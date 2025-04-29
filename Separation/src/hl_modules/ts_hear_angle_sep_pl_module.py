import os

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import torch
from numpy import mean
from src.metrics.metrics import Metrics
from src.metrics.metrics import compute_decay
import src.utils as utils
from transformers.debug_utils import DebugUnderflowOverflow
import numpy as np 



class PLModule(object):
    def __init__(self, model, model_params, sr,
                 optimizer, optimizer_params,
                 scheduler=None, scheduler_params=None,
                 loss=None, loss_params=None, 
                 metrics=[], init_ckpt=None,
                 grad_clip = None,
                 use_dp=True,
                 val_log_interval=10, # Unused, only kept for compatibility TODO: Remove
                 samples_per_speaker_number=3):
        print(init_ckpt) 

        self.output_src = model_params["num_src"]
        self.model = utils.import_attr(model)(**model_params)
        self.use_dp = use_dp
        if use_dp:
            self.model = nn.DataParallel(self.model)
        
        self.sr = sr
        
        #debug_overflow = DebugUnderflowOverflow(self.model)
        # Log a val sample every this many intervals
        # self.val_log_interval = val_log_interval
        self.samples_per_speaker_number = samples_per_speaker_number
        
        # Initialize metrics
        self.metrics_list = ["val/" + m for m in metrics]
        self.metrics = [Metrics(metric) for metric in metrics]

        # Metric values
        self.metric_values = {}
        
        # Dataset statistics
        self.statistics = {}
       
        # Assine metric to monitor, and how to judge different models based on it
        # i.e. How do we define the best model (Here, we minimize val loss)
        self.monitor = 'val/loss'
        self.monitor_mode = 'min'

        # Mode, either train or val
        self.mode = None

        self.val_samples = {}
        self.train_samples = {}

        # Initialize loss function
        self.loss_fn = utils.import_attr(loss)(**loss_params)
        
        # Initaize weights if checkpoint is provided
        # Warning: This will only load the weights of the module
        # called "model" in this class
        if init_ckpt is not None:
            print("load", init_ckpt)
        
            state = torch.load(init_ckpt)
            print(state["current_epoch"])    
            state = state["model"]
            if self.use_dp:
                self.model.module.load_state_dict(state)
            else:
                self.model.load_state_dict(state)

         # Initialize optimizer
        self.optimizer = utils.import_attr(optimizer)(self.model.parameters(), **optimizer_params)
        self.optim_name = optimizer
        self.opt_params = optimizer_params

        # Grad clip
        self.grad_clip = grad_clip

        if self.grad_clip is not None:
            print(f"USING GRAD CLIP: {self.grad_clip}")
        else:
            print("ERROR! NOT USING GRAD CLIP" * 100)

        # Initialize scheduler
        self.scheduler = self.init_scheduler(scheduler, scheduler_params)
        self.scheduler_name = scheduler
        self.scheduler_params = scheduler_params
        
        self.epoch = 0
    
    def load_state(self, path, map_location=None):
        state = torch.load(path, map_location=map_location)

        if self.use_dp:
            self.model.module.load_state_dict(state['model'])
        else:
            self.model.load_state_dict(state['model'])
        
        # Re-initialize optimizer
        self.optimizer = utils.import_attr(self.optim_name)(self.model.parameters(), **self.opt_params)
        
        # Re-initialize scheduler (Order might be important?)
        if self.scheduler is not None:
            self.scheduler = self.init_scheduler(self.scheduler_name, self.scheduler_params)

        self.optimizer.load_state_dict(state['optimizer'])

        if self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        
        self.epoch = state['current_epoch']
        print("Load model from epoch", self.epoch)
        self.metric_values = state['metric_values']
        
        if 'statistics' in self.statistics:
            self.statistics = state['statistics']

    def dump_state(self, path):
        if self.use_dp:
            _model = self.model.module
        else:
            _model = self.model
        
        state = dict(model = _model.state_dict(),
                     optimizer = self.optimizer.state_dict(),
                     current_epoch = self.epoch,
                     metric_values=self.metric_values, 
                     statistics = self.statistics)
        
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        print("save to " + path)
        torch.save(state, path)

    def get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def on_epoch_start(self):
        print()
        print("=" * 25, "STARTING EPOCH", self.epoch, "=" * 25)
        print()

    def get_avg_metric_at_epoch(self, metric, epoch = None):
        if epoch is None:
            epoch = self.epoch
        
        return self.metric_values[epoch][metric]['epoch'] / \
            self.metric_values[epoch][metric]['num_elements']

    def on_epoch_end(self, best_path, wandb_run):
        assert self.epoch + 1 == len(self.metric_values), \
            "Current epoch must be equal to length of metrics (0-indexed)"

        monitor_metric_last = self.get_avg_metric_at_epoch(self.monitor)

        # Go over all epochs
        save = True
        for epoch in range(len(self.metric_values) - 1):
            monitor_metric_at_epoch = self.get_avg_metric_at_epoch(self.monitor, epoch)
            
            if self.monitor_mode == 'max':
                # If there is any model with monitor larger than current, then
                # this is not the best model
                if monitor_metric_last < monitor_metric_at_epoch:
                    save = False
                    break

            if self.monitor_mode == 'min':
                # If there is any model with monitor smaller than current, then
                # this is not the best model
                if monitor_metric_last > monitor_metric_at_epoch:
                    save = False
                    break
        
        # If this is best, save it
        if save:
            print("Current checkpoint is the best! Saving it...")
            self.dump_state(best_path)
        
        val_loss = self.get_avg_metric_at_epoch('val/loss')
        print(f'Val loss: {val_loss:.02f}')

        for m in self.metrics_list:
            m_value = self.get_avg_metric_at_epoch(m)
            print(f'{m}: {m_value:.02f}')

        
        # print(f'Val SNRi: {val_snr_i:.02f}dB')
        # print(f'Val SI-SDRi: {val_si_snr_i:.02f}dB')


        # def log_audio(run, key, samples, sr):
        #     columns = ['mixture', 'target', 'output']
        #     wandb_samples = []
        #     for i, sample in enumerate(samples):
        #         # TODO: Save spectrograms as well
        #         for k in columns:
        #             wandb_samples.append(wandb.Audio(
        #                 sample[k].permute(1, 0).cpu().numpy(),
        #                 sample_rate=sr, caption=f'{i}/{k}'))
        #     run.log({key: wandb_samples}, commit=False, step=self.epoch + 1)

        # Log stuff on wandb
        wandb_run.log({'lr-Adam': self.get_current_lr()}, commit=False, step=self.epoch + 1)

        for metric in self.metric_values[self.epoch]:
            wandb_run.log({metric: self.get_avg_metric_at_epoch(metric)}, commit=False, step=self.epoch + 1)
        
        for statistic in self.statistics:
            if not self.statistics[statistic]['logged']:
                data = self.statistics[statistic]['data']
                reduction = self.statistics[statistic]['reduction']
                if reduction == 'mean':
                    val = mean(data)
                elif reduction == 'sum':
                    val = sum(data)
                elif reduction == 'histogram':
                    data = [[d] for d in data]
                    table = wandb.Table(data=data, columns=[statistic])
                    val = wandb.plot.histogram(table, statistic, title=statistic)
                else:
                    assert 0, f"Unknown reduction {reduction}."
                wandb_run.log({statistic: val}, commit=False)
                self.statistics[statistic]['logged'] = True
        
        # for spk_num in self.train_samples:
        #     log_audio(wandb_run, f"train/audio_samples_{spk_num}spk", self.train_samples[spk_num], sr=self.sr)
        # self.train_samples.clear()

        # for spk_num in self.val_samples:
        #     log_audio(wandb_run, f"val/audio_samples_{spk_num}spk", self.val_samples[spk_num], sr=self.sr)
        # self.val_samples.clear()

        wandb_run.log({'epoch': self.epoch}, commit=True, step=self.epoch + 1)
        
        if self.scheduler is not None:
            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                # Get last metric
                self.scheduler.step(monitor_metric_last)
            else:
                self.scheduler.step()

        self.epoch += 1

    def log_statistic(self, name, value, reduction='mean'):
        if name not in self.statistics:
            self.statistics[name] = dict(logged=False, data=[], reduction=reduction)
        
        self.statistics[name]['data'].append(value)

    def log_metric(self, name, value, batch_size=1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True):
        """
        Logs a metric
        value must be the AVERAGE value across the batch
        Must provide batch size for accurate average computation
        """
        
        epoch_str = self.epoch
        if epoch_str not in self.metric_values:
            self.metric_values[epoch_str] = {}

        if (name not in self.metric_values[epoch_str]):
            self.metric_values[epoch_str][name] = dict(step=None, epoch=None)
        
        if type(value) == torch.Tensor:
            value = value.item()

        if on_step:            
            if self.metric_values[epoch_str][name]['step'] is None:
                self.metric_values[epoch_str][name]['step'] = []
            
            self.metric_values[epoch_str][name]['step'].append(value)
        
        if on_epoch:
            if self.metric_values[epoch_str][name]['epoch'] is None:
                self.metric_values[epoch_str][name]['epoch'] = 0
                self.metric_values[epoch_str][name]['num_elements'] = 0
            
            self.metric_values[epoch_str][name]['epoch'] += (value * batch_size)
            self.metric_values[epoch_str][name]['num_elements'] += batch_size

    def _step(self, batch, batch_idx, step='train'):
        inputs, targets = batch
        batch_size = inputs['mixture'].shape[0]
       
        #print(targets['num_target_speakers'])
        # Forward pass
        #print(inputs['mixture'].shape)
        outputs = self.model(inputs)
        #print(outputs['output'].shape, targets['target'].shape) 

        mix = inputs['mixture'].clone() # Take first channel in mixture as reference
        est = outputs['output'].clone()
        
        gt = targets['target'].clone()


        POS_OR_NEG = targets['sample_pos'].clone()
        # Compute loss
        loss = 0
        for i in range(self.output_src):
            loss += self.loss_fn(est=est[:,i:i+1,:], gt=gt[:, i:i+1,:]).mean()
        loss /= self.output_src
        est_detached = est.detach().clone()
        

        mix = mix[:, :self.output_src, :]
        gt = gt[:, :self.output_src, :]
        
        # print(POS_OR_NEG)
        with torch.no_grad():
            # Log loss
            self.log_metric(f'{step}/loss', loss.item(), batch_size=batch_size, on_step=(step == 'train'), on_epoch=True, prog_bar=True, sync_dist=True)

            # Log metrics
            for metric in self.metrics:
                if step == "train" and (metric.name == "PESQ" or metric.name == "STOI"):
                    continue
                metric_val = metric(est=est_detached, gt=gt, mix=mix)
                for i in range(batch_size):
                    if POS_OR_NEG[i] > 0:
                        assert torch.abs(gt[i]).max() > 1e-5, "Expected gt > 0"

                        val = metric_val[i].item()
                        self.log_metric(f'{step}/{metric.name}', val, batch_size=1,
                                on_step=False, on_epoch=True, prog_bar=True,
                                sync_dist=True)
            
            # Log metrics for zero speakers
            for i in range(batch_size):
                if POS_OR_NEG[i] == 0:
                    assert torch.abs(gt[i]).max() < 1e-4, "Expected gt close to 0"
                    decay = compute_decay(est_detached[i].unsqueeze(0), mix[i].unsqueeze(0)).item()
                    # print("decay",decay)
                    self.log_metric(f'{step}/decay', decay, batch_size=1,
                            on_step=False, on_epoch=True, sync_dist=True)
            

        sample = {
            'mixture': mix,
            'output': est_detached,
            'target': gt,
            'POS_OR_NEG': POS_OR_NEG,
        }

        return loss, sample

    def train(self):
        self.model.train()
        self.mode = 'train'
    
    def eval(self):
        self.model.eval()
        self.mode = 'val'

    def training_step(self, batch, batch_idx):
        loss, sample = self._step(batch, batch_idx, step='train')

        POS_OR_NEG = sample['POS_OR_NEG']

        
        return loss, POS_OR_NEG.shape[0]

    def validation_step(self, batch, batch_idx):
        loss, sample = self._step(batch, batch_idx, step='val')
        
        POS_OR_NEG = sample['POS_OR_NEG']

        return loss, POS_OR_NEG.shape[0]
    
    def reset_grad(self):
        self.optimizer.zero_grad()

    def backprop(self):
        #print("BACKPROP")
        #print(self.grad_clip)
        # Gradient clipping
        if self.grad_clip is not None:
            #print("Clipping grad norm")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip) 
        
        self.optimizer.step()
    
    def configure_optimizers(self):        
        if self.scheduler is not None:
            # For reduce LR on plateau, we need to provide more information
            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler_cfg = {
                    "scheduler": self.scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": self.monitor,
                    "strict": False
                }
            else:
                scheduler_cfg = self.scheduler
            return [self.optimizer], [scheduler_cfg]
        else:
            return self.optimizer

    def init_scheduler(self, scheduler, scheduler_params):
        if scheduler is not None:
            if scheduler == 'sequential':
                schedulers = []
                milestones = []
                for scheduler_param in scheduler_params:
                    sched = utils.import_attr(scheduler_param['name'])(self.optimizer, **scheduler_param['params'])
                    schedulers.append(sched)
                    milestones.append(scheduler_param['epochs'])

                # Cumulative sum for milestones
                for i in range(1, len(milestones)):
                    milestones[i] = milestones[i-1] + milestones[i]

                # Remove last milestone as it is implied by num epochs
                milestones.pop()

                scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers, milestones)
            else:
                scheduler = utils.import_attr(scheduler)(self.optimizer, **scheduler_params)

        return scheduler
