import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import csv
from torch.utils.data import Dataset
import argparse
from pathlib import Path
from seamless_communication.models.generator.loader import load_pretssel_vocoder_model, PretsselVocoder
from seamless_communication.store import add_gated_assets
from seamless_communication.models.unity import load_gcmvn_stats
from fairseq2.data.audio import WaveformToFbankConverter, WaveformToFbankInput

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
add_gated_assets(Path("/gscratch/intelligentsystems/tuochao/Large_Model/SeamlessExpressive/"))
import os
os.environ['FAIRSEQ2_CACHE_DIR'] = '/gscratch/intelligentsystems/tuochao/Large_Model/seamless/'


class VariancePredictor(nn.Module):
    def __init__(self, args, embedding_tokens, use_rnn = False):
        super().__init__()
        self.embedding_tokens = embedding_tokens
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                args.encoder_embed_dim,
                args.var_pred_hidden_dim,
                kernel_size=args.var_pred_kernel_size,
                padding=(args.var_pred_kernel_size - 1) // 2,
            ),
            nn.ReLU(),
        )
        self.ln1 = nn.LayerNorm(args.var_pred_hidden_dim)
        self.dropout_module = nn.Dropout(p=args.var_pred_dropout)
        self.use_rnn = use_rnn
        if use_rnn:
            self.rnn = torch.nn.LSTM(args.var_pred_hidden_dim, args.var_pred_hidden_dim, num_layers=1, batch_first = True)

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                args.var_pred_hidden_dim,
                args.var_pred_hidden_dim,
                kernel_size=args.var_pred_kernel_size,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.ln2 = nn.LayerNorm(args.var_pred_hidden_dim)
        self.proj = nn.Linear(args.var_pred_hidden_dim, 1)
        self.freeze_embed()

    def freeze_embed(self):
        for param in self.embedding_tokens.parameters():
            param.requires_grad = False

    def forward(self, unit):
        # Input: B x T x C; Output: B x T
        x = self.embedding_tokens(unit)
        x = x.float()
        # print(x.shape)
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout_module(self.ln1(x))
        if self.use_rnn:
            x, _ = self.rnn(x)
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout_module(self.ln2(x))
        return self.proj(x).squeeze(dim=2)


class DurDataset(Dataset):
    def __init__(self, tsv_file):
        """
        Args:
            data (array-like or tensor): Your data.
            labels (array-like or tensor): Corresponding labels for the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_list = []

        # Open and read the CSV file
        with open(tsv_file, mode='r') as file:
            # Create a CSV DictReader
            tsv_reader = csv.DictReader(file, delimiter='\t')
            # Iterate over each row and convert it to a dictionary
            for row in tsv_reader:
                self.data_list.append(dict(row))

    def __len__(self):
        # Return the total number of samples
        return min([len(self.data_list)])

    def str2unit(self, unit_str):
        unit = []
        for a in unit_str.split(" "):
            unit.append(int(a))
        unit = torch.tensor(unit, dtype=torch.long)
        return unit

    def __getitem__(self, idx):
        # Retrieve the data and label for a given index
        sample = self.data_list[idx]
        tgt = sample["tgt_audio"]
        raw = sample["tgt_audio_raw"]
        
        raw = self.str2unit(raw) + 4
        unit, duration = torch.unique_consecutive(raw, return_counts=True)

        return unit, duration.float()

def collate_fn(batch):
    # Batch is a list of (sequence, label) tuples
    sample, unit = zip(*batch)
    
    # Pad sequences to the length of the longest sequence in the batch
    padded_sequences = pad_sequence(sample, batch_first=True, padding_value=0) # B x T
    padded_units = pad_sequence(unit, batch_first=True, padding_value=0) # B x T
    
    
    # Compute the length of each sequence before padding
    lengths = torch.tensor([len(s) for s in sample])
    
    
    return padded_sequences, padded_units, lengths


def build_model(use_rnn, device):
    args = argparse.Namespace(
        encoder_embed_dim = 256,
        var_pred_hidden_dim = 256,
        var_pred_kernel_size = 3,
        var_pred_dropout = 0.5
    )
    vocoder_name = "vocoder_pretssel_16khz"
    vocoder = load_pretssel_vocoder_model(vocoder_name, device=device, dtype=torch.float16)
    embed_tokens = vocoder.encoder_frontend.embed_tokens
    model = VariancePredictor(args, embed_tokens, use_rnn)
    return model


def build_model2(embed_tokens, use_rnn):
    args = argparse.Namespace(
        encoder_embed_dim = 256,
        var_pred_hidden_dim = 256,
        var_pred_kernel_size = 3,
        var_pred_dropout = 0.5
    )
    # vocoder_name = "vocoder_pretssel_16khz"
    # vocoder = load_pretssel_vocoder_model(vocoder_name, device=device, dtype=torch.float16)
    # embed_tokens = vocoder.encoder_frontend.embed_tokens
    model = VariancePredictor(args, embed_tokens, use_rnn)
    return model


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.l1_loss_fn = nn.MSELoss(reduction='none')  # Use 'none' to keep per-element loss
    def forward(self, outputs, targets, lengths):
        """
        Args:
            outputs (torch.Tensor): Predicted outputs (batch_size, max_seq_len)
            targets (torch.Tensor): Ground truth targets (batch_size, max_seq_len)
            lengths (torch.Tensor): Lengths of each sequence in the batch (batch_size).
        
        Returns:
            torch.Tensor: The mean L1 loss, masked to ignore padding.
        """
        assert outputs.shape == targets.shape, "Outputs and targets must have the same shape."

        # Create a mask based on the lengths
        # print(lengths)
        max_len = outputs.size(1)
        mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        mask = mask.to(outputs.device)
        
        # Calculate the L1 loss (per element)
        loss = self.l1_loss_fn(outputs, targets) # B T
        # print(mask.shape, loss.shape)
        masked_loss = loss * mask
        masked_loss_sum = masked_loss.sum(dim=1)  # Sum over the sequence length dimension
        num_valid_elements = mask.sum(dim=1).unsqueeze(-1).float()
        num_valid_elements = torch.clamp(num_valid_elements, min=1.0)

        final_loss = masked_loss_sum / num_valid_elements  # Mean loss per sequence

        # Return the mean loss over the batch
        return final_loss.mean()

if __name__ == "__main__":
    from transformers import get_linear_schedule_with_warmup
    criterion = MaskedL1Loss()
    device  = torch.device("cuda")
    checkpoints = "agent/unit_dur/dur_predictor.pt"
    model = build_model(device = device, use_rnn = False)
    model = model.to(device)
    print(model)
    data_folder = "/scr/data_streamspeech_es/cvss/cvss-c/es-en/fbank2unit/"
    batch_size = 8
    val_set = DurDataset(data_folder + "dev.tsv")
    train_set = DurDataset(data_folder + "train.tsv")

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50
    total_steps = len(train_dataloader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # Number of warmup steps (can be set as needed)
        num_training_steps=total_steps * 5
    )

    start_epoch = 0
    best_val_loss = 99999
    if os.path.exists(checkpoints):
        checkpoint = torch.load(checkpoints)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['loss']
        print(f"Checkpoint loaded from {checkpoints}")
    
    # Training loop

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for param_group in optimizer.param_groups:
            print(f"  Learning Rate: {param_group['lr']:.6f}")
        ### training
        cumm = 0
        cumm_batch = 0
        for batch in tqdm(train_dataloader):
            # if cumm > early_quit:
            #     break
            # else:
            #     cumm += 1
            inputs, labels, lengths = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            # lengths = lengths.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels, lengths)
            # print(loss.dtype)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Step the learning rate scheduler
            scheduler.step()
            
            running_loss += loss.item()
            cumm_batch += inputs.shape[0]
            # print(loss.item())
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader):.4f}")

        
        ### evals
        model.eval()  # Set the model to evaluation mode
        eval_loss = 0.0
        correct_predictions = 0

        with torch.no_grad():  # Disable gradient computation for evaluation
            for inputs, labels, lengths in tqdm(val_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # lengths = lengths.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels, lengths)
                eval_loss += loss.item()
    
        print(f"Eval Epoch [{epoch+1}/{num_epochs}], Loss: {eval_loss/len(val_dataloader):.4f}")
        val_loss = eval_loss/len(val_dataloader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
            }, checkpoints)
            print("Checkpoint saved with validation loss:", val_loss)


    print("Training finished.")    

