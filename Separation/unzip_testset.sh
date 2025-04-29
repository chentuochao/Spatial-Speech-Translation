data_path=/data
tar -xvf XXXXX/data_blind_separation_multilang.tar -C $data_path
tar -xvf XXXXX/data_stream_1channel_processed.tar.gz  -C $data_path
rm -rf $data_path/data_streamspeech/covost2/
rm -rf $data_path/data_streamspeech/cvss/cvss-c/fr-en/src_fbank80.zip