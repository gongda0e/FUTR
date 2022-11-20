python main.py \
    --task long \
    --seg --anticipate --pos_emb\
    --n_encoder_layer 2 --n_decoder_layer 1 --batch_size 16 --hidden_dim 128 --max_pos_len 2000\
    --epochs 60 --mode=train --input_type=i3d_transcript --split=$1

