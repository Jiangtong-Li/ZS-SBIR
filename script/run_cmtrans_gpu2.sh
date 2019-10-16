python ./src/main_cmdtrans.py  --sketch_dir ./data/256x256/sketch/tx_000100000000 \
                               --image_dir /./data/256x256/EXTEND_image_sketchy \
                               --stats_file ./data/info/stats.csv \
                               --packed_pkl_nozs ./data/preprocessed/cm_trans_sketch_all_unpair/nozs_packed.pkl \
                               --packed_pkl_zs ./data/preprocessed/cm_trans_sketch_all_unpair/zs_packed.pkl \
                               --embedding_file ./data/GoogleNews-vectors-negative300.bin \
                               --preprocess_data ./data/preprocessed/cm_trans_sketch_all_unpair/CNN_feature_1024.h5py \
                               --log_file ./log/cmd_trans_1024.log \
                               --shuffle \
                               --pca_size 1024 \
                               --hidden_size 1024 \
                               --semantics_size 300 \
                               --fix_embedding \
                               --seman_dist l2 \
                               --triplet_dist l2 \
                               --margin 10.0 \
                               --patience 30 \
                               --batch_size 128 \
                               --num_worker 8 \
                               --dropout 0.5 \
                               --warmup_steps 500 \
                               --lr 1e-4 \
                               --print_every 125 \
                               --save_every 250 \
                               --save_dir ./model/cmd_trans_1024/ \
                               --gpu_id 1 \
                               --cum_num 1 \
                               --zs
