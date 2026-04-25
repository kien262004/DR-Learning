CUDA_VISIBLE_DEVICES=0 python main_supcon.py --batch_size 2 --learning_rate 0.001  --temp 0.1 --cosine --num_workers 4 \
--dataset path \
--data_folder ../dataset/train_re \
--mean "(0.4914, 0.4822, 0.4465)" \
--std "(0.2675, 0.2565, 0.2761)" \
--method SupCon \
--print_freq 1 --save_freq 5