## Dual-Directional Training

TEXT=./download_prepare/data_mixed/
SAVE_DIR=./models/bibert-r-drop-8/
LOG_DIR_NAME=bibert-r-drop-8

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train ${TEXT}de-en-databin/ --arch transformer_iwslt_de_en --ddp-backend no_c10d --optimizer adam --adam-betas '(0.9, 0.98)' \
--clip-norm 1.0 --lr 0.0004 --lr-scheduler inverse_sqrt --warmup-updates 1000 --warmup-init-lr 1e-07 --dropout 0.3 \
--weight-decay 0.00002 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 2048 --update-freq 4 \
--attention-dropout 0.1 --activation-dropout 0.1 --save-dir ${SAVE_DIR}  --encoder-embed-dim 768 --decoder-embed-dim 768 \
--no-epoch-checkpoints --save-interval 1 --pretrained_model jhu-clsp/bibert-ende --use_drop_embedding 8 \
--tensorboard-logdir ${LOG_DIR_NAME} --patience 5 \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
--user-dir examples/translation_rdrop \
--task rdrop_translation \
--reg-alpha 5 \
--criterion reg_label_smoothed_cross_entropy
## Fine-Tuning

TEXT_FT=./download_prepare/data_mixed_ft/
SAVE_DIR_FT=./models/bibert-r-drop-8-ft/
LOG_DIR_NAME_FT=bibert-r-drop-8-ft

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train ${TEXT_FT}de-en-databin/ --arch transformer_iwslt_de_en --ddp-backend no_c10d --optimizer adam --adam-betas '(0.9, 0.98)' \
--clip-norm 1.0 --lr 0.0004 --lr-scheduler inverse_sqrt --warmup-updates 1000 --warmup-init-lr 1e-07 --dropout 0.3 \
--weight-decay 0.00002 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 2048 --update-freq 4 \
--attention-dropout 0.1 --activation-dropout 0.1 --max-epoch 20 --save-dir ${SAVE_DIR_FT}  --encoder-embed-dim 768 --decoder-embed-dim 768 \
--no-epoch-checkpoints --save-interval 1 --pretrained_model jhu-clsp/bibert-ende --use_drop_embedding 8 \
--restore-file ${SAVE_DIR}checkpoint_best.pt --reset-lr-scheduler --reset-dataloader --reset-meters \
--tensorboard-logdir ${LOG_DIR_NAME_FT} --patience 5 \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--best-checkpoint-metric bleu --maximize-best-checkpoint-metric  \
--user-dir examples/translation_rdrop \
--task rdrop_translation \
--reg-alpha 5 \
--criterion reg_label_smoothed_cross_entropy
