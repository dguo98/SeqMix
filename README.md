# Sequence-Level Mixed Sample Data Augmentation
Despite their empirical success, neural networks still have difficulty capturing compositional aspects of natural language. This work proposes a simple data augmentation approach to encourage compositional behavior in neural models for sequence-to-sequence problems. Our approach, SeqMix, creates new synthetic examples by softly combining input/output sequences from the training set. We connect this approach to existing techniques such as SwitchOut and word dropout, and show that these techniques are all approximating variants of a single objective. SeqMix consistently yields approximately 1.0 BLEU improvement on five different translation datasets over strong Transformer baselines. On tasks that require strong compositional generalization such as SCAN and semantic parsing, SeqMix also offers further improvements.

[[EMNLP 2020 Paper]](https://arxiv.org/abs/2011.09039)

# Prerequisites
This is example codebase for NMT experiments, which is based on fairseq. Please follow fairseq set up instructions [here](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8)

# Data Preprocessing
First download the [preprocessed WMT'16 data](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8). <br />
Then, extract the WMT'16 En-De data.
```
TEXT=wmt16_en_de_bpe32k
mkdir -p $TEXT
tar -xzvf wmt16_en_de.tar.gz -C $TEXT
```

Preprocess the data with a joined dictionary.
```
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train.tok.clean.bpe.32000 \
    --validpref $TEXT/newstest2013.tok.bpe.32000 \
    --testpref $TEXT/newstest2014.tok.bpe.32000 \
    --destdir data-bin/wmt16_en_de_bpe32k \
    --nwordssrc 32768 --nwordstgt 32768 \
    --joined-dictionary \
    --workers 20
```

# Training and Evaluation
Train a model.
```
EXP_NAME=WMT
ALPHA=0.10

mkdir -p checkpoints/${EXP_NAME}

CUDA_VISIBLE_DEVICES=0,1,2,3  python train.py data-bin/wmt16_en_de_bpe32k \
        --arch transformer_wmt_en_de --share-all-embeddings --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --ddp-backend=no_c10d \
        --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000 \
              --lr 0.0007 --min-lr 1e-09 \
             --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 --dropout 0.1 \
              --max-tokens  3072   --save-dir checkpoints/${EXP_NAME}  \
              --update-freq 3 --no-progress-bar --log-format json --log-interval 50\
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "lenpen": 0.6}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
              --validate-interval 1 \
              --save-interval 1 --keep-last-epochs 5 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --alpha ${ALPHA} 1>logs/${EXP_NAME}.out 2>logs/${EXP_NAME}.err

```

Finally, evaluate.
```
python scripts/average_checkpoints.py --inputs checkpoints/${EXP_NAME} --num-epoch-checkpoints 5 --output checkpoints/${EXP_NAME}/avg5.pt

fairseq-generate data-bin/wmt16_en_de_bpe32k --path checkpoints/${EXP_NAME}/avg5.pt --batch-size 32 --beam 4 --lenpen 0.6 --remove-bpe --gen-subset test > logs/${EXP_NAME}.avg5.raw_result

GEN=logs/${EXP_NAME}.avg5.raw_result

SYS=$GEN.sys
REF=$GEN.ref

grep ^H $GEN | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $SYS
grep ^T $GEN | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $REF
python score.py --sys $SYS --ref $REF > logs/${EXP_NAME}.avg5.final_result

```

