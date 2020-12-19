# Sequence-Level Mixed Sample Data Augmentation
Despite their empirical success, neural networks still have difficulty capturing compositional aspects of natural language. This work proposes a simple data augmentation approach to encourage compositional behavior in neural models for sequence-to-sequence problems. Our approach, SeqMix, creates new synthetic examples by softly combining input/output sequences from the training set. We connect this approach to existing techniques such as SwitchOut and word dropout, and show that these techniques are all approximating variants of a single objective. SeqMix consistently yields approximately 1.0 BLEU improvement on five different translation datasets over strong Transformer baselines. On tasks that require strong compositional generalization such as SCAN and semantic parsing, SeqMix also offers further improvements.

[[EMNLP 2020 Paper]](https://arxiv.org/abs/2011.09039)

## Running Experiments
Code will be released soon! Hopefully, some time in late December.
