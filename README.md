## MobileBERT_pytorch

This repository contains a PyTorch implementation of the **MobileBERT** model from the paper 

[MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/pdf/2004.02984.pdf)

by Zhiqing Sun1∗, Hongkun Yu2, Xiaodan Song....

## Dependencies

- pytorch=1.10
- cuda=9.0
- cudnn=7.5
- scikit-learn
- sentencepiece
- tokenizers

## Download Pre-trained Models of English

Official download links: [google mobilebert](https://github.com/google-research/google-research/tree/master/mobilebert)

## Fine-tuning

１. Place `config.json` and `vocab.txt` into the `prev_trained_model/mobilebert` directory.
example:
```text
├── prev_trained_model
|  └── mobilebert
|  |  └── pytorch_model.bin
|  |  └── config.json
|  |  └── vocab.txt
```
2．convert mobilebert tf checkpoint to pytorch
```python
python convert_mobilebert_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path=./prev_trained_model/mobilebert \
    --mobilebert_config_file=./prev_trained_model/mobilebert/config.json \
    --pytorch_dump_path=./prev_trained_model/mobilebert/pytorch_model.bin
```
The [General Language Understanding Evaluation (GLUE) benchmark](https://gluebenchmark.com/) is a collection of nine sentence- or sentence-pair language understanding tasks for evaluating and analyzing natural language understanding systems.

Before running anyone of these GLUE tasks you should download the [GLUE data](https://gluebenchmark.com/tasks) by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to some directory $DATA_DIR.

3．run `sh scripts/run_classifier_sst2.sh`to fine tuning mobilebert model

## Result

Performance of MobileBert on GLUE benchmark results using a single-model setup on **dev**:

|  | Cola| Sst-2| Sts-b|
| :------- | :---------: | :---------: | :---------: |
| metric | matthews_corrcoef |accuracy | pearson |
| mobilebert | 0.5837 | 0.922 | 0.8839 |



