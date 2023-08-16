# TS-HuBERT

## Reference

```bibtex
# Interspeech 2023
# https://isca-speech.org/archive/interspeech_2023/zhang23w_interspeech.html
@inproceedings{zhang23w_interspeech,
  title={Weakly-Supervised Speech Pre-training: A Case Study on Target Speech Recognition},
  author={Wangyou Zhang and Yanmin Qian},
  booktitle={Proc. INTERSPEECH},
  pages={3517--3521},
  year=2023,
}

# Preprint
# https://arxiv.org/abs/2305.16286

@article{TSHUBERT-Zhang2023,
  title={Weakly-Supervised Speech Pre-training: A Case Study on Target Speech Recognition},
  author={Zhang, Wangyou and Qian, Yanmin},
  journal={arXiv preprint arXiv:2305.16286},
  year={2023},
}
```

<!-- ## Pre-trained models
Model | Pre-training Dataset | Fine-tuning Dataset | Model
|---|---|---|---
TS-HuBERT Base |  [960 hrs LibriSpeech](http://www.openslr.org/12)| [Libri2Mix](https://github.com/JorisCos/LibriMix/) | [Google Drive]() -->

## Load a model
```python
ckpt_path = "/path/to/the/checkpoint.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
```

## Load Pre-Trained Models for Inference
```python
import fairseq
import torch

# load the pre-trained checkpoints
ckpt_path = "/path/to/the/checkpoint.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
model.eval()

# extract the the representation of last layer
wav_input_16khz = torch.randn(1,10000)
enroll_wav_16khz = torch.randn(1,8000)
rep = model.extract_features(wav_input_16khz, enroll_wav_16khz)[0]

# extract the the representation of each layer
wav_input_16khz = torch.randn(1,10000)
enroll_wav_16khz = torch.randn(1,8000)
rep, layer_results = model.extract_features(wav_input_16khz, enroll_wav_16khz, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
```

## Train a new model

### Data preparation (Pretraining)

Run `local/prepare_tshubert_pretraining_data.sh` to create:
- `{train960,dev}.tsv` waveform list files
- `{train960,dev}.km` frame-aligned pseudo label files.
The `label_rate` is the same as the feature frame rate used for clustering,
which is 100Hz for MFCC features and 50Hz for TS-HuBERT features by default.

### Pre-train a TS-HuBERT model

Suppose `{train960,dev}.tsv` are saved at `/path/to/data/tsv_files`, `{train960,dev}.km`
are saved at `/path/to/kmeans_model/km_train960_TS-HuBERT9/kmeans_labels`, and the label rate is 100Hz.

To train a base model (12 layer transformer), run:
```sh
$ python fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq-py/examples/tshubert/config/pretrain \
  --config-name tshubert_base_iter1_librispeech_2x4gpu \
  task.data=/path/to/data/tsv_files task.label_dir=/path/to/kmeans_model/km_train960_TS-HuBERT9/kmeans_labels model.label_rate=100
```
Alternatively, run `run/pretrain_tshubert_iter1.sh`.


### Data preparation (Fine-tuning on mixture data for target speech recognition)
Run `local/prepare_tshubert_finetuning_data.sh` to create:
- `{train,valid}.tsv` waveform list files
- `{train,valid}.ltr` character-level label files.
- `{train,valid}.wrd` word-level label files.
- `{train,valid}.enroll` enrollment list files.

### Fine-tune a TS-HuBERT model on mixture data with a CTC loss for target speech recognition

Suppose `{train,valid}.tsv` are saved at `/path/to/train_2mix`, and their
corresponding character transcripts `{train,valid}.ltr` are saved at
`/path/to/trans`.

To fine-tune a pre-trained TS-HuBERT model at `/path/to/checkpoint`, run
```sh
$ python fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq-py/examples/tshubert/config/finetune \
  --config-name tshubert_base_iter1_libri2mix_train100_8x1gpu \
  task.data=/path/to/train_2mix task.label_dir=/path/to/trans \
  model.w2v_path=/path/to/checkpoint
```
Alternatively, run `run/finetune_tshubert_iter1_libri2mix.sh`.

### Data preparation (Decoding on mixture data for target speech recognition)
Run `local/prepare_tshubert_decoding_data.sh` to create:
- `test.tsv` waveform list files
- `test.ltr` character-level label files.
- `test.wrd` word-level label files.
- `test.enroll` enrollment list files.

### Decode a TS-HuBERT model on mixture data for target speech recognition

Suppose the `test.tsv` and `test.ltr` are the waveform list and transcripts of
the split to be decoded, saved at `/path/to/test_2mix`, and the fine-tuned model is
saved at `/path/to/checkpoint`. We support three decoding modes:
- Viterbi decoding: greedy decoding without a language model
- KenLM decoding: decoding with an arpa-format KenLM n-gram language model
- Fairseq-LM deocding: decoding with a Fairseq neural language model


#### Viterbi decoding on mixture data for target speech recognition

`task.normalize` needs to be consistent with the value used during fine-tuning.
Decoding results will be saved at
`/path/to/experiment/directory/decode/viterbi/test_2mix`.

```sh
$ python examples/speech_recognition/new/infer.py \
  --config-dir /path/to/fairseq-py/examples/tshubert/config/decode \
  --config-name infer_viterbi \
  task.data=/path/to/test_2mix \
  task.normalize=[true|false] \
  decoding.exp_dir=/path/to/experiment/directory \
  common_eval.path=/path/to/checkpoint
  dataset.gen_subset=test \
```
Alternatively, run `run/decode_tshubert_iter1_libri2mix.sh`.

#### KenLM / Fairseq-LM decoding on mixture data for target speech recognition

Suppose the pronunciation lexicon and the n-gram LM are saved at
`/path/to/lexicon` and `/path/to/arpa`, respectively. Decoding results will be
saved at `/path/to/experiment/directory/decode/kenlm/test`.

```sh
$ python examples/speech_recognition/new/infer.py \
  --config-dir /path/to/fairseq-py/examples/tshubert/config/decode \
  --config-name infer_kenlm \
  task.data=/path/to/test_clean \
  task.normalize=[true|false] \
  decoding.exp_dir=/path/to/experiment/directory \
  common_eval.path=/path/to/checkpoint
  dataset.gen_subset=test \
  decoding.decoder.lexicon=/path/to/lexicon \
  decoding.decoder.lmpath=/path/to/arpa
```

The command above uses the default decoding hyperparameter, which can be found
in `examples/speech_recognition/hydra/decoder.py`. These parameters can be
configured from the command line. For example, to search with a beam size of
500, we can append the command above with `decoding.decoder.beam=500`.
Important parameters include:
- decoding.decoder.beam
- decoding.decoder.beamthreshold
- decoding.decoder.lmweight
- decoding.decoder.wordscore
- decoding.decoder.silweight

To decode with a Fairseq LM, use `--config-name infer_fsqlm` instead, and
change the path of lexicon and LM accordingly.
