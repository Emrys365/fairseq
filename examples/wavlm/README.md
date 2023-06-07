# WavLM

## Reference

```bibtex
@article{WavLM-Chen2022,
  title={{WavLM}: Large-scale self-supervised pre-training for full stack speech processing},
  author={Chen, Sanyuan and Wang, Chengyi and Chen, Zhengyang and Wu, Yu and Liu, Shujie and Chen, Zhuo and Li, Jinyu and Kanda, Naoyuki and Yoshioka, Takuya and Xiao, Xiong and Wu, Jian and Zhou, Long and Ren, Shuo and Qian, Yanmin and Qian, Yao and Wu, Jian and Zeng, Michael and Yu, Xiangzhan and Wei, Furu},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  volume={16},
  number={6},
  pages={1505--1518},
  year={2022},
}
```

Check https://github.com/microsoft/unilm/blob/master/wavlm/README.md for more information.

## Pre-trained models
| Model | Pre-training Dataset | Fine-tuning Dataset | Official Model | Adapted checkpoint for Fairseq finetuning |
|---|---|---|---|---|
| WavLM Base |  [960 hrs LibriSpeech](http://www.openslr.org/12)| -  | [Azure Storage](https://valle.blob.core.windows.net/share/wavlm/WavLM-Base.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D) <br> [Google Drive](https://drive.google.com/file/d/1BhTPLUkfN6e2xkqR8LEm9lByXbLY1IYd/view?usp=share_link) | [HuggingFace Repo](https://huggingface.co/wyz/WavLM_models_for_fairseq_finetuning/blob/main/WavLM-Base_cpt.pt) |
| WavLM Base+ | [60k hrs Libri-Light](https://github.com/facebookresearch/libri-light) + [10k hrs GigaSpeech](https://github.com/SpeechColab/GigaSpeech) + [24k hrs VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main)| -  |  [Azure Storage](https://valle.blob.core.windows.net/share/wavlm/WavLM-Base+.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D) <br> [Google Drive](https://drive.google.com/file/d/1-zlAj2SyVJVsbhifwpTlAfrgc9qu-HDb/view?usp=share_link) | [HuggingFace Repo](https://huggingface.co/wyz/WavLM_models_for_fairseq_finetuning/blob/main/WavLM-Base%2B_cpt.pt) |
| WavLM Large | [60k hrs Libri-Light](https://github.com/facebookresearch/libri-light) + [10k hrs GigaSpeech](https://github.com/SpeechColab/GigaSpeech) + [24k hrs VoxPopuli](https://github.com/facebookresearch/voxpopuli/tree/main)| -  | [Azure Storage](https://valle.blob.core.windows.net/share/wavlm/WavLM-Large.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D) <br> [Google Drive](https://drive.google.com/file/d/12-cB34qCTvByWT-QtOcZaqwwO21FLSqU/view?usp=share_link) | [HuggingFace Repo](https://huggingface.co/wyz/WavLM_models_for_fairseq_finetuning/blob/main/wavlm_large_cpt.pt) |

## Load a model
```python
ckpt_path = "/path/to/the/checkpoint.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
```
## Load Official Pre-Trained Models for Inference
```python
import torch
from fairseq.models.wavlm import WavLM, WavLMConfig

# load the pre-trained checkpoints
checkpoint = torch.load('/path/to/wavlm.pt')
cfg = WavLMConfig(**checkpoint['cfg'])
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()

# extract the the representation of last layer
wav_input_16khz = torch.randn(1,10000)
rep = model.extract_features(wav_input_16khz)[0]

# extract the the representation of each layer
wav_input_16khz = torch.randn(1,10000)
rep, layer_results = model.extract_features(wav_input_16khz, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
```
## Load adapted checkpoints for Fairseq finetuning
> NOTE: The adapted checkpoints are only guaranteed to have the same `model configuration` and `model parameters` as the official released models. The other parameters (such as `task hyperparameters` and `optimizer hyperparameters`) are just a template.
>
> Therefore, these adapted checkpoints are only suitable to be used as an initialization for fine-tuning in the downstream tasks, **NOT** for continuing pre-training.
>
> A typical use case is to set `--init /path/to/wavlm_xxxxx_cpt.pt` in `run/finetune_*.sh` scripts.

```python
import torch
from fairseq import checkpoint_utils


# Load the adapted checkpoints
ckpt_path = "/path/to/wavlm_large_cpt.pt"
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0]
# Load the official checkpoints (for comparison)
org_ckpt = torch.load("/path/to/wavlm_large.pt")

# Ensure that model parameters are the same as the official ones
state_dict = model.state_dict()
for k in org_ckpt['model']:
    torch.testing.assert_allclose(org_ckpt['model'][k], state_dict[k])

# Ensure that model configuration is the same as the official one
for k in org_ckpt['cfg']:
    if k == 'normalize':
        assert org_ckpt['cfg'][k] == cfg.task[k], k
        continue
    assert cfg.model[k] == org_ckpt['cfg'][k], k
```

## Train a new model

### Data preparation (Pretraining)

Run `local/prepare_wavlm_pretraining_data.sh` to create:
- `{train960,dev}.tsv` waveform list files
- `{train960,dev}.km` frame-aligned pseudo label files.
The `label_rate` is the same as the feature frame rate used for clustering,
which is 100Hz for MFCC features and 50Hz for WavLM features by default.

### Pre-train a WavLM model

Suppose `{train960,dev}.tsv` are saved at `/path/to/data/tsv_files`, `{train960,dev}.km`
are saved at `/path/to/kmeans_model/km_train960_WavLM9/kmeans_labels`, and the label rate is 100Hz.

To train a base model (12 layer transformer), run:
```sh
$ python fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq-py/examples/wavlm/config/pretrain \
  --config-name wavlm_base_iter1_librispeech_2x4gpu \
  task.data=/path/to/data/tsv_files task.label_dir=/path/to/kmeans_model/km_train960_WavLM9/kmeans_labels model.label_rate=100
```
Alternatively, run `run/pretrain_wavlm_iter1.sh`.


### Data preparation (Fine-tuning)
Run `local/prepare_wavlm_finetuning_data.sh` to create:
- `{train,valid}.tsv` waveform list files
- `{train,valid}.ltr` character-level label files.
- `{train,valid}.wrd` word-level label files.

### Fine-tune a WavLM model with a CTC loss

Suppose `{train,valid}.tsv` are saved at `/path/to/ll_10h`, and their
corresponding character transcripts `{train,valid}.ltr` are saved at
`/path/to/trans`.

To fine-tune a pre-trained WavLM model at `/path/to/checkpoint`, run
```sh
$ python fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq-py/examples/wavlm/config/finetune \
  --config-name wavlm_base_iter1_librispeech_10h_2x4gpu \
  task.data=/path/to/ll_10h task.label_dir=/path/to/trans \
  model.w2v_path=/path/to/checkpoint
```
Alternatively, run `run/finetune_wavlm_iter1.sh`.

### Data preparation (Decoding)
Run `local/prepare_wavlm_decoding_data.sh` to create:
- `test.tsv` waveform list files
- `test.ltr` character-level label files.
- `test.wrd` word-level label files.

### Decode a WavLM model

Suppose the `test.tsv` and `test.ltr` are the waveform list and transcripts of
the split to be decoded, saved at `/path/to/test_clean`, and the fine-tuned model is
saved at `/path/to/checkpoint`. We support three decoding modes:
- Viterbi decoding: greedy decoding without a language model
- KenLM decoding: decoding with an arpa-format KenLM n-gram language model
- Fairseq-LM deocding: decoding with a Fairseq neural language model


#### Viterbi decoding

`task.normalize` needs to be consistent with the value used during fine-tuning.
Decoding results will be saved at
`/path/to/experiment/directory/decode/viterbi/test`.

```sh
$ python examples/speech_recognition/new/infer.py \
  --config-dir /path/to/fairseq-py/examples/wavlm/config/decode \
  --config-name infer_viterbi \
  task.data=/path/to/test_clean \
  task.normalize=[true|false] \
  decoding.exp_dir=/path/to/experiment/directory \
  common_eval.path=/path/to/checkpoint
  dataset.gen_subset=test \
```
Alternatively, run `run/decode_wavlm_iter1.sh`.

#### KenLM / Fairseq-LM decoding

Suppose the pronunciation lexicon and the n-gram LM are saved at
`/path/to/lexicon` and `/path/to/arpa`, respectively. Decoding results will be
saved at `/path/to/experiment/directory/decode/kenlm/test`.

```sh
$ python examples/speech_recognition/new/infer.py \
  --config-dir /path/to/fairseq-py/examples/wavlm/config/decode \
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


---------

### Data preparation (Fine-tuning on mixture data for target speech recognition)
Run `../tshubert/local/prepare_tshubert_finetuning_data.sh` to create:
- `{train,valid}.tsv` waveform list files
- `{train,valid}.ltr` character-level label files.
- `{train,valid}.wrd` word-level label files.
- `{train,valid}.emb` speaker embedding list files.

### Fine-tune a WavLM model on mixture data (for target speech recognition) with a CTC loss

Suppose `{train,valid}.tsv` are saved at `/path/to/train_2mix`, and their
corresponding character transcripts `{train,valid}.ltr` are saved at
`/path/to/trans`.

To fine-tune a pre-trained WavLM model at `/path/to/checkpoint`, run
```sh
$ python fairseq_cli/hydra_train.py \
  --config-dir /path/to/fairseq-py/examples/wavlm/config/finetune \
  --config-name wavlm_base_iter1_libri2mix_train100_8x1gpu \
  task.data=/path/to/train_2mix task.label_dir=/path/to/trans \
  model.w2v_path=/path/to/checkpoint
```
Alternatively, run `run/finetune_mix_wavlm_iter1.sh`.

### Data preparation (Decoding on mixture data for target speech recognition)
Run `../tshubert/local/prepare_tshubert_decoding_data.sh` to create:
- `test.tsv` waveform list files
- `test.ltr` character-level label files.
- `test.wrd` word-level label files.
- `test.emb` speaker embedding list files.

### Decode a WavLM model on mixture data for target speech recognition

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
  --config-dir /path/to/fairseq-py/examples/wavlm/config/decode \
  --config-name infer_viterbi_mix \
  task.data=/path/to/test_2mix \
  task.normalize=[true|false] \
  decoding.exp_dir=/path/to/experiment/directory \
  common_eval.path=/path/to/checkpoint
  dataset.gen_subset=test \
```
Alternatively, run `run/decode_mix_wavlm_iter1.sh`.
