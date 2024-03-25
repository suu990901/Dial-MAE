# Dial-MAE
Dial-MAE is a transformers based Masked Auto-Encoder pretraining architecture designed for Retrieval-based Dialogue Systems. Details can be found in [Dial-MAE:ConTextual Masked Auto-Encoder for Retrieval-based Dialogue Systems](https://arxiv.org/abs/2306.04357)(NAACL 2024)

## Dependencies

Please refer to [PyTorch Homepage](https://pytorch.org/get-started/previous-versions/) to install a pytorch version suitable for your system.

Dependencies can be installed by running codes below. Specifically, we use transformers=4.17.0 for our experiments. Other versions should also work well.

```
apt-get install parallel
pip install transformers==4.17.0 datasets nltk tensorboard pandas tabulate
```
## Training
Please refer to examples below for reproducing our works.
1. [Post-training](https://github.com/suu990901/Dial-MAE/tree/main/dialogue_post_train)
2. [Fine-tuning](https://github.com/suu990901/Dial-MAE/tree/main/dialogue_finetune)

## Citation
If you find our work useful, please consider to cite our paper.
```
@article{su2023contextual,
  title={ConTextual Masked Auto-Encoder for Retrieval-based Dialogue Systems},
  author={Su, Zhenpeng and Wu, Xing and Zhou, Wei and Ma, Guangyuan and others},
  journal={arXiv preprint arXiv:2306.04357},
  year={2023}
}
```
