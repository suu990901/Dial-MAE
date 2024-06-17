# Dial-MAE
Dial-MAE is a transformers based Masked Auto-Encoder post-training architecture designed for Retrieval-based Dialogue Systems. Details can be found in [Dial-MAE:ConTextual Masked Auto-Encoder for Retrieval-based Dialogue Systems](https://aclanthology.org/2024.naacl-long.47.pdf)(NAACL 2024)

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
@inproceedings{su-etal-2024-dial,
    title = "Dial-{MAE}: {C}on{T}extual Masked Auto-Encoder for Retrieval-based Dialogue Systems",
    author = "Su, Zhenpeng  and
      W, Xing  and
      Zhou, Wei  and
      Ma, Guangyuan  and
      Hu, Songlin",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.47",
    pages = "820--830",
    abstract = "Dialogue response selection aims to select an appropriate response from several candidates based on a given user and system utterance history. Most existing works primarily focus on post-training and fine-tuning tailored for cross-encoders. However, there are no post-training methods tailored for dense encoders in dialogue response selection. We argue that when the current language model, based on dense dialogue systems (such as BERT), is employed as a dense encoder, it separately encodes dialogue context and response, leading to a struggle to achieve the alignment of both representations. Thus, we propose Dial-MAE (Dialogue Contextual Masking Auto-Encoder), a straightforward yet effective post-training technique tailored for dense encoders in dialogue response selection. Dial-MAE uses an asymmetric encoder-decoder architecture to compress the dialogue semantics into dense vectors, which achieves better alignment between the features of the dialogue context and response. Our experiments have demonstrated that Dial-MAE is highly effective, achieving state-of-the-art performance on two commonly evaluated benchmarks.",
}
```
