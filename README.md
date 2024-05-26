# VisLingInstruct
This project is the key code (such as Instruction Alignment Score, IAS) for the paper "[VisLingInstruct: Elevating Zero-Shot Learning in Multi-Modal Language Models with Autonomous Instruction Optimization](https://arxiv.org/abs/2402.07398)".

The IAS is the most critical component of VisLingInstruct, utilized for assessing the quality of instructions. Our experiment is based on the [LAVIS](https://github.com/salesforce/LAVIS) library, which we expanded upon. For example, in the `lavis/models/blip2_models/blip2_vicuna_instruct.py` file, we added the `calculate_ias` function. For further details, please refer to `mmlm_vicuna.py`.

# Train
The entry point for the training script is in the `train.py` file, and the training can be initiated by maintaining the `train.sh` script. The related configuration files are located in the `train_configs` folder. Parameters prefixed with "need:" indicate mandatory adjustments, which are typically paths.

# Inference
Some inference-related examples are demonstrated in the `test.py` file, and the related configuration files are located in the `eval_configs` folder.

## Acknowledgement
- [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2) The model architecture of BLIVA follows BLIP-2. Don't forget to check this great open-source work if you don't know it before. 
- [Lavis](https://github.com/salesforce/LAVIS) The codebase we built upon.
- [Vicuna](https://github.com/lm-sys/FastChat) Vicuna-13B demonstrates fantastic language ability and it's open source. 
- [BLIVA](https://github.com/mlpc-ucsd/BLIVA) A Simple Multimodal LLM for Better Handling of Text-rich Visual Questions.

## Citation
```bibtex
@article{zhu2024vislinginstruct,
  title={VisLingInstruct: Elevating Zero-Shot Learning in Multi-Modal Language Models with Autonomous Instruction Optimization},
  author={Zhu, Dongsheng and Tang, Xunzhu and Han, Weidong and Lu, Jinghui and Zhao, Yukun and Xing, Guoliang and Wang, Junfeng and Yin, Dawei},
  journal={arXiv preprint arXiv:2402.07398},
  year={2024}
}
```