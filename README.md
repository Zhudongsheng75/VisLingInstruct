# VisLingInstruct
This project is the Instruction Alignment Score (IAS) code for the paper "[VisLingInstruct: Elevating Zero-Shot Learning in Multi-Modal Language Models with Autonomous Instruction Optimization](https://arxiv.org/abs/2402.07398)".

The IAS is the most critical component of VisLingInstruct, utilized for assessing the quality of instructions.

Our experiment is based on the [LAVIS](https://github.com/salesforce/LAVIS) library, which we expanded upon. For example, in the `lavis/models/blip2_models/blip2_vicuna_instruct.py` file, we added the `calculate_ias` function. For further details, please refer to `mmlm_vicuna.py`.

