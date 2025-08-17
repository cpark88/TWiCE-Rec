# Think Wise, Collaborate Effectively: A Rationale-Aware LLM-Based Recommender with Reinforcement Learning from Collaborative Signals

## Overview
***
Code for our Paper "Think Wise, Collaborate Effectively: A Rationale-Aware LLM-Based Recommender with Reinforcement Learning from Collaborative Signals." 
We referred to the source code of Open-R1 (<https://github.com/huggingface/open-r1>).

## Abstract
***
Large Language Models (LLMs) have recently emerged as powerful reasoning engines in recommender systems, generating natural-language explanations that foster user engagement.
However, their recommendation performance remains limited, as they lack exposure to collaborative user-item interaction patterns.
In contrast, collaborative filtering (CF) models achieve strong performance by learning from these behavioral patterns at scale.
To unify the strengths of \textbf{both paradigms}, we propose \textbf{TWiCE-Rec} (\underline{T}hink \underline{Wi}se, \underline{C}ollaborate \underline{E}ffectively), a rationale-aware LLM-based recommender that incorporates collaborative user-item interactions.
In the first stage, we construct a rationale dataset by applying in-context learning with self-annotated curation. 
A state-of-the-art LLM is guided to generate persuasive rationales that explain the causal relationship between the user’s interaction sequence and the ground-truth next item, resulting in a curated post-hoc training dataset.
In the second stage, we perform multi-task instruction-tuned adaptation—based on the rationale-augmented training dataset—comprising item description generation and both non-reasoning and reasoning-based sequential recommendation, to equip the LLM with the ability to generate rationales that reflect how user preferences align with item characteristics.
Finally, we aim to enhance the LLM’s recommendation performance by incorporating user-item interaction patterns derived from the CF-Rec model.
To achieve this, we propose a confidence-weighted reinforcement learning strategy that adjusts rewards in proportion to both the LLM’s prediction alignment with the ground-truth and the confidence from the pretrained CF-Rec model.
Our method outperforms both CF- and LLM-Rec models on Amazon datasets in terms of recommendation performance and rationale quality. 
In an online A/B test, it achieved about 8\% higher click-through rate than existing models, demonstrating practical value.

## Environment Setting
***
```bash
pip install -r requirements.txt
```


## Dataset
***
To facilitate smooth testing, we have uploaded a small-sized temporary raw data in src/open_r1/sasrec/amazon_dataset. We have also uploaded the full dataset for the Grocery and Gourmet Food domain at the following path, so our paper can be reproduced using this data.  
(Updated) We openly release all the data used in our paper. This dataset is reconstructed from the Amazon Review 2023 corpus, and the recommendation rationales were generated and refined using a large-scale LLM.

https://drive.google.com/drive/folders/1k8FS7F3woEZJhBabd5_BQn56I8bd5abP

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

## Stage 2 (Rationale-Aware Knowledge Alignment)
***

```bash
bash sft.sh
```

## Stage 3 (Collaborative Preference Alignment)
1. **Start the Stage-2 SFT Model Server (vLLM-based)**

On a machine with a specific GPU, launch the Stage-2 SFT model using `vLLM`:
***

```bash
bash grpo_step1.sh
```

2.	**Run GRPO Training**
On a different GPU (or machine), run the GRPO training by executing:

```bash
bash grpo_step2.sh
```
