# MaoMao Disambiguator Dictionary

![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](#)

Context-aware dictionary that automatically ranks definitions by semantic relevance. This project will be used to enhance ![MaoMao-Dict](https://github.com/MaoMao-Corp/MaoMao-Dict) performance and making it fully local, privacy-respecting, free of charge, and released as free software in the “free as in freedom” sense. Uses the GNU Collaborative International Dictionary of English for the definitions.

## Motivation
I'm tired of having to stop every time I encounter a word I don't know, or don't recognize in a particular context. Existing solutions such as popup dictionaries save you from opening new tabs but they still force you to read through every definition entry - sometimes there are a lot - and make a bold guess about which one better fits the context.

Given that English is not my first language, I sometimes struggle to infer the correct definition, which can be really time consuming.

This word disambiguator was created for personal use in mind, to make our daily lives better.

## General Overview
Given a target word and its surrounding context window, the system predicts its part of speech and lemma, then filters out dictionary entries that don't match these attributes. The remaining candidate meanings are passed through a two-stage pipeline:

- **Bi-encoder retrieval**:
Both the context window and all candidate meanings are embedded using a bi-encoder. Candidates are ranked by similarity, computed via dot product with the context embedding.

- **Cross-encoder reranking**:
The top candidates from the retrieval stage are then evaluated by a cross-encoder, which provides a more precise ranking.

Finally, the scores from both stages are combined using a fusion ensemble strategy to produce the final ranked list of candidate meanings presented to the user.

## 🚧 Status 🚧
The project runs successfully and includes a functional CLI, but it is not yet fully integrated with the frontend. We are actively experimenting with alternative models and optimization strategies to reduce computational cost and make the tool accessible to as many users as possible.

Looking forward, we plan to allow users to contribute to model improvement through federated learning, enabling community-driven fine-tuning while preserving privacy.

## Showcase
<img width="822" height="703" alt="image(1)" src="https://github.com/user-attachments/assets/eef2c55e-fadd-4d52-b79b-dbd4170e4195" />
