---
library_name: transformers
language:
- ta
license: mit
base_model: microsoft/speecht5_tts
tags:
- generated_from_trainer
datasets:
- Common Voice (Tamil)
model-index:
- name: SpeechT5 fine-tuning for Tamil
  results: [accurate result]
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# SpeechT5 fine-tuning for Tamil

![Banner](https://github.com/anirxudh/Python/blob/main/speech-to-text.jpg)

This model is a fine-tuned version of [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts) on the Tamil language using the **Common Voice** dataset. It achieves the following results on the evaluation set:
- Loss: 0.4123

# SpeechT5 Model Fine-tuned for Tamil Language

## Model Description
The SpeechT5 model is a versatile speech processing model designed for tasks such as text-to-speech (TTS), automatic speech recognition (ASR), and other speech-related applications. This version of the model has been **fine-tuned specifically for the Tamil language**, making it well-suited for Tamil text-to-speech and automatic speech recognition tasks.

The fine-tuned model has been trained to:
- Accurately recognize and synthesize **Tamil speech**.
- Improve the pronunciation and understanding of diverse Tamil dialects, ensuring higher accuracy in tasks like speech synthesis and recognition for Tamil.

## Intended Uses & Limitations

### Intended Uses:
- **Text-to-Speech (TTS)**: The model can convert Tamil text into accurate and natural-sounding speech.
- **Automatic Speech Recognition (ASR)**: This model can transcribe Tamil speech with higher accuracy than general-purpose ASR systems.

### Limitations:
- **Limited Domains**: The model is trained on a general Tamil dataset and may not perform as well on technical or domain-specific vocabulary.
- **Language Restriction**: This fine-tuned version is specific to **Tamil speech** and is not trained for other languages.

## Training and Evaluation Data
- **Training Dataset**: The model was fine-tuned using the **Common Voice** dataset for Tamil. This dataset is designed to cover a wide range of Tamil dialects and styles of speaking.
  - **Size**: It contains 5 hours of Tamil speech data.
  - **Content**: The dataset includes diverse voices and speech patterns, helping the model generalize to various Tamil dialects.

## Training Procedure
1. **Preprocessing**: The input data was preprocessed to ensure that both the text and audio were properly aligned, with steps such as text normalization and feature extraction from audio samples.
   - **Text Normalization**: The text normalization process handled abbreviations and common Tamil speech constructs.
   - **Audio Preprocessing**: Spectrograms were generated from raw audio files using a standard feature extraction pipeline.

2. **Model Architecture**: SpeechT5's architecture was leveraged for its ability to handle both ASR and TTS tasks effectively. The fine-tuning process adapted the output layer to better handle Tamil speech.

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 32
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 4000
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch   | Step | Validation Loss |
|:-------------:|:-------:|:----:|:---------------:|
| 0.5031        | 3.5778  | 1000 | 0.4321          |
| 0.4908        | 7.1556  | 2000 | 0.4189          |
| 0.4812        | 10.7335 | 3000 | 0.4152          |
| 0.4709        | 14.3113 | 4000 | 0.4123          |


### Framework versions

- Transformers 4.46.0.dev0
- Pytorch 2.4.1+cu121
- Datasets 3.0.1
- Tokenizers 0.20.1
