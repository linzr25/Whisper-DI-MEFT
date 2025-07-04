# APSIPA 2025 ASC Submission

The project is developed based on [Transformers](https://github.com/huggingface/transformers) 4.47.0

[whisper_modified](./whisper_modified) contains the implementations of Memory-Efficient Fine-Tuninig (MEFT) methods on the WhisperForAudioClassification class.

[train](./train) includes the Python scripts for training.

Methods are evaluated on the Mandarin subdialects (Ji-Lu, Jiang-Huai, Jiao-Liao, Lan-Yin, Southwestern, Zhongyuan) from [KeSpeech](https://github.com/KeSpeech/KeSpeech) Dataset. The dataset used in the training only includes `input_features` and `labels` items, and has been preprocessed by WhisperProcessor.

[mezo_trainer](./mezo_trainer) is the implementation of [MeZO](https://github.com/princeton-nlp/MeZO) on the trainer of [Transformers](https://github.com/huggingface/transformers) 4.47.0. Please refer to [Fine-Tuning Language Models with Just Forward Passes](https://arxiv.org/pdf/2305.17333) for more details of MeZO.
