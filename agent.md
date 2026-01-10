# AI SMS Spam Filter (Industrial Grade)

This file captures the current project context for continuity across sessions.

Project Name: AI SMS Spam Filter (Industrial Grade)
Tech Stack: Python 3.12, Hugging Face Transformers (DistilBERT), TensorFlow/Keras, Regex

Key Features:
- Hybrid Architecture: DistilBERT classifier + regex-based rule engine
- Data Augmentation: adversarial samples for sextortion, insider stock, pig butchering
- Whitelist mechanism and false positive fixes for common domains

Current Status:
- Model migration complete (Bi-LSTM -> DistilBERT)
- Passed 20 hard test cases

Notes:
- Uses TFDistilBertForSequenceClassification fine-tuning (lr 5e-5, 2 epochs)
- Hybrid rule engine includes context-aware regex logic and URL whitelisting
