# Arabic Named Entity Recognition (NER) with AraBERT
1. Project Overview

This project implements an Arabic Named Entity Recognition (NER) system using a transformer-based architecture. The model is fine-tuned on the ANERCorp dataset and is designed to handle the linguistic and morphological complexity of the Arabic language.

The system performs token-level classification to identify the following entity types:

PERS – Person

LOC – Location

ORG – Organization

MISC – Miscellaneous

The model is built on top of AraBERT, a pretrained language model specifically optimized for Arabic.

2. Technical Architecture

The core model used in this project is:

aubmindlab/bert-base-arabertv02


Due to the highly inflectional nature of Arabic, several custom strategies were implemented to ensure accurate entity recognition.

2.1 Sentence-Boundary Awareness

Instead of arbitrarily chunking long sequences, the text is segmented into sentences using punctuation delimiters:

.

!

?

This preserves semantic coherence and improves contextual understanding during token classification.

2.2 Sub-word Label Alignment

AraBERT relies on WordPiece tokenization, which may split a single word into multiple sub-tokens. To correctly align labels:

The first sub-token of a word receives the original NER label.

All subsequent sub-tokens are assigned a label value of -100.

This ensures that only valid tokens contribute to the loss during training.

2.3 Dynamic Padding

The project uses Hugging Face’s:

DataCollatorForTokenClassification


Dynamic padding is applied at the batch level, meaning sequences are padded only to the length of the longest sample in each batch. This significantly reduces memory usage and improves training efficiency.

3. Dataset Implementation

The model is trained and evaluated using the ANERCorp dataset, which provides word-level annotations for Arabic NER tasks.

Feature	Description
Dataset Source	asas-ai/ANERCorp
Annotation Level	Word-level
Labeling Scheme	BIO (Beginning, Inside, Outside)
Preprocessing	Sentence grouping based on punctuation
Data Split	Standard train/test distribution
4. How to Use
4.1 Installation

Install the required dependencies:

pip install transformers datasets seqeval evaluate accelerate -q

4.2 Running Inference

You can perform inference using the Hugging Face pipeline API after loading the fine-tuned model.

from transformers import pipeline

Load the fine-tuned model
ner_pipeline = pipeline(
    "token-classification",
    model="./arabert-ner",
    aggregation_strategy="simple"
)

text = "أعلن المدير التنفيذي لشركة أبل تيم كوك عن افتتاح فرع جديد في الرياض."
results = ner_pipeline(text)

for entity in results:
    print(f"Entity: {entity['word']}, Label: {entity['entity_group']}, Score: {entity['score']:.2f}")

5. Performance Evaluation

The model is evaluated using the seqeval metric, which measures performance at the entity level rather than individual tokens.

Precision: High accuracy in identifying relevant entities

Recall: Strong ability to detect all entity instances

F1-Score: Balanced harmonic mean of precision and recall

This evaluation approach provides a realistic assessment of NER performance in real-world Arabic text.

6. Conclusion

This project demonstrates an effective approach to Arabic NER by combining:

A pretrained Arabic transformer (AraBERT)

Careful sentence segmentation

Accurate sub-word label alignment

Efficient dynamic padding

The result is a robust and scalable NER system suitable for downstream Arabic NLP applications.
