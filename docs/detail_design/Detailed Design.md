# Detailed Design

## Background

As a key concept in natural language processing, knowledge graph (KG) presents an approach to organize knowledge points in certain fields and visualize their relations. Several successful instances of KG, such as *BioBERT* in biological medicine, has been published. But up till now, no instance of KG in civil engineering has been published yet. Thus, this project has an objective of constructing a knowledge graph in the field of civil engineering.

Following steps are taken in the construction of our KG: named entity recognition (NER), word embedding and relation extraction. NER recognize named entities (mostly field-related terms) and extracts them from sentence. Word embeddings maps entities to vectors, which is later used in finding the most unrelated entity in the sentence and giving related words of entity specified. Relation extraction classifies the relations between every two entities in the sentence given.

The KG constructed by us was initially aimed at tunnel engineering. With the generalization ability of model, we has extended the scope of KG into civil engineering currently.

