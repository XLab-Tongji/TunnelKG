# 需求规约

​                                        

---

### 详述

   ![NER_use_case](image/ner_use_case.png)

### Use Case: View Results

#### Use Case Name

-  NER Model Manage

#### Identifier

- UC01

#### **Summary Description**

- Background Program manage the model for prediction

#### Actor

- Background Program

#### Frequency

- high

#### State

- pass

#### **Pre-Condition**

- The server storing model is working

#### **Post-Condition**

- The background program can train model or predict result.

#### Extended Use Case

- None

#### Included Use Case

- Train Model

#### Basic Flows

1. Background program turn on model service
2. Others use the model.

#### Alternative Flows

- None

-----------

### Included Use Case: Prediction of Entities

#### Use Case Name

-  Prediction of Entities

#### Identifier

- UC02

#### **Summary Description**

- Background program prepare the model for prediction.
- User input the sentence for prediction of entities. 

#### Actor

- User
- Background program

#### Frequency

- high

#### State

- pass

#### **Pre-Condition**

- There is an active connection to the model and network.

#### **Post-Condition**

- The background program receives the input sentence
- The background program outputs the prediction result.

#### Extended Use Case

- None

#### Included Use Case

- None

#### Basic Flows

1. User open the website
2. User click the input bar
3. User input the sentence

#### Alternative Flows

- User can rewrite the sentence in input bar again





------

### Included Use Case: Train model

#### Use Case Name

-  Train model

#### Identifier

- UC03

#### **Summary Description**

- Background program train model to achieve better performance

#### Actor

- Background program

#### Frequency

- high

#### State

- pass

#### **Pre-Condition**

- There is a suitable device for training

#### **Post-Condition**

- The performance of model changes.

#### Extended Use Case

- None

#### Included Use Case

- Process Data
- Load Pretrained Model
- Fine Tuning
- Evaluate

#### Basic Flows

1. User click the confirm button
2. The website received the unmatched word from background program
3. The website show the unmatched word to user
4. User view unmatched word in its area

#### Alternative Flows

- None



-----

### Included Use Case: Process Data

#### Use Case Name

-  Process data

#### Identifier

- UC04

#### **Summary Description**

- Data manager provide the original dataset.
- Background program process the data to meet the needs of training

#### Actor

- Data manager
- Background program

#### Frequency

- high

#### State

- pass

#### **Pre-Condition**

- Original dataset is ready

#### **Post-Condition**

-  Produce a dataset tailored for training

#### Extended Use Case

- None

#### Included Use Case

-  None

#### Basic Flows

1. Data manager provides original dataset.
2. The background program runs a script to modify dataset
3. The background program saves modified dataset.

#### Alternative Flows

- None



----

### Included Use Case: Load Pretrained Model

#### Use Case Name

-  Load Pretrained Model

#### Identifier

- UC05

#### **Summary Description**

- Background program load ALBERT or BERT for training

#### Actor

- Background program

#### Frequency

- high

#### State

- pass

#### **Pre-Condition**

- There is an downloaded pretrained model
- There is enough free memory in CUDA

#### **Post-Condition**

- The pretrained model is loaded into CUDA

#### Extended Use Case

- None

#### Included Use Case

- None

#### Basic Flows

1. Background program runs script
2. The pretrained model is loaded

#### Alternative Flows

- Return an error if there are no space left in device



----

### Included Use Case: Fine Tuning

#### Use Case Name

-  Fine Tuning

#### Identifier

- UC06

#### **Summary Description**

- Background program fine-tune pretrained model to achieve better performance

#### Actor

- Background program

#### Frequency

- high

#### State

- pass

#### **Pre-Condition**

- Pretrained model is loaded
- Pre-processed dataset is ready

#### **Post-Condition**

- The performance of model is changed

#### Extended Use Case

- None

#### Included Use Case

- None

#### Basic Flows

1. Background program modify model's parameters
2. Background program run model on training dataset

#### Alternative Flows

- None



----

### Included Use Case: Evaluate

-  Load Embedding

#### Identifier

- UC07

#### **Summary Description**

- Background program evaluate the performance of the model

#### Actor

- Background program

#### Frequency

- high

#### State

- pass

#### **Pre-Condition**

- Model is trained for some epochs

#### **Post-Condition**

- The background program return a score

#### Extended Use Case

- None

#### Included Use Case

- None

#### Basic Flows

1. Background program start evaluate
2. Background program run model on testing dataset
3. Background program return precision and f1 score

#### Alternative Flows

- None


