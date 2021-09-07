# RNN-Language-Classifier
A Language Classifier powered by [Recurrent Neural Network(RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network) implemented in Python without AI libraries.

## Features
The classifier classifies a word in **English**, **Spanish**, **Finnish**, **Dutch**, or **Polish**. The classifier outputs correctly at a rate of approximately 85%. 
It is purely implemented with numpy and built-in libraries.


## Model Architecture
- Input Layer: 47 nodes representing 47 different characters
- Output Layer: 5 nodes representing 5 languages

The technique used in this project is called  [Recurrent Neural Network(RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network):<br/>
<br/><img src="https://cs188.ml/assets/images/rnn.png" height="150px"/><br/><br/>
Here, an RNN is used to encode the word “c-a-t” into a fixed-size vector h3.

## Sample Run
#### Training until validation accuracy achieve a certain level:
```
epoch 1 iteration 24 validation-accuracy 43.0%
  shaking    English ( 22.4%) Pred: Dutch   |en 22%|es 20%|fi 18%|nl 26%|pl 14%
  relaxing   English ( 23.7%) Pred: Dutch   |en 24%|es 20%|fi 18%|nl 25%|pl 13%
  prophecy   English ( 17.6%) Pred: Spanish |en 18%|es 24%|fi 24%|nl 16%|pl 19%
  tiroteo    Spanish ( 25.8%)               |en 21%|es 26%|fi 18%|nl 18%|pl 17%
  vientre    Spanish ( 24.2%)               |en 17%|es 24%|fi 21%|nl 21%|pl 17%
  estupenda  Spanish ( 31.4%)               |en 16%|es 31%|fi 18%|nl 19%|pl 16%
  osti       Finnish ( 21.2%) Pred: Polish  |en 15%|es 19%|fi 21%|nl 20%|pl 25%
  veljensä   Finnish ( 19.8%) Pred: Spanish |en 21%|es 22%|fi 20%|nl 20%|pl 18%
  aikoinaan  Finnish ( 22.3%)               |en 15%|es 21%|fi 22%|nl 21%|pl 21%
  betwijfel  Dutch   ( 22.8%) Pred: English |en 24%|es 23%|fi 15%|nl 23%|pl 15%
  merkte     Dutch   ( 17.1%) Pred: Spanish |en 17%|es 22%|fi 22%|nl 17%|pl 21%
  beseffen   Dutch   ( 24.5%)               |en 21%|es 19%|fi 21%|nl 25%|pl 15%
  kończę     Polish  ( 21.5%) Pred: Spanish |en 17%|es 23%|fi 20%|nl 18%|pl 21%
  firmy      Polish  ( 20.7%) Pred: Finnish |en 15%|es 22%|fi 23%|nl 19%|pl 21%
  decyzje    Polish  ( 16.2%) Pred: Dutch   |en 19%|es 22%|fi 20%|nl 23%|pl 16%

.
.
.

epoch 6 iteration 153 validation-accuracy 84.2%
  shaking    English ( 86.4%)               |en 86%|es  0%|fi  1%|nl 12%|pl  1%
  relaxing   English ( 84.6%)               |en 85%|es  0%|fi  0%|nl 15%|pl  0%
  prophecy   English ( 54.2%)               |en 54%|es  0%|fi  0%|nl  4%|pl 41%
  tiroteo    Spanish ( 38.9%)               |en 12%|es 39%|fi 36%|nl  6%|pl  8%
  vientre    Spanish ( 43.4%)               |en 19%|es 43%|fi  2%|nl 29%|pl  7%
  estupenda  Spanish ( 75.2%)               |en  1%|es 75%|fi 15%|nl  2%|pl  7%
  osti       Finnish ( 75.7%)               |en  1%|es  1%|fi 76%|nl  3%|pl 20%
  veljensä   Finnish ( 81.7%)               |en  0%|es  1%|fi 82%|nl 17%|pl  0%
  aikoinaan  Finnish ( 99.9%)               |en  0%|es  0%|fi100%|nl  0%|pl  0%
  betwijfel  Dutch   ( 98.7%)               |en  1%|es  0%|fi  0%|nl 99%|pl  1%
  merkte     Dutch   ( 71.9%)               |en 10%|es  1%|fi  6%|nl 72%|pl 10%
  beseffen   Dutch   ( 96.6%)               |en  2%|es  0%|fi  0%|nl 97%|pl  0%
  kończę     Polish  (100.0%)               |en  0%|es  0%|fi  0%|nl  0%|pl100%
  firmy      Polish  ( 29.4%) Pred: English |en 59%|es  5%|fi  2%|nl  5%|pl 29%
  decyzje    Polish  ( 87.7%)               |en  1%|es  1%|fi  0%|nl 10%|pl 88%
```

#### Test Results:
```
test set accuracy is: 83.800000%
```

#### User Input:
```
word: tervetuloa # welcome
predicted language is: Finnish, with a confidence of 80.011147%

word: ciudades # cities
predicted language is: Spanish, with a confidence of 88.442353%

word: właź # hatch
predicted language is: Polish, with a confidence of 99.979566%

word: algorithm
predicted language is: English, with a confidence of 79.893499%

word: resolution
predicted language is: English, with a confidence of 94.786443%

word: ademt # breathe
predicted language is: Dutch, with a confidence of 47.399565%

word: invitar # invite
predicted language is: Spanish, with a confidence of 93.986880%
```

## Dependencies
You will need `numpy` for this project
```
pip install numpy
```

## How To Use
clone this project or download the zip file
```
py run.py
```

## Improvements To Make
- support save & load models
- classify more languages
- improve accuracy
- classify a sentence or paragraph instead of words
- ...
 
## *Reference*
The dataset `lang_id.npz`, image demonstrating `RNN`, and project skeleton are from [cs188.ml](https://cs188.ml/).
