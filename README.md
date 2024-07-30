# Respiratory Prediction in Radiation Therapy


## Introduction
In radiation therapy, the respiratory gating method is widely used to account for the patient's breathing during treatment. However, this method introduces delays between beam on and off times. These delays can reduce the accuracy of the treatment and potentially harm normal tissues, making delay compensation necessary.

## Models

The models used in our research are:
- **LSTM (Long Short-Term Memory)**
- **Bi-LSTM (Bidirectional Long Short-Term Memory)**
- **Transformer**

## Delay Times

The delay times considered in our study are:
- **300 ms**
- **500 ms**
- **700 ms**

## Repository Structure
- **data/**: Contains the datasets used for training and testing.
- **models/**: Includes the implementation of the LSTM, Bi-LSTM, and Transformer models.
- **results/**: Stores the results and performance metrics of the models.

## Publications

For more detailed information, refer to our published paper:
- **Clinical applicability of deep learning-based respiratory signal prediction models for four-dimensional radiation therapy**
- Authors: [Sangwoon Jeong]
- Published in: [PLOS ONE]
- [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0275719]
  
## Contact
For any questions or inquiries, please contact [sangwoonjeong93@gmail.com].











