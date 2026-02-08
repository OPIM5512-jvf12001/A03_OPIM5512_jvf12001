# This is my A03 Sampling Assignment. 

This project compares and contrasts three possible sampling methods to deal with imbalanced data and it's usage in training and testing a machine learning algorithm.

## How to Run:

1. Clone the repository

2. Install dependencies using pip install -r requirements.txt

3. Check line 169 of sampling_compare.py and adjust the number of loop iterations. This value is currently set to 1,000 which may take a while to run.

4. Run sampling_compare.py


## Expected Output:
sampling_compare.py will output a number of classification reports and confusion matrices comparing the performance of three different sampling methods used to deal
with imbalanced data. The methods are Majority Undersampling, Minority Oversampling, and SMOTE. Finally, the file will produce a histogram smote_loop_performance.png
showing the classification report stats of the model when run over "i" different training/test splits of the data.