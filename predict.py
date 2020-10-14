import numpy as np
import csv
import sys
import pickle
from validate import validate
"""
Predicts the target values for data in the file at 'test_X_file_path', using the weights learned during training.
Writes the predicted values to the file named "predicted_test_Y_lg.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data():
    X = np.genfromtxt('train_X_lg_v2.csv', dtype = 'float64',\
                      skip_header = 1 ,delimiter= ',')
    Y = np.genfromtxt('train_Y_lg_v2.csv', dtype = np.int32,\
                      delimiter = ',')

    return X,Y

def data_per_class(X,Y, class_label):
    class_X = np.copy(X)
    class_Y = np.copy(Y)
    class_Y = np.where(class_Y == class_label, 1, 0)
    return class_Y

def import_data_and_weights(test_X_file_path, weights_file_path):
    X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    weights = pickle.load(open(weights_file_path, 'rb'))
    return X, weights

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    s = 1.0/(1.0 + np.exp(-x))
    return s
    

def predict_target_values(test_X, weights):
    pred_Y = []
    pred_Y0 = sigmoid(np.dot(weights[0][0].T, test_X.T) + weights[0][1])
    pred_Y1 = sigmoid(np.dot(weights[1][0].T, test_X.T) + weights[1][1])
    pred_Y2 = sigmoid(np.dot(weights[2][0].T, test_X.T) + weights[2][1])
    pred_Y3 = sigmoid(np.dot(weights[3][0].T, test_X.T) + weights[3][1])
    pred_Y0, pred_Y1, pred_Y2, pred_Y3 = pred_Y0.T, pred_Y1.T, pred_Y2.T, pred_Y3.T
    for i in range(len(test_X)):
        if(pred_Y1[i] > pred_Y0[i] and pred_Y1[i] > pred_Y2[i] and pred_Y1[i] > pred_Y3[i]):
            pred_Y.append(1)
        elif(pred_Y2[i] > pred_Y0[i] and pred_Y2[i] > pred_Y1[i] and pred_Y2[i] > pred_Y3[i]):
            pred_Y.append(2)
        elif(pred_Y3[i] > pred_Y0[i] and pred_Y3[i] > pred_Y2[i] and pred_Y3[i] > pred_Y1[i]):
            pred_Y.append(3)
        else:
            pred_Y.append(0)
    pred_Y = np.array(pred_Y)  
    return pred_Y
            
            
    
    # Write your code to Predict Target Variables
    # HINT: You can use other functions which you've already implemented in coding assignments.


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    X, Y = import_data()
    test_X, weights = import_data_and_weights(test_X_file_path, "weights.pkl")
    pred_Y = predict_target_values(test_X, weights)
    write_to_csv_file(pred_Y, "predicted_test_Y_lg.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    validate(test_X_file_path, actual_test_Y_file_path="train_Y_lg_v2.csv") 