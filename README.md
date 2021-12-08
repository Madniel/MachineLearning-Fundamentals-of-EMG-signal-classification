# MachineLearning-Fundamentals-of-EMG-signal-classification

Tasks

1. Load the MVC (https://chmura.put.poznan.pl/s/4UuSx0lfK53FA7I) signal, and the training signal (https://chmura.put.poznan.pl/s/38aeyGzigLEHLbp)
2. Determine its RMS and ZC features and their associated labels ('TRAJ_GT')
3. Assume that only samples for which the label has a value >=0 are processed in the classification task
4. Analyze the operation of the following program, which aims to estimate the accuracy of the classifier by k-fold validation method 
(https://scikit-learn.org/stable/modules/cross_validation.html)

5. Compare the results obtained with the classification results of the test signal (https://chmura.put.poznan.pl/s/7g3b2p7tljJJaNc) recorded in the same session. For this, train the classifier on the learning data and then make a prediction for the test set. 
6. For the results from task. 5, check the confusion matrix and consider which gestures are most frequently confused with each other, try to explain why?

7. Get the validation signal )https://chmura.put.poznan.pl/s/wuu8IZDRHxUrgXX) and determine its features. Generate a gesture prediction for moments of time of the index data.
(https://chmura.put.poznan.pl/s/3S2QorjXu0tUM0h) Remember not to change the value of random_state of the classifier if you do not implement task 7*. Choose the classifier with the best generalization properties for implementation, you can overtrain it according to your skills. Save the resulting label prediction file to an hdf file. The file should have an index corresponding to the index from the downloaded index file and a predictions column.

8. As homework, submit the source file implementing the learning and the predictions.hdf file containing the classification results for the validation set.
