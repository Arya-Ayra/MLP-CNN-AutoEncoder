Statistical Methods in AI

Instructor : Prof Ravi Kiran Sarvadevabhatla



In this assignment, you are required to work with W&B library. Here are some resources on W&B logging and reporting :

1. [Experiment management with Weights and Biases platform](https://broutonlab.com/blog/data-science-experiments-management-with-weights-and-biases-platform)
2. [Reports by W&B](https://www.youtube.com/watch?app=desktop&v=o2dOSIDDr1w)


1  Multinomial Logistic Regression

[ Marks : 40, Estimated Time : 2-3 days ]

Implement a Multinomial logistic regression model from scratch using numpy and pandas. You have to train this model on Wine Quality Dataset to classify

a wine’s quality based on the values of its various contents.

1. Dataset Analysis and Preprocessing [5 marks]
2. Describe the dataset using mean, standard deviation, min, and max values for all attributes.
3. Draw a graph that shows the distribution of the various labels across the entire dataset. You are allowed to use standard libraries like Matplotlib.
4. Partition the dataset into train, validation, and test sets. You can use sklearn for this.
5. Normalise and standarize the data. Make sure to handle the missing or inconsistent data values if necessary. You can use sklearn for this.
6. Model Building from Scratch [20 marks]
   1. Create a Multinomial Logistic Regression model from scratch and Use cross entropy loss as loss function and Gradient descent as the optimization algorithm (write seperate methods for these).
   2. Train the model, use sklearn classification report and print metrics on the validation set while training. Also, report loss and accuracy on train set.
7. Hyperparameter Tuning and Evaluation [15 marks]
8. Use your validation set and W&B logging to fine-tune the hyperparameters ( learning rate , epochs) for optimal results.
9. Evaluate your model on test dataset and print sklearn classification report.
2  Multi Layer Perceptron Classification [ Marks : 60, Estimated Time : 4-5 days ]

In this part, you are required to implement MLP classification from scratch using numpy, pandas and experiment with various activation functions and op- timization techniques, evaluate the model’s performance, and draw comparisons with the previously implemented multinomial logistic regression.

Use the same dataset as Task 1

1. Model Building from Scratch [20 marks]

Build an MLP classifier class with the following specifications:

1. Create a class where you can modify and access the learning rate, activa- tion function, optimisers, number of hidden layers and neurons.
1. Implement methods for forward propagation, backpropagation, and train- ing.
1. Different activation functions introduce non-linearity to the model and affect the learning process. Implement the Sigmoid, Tanh, and ReLU activation functions and make them easily interchangeable within your MLP framework.
1. Optimization techniques dictate how the neural network updates its weights during training. Implement methods for the Stochastic Gradient Descent (SGD), Batch Gradient Descent, and Mini-Batch Gradient Descent algo- rithms from scratch, ensuring that they can be employed within your MLP architecture.
2. Model Training & Hyperparameter Tuning using W&B [10 marks]

Effective tuning can vastly improve a model’s performance. Integrate Weights & Biases (W&B) to log and track your model’s metrics. Using W&B and your validation set, experiment with hyperparameters such as learning rate, epochs, hidden layer neurons, activation functions, and optimization techniques. You have to use W&B for loss tracking during training and to log effects of different activation functions and optimizers on model performance.

1. Log your scores - loss and accuracy on validation set and train set using W&B.
1. Report metrics: accuracy, f-1 score, precision, and recall. You are allowed to use sklearn metrics for this part.
1. You have to report the scores(ordered) for all the combinations of :
- Activation functions : sigmoid, tanh and ReLU (implemented from scratch)
- Optimizers : SGD, batch gradient descent, and mini-batch gradient descent (implemented from scratch).
4. Tune your model on various hyperparameters, such as learning rate, epochs, and hidden layer neurons.
- Plot the trend of accuracy scores with change in these hyperparam- eters.
- Report the parameters for the best model that you get (for the various values you trained the model on).
- Report the scores mentioned in 2.2.2 for all values of hyperparameters in a table.
3. Evaluating Model [10 marks]
1. Test and print the classification report on the test set. (use sklearn)
1. Compare the results with the results of the logistic regression model.
4. Multi-Label Classification [20 marks]

For this part, you will be training and testing your model on Multilabel dataset: ”advertisement.csv” as provided in Assignment 1.

1. Modify your model accordingly to classify multilabel data.
1. (a) Log your scores - loss and accuracy on validation set and train set using W&B.
2) Report metrics: accuracy, f-1 score, precision, and recall.
2) You have to report the scores(ordered) for all the combinations of :
- Activation functions : sigmoid, tanh and ReLU (implemented from scratch)
- Optimizers : SGD, batch gradient descent and mini-batch gra- dient descent (implemented from scratch).
4) Tune your model on various hyperparameters, such as learning rate, epochs, and hidden layer neurons.
- Plot the trend of accuracy scores with change in these hyperpa- rameters.
- Report the parameters for the best model that you get (for the various values you trained the model on).
- Report the scores mentioned in Point b for all values of hyper- parameters in a table.
3. Evaluate your model on the test set and report accuracy, f1 score, preci- sion, and recall.
3  Multilayer Perceptron Regression

[ Marks : 50, Estimated Time : 3-4 days ]

In this task, you will implement a Multi-layer Perceptron (MLP) for regression from scratch, and integrate Weights & Biases (W&B) for tracking and tun- ing. Using the Boston Housing dataset, you have to predict housing prices while following standard machine learning practices. In this dataset, the column MEDV gives the median value of owner-occupied homes in $1000’s.

1. Data Preprocessing [5 marks]
1. Describe the dataset using mean, standard deviation, min, and max values for all attributes.
1. Draw a graph that shows the distribution of the various labels across the entire dataset. You are allowed to use standard libraries like Matplotlib.
1. Partition the dataset into train, validation, and test sets.
1. Normalise and standarize the data. Make sure to handle the missing or inconsistent data values if necessary.
2. MLP Regression Implementation from Scratch [20 marks]

In this part, you are required to implement MLP regression from scratch using numpy, pandas and experiment with various activation functions and optimiza- tion techniques, and evaluate the model’s performance.

1. Create a class where you can modify and access the learning rate, activa- tion function, optimisers, number of hidden layers and neurons.
1. Implement methods for forward propagation, backpropagation, and train- ing.
1. Implement the Sigmoid, Tanh, and ReLU activation functions and make them easily interchangeable within your MLP framework.
1. Implement methods for the Stochastic Gradient Descent (SGD), Batch Gradient Descent, and Mini-Batch Gradient Descent algorithms from scratch, ensuring that they can be employed within your MLP architecture.
3. Model Training & Hyperparameter Tuning using W&B [20 marks]
1. Log your scores - loss (Mean Squared Error) on the validation set using W&B.
1. Report metrics: MSE, RMSE, R-squared.
1. You have to report the scores(ordered) for all the combinations of :
- Activation functions : sigmoid, tanh and ReLU (implemented from scratch)
- Optimizers : SGD, batch gradient descent and mini-batch gradient descent (implemented from scratch).
4. Tune your model on various hyperparameters, such as learning rate, epochs, and hidden layer neurons.
- Report the parameters for the best model that you get (for the various values you trained the model on).
- Report the scores mentioned in 3.3.2 for all values of hyperparameters in a table.
4. Evaluating Model [5 marks]

1\. Test your model on the test set and report loss score (MSE, RMSE, R- squared).

4  CNN and AutoEncoders

[ Marks : 100, Estimated Time : 5-6 days ]

Welcome to the Hello World of image classification - a CNN trained on MNIST dataset. You can use Pytorch for this Task. You can load the MNIST dataset using PyTorch’s torchvision.

1. Data visualization and Preprocessing [10 marks]
1. Draw a graph that shows the distribution of the various labels across the entire dataset. You are allowed to use standard libraries like Matplotlib.
1. Visualize several samples (say 5) of images from each class.
1. Check for any class imbalance and report.
1. Partition the dataset into train, validation, and test sets.
1. Write a function to visualize the feature maps. Your code should be able to visualize feature maps of a trained model for any layer of the given image.
2. Model Building [20 marks]
1. Construct a CNN model for Image classification using pytorch.
1. Your network should include convolutional layers, pooling layers, dropout layers, and fully connected layers.
1. Construct and train a baseline CNN using the following architecture: 2 convolutional layers each with ReLU activation and subsequent max pool- ing, followed by a dropout and a fully-connected layer with softmax acti- vation, optimized using the Adam optimizer and trained with the cross- entropy loss function.
1. Display feature maps after applying convolution and pooling layers for any one class and provide a brief analysis.
1. Report the training and validation loss and accuracy at each epoch.
3. Hyperparameter Tuning and Evaluation [20 marks]
   1. Use W&B to facilitate hyperparameter tuning. Experiment with various architectures and hyperparameters: learning rate, batch size, kernel sizes (filter size), strides, number of epochs, and dropout rates.
   1. Compare the effect of using and not using dropout layers.
   1. Log training/validation loss and accuracy, confusion matrices, and class- specific metrics using W&B.
4. Model Evaluation and Analysis [10 marks]
1. Evaluate your best model on the test set and report accuracy, per-class accuracy, and classification report.
1. Provide a clear visualization of the model’s performance, e.g., confusion matrix.
3. Identify a few instances where the model makes incorrect predictions and analyze possible reasons behind these misclassifications.
5. Train on Noisy Dataset [10 marks]

In the subsequent parts, you have to work with a noisy mnist dataset. Download the mnist-with-awgn.mat from [here.](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/khushi_agarwal_students_iiit_ac_in/EnXYjh3ZU7xAtuz_NsvtB_MBhJHSL7EQzSJdeyPu8j8Khg?e=c09dfv ) You can load .mat file using scipy.io.

1. Train your best model from the previous parts of Task 4 on the noisy mnist dataset which contains noise in it (additive white gaussian noise, don’t worry it’s just a fancy name).
1. Report validation losses, validation scores, training losses, training scores.
1. Evaluate your model on test data and print the classification report.
6. AutoEncoders to Save the Day [30 marks]
1. Implement an Autoencoder class which will help you de-noise the noisy mnist dataset from Part 4.5.
1. Visualise the classes and feature space before and after de-noising.
1. Now using the de-noised dataset, train your best model from the previous parts.
1. Report validation losses, validation scores, training losses, training scores.
1. Evaluate your model on test data and print the classification report.
1. Analyse and compare the results/accuracy scores as obtained in Part 4.5 and 4.6.
5  Some Other Variants

[ Marks : 50, Estimated Time : 2-3 days ]

You can use PyTorch for this Task.

1. Multi-digit Recognition on Multi-MNIST Dataset [25 marks]

Download the DoubleMNIST dataset from [here. ](https://drive.google.com/file/d/1MqQCdLt9TVE3joAMw4FwJp_B8F-htrAo/view)The DoubleMNIST dataset contains images with two handwritten digits, and the task is to correctly identify and classify each digit within the image.

Build and train models that can simultaneously recognize and predict the two digits from a single image.This is basically another version of a multilabel classification.

- Display several images from the filtered dataset to familiarize yourself with the dual-digit nature of the images.
- From the Multi-MNIST dataset, filter out and exclude images where the same digit appears twice. Your model will be trained only on images that have distinct digits
- Ensure you split the datasets into training and validation sets to evaluate the model’s performance during hyperparameter tuning.
1. MLP on Multi-MNIST
   1. Implement and train an MLP model on the MultiMNIST dataset.
   1. Hyperparameter Tuning: Adjust the number of hidden layers and the number of neurons within each layer to optimize performance and find the best model.
   1. Report the accuracies on the train and validation set.
   1. Evaluate your trained model on test set and report the accuracy.
2. CNN on Multi-MNIST
1. Design and train a CNN model on the MultiMNIST dataset.
1. Hyperparameter Tuning: Experiment with different learning rates, kernel sizes, and dropout rates to determine the optimal configuration.
1. Report the accuracies on the train and validation set.
1. Evaluate your trained model on test set and report the accuracy.
3. Testing on Single digit MNIST (regular MNIST)

Evaluate your trained model on the regular MNIST dataset with single-digit images. See how the model, initially trained on images with two digits, performs on these single-digit images. Report the accuracies.

2. Permuted MNIST [15 marks]

The Permuted MNIST dataset is a variation of the original MNIST dataset where the pixels of each image are randomly permuted. This permutation makes the task significantly more challenging because it destroys the spatial structure of the images. In Permuted MNIST, the goal is still to recognize the digits but now without relying on the spatial relationships between pixels.

Ensure you split the datasets into training and validation sets to evaluate the model’s performance during hyperparameter tuning.

1. MLP on Permuted-MNIST
   1. Implement and train an MLP model on the Permuted-MNIST dataset.
   1. Hyperparameter Tuning: Adjust the number of hidden layers and the number of neurons within each layer to optimize performance and find the best model.
   1. Report the accuracies on the train and validation set.
   1. Evaluate your trained model on test set and report the accuracy.
2. CNN on Permuted-MNIST
1. Design and train a CNN model on the Permuted-MNIST dataset.
1. Hyperparameter Tuning: Experiment with different learning rates, kernel sizes, and dropout rates to determine the optimal configuration.
1. Report the accuracies on the train and validation set.
1. Evaluate your trained model on test set and report the accuracy.
3. Analysis [10 marks]
1. Contrast the performances of MLP vs. CNN for both datasets.
1. Discuss the observed differences and any challenges faced during training and evaluation.
1. Compare the potential for overfitting between a CNN and an MLP in the context of datasets in Task 5. Use training vs. validation loss/accuracy plots to support your observations.
6  Report
1. The submission should consist of two separate files one for Task 1,2 and 3, one for Task 4 and 5.
1. Ensure you submit a clear and organized report using W&B. The quality of your report is essential for this assignment and holds significant weightage. Proper documentation and presentation of results align with good ML practices, so take the time to structure your W&B report effectively. You can submit 5 different W&B reports for all the 5 tasks.
10
