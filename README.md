# Keras_cifar10

### Why cifar10?
Cifar 10 is a dataset that many researchers use to benchmark their deep learning models. Papers which use the cifar10 dataset are easy to find (https://benchmarks.ai/cifar-10) which allows me to learn about some of the most recent discoveries in object classification through hands-on practice. 

### Why Keras?
My aim in working on this project is to introduce myself to deep learning and image processing. Keras allows me to experiment with techniques discussed in peer-reviewed research without worrying about what goes on at a lower level of abstraction. Unfortunately, the drawback to using Keras is that many ideas presented in the more recent research papers are not supported in Keras.

### Intuition I've built from this project
1. Use a low dropout rate for convolutional layers (p= ~0.1) and a high rate for fully connected layers (p= ~0.5)
2. ELu and SELu activations are better than ReLu activations + batch norm since SELu's and ELu's can have negative values and prevent bias shift. Batch norm is costly so using ELu's can speed up the network. (However I did have an error when I tried to remove batchnorm from a fully connected layer where the network did not learn. I still have to find out why).
3. Data augmentation works REALLY well to prevent overfitting.
