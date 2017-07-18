import mnist_loader
import network2
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()
net = network2.Network([784,60,20,10])
net.SGD(training_data, 30, 10, 1.25,lmbda=100,\
		evaluation_data = validation_data,
		monitor_evaluation_accuracy=True,
		monitor_training_accuracy=True)
