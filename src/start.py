import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 20
net = Network([
	ConvPoolLayer(filter_shape=(4,1,5,5), image_shape=(mini_batch_size,1,28,28)),
	FullyConnectedLayer(n_in=576, n_out=100),
	SoftmaxLayer(n_in=100, n_out=10)], 
	mini_batch_size)
net.SGD(training_data, 50, mini_batch_size, 0.1,
	validation_data, test_data)
