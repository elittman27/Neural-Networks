# Neural-Networks
Derives and implements neural networks with arbitrarily deep layers, convolutional layers, and pooling layers. Enables forward and backward propagation to make predictions and compute the gradient of the network respectively. Uses Cross Entropy Loss for the cost function and exposes different activation functions including SoftMax, ReLU, and Sigmoid.

To run all the tests, run:
```sh
python -m unittest -v
```

The most relevant code samples exist in the following files:
- Convolutions and Layers: neural_networks/layers.py
- Overall network: neural_networks/models.py
- Cost functions: neural_networks/losses.py
- Activation functions: neural_networks/activations.py
