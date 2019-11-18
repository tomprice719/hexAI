from keras.layers import Layer
from keras.initializers import Constant
from keras.activations import softplus

class Gate(Layer):

    def __init__(self, var_init, **kwargs):
        self.var_init = var_init
        super(Gate, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.var = self.add_weight(initializer=Constant(self.var_init), trainable=True)
        super(Gate, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        return x * softplus(self.var)

    def compute_output_shape(self, input_shape):
        return input_shape