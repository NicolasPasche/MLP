import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_random_state

class MLP(object):
    def func_id(self, x):
        return x
    
    def func_sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def __init__(self, 
                 n_input_neurons=2,
                 n_hidden_neurons=2,
                 n_output_neurons=1,
                 weights=None, 
                 eta=0.01, n_iterations=10, random_state=2, 
                 *args, **kwargs):
        """
        Description of the matrix structure:
        
        1. Input Layer (self.inputLayer):
           - Column 0: Bias Neuron (always 1.0)
           - Column 1: Placeholder for input values (X)
           - Column 2: Input values (X)
        
        2. Hidden Layer (self.hiddenLayer):
           - Column 0: Net input (net_j), calculated as W_ij . x
           - Column 1: Neuron activation (a_j), calculated through sigmoid of the net input
           - Column 2: Neuron output (o_j), same as activation in this implementation
           - Column 3: Derivative of sigmoid (der_j), calculated as o_j * (1 - o_j)
           - Column 4: Error term (delta_j), based on derivative and error backpropagation
        
        3. Output Layer (self.outputLayer):
           - Column 0: Net input (net_k), calculated as W_jk . h
           - Column 1: Neuron activation (a_k), calculated through sigmoid of the net input
           - Column 2: Neuron output (o_k), same as activation in this implementation
           - Column 3: Derivative of sigmoid (der_k), calculated as o_k * (1 - o_k)
           - Column 4: Error term (delta_k), based on derivative and error backpropagation
        """
        self.n_input_neurons = n_input_neurons
        self.n_hidden_neurons = n_hidden_neurons
        self.n_output_neurons = n_output_neurons
        self.weights = weights
        W_IH = []
        W_HO = []
        # Learning rate
        self.eta = eta
        # Iterations
        self.n_iterations = n_iterations
        # Random Number Generator
        self.random_state = random_state
        # Creation of RNG
        self.random_state_ = check_random_state(self.random_state)
        # Error at fit
        self.errors = []        
        self.network = []
        self.inputLayer = np.zeros((self.n_input_neurons + 1, 5))
        self.inputLayer[0] = 1.0 
        self.network.append(self.inputLayer)
        if weights:
            W_IH = self.weights[0]    
        else:
            W_IH = 2 * self.random_state_.random_sample(\
            (self.n_hidden_neurons + 1, self.n_input_neurons + 1)) - 1
        self.network.append(W_IH)
        # Hidden Layer + Bias Neuron: 
        # Columns = net_i, a_i, o_i, d_i, delta_i
        self.hiddenLayer = np.zeros((self.n_hidden_neurons + 1, 5))
        self.hiddenLayer[0] = 1.0 
        self.network.append(self.hiddenLayer)
        if weights:
            W_HO = self.weights[1]
        else:
            W_HO = 2 * self.random_state_.random_sample(\
            (self.n_output_neurons + 1, self.n_hidden_neurons + 1)) - 1
        self.network.append(W_HO)
        # Output Layer + Bias Neuron: 
        # Columns = net_i, a_i, o_i, d_i, delta_i
        self.outputLayer = np.zeros((self.n_output_neurons + 1, 5)) 
        self.outputLayer[0] = 0.0 
        self.network.append(self.outputLayer)

    def print(self):
        print("************************************************")
        print('Multi-Layer-Perceptron - Network Architecture')
        print("************************************************")
        np.set_printoptions(formatter={'float': lambda x: "{0:7.3f}".format(x)})
        for idx, nn_part in enumerate(self.network):
            print(nn_part)
            print('----------v----------')  
            
    def predict(self, x):
        ###############
        # Input Layer
        # Set the inputs: All lines, Row 2
        self.network[0][:, 2] = x
        ###############
        # Hidden Layer
        # Start from line 1, because of the Bias Neuron on index 0
        # net_j = W_ij . x
        self.network[2][1:, 0] = np.dot(self.network[1][1:, :],\
                                       self.network[0][:, 2])
        # a_j
        self.network[2][1:, 1] = self.func_sigmoid(\
                                       self.network[2][1:, 0]) 
        # o_j
        self.network[2][1:, 2] = self.func_id(self.network[2][1:, 1]) 
        # der_j = o_j*(1 - o_j) Derivative of sigmoid
        self.network[2][1:, 3] = self.network[2][1:, 2] \
                                * (1.0 - self.network[2][1:, 2])
        ###############
        # Output Layer
        # Start from line 1 because of the Bias Neuron on index 0
        # net_k = W_jk . h
        self.network[4][1:, 0] = np.dot(self.network[3][1:, :],\
                                       self.network[2][:, 2])
        # a_k
        self.network[4][1:, 1] = self.func_sigmoid(\
                                       self.network[4][1:, 0]) 
        # o_k
        self.network[4][1:, 2] = self.func_id(self.network[4][1:, 1])
        # der_k = o_k*(1 - o_k) Derivative of sigmoid
        self.network[4][1:, 3] = self.network[4][1:, 2] \
                                * (1.0 - self.network[4][1:, 2])
        
        return self.network[4][:, 2]   
    
    def fit(self, X, Y):       
        delta_w_jk = []
        delta_w_ij = []
        self.errors = []
        for iteration in range(self.n_iterations):
            error = 0.0
            for x, y in zip(X, Y):    
                y_hat = self.predict(x)
                diff = y - y_hat
                error += 0.5 * np.sum(diff * diff)
                
                #####################
                # Output Layer
                # delta_k in the Output Layer = der_k * diff
                self.network[4][:, 4] = self.network[4][:, 3] * diff
                
                #####################
                # Hidden Layer
                # delta_j in the Hidden Layer = 
                #   der_j * dot(W_kj^T, delta_k)
                self.network[2][:, 4] = \
                             self.network[2][:, 3] * \
                             np.dot(self.network[3][:].T,\
                                    self.network[4][:, 4])                 
                
                #####################
                # Weight deltas of W_kj
                # delta_w = eta * delta_k . o_j^T
                delta_w_jk = self.eta * \
                             np.outer(self.network[4][:, 4],\
                                      self.network[2][:, 2].T)
                # Weight deltas of W_ji
                # delta_w = eta * delta_j . o_i^T
                delta_w_ij = self.eta * \
                             np.outer(self.network[2][:, 4],\
                                      self.network[0][:, 2].T)
                
                #####################
                # Adjust the Weights
                self.network[1][:, :] += delta_w_ij               
                self.network[3][:, :] += delta_w_jk
                
            # Collect the error for every example 
            self.errors.append(error)

    def plot(self):       
        fignr = 1
        plt.figure(fignr, figsize=(5, 5))
        plt.plot(self.errors) 
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.title('Training Error Over Iterations')  # Added title for clarity
        plt.grid(True)  # Added grid for better readability
        plt.legend(['Error'], loc='upper right')  # Added legend
        plt.show()  # Added to ensure the plot is displayed


def main():
    X = np.array([[1.0, 1.0, 1.0], [1.0, 0, 1.0], [1.0, 1.0, 0], [1.0, 0, 0]])
    Y = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    nn = MLP(eta=0.03, n_iterations=40000, random_state=42)

    nn.fit(X, Y)                        
    nn.print()

    nn.plot()

    print('Predict:')
    for x, y in zip(X, Y):
        print('{} {} -> {}'.format(x, y[1], nn.predict(x)[1:2]))


main()
