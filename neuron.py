import numpy as np

class Neuron():
    class Model():
        def __init__(self, input, output):
            self.input = input
            self.output = output

    def __init__(self, model):
        self.model = model
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 3)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_diff(self, x):
        return x * (1 - x)

    def train(self, training_iterations):
        for iteration in range(training_iterations):
            output = self.output(self.model.input)
            error = self.model.output - output
            adjustment_factor = error * self.sigmoid_diff(output)
            adjustment_diff = np.dot(self.model.input.T, adjustment_factor)
            self.synaptic_weights += adjustment_diff

    def output(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

def test():
    # Guess the bitwise operator << 1
    model = Neuron.Model(
        input=np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0]]), 
        output=np.array([[0,0,0], [0,1,0], [1,0,0], [1,1,0], [0,0,0], [0,1,0], [1,0,0]])
    )
    neuron = Neuron(model=model)
    neuron.train(100)
    
    new_situation = [1, 1, 1]
    output = neuron.output(np.array(new_situation))
    sum = 0
    for value in output:
        sum += value
    error_percent = sum / len(output)
    output = [round(x) for x in output] 
    print(f'Bitwise left (<< 1) guess for: {new_situation} = {output} | error: {error_percent}%')

    # Guess the bitwise operator >> 1 
    model = Neuron.Model(
        input=np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0]]), 
        output=np.array([[0,0,0], [0,0,0], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,1]])
    )
    neuron = Neuron(model=model)
    neuron.train(100)
    
    new_situation = [1, 1, 1]
    output = neuron.output(np.array(new_situation))
    sum = 0
    for value in output:
        sum += value
    error_percent = sum / len(output)
    output = [round(x) for x in output]
    print(f'Bitwise right (>> 1) guess for: {new_situation} = {output} | error: {error_percent}%')

    # Guess the opposite (!) operator
    model = Neuron.Model(
        input=np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0]]), 
        output=np.array([[1,1,1], [1,1,0], [1,0,1], [1,0,0], [0,1,1], [0,1,0], [0,0,1]])
    )
    neuron = Neuron(model=model)
    neuron.train(100)
    
    new_situation = [1, 1, 1]
    output = neuron.output(np.array(new_situation))
    sum = 0
    for value in output:
        sum += value
    error_percent = sum / len(output)
    output = [round(x) for x in output] 
    print(f'Opposite (!) guess for: {new_situation} = {output} | error: {error_percent}%')

    new_situation = [1, 1, 0]
    output = neuron.output(np.array(new_situation))
    sum = 0
    for value in output:
        sum += value
    error_percent = sum / len(output)
    output = [round(x) for x in output]
    print(f'Opposite (!) guess for: {new_situation} = {output} | error: {error_percent}%')


if __name__ == "__main__":
    test()
    