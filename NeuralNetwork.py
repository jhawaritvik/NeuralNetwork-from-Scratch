import numpy as np

class NeuralNetwork:
    class Linear_layer:
        def __init__(self,features,neurons):
            #neurons  --> No of neurons in this layer
            #features --> No of neurons/features in the previous layer
            k=np.sqrt(2.0/(features))
            self.weights=np.random.uniform(-k,k,size=(neurons,features))
            self.bias=np.zeros((neurons,1))
        
        def forward(self,x):
            self.input=x #x has shape (samples,features)
            self.output=np.dot(self.weights,self.input.T)+self.bias #will give output of shape (neurons x samples)
            return self.output

        def backward(self,dL_dZ):
            samples=self.input.shape[0]
            dL_dW=np.dot(dL_dZ,self.input)/samples #shape is (neurons,samples)*(samples,features)=(neurons,features) 
            dL_dB=np.sum(dL_dZ,axis=1).reshape(-1,1) /samples #shape before reshape will be (1,classes)  sum of column1,column2,...

            return dL_dW,dL_dB

    class activation:
        def ReLU(layer_output):
            return np.maximum(0,layer_output)

        def ReLU_derivative(layer_output):
            return (layer_output>0).astype(float) #1 if layer_output>0 else 0
            
        def softmax(layer_output):
            numerator=np.exp(layer_output-np.max(layer_output,axis=0,keepdims=True))
            return np.divide(numerator,np.sum(numerator,axis=0))
    
    class loss_functions:
        def CrossEntropyLoss(self,y_predicted,y):
            self.y_predicted=y_predicted
            self.y=y
            epsilon=1e-9 #added to the probailities to avoid 0 in log
            self.y_predicted=NeuralNetwork.activation.softmax(self.y_predicted)#softmax application on output logits
            self.loss=-np.mean(np.sum((y * np.log(self.y_predicted.T+epsilon)),axis=1)) #negative log likelihood of the true class probabilities
            return self.loss
        
        def backward(self,model): #Only for 2 layers
            dL_dZ2=self.y_predicted - self.y.T #gradient of loss wrt softmax output

            W2=model.l2.weights

            dL_dW2,dL_dB2=model.l2.backward(dL_dZ2) #dLoss/W2 and dLoss/B2, gradient wrt layer2 weights and bias

            dL_dA1=np.dot(W2.T,dL_dZ2) #dZ2/dA1*dLoss/dZ2 , gradient wrt layer1 output with ReLU. Dot of

            dL_dZ1=dL_dA1*NeuralNetwork.activation.ReLU_derivative(model.l1.output) #Gradient wrt layer1 output

            dL_dW1,dL_dB1=model.l1.backward(dL_dZ1) #gradient wrt to layer 1 weights and bias

            return dL_dW2, dL_dB2, dL_dW1, dL_dB1

    class gradient_descent:
        def __init__(self,learning_rate):
            self.learning_rate=learning_rate
        
        def step(self,model,loss):
            dL_dW2, dL_dB2, dL_dW1, dL_dB1 =loss.backward(model)
            model.l2.weights-=self.learning_rate*dL_dW2
            model.l2.bias-=self.learning_rate*dL_dB2
            model.l1.weights-=self.learning_rate*dL_dW1
            model.l1.bias-=self.learning_rate*dL_dB1


class model:
    def __init__(self,layer1neurons,input_size):
        self.l1=NeuralNetwork.Linear_layer(input_size,layer1neurons) #1st neural layer with inputsize i.e features of x and neuron count as desired
        self.l2=NeuralNetwork.Linear_layer(layer1neurons,3) #2nd neural layer with input 

    def forward(self,x):
        out=self.l1.forward(x)
        out=NeuralNetwork.activation.ReLU(out) #out has shape (neurons x samples)
        out=self.l2.forward(out.T) #transposing out because input should have shape (samples,features or neurons)

        return out #has shape (neurons,samples)
    