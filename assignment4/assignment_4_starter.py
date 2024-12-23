import plotly
import math
import csv
import sys

class LogisticReg:
    def __init__(self, W, learning_rate):
        self.weights = W
        self.learning_rate = learning_rate

    #Calculates the dot product of two vectors X, Y.
    def dot_product(self, X, Y):
        return sum(x *y for x,y in zip(X, Y))

    #Calculates the sigmoid of the input Z
    def sigmoid(self, Z):
        return 1/(1 +math.exp(-Z))

    #Calculates the gradient when the model is given the input vector X and true label Y
    def gradient(self, X, Y):
        z =self.dot_product(self.weights, X)
        y_pred = self.sigmoid(z)
        error = y_pred- Y
        return [error*x for x in X]

    #Updates the weights of the model given the calculated gradient
    #self.weights are updated and not returned
    def update_weights(self, gradient):
        self.weights = [
            w - self.learning_rate*g for w,g in zip(self.weights,gradient)
        ]

    #Calculate the cross_entropy_loss when the model is given the input vector X and true label Y
    def cross_entropy_loss(self, X, Y):
        z = self.dot_product(self.weights,X)
        y_pred = self.sigmoid(z)
        if Y==1:
            return -math.log(y_pred)
        else:
            return -math.log(1 -y_pred)

    #Create a plot that shows the loss function after each instance
    #Train over dataset for the given nummber of epochs.
    #Show the loss on the instance after you update the weights
    #plot the cross_entropy_loss for each isntance while training so 8*num_epochs points
    #Also, plot the mean of the cross entropy loss across the entire dataset after each epoch
    #To calculate the mean of the cross entropy loss, you will sum the cross entropy loss for each instance as you train then divide by the number of instances.
    #Set the upper limit to 0.1 for the y-axis so that you can see the differences among the points
    #You will have some values that go off the graph, which is okay.
    def train_and_plot_loss(self, dataset, num_epochs):
        instance_losses = []
        epoch_mean_losses = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            for instance in dataset:
                X,Y=instance
                loss =self.cross_entropy_loss(X, Y)
                instance_losses.append(loss)
                epoch_loss+=loss
                gradient=self.gradient(X, Y)
                self.update_weights(gradient)
            mean_loss = epoch_loss / len(dataset)
            epoch_mean_losses.append(mean_loss)
        fig1 = plotly.graph_objects.Figure()
        fig1.add_trace(plotly.graph_objects.Scatter(y=instance_losses, mode="lines", name="Instance Loss"))
        fig1.update_layout(
            title="Cross-Entropy Loss After Each Weight Update",
            xaxis_title="Weight Updates",
            yaxis_title="Loss",
            yaxis=dict(range=[0, 0.1]),
        )
        fig1.show()
        fig2 = plotly.graph_objects.Figure()
        fig2.add_trace(plotly.graph_objects.Scatter(y=epoch_mean_losses, mode="lines", name="Mean Loss"))
        fig2.update_layout(
            title="Mean Cross-Entropy Loss After Each Epoch",
            xaxis_title="Epochs",
            yaxis_title="Loss",
        )
        fig2.show()


def read_in_data(dataset_path):
    my_dataset = []
    with open(dataset_path, "r", newline="", errors="ignore") as csvfile:
        my_reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for row in my_reader:
            my_dataset.append(row)
    return my_dataset


#Given a dataset, create three visual represenations of some data from the dataset using three different visualization styles
#The data that you choose to visualize from the given dataset is up to you.
#The grpahs should be created and displayed when the program is called
def visualize_data(dataset):
    ratings = []
    release_years = []
    genres = {}
    for row in dataset[1:]:
        try:
            if row[5]:
                ratings.append(float(row[5]))
            if row[3]:
                release_years.append(int(float(row[3])))
            if row[2]:
                for genre in row[2].split(","):
                    genres[genre] = genres.get(genre, 0) + 1
        except ValueError:
            continue
    fig1 = plotly.graph_objects.Figure()
    fig1.add_trace(plotly.graph_objects.Histogram(x=ratings))
    fig1.update_layout(
        title="Distribution of imdb ratings",
        xaxis_title="Imdb ratings",
        yaxis_title="Frequency",
    )
    fig1.show()
    fig2 = plotly.graph_objects.Figure()
    fig2.add_trace(plotly.graph_objects.Histogram(x=release_years))
    fig2.update_layout(
        title="Distribution of release years",
        xaxis_title="Release year",
        yaxis_title="Frequency",
    )
    fig2.show()
    fig3 = plotly.graph_objects.Figure()
    fig3.add_trace(plotly.graph_objects.Bar(x=list(genres.keys()), y=list(genres.values())))
    fig3.update_layout(
        title="Distribution of genres",
        xaxis_title="Genres",
        yaxis_title="Count",
    )
    fig3.show()  

###TESTS
###YOU MAY NOT REMOVE OR EDIT THE CODE BELOW THIS POINT!
num_incorrect = 0
num_tests = 6
weights = [0.1, 2.5, -5.0, -1.2, 0.5, 2.0, 0.7]
X = [1,3,2,1,3,0,4.19]
logReg = LogisticReg(weights,0.5 )
dot_prod = logReg.dot_product(X,weights)
if(round(dot_prod, 2) != 0.83):
    num_incorrect += 1

sig = logReg.sigmoid(dot_prod)
if(round(sig,3) != 0.697):
    num_incorrect += 1

loss1 = logReg.cross_entropy_loss(X,1)
if(round(loss1,3) != 0.361):
    num_incorrect += 1

loss0 = logReg.cross_entropy_loss(X,0)
if(round(loss0, 3) != 1.194):
    num_incorrect += 1

weights_2 = [0,0,0]
X_2 = [1,3,2]
logReg_2 = LogisticReg(weights_2, 0.1)
grad = logReg_2.gradient(X_2, 1)
if(grad != [-0.5, -1.5, -1.0]):
    num_incorrect += 1

logReg_2.update_weights(grad)
correct_weights = [0.05, 0.15, 0.10]
found_error = False
for i in range(len(logReg_2.weights)):
    if(round(logReg_2.weights[i],2) != correct_weights[i]):
        found_error = True
if(found_error):
    num_incorrect += 1

print(num_tests-num_incorrect, "/",num_tests, "test passed")

#1 is alsways the left most input which corresponds to the bias
simple_dataset = [[[1,6,5,1,1],1],
                  [[1,7,7,2,3],1],
                  [[1,0,2,4,6],0],
                  [[1,1,3,5,8],0],
                  [[1,6,7,2,3],1],
                  [[1,1,2,8,9],0],
                  [[1,8,9,2,0],1]]
start_weights = [-1, -0.8, -2, 1, 2] #The left most weight is the bias
my_log_reg = LogisticReg(start_weights,0.2 )
my_log_reg.train_and_plot_loss(simple_dataset, 20)


#Visualize dataset
my_dataset = read_in_data(sys.argv[1])
visualize_data(my_dataset)

