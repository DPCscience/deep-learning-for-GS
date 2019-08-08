#http://gradientdescending.com/deep-neural-network-from-scratch-in-r/

# libraries
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(caret))

# set up data
id <- sample(rep(1:4, 2), 8)
X <- matrix(c(0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1), nrow = 4, byrow = FALSE)
X <- X[id,]
y <- matrix(c(0, 1, 1, 0, 1, 0, 0, 1), nrow = 4)
y <- y[id,]

# activation function
# sigmoid
sigmoid <- function(x) return(1/(1+exp(-x)))
d.sigmoid <- function(x) return(x*(1-x))

# neural net function with 1 hidden layer - user specifies number of nodes
myNeuralNet <- function(X, y, hl, niters, learning.rate){
  
  # add in intercept
  X <- cbind(rep(1, nrow(X)), X)
  
  # set error array
  error <- rep(0, niters)
  
  # set up weights
  # the +1 is to add in the intercept/bias parameter
  W1 <- matrix(runif(ncol(X)*hl[1], -1, 1), nrow = ncol(X))
  W2 <- matrix(runif((hl[1]+1)*hl[2], -1, 1), nrow = hl[1]+1)
  W3 <- matrix(runif((hl[2]+1)*ncol(y), -1, 1), nrow = hl[2]+1)
  
  for(k in 1:niters){
    
    # calculate the hidden and output layers using X and hidden layer as inputs
    # hidden layer 1 and 2 have a column of ones appended for the bias term
    hidden1 <- cbind(matrix(1, nrow = nrow(X)), sigmoid(X %*% W1))
    hidden2 <- cbind(matrix(1, nrow = nrow(X)), sigmoid(hidden1 %*% W2))
    y_hat <- sigmoid(hidden2 %*% W3)
    
    # calculate the gradient and back prop the errors
    # see theory above
    y_hat_del <- (y-y_hat)*(d.sigmoid(y_hat))
    hidden2_del <- y_hat_del %*% t(W3)*d.sigmoid(hidden2)
    hidden1_del <- hidden2_del[,-1] %*% t(W2)*d.sigmoid(hidden1)
    
    # update the weights
    W3 <- W3 + learning.rate*t(hidden2) %*% y_hat_del
    W2 <- W2 + learning.rate*t(hidden1) %*% hidden2_del[,-1]
    W1 <- W1 + learning.rate*t(X) %*% hidden1_del[,-1]
    
    # storing error (MSE)
    error[k] <- 1/nrow(y)*sum((y-y_hat)^2)
    if((k %% (10^4+1)) == 0) cat("mse:", error[k], "\n")
  }
  
  # plot loss
  xvals <- seq(1, niters, length = 1000)
  print(qplot(xvals, error[xvals], geom = "line", main = "MSE", xlab = "Iteration"))
  
  return(y_hat)
}

# set parameters
hidden.layers <- c(6, 6)
iter <- 50000
lr <- 0.02

# run neural net
out <- myNeuralNet(X, y, hl = hidden.layers, niters= iter, learning.rate = lr)

pred <- apply(out, 1, which.max)
true <- apply(y, 1, which.max)
cbind(true, pred)

# try on iris data set
data(iris)
Xiris <- as.matrix(iris[, -5])
yiris <- model.matrix(~ Species - 1, data = iris)
out.iris <- myNeuralNet(Xiris, yiris, hl = hidden.layers, niters = iter, learning.rate = lr)
labels <- c("setosa", "versicolor", "virginica")
pred.iris <- as.factor(labels[apply(out.iris, 1, which.max)])
confusionMatrix(table(iris$Species, pred.iris))


# comparing with the neuralnet package
suppressPackageStartupMessages(library(neuralnet))

df <- data.frame(X1 = X[,1], X2 = X[,2], X3 = X[,3], y1 = y[,1], y2 = y[,2])
nn.mod <- neuralnet(y1 + y2 ~ X1 + X2 + X3, data = df, hidden = hidden.layers, 
                    algorithm = "backprop", learningrate = lr, act.fct = "logistic")
nn.pred <- apply(nn.mod$net.result[[1]], 1, which.max)
cbind(true, nn.pred)
plot(nn.mod)
# and on the iris package
iris.df <- cbind(iris[,-5], setosa = yiris[,1], versicolor = yiris[,2], virginica = yiris[,3])
nn.iris <- neuralnet(setosa + versicolor + virginica ~ Petal.Length + Petal.Width + Sepal.Length + Sepal.Width, 
                     data = iris.df, hidden = c(6, 6), algorithm = "backprop", learningrate = lr, act.fct = "logistic", 
                     linear.output = FALSE)
pred.iris <- labels[apply(nn.iris$net.result[[1]], 1, which.max)]
confusionMatrix(table(iris$Species, pred.iris))
