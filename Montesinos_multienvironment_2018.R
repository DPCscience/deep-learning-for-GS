#Paper Multi-environment Genomic Prediction of Plant Traits Using Deep Learners With Dense Architecture
#Abelardo Montesinos-López, Osval A. Montesinos-López, Daniel Gianola, José Crossa and Carlos M. Hernández-Suárez

DEEP LEARNING R CODES FOR A DENSELY CONNECTED NETWORK
setwd(“C:\\TELEMATICA 2017\\Deep Learning CONTINUOUS”)

rm(list = ls())

######Libraries required##################################

library(tensorflow)

library(keras)

#############Loading data###############################

load(“Data_Maize_1to3.RData”)

####Genomic relationship matrix (GRM)) and phenotipic data#####

G=G_maize_1to3

Pheno=Pheno_maize_1to3

head(Pheno)

###########Cholesky decomposition of the GRM##############

LG=t(chol(G))

########Creating the desing matrices ########################

Z1G=model.matrix(∼0+as.factor(Pheno$Line))

ZE=model.matrix(∼0+as.factor(Pheno$Env))

Z1G=Z1G%*%LG ####Incorporating marker information to lines

Z2GE=model.matrix(∼0+as.factor(Pheno$Line):as.factor(Pheno$Env))

G2=kronecker(diag(3),data.matrix(G))

LG2=t(chol(G2))

Z2GE=Z2GE%*%LG2

###Defining the number of epoch and units#####################

units_M=50

epochs_M=20

##########Data for trait GY#################################

y =Pheno$Yield

X = cbind(ZE, Z1G, Z2GE)

#############Training and testing sets########################

n=dim(X)[1]

Post_trn=sample(1:n,round(n*0.65))

X_tr = X[Post_trn,]

X_ts = X[-Post_trn,]

y_tr = scale(y[Post_trn])

Mean_trn=mean(y[Post_trn])

SD_trn=sd(y[Post_trn])

y_ts = (y[-Post_trn]- Mean_trn)/SD_trn

#########Model fitting in Keras################################

model <- keras_model_sequential()

#########Layers specification ################################

model %>%

layer_dense(

units =units_M,

activation = “relu”,

input_shape = c(dim(X_tr)[2])) %>%

layer_dropout(rate = 0.3) %>% ###Input Layer

layer_dense(units = units_M, activation = “relu”) %>%

layer_dropout(rate = 0.3) %>% ###Hidden layer 1

layer_dense(units = units_M, activation = “relu”) %>%

layer_dropout(rate = 0.3) %>% ####Hidden layer 2

layer_dense(units = 1) ####Output layer

model %>% compile(

loss = “mean_squared_error”,

optimizer = optimizer_adam(),

metrics = c(“mean_squared_error”))

history <- model %>% fit(

X_tr, y_tr, epochs = epochs_M, batch_size = 30,

verbose = FALSE)

#######Evaluating the performance of the model###################

pf = model %>% evaluate(x = X_ts, y = y_ts, verbose = 0)

y_p = model %>% predict(X_ts)

y_p=y_p*SD_trn+ Mean_trn

y_ts=y_ts

y_ts=y_ts*SD_trn+ Mean_trn

###############Observed and predicted values of the testing set#

Y_all_tst = data.frame(cbind(y_ts, y_p))

cor(Y_all_tst[,1],Y_all_tst[,2])

plot(Y_all_tst)
