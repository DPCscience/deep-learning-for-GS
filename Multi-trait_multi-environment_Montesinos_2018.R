############################################################################
##Multi-trait, multi-environment deep learning modeling for genomic-enabled prediction of plant traits
##Montesinos-López O.A., Montesinos-López A., Crossa J., Gianola D., Hernández-Suárez C.M., et al. 2018
############################################################################

#R code for implementing MTDL models
setwd(“C:/TELEMATICA 2017/Deep Learning Multi-trait”)

rm(list = ls())

######Libraries required#################################
library(tensorflow)
library(keras)

#############Loading data################################
load(“Data_Maize_set_1.RData”)
ls()

####Genomic relationship matrix (GRM)) and phenotipic data######
G=G_maize_1to3
Pheno=Pheno_maize_1to3
head(Pheno)

###########Cholesky decomposition of the GRM######
LG=t(chol(G))

#################
########Creating the design matrices 
#################

Z1=model.matrix(~0+as.factor(Pheno$Line))
ZE=model.matrix(~0+as.factor(Pheno$Env))
Z1G=Z1%%LG
Z2GE=model.matrix(~0+as.factor(Pheno$Line):as.factor(Pheno$Env))
G=data.matrix(G)
G2=kronecker(diag(3),G)
LG2=t(chol(G2))
Z2GE=Z2GE%%LG2

###Defining the number of epoch and units#####################
units_M=50
epochs_M=50

##########Data for trait GY#################################
y = Pheno[,3:5]
X = cbind(ZE,Z1G,Z2GE)
head(y)

#############Training and testing sets###################
n=dim(X)[1]
nt=dim(y)[2]
Post_trn=sample(1:n,round(n0.8))
X_tr = X[Post_trn,]
X_ts = X[-Post_trn,]
y_tr = scale(y[Post_trn,])
Mean_trn=apply(y[Post_trn,],2,mean)
SD_trn=apply(y[Post_trn,],2,sd)
y_ts=matrix(NA,ncol=nt,nrow=dim(X_ts)[1])
for (t in 1:nt){
y_ts[,t] =(y[-Post_trn,t]- Mean_trn[t])/SD_trn[t] }


# add covariates (independent variables)
input ,- layer_input(shape=dim(X_tr)[2],name=“covars”)


# add hidden layers
base_model <- input %.%
layer_dense(units =units_M, activation=’relu’) %>%
layer_dropout(rate = 0.3) %>%
layer_dense(units = units_M, activation = “relu”) %>%
layer_dropout(rate = 0.3) %>%
layer_dense(units = units_M, activation = “relu”) %>%
layer_dropout(rate = 0.3)


# add output 1
yhat1 <- base_model %>%
layer_dense(units = 1, name=“yhat1”)

# add output 2
yhat2 <- base_model %>%
layer_dense(units = 1, name=“yhat2”)

# add output 3
yhat3 <- base_model %>%
layer_dense(units = 1, name=“yhat3”)

# build multi-output model
model <- keras_model(input,list(yhat1,yhat2,yhat3)) %>%
compile(optimizer = “rmsprop”,
loss=”mse”,
metrics=”mae”,
loss_weights=c(0.3333,0.3333,0.3333))


# fit model
model_fit <- model %>%
fit(x=X_tr,
y=list(y_tr[,1],y_tr[,2],y_tr[,3]),
epochs=epochs_M,
batch_size = 50,
verbose=0)

# predict values for test set
Yhat <- predict(model, X_ts) %>%
data.frame() %.%
setNames(colnames(y_tr))
predB=Yhat
y_p=predB
for (s in 1:nt){
y_p[,s]=y_p[,s]SD_trn[s]+ Mean_trn[s]
y_ts[,s]=y_ts[,s]SD_trn[s]+ Mean_trn[s]
}


#################Observed and predicted values############
Y_all_tst = data.frame(cbind(y_ts, y_p))
Y_all_tst

########Prediction accuracy with Pearson Correlation######
Cor_Mat=cor(Y_all_tst)
Cor_Traits=diag(Cor_Mat[(nt+1):(2nt),1:nt])
Cor_Traits

########Plots of observed and predicted values############
plot(Y_all_tst[,1],Y_all_tst[,4])
plot(Y_all_tst[,2],Y_all_tst[,5])
plot(Y_all_tst[,3],Y_all_tst[,6])

