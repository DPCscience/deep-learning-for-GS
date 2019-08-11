#' @title Build a genomic selection prediction model using the deep learning technique
#' @description The function applies the deep convolutional neural network to build a prediction model for genomic selection.
#' @param trainMat  A genotype matrix (N x M; N individuals, M markers) for training model.
#' @param trainPheno  Vector (N * 1) of phenotype for training model.
#' @param validMat A genotype matrix for validing trained model.
#' @param validPheno Vector (N * 1) of phenotype for validing trained model.
#' @param markerImage  (String) This gives a "i * j" image format that the (M x1) markers informations of each individual will be encoded.
#'if the image size exceeds the original snp number, 0 will be polished the lack part,
#' if the image size is less than the original snp number, the last snp(s) will be descaled.
#' @param cnnFrame  A list containing the following element for convolutional neural network (CNN) framework:
#' \itemize{
#'     \item{conv_kernel:} {A vector (K * 1) gives convolutional kernel sizes (width x height) to filter image matrix for K convolutional layers, respectively. }
#'     \item{conv_num_filter:} { A vector (K * 1) gives number of convolutional kernels for K convolutional layers, respectively.}
#'     \item{pool_act_type:} {A vector (K * 1) gives types of active function will define outputs of K convolutional layers which will be an input of corresponding pool layer,
#'     respectively. It include "relu", "sigmoid", "softrelu" and "tanh". }
#'     \item{conv_stride:} {A character (K * 1) strides for K convolutional kernel.}
#'     \item{pool_type:} {A character (K * 1) types of K pooling layers select from "avg", "max", "sum", respectively.}
#'     \item{pool_kernel:} {A character (K * 1) K pooling kernel sizes (width * height) for K pooling layers. }
#'     \item{pool_stride:} {A Character (K * 1) strides for K pooling kernels.}
#'     \item{fullayer_num_hidden:} {A numeric (H * 1) number of hidden neurons for H full connected layers, respectively.
#'     The last full connected layer's number of hidden nerurons must is one.  }
#'     \item{fullayer_act_type:} {A numeric ((H-1) * 1) selecting types of active function from "relu", "sigmoid", "softrelu" and "tanh" for full connected layers.}
#'     \item{drop_float:} {Numeric.}
#' }
#' @param device_type  Selecting "cpu" or "gpu" device to  construct predict model.
#' @param gpuNum  (Integer) Number of GPU devices, if using multiple GPU (gpuNum > 1), the parameter momentum must greater than 0.
#' @param eval_metric (String) A approach for evaluating the performance of training process, it include "mae", "rmse" and "accuracy", default "mae".
#' @param num_round (Integer) The number of iterations over training data to train the model, default = 10.
#' @param  array_batch_size (Integer) It defines number of samples that going to be propagated through the network for each update weight, default 128.
#' @param learning_rate  The learn rate for training process.
#' @param momentum  (Float, 0~1) Momentum for moving average, default 0.9.
#' @param wd (Float, 0~1) Weight decay, default 0.
#' @param randomseeds  Set the seed used by mxnet device-specific random number.
#' @param initializer_idx  The initialization scheme for parameters.
#' @param verbose  logical (default=TRUE) Specifies whether to print information on the iterations during training.
#' @param \dots Parameters for construncting neural networks used in package "mxnet" (\url{http://mxnet.io/}).
#'  
#' @author Chuang Ma , Zhixu Qiu, Qian Cheng and Wenlong Ma
#' @export
#' @examples 
#' data(wheat_example)
#' Markers <- wheat_example$Markers
#' y <- wheat_example$y
#' cvSampleList <- cvSampleIndex(length(y),10,1)
#' # cross validation set
#' cvIdx <- 1
#' trainIdx <- cvSampleList[[cvIdx]]$trainIdx
#' testIdx <- cvSampleList[[cvIdx]]$testIdx
#' trainMat <- Markers[trainIdx,]
#' trainPheno <- y[trainIdx]
#' validIdx <- sample(1:length(trainIdx),floor(length(trainIdx)*0.1))
#' validMat <- trainMat[validIdx,]
#' validPheno <- trainPheno[validIdx]
#' trainMat <- trainMat[-validIdx,]
#' trainPheno <- trainPheno[-validIdx]
#' conv_kernel <- c("1*18") ## convolution kernels (fileter shape)
#' conv_stride <- c("1*1")
#' conv_num_filter <- c(8)  ## number of filters
#' pool_act_type <- c("relu") ## active function for next pool
#' pool_type <- c("max") ## max pooling shape
#' pool_kernel <- c("1*4") ## pooling shape
#' pool_stride <- c("1*4") ## number of pool kernerls
#' fullayer_num_hidden <- c(32,1)
#' fullayer_act_type <- c("sigmoid")
#' drop_float <- c(0.2,0.1,0.05)
#' cnnFrame <- list(conv_kernel =conv_kernel,conv_num_filter = conv_num_filter,
#'                  conv_stride = conv_stride,pool_act_type = pool_act_type,
#'                  pool_type = pool_type,pool_kernel =pool_kernel,
#'                  pool_stride = pool_stride,fullayer_num_hidden= fullayer_num_hidden,
#'                  fullayer_act_type = fullayer_act_type,drop_float = drop_float)
#' 
#' markerImage = paste0("1*",ncol(trainMat))
#' 
#' trainGSmodel <- train_deepGSModel(trainMat = trainMat,trainPheno = trainPheno,
#'                 validMat = validMat,validPheno = validPheno, markerImage = markerImage, 
#'                 cnnFrame = cnnFrame,device_type = "cpu",gpuNum = 1, eval_metric = "mae",
#'                 num_round = 6000,array_batch_size= 30,learning_rate = 0.01,
#'                 momentum = 0.5,wd = 0.00001, randomseeds = 0,initializer_idx = 0.01,
#'                 verbose =TRUE)
#' predscores <- predict_GSModel(GSModel = trainGSmodel,testMat = Markers[testIdx,],
#'               markerImage = markerImage )


train_deepGSModel <- function(trainMat,trainPheno,validMat,validPheno,markerImage,cnnFrame,device_type = "cpu",gpuNum = "max",
                              eval_metric = "mae",num_round = 6000,array_batch_size= 30,learning_rate = 0.01,
                              momentum = 0.5,wd = 0.00001 ,randomseeds = NULL,initializer_idx = 0.01,verbose =TRUE...){
  requireNamespace("mxnet")
  require(mxnet)
  # demo.metric.mae <- mx.metric.custom("mae", function(label, pred) {
  #   res <- mean(abs(label-pred))
  #   return(res)
  # })
  # this function is used to evluate metrics provide a way to evaluate the performance of a learned model.
  evalfun <- switch(eval_metric,
                    accuracy = mx.metric.accuracy,
                    mae = mx.metric.mae,
                    rmse = mx.metric.rmse)
  # select device type(cpu/gpu) and device number according you computer and task.
  if(device_type == "cpu") { device <- mx.cpu()}
  if(device_type == "gpu") { ifelse(gpuNum == "max", device <- mx.gpu(),device <- lapply(0:(gpuNum -1), function(i) {  mx.gpu(i)}))}
  
  # transform marker matrices into image format
  markerImage <- as.numeric(unlist(strsplit(markerImage,"\\*")))
  trainMat <- t(trainMat)
  validMat <- t(validMat)
  dim(trainMat) <- c(markerImage[1], markerImage[2],1,ncol(trainMat))
  dim(validMat) <- c(markerImage[1], markerImage[2],1,ncol(validMat))
  eval.data <- list(data=validMat, label=validPheno)
  # extract Convolution set from the cnn frame list.
  conv_kernel <- unlist(strsplit(cnnFrame$conv_kernel,"\\*"))
  conv_kernel <- matrix(as.numeric(conv_kernel),ncol = 2,byrow = TRUE)
  conv_stride <- unlist(strsplit(cnnFrame$conv_stride,"\\*"))
  conv_stride <-  matrix(as.numeric(conv_stride),ncol = 2,byrow = TRUE)
  conv_num_filter <- cnnFrame$conv_num_filter
  pool_act_type <- cnnFrame$pool_act_type
  pool_type <- cnnFrame$pool_type
  pool_kernel <-  unlist(strsplit(cnnFrame$pool_kernel,"\\*"))
  pool_kernel <- matrix(as.numeric(pool_kernel),ncol = 2,byrow = TRUE)
  pool_stride <-  unlist(strsplit(cnnFrame$pool_stride,"\\*"))
  pool_stride <- matrix(as.numeric(pool_stride),ncol = 2,byrow = TRUE)
  drop_float <- cnnFrame$drop_float
  if(nrow(conv_kernel) != length(conv_num_filter) ||nrow(conv_kernel)!= nrow(conv_stride)|| nrow(conv_kernel)!= length(pool_act_type)){
    stop("Error: a convolutional layer is only matched with one convolution kernel set one stride,and one activation." )
  }
  if( nrow(conv_kernel) != nrow(pool_kernel)){
    stop("Error: the number of convolutional layers must equal the number of pooling layers." )
  }
  if(nrow(pool_kernel)!= nrow(pool_stride) || nrow(pool_kernel) != length(pool_type)){
    stop("Error: pooling framwork is inconsistent" )
  }
  # extract full connect set from the cnn frame list.
  fullayer_num_hidden <- cnnFrame$fullayer_num_hidden
  fullayer_act_type <- cnnFrame$fullayer_act_type
  if(length(fullayer_num_hidden) - length(fullayer_act_type) != 1){
    stop("Error: the last full connected layer don't need activation.")
  }
  conv_layer_num <- nrow(conv_kernel)
  fullayer_num <- length(fullayer_num_hidden)
  if(fullayer_num_hidden[fullayer_num] != 1){
    stop("Error: the last full connected layer's number of hidden nerurons must is one.")
  }
  if(length(drop_float)- length(fullayer_num_hidden) != 1){
    stop("Error:  the number of dropout layers must one more layer than the full connected  layers.")
  }
  # set Convolution frame
  data <- mx.symbol.Variable('data')
  for(cc in 1:conv_layer_num){
    if(cc == 1){
      assign(paste0("conv",cc),mx.symbol.Convolution(data= data, kernel=conv_kernel[cc,],stride=conv_stride[cc,], num_filter= conv_num_filter[cc])) #Convolution layer
    }else if(cc > 1){
      assign(paste0("conv",cc),mx.symbol.Convolution(data= get(paste0("pool",cc-1)), kernel=conv_kernel[cc,],stride=conv_stride[cc,], num_filter= conv_num_filter[cc])) #Convolution layer
    } 
    
    assign(paste0("conv_Act",cc), mx.symbol.Activation(data= get(paste0("conv",cc)), act_type= pool_act_type[cc])) ### active function
    assign(paste0("pool",cc),mx.symbol.Pooling(data= get(paste0("conv_Act",cc)), pool_type= pool_type[cc] ,  
                                               kernel= pool_kernel[cc,], stride= pool_stride[cc,])) # pool layer
  }
  # set full connect frame
  drop_initial <- mx.symbol.Dropout(data = get(paste0("pool",conv_layer_num)),p =drop_float[1])
  fullconnect_initial <- mx.symbol.Flatten(data= drop_initial)
  for(ss in 1:max(c(fullayer_num -1,1))){
    if(ss == 1){
      assign(paste0("fullconnect_layer",ss),mx.symbol.FullyConnected(data= fullconnect_initial, num_hidden= fullayer_num_hidden[ss]))
      
    } else if(ss > 1){
      assign(paste0("fullconnect_layer",ss),mx.symbol.FullyConnected(data= get(paste0("drop_layer",ss -1)), num_hidden= fullayer_num_hidden[ss]))
    }
    # 
    if(fullayer_num == 1){
      assign(paste0("drop_layer",ss),mx.symbol.Dropout(data= get(paste0("fullconnect_layer",ss)), p = drop_float[ss +1]))
    }
    #  performed below when more than more than one full connnect layer
    if(fullayer_num > 1){
      assign(paste0("fullconnect_Act",ss), mx.symbol.Activation(data= get(paste0("fullconnect_layer",ss)), act_type= fullayer_act_type[ss]))
      assign(paste0("drop_layer",ss),mx.symbol.Dropout(data= get(paste0("fullconnect_Act",ss)), p = drop_float[ss +1]))
    }
    
  }
  # performed below when more than one full connnect layer
  if(fullayer_num > 1){
    assign(paste0("fullconnect_layer",fullayer_num),mx.symbol.FullyConnected(data= get(paste0("drop_layer",ss)), num_hidden= fullayer_num_hidden[fullayer_num]))
    assign(paste0("drop_layer",fullayer_num),mx.symbol.Dropout(data= get(paste0("fullconnect_layer",fullayer_num)), p = drop_float[fullayer_num +1]))
  }
  
  # cnn network
  cnn_nerwork <- mx.symbol.LinearRegressionOutput(data= get(paste0("drop_layer",fullayer_num)))  
  if(!is.null(randomseeds)){mx.set.seed(randomseeds)}
  cnn.object <- mx.model.FeedForward.create(cnn_nerwork, X=trainMat, y=trainPheno,eval.data = eval.data,
                                            ctx= device, num.round= num_round, array.batch.size=array_batch_size,
                                            learning.rate=learning_rate, momentum=momentum, wd=wd,
                                            eval.metric= evalfun,initializer = mx.init.uniform(initializer_idx),
                                            verbose = verbose,
                                            epoch.end.callback=mx.callback.early.stop(bad.steps = 600,verbose = verbose))
}
