###deep learning with Keras in Python

# Import the Sequential model and Dense layer
from keras.models import Sequential
from keras.layers import Dense

# Create a Sequential model
model = Sequential()

# Add an input layer and a hidden layer with 10 neurons
model.add(Dense(10, input_shape=(2,), activation="relu"))

# Add a 1-neuron output layer
model.add(Dense(1))

# Summarise your model
model.summary()


#######################
### MORE LAYERS

# Instantiate a Sequential model
model = Sequential()

# Add a Dense layer with 50 neurons and an input of 1 neuron
model.add(Dense(50, input_shape=(1,), activation='relu'))

# Add two Dense layers with 50 neurons and relu activation
model.add(Dense(50,activation="relu"))
model.add(Dense(50,activation="relu"))

# End your model with a Dense layer and no activation
model.add(Dense(1))

########################
## compile and evaluate
########################
# Compile your model
model.compile(optimizer="adam",loss="mse")

print("Training started..., this can take a while:")

# Fit your model on your data for 30 epochs
model.fit(time_steps, y_positions, epochs=30)

# Evaluate your model 

print("Final lost value:",model.evaluate(time_steps, y_positions))


#######################
## CLASSIFICATION
# Import seaborn
import seaborn as sns

# Use pairplot and set the hue to be our class
sns.pairplot(banknotes, hue="class") 

# Show the plot
plt.show()

# Describe the data
print('Dataset stats: \n', banknotes.describe())

# Count the number of observations of each class
print('Observations per class: \n', banknotes["class"].value_counts())


#####################
# CLASSIFICATION MODEL

# Import the sequential model and dense layer
from keras.models import Sequential
from keras.layers import Dense

# Create a sequential model
model = Sequential()

# Add a dense layer 
model.add(Dense(1, input_shape=(4,), activation="sigmoid"))

# Compile your model
model.compile(loss='binary_crossentropy', optimizer="sgd", metrics=['accuracy'])

# Display a summary of your model
model.summary()

# Train your model for 20 epochs
model.fit(X_train, y_train, epochs=20)

# Evaluate your model accuracy on the test set
accuracy = model.evaluate(X_test,y_test)[1]

# Print accuracy
print('Accuracy:',accuracy)

#######################
#multiclass classification model

# Instantiate a sequential model
model = Sequential()
  
# Add 3 dense layers of 128, 64 and 32 neurons each
model.add(Dense(128, input_shape=(2,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
  
# Add a dense layer with as many neurons as competitors
model.add(Dense(4, activation="softmax"))
  
# Compile your model using categorical_crossentropy loss
model.compile(loss="categorical_crossentropy",
              optimizer='adam',
              metrics=['accuracy'])
              
#######################
###Prepare your dataset

# Transform into a categorical variable
darts.competitor = pd.Categorical(darts.competitor)

# Assign a number to each category (label encoding)
darts.competitor = darts.competitor.cat.codes 

# Import to_categorical from keras utils module
from keras.utils import to_categorical

# Use to_categorical on your labels
coordinates = darts.drop(['competitor'], axis=1)
competitors = to_categorical(darts.competitor)

# Now print the to_categorical() result
print('One-hot encoded competitors: \n',competitors)
              
# Train your model on the training data for 200 epochs
model.fit(coord_train,competitors_train,epochs=200)

# Evaluate your model accuracy on the test data
accuracy = model.evaluate(coord_test, competitors_test)[1]

# Print accuracy
print('Accuracy:', accuracy)


#####################################################################
# Predict on X_small_test
preds = model.predict(coords_small_test)

# Print preds vs true values
print("{:45} | {}".format('Raw Model Predictions','True labels'))
for i,pred in enumerate(preds):
  print("{} | {}".format(pred,competitors_small_test[i]))

# Extract the indexes of the highest probable predictions
preds = [np.argmax(pred) for pred in preds]

# Print preds vs true values
print("{:10} | {}".format('Rounded Model Predictions','True labels'))
for i,pred in enumerate(preds):
  print("{:25} | {}".format(pred,competitors_small_test[i]))         
#####################################################################
#####################################################################
###multilabel
#####################################################################
#####################################################################

# Instantiate a Sequential model
model = Sequential()

# Add a hidden layer of 64 neurons and a 20 neuron's input
model.add(Dense(64,input_shape=(20,),activation="relu"))

# Add an output layer of 3 neurons with sigmoid activation
model.add(Dense(3,activation="sigmoid"))

# Compile your model with adam and binary crossentropy loss
model.compile(loss="binary_crossentropy",
           optimizer="adam",
           metrics=['accuracy'])

model.summary()


#Training with multiple labels
# Train for 100 epochs using a validation split of 0.2
model.fit(sensors_train, parcels_train, epochs = 100,validation_split=0.2)

# Predict on sensors_test and round up the predictions
preds = model.predict(sensors_test)
preds_rounded = np.round(preds)

# Print rounded preds
print('Rounded Predictions: \n', preds_rounded)

# Evaluate your model's accuracy on the test data
accuracy = model.evaluate(sensors_test,parcels_test)[1]

# Print accuracy
print('Accuracy:', accuracy)


##############################
###The history callback
##############################
#check overfitting

# Train your model and save it's history
history = model.fit(X_train, y_train, epochs=50,
               validation_data=(X_test, y_test))

# Plot train vs test loss during training
plot_loss(history.history["loss"], history.history["val_loss"])

# Plot train vs test accuracy during training
plot_accuracy(history.history["acc"], history.history["val_acc"])


###early stopping
# Import the early stopping callback
from keras.callbacks import EarlyStopping

# Define a callback to monitor val_acc
monitor_val_acc = EarlyStopping(monitor="val_acc", patience=5)

# Train your model using the early stopping callback
model.fit(X_train, y_train, 
           epochs=1000, validation_data=(X_test,y_test),
           callbacks=[monitor_val_acc])



####ModelCheckpoint

# Import the EarlyStopping and ModelCheckpoint callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Early stop on validation accuracy
monitor_val_acc = EarlyStopping(monitor="val_acc", patience= 3)

# Save the best model as best_banknote_model.hdf5
modelCheckpoint = ModelCheckpoint("best_banknote_model.hdf5", save_best_only=True)

# Fit your model for a stupid amount of epochs
history = model.fit(X_train, y_train,
                    epochs=10000000,
                    callbacks=[monitor_val_acc,modelCheckpoint],
                    validation_data=(X_test,y_test))




####chapter 3
##Learning the digits

#####################################################################

# Instantiate a Sequential model
model = Sequential()

# Input and hidden layer with input_shape, 16 neurons, and relu 
model.add(Dense(16, input_shape=(64,), activation="relu"))

# Output layer with 10 neurons (one per digit) and softmax
model.add(Dense(10,activation="softmax"))

# Compile your model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Test if your model works and can process input data
print(model.predict(X_train))

####Is the model overfitting?
#Not quite, the training loss is indeed lower than the test loss, but overfitting 
#happens when, as epochs go by, the test loss gets worse because of the model starting to lose generalization power

# Train your model for 60 epochs, using X_test and y_test as validation data
history = model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test), verbose=0)

# Extract from the history object loss and val_loss to plot the learning curve
plot_loss(history.history["loss"], history.history["val_loss"])

######################
#Do we need more data?
######################


for training_sizes in train_sizes :
  	# Get a fraction of training data
    X_train_frac,_, y_train_frac,_ = train_test_split(X_train, y_train , 
                                                        train_size=training_sizes)
    
    # Set the model weights to the initial weights and fit the model
    model.set_weights(initial_weights)
    model.fit(X_train_frac, y_train_frac, epochs=50, callbacks=[early_stop], verbose=0)

    # Evaluate and store results for the training fraction and the test set
    train_accs.append(model.evaluate(X_train_frac, y_train_frac, verbose=0)[1])
    test_accs.append(model.evaluate(X_test, y_test, verbose=0)[1])

# Plot train vs test accuracies
plot_results(train_accs, test_accs)

################################
#Comparing activation functions
################################
# Set a seed
np.random.seed(27)

# Activation functions to try
activations = ["relu", "leaky_relu", "sigmoid", "tanh"]

# Loop over the activation functions
activation_results = {}

for act in activations:
  # Get a new model with the current activation
  model = get_model(act)
  # Fit the model
  history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20,verbose=-0)
  activation_results[act] = history


##with 100 epochs
# Create a dataframe from val_loss_per_function
val_loss= pd.DataFrame(val_loss_per_function)

# Call plot on the dataframe
val_loss.plot()
plt.show()

# Create a dataframe from val_acc_per_function
val_acc = pd.DataFrame(val_acc_per_function)

# Call plot on the dataframe
val_acc.plot()
plt.show()

#######################
###changing batch sizes
#######################

model = get_model()

# Fit your model for 5 epochs with a batch of size the training set
model.fit(X, y, epochs=5, batch_size=X_train.shape[0])
print("The accuracy when using the whole training set as a batch was: ",
      model.evaluate(X_test, y_test)[1])



# Get a fresh new model with get_model
model = get_model()

# Train your model for 5 epochs with a batch size of 1
model.fit(X_train, y_train, epochs=5, batch_size=1)
print("The accuracy when using a batch of size 1 is: ",
      model.evaluate(X_test, y_test)[1])


#########################
###batch normalization
#########################

# Import batch normalization from keras layers
from keras.layers import BatchNormalization

# Build your deep network
batchnorm_model = Sequential()
batchnorm_model.add(Dense(50, input_shape=(64,), activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(10, activation='softmax', kernel_initializer='normal'))

# Compile your model with sgd
batchnorm_model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])

#########################
#####Batch normalization effects
#########################

# Train your standard model, storing its history
history1 = standard_model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, verbose=0)

# Train the batch normalized model you recently built, store its history
history2 = batchnorm_model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, verbose=0)

# Call compare_acc_histories passing in both model histories
compare_histories_acc(history1, history2)

#########################
###hyperparameter tuning
#########################

# Creates a model given an activation and learning rate
def create_model(learning_rate=0.01, activation='relu'):
  
  	# Create an Adam optimizer with the given learning rate
  	opt = Adam(lr=learning_rate)
  	
  	# Create your binary classification model  
  	model = Sequential()
  	model.add(Dense(128, input_shape=(30,), activation=activation))
  	model.add(Dense(256, activation=activation))
  	model.add(Dense(1, activation='sigmoid'))
  	
  	# Compile your model with your optimizer, loss, and metrics
  	model.compile(optimizer=opt, loss="cross-entropy", metrics=['accuracy'])
  	return model

#########################
###tuning parameters
#########################

# Import KerasClassifier from keras wrappers
from keras.wrappers.scikit_learn import KerasClassifier

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model)

# Define the parameters to try out
params = {'activation':["relu", "tanh"], 'batch_size':[32, 128, 256], 
          'epochs':[50, 100, 200], 'learning_rate':[0.1, 0.01, 0.001]}

# Create a randomize search cv object and fit it on the data to obtain the results
random_search = RandomizedSearchCV(model, param_distributions=params, cv=KFold(3))

# random_search.fit(X,y) takes too long! But would start the search.
show_results()

#########################
#Training with crossvalidation
#########################

# Import KerasClassifier from keras wrappers
from keras.wrappers.scikit_learn import KerasClassifier

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=128, verbose=0)

# Calculate the accuracy score for each fold
kfolds = cross_val_score(model, X, y, cv=3)

# Print the mean accuracy
print('The mean accuracy was:', kfolds.mean())

# Print the accuracy standard deviation
print('With a standard deviation of:', kfolds.std())

#################################
### CHAPTER 4
#################################

#It's a flow of tensors provided with a valid input tensor, return the corresponding output tensor.

# Import keras backend
import keras.backend as K

# Input tensor from the 1st layer of the model
inp = model.layers[0].input

# Output tensor from the 1st layer of the model
out = model.layers[0].output

# Define a function from inputs to outputs
inp_to_out = K.function([inp],[out])

# Print the results of passing X_test through the 1st layer
print(inp_to_out([X_test]))

#make use of the inp_to_out() function, built in the previous exercise, 
#to visualize how neurons learn to separate real from fake dollar bills




