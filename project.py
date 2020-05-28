import progressbar
from time import sleep
import pandas

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from numpy import loadtxt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline



# load dataset
dataset = loadtxt('wine.csv', delimiter=',')

X = dataset[:, 1:13].astype(float)#input
Y = dataset[:, 0]#output

# devide data as input and output
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# define model
def modeling():
    # create model
    model = Sequential()
    #relu or sigmoid is not changed anything but result are very good in relu function
    model.add(Dense(8, input_dim=12, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    #I use softmax funtion in the last layer because problem type is Multi-Class Classification problem.
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #adam is comman opitmizer
    #I want to get accuracy results so metrices is accuracy
    #as a lost function i use categorical crossentropy because i can not
    # use binary_cross entropy because its multi-Class classification problem there are 3 types of output.
    return model

print(dummy_y)
print(Y)
#checked outputs


estimator = KerasClassifier(build_fn=modeling, epochs=15000, batch_size=15, verbose=0)#I tried lots of epochs and batch size this one is very good working.

#used Keras Classifier.
kfold = KFold(n_splits=3, shuffle=True)
#shuffle is true because 3 splits must be selected non linear or disorganized
#tree splits are training-validation and testing
a=kfold.get_n_splits()
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (± %.2f%%)" % (results.mean() * 100, results.std() * 100)+" result"+str(results))#I printed the result and standart deviation
