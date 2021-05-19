from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import text_to_word_sequence
from preprocessing_ap import PatternTokenizer
from preprocessing_ap import listToString
from keras.preprocessing import sequence
from keras.layers import Bidirectional, GlobalMaxPool1D
from sklearn.metrics import classification_report
import preprocessing_ap as pap 
import scrapping_ap as sap

MAX_LEN=150
max_words=20000
# def create_model(max_len=MAX_LEN,max_words=20000):
#     inputs = Input(name='inputs',shape=[max_len])
#     #current_layer=.........(previous_layer)
#     inp = Input(shape=(maxlen,))
#     x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
#     x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
#     x = GlobalMaxPool1D()(x)
#     x = Dense(50, activation="relu")(x)
#     x = Dropout(0.1)(x)
#     x = Dense(6, activation="sigmoid")(x)
#     model = Model(inputs=inp, outputs=x)
#     #the model just connect the first and last all other interactions between layers are already cached
#     return model

def MyBaseline_Model(maxlen=MAX_LEN, max_features=20000):
    embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(100, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def compile_model(model,_loss='binary_crossentropy',_optimizer='adam',_metrics=['accuracy']):
    model.compile(loss=_loss,optimizer=_optimizer,metrics=_metrics)


def fit_model(model,sequences_matrix,Y_train,_batch_size=128,_epochs=10,_validation_split=0.2,_callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)]):
    model.fit(sequences_matrix,Y_train,batch_size=_batch_size,epochs=_epochs,validation_split=_validation_split)
def evaluate_model(model,test_sequences_matrix,Y_test):
    return model.evaluate(test_sequences_matrix,Y_test) 

def predict(model,our_string,tok,max_len=MAX_LEN):
    
    tokenizer = PatternTokenizer()
    my_text = listToString(tokenizer.process_text(our_string))
    l=[]
    l.append(my_text)
    test_sequences = tok.texts_to_sequences(l)
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
    return model.predict(test_sequences_matrix) 

def classification_Report(model, Y_test, test_sequences_matrix):
    predictions=model.predict(test_sequences_matrix)
    predictions=predictions.reshape(1563,)
    predictions=list(map(lambda x: 1 if x > 0.5 else 0 , predictions))
    print(classification_report(Y_test, predictions))

def predict_url(url,model,tok):
    max_len = 150
    #It is the process of separating each word in a text as a unit and you can later
    #you use the tokenize data for things like term frequency and word clouds
    corpus=sap.scrap_raw_text(url)
    corpus=pap.text_preprocessing(corpus)
    tokenizer=pap.PatternTokenizer()
    corpus=' '.join(corpus)
    corpus=tokenizer.process_text(corpus)
    corpus=[corpus]
    c = tok.texts_to_sequences(corpus)
    c = sequence.pad_sequences(c,maxlen=max_len)
    prediction= model.predict(c)
    print(prediction)
    return prediction