import unittest
class Test_Model(unittest.TestCase):
    # def test_model(self):
    #     try:
            
    #         import preprocessing_ap as pap
    #         import model_ap as mp
    #         import pandas as pd
    #         from keras.preprocessing.text import Tokenizer
    #         from keras.preprocessing import sequence
    #         import pickle5 as pickle

    #         df_train=pd.read_csv('train_data.csv')
    #         df_test=pd.read_csv('test_data.csv')
    #         X_train,X_test,Y_train,Y_test = pap.x_y_train_test(df_train, df_test)
    #         print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
    #         sequences_matrix,test_sequences_matrix,tokenizer= pap.sequence_matrix(X_train,X_test)
    #         # saving the tokenizer
    #         with open('tokenizer.pickle', 'wb') as handle:
    #             pickle.dump(tokenizer, handle)
    #         model = mp.MyBaseline_Model()
    #         mp.compile_model(model)
    #         mp.fit_model(model, sequences_matrix,Y_train)
    #         accr = mp.evaluate_model(model,test_sequences_matrix,Y_test)
    #         print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    #         model.save("model_Bi_LSTM.h5")
           
    #     except Exception as e:
    #         print(str(e))  

    def test_prediction(self):
        import keras
        import model_ap as mp
        import pickle5 as pickle

        # loading
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        reconstructed_model = keras.models.load_model("model_Bi_LSTM.h5")
        mp.predict_url("https://stackoverflow.com/questions/45735070/keras-text-preprocessing-saving-tokenizer-object-to-file-for-scoring",reconstructed_model,tokenizer)
     

if __name__ == "__main__":
    unittest.main()       
