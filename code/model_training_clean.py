# -*- coding: utf-8 -*-
"""
Myers–Briggs Type Indicator classification based on Twitter data

Model training
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix 
import pickle
from keras.models import Sequential
from keras import layers
from keras import backend
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from keras.optimizers import RMSprop
from keras.layers import Dropout

#load
preprocessed_featuresdf=pd.read_feather('preprocessed_features')
preprocessed_featuresdf=preprocessed_featuresdf.drop(['posts'], axis=1)
labelsdf=pd.read_feather('labels')
PERSONALITY_DIM=['ie', 'ns', 'ft', 'pj']  
    
# onehot encoding
onehotEncoder = OneHotEncoder(sparse=False)
oh_labels = labelsdf.loc[:,'type'].values.reshape(labelsdf.shape[0], 1)
labelsdf['oh_label']=list(onehotEncoder.fit_transform(oh_labels))
labelsdf['oh_label_argmax']=[label.argmax() for label in labelsdf['oh_label'].values]

# split
def pd_train_test_split(Xdf,ydf, test_size=0.2, randomstate=123):
    array_size=Xdf.shape[0]
    suffle_idx=np.arange(array_size)
    np.random.seed(randomstate)
    np.random.shuffle(suffle_idx)
    test_portion=int(array_size*test_size)
    
    test_shuffle=suffle_idx[:test_portion]
    train_shuffle=suffle_idx[test_portion:]
    
    X_test_df=Xdf.iloc[test_shuffle,:].reset_index(drop=True)
    X_train_df=Xdf.iloc[train_shuffle,:].reset_index(drop=True)
    y_test_df=ydf.iloc[test_shuffle,:].reset_index(drop=True)
    y_train_df=ydf.iloc[train_shuffle,:].reset_index(drop=True)
    return X_train_df, y_train_df, X_test_df, y_test_df
    
    
X_train_df, y_train_df, X_test_df, y_test_df=\
    pd_train_test_split(preprocessed_featuresdf, labelsdf, randomstate=42)

# tfidf encoding
tfidfvectorizer = TfidfVectorizer(max_features=3000, min_df=1, max_df=0.8)  
train_tfidf = tfidfvectorizer.fit_transform(X_train_df['clean_tweets']).toarray()
test_tfidf = tfidfvectorizer.transform(X_test_df['clean_tweets']).toarray()

# scaling
X_train_toscale=X_train_df.drop(['clean_tweets'], axis=1)
X_test_toscale=X_test_df.drop(['clean_tweets'], axis=1)
stdscaler=StandardScaler()
X_train_sclaed=stdscaler.fit_transform(X_train_toscale)
X_test_sclaed=stdscaler.transform(X_test_toscale)

train_allfeatures=np.concatenate([train_tfidf, X_train_sclaed], axis=1)
test_allfeatures=np.concatenate([test_tfidf, X_test_sclaed], axis=1)

# performance metrics
def recall_m(y_true, y_pred):
        true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + backend.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + backend.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+backend.epsilon()))

def roc_auc_m(y_true, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true = lb.transform(y_true)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_true, y_pred, average=average)

def plot_history(history, savename=False, verbose=False):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 12))
    fig, axs = plt.subplots(2,2,figsize=(12,10))
    axs[0,0].plot(x, acc, 'b', label='Training acc')
    axs[0,0].plot(x, val_acc, 'r', label='Validation acc')
    axs[0,0].set_title('Training and validation accuracy')
    axs[0,0].legend()
    axs[0,1].plot(x, loss, 'b', label='Training loss')
    axs[0,1].plot(x, val_loss, 'r', label='Validation loss')
    axs[0,1].set_title('Training and validation loss')
    axs[1,1].plot(x, history.history['recall_m'], 'b', label='Training recall')
    axs[1,1].plot(x, history.history['val_recall_m'], 'r', label='Validation recall')
    axs[1,1].set_title('Training and validation recall')
    axs[1,1].legend()
    axs[1,0].plot(x, history.history['precision_m'], 'b', label='Training precision')
    axs[1,0].plot(x, history.history['val_precision_m'], 'r', label='Validation precision')
    axs[1,0].set_title('Training and validation precision')
    axs[1,0].legend()
    if(verbose):
        print('Min validation loss on epoch '+ str(x[np.array(val_loss).argmin()]))
    if(savename):
        plt.savefig(savename)
    else:
        plt.show()

def test_run(trainX,trainy,testX,testy,epochs=1000,
             validation_split=0.2,batch_size=128, dim_asix='MBTI',
             clf_name='default'):
    def mlp_model(input_size, output_size=1, softmax=False):
        model = Sequential()
        model.add(layers.Dense(64, input_dim=input_size,
                               activation='relu'))
        model.add(Dropout(0.5))
        model.add(layers.Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        if(output_size==1):
            model.add(layers.Dense(output_size, activation='sigmoid'))
            lossf='binary_crossentropy'
        else:
            model.add(layers.Dense(output_size, activation='softmax'))
            lossf='categorical_crossentropy'
        model.compile(optimizer=RMSprop(lr=0.000015),
                      loss=lossf,
                      metrics=['acc',f1_m,precision_m, recall_m])
        return model
    if(len(trainy.shape)==1):
        trainysize=1
    else:
        trainysize=trainy.shape[1]
    clfmodel=mlp_model(trainX.shape[1],trainysize)
    history = clfmodel.fit(trainX ,
                           trainy ,
                           batch_size=batch_size,
                           validation_split=validation_split,
                           epochs=epochs, 
                           verbose=0)
    plot_history(history,savename=dim_asix+' '+clf_name+'_history.png') 
    loss, accuracy, f1_score, precision, recall = \
        clfmodel.evaluate(testX, testy, verbose=0)
    test_predictions=clfmodel.predict(testX)
    # convert to labeled class
    test_predictions_labels=[]
    actual_predictions_labels=[]
    if(trainysize==1):
        for prediction in test_predictions:
            test_predictions_labels.append(dim_asix[round(prediction)])
        for actual in testy:
            actual_predictions_labels.append(dim_asix[actual])
    else:
        test_predictions_labels=onehotEncoder.inverse_transform(test_predictions)
        actual_predictions_labels=onehotEncoder.inverse_transform(testy)

    roc_auc=roc_auc_m(actual_predictions_labels,test_predictions_labels)
    cm=confusion_matrix(actual_predictions_labels,test_predictions_labels)

    return clfmodel, history, cm,\
            test_predictions_labels,\
            actual_predictions_labels,\
            loss, accuracy, f1_score, precision, recall, roc_auc

# full MBTI classifiers

# def
allfeatures_nosmote_X=train_allfeatures
tfidf_nosmote_X=train_tfidf
nosmote_y=np.array(y_train_df['oh_label'].tolist())
tosmote_y=np.array(y_train_df['oh_label_argmax'].tolist())
test_results=[]

# all features test
clfmodel, history, cm,\
        test_predictions_labels,\
        actual_predictions_labels,\
        loss, accuracy, f1_score, precision, recall, roc_auc=\
    test_run(allfeatures_nosmote_X,
             nosmote_y,
             test_allfeatures,
             np.array(y_test_df['oh_label'].tolist()))
test_results.append(('all features',(clfmodel, history, cm,\
        test_predictions_labels,\
        actual_predictions_labels,\
        loss, accuracy, f1_score, precision, recall, roc_auc)))
print('All features :',precision, roc_auc)

# SMOTE test
smt = SMOTE()
allfeatures_smote_X, smote_y =\
    smt.fit_sample(allfeatures_nosmote_X,
                   tosmote_y)
oh_smote_y=[]
for slabel in smote_y:
    ohl=np.zeros((len(nosmote_y[0],)))
    ohl[slabel]=1
    oh_smote_y.append(ohl)

clfmodel, history, cm,\
        test_predictions_labels,\
        actual_predictions_labels,\
        loss, accuracy, f1_score, precision, recall, roc_auc=\
    test_run(allfeatures_smote_X,
             np.array(oh_smote_y),
             test_allfeatures,
             np.array(y_test_df['oh_label'].tolist()))
test_results.append(('SMOTE',(clfmodel, history, cm,\
        test_predictions_labels,\
        actual_predictions_labels,\
        loss, accuracy, f1_score, precision, recall, roc_auc)))
print('SMOTE:',precision, roc_auc)

# NearMiss test
nr = NearMiss()
allfeatures_nearmiss_X, nearmiss_y =\
    nr.fit_sample(allfeatures_nosmote_X,
                  tosmote_y)
oh_nearmiss_y=[]
for slabel in nearmiss_y:
    ohl=np.zeros((len(nosmote_y[0],)))
    ohl[slabel]=1
    oh_nearmiss_y.append(ohl)

clfmodel, history, cm,\
        test_predictions_labels,\
        actual_predictions_labels,\
        loss, accuracy, f1_score, precision, recall, roc_auc=\
    test_run(allfeatures_nearmiss_X,
             np.array(oh_nearmiss_y),
             test_allfeatures,
             np.array(y_test_df['oh_label'].tolist()))

test_results.append(('NearMiss',(clfmodel, history, cm,\
        test_predictions_labels,\
        actual_predictions_labels,\
        loss, accuracy, f1_score, precision, recall, roc_auc)))
print('NearMiss:',precision, roc_auc)

# only tfidf features test
clfmodel, history, cm,\
        test_predictions_labels,\
        actual_predictions_labels,\
        loss, accuracy, f1_score, precision, recall, roc_auc=\
    test_run(tfidf_nosmote_X,
             nosmote_y,
             test_tfidf,
             np.array(y_test_df['oh_label'].tolist()))
test_results.append(('tfidf',(clfmodel, history, cm,\
        test_predictions_labels,\
        actual_predictions_labels,\
        loss, accuracy, f1_score, precision, recall, roc_auc)))
print('tfidf:',precision, roc_auc)

# personality dimension classifier
for pdim in PERSONALITY_DIM:
    print('personality dimension ',pdim)
    nosmote_y=np.array(y_train_df[pdim].tolist())
    tosmote_y=np.array(y_train_df[pdim].tolist())
    clfmodel, history, test_predictions, testy, loss, accuracy, f1_score, precision, recall, roc_auc=\
        test_run(allfeatures_nosmote_X,
                 nosmote_y,
                 test_allfeatures,
                 np.array(y_test_df[pdim].tolist()))
    plot_history(history) 
    test_results.append(('all features '+pdim,(clfmodel, history, test_predictions, testy, loss, accuracy, f1_score, precision, recall, roc_auc)))
    print(pdim,'all features ',precision, roc_auc)
    smt = SMOTE()
    allfeatures_smote_X, smote_y =\
        smt.fit_sample(allfeatures_nosmote_X,
                       tosmote_y)
    clfmodel, history, test_predictions, testy, loss, accuracy, f1_score, precision, recall, roc_auc=\
        test_run(allfeatures_smote_X,
                 np.array(smote_y),
                 test_allfeatures,
                 np.array(y_test_df[pdim].tolist()))
    test_results.append(('SMOTE'+pdim,(clfmodel, history, test_predictions, testy, loss, accuracy, f1_score, precision, recall, roc_auc)))
    print(pdim,'SMOTE:',precision, roc_auc)
    nr = NearMiss()
    allfeatures_nearmiss_X, nearmiss_y =\
        nr.fit_sample(allfeatures_nosmote_X,
                      tosmote_y)
    clfmodel, history, test_predictions, testy, loss, accuracy, f1_score, precision, recall, roc_auc=\
        test_run(allfeatures_nearmiss_X,
                 np.array(nearmiss_y),
                 test_allfeatures,
                 np.array(y_test_df[pdim].tolist()))
    test_results.append(('NearMiss'+pdim,(clfmodel, history, test_predictions, testy, loss, accuracy, f1_score, precision, recall, roc_auc)))
    print(pdim,'NearMiss:',precision, roc_auc)
    clfmodel, history, test_predictions, testy, loss, accuracy, f1_score, precision, recall, roc_auc=\
        test_run(tfidf_nosmote_X,
                 nosmote_y,
                 test_tfidf,
                 np.array(y_test_df[pdim].tolist()))
    test_results.append(('tfidf'+pdim,(clfmodel, history, test_predictions, testy, loss, accuracy, f1_score, precision, recall, roc_auc)))
    print(pdim,'tfidf:',precision, roc_auc)

    
# gather results
result_names=[testtuple[0] for testtuple in test_results]
result_classifiers=[testtuple[1][0] for testtuple in test_results]
result_history=[testtuple[1][1] for testtuple in test_results]
result_cm=[testtuple[1][2] for testtuple in test_results]
results_predictions=[testtuple[1][3] for testtuple in test_results]
results_predictions=[np.array(predictions).reshape(len(predictions),) for\
                     predictions in results_predictions]
results_actual=[testtuple[1][4] for testtuple in test_results]

result_performance=np.array([testtuple[1][5:] for testtuple in test_results])
performance_col=['loss', 'accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']

predictionsdf=pd.DataFrame(np.array(results_predictions).T,columns=result_names)
predictionsdf=pd.concat([predictionsdf,y_test_df],axis=1)

performancedf=pd.DataFrame(result_performance,
                           columns=performance_col,
                           index=result_names)

performancedf.to_excel('performance.xlsx')
predictionsdf.to_excel('predictions.xlsx')
pickle.dump((result_classifiers, result_history),open( 'classifiers.pkl', "wb" ))
pickle.dump(result_cm ,open( 'confusion_matrices.pkl', "wb" ))

#eof