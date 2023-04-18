import datetime
import glob
import os
from pathlib import Path
import pickle
import random
import dill
from imutils import paths
from numpy import size
from Global_Model import *
from Model import *
import copy
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import sys
from selection import Selection

def main(participants,rounds,epochs,batch):
    
    # Model Folder
    local_model_path='..\..\models\localModels\\'
    global_model_path='..\..\models\globalModel\\'
    eval_df =[]

    # Selection module
    select=Selection()
    
    # Data path
    paths_clients=glob.glob(os.path.abspath('..\..')+"\data\ClientsData\*")
    
    # Rounds
    # rounds=int(sys.argv[1])

    # Participants
    # participants=2
    
    
    # Optimiser
    learning_rate = 0.01 
    loss='categorical_crossentropy'
    metrics = ['accuracy']
    optimizer = SGD(learning_rate=learning_rate,decay=learning_rate /rounds,momentum=0.9) 
    
    # Intialize global models
    model=Model()
    init_model=model.build_model(784,10)
    init_weights=init_model.get_weights()

    gm=ModelClass(model_name="Global_Model",class_name=init_model.__class__.__name__,model_init=init_model,model_local=None)
    
    # Cleaning the models folder
    files_local=  glob.glob(local_model_path+'*')
    files_global= glob.glob(global_model_path+'*')
   
    for f in files_local:
        os.remove(f)

    for f in files_global:
        os.remove(f)

    # Start training by rounds
    for round in range(rounds):
       
        clts_models=[]
        scaled_local_weight_list = list()
       
        print('start executing the round : ' + str(rounds))
        # Participant Selection 
        if participants==len(glob.glob(os.path.abspath('..\..')+"\data\ClientsData\*")):
            print("No need for selection")
        else:
            paths_clients=random.choice(list(select.randomselection(os.path.abspath('..\..')+"\data\ClientsData\\",2)))
        print("Used clients...")
        print(paths_clients)
        print(paths_clients[0].split('\\')[-1])
        for clt in range(len(paths_clients)):
            print("Train the client : " +  paths_clients[clt].split('\\')[-1])
            data=open(paths_clients[clt],'rb')
            data_clt=pickle.load(data)
            local_model=copy.copy(init_model)
            local_model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
            local_model.set_weights(init_weights)
            clt_model=local_model.fit(tf.stack(data_clt.X_train),tf.stack(data_clt.y_train),epochs=epochs,batch_size=batch)
            local_model.save(local_model_path+paths_clients[clt].split('\\')[-1]+'.h5')
            mdl_obj=ModelClass(model_name=paths_clients[clt].split('\\')[-1],class_name=local_model.__class__.__name__,model_init=None,model_local=os.path.abspath(local_model_path+paths_clients[clt].split('\\')[-1]+'.h5'))
            print(mdl_obj)
            clts_models.append(mdl_obj)


        # Server step Global model
        print("Global model...")
        for local_client in range(len(clts_models)):
            f=open('..\..\data\ClientsData\Client_'+str(local_client),'rb')
            scaled_local_weight_list.append(model.scale_weights(load_model(clts_models[local_client].model_local).get_weights(),pickle.load(f)))
            f.close()
        
        # Clean tf session 
        tf.keras.backend.clear_session()

        # Averging the weights
        print("Avereging the weights")
        new_weights=model.avg_weights(scaled_local_weight_list)
        init_model.set_weights(new_weights)
        init_model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
        init_model.save(global_model_path+datetime.datetime.today().strftime ('%H-%M-%S-%d-%b-%Y')+"_round_"+str(round)+'.h5')
        print("Hado init weights :")

        # Evaluation Step
        print('Evaluating gloabal models')
        for clt in range(len(paths_clients)):
            print("Evaluate the client : " +  paths_clients[clt].split('\\')[-1])
            data=open(paths_clients[clt],'rb')
            data_clt=pickle.load(data)
            print('Local models evaluation of client : ' +  paths_clients[clt].split('\\')[-1])
            eval_results=model.evaluation(X_test=tf.stack(data_clt.X_test),Y_test=tf.stack(data_clt.y_test),model=load_model(local_model_path+paths_clients[clt].split('\\')[-1]+'.h5'),comm_round=rounds)
            tmp_eval_l={'acc':eval_results[0],'loss':float(eval_results[1]),'round':round,'client': paths_clients[clt].split('\\')[-1],'type':'local'}
            print('Global model evaluation of client : ' +  paths_clients[clt].split('\\')[-1])
            list_of_files = glob.glob(global_model_path+'*') # * means all if need specific format then *.csv
            latest_file = max(list_of_files, key=os.path.getctime)
            print('Used model : '+ latest_file)      
            eval_results=model.evaluation(X_test=tf.stack(data_clt.X_test),Y_test=tf.stack(data_clt.y_test),model=load_model(latest_file),comm_round=1)
            tmp_eval_g={'acc':eval_results[0],'loss':float(eval_results[1]),'round':round,'client': paths_clients[clt].split('\\')[-1],'type':'global'}
            eval_df.append(tmp_eval_l)
            eval_df.append(tmp_eval_g)
        df=pd.DataFrame(eval_df)
    df.to_csv(global_model_path+datetime.datetime.today().strftime ('%H-%M-%S-%d-%b-%Y')+'.csv')




    