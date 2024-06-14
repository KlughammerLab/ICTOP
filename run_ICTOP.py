#!/usr/bin/env python3
# python script for loading + using the classifier

import xgboost as xgb
import argparse
import pandas as pd
import anndata as ad
import numpy as np
import scanpy as sc
import sklearn
from scipy.sparse import vstack
from sklearn import metrics
from sklearn.metrics import *
from sklearn.model_selection import (
    # LearningCurveDisplay,
    # learning_curve,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_mapping = {
            0: 'BLD',
            1: 'BMA',
            2: 'CAE',
            3: 'DUO',
            4: 'ILE',
            5: 'JEJEPI',
            6: 'JEJLP',
            7: 'LIV',
            8: 'LLN',
            9: 'LNG',
            10: 'MLN',
            11: 'OME',
            12: 'SCL',
            13: 'SKM',
            14: 'SPL',
            15: 'TCL',
            16: 'THY'
}

# list of the corresponding model for the Celltype 
model_names = {
    'bestT_softprob': ['T', 'T cells', 'Tcells', 't', 't cells', 'tcells'],
    'bestIlc_softprob': ['Innate Lymphoid Cells', 'ilc', 'ILC', 'Ilc'],
    'bestB_softprob': ['B', 'B cells', 'Bcells', 'b', 'b cells', 'bcells'],
    'bestPdc_softprob':['pDC', 'Plasmacytoid Dendritic Cells', 'Pdc'],
    'bestMgk_softprob':['Megakaryocytes/platelets', 'MGK', 'mgk'],
    'bestMono_softprob':['mono', 'Mono', 'Monocytes'],
    'bestDc_softprob': ['DC', 'dc', 'Dc']
}
    
# the softmax models 
model_names_softmax = {
    'Mgk': ['Megakaryocytes/platelets', 'MGK', 'mgk'], 
    'Pdc': ['pDC', 'Plasmacytoid Dendritic Cells', 'Pdc'],
    'Dc': ['DC', 'dc', 'Dc'],
    'T': [ 'T', 'T cells', 'Tcells', 't', 't cells', 'tcells'],
    'B': ['B', 'B cells', 'Bcells', 'b', 'b cells', 'bcells'],
    'Ilc': ['Innate Lymphoid Cells', 'ilc', 'ILC', 'Ilc'],
    'Mono': ['mono', 'Mono', 'Monocytes']
}
    
def pred_split(data): # for each celltype, call the corresponding model
    first = True
    adata = data
    combined_adata = ad.AnnData() # creation of the output object
    
    for celltype in adata.obs["celltypist_cell_label_coarse"].unique():
    
        for modelname in model_names.keys():
        
            for submodel in model_names[modelname]:
        
                if celltype == submodel:  # Find the corresponding model for that celltype
                    print(celltype)
                    print("Modelname: " , modelname)
                    sub_type = adata[adata.obs["celltypist_cell_label_coarse"] == celltype].copy()
    
                    for person in sub_type.obs["participant"].unique():
                        sub_person = sub_type[sub_type.obs["participant"] == person].copy()
    
                        for tp in sub_person.obs["timepoint"].unique(): 
                            
                            subset = sub_person[sub_person.obs["timepoint"] == tp].copy()
        
                            X = subset.to_df()
                            deval = xgb.DMatrix(X) 
                            
                            model_name = f"Models/{modelname}.model"
                            print("loading model: ", model_name)
                    
                            bst = load_model(model_name)
                            
                            predictions = prediction(bst, deval)
                            
                            
                            # Add predictions to the subset
                            subset.obs["Predicted_tissue"] = [label_mapping[int(pred)] for pred in predictions]
                            subset.obs["celltype_model"] = modelname
                    
                            # Append the subset to the combined AnnData object
                            if first:
                                combined_adata = subset.copy()
                                first = False
                            else:
                                combined_adata = ad.concat([combined_adata, subset], axis=0)

                            
                    
        else:
            print("no submodel : ", submodel , " for celltype: " , celltype)

    return combined_adata # return of the input adata with an additional .obs for the predictions
                                

# loads the corresponding model
def load_model(modelname: str) -> xgb.Booster:
    bst = xgb.Booster()
    bst.load_model(modelname)
    return bst

# calls the prediction function and returns the predicted tissues for that data
def prediction(bst: xgb.Booster, deval: xgb.DMatrix) -> list:
    y_pred_prob = bst.predict(deval) # predict
    predictions = np.argmax(y_pred_prob, axis =1) # get the tissue with the highest prediction probability
    return predictions
        
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Classifier for single-cell data")
    parser.add_argument("filename", type=str, help="Path to the input .h5ad file")
    args = parser.parse_args()
    print("reading file")
    
    adata = sc.read_h5ad(args.filename) # read the dataset

    result_adata = pred_split(adata) # call the prediction method
    
    for cellt in result_adata.obs["celltype_model"].unique(): # print the results
        print(cellt)
        adata_subset = result_adata[(result_adata.obs["celltype_model"] == cellt)]
        print(adata_subset.obs["Predicted_tissue"].value_counts())

    # save the results as an individual file, ready for further usage
    result_adata.write('predictedAnndata.h5ad')
        
    
