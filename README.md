This is the official GitHub page for the Immune Cell Tissue Origin Predictor (ICTOP). 

In the context of a bachelor's thesis, ICTOP, a classifier capable of predicting the tissue of origin for selected immune cell types was developed.
By applying it across different time points, one can track the immune system’s response to external stressors, such as physical activity. 
The classifier was trained on the Cross-tissue Immune Cell Atlas (https://www.tissueimmunecellatlas.org) and is based on the XGBoost ```mulit:softmax``` algorithm. 
ICTOP learned the patterns between a cell’s gene expression and origin tissue during training. Per cell type, one individual model is availale. 
ICTOP will automatically choose the corresponding model for the individual cell types contained in the input dataset. 


ICTOP can be run via the commandline. Here, a cloning of the git repository is necessary. It can be utilized:

```
run_ICTOP.py <input_filename.h5ad>
```

At the moment, models are available for:
- T cells
- B cells
- ILCs
- Monocytes
- DCs
- pDCs
- Megakaryoctes
