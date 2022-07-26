# **Codes of QDU team**

## **1. Environment**

- pandas==1.1.5
- numpy==1.19.5
- paddlepaddle-gpu==2.3.1.post101
- python--3.7

## **2. Catalogue**

```
./
├── README.md
├── train
│   ├── TSModel
 |       ├──TSModel.ipynb
│   ├── CNN
 |       ├──CNN_nor.ipynb
 |       ├──CNN_42.ipynb
 |       ├──CNN_2022.ipynb
│   ├── GRU
 |       ├──GRU_nor.ipynb
 |       ├──GRU_42.ipynb
 |       ├──GRU_2022.ipynb
├── inference
│   ├── model
│   ├── Common.py
│   ├── dataset.py
│   ├── predict.py
│   ├── prepare.py
│   ├── testf.py
│   ├── TSModel.py
│   ├── dist_dict.pickle
```


## **4. Instructions  **
- Codes for training models are in the train/ file including three kinds of models and a jupyter to generate training sets.
The TsModel/ file contains the training code of the CNN model with multichannel time series. The CNN/ file contains three 
training jupyters of the CNN model with spatial-temporal information using three different datasets. The GRU/ file includes
three jupyters of the GRU model with spatial-temporal information using three different datasets.
- The final score in phase 3 can be reproduced by evaluating the codes in inference/ file, which is same as the final submitted 
file.

## **5. Our Framework**
Firstly, five variables including Wspd, Pab1, Etmp, Itmp, Patv and spatial distribution information were selected from 
all the available information according to our multiple attempts. Secondly, we generated sequence features from 
the five variables and obtained three small datasets by random sampling and outlier processing. 
Then, three deep learning models based on CNN and GRU were constructed to integrate the spatial-temporal information. 
Finally, the forecast values were assembled by model averaging to improve the robustness.




