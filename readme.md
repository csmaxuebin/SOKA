# This code is the source code implementation for the paper "SOKA：An Optimized K-Anonymity Algorithm in Location-based Services"！
# Abstract

In recent years, the demand for location-based services (LBS) has grown explosively with the development of wireless networks and mobile services. It is of paramount importance to protect location privacy which may disclose users’ private information. Previous studies have focused on location obfuscation-based scheme such as perturbing and location spoofing. However, these methods reduce data availability and increase time overhead. In this paper, we apply the self-organizing map network in LBS for the first time to transform the highdimensional data into two-dimensional data while preserving the topology of the original data. In addition, we propose a constraint-based k-anonymity region construction algorithm which considering l-diversity and t-closeness. For cloaking regions that do not satisfy the conditions, a cloaking region adjustment strategy is used to add a varying number of optimal dummy locations based
on location semantics, making it difficult for attackers to distinguish them. Experimental results show that our approach is able to reduce the information loss while significantly reducing the running time, achieving a balance between privacy-preserving capability and data utility

# Experimental Environment

```
- Python 3.7 (Anaconda Python recommended)
- MiniSom 2.2.2
- PyTorch 1.10
- torchvision
- nltk
- pandas
- scipy
- tqdm
- backpack-for-pytorch
- scikit-image
- scikit-learn
- tensorboard
- tensorflow==2.5.0 (for tensorboard visualizations)
- Intel Core i5 CPU 2.3GHz and 8GB RAM 
```

## Datasets

```
- Gowalla datase
- Adult Dataset<br>
   https://www.kaggle.com/wenruliu/adult-income-dataset/data
   Create an empty folder called ***"data"*** and put the "adult.csv" under it before using the code.
```

## Experimental Setup

 **Evaluation Metrics:**
 -   **Information Loss:** Information loss is an important evaluation metric for measuring big data privacy protection algorithms, which can be classified as numerical and categorical based on data types.
 -  **Anonymous success rate**
 -  **Running time**
## Python Files
 -   **som.py**: This code is an implementation of a Self-Organizing Map (SOM), which is used to train and visualize the results of the SOM algorithm on a given set of feature data. The `train` method is used to train the SOM model using the SOM algorithm, and the `show` method is used to visualize the mapping results of the SOM.
-   **runner.py**: This code defines a `TrainRunner` class that automates the process of data preprocessing and model training.
-   **kdtree.py**:  is code defines two classes: `Mondrian` and `K_Anonymity`, which are used for data perturbation and achieving k-anonymity. These classes provide different levels of data privacy protection by applying the Mondrian algorithm and k-anonymity techniques, with the additional option of utilizing a Self-Organizing Map (SOM) for perturbation..
-   **main.ipynb**: The code cell imports some Python libraries and defines variables and functions.
-   **model.py**: The code defines a neural network model, trains it, and provides methods for prediction and evaluation of the model's performance.

## Experimental Results
![输入图片说明](https://github.com/csmaxuebin/SOKA/blob/main/picture/3.png)Fig.7 shows the effects of the number of records and the number of attributes in the dataset on the information loss performance of the algorithms, respectively
![输入图片说明](https://github.com/csmaxuebin/SOKA/blob/main/picture/2.png)Fig.8 shows the variation of anonymization processing time with k for different thresholds for Greedy Search, Improved Clustering and SOKA algorithms.
![输入图片说明](https://github.com/csmaxuebin/SOKA/blob/main/picture/3.png)The effect of location semantic difference 𝑢 on the anonymization success rate is shown in Fig.9(a).
Fig.9(b) shows the ratio of the number of false locations to the total number of locations in the anonymization region when the number of locations in the map is 10000.
Fig.9(c) and Fig.9(d) shows the anonymization success rate decreases when the number of location queries is small and the users are dispersed under a certain k. 


## Update log

```
- {24.06.17} Uploaded overall framework code and readme file
```



