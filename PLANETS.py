#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math 


# In[2]:


planets= pd.read_csv("D:\Downloads\exoplanets\cumulative.csv")


# In[3]:


planets.head()


# In[4]:


print(planets.shape)


# In[5]:


confirmed=[]

for i in range(planets.shape[0]):
    if planets.koi_disposition[i] == "CONFIRMED":
        confirmed.append(1)
    else:
        confirmed.append(0)
print(confirmed[:10])





    


# In[6]:


planets.drop(columns='koi_disposition')


# In[ ]:





# In[7]:


planets.drop(columns=['koi_pdisposition','koi_disposition'], inplace=True)


# In[8]:


planets['confirmed']=confirmed


# In[9]:


planets.head()


# In[10]:


plt.figure(figsize=(50,50))
sns.heatmap(corr, cmap='mako_r',annot=True)
plt.show()


# In[12]:


print(planets.isnull().mean() * 100)


# In[13]:


planets.drop(columns=['koi_teq_err1','koi_teq_err2'], inplace=True)


# In[14]:


print(planets.median(skipna=True, numeric_only=True))


# In[15]:


planets.fillna(planets.median(numeric_only = True))


# In[16]:


print(planets.isnull().mean() * 100)


# In[17]:


planets.fillna(planets.median(numeric_only = True), inplace=True)


# In[18]:


print(planets.isnull().mean() * 100)


# In[19]:


planets.koi_teq.hist(bins=50, figsize=(20,16),range=(273,373))
plt.show()


# In[20]:


print(planets.query("koi_teq<=373 and koi_teq>=273"))


# In[21]:


print(planets.koi_prad.mean())


# In[22]:


X_train["koi_slogg"].hist(bins=50, figsize=(50,30))
plt.show()


# In[23]:


planets["koi_slogg"].hist(bins=50, figsize=(50,30))
plt.show()


# In[ ]:





# SPLITTING

# In[24]:


import sklearn 


# In[25]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


planets["rocky"] = (planets["koi_prad"] <= 4).astype(int)
planets["habitable"] = ((planets["koi_teq"] >= 200) & (planets["koi_teq"] <= 350)).astype(int)

# --- Step 2: Create stratification categories ---
planets["rad"] = pd.cut(planets["koi_prad"], bins=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,np.inf],labels=np.arange(1,16))

planets["teq"] = pd.cut(planets["koi_teq"], bins=[0,200,400,600,800,1000,1200,1400,1600,1800,2000,np.inf],labels=np.arange(1,12))

planets["slogg"] = pd.cut(planets["koi_slogg"], bins =[-np.inf,3,4,5], labels=np.arange(1,4))
# --- Step 3: Combine both categories into a single stratification key ---
planets["strata"] = planets["rad"].astype(str) + "_" + planets["teq"].astype(str)+ "_" +planets["slogg"].astype(str)

# --- Step 4: Drop rare strata (<2 samples) ---
counts = planets["strata"].value_counts()
rare_strata = counts[counts < 2].index
planets = planets[~planets["strata"].isin(rare_strata)].copy()



X = ["koi_prad", "koi_steff", "koi_slogg", "koi_srad","koi_teq", "koi_insol", "rocky", "habitable"]
y = planets["confirmed"]

# --- Step 6: Stratified splitting ---
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in splitter.split(planets, planets["strata"]):
    strat_train_set_n = planets.iloc[train_index]
    strat_test_set_n = planets.iloc[test_index]

# --- Step 7 (optional): Verify stratification ---
def strata_proportions(df):
    return df["strata"].value_counts(normalize=True)

compare = pd.DataFrame({
    "Overall": strata_proportions(planets),
    "Train": strata_proportions(strat_train_set_n),
    "Test": strata_proportions(strat_test_set_n)
}).sort_index()

print(compare.head(10))


planets_train=strat_train_set_n.copy()
planets_test=strat_test_set_n.copy()

X_train = planets_train[["koi_prad", "koi_steff", "koi_slogg", "koi_srad","koi_teq", "koi_insol", "rocky", "habitable"]]
y_train = planets_train["confirmed"]

X_test = planets_test[["koi_prad", "koi_steff", "koi_slogg", "koi_srad","koi_teq", "koi_insol", "rocky", "habitable"]]
y_test = planets_test["confirmed"]
print(X_train.shape)
X_train.head()


# In[26]:


planets.head()


# Feature SCALING

# In[27]:


import pandas as pd


cols_to_scale = [
    'koi_prad', 'koi_steff', 'koi_slogg', 'koi_srad',
    'koi_teq', 'koi_insol'
]

# Compute mean and std for each feature (using training data only if you split)
means = X_train[cols_to_scale].mean()
stds  = X_train[cols_to_scale].std()

# Apply z-score scaling
planets_scaled = X_train.copy()
planets_scaled[cols_to_scale] = (X_train[cols_to_scale] - means) / stds
X_train = planets_scaled.copy()



# In[102]:


import pandas as pd


cols_to_scale = [
    'koi_prad', 'koi_steff', 'koi_slogg', 'koi_srad',
    'koi_teq', 'koi_insol'
]


means = X_test[cols_to_scale].mean()
stds  = X_test[cols_to_scale].std()

# Apply z-score scaling
planets_scaled = X_test.copy()
planets_scaled[cols_to_scale] = (X_test[cols_to_scale] - means) / stds
X_test = planets_scaled.copy()



# In[28]:


X_train.drop(columns='box', inplace =True)
X_train.drop(columns='scale', inplace=True)


# In[ ]:





# In[29]:


print(X_train)


# In[30]:


slogg_mean = np.mean(X_train['koi_slogg'])
teq_mean = np.mean(X_train['koi_teq'])


# In[31]:


X_train.hist(figsize=(20,16), bins=50)
print(X_train.shape)


# ______________________MODEL TRAINING_________________

# In[68]:


def sigmoid(z):
 
    return 1 / (1 + np.exp(-z))
    


# In[69]:


def compute_cost(X, y, w, b, *argv):
    m, n = X.shape
    cost = 0.0
   

    
    z = np.dot(X, w) + b
    sig = sigmoid(z)

    cost = np.sum(-y *np.log(sig) - (1 - y)*np.log(1 - sig))

    total_cost = cost / m
    
    return total_cost
    


# In[ ]:





# In[70]:


m,n=X_train.shape
print(m,n)


# In[71]:


def compute_gradient(X, y, w, b, *argv): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    z_wb = np.dot(X,w)+b
       
    f_wb = sigmoid(z_wb)

    
    
       
        
    dj_db = 1/m*(np.sum(f_wb - y))
       
    dj_dw =  1/m*(X.T@(f_wb-y))
            
    
    
   

        
    return dj_db, dj_dw


# In[72]:


initial_w = np.zeros(n)
initial_b = 0.


dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
print(f'dj_db at initial w and b (zeros):{dj_db}' )
print(f'dj_dw at initial w and b (zeros):{dj_dw.tolist()}' )


# In[73]:


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
  
    
    # number of training examples
    m = len(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history #return w and J,w history for graphing


# In[224]:


np.random.seed(1)
initial_w = np.array([0.1,0.0001,0.1,0.01,0.1,0.0001,1,1])
initial_b = 5

# Some gradient descent settings
iterations = 10000
alpha = 0.001

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, 
                                   compute_cost, compute_gradient, alpha, iterations, 0)
print("Learned w:", w)
print("Learned b:", b)


# In[62]:


print(planets.shape[1])
print(X_train.shape)


# In[189]:


# UNQ_C4
# GRADED FUNCTION: predict

def predict(X, w, b): 
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w
    
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    # number of training examples
    m, n = X.shape   
    p = np.zeros(m)
    f_wb=np.zeros(m)
    z_wb=np.zeros(m)   
    ### START CODE HERE ### 
    # Loop over each example
    for i in range(m):   
        z_wb[i] = np.dot(X.iloc[i],w)+b
        
        # Add bias term 
        
        
        # Calculate the prediction for this example
        f_wb[i] = sigmoid(z_wb[i])

        # Apply the threshold
        if f_wb[i]>=0.35:
            p[i]=1
        else:
            p[i]=0
        
        
    ### END CODE HERE ### 
    return p


# In[200]:


# Test your predict code

w = [-0.000452, -0.272556, 0.187035, -0.024772, -0.080476, -0.002733, -0.242671, 0.836049]
b =  -0.752340954157874
tmp_p=predict(X_test, w, b)
print(f'Output of predict: shape {y_test[0:50]}, value {tmp_p[0:50]}')

# UNIT TESTS        


# In[99]:


X_train.hist()


# In[110]:


print(y_test[0:50])


# In[148]:


X_test.hist(figsize=(20,16), bins=50)


# In[216]:


def predict(X, w, b, threshold=0.2):
    """
    Predict binary labels (0/1) for input features X using learned weights and bias.
    """
    z = np.dot(X, w) + b
    y_pred = 1 / (1 + np.exp(-z))   # sigmoid
    return (y_pred >= threshold).astype(int)


def compute_accuracy(y_true, y_pred):
    """
    Compute accuracy as % of correctly predicted samples.
    """
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)


# In[ ]:





# In[218]:


w = [-0.000452, -0.272556, 0.187035, -0.024772, -0.080476, -0.002733, -0.242671, 0.836049]
b =  -0.1
y_pred = predict(X_test, w, b)
acc = compute_accuracy(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%")


# In[219]:


import numpy as np

def precision_recall_f1(y_true, y_pred):
    """
    Computes Precision, Recall, and F1 Score for binary classification.
    
    Args:
        y_true : (ndarray, shape (m,)) true labels (0 or 1)
        y_pred : (ndarray, shape (m,)) predicted labels (0 or 1)
    
    Returns:
        precision : float
        recall : float
        f1 : float
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp + 1e-15)
    recall = tp / (tp + fn + 1e-15)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-15)

    return precision, recall, f1


# In[222]:


# Get predicted probabilities
w = [-0.000452, -0.272556, 0.187035, -0.024772, -0.080476, -0.002733, -0.242671, 0.836049]
b =  0
y_pred_probs = predict(X_test, w, b)


# Convert to binary predictions with threshold 0.5
y_pred = (y_pred_probs >= 0.2).astype(int)

# Evaluate
precision, recall, f1 = precision_recall_f1(y_test, y_pred)

print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")


# In[223]:


import numpy as np

def precision_recall_f1(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = np.sum((y_true==1) & (y_pred==1))
    fp = np.sum((y_true==0) & (y_pred==1))
    fn = np.sum((y_true==1) & (y_pred==0))
    precision = tp / (tp + fp + 1e-15)
    recall    = tp / (tp + fn + 1e-15)
    f1        = 2 * precision * recall / (precision + recall + 1e-15)
    return precision, recall, f1

# probabilities from your trained model
probs = sigmoid(np.dot(X_test, w) + b)

best = {"thr": None, "prec":0, "rec":0, "f1":-1}
for thr in np.linspace(0.05, 0.5, 20):  # sweep 0.05 → 0.50
    pred = (probs >= thr).astype(int)
    p,r,f1 = precision_recall_f1(y_test, pred)
    if f1 > best["f1"]:
        best = {"thr":thr, "prec":p, "rec":r, "f1":f1}

print(f"Best threshold = {best['thr']:.2f} | "
      f"Precision: {best['prec']*100:.2f}%  "
      f"Recall: {best['rec']*100:.2f}%  "
      f"F1: {best['f1']*100:.2f}%")


# In[ ]:





# In[ ]:




