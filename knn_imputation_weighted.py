from pandas import read_csv
import numpy as np
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.impute import KNNImputer

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
dataframe = read_csv(url, header=None, na_values='?')

data = dataframe.values

def nan_distance(x,y):
    col = len(x)
    dist = 0
    # number of present coordinates
    pres_cand =  len(x)
     # number of total coordinates
    tot_cand = len(x)
    for i in range(col):
        if(np.isnan(x[i]) or np.isnan(y[i])):
            pres_cand = pres_cand-1
            # if nan is there so we will decrease our present coordinates count
            continue
        dist = dist + (x[i]-y[i])**2
    ## formula
    dist = dist*tot_cand/pres_cand
    return dist**0.5

def nan_matrix(data):
    n_row = data.shape[0]
    dist_matrix = np.zeros((n_row,n_row))
    for i in range(n_row):
        for j in range(n_row):
            if(j==i):
                continue
            elif(j<i):
                dist_matrix[i,j] = dist_matrix[j,i]
            else:
                dist_matrix[i,j] = nan_distance(data[i],data[j])
    return(dist_matrix)
            

def knn_imputation(data,n_neighbors = 2):
    count = 0
    ## Where are the null values in the data set
    indexes_nan = np.argwhere(np.isnan(data))
    n_col = data.shape[1]
    n_row = data.shape[0]
    final_data = data.copy()
    for index in indexes_nan:
        print(count)
        count = count + 1 
        distances = []
        row = index[0]
        col = index[1]
        x = data[row]
        for i in range(n_row):
            if(i==row):
                continue
            dist = nan_euclidean_distances(x.reshape(1,-1),data[i].reshape(1,-1))[0][0]
            ## or u can also use the above distance function, both produce same answer
            ## dist = nan_distance(x,data[i])
            distances.append([dist, data[i][col]])
        ## key=lambda x: x[0]  here x:x[0] signifies that we will sort this list using 
        ## 1st(in python indexing starts from 0) column i.e. distance
        distances.sort(key=lambda x: x[0])
         ## Considering k closest points
        distances = np.array(distances)
        neighbors = []
        for i in range(len(distances)):
            if(np.isnan(distances[i,1])):
                continue
            neighbors.append(distances[i,1])
            if(len(neighbors)==n_neighbors):  
                break
        ## replacing the nan with the mean of the closest points
        adjusted_value = np.mean(neighbors)
        final_data[row][col] = adjusted_value
    
    return(final_data)
    

def optimized(data,n_neighbors = 2):
    count = 0
    ## Where are the null values in the data set
    indexes_nan = np.argwhere(np.isnan(data))
    final_data = data.copy()
    ## it will produce a distance matrix with row in the dataset
    dist = nan_matrix(data)
    for index in indexes_nan:
        print(count)
        count = count + 1 
        row = index[0]
        col = index[1]
        ## Sorting the distance matrix for the particular row and getting top n values
    sort_indxs = dist[row].argsort()
    close_val = []
    total_dist = 0
    for i in sort_indxs:
        if(np.isnan(data[i,col])):
            continue
        inv_dist = 1/dist[row,i]
        total_dist = total_dist + inv_dist
        close_val.append(data[i,col]*inv_dist)
        if(len(close_val)==n_neighbors):
            break
    adjusted_value = np.sum(close_val)/total_dist
    final_data[row][col] = adjusted_value
    return(final_data)
    
final = optimized(data,5)
f = knn_imputation(data,5)
imputer = KNNImputer(n_neighbors=5)
sklearn_out = imputer.fit_transform(data)