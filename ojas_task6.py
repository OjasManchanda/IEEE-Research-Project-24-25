import cupy as cp
import numpy as np 
from sklearn.neighbors import NearestNeighbors

def compute_mnn_score_gpu(data, index, n_neighbors=5):
   
    data_gpu = cp.asarray(data)
    
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    nn_model.fit(data)
   
    distances, indices = nn_model.kneighbors(data)
    
    indices_gpu = cp.asarray(indices)
    
    neighbors = indices_gpu[index]
    
    neighbors_set = set(neighbors.tolist())
   
    mnn_count = 0
    for neighbor in neighbors:
        
        neighbor_neighbors = set(indices_gpu[neighbor].tolist())
        if index in neighbor_neighbors:
            mnn_count += 1
    
    return mnn_count

def main():
    
    num_rows = 10
    num_cols = 2
    data = np.random.randint(0, 100, size=(num_rows, num_cols)) 
    
    print("Generated Dataset:")
    print(data)
    
    input_index = int(input("\nEnter the index to calculate the MNN score: "))

    if 0 <= input_index < num_rows:
        mnn_score = compute_mnn_score_gpu(data, input_index, n_neighbors=5)
        print(f"\nMNN score for index {input_index}: {mnn_score}")
    else:
        print("Invalid index, Please enter a value within the dataset range.")

main()