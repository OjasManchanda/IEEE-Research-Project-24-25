import numpy as np
from sklearn.neighbors import NearestNeighbors
from numba import cuda
import numba

class MutualNNCalculator:
    def __init__(self, num_samples=1000, num_features=2, num_neighbors=5):
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_neighbors = num_neighbors
        self.dataset = None
        self.neighbor_indices = None
        self.neighbor_dict = None

    def generate_dataset(self):
        self.dataset = np.random.rand(self.num_samples, self.num_features)

    def compute_neighbors(self):
        knn_model = NearestNeighbors(n_neighbors=self.num_neighbors, algorithm='auto').fit(self.dataset)
        _, self.neighbor_indices = knn_model.kneighbors(self.dataset)
        self.neighbor_indices = np.ascontiguousarray(self.neighbor_indices[:, 1:])
        self.neighbor_dict = {i: set(self.neighbor_indices[i]) for i in range(self.num_samples)}

    @staticmethod
    @cuda.jit
    def mnn_kernel(neighbor_array, mnn_scores_gpu):
        thread_id = cuda.grid(1)
        if thread_id < neighbor_array.shape[0]:
            current_neighbors = neighbor_array[thread_id]
            mutual_score = 0

            for i in range(current_neighbors.shape[0]):
                neighbor_id = current_neighbors[i]
                neighbors_of_neighbor = neighbor_array[neighbor_id]

                for j in range(neighbors_of_neighbor.shape[0]):
                    if neighbors_of_neighbor[j] == thread_id:
                        mutual_score += 1
                        break

            mnn_scores_gpu[thread_id] = mutual_score

    def calculate_mnn_scores(self):
        neighbor_array_gpu = cuda.to_device(self.neighbor_indices)
        mnn_scores = np.zeros(self.num_samples, dtype=np.int32)
        mnn_scores_gpu = cuda.to_device(mnn_scores)

        threads_per_block = 256
        blocks_per_grid = (self.num_samples + threads_per_block - 1) // threads_per_block

        self.mnn_kernel[blocks_per_grid, threads_per_block](neighbor_array_gpu, mnn_scores_gpu)

        return mnn_scores_gpu.copy_to_host()

if __name__ == "__main__":
    mnn_calculator = MutualNNCalculator(num_samples=1000, num_features=2, num_neighbors=5)

    mnn_calculator.generate_dataset()

    mnn_calculator.compute_neighbors()

    mnn_scores = mnn_calculator.calculate_mnn_scores()

    print("MNN scores for all points:")
    for i, score in enumerate(mnn_scores):
        print(f"Point {i}: {score}")
