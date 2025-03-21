import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class SimpleTMP:
    """
    A simplified Temporal Movement Primitive model
    """
    
    def __init__(self, num_t_points, num_MPs, kernel_width=10.0, kernel_var=0.16, noise_level=0.03):
        """
        Initialize a simple TMP model.
        
        Parameters:
        -----------
        num_t_points : int
            Number of time discretization points
        num_MPs : int
            Number of movement primitives
        kernel_width : float
            Width of the RBF kernel
        kernel_var : float
            Variance of the RBF kernel
        noise_level : float
            Observation noise level
        """
        self.num_t_points = num_t_points
        self.num_MPs = num_MPs
        self.kernel_width = kernel_width
        self.kernel_var = kernel_var
        self.noise_level = noise_level
        
        # Create kernel matrix
        x = np.arange(num_t_points)
        self.K = self.kernel_matrix(x, x, kernel_var, kernel_width)
        self.invK = np.linalg.inv(self.K)
        
        # Dictionary for resampling matrices
        self.resampling_matrix = {}
        
        # Placeholder for MPs and weights
        self.MPs = None
        self.weights = None
    
    def kernel_matrix(self, x, y, variance, width):
        """Compute RBF kernel matrix between time points x and y"""
        if np.array_equal(x, y):
            return variance * np.exp(-0.5 * (np.subtract.outer(x, y))**2 / (width**2)) + 1e-6 * np.eye(len(x))
        else:
            return variance * np.exp(-0.5 * (np.subtract.outer(x, y))**2 / (width**2))
    
    def init_model_params(self, data):
        """
        Initialize model parameters from data using PCA
        
        Parameters:
        -----------
        data : list of arrays
            List of movement segments, each with shape [sensors, timesteps]
        """
        sensors = data[0].shape[0]
        segments = []
        
        # Resample all segments to the same length
        for segment in data:
            slen2 = segment.shape[1] // 2
            # Pad start and end to avoid resampling artifacts
            padded = np.hstack([
                np.ones((sensors, slen2)) * segment[:, 0:1],
                segment,
                np.ones((sensors, slen2)) * segment[:, -2:-1]
            ])
            # Resample and cut out the middle
            resampled = signal.resample(padded, self.num_t_points*2, axis=1)[:, self.num_t_points//2:3*self.num_t_points//2]
            segments.append(resampled)
        
        # Compute kernel variance from data
        concat_segments = np.concatenate(segments, axis=0)
        self.kernel_var = concat_segments.var(axis=1).mean()
        
        # SVD for initialization
        U, S, V = np.linalg.svd(concat_segments)
        
        # Initialize weights and MPs
        self.weights = []
        for trial in range(len(segments)):
            self.weights.append(U[trial*sensors:(trial+1)*sensors, :self.num_MPs])
        
        # Initialize MPs
        all_mps = np.dot(np.diag(S[:self.num_MPs]), V[:self.num_MPs])
        self.MPs = all_mps
    
    def predict(self, segment_lengths):
        """
        Predict data by multiplying weights with MPs and resampling
        
        Parameters:
        -----------
        segment_lengths : list of int
            Lengths of each segment to predict
            
        Returns:
        --------
        list of arrays
            Predicted segments
        """
        predictions = []
        
        for segidx, seg_len in enumerate(segment_lengths):
            predictions.append(self.predict_one_segment(seg_len, segidx))
            
        return predictions
    
    def predict_one_segment(self, seg_len, segidx):
        """
        Predict one segment with specified length
        
        Parameters:
        -----------
        seg_len : int
            Length of the segment to predict
        segidx : int
            Index of the segment (for weight selection)
            
        Returns:
        --------
        array
            Predicted segment
        """
        # Create resampling matrix if it doesn't exist
        if seg_len not in self.resampling_matrix:
            resample_t = np.arange(seg_len) * (self.num_t_points / seg_len)
            self.resampling_matrix[seg_len] = np.dot(
                self.kernel_matrix(resample_t, np.arange(self.num_t_points), self.kernel_var, self.kernel_width),
                self.invK
            )
        
        # Predict using weights and MPs
        pred_seg = np.tensordot(
            np.dot(self.weights[segidx], self.MPs), 
            self.resampling_matrix[seg_len], 
            axes=((1,), (1,))
        )
        
        return pred_seg
    
    def compute_VAF(self, data, predictions):
        """
        Compute Variance Accounted For (VAF)
        
        Parameters:
        -----------
        data : list of arrays
            Original data segments
        predictions : list of arrays
            Predicted data segments
            
        Returns:
        --------
        float
            VAF score (1.0 is perfect reconstruction)
        """
        all_errors = []
        variances = []
        
        for i, (original, prediction) in enumerate(zip(data, predictions)):
            error = original - prediction
            all_errors.append(error)
            variances.append(original.var(axis=1).mean())
        
        # Calculate squared error
        squared_error = np.hstack(all_errors)**2
        mean_variance = np.mean(variances)
        
        # VAF calculation
        vaf = 1.0 - squared_error.mean() / mean_variance
        
        return vaf
    
    def learn(self, data, max_iter=100):
        """
        Simple learning algorithm using alternating optimization
        
        Parameters:
        -----------
        data : list of arrays
            Training data segments
        max_iter : int
            Maximum number of iterations
            
        Returns:
        --------
        list
            VAF values during learning
        """
        # Initialize model if not already
        if self.MPs is None:
            self.init_model_params(data)
        
        segment_lengths = [segment.shape[1] for segment in data]
        vaf_history = []
        
        for iteration in range(max_iter):
            # Fix MPs, update weights
            for segidx, segment in enumerate(data):
                # Solve for optimal weights using least squares
                proj_MPs = np.zeros((self.num_MPs, segment.shape[1]))
                for mp_idx in range(self.num_MPs):
                    proj_MPs[mp_idx] = np.dot(
                        self.resampling_matrix.get(segment.shape[1], 
                            np.dot(
                                self.kernel_matrix(
                                    np.arange(segment.shape[1]) * (self.num_t_points / segment.shape[1]),
                                    np.arange(self.num_t_points),
                                    self.kernel_var, self.kernel_width
                                ),
                                self.invK
                            )
                        ),
                        self.MPs[mp_idx]
                    )
                
                # Update weights for this segment
                self.weights[segidx] = np.linalg.lstsq(proj_MPs.T, segment.T, rcond=None)[0].T
            
            # Fix weights, update MPs
            weighted_data = []
            for segidx, segment in enumerate(data):
                weighted_data.append(self.weights[segidx].T @ segment)
            
            # Stack weighted data
            stacked_weighted_data = np.hstack(weighted_data)
            
            # Create design matrix
            design_matrix = np.zeros((stacked_weighted_data.shape[1], self.num_t_points))
            col_idx = 0
            for segidx, segment in enumerate(data):
                seg_len = segment.shape[1]
                design_matrix[col_idx:col_idx+seg_len] = self.resampling_matrix.get(seg_len, 
                    np.dot(
                        self.kernel_matrix(
                            np.arange(seg_len) * (self.num_t_points / seg_len),
                            np.arange(self.num_t_points),
                            self.kernel_var, self.kernel_width
                        ),
                        self.invK
                    )
                )
                col_idx += seg_len
            
            # Update MPs using regularized least squares
            for mp_idx in range(self.num_MPs):
                self.MPs[mp_idx] = np.linalg.lstsq(
                    design_matrix.T @ design_matrix + 0.01 * np.eye(self.num_t_points),
                    design_matrix.T @ stacked_weighted_data[mp_idx],
                    rcond=None
                )[0]
            
            # Compute current predictions and VAF
            predictions = self.predict(segment_lengths)
            vaf = self.compute_VAF(data, predictions)
            vaf_history.append(vaf)
            
            print(f"Iteration {iteration+1}, VAF: {vaf:.4f}")
            
            # Check convergence
            if iteration > 1 and abs(vaf_history[-1] - vaf_history[-2]) < 0.0001:
                print("Converged!")
                break
                
        return vaf_history
    
    def plot_MPs(self):
        """Plot the movement primitives"""
        if self.MPs is None:
            print("No MPs to plot. Initialize or train the model first.")
            return
            
        plt.figure(figsize=(10, 6))
        for i in range(min(self.num_MPs, 5)):  # Plot up to 5 MPs
            plt.plot(self.MPs[i], label=f'MP {i+1}')
        
        plt.title('Movement Primitives')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_reconstruction(self, original, prediction, title="Data Reconstruction"):
        """Plot original vs reconstructed data"""
        plt.figure(figsize=(12, 8))
        n_signals = min(original.shape[0], 4)  # Plot up to 4 signals
        
        for i in range(n_signals):
            plt.subplot(n_signals, 1, i+1)
            plt.plot(original[i], 'b-', label='Original')
            plt.plot(prediction[i], 'r--', label='Reconstruction')
            plt.ylabel(f'Signal {i+1}')
            plt.grid(True)
            if i == 0:
                plt.title(title)
            if i == n_signals-1:
                plt.xlabel('Time')
            plt.legend()
        
        plt.tight_layout()
        plt.show()