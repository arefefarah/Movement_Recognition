#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 01:33:13 2019

@author: dom
"""

import matplotlib.pyplot as plt
import numpy
import torch
import numpy as np
from scipy import signal
import unittest
import os
from matplotlib.patches import Patch
from bvh import Bvh


###### Plotting code ######

def plot_eigenvalues(prec_mat,numw,title=""):            
    cv_mat=numpy.linalg.inv(prec_mat)
    ev=numpy.linalg.eigvals(cv_mat[:numw,:numw])
    ev.sort()
    ev=ev[::-1]
    
    plt.clf()
    plt.plot(ev,linewidth=2)
    plt.title("EVs weights, "+title)
    plt.savefig("ev_weights_{0:d}_".format(numw)+title+".png".format(numw))
    plt.show()

    ev=numpy.linalg.eigvals(cv_mat[numw:,numw:])
    ev.sort()
    ev=ev[::-1]
    plt.clf()
    plt.plot(ev,linewidth=2)
    plt.title("EVs MPs, "+title)
    plt.savefig("ev_MPs_{0:d}_".format(numw)+title+".png".format(numw))
    plt.show()
        
        
def plot_mp(MPs,title=""):
    
    MPs=MPs.detach().numpy()
    plt.clf()
    for i in range(MPs.shape[0]):
        plt.plot(MPs[i],linewidth=2,label="MP {0:d}".format(i))
        
    plt.suptitle("MPs: "+title)
    plt.xlabel("t")
    plt.ylabel("joint angles [rad]")
    plt.subplots_adjust(left=0.2)
    plt.savefig("MPs_"+title+".png")
    plt.show()
    
def plot_learn_curve(epochs,lc,vc,title=""):
    
    plt.clf()
    plt.plot(epochs,lc,linewidth=2)
    plt.xlabel("epochs")
    plt.ylabel("log(p(X,W,MP))")
    plt.title("Learning curve: "+title)
    plt.subplots_adjust(left=0.2)
    plt.savefig("learning_curve_"+title+".png")
    plt.show()
    
    plt.clf()
    plt.plot(epochs,vc,linewidth=2)
    plt.xlabel("epochs")
    plt.ylabel("VAF")
    plt.title("VAF: "+title)
    plt.subplots_adjust(left=0.2)
    plt.savefig("VAF_curve_"+title+".png")
    plt.show()
    
    

def plot_reconstructions(orig, recon, title=""):
    plt.clf()
    
    for i in range(6):
        plt.subplot(2, 3, i + 1) 
        if i == 0:
            plt.plot(orig[i], linewidth=2, linestyle="dotted", label="data")
            plt.plot(recon[i], linewidth=1, label="model")
            plt.legend()
        else:
            plt.plot(orig[i], linewidth=2, linestyle="dotted")
            plt.plot(recon[i], linewidth=1)
        
    plt.tight_layout() 
    plt.suptitle("Reconstructions: " + title, y=1.02) 
    plt.savefig("recon_" + title + ".png", bbox_inches='tight') 
    plt.show()

    
    
def plot_kernel(K):
    
    kvals=K[len(K)//2]
    idx=numpy.arange(len(K))-len(K)//2
    plt.clf()
    plt.plot(idx,kvals,linewidth=2)
    plt.xlabel("$\Delta t$")
    plt.ylabel("covariance")
    plt.title("Kernel function")
    plt.savefig("kernel.png")
    plt.show()
    
    plt.clf()
    plt.title("Kernel matrix")
    plt.imshow(K)
    plt.xlabel("$t$")
    plt.ylabel("$t^\prime$")
    plt.savefig("kernel_matrix.png")
    plt.show()
    
    
def plot_model_comparison(model_evidences,VAFs,ground_truth_num_MPs,title=""):
    plt.clf()
    
    plt.subplot(1,2,1)
    plt.bar(range(1,10),model_evidences)
    plt.ylabel("LAP")
    
    plt.subplot(1,2,2)
    plt.bar(range(1,10),VAFs)
    plt.ylabel("VAF")
    
    plt.suptitle("Model comparison (Ground truth: "+str(ground_truth_num_MPs)+" MPs)")
    plt.savefig("model_comparison.png")




def read_bvh_files(folder_path):
    bvh_data = []
    bvh_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.bvh')]
    
    for bvh_file in bvh_files:
        file_path = os.path.join(folder_path, bvh_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                mocap = Bvh(f.read())
                bvh_data.append(mocap)
                # print(f"Successfully opened {bvh_file}")
        except Exception as e:
            print(f"Error reading file {bvh_file}: {str(e)}")
    return bvh_data

def filter_motion_data(data, cutoff_freq=6, sampling_rate=120):
    """Apply 4th order Butterworth filter as specified in paper"""
    nyquist_freq = 0.5 * sampling_rate
    order = 4
    b, a = signal.butter(order, cutoff_freq/nyquist_freq)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def process_bvh_data(bvh_data, num_points=50):
    """Process BVH data according to paper specifications"""
    processed_segments = []
    
    for mocap in bvh_data:
        try:
            # Convert frames to numpy array
            frames = np.array(mocap.frames, dtype=np.float64)
            
            # Filter the data with 4th order Butterworth (6Hz cutoff)
            filtered_frames = filter_motion_data(frames)
            
            # Resample to fixed length (50 points) with padding
            # slen2 = filtered_frames.shape[1] // 2
            
            # # Pad start and end to avoid resampling artifacts
            # padded = np.hstack([
            #     np.ones((filtered_frames.shape[0], slen2)) * filtered_frames[:, 0:1],
            #     filtered_frames,
            #     np.ones((filtered_frames.shape[0], slen2)) * filtered_frames[:, -2:-1]
            # ])
            
            # # Resample and cut middle section as per paper
            # resampled = signal.resample(
            #     padded, 
            #     num_points*2, 
            #     axis=1
            # )[:, num_points//2:3*num_points//2]
            
            processed_segments.append(filtered_frames.T) # the format of each segment should be [signals,time]
            # print(f"Processed segment shape: {filtered_frames.T.shape}") 
            
        except Exception as e:
            print(f"Error processing mocap data: {str(e)}")
            continue
    
    if not processed_segments:
        raise ValueError("No segments could be processed")
    
    return processed_segments


def plot_weights_for_signal(model, signal_idx, title=None):
    """
    Plot weights of a specific signal for all MPs across all segments.
    
    Parameters:
    -----------
    model : MP_model
        The trained temporal movement primitive model
    signal_idx : int
        Index of the signal to plot (0-53)
    title : str, optional
        Title for the plot
    """
    # Get number of segments and MPs
    num_segments = len(model.weights)
    num_MPs = model.num_MPs
    
    # Create figure
    plt.figure(figsize=(8, 4))
    
    # For each MP, collect weights from all segments for the chosen signal
    for mp_idx in range(num_MPs):
        # Extract weights for this MP and signal across all segments
        weights = [model.weights[seg_idx][signal_idx, mp_idx].item() for seg_idx in range(num_segments)]
        
        # Create x positions for this MP (add small jitter to separate points)
        x_positions = np.random.normal(mp_idx + 1, 0.05, len(weights))
        
        # Plot points for this MP
        plt.scatter(x_positions, weights, label=f'MP {mp_idx+1}', alpha=0.7)
    
    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Set plot labels and title
    plt.xlabel('Movement Primitive (MP)')
    plt.ylabel(f'Weight for Signal {signal_idx+1}')
    if title:
        plt.title(title)
    else:
        plt.title(f'Weights for Signal(joint) {signal_idx+1} across all segments')
    
    # Set x-ticks to MP numbers
    plt.xticks(range(1, num_MPs + 1))
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Show the plot
    plt.tight_layout()
    plt.show()



def identify_signals(num_signals=54):
    """
    Create a mapping between signal indices and joint names with rotation/position axes
    based on Human3.6M keypoint structure.
    
    Parameters:
    -----------
    num_signals : int
        Total number of signals (default=54)
    
    Returns:
    --------
    signal_names : list
        List of signal names in format "JointName_Axis"
    signal_mapping : dict
        Dictionary mapping signal indices to signal names
    """
    # Human3.6M keypoint names
    H36M_KEYPOINT_NAMES = [
        'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
        'Spine', 'Thorax', 'Neck', 'Head',
        'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
    ]
    
    # Create signal names
    signal_names = []
    signal_mapping = {}
    
    signal_idx = 0
    
    # Hip has 6 signals: X/Y/Z position and Z/X/Y rotation
    hip_signals = ['Xposition', 'Yposition', 'Zposition', 'Zrotation', 'Xrotation', 'Yrotation']
    for signal in hip_signals:
        signal_name = f"Hip_{signal}"
        signal_names.append(signal_name)
        signal_mapping[signal_idx] = signal_name
        signal_idx += 1
    
    # All other joints have 3 rotation signals: Z/X/Y rotation
    rotation_signals = ['Zrotation', 'Xrotation', 'Yrotation']
    for joint in H36M_KEYPOINT_NAMES[1:]:  # Skip Hip as it's already processed
        for signal in rotation_signals:
            signal_name = f"{joint}_{signal}"
            signal_names.append(signal_name)
            signal_mapping[signal_idx] = signal_name
            signal_idx += 1
    
    # Verify we have the expected number of signals
    expected_signals = 6 + (len(H36M_KEYPOINT_NAMES) - 1) * 3  # 6 for Hip + 3 for each other joint
    
    if expected_signals != num_signals:
        print(f"Warning: Expected {expected_signals} signals based on H36M structure.")
        print(f"Actual number of signals: {num_signals}")
        
        # Fill any remaining indices if needed
        for i in range(signal_idx, num_signals):
            signal_name = f"Unknown_{i}"
            signal_names.append(signal_name)
            signal_mapping[i] = signal_name
    
    return signal_names, signal_mapping

def plot_weights_by_joint(model, motion_label):
    """
    Plot weight distribution for all signals grouped by joint, with one subplot per MP.
    
    Parameters:
    -----------
    model : MP_model
        The trained temporal movement primitive model
    title : str, optional
        Title for the plot
    """
    # Get dimensions
    num_segments = len(model.weights)
    num_MPs = model.num_MPs
    num_signals = model.weights[0].shape[0]
    
    # Get signal names and mapping
    signal_names, signal_mapping = identify_signals(num_signals)
   
    # Create figure with subplots
    fig, axes = plt.subplots(num_MPs, 1, figsize=(18, 4*num_MPs), sharex=True)
    if num_MPs == 1:
        axes = [axes]  # Make it iterable for a single subplot
    
    # Define colors for different signal types
    signal_colors = {
        'Xposition': 'tab:red',
        'Yposition': 'tab:green',
        'Zposition': 'tab:blue',
        'Zrotation': 'tab:purple',
        'Xrotation': 'tab:orange',
        'Yrotation': 'tab:brown'
    }
    
    # For each MP
    for mp_idx in range(num_MPs):
        ax = axes[mp_idx]
        
        # For each signal, collect weights from all segments
        for signal_idx in range(num_signals):
            signal_name = signal_mapping[signal_idx]
            signal_type = signal_name.split('_')[1]
            
            # Extract weights for this MP and signal across all segments
            weights = [model.weights[seg_idx][signal_idx, mp_idx].item() for seg_idx in range(num_segments)]
            
            # Create x positions for this signal
            x_positions = np.random.normal(signal_idx + 1, 0.1, len(weights))
            
            # Plot points for this signal
            ax.scatter(x_positions, weights, alpha=0.6, s=30, color=signal_colors.get(signal_type, 'gray'))
        
        # Add a horizontal line at y=0 for reference
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Set plot labels
        ax.set_ylabel(f'Weights for MP {mp_idx+1}')
        ax.set_title(f'Movement Primitive {mp_idx+1} Weight Distribution')
        
        ax.grid(True, alpha=0.3)
    
    plt.xlabel('Joint Signal')
    
    # Create custom x-tick labels
    tick_positions = list(range(1, num_signals + 1))
    tick_labels = [f"{signal_idx+1}: {name}" for signal_idx, name in signal_mapping.items()]
    
    # Use every few ticks to avoid overcrowding
    step = max(1, num_signals // 20)
    plt.xticks(tick_positions[::step], tick_labels[::step], rotation=45, ha='right', fontsize=8)
    
    # Add a legend for signal types
    legend_elements = [Patch(facecolor=color, label=signal_type) 
                      for signal_type, color in signal_colors.items()]
    fig.legend(handles=legend_elements, loc='upper right', title='Signal Types')
    
    fig.suptitle(f'Weight Distribution by Joint in {motion_label}', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.2)  # Adjust to make room for the suptitle and rotated labels
    plt.savefig(f'weights_{motion_label}.png')
    plt.show()
    

