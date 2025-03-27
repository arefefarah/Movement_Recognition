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
    
    
    
def plot_reconstructions(orig,recon,title=""):
    
    plt.clf()
    for i in range(6):
        plt.subplot(2,3,i+1)
        if i==0:
            plt.plot(orig[i],linewidth=2,linestyle="dotted",label="data")
            plt.plot(recon[i],linewidth=1,label="model")
            plt.legend()
        else:
            plt.plot(orig[i],linewidth=2,linestyle="dotted")
            plt.plot(recon[i],linewidth=1)
            
    
    plt.suptitle("Reconstructions: "+title)
    plt.savefig("recon_"+title+".png")
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
    
    # Ensure all segments have same number of sensors
    # min_sensors = min(seg.shape[0] for seg in processed_segments)
    # normalized_segments = [seg[:min_sensors, :] for seg in processed_segments]

    return processed_segments
    # return normalized_segments

