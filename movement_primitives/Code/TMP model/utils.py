import numpy as np
import torch
import pandas as pd
import glob
import re
import os
import yaml
import math
from pathlib import Path
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import cv2
import scipy.io as sio



from plotting import *
from TMP_model import MP_model,TestTMPModel

H36M_KEYPOINT_NAMES = [
    'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Neck', 'Head',
    'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
]

# Define connections between joints for visualization (based on Human3.6M skeleton)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 4),       # Hip to RHip, Hip to LHip
    (1, 2), (2, 3),       # Right leg
    (4, 5), (5, 6),       # Left leg
    (0, 7), (7, 8),       # Spine to thorax
    (8, 9), (9, 10),      # Thorax to head
    (8, 11), (11, 12), (12, 13),  # Left arm
    (8, 14), (14, 15), (15, 16)   # Right arm
]


def visualize_frame_3d(df, frame_id=0):
    """
    Visualize 3D pose data for a specific frame from the DataFrame.
    
    Args:
        df: DataFrame containing the 3D pose data
        frame_id: The frame ID to visualize
    """
    # Filter data for the specific frame
    frame_data = df[df['frame_id'] == frame_id]
    
    # Create a new figure
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates for each joint
    joint_data = {}
    for _, row in frame_data.iterrows():
        joint_data[row['joint_name']] = (row['x_3d'], row['y_3d'], row['z_3d'])
    
    # Prepare coordinate arrays
    x = [joint_data[name][0] for name in H36M_KEYPOINT_NAMES]
    y = [joint_data[name][1] for name in H36M_KEYPOINT_NAMES]
    z = [joint_data[name][2] for name in H36M_KEYPOINT_NAMES]
    
    # Plot the joints
    ax.scatter(x, y, z, c='blue', marker='o', s=50)
    
    # Plot the skeleton connections
    for connection in SKELETON_CONNECTIONS:
        ax.plot([x[connection[0]], x[connection[1]]],
                [y[connection[0]], y[connection[1]]],
                [z[connection[0]], z[connection[1]]], 'r-', linewidth=2)
    
    # Add joint labels
    for i, (joint_x, joint_y, joint_z) in enumerate(zip(x, y, z)):
        ax.text(joint_x, joint_y, joint_z, H36M_KEYPOINT_NAMES[i], size=8)
    
    # Set axis properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Frame {frame_id}')
    
    # Set view limits based on data range with some padding
    x_range = max(x) - min(x)
    y_range = max(y) - min(y)
    z_range = max(z) - min(z)
    
    x_mid = (max(x) + min(x)) / 2
    y_mid = (max(y) + min(y)) / 2
    z_mid = (max(z) + min(z)) / 2
    
    max_range = max(x_range, y_range, z_range) * 0.6
    
    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)
    
    # Show the plot
    # plt.tight_layout()
    # plt.show()
    
    return fig



# def calculate_joint_angles(df_3d):
#     # Get unique frames
#     frames = df_3d['frame_id'].unique()
    
#     # Define joint hierarchy (parent-child relationships)
#     # For Human3.6M skeleton
#     parent_indices = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
#     H36M_KEYPOINT_NAMES = [
#     'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
#     'Spine', 'Thorax', 'Neck', 'Head',
#     'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
# ]
#     joint_angles = []
    
#     for frame_id in frames:
#         frame_data = df_3d[df_3d['frame_id'] == frame_id]
        
#         # Extract joint positions for this frame
#         positions = np.zeros((len(H36M_KEYPOINT_NAMES), 3))
#         for i, joint_name in enumerate(H36M_KEYPOINT_NAMES):
#             joint_row = frame_data[frame_data['joint_name'] == joint_name].iloc[0]
#             positions[i] = [joint_row['x_3d'], joint_row['y_3d'], joint_row['z_3d']]
        
#         # Calculate rotation for each joint
#         rotations = []
        
#         for joint_idx in range(len(H36M_KEYPOINT_NAMES)):
#             if parent_indices[joint_idx] == -1:
#                 # Root joint - store global position and orientation
#                 rotations.append(np.eye(3))  # Identity rotation for root
#                 continue
            
#             parent_idx = parent_indices[joint_idx]
            
#             # Calculate bone direction vectors
#             child_pos = positions[joint_idx]
#             parent_pos = positions[parent_idx]
#             bone_vec = child_pos - parent_pos
            
#             if np.linalg.norm(bone_vec) < 1e-10:
#                 # Bones with zero length - use identity
#                 rotations.append(np.eye(3))
#                 continue
                
#             # Normalize bone vector
#             bone_vec = bone_vec / np.linalg.norm(bone_vec)
            
#             # Create local coordinate system
#             # Y axis along the bone
#             y_axis = bone_vec
            
#             # Find a perpendicular vector for X axis
#             # Usually choose a direction that aligns with anatomical planes
#             if abs(y_axis[1]) < 0.9:  # If Y isn't nearly parallel to world Y
#                 x_axis = np.cross(np.array([0, 1, 0]), y_axis)
#             else:
#                 x_axis = np.cross(np.array([1, 0, 0]), y_axis)
                
#             x_axis = x_axis / np.linalg.norm(x_axis)
            
#             # Z completes the right-handed coordinate system
#             z_axis = np.cross(x_axis, y_axis)
#             z_axis = z_axis / np.linalg.norm(z_axis)
            
#             # Create rotation matrix (local coordinate system)
#             rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
            
#             rotations.append(rot_matrix)
        
#         # Convert rotation matrices to Euler angles or quaternions
#         euler_angles = []
#         for rot_matrix in rotations:
#             r = Rotation.from_matrix(rot_matrix)
#             # Use 'xyz' for typical BVH format
#             euler = r.as_euler('xyz', degrees=True)
#             euler_angles.append(euler)
        
#         # Store root position and all joint rotations for this frame
#         frame_angles = {
#             'frame_id': frame_id,
#             'root_position': positions[0],  # Hip position
#             'joint_rotations': euler_angles
#         }
        
#         joint_angles.append(frame_angles)
    
#     return joint_angles


def create_h36m_bvh(df_3d, output_path, fps=120):
    """
    Create a BVH file from H36M 3D pose data with proper BVH format.
    
    Args:
        df_3d: DataFrame with 3D joint positions
        output_path: Path to save the BVH file
        fps: Frames per second (120 for your reference BVH)
    """
    # Define joint hierarchy (which joint is parent of which)
    h36m_parent_indices = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    
    # Use H36M joint names directly
    joint_names = H36M_KEYPOINT_NAMES.copy()
    
    # Get unique frames
    frames = sorted(df_3d['frame_id'].unique())
    
    # Write BVH file
    with open(output_path, 'w') as f:
        # Write HIERARCHY section
        f.write("HIERARCHY\n")
        f.write(f"ROOT {joint_names[0]}\n")  # Hip as root
        f.write("{\n")
        f.write("  OFFSET 0 0 0\n")  # Root starts at origin
        f.write("  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
        
        # Function to recursively write joint hierarchy
        def write_joint_hierarchy(joint_idx, level=1):
            indent = "  " * level
            
            # Find children of this joint
            children = [i for i, parent in enumerate(h36m_parent_indices) if parent == joint_idx]
            
            for child_idx in children:
                # Get positions from first frame to calculate offset
                first_frame = df_3d[df_3d['frame_id'] == frames[0]]
                parent_pos = first_frame[first_frame['joint_name'] == joint_names[joint_idx]][['x_3d', 'y_3d', 'z_3d']].values[0]
                child_pos = first_frame[first_frame['joint_name'] == joint_names[child_idx]][['x_3d', 'y_3d', 'z_3d']].values[0]
                
                # Calculate offset (scale for better visibility)
                offset = (child_pos - parent_pos) * 100  # Scale to centimeters
                
                # Write joint definition
                f.write(f"{indent}JOINT {joint_names[child_idx]}\n")
                f.write(f"{indent}{{\n")
                f.write(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
                f.write(f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation\n")
                
                # Recursively process children
                write_joint_hierarchy(child_idx, level + 1)
                
                f.write(f"{indent}}}\n")
            
            # Add End Site for terminal joints
            if not children:
                f.write(f"{indent}End Site\n")
                f.write(f"{indent}{{\n")
                f.write(f"{indent}  OFFSET 0 10 0\n")  # Default end site offset
                f.write(f"{indent}}}\n")
        
        # Write the full hierarchy
        write_joint_hierarchy(0)  # Start with Hip (index 0)
        f.write("}\n\n")
        
        # Write MOTION section
        f.write("MOTION\n")
        f.write(f"Frames: {len(frames)}\n")
        f.write(f"Frame Time: {1.0/fps}\n")
        
        # Process each frame
        for frame_id in frames:
            frame_data = df_3d[df_3d['frame_id'] == frame_id]
            
            # Get positions for all joints
            positions = {}
            for idx, joint_name in enumerate(joint_names):
                joint_row = frame_data[frame_data['joint_name'] == joint_name]
                if not joint_row.empty:
                    positions[idx] = joint_row[['x_3d', 'y_3d', 'z_3d']].values[0]
            
            # Calculate rotations for all joints
            rotations = calculate_joint_rotations(positions, h36m_parent_indices)
            
            # Format motion data line
            motion_line = []
            
            # Root position (scale to match your reference)
            root_pos = positions[0] * 100  # Scale to centimeters
            motion_line.extend([root_pos[0], root_pos[1], root_pos[2]])
            
            # Add rotations for all joints in the correct order
            for joint_idx in range(len(joint_names)):
                if joint_idx in rotations:
                    # Get Euler angles in ZXY order (match your BVH format)
                    euler_angles = rotations[joint_idx].as_euler('zxy', degrees=True)
                    motion_line.extend(euler_angles)
                else:
                    # Default rotation for missing joint
                    motion_line.extend([0.0, 0.0, 0.0])
            
            # Write the frame data
            f.write(' '.join(f"{val:.6f}" for val in motion_line) + '\n')
    
    print(f"BVH file created at {output_path}")

def calculate_joint_rotations(positions, parent_indices):
    """Calculate joint rotations from positions"""
    rotations = {}
    
    for joint_idx in range(len(positions)):
        if parent_indices[joint_idx] == -1:
            # Root joint - use identity rotation
            rotations[joint_idx] = Rotation.from_euler('zxy', [0, 0, 0], degrees=True)
            continue
        
        parent_idx = parent_indices[joint_idx]
        
        # Calculate bone vector
        child_pos = positions[joint_idx]
        parent_pos = positions[parent_idx]
        bone_vec = child_pos - parent_pos
        
        # Skip if bone length is negligible
        if np.linalg.norm(bone_vec) < 1e-6:
            rotations[joint_idx] = Rotation.from_euler('zxy', [0, 0, 0], degrees=True)
            continue
        
        # Normalize bone vector
        bone_vec = bone_vec / np.linalg.norm(bone_vec)
        
        # Create local coordinate system
        # Y axis along bone direction
        y_axis = bone_vec
        
        # Create X axis perpendicular to Y and global Z
        z_global = np.array([0, 0, 1])
        x_axis = np.cross(y_axis, z_global)
        
        if np.linalg.norm(x_axis) < 1e-6:
            # If Y is parallel to Z, use another reference vector
            x_axis = np.cross(y_axis, np.array([1, 0, 0]))
        
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Z completes right-handed system
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # Create rotation matrix and convert to scipy rotation
        rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
        rotations[joint_idx] = Rotation.from_matrix(rot_matrix)
    
    return rotations


def parse_bvh_file(bvh_file_path):
    """Parse a BVH file to understand its structure"""
    with open(bvh_file_path, 'r') as f:
        content = f.read()
    
    print("BVH File Structure Analysis:")
    
    hierarchy_match = re.search(r'HIERARCHY(.*?)MOTION', content, re.DOTALL)
    if hierarchy_match:
        hierarchy = hierarchy_match.group(1).strip()
        print("\nHIERARCHY Section Sample:")
        print(hierarchy[:500] + "...")  # Print first 500 chars
    
    channels = re.findall(r'CHANNELS\s+(\d+)\s+(.*?)(?:\n|$)', content)
    print("\nChannel Definitions:")
    for num, ch_def in channels[:5]:  # Show first 5 channel definitions
        print(f"{num} channels: {ch_def}")
    
    # Extract motion data format
    motion_match = re.search(r'MOTION\s+Frames:\s+(\d+)\s+Frame Time:\s+([\d\.]+)(.*)', content, re.DOTALL)
    if motion_match:
        frames = int(motion_match.group(1))
        frame_time = float(motion_match.group(2))
        motion_data = motion_match.group(3).strip()
        
        print(f"\nMOTION Section: {frames} frames, {frame_time} sec/frame")
        first_frame = motion_data.split('\n')[0]
        print(f"First frame data: {first_frame[:100]}...")
        
        values_per_frame = len(first_frame.split())
        print(f"Values per frame: {values_per_frame}")
    
    return {
        'frames': frames,
        'frame_time': frame_time,
        'values_per_frame': values_per_frame,
        'channels': channels
    }



# Global dictionary to persist motion mappings across function calls
MOTION_MAPPING = {}
MOTION_ID_COUNTER = 0
MAPPING_FILE = "../../../data/motion_mapping.json"

def load_existing_mapping():
    """Load existing motion mapping from file if it exists"""
    global MOTION_MAPPING, MOTION_ID_COUNTER
    try:
        if os.path.exists(MAPPING_FILE):
            with open(MAPPING_FILE, 'r') as f:
                data = json.load(f)
                MOTION_MAPPING = data["mapping"]
                MOTION_ID_COUNTER = data["counter"]
                print(f"Loaded existing motion mapping with {len(MOTION_MAPPING)} entries.")
    except Exception as e:
        print(f"Error loading mapping file: {e}")
        MOTION_MAPPING = {}
        MOTION_ID_COUNTER = 0

def save_mapping():
    """Save the current motion mapping to file"""
    try:
        with open(MAPPING_FILE, 'w') as f:
            json.dump({
                "mapping": MOTION_MAPPING,
                "counter": MOTION_ID_COUNTER
            }, f, indent=2)
        print(f"Saved motion mapping with {len(MOTION_MAPPING)} entries.")
    except Exception as e:
        print(f"Error saving mapping file: {e}")

def create_motion_mapping(motions_list):
    """
    Create a mapping from motion sequence numbers to standardized motion IDs.
    If a motion is new, it gets a new ID. If it's already known, it keeps its ID.
    
    Parameters:
        motions_list: List of motion names from the .mat file
        
    Returns:
        Dictionary mapping sequence numbers to motion IDs
    """
    global MOTION_MAPPING, MOTION_ID_COUNTER
    
    # Load existing mapping if it's the first call
    if not MOTION_MAPPING:
        load_existing_mapping()
    
    # Create a mapping for this specific file
    sequence_to_id = {}
    print("Available motions in this file:")
    
    for i, motion in enumerate(motions_list):
        motion_name = str(motion).lower().strip()
        # print(f"  {i+1}: {motion_name}")
        
        # Check if this motion is already in our mapping
        if motion_name in MOTION_MAPPING:
            motion_id = MOTION_MAPPING[motion_name]
        else:
            # New motion, assign a new ID
            motion_id = MOTION_ID_COUNTER
            MOTION_MAPPING[motion_name] = motion_id
            MOTION_ID_COUNTER += 1
            print(f"  Added new motion: '{motion_name}' with ID {motion_id}")
        
        sequence_to_id[i+1] = motion_id
    
    # Save the updated mapping
    save_mapping()
   
    return sequence_to_id

def single_videos(main_video, main_rub):
    """
    Extract the single video motions using "flags" field from
    the corresponding rub file and save them in a folder named after the original video.
    Files are named using subject ID and a consistent motion ID.
    
    Parameters:
        main_video (str): The full name of the main video file which contains all motions
        main_rub (dict): Corresponding rub style struct (loaded from .mat file)
    """
    
    move_num = 1
    video_obj = cv2.VideoCapture(main_video)
    total_frames = int(video_obj.get(cv2.CAP_PROP_FRAME_COUNT))
    
    filepath, filename = os.path.split(main_video)
    name, ext = os.path.splitext(filename)
    
    # Extract subject ID from the filename using regex
    subject_id_match = re.search(r'Subject_(\d+)', filename)
    if subject_id_match:
        subject_id = subject_id_match.group(1)
    else:
        # If no match, use the full filename
        subject_id = name
    
    # Create a directory with the same name as the original video
    output_dir = os.path.join(filepath, name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Based on the output, 'Subject_20_F' is a key in main_rub
    subject_key = list(filter(lambda k: k not in ['__header__', '__version__', '__globals__'], main_rub.keys()))[0]
    subject_struct = main_rub[subject_key]
    
    
    try:
        main_move = subject_struct.move
        
        if hasattr(main_move, 'flags30'):
            flags30 = main_move.flags30
            # print("Found flags30 directly in move")
        else:
            # The move might be a struct with numbered fields
            for i in range(1, 10):  # Try reasonable field names
                field_name = str(i)
                if hasattr(main_move, field_name):
                    sub_move = getattr(main_move, field_name)
                    if hasattr(sub_move, 'flags30'):
                        flags30 = sub_move.flags30
                        # print(f"Found flags30 in move.{field_name}")
                        break
            else:
                raise KeyError("Could not find flags30 field")
        
        # Try to get the motions_list from the same place as flags30
        if hasattr(main_move, 'motions_list'):
            motions_list = main_move.motions_list
            num_motions = len(motions_list)
            
            motion_mapping = create_motion_mapping(motions_list)
                
        else:
            # If there's no motions_list, just use the number of segments in flags30
            num_motions = len(flags30)
            
            # Create a simple mapping without motion names
            # We'll just use sequential IDs starting from the current counter
            sequence_to_id = {}
            for i in range(num_motions):
                sequence_to_id[i+1] = MOTION_ID_COUNTER + i
            
            # Update the counter for next time
            MOTION_ID_COUNTER += num_motions
            
            motion_mapping = sequence_to_id
        
    except (AttributeError, KeyError) as e:
        print(f"Error accessing move structure: {e}")
        raise
    
    # print(f"Found {num_motions} motion segments")
    
    # Reading the frames one by one and saving the videos
    counter = 1
    frame_idx = 0
    
    while frame_idx < total_frames:
        ret, frame = video_obj.read()
        if not ret:
            break
            
        frame_idx += 1  # Python uses 0-indexing but we need 1-indexing to match MATLAB
        
        # Check if current frame is within a motion segment
        if counter <= num_motions:
            start_frame = flags30[counter-1][0] if isinstance(flags30[counter-1], np.ndarray) else flags30[counter-1, 0]
            end_frame = flags30[counter-1][1] if isinstance(flags30[counter-1], np.ndarray) else flags30[counter-1, 1]
            
            if frame_idx >= start_frame and frame_idx <= end_frame:
                if frame_idx == start_frame:
                    # Get the motion ID from our mapping
                    motion_id = motion_mapping.get(counter, counter-1)
                    
                    # Format: subject_id_motion_id.avi
                    output_videofilename = os.path.join(output_dir, f"subject_{subject_id}_motion_{motion_id:02d}.avi")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    fps = 30
                    frame_size = (frame.shape[1], frame.shape[0])
                    output_video = cv2.VideoWriter(output_videofilename, fourcc, fps, frame_size)
                    # print(f"Creating video: {output_videofilename}")
                
                output_video.write(frame)
                
                if frame_idx == end_frame:
                    motion_id = motion_mapping.get(counter, counter-1)
                    counter += 1
                    output_video.release()
                    # print(f"Finished writing motion ID {motion_id:02d} for subject {subject_id}")
                    if counter > num_motions:
                        break
    
    video_obj.release()
    cv2.destroyAllWindows()
    print(f"Video processing complete. All segments saved to {output_dir}")