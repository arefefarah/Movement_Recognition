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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from matplotlib.gridspec import GridSpec



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

import numpy as np
from scipy.spatial.transform import Rotation

def create_h36m_bvh(df_3d, output_path, fps=120):
    """
    Create a BVH file from H36M 3D pose data with proper BVH format.
    
    Args:
        df_3d: DataFrame with 3D joint positions
        output_path: Path to save the BVH file
        fps: Frames per second
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
        f.write("  OFFSET 0 100 0\n")  # Root starts elevated like in the working example
        # Change position order to match working example
        f.write("  CHANNELS 6 Xposition Zposition Yposition Zrotation Xrotation Yrotation\n")
        
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
                # Swap y and z coordinates to match BVH convention
                f.write(f"{indent}  OFFSET {offset[0]:.6f} {offset[2]:.6f} {offset[1]:.6f}\n")
                f.write(f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation\n")
                
                # Recursively process children
                write_joint_hierarchy(child_idx, level + 1)
                
                f.write(f"{indent}}}\n")
            
            # Add End Site for terminal joints
            if not children:
                f.write(f"{indent}End Site\n")
                f.write(f"{indent}{{\n")
                # Match the end site in working BVH
                f.write(f"{indent}  OFFSET 0 -2 15\n")
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
            rotations = calculate_joint_rotations_fixed(positions, h36m_parent_indices)
            
            # Format motion data line
            motion_line = []
            
            # Root position (scale to match your reference)
            root_pos = positions[0] * 100  # Scale to centimeters
            
            # Apply the Y-Z swap to match working example
            motion_line.extend([root_pos[0], root_pos[2], root_pos[1]])
            
            # Add rotations for all joints in the correct order
            for joint_idx in range(len(joint_names)):
                if joint_idx in rotations:
                    # Calculate Euler angles in ZXY order
                    euler_angles = rotations[joint_idx].as_euler('zxy', degrees=True)
                    
                    # Root rotation correction (match working example)
                    if joint_idx == 0:
                        # Rotate 180 degrees around Z-axis to match working BVH
                        euler_angles[0] += 180.0
                        # Ensure angles are in -180 to 180 range
                        if euler_angles[0] > 180:
                            euler_angles[0] -= 360
                        
                    motion_line.extend(euler_angles)
                else:
                    # Default rotation for missing joint
                    motion_line.extend([0.0, 0.0, 0.0])
            
            # Write the frame data
            f.write(' '.join(f"{val:.6f}" for val in motion_line) + '\n')
    
    print(f"BVH file created at {output_path}")

def calculate_joint_rotations_fixed(positions, parent_indices):
    """Calculate joint rotations from positions with fixed coordinate system"""
    rotations = {}
    
    # Define the BVH coordinate system convention
    FORWARD_AXIS = np.array([0, 0, 1])  # Z-forward 
    UP_AXIS = np.array([0, 1, 0])       # Y-up
    
    # First, calculate orientation for root
    if 0 in positions:
        # Find spine index which is typically at index 7 in H36M
        spine_idx = 7  # Usually this is the spine in H36M
        
        if spine_idx in positions:
            # Use spine direction to determine orientation
            spine_dir = positions[spine_idx] - positions[0]
            spine_dir = spine_dir / np.linalg.norm(spine_dir)
            
            # Set up initial coordinate system for root
            # We want spine_dir to align with Y-axis in BVH
            y_axis = spine_dir
            
            # Find hip joints for left-right axis
            left_hip_idx = 4  # LHip in H36M
            right_hip_idx = 1  # RHip in H36M
            
            if left_hip_idx in positions and right_hip_idx in positions:
                # Use hips to define left-right axis
                left_hip = positions[left_hip_idx]
                right_hip = positions[right_hip_idx]
                side_dir = right_hip - left_hip
                side_dir = side_dir / np.linalg.norm(side_dir)
                
                # Make sure side_dir is perpendicular to spine_dir
                side_dir = side_dir - np.dot(side_dir, y_axis) * y_axis
                side_dir = side_dir / np.linalg.norm(side_dir)
                
                x_axis = side_dir
            else:
                # Fallback if hips not available
                x_axis = np.cross(FORWARD_AXIS, y_axis)
                if np.linalg.norm(x_axis) < 1e-6:
                    x_axis = np.cross(UP_AXIS, y_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
            
            # Complete the coordinate system
            z_axis = np.cross(x_axis, y_axis)
            z_axis = z_axis / np.linalg.norm(z_axis)
            
            # Create rotation matrix for root
            root_matrix = np.column_stack((x_axis, y_axis, z_axis))
            
            # Apply rotation to transform coordinate system to match BVH viewer's expectation
            # This is the key correction to fix the top-down view issue
            fix_rotation = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_matrix()
            adjusted_matrix = np.dot(fix_rotation, root_matrix)
            
            rotations[0] = Rotation.from_matrix(adjusted_matrix)
        else:
            # Fallback if no spine found
            rotations[0] = Rotation.from_euler('xyz', [90, 0, 0], degrees=True)
    
    # Second pass: compute joint orientations based on bone directions
    for joint_idx in sorted(positions.keys()):
        if joint_idx == 0 or parent_indices[joint_idx] == -1:
            # Root already handled
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
        
        # In BVH convention, bones typically point along Y axis
        parent_y = np.array([0, 1, 0])  # Y axis in local space
        
        # Calculate rotation that aligns parent_y with bone_vec
        if parent_idx in rotations:
            # Get parent's global orientation
            parent_matrix = rotations[parent_idx].as_matrix()
            
            # Transform parent_y to global space
            parent_y_global = parent_matrix @ parent_y
            
            # Calculate rotation from parent_y_global to bone_vec
            rot_axis = np.cross(parent_y_global, bone_vec)
            
            if np.linalg.norm(rot_axis) < 1e-6:
                # Vectors are parallel
                dot_product = np.dot(parent_y_global, bone_vec)
                if dot_product > 0:
                    # Same direction, no rotation
                    local_rotation = Rotation.from_euler('zxy', [0, 0, 0], degrees=True)
                else:
                    # Opposite direction, 180° rotation around X axis
                    local_rotation = Rotation.from_euler('x', 180, degrees=True)
            else:
                # Normalize rotation axis
                rot_axis = rot_axis / np.linalg.norm(rot_axis)
                
                # Calculate rotation angle
                cos_angle = np.clip(np.dot(parent_y_global, bone_vec), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                # Create rotation from axis and angle
                local_rotation = Rotation.from_rotvec(rot_axis * angle)
            
            # Combine parent rotation with local rotation
            rotations[joint_idx] = rotations[parent_idx] * local_rotation
        else:
            # No parent rotation, use default
            rotations[joint_idx] = Rotation.from_euler('zxy', [0, 0, 0], degrees=True)
    
    # Convert global rotations to local rotations
    local_rotations = {}
    for joint_idx in rotations:
        if parent_indices[joint_idx] == -1 or parent_indices[joint_idx] not in rotations:
            # Root or joint with no valid parent rotation
            local_rotations[joint_idx] = rotations[joint_idx]
        else:
            # Calculate local rotation relative to parent
            parent_rot = rotations[parent_indices[joint_idx]]
            local_rot = parent_rot.inv() * rotations[joint_idx]
            local_rotations[joint_idx] = local_rot
    
    return local_rotations




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


def create_csv_from_json(source_folder, destination_folder, subjects, equivalent_motions, id_to_motion):
    """
    Function to create CSV files from JSON motion files
    
    Args:
        source_folder: Directory containing subject folders with JSON files
        destination_folder: Directory to save individual CSV files
        subjects: List of subject IDs to process
        equivalent_motions: Dictionary mapping motion IDs to canonical IDs
        id_to_motion: Dictionary mapping motion IDs to motion names
    
    Returns:
        dict: Dictionary mapping subject IDs to their common motions
        list: List of motion IDs common to all subjects
    """
    os.makedirs(destination_folder, exist_ok=True)
    
    # Find subjects and their available motions
    subject_motions = {}

    for subject_id in subjects:
        subject_folder = os.path.join(source_folder, f"pred_Subject_{subject_id}")
        if os.path.exists(subject_folder):
            json_files = glob.glob(os.path.join(subject_folder, "*.json"))
            motion_ids = set()  # Using a set to automatically handle duplicates
            
            for json_file in json_files:
                filename = os.path.basename(json_file)
                # Extract motion ID from filename
                match = re.search(f"subject_{subject_id}_motion_(\\d+)", filename)
                if match:
                    motion_id = match.group(1)
                    # Convert to canonical ID if it's an equivalent motion
                    canonical_id = equivalent_motions.get(motion_id, motion_id)
                    motion_ids.add(canonical_id)
            
            subject_motions[subject_id] = list(motion_ids)

    # Find common motions across all subjects
    all_motion_sets = [set(motions) for motions in subject_motions.values()]
    if all_motion_sets:
        common_motions = set.intersection(*all_motion_sets)
        common_motions = sorted([int(m) for m in common_motions])
        print(f"Common motions across all subjects: {common_motions}")
        print(f"Motion names: {[id_to_motion.get(str(m), f'Unknown-{m}') for m in common_motions]}")
    else:
        common_motions = []
        print("No subjects with motions found.")

    # Process each subject
    for subject_id in subjects:
        subject_folder = os.path.join(source_folder, f"pred_Subject_{subject_id}")
        
        if not os.path.exists(subject_folder):
            print(f"Subject folder not found: {subject_folder}")
            continue
        
        # Process each motion for this subject
        for canonical_motion_id in common_motions:
            canonical_motion_id_str = str(canonical_motion_id)
            
            # Find all possible equivalent motion IDs for this canonical ID
            possible_motion_ids = [motion_id for motion_id, canon_id in equivalent_motions.items() 
                                    if canon_id == canonical_motion_id_str]
            # Also include the canonical ID itself
            if canonical_motion_id_str not in possible_motion_ids:
                possible_motion_ids.append(canonical_motion_id_str)
            
            # Try all possible motion IDs
            found = False
            for motion_id in possible_motion_ids:
                # Pad with leading zeros (both 2-digit and 1-digit formats for compatibility)
                padded_motion_id_2 = motion_id.zfill(2)
                padded_motion_id_1 = motion_id.zfill(1)
                
                # Try both padding formats
                for padded_id in [padded_motion_id_2, padded_motion_id_1]:
                    motion_pattern = f"subject_{subject_id}_motion_{padded_id}.json"
                    motion_files = glob.glob(os.path.join(subject_folder, motion_pattern))
                    
                    if motion_files:
                        json_file_path = motion_files[0]
                        filename = os.path.basename(json_file_path)
                        filename_without_ext = os.path.splitext(filename)[0]
                        
                        # Create CSV for this motion
                        output_path = os.path.join(destination_folder, f"{filename_without_ext}.csv")
                        
                        # Process the JSON file and create CSV
                        with open(json_file_path, 'r') as f:
                            predictions = json.load(f)
                        
                        # Create empty lists to store the data
                        frame_ids = []
                        joint_names = []
                        x_3d = []
                        y_3d = []
                        z_3d = []
                        confidence = []
                        
                        # Process each frame
                        for frame_data in predictions:
                            frame_id = frame_data["frame_id"]
                            
                            # Check if there are instances in this frame
                            if 'instances' in frame_data and len(frame_data["instances"]) > 0:
                                # Get the first person (or you can loop through all if needed)
                                person_data = frame_data["instances"][0]
                                
                                # Extract 3D keypoints
                                keypoints_3d = person_data['keypoints']
                                scores = person_data['keypoint_scores']
                                
                                # Process each keypoint
                                for idx, (point, score) in enumerate(zip(keypoints_3d, scores)):
                                    if idx < len(H36M_KEYPOINT_NAMES):
                                        joint_name = H36M_KEYPOINT_NAMES[idx]
                                        
                                        frame_ids.append(frame_id)
                                        joint_names.append(joint_name)
                                        x_3d.append(point[0])
                                        y_3d.append(point[1])
                                        z_3d.append(point[2])
                                        confidence.append(score)
                        
                        # Create a DataFrame with the 3D keypoint data
                        df_3d = pd.DataFrame({
                            'frame_id': frame_ids,
                            'joint_name': joint_names,
                            'x_3d': x_3d,
                            'y_3d': y_3d,
                            'z_3d': z_3d,
                            'confidence': confidence,
                            'motion_id': canonical_motion_id,
                            'original_motion_id': motion_id,
                            'motion_name': id_to_motion.get(canonical_motion_id_str, f'Unknown-{canonical_motion_id}'),
                            'subject_id': subject_id
                        })
                        
                        # Save the DataFrame to CSV
                        df_3d.to_csv(output_path, index=False)
                        
                        found = True
                        break
                
                if found:
                    break
    
    return subject_motions, common_motions

def merge_subject_csv_files(destination_folder, merged_folder, subjects, common_motions):
    """
    Function to merge CSV files for each subject
    
    Args:
        destination_folder: Directory containing individual CSV files
        merged_folder: Directory to save merged CSV files
        subjects: List of subject IDs to process
        common_motions: List of motion IDs common to all subjects
    """
    os.makedirs(merged_folder, exist_ok=True)
    
    for subject_id in subjects:
        # Dictionary to store dataframes for common motions
        subject_dfs = {}
        
        # Process each motion for this subject
        for canonical_motion_id in common_motions:
            # Try different padding formats for motion IDs in filenames
            found = False
            
            # Try with 2-digit padding (00, 01, etc.)
            padded_id = str(canonical_motion_id).zfill(2)
            csv_pattern = f"subject_{subject_id}_motion_{padded_id}.csv"
            csv_files = glob.glob(os.path.join(destination_folder, csv_pattern))
            
            if not csv_files:
                # Try with 1-digit padding (0, 1, etc.)
                padded_id = str(canonical_motion_id).zfill(1)
                csv_pattern = f"subject_{subject_id}_motion_{padded_id}.csv"
                csv_files = glob.glob(os.path.join(destination_folder, csv_pattern))
            
            if csv_files:
                # Load CSV file
                csv_file_path = csv_files[0]
                df = pd.read_csv(csv_file_path)
                subject_dfs[canonical_motion_id] = df
        
        # Merge all dataframes for this subject
        if subject_dfs:
            # Sort by motion_id to ensure consistent order
            sorted_dfs = [subject_dfs[motion_id] for motion_id in sorted(subject_dfs.keys()) if motion_id in subject_dfs]
            
            if sorted_dfs:  # Check if we have any dataframes to merge
                merged_df = pd.concat(sorted_dfs, ignore_index=True)
                
                # Save merged dataframe
                merged_output_path = os.path.join(merged_folder, f"subject_{subject_id}_all_motions.csv")
                merged_df.to_csv(merged_output_path, index=False)
                print(f"Created merged file for subject {subject_id} with {len(sorted_dfs)} motions")
            else:
                print(f"No common motions found for subject {subject_id}")

def visualize_joint_trajectories(merged_folder, output_folder, common_motions, subjects, id_to_motion):
    """
    Visualize 3 sample joint trajectories for a specific motion across all subjects.
    
    Args:
        merged_folder: Directory containing merged CSV files
        output_folder: Directory to save visualization plots
        common_motions: List of common motion IDs
        subjects: List of subject IDs
        id_to_motion: Dictionary mapping motion IDs to motion names
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through each common motion
    for motion_id in common_motions:
        motion_name = id_to_motion.get(str(motion_id), f'Unknown-{motion_id}')
        print(f"Visualizing motion {motion_id}: {motion_name}")
        
        # Create a folder for this motion
        motion_folder = os.path.join(output_folder, f"motion_{motion_id}_{motion_name}")
        os.makedirs(motion_folder, exist_ok=True)
        
        # Dictionary to store data for each subject
        subject_data = {}
        
        # Load data for all subjects for this motion
        for subject_id in subjects:
            subject_file = os.path.join(merged_folder, f"subject_{subject_id}_all_motions.csv")
            if os.path.exists(subject_file):
                df = pd.read_csv(subject_file)
                # Extract only data for the current motion
                motion_df = df[df['motion_id'] == motion_id]
                if not motion_df.empty:
                    subject_data[subject_id] = motion_df
        
        if not subject_data:
            print(f"No data found for motion {motion_id}")
            continue
        
        # Get unique joints from the data
        sample_df = next(iter(subject_data.values()))
        all_joints = sample_df['joint_name'].unique()
        
        # Visualize trajectory for each joint
        sample_joints = all_joints[0:3]
        for joint in sample_joints:
            print(f"  Visualizing joint: {joint}")
            
            # Figure for 3D visualization
            # fig_3d = plt.figure(figsize=(12, 10))
            # ax_3d = fig_3d.add_subplot(111, projection='3d')
            # ax_3d.set_title(f'3D Trajectory of {joint} - Motion: {motion_name} (ID: {motion_id})')
            # ax_3d.set_xlabel('X')
            # ax_3d.set_ylabel('Y')
            # ax_3d.set_zlabel('Z')
            
            # 2D visualization of X, Y, Z coordinates over time
            fig_2d = plt.figure(figsize=(16, 10))
            gs = GridSpec(3, 1, figure=fig_2d)
            ax_x = fig_2d.add_subplot(gs[0, 0])
            ax_y = fig_2d.add_subplot(gs[1, 0])
            ax_z = fig_2d.add_subplot(gs[2, 0])
            
            ax_x.set_title(f'X Coordinate of {joint} Over Time - Motion: {motion_name} (ID: {motion_id})')
            ax_y.set_title(f'Y Coordinate of {joint} Over Time')
            ax_z.set_title(f'Z Coordinate of {joint} Over Time')
            
            ax_x.set_ylabel('X Position')
            ax_y.set_ylabel('Y Position')
            ax_z.set_ylabel('Z Position')
            ax_z.set_xlabel('Frame Number')
            
            # Color map for different subjects
            cmap = plt.cm.get_cmap('tab20', len(subject_data))
            
            # Plot data for each subject
            for i, (subject_id, df) in enumerate(subject_data.items()):
                # Get data for this joint
                joint_df = df[df['joint_name'] == joint].sort_values('frame_id')
                
                if joint_df.empty:
                    continue
                
                color = cmap(i)
                label = f'Subject {subject_id}'
                
                # Extract coordinates
                x = joint_df['x_3d'].values
                y = joint_df['y_3d'].values
                z = joint_df['z_3d'].values
                frames = joint_df['frame_id'].values
                
                # 3D trajectory plot
                # ax_3d.plot(x, y, z, marker='o', markersize=2, linestyle='-', linewidth=1, color=color, label=label)
                
                # 2D plots over time
                ax_x.plot(frames, x, marker='o', markersize=2, linestyle='-', linewidth=1, color=color, label=label)
                ax_y.plot(frames, y, marker='o', markersize=2, linestyle='-', linewidth=1, color=color)
                ax_z.plot(frames, z, marker='o', markersize=2, linestyle='-', linewidth=1, color=color)
            
            # Add legends
            # ax_3d.legend()
            ax_x.legend()
            
            # Adjust layout
            # fig_3d.tight_layout()
            fig_2d.tight_layout()
            
            # Save plots
            # fig_3d.savefig(os.path.join(motion_folder, f"{joint}_3D_trajectory.png"), dpi=300)
            fig_2d.savefig(os.path.join(motion_folder, f"{joint}_coordinates_over_time.png"), dpi=300)
            
            # Close figures to save memory
            # plt.close(fig_3d)
            plt.close(fig_2d)
            
        # Create a combined visualization for all joints in this motion
        visualize_motion_overview(subject_data, motion_id, motion_name, motion_folder, all_joints)

def visualize_motion_overview(subject_data, motion_id, motion_name, output_folder, all_joints):
    """
    Create an overview visualization showing all joints for a single subject
    
    Args:
        subject_data: Dictionary of dataframes by subject ID
        motion_id: Motion ID being visualized
        motion_name: Name of the motion
        output_folder: Folder to save the visualization
        all_joints: List of all joint names
    """
    # Choose a sample subject for overview visualization
    sample_subject_id = next(iter(subject_data.keys()))
    sample_df = subject_data[sample_subject_id]
    
    # Create figures for overview
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], figure=fig)
    
    # 3 subplots for X, Y, Z coordinates
    ax_x = fig.add_subplot(gs[0, 0])
    ax_y = fig.add_subplot(gs[1, 0])
    ax_z = fig.add_subplot(gs[2, 0])
    
    ax_x.set_title(f'X Coordinates of All Joints - Motion: {motion_name} (ID: {motion_id}) - Subject {sample_subject_id}')
    ax_y.set_title('Y Coordinates of All Joints')
    ax_z.set_title('Z Coordinates of All Joints')
    
    ax_x.set_ylabel('X Position')
    ax_y.set_ylabel('Y Position')
    ax_z.set_ylabel('Z Position')
    ax_z.set_xlabel('Frame Number')
    
    # Color map for different joints
    cmap = plt.cm.get_cmap('tab20', len(all_joints))
    
    # Plot each joint
    for i, joint in enumerate(all_joints):
        # Get data for this joint
        joint_df = sample_df[sample_df['joint_name'] == joint].sort_values('frame_id')
        
        if joint_df.empty:
            continue
        
        color = cmap(i)
        
        # Extract coordinates
        x = joint_df['x_3d'].values
        y = joint_df['y_3d'].values
        z = joint_df['z_3d'].values
        frames = joint_df['frame_id'].values
        
        # Plot coordinates over time
        ax_x.plot(frames, x, marker='', linestyle='-', linewidth=1, color=color, label=joint)
        ax_y.plot(frames, y, marker='', linestyle='-', linewidth=1, color=color)
        ax_z.plot(frames, z, marker='', linestyle='-', linewidth=1, color=color)
    
    # Add legend
    ax_x.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    
    # Adjust layout
    fig.tight_layout()
    
    # Save plot
    fig.savefig(os.path.join(output_folder, f"all_joints_overview_subject_{sample_subject_id}.png"), dpi=300)
    plt.close(fig)

