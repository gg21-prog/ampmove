#!/usr/bin/env python3
"""
Parse MotionBuilder-style BVH file (e.g., 7_12_motionbuilder.bvh)
and save raw motion data as a structured .npy file.

Output format:
{
  "joint_names": [...],         # list of joint names (Hips first)
  "frames": np.array,           # (T, num_channels): [root_x, root_y, root_z, root_rotZ, root_rotX, root_rotY, joint1, joint2, ...]
}
"""

import os
import argparse
import numpy as np
from bvh import Bvh
from typing import List

def parse_bvh(bvh_path):
    """
    Parse a MotionBuilder-style BVH file.
    
    Args:
        bvh_path: Path to the BVH file
        
    Returns:
        Dictionary containing joint names, frame data, and frame time
    """
    with open(bvh_path, "r") as f:
        mocap = Bvh(f.read())

    # Get joint hierarchy (Hips is root)
    joint_names = [j.name for j in mocap.get_joints()]
    print(f"Found joints: {joint_names}")

    # Build frame data
    frames = []
    for i in range(mocap.nframes):
        frame = []

        # Root position (X, Y, Z)
        hips_pos = [float(mocap.frame_joint_channel(i, "Hips", c)) for c in ["Xposition", "Yposition", "Zposition"]]
        frame.extend(hips_pos)

        # Root rotation in MotionBuilder order: Z, X, Y (Euler angles, degrees)
        hips_rot = [float(mocap.frame_joint_channel(i, "Hips", c)) for c in ["Zrotation", "Xrotation", "Yrotation"]]
        frame.extend(hips_rot)

        # All other joints: assume single Zrotation (or dominant axis)
        for joint in joint_names[1:]:
            # Try Z, then Y, then X — fallback to first available
            rot_val = 0.0
            for axis in ["Zrotation", "Yrotation", "Xrotation"]:
                try:
                    rot_val = float(mocap.frame_joint_channel(i, joint, axis))
                    break
                except KeyError:
                    continue
            frame.append(rot_val)

        frames.append(frame)

    frames = np.array(frames, dtype=np.float32)  # (T, D)

    return {
        "joint_names": joint_names,
        "frames": frames,
        "frame_time": mocap.frame_time
    }

def save_as_npy(data, output_path):
    """
    Save parsed BVH data as a structured .npy file.
    
    Args:
        data: Dictionary containing joint_names and frames
        output_path: Path to save the .npy file
    """
    np.save(output_path, data)
    print(f"✅ Saved parsed data to {output_path}")
    print(f"   Shape: {data['frames'].shape}")
    print(f"   Joints: {len(data['joint_names'])} | Duration: {data['frames'].shape[0] * data['frame_time']:.2f}s")


def convert_bvh_to_npy(bvh_path, output_path=None):
    """
    Convert a BVH file to a structured .npy file.
    
    Args:
        bvh_path: Path to input .bvh file
        output_path: Path to output .npy file (default: same name as input)
    """
    if not os.path.exists(bvh_path):
        raise FileNotFoundError(f"BVH file not found: {bvh_path}")

    data = parse_bvh(bvh_path)
    
    output_path = output_path or bvh_path.replace(".bvh", "_parsed.npy")
    save_as_npy(data, output_path)
    
    return data


def get_available_bvh_files(base_path: str = "motion_priors") -> List[str]:
    """
    Get paths to all BVH files under the motion_priors directory.
    
    Args:
        base_path: Base path to search for BVH files (default: "motion_priors")
        
    Returns:
        List of paths to BVH files
    """
    import os
    bvh_files = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.bvh'):
                bvh_files.append(os.path.join(root, file))
    
    return bvh_files


def main():
    parser = argparse.ArgumentParser(description="Parse BVH file to .npy")
    parser.add_argument("input_bvh", type=str, help="Path to input .bvh file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output .npy path (default: same name as input)")
    args = parser.parse_args()

    if not os.path.exists(args.input_bvh):
        raise FileNotFoundError(f"BVH file not found: {args.input_bvh}")

    data = parse_bvh(args.input_bvh)

    output_path = args.output or args.input_bvh.replace(".bvh", "_parsed.npy")
    save_as_npy(data, output_path)


if __name__ == "__main__":
    main()