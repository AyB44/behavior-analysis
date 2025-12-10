"""
The purpose of this code is to generate sample Open Field Test data
It will create a CSV file with simulated rodent trajectories
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# Setting the random seed
np.random.seed(42)

# Defining a funciton to simulate movement for animal
def generate_trajectory(animal_id, group, arena_size=50, duration=300, fps=30):
    """Generate a single animal trajectory"""
    n_frames = duration * fps
    
    # Initializing at edge
    x = np.zeros(n_frames)
    y = np.zeros(n_frames)
    x[0] = 5
    y[0] = arena_size / 2
    
    # Setting the behavior based on group
    if group == 'control':
        center_avoidance = 0.4
        speed = 1.2
    else:  # anxious group
        center_avoidance = 0.85
        speed = 0.6
    
    # Generating movement
    for i in range(1, n_frames):
        center_x, center_y = arena_size/2, arena_size/2
        dist_from_center = np.sqrt((x[i-1] - center_x)**2 + (y[i-1] - center_y)**2)
        
        # Avoid center (thigmotaxis)
        if dist_from_center < arena_size * 0.3:
            angle = np.arctan2(y[i-1] - center_y, x[i-1] - center_x)
            push = center_avoidance * (1 - dist_from_center / (arena_size * 0.3))
            dx = np.cos(angle) * push
            dy = np.sin(angle) * push
        else:
            dx = np.random.randn() * 0.3 * speed
            dy = np.random.randn() * 0.3 * speed
        
        # Adding momentum
        if i > 1:
            dx += 0.3 * (x[i-1] - x[i-2])
            dy += 0.3 * (y[i-1] - y[i-2])
        
        x[i] = x[i-1] + dx
        y[i] = y[i-1] + dy
        
        # Boundary conditions
        x[i] = np.clip(x[i], 0, arena_size)
        y[i] = np.clip(y[i], 0, arena_size)
    
    # Smooth trajectories
    x = gaussian_filter1d(x, sigma=2)
    y = gaussian_filter1d(y, sigma=2)
    
    time = np.arange(n_frames) / fps
    
    return pd.DataFrame({
        'animal_id': animal_id,
        'group': group,
        'time': time,
        'x': x,
        'y': y
    })

def generate_dataset():
    """Generate complete dataset"""
    trajectories = []
    
    # Control group
    for i in range(8):
        df = generate_trajectory(f'control_{i:02d}', 'control')
        trajectories.append(df)
    
    # Treatment group (more anxious)
    for i in range(8):
        df = generate_trajectory(f'anxious_{i:02d}', 'anxious')
        trajectories.append(df)
    
    # Combining all
    all_data = pd.concat(trajectories, ignore_index=True)
    
    # Saving output 
    all_data.to_csv('oft_data.csv', index=False)
    print(f"Generated data for {all_data['animal_id'].nunique()} animals")
    print(f"Total data points: {len(all_data)}")
    print(f"Groups: {all_data['group'].unique()}")
    print("\nData saved to: oft_data.csv")
    print("\nFirst few rows:")
    print(all_data.head())

if __name__ == '__main__':
    generate_dataset()
