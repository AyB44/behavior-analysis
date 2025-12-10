"""
Simple Rodent Behavioral Analysis
Analyzes Open Field Test data from CSV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data(filename):
    """Load trajectory data from CSV"""
    df = pd.read_csv(filename)
    return df

def calculate_metrics(df, arena_size=50):
    """Calculate behavioral metrics for each animal"""
    results = []
    
    for animal_id in df['animal_id'].unique():
        animal_data = df[df['animal_id'] == animal_id]
        
        # Extracting the coordinates
        x = animal_data['x'].values
        y = animal_data['y'].values
        time = animal_data['time'].values
        
        # Calculating distance traveled
        dx = np.diff(x)
        dy = np.diff(y)
        distances = np.sqrt(dx**2 + dy**2)
        total_distance = np.sum(distances)
        
        # Calculating velocity
        dt = np.diff(time)
        velocities = distances / dt
        mean_velocity = np.mean(velocities)
        
        # Center zone analysis (middle part of arena)
        center_boundary = arena_size * 0.25
        in_center = ((x > center_boundary) & 
                    (x < arena_size - center_boundary) &
                    (y > center_boundary) & 
                    (y < arena_size - center_boundary))
        
        time_in_center = np.sum(in_center) / len(in_center) * (time[-1] - time[0])
        center_percentage = (np.sum(in_center) / len(in_center)) * 100
        
        # Counting entries to center
        center_entries = np.sum(np.diff(in_center.astype(int)) > 0)
        
        # Storing the  results
        results.append({
            'animal_id': animal_id,
            'group': animal_data['group'].iloc[0],
            'total_distance_cm': total_distance,
            'mean_velocity_cm_s': mean_velocity,
            'time_in_center_s': time_in_center,
            'center_percentage': center_percentage,
            'center_entries': center_entries
        })
    
    return pd.DataFrame(results)

def plot_trajectories(df, save_path='trajectories.png'):
    """Plot example trajectories from each group"""
    groups = df['group'].unique()
    fig, axes = plt.subplots(1, len(groups), figsize=(14, 6))
    
    if len(groups) == 1:
        axes = [axes]
    
    for idx, group in enumerate(groups):
        # Get first animal from group
        animal = df[df['group'] == group]['animal_id'].iloc[0]
        animal_data = df[df['animal_id'] == animal]
        
        x = animal_data['x'].values
        y = animal_data['y'].values
        
        # Plotting trajectory
        axes[idx].plot(x, y, alpha=0.6, linewidth=1)
        axes[idx].plot(x[0], y[0], 'go', markersize=10, label='Start')
        axes[idx].plot(x[-1], y[-1], 'ro', markersize=10, label='End')
        
        # Drawing our center zone
        center_size = 50 * 0.5
        center_start = 50 * 0.25
        axes[idx].add_patch(plt.Rectangle((center_start, center_start), 
                                         center_size, center_size,
                                         fill=False, edgecolor='red', 
                                         linestyle='--', linewidth=2))
        
        axes[idx].set_xlim(0, 50)
        axes[idx].set_ylim(0, 50)
        axes[idx].set_xlabel('X position (cm)')
        axes[idx].set_ylabel('Y position (cm)')
        axes[idx].set_title(f'{group.upper()}: {animal}')
        axes[idx].legend()
        axes[idx].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Trajectories saved to {save_path}")

def plot_metrics_comparison(metrics_df, save_path='comparison.png'):
    """Plot comparison of key metrics between groups"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics_to_plot = ['total_distance_cm', 'mean_velocity_cm_s', 
                       'center_percentage', 'center_entries']
    titles = ['Total Distance Traveled', 'Mean Velocity', 
              'Time in Center (%)', 'Center Entries']
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx]
        
        # Box plotting with individual points
        sns.boxplot(data=metrics_df, x='group', y=metric, ax=ax)
        sns.swarmplot(data=metrics_df, x='group', y=metric, 
                     color='black', alpha=0.5, size=6, ax=ax)
        
        # Adding statistical test
        groups = metrics_df['group'].unique()
        if len(groups) == 2:
            g1 = metrics_df[metrics_df['group'] == groups[0]][metric]
            g2 = metrics_df[metrics_df['group'] == groups[1]][metric]
            t_stat, p_val = stats.ttest_ind(g1, g2)
            
            ax.text(0.5, 0.95, f'p = {p_val:.4f}', 
                   transform=ax.transAxes,
                   ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('')
    
    plt.suptitle('Behavioral Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {save_path}")

def statistical_analysis(metrics_df):
    print("\nSTATISTICAL ANALYSIS REPORT")
    print("="*60 + "\n")
    
    groups = metrics_df['group'].unique()
    
    if len(groups) == 2:
        print(f"Comparing groups: {groups[0]} vs {groups[1]}\n")
        
        metrics = ['total_distance_cm', 'mean_velocity_cm_s', 
                  'center_percentage', 'center_entries']
        
        for metric in metrics:
            g1 = metrics_df[metrics_df['group'] == groups[0]][metric]
            g2 = metrics_df[metrics_df['group'] == groups[1]][metric]
            
            # Descriptive stats
            print(f"{metric}:")
            print(f"  {groups[0]}: {g1.mean():.2f} ± {g1.std():.2f} (n={len(g1)})")
            print(f"  {groups[1]}: {g2.mean():.2f} ± {g2.std():.2f} (n={len(g2)})")
            
            # t-test
            t_stat, p_val = stats.ttest_ind(g1, g2)
            print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")
            
            if p_val < 0.05:
                print(f"  *** SIGNIFICANT DIFFERENCE (p < 0.05) ***")
            
            print()

def main():
    """Main analysis pipeline"""
    print("Starting behavioral analysis...\n")
    
    # Load data
    print("Loading data...")
    df = load_data('oft_data.csv')
    print(f"Loaded data for {df['animal_id'].nunique()} animals")
    print(f"Groups: {df['group'].unique()}\n")
    
    # Calculating metrics
    print("Calculating behavioral metrics...")
    metrics_df = calculate_metrics(df)
    
    # Saving metrics
    metrics_df.to_csv('behavioral_metrics.csv', index=False)
    print("Metrics saved to behavioral_metrics.csv\n")
    
    # Print summary - only include numeric columns to avoid the error
    print("Summary statistics by group:")
    numeric_cols = metrics_df.select_dtypes(include=['number']).columns
    print(metrics_df.groupby('group')[numeric_cols].mean())
    print()
    
    # Creating visualizations
    print("Creating visualizations...")
    plot_trajectories(df)
    plot_metrics_comparison(metrics_df)
    
    # Statistical analysis
    statistical_analysis(metrics_df)
    
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
