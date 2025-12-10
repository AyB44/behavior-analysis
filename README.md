# Behavioral Analysis Pipeline

A Python-based analysis pipeline for Open Field Test (OFT) behavioral data in rodents, demonstrating computational skills for behavioral neuroscience research.

## Overview

This project demonstrates automated behavioral analysis skills relevant to neuroscience research, specifically:
- Processing trajectory data from behavioral experiments
- Extracting relevant behavioral metrics
- Statistical comparison between experimental groups
- Data visualization for publication

**Note**: This is a demonstration project using simulated data. Parameters were chosen to showcase the analysis pipeline. In real experiments, effect sizes and behavioral patterns would be determined by the specific biological model being studied.

## Features

- **Automated metric calculation**: Distance traveled, velocity, center time, entries
- **Statistical analysis**: Group comparisons with appropriate statistical tests
- **Publication-quality visualizations**: Trajectory plots, heatmaps, group comparisons
- **Reproducible pipeline**: Complete workflow from raw data to figures
- **Modular design**: Easy to adapt for real experimental data

## Project Structure

```
behavior-analysis/
├── README.md
├── requirements.txt
├── generate_sample_data.py    # Creates simulated OFT data
├── analyze_behavior.py         # Main analysis script
├── data/
│   ├── oft_data.csv           # Trajectory data (x, y, time)
│   └── behavioral_metrics.csv # Calculated metrics
└── outputs/
    ├── trajectories.png       # Example movement paths
    └── comparison.png         # Statistical comparisons
```

## Installation

```bash
# Clone the repository
git clone https://github.com/AyB44/behavior-analysis.git
cd behavior-analysis

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Sample Data

```bash
python generate_sample_data.py
```

This creates `oft_data.csv` with simulated trajectory data for 16 animals (8 control, 8 anxious).

### 2. Run Analysis

```bash
python analyze_behavior.py
```

This will:
- Load trajectory data
- Calculate behavioral metrics for each animal
- Generate visualizations
- Perform statistical comparisons
- Save results to CSV and PNG files

## Behavioral Metrics

The pipeline calculates the following metrics for each animal:

### Locomotor Activity
- **Total distance traveled (cm)**: Overall locomotor activity throughout the test
- **Mean velocity (cm/s)**: Average movement speed
- **Max velocity (cm/s)**: Peak movement speed

### Anxiety-Related Measures
- **Time in center (s, %)**: Duration spent in the center zone of the arena
  - *Interpretation*: Reduced center time indicates increased anxiety-like behavior
- **Center entries**: Number of times the animal enters the center zone
  - *Interpretation*: Fewer entries suggest heightened anxiety and reduced exploration
- **Mean distance from center (cm)**: Average distance from arena center
  - *Interpretation*: Greater distance indicates thigmotaxis (wall-hugging behavior)

### Exploratory Behavior
- **Path tortuosity**: Ratio of total distance to straight-line distance
  - *Interpretation*: Higher values indicate more exploratory, less directed movement
- **Immobility time (s)**: Duration of low movement (velocity < 1 cm/s)
  - *Interpretation*: Increased immobility may indicate freezing behavior

## Example Results

### Trajectory Analysis
![Trajectories]<img width="3809" height="1774" alt="trajectories" src="https://github.com/user-attachments/assets/5df5ed02-d7cc-436f-8282-548e04fdda28" />
```

```
*Representative trajectories showing control (left) and anxious (right) behavioral patterns. Green dot = start position, red dot = end position. Red dashed box indicates center zone.*

### Group Comparison
![Comparison]<img width="4170" height="2957" alt="comparison" src="https://github.com/user-attachments/assets/269e3678-4c1a-4212-8c25-897d4252e110" />

```
```
*Statistical comparison of key behavioral metrics between control and anxious groups. Box plots show median, quartiles, and individual data points. P-values from independent t-tests are displayed.*

## Interpretation of Results

This analysis demonstrates behavioral patterns consistent with anxiety-like phenotypes:

**Key Findings:**
- **Center zone exploration**: Control animals showed greater time in center (mean: 89.2%) compared to anxious group (15.3%), though this did not reach statistical significance (p=0.49)
- **Exploratory entries**: Control animals made more center entries (1.0 vs 0.25), suggesting reduced exploration in the anxious group
- **Locomotor activity**: Both groups showed similar overall distance traveled and velocity patterns

**Statistical Considerations:**
The lack of statistical significance (all p > 0.05) may reflect:
1. Small sample size (n=8 per group) - typical behavioral studies use n=12-15
2. High inter-individual variability (common in behavioral assays)
3. Simulated data parameters chosen for demonstration purposes

**Biological Relevance:**
Similar behavioral patterns are observed in:
- Autism Spectrum Disorder (ASD) rodent models (e.g., Shank3, Cntnap2 knockouts)
- Anxiety disorder models (genetic or pharmacological)
- Stress-induced behavioral alterations

## Use Case: Relevance to ASD Research

The Open Field Test is a standard behavioral assay for characterizing ASD-related phenotypes in rodent models:

### Translational Value
- **Repetitive behaviors**: Path patterns can reveal stereotyped movements
- **Anxiety comorbidity**: Center avoidance correlates with anxiety symptoms common in ASD
- **Exploratory drive**: Reduced exploration may reflect altered novelty responses
- **Developmental trajectories**: Can be performed across age to track symptom emergence

### Integration with Circuit Neuroscience
This pipeline can be extended to analyze behavior during:
- Optogenetic manipulation of specific neural circuits
- Chemogenetic activation/inhibition of cell populations  
- Pharmacological interventions
- Developmental time-course studies

## Technical Skills Demonstrated

- **Python programming**: Object-oriented design, clean code structure
- **Data analysis**: Pandas, NumPy for efficient data manipulation
- **Statistics**: Hypothesis testing, normality checks, effect sizes
- **Visualization**: Matplotlib, Seaborn for publication-quality figures
- **Scientific computing**: Spatial trajectory analysis, kinematic calculations
- **Version control**: Git/GitHub workflow
- **Documentation**: Clear README, code comments, reproducible workflow

## Adaptability to Real Data

This pipeline is designed to easily integrate with real experimental data from:

### Automated Tracking Systems
- **DeepLabCut**: Markerless pose estimation
- **SLEAP**: Multi-animal tracking
- **Bonsai**: Real-time behavioral tracking
- **EthoVision**: Commercial tracking software
- **Kinect-based systems**: 3D motion capture

### Integration Example
```python
# Load DeepLabCut output
dlc_data = pd.read_csv('DLC_output.csv', header=[1,2])
x = dlc_data['bodypart']['x'].values
y = dlc_data['bodypart']['y'].values
time = np.arange(len(x)) / fps

# Create standardized format
trajectory_df = pd.DataFrame({
    'animal_id': 'mouse_01',
    'group': 'experimental',
    'time': time,
    'x': x,
    'y': y
})

# Run analysis pipeline
metrics = calculate_metrics(trajectory_df)
```

## Future Extensions

Potential enhancements for more comprehensive analysis:

### Additional Behavioral Metrics
- Rearing behavior (vertical exploration)
- Grooming bouts (stereotyped behavior)
- Freezing episodes (fear/anxiety)
- Corner preference (spatial bias)
- Acceleration patterns (movement dynamics)

### Advanced Analysis
- **Machine learning classification**: Automated behavioral state detection
- **Temporal dynamics**: Time-bin analysis, habituation curves
- **Path complexity**: Fractal dimension, entropy measures
- **Social behavior**: Multi-animal tracking and interaction analysis
- **Circadian patterns**: Long-duration recording analysis

### Visualization Enhancements
- Interactive dashboards (Plotly, Dash)
- Real-time plotting during experiments
- 3D trajectory visualization
- Video overlay with tracking data

## Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum
- Works on Windows, macOS, Linux

## Contributing

Contributions are welcome! Areas for improvement:
- Additional behavioral metrics
- Support for more tracking formats
- Machine learning features
- Real-time analysis capabilities

## Citation

If you use or adapt this code for your research, please cite:

```
Ayaan Borthakur. (2025). Behavioral Analysis Pipeline. 
GitHub: https://github.com/AyB44/behavior-analysis
```

## Related Resources

### Tools & Software
- [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut): Markerless pose estimation
- [SLEAP](https://sleap.ai/): Multi-animal tracking
- [MoSeq](https://dattalab.github.io/moseq2-website/): Unsupervised behavior analysis

### Literature
- Seibenhener, M.L. and Wooten, M.C., 2015. Use of the open field maze to measure locomotor and anxiety-like behavior in mice. Journal of visualized experiments: JoVE, (96), p.52434.
- Silverman, J.L., Yang, M., Lord, C. and Crawley, J.N., 2010. Behavioural phenotyping assays for mouse models of autism. Nature Reviews Neuroscience, 11(7), pp.490-502.

## Contact

**Ayaan Borthakur**    
Email: [ayaanborthakur2019@gmail.com]

---

## License

MIT License - feel free to use and modify for your research or educational purposes.
