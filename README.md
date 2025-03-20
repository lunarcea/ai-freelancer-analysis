# Digital Labor Markets Skill Metrics Analysis

This repository contains code and tools for analyzing skill metrics in digital labor markets, with a particular focus on how these metrics evolve over time and how they relate to various market outcomes.

## Project Overview

The digital labor market is constantly evolving, with new technologies and platforms changing how work is organized and evaluated. This project provides tools to:

- Calculate and track skill metrics over time (skill depth, skill diversity)
- Analyze the relationship between skill metrics and market outcomes (contract success, ratings, completion time)
- Identify patterns in skill acquisition and specialization
- Examine how market changes affect different types of workers

## Repository Structure
digital-labor-markets/

├── src/

│ ├── python/

│ │ ├── skill_metrics_processor.py # Core skill metric calculations

│ │ ├── visualization_generator.py # Creates data visualizations

│ │ └── nlp_processor.py # NLP-based skill analysis

│ ├── sql/

│ │ └── db_schema_setup.sql # Database schema for project data

│ └── stata/

│ └── statistical_analysis.do # Statistical modeling and hypothesis testing

├── notebook/ # Jupyter notebooks for exploratory analysis

└── README.md # Project documentation



## Key Features

### Skill Metrics

- **Skill Depth**: Measures how interconnected a user's skills are, indicating specialization
- **Skill Diversity**: Measures the variety of skill categories a user possesses, indicating breadth
- **Temporal Analysis**: Tracks how these metrics change over time

### Analysis Capabilities

- Time-series analysis of skill metrics
- Relationship between skill metrics and market outcomes
- Difference-in-differences analysis for policy/platform changes
- Heterogeneous effects analysis by user characteristics
- Visualization of key relationships and trends

## Getting Started

### Prerequisites

- Python 3.7+
- Stata 16+ (for statistical analysis)
- SQL Server or compatible database (for data storage)

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/digital-labor-markets.git
   cd digital-labor-markets

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   
3. Create the necessary folders:
   ```bash
   mkdir -p data/input data/output

### Usage
Processing Skill Metrics
   ```bash
   python src/python/skill_metrics_processor.py
```
This script calculates skill depth and diversity metrics for users in the dataset. It requires input data in CSV format with columns for user IDs, skills, and dates.

Generating Visualizations
   ```bash
  python src/python/visualization_generator.py
```
Creates visualizations for skill metrics, including time series and relationship plots.

Statistical Analysis
To run the Stata analysis:
```bash
 stata -b do src/stata/statistical_analysis.do
```
This performs regression analysis and hypothesis testing on the processed data.

## Data Protection

This repository does not include real data due to confidentiality requirements. To use this codebase:

1. **Prepare your own data** following the format described in `DATA.md`
2. **Place your data files** in the `data/input/` directory
3. **Ensure your `.gitignore` is properly set** to avoid accidentally committing sensitive data
4. **Use synthetic data** for testing and demonstration purposes

### Data Security Best Practices

When working with sensitive labor market data:
- Never commit data files to Git repositories
- Consider encrypting data files when stored locally
- Be cautious about sharing outputs that might reveal personal information
- Aggregate results when possible to prevent identification of individuals

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments
This research builds on literature in labor economics and digital platforms

Special thanks to researchers and practitioners in the digital labor markets space
