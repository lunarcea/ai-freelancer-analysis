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

├── data/

│ ├── input/ # Place input data files here

│ └── output/ # Output files will be stored here


├── src/

│ ├── python/

│ │ ├── skill_metrics_processor.py # Core skill metric calculations

│ │ ├── visualization_generator.py # Creates data visualizations

│ │ └── nlp_processor.py # NLP-based skill analysis

│ ├── sql/

│ │ └── db_schema_setup.sql # Database schema for project data

│ └── stata/

│ └── statistical_analysis.do # Statistical modeling and hypothesis testing

├── notebooks/ # Jupyter notebooks for exploratory analysis

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

### Data Format
The input data should include:
- User information: IDs, characteristics, performance metrics
- Skill information: Skills associated with each user
- Temporal information: Dates for tracking changes over time
- Sample data format:
```bash
   user_id,date,skills,rating,success_rate
   user123,2023-01-15,"python,data analysis,machine learning",4.8,0.75
   user456,2023-01-20,"javascript,html,css",4.5,0.68
```

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments
This research builds on literature in labor economics and digital platforms

Special thanks to researchers and practitioners in the digital labor markets space
