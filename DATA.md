# Data Documentation

## Overview
This repository uses proprietary data that cannot be shared publicly due to confidentiality agreements and data protection requirements.

## Required Data Structure
To use this code, you'll need to prepare your own data with the following structure:

### Input Data Format
The main analysis requires a CSV file with the following columns:
- `user_id`: Unique identifier for each user (string)
- `date`: Date of the observation (YYYY-MM-DD format)
- `skills`: Comma-separated list of skills (string)
- `category`: Skill category (string)
- `rating`: User rating (float, 0-5 scale)
- `success_rate`: Contract success rate (float, 0-1 scale)
- `earnings`: User earnings (numeric)
- [Additional columns as needed]

### Sample Data (Synthetic)
```csv
user_id,date,skills,category,rating,success_rate,earnings
user123,2023-01-15,"python,data analysis,machine learning",Technical,4.8,0.75,1250
user456,2023-01-20,"javascript,html,css",Design,4.5,0.68,950