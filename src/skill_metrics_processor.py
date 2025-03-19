"""
Skill Metrics Processor

This script calculates various skill-related metrics for users on digital platforms:
- Skill depth: measures how interconnected a user's skills are
- Skill diversity: measures the variety of skill categories a user possesses
- Accumulated metrics: tracks how metrics change over time

These metrics can be used to analyze how skill profiles evolve over time,
particularly useful for studying market changes or technology adoption effects.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import psutil

def debug_memory_usage():
    """Print current memory usage."""
    process = psutil.Process()
    print(f"Current memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")


class SkillDepthCalculator:
    """
    Calculates the interconnectedness (depth) of skills based on co-occurrence patterns.
    Skill depth measures how related the skills in a freelancer's portfolio are to each other.
    """
    def __init__(self):
        self.skill_to_index = {}
        self.index_to_skill = []
        self.co_occurrence = None
        self.skill_totals = None
    
    def _get_skill_index(self, skill):
        """Get index for a skill, creating new index if needed."""
        if skill not in self.skill_to_index:
            self.skill_to_index[skill] = len(self.index_to_skill)
            self.index_to_skill.append(skill)
            
            # Expand co-occurrence matrix if needed
            if self.co_occurrence is None:
                self.co_occurrence = np.zeros((1, 1), dtype=int)
                self.skill_totals = np.zeros(1, dtype=int)
            else:
                new_size = len(self.index_to_skill)
                new_matrix = np.zeros((new_size, new_size), dtype=int)
                new_matrix[:-1, :-1] = self.co_occurrence
                self.co_occurrence = new_matrix
                
                new_totals = np.zeros(new_size, dtype=int)
                new_totals[:-1] = self.skill_totals
                self.skill_totals = new_totals
                
        return self.skill_to_index[skill]
    
    def update_co_occurrence(self, skills_list):
        """Update co-occurrence matrix with new skills."""
        if not skills_list:
            return
            
        # Get indices for all skills
        indices = [self._get_skill_index(skill.strip()) for skill in skills_list]
        
        # Update co-occurrence matrix
        for i, idx1 in enumerate(indices):
            for idx2 in indices[i+1:]:
                self.co_occurrence[idx1, idx2] += 1
                self.co_occurrence[idx2, idx1] += 1
                self.skill_totals[idx1] += 1
                self.skill_totals[idx2] += 1
    
    def compute_similarity_weights(self):
        """Compute similarity weights matrix based on co-occurrence."""
        weights = np.zeros_like(self.co_occurrence, dtype=float)
        for i in range(len(self.index_to_skill)):
            for j in range(i + 1, len(self.index_to_skill)):
                denominator = self.skill_totals[i] + self.skill_totals[j]
                if denominator > 0:
                    weight = self.co_occurrence[i, j] / denominator
                    weights[i, j] = weight
                    weights[j, i] = weight
        return weights
    
    def calculate_skill_depth(self, skills, similarity_weights):
        """
        Calculate skill depth for a set of skills.
        
        Higher values indicate skills that frequently appear together (more specialized).
        Lower values indicate more diverse skill combinations.
        """
        if not skills:
            return 0
            
        indices = []
        for skill in skills:
            if skill in self.skill_to_index:
                indices.append(self.skill_to_index[skill])
        
        if len(indices) < 2:
            return 0
            
        connections = []
        for i, idx1 in enumerate(indices):
            for idx2 in indices[i+1:]:
                connections.append(similarity_weights[idx1, idx2])
        
        return np.mean(connections) if connections else 0


def calculate_skill_depth_accumulated(df):
    """
    Calculate skill depth by bidder_id, accumulating skills over time.
    
    This allows tracking how a freelancer's skill depth evolves as they
    acquire new skills over time.
    """
    print("Starting data preprocessing...")
    debug_memory_usage()
    
    # Convert biddate to datetime and sort
    print("Converting dates...")
    df['biddate'] = pd.to_datetime(df['biddate'])
    df['date'] = df['biddate'].dt.date
    df = df.sort_values(['biddate', 'bidder_id'])
    
    # Get all unique dates and bidders upfront
    all_dates = sorted(df['date'].unique())
    all_bidders = df['bidder_id'].unique()
    print(f"Total unique dates to process: {len(all_dates)}")
    print(f"Total unique bidders: {len(all_bidders)}")
    
    # Initialize results list
    intermediate_results = []
    
    # Dictionary to track accumulated data by date
    date_accumulated_data = {}
    
    # Process each date
    print("Processing dates and accumulating skills...")
    for current_date in all_dates:
        print(f"Processing date: {current_date}")
        
        # Get all data up to current date
        if current_date not in date_accumulated_data:
            date_accumulated_data[current_date] = df[df['date'] <= current_date].copy()
        
        accumulated_data = date_accumulated_data[current_date]
        
        # Build co-occurrence matrix for this date
        calculator = SkillDepthCalculator()
        for _, row in accumulated_data.iterrows():
            if pd.isna(row['review_skills']) or not isinstance(row['review_skills'], str):
                continue
            skills = [s.strip() for s in row['review_skills'].split(',') if s.strip()]
            calculator.update_co_occurrence(skills)
        
        # Compute similarity weights for this date
        similarity_weights = calculator.compute_similarity_weights()
        
        # Get current date's entries
        current_date_data = df[df['date'] == current_date]
        
        # For each bidder on current date
        for bidder_id in current_date_data['bidder_id'].unique():
            # Get accumulated skills for this bidder up to current date
            bidder_accumulated = accumulated_data[accumulated_data['bidder_id'] == bidder_id]
            bidder_skills = set()
            
            for _, row in bidder_accumulated.iterrows():
                if pd.isna(row['review_skills']) or not isinstance(row['review_skills'], str):
                    continue
                skills = [s.strip() for s in row['review_skills'].split(',') if s.strip()]
                bidder_skills.update(skills)
            
            # Calculate skill depth for this bidder using their accumulated skills
            if not bidder_skills:
                skill_depth = 0
            else:
                skill_depth = calculator.calculate_skill_depth(list(bidder_skills), similarity_weights)
            
            # Get current date entries for this bidder
            current_bidder_entries = current_date_data[current_date_data['bidder_id'] == bidder_id]
            
            # Add results for each entry of this bidder on current date
            for _, row in current_bidder_entries.iterrows():
                intermediate_results.append({
                    'bidder_id': bidder_id,
                    'biddate': row['biddate'],
                    'date': current_date,
                    'individual_skill_depth': skill_depth,
                    'accumulated_skill_count': len(bidder_skills)
                })
        
        # Clean up
        del calculator
        if current_date in date_accumulated_data:
            del date_accumulated_data[current_date]
        
        if len(intermediate_results) % 10000 == 0:
            print(f"Processed {len(intermediate_results)} records")
            debug_memory_usage()
    
    # Convert results to DataFrame and save intermediate results
    print("Saving intermediate results...")
    results_df = pd.DataFrame(intermediate_results)
    intermediate_file = 'intermediate_skill_depth_results.csv'
    results_df.to_csv(intermediate_file, index=False)
    print(f"Intermediate results saved to: {intermediate_file}")
    
    return results_df


def calculate_skill_diversity(category_series):
    """
    Calculate Blau Index of Diversity using predefined categories.
    
    The Blau Index measures the diversity of skills across different categories:
    1 - Σ(p_i^2) where p_i is the proportion of skills in category i
    
    Higher values indicate more diverse skills across categories.
    """
    # Count occurrences of each category
    category_counts = category_series.value_counts()
    
    # Calculate total number of entries
    total_entries = len(category_series)
    
    # Calculate proportions and sum of squared proportions
    if total_entries == 0:
        return 0
    
    # Calculate proportion of each category and square it
    proportions = category_counts / total_entries
    squared_proportions = proportions ** 2
    
    # Blau Index: 1 - Σ(p_i^2)
    diversity = 1 - squared_proportions.sum()
    
    return diversity


def process_skill_diversity_multiple_periods(df):
    """
    Process skill diversity for multiple test periods including the main cutoff 
    and placebo test periods.
    
    Allows comparison of pre-post changes around different dates to validate findings.
    """
    # Ensure biddate is datetime
    df['biddate'] = pd.to_datetime(df['biddate'])
    
    # Define test periods (matching your Stata code)
    test_periods = {
        'main': {
            'cutoff': '2022-11-30',
            'prefix': 'skilldiversity'
        },
        'falsification': {
            'before_start': '2022-10-15',
            'before_end': '2022-10-31',
            'after_start': '2022-11-08',
            'after_end': '2022-11-29',
            'prefix': 'skilldiversity_falsification'
        },
        'falsification_balanced': {
            'before_start': '2022-10-30',
            'before_end': '2022-11-09',
            'after_start': '2022-11-20',
            'after_end': '2022-11-29',
            'prefix': 'skilldiversity_balanced'
        }
    }
    
    result_df = df.copy()
    
    # Process each test period
    for period_name, period_info in test_periods.items():
        if 'cutoff' in period_info:
            # Process main period
            cutoff_date = pd.to_datetime(period_info['cutoff'])
            df_before = df[df['biddate'] < cutoff_date]
            df_after = df[df['biddate'] >= cutoff_date]
        else:
            # Process placebo periods
            before_mask = (df['biddate'] >= pd.to_datetime(period_info['before_start'])) & \
                         (df['biddate'] <= pd.to_datetime(period_info['before_end']))
            after_mask = (df['biddate'] >= pd.to_datetime(period_info['after_start'])) & \
                        (df['biddate'] <= pd.to_datetime(period_info['after_end']))
            
            df_before = df[before_mask]
            df_after = df[after_mask]
        
        # Calculate diversity for before period
        skilldiversity_before = (
            df_before.groupby('bidder_id')['category']
            .apply(calculate_skill_diversity)
            .reset_index(name=f'{period_info["prefix"]}_before')
        )
        
        # Calculate diversity for after period
        skilldiversity_after = (
            df_after.groupby('bidder_id')['category']
            .apply(calculate_skill_diversity)
            .reset_index(name=f'{period_info["prefix"]}_after')
        )
        
        # Merge results
        result_df = result_df.merge(skilldiversity_before, on='bidder_id', how='left')
        result_df = result_df.merge(skilldiversity_after, on='bidder_id', how='left')
        
        # Fill NaN values with 0
        result_df[f'{period_info["prefix"]}_before'] = result_df[f'{period_info["prefix"]}_before'].fillna(0.0)
        result_df[f'{period_info["prefix"]}_after'] = result_df[f'{period_info["prefix"]}_after'].fillna(0.0)
    
    return result_df


def main():
    """Main execution function."""
    try:
        print(f"Start time: {datetime.now()}")
        debug_memory_usage()
        
        # Read the input file - replace with your actual file path
        input_file = "data/input/user_skills_data.csv"
        print(f"Reading data from: {input_file}")
        df = pd.read_csv(input_file, low_memory=False)
        print(f"Data loaded. Shape: {df.shape}")
        
        # Process skill diversity with multiple periods (including placebo tests)
        print("Calculating skill diversity metrics...")
        diversity_df = process_skill_diversity_multiple_periods(df)
        diversity_output = 'output/skill_diversity_metrics.csv'
        os.makedirs('output', exist_ok=True)
        diversity_df.to_csv(diversity_output, index=False)
        print(f"Skill diversity metrics saved to: {diversity_output}")
        
        # Check if intermediate results exist
        intermediate_file = 'output/intermediate_skill_depth_results.csv'
        if os.path.exists(intermediate_file):
            print("Loading existing intermediate skill depth results...")
            skill_depth_results = pd.read_csv(intermediate_file)
            skill_depth_results['date'] = pd.to_datetime(skill_depth_results['date']).dt.date
        else:
            print("Calculating accumulated skill depth metrics...")
            skill_depth_results = calculate_skill_depth_accumulated(df)
            os.makedirs('output', exist_ok=True)
            skill_depth_results.to_csv(intermediate_file, index=False)
        
        # Save final accumulated skill depth results
        output_path = 'output/accumulated_skill_depth.csv'
        final_df = df.merge(
            skill_depth_results[['bidder_id', 'date', 'individual_skill_depth', 'accumulated_skill_count']],
            on=['bidder_id', 'date'],
            how='left'
        )
        final_df.to_csv(output_path, index=False)
        
        print("\nProcessing completed!")
        print(f"End time: {datetime.now()}")
        print(f"Results saved to: {output_path} and {diversity_output}")
        debug_memory_usage()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()