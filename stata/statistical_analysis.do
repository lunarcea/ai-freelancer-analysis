/*
Statistical Analysis for Skill Metrics Impact Study

This Stata script performs statistical analysis to examine:
1. The impact of skill metrics (depth/diversity) on various outcomes
2. Difference-in-differences analysis with a reference period
3. Heterogeneous effects by user characteristics

The script cleans data, handles outliers, and performs regression analysis
with proper controls and robustness checks.
*/

clear all
set more off

// ---------------------------------------------------------------------------
// Data Import and Preparation
// ---------------------------------------------------------------------------

// Import data - replace path with local data path 
import delimited "data/input/skill_metrics_data.csv", clear

// Convert string date to Stata date format
gen date_numeric = date(date, "YMDhms")
format date_numeric %td

// Convert categorical variables to numeric
local categorical_vars "status project_type currency"
foreach var of local categorical_vars {
    encode `var', gen(`var'_numeric)
}

// Define analysis period indicator
// Change the reference date as needed for your study
gen period = .
replace period = 0 if date_numeric < date("2023-01-01", "YMD")
replace period = 1 if date_numeric >= date("2023-01-01", "YMD")

// Keep only observations with data in both periods
keep if has_data_both_periods == 1

// Sort by user ID
sort user_id

// ---------------------------------------------------------------------------
// Variable Transformations and Outlier Handling
// ---------------------------------------------------------------------------

// Generate log variables
foreach var of varlist rating earnings_score completion_rate tenure_months {
    // Handle zeros and negative values for log transformation
    gen temp_`var' = `var'
    replace temp_`var' = 0.01 if temp_`var' <= 0
    gen log_`var' = log(temp_`var')
    drop temp_`var'
}

// Winsorize variables at 5th and 95th percentiles
foreach var of varlist bid_amount paid_amount bid_order num_ratings client_experience {
    // Calculate percentiles
    _pctile `var', p(5 95)
    scalar p5_`var' = r(r1)
    scalar p95_`var' = r(r2)
    
    // Replace outliers
    replace `var' = p5_`var' if `var' < p5_`var'
    replace `var' = p95_`var' if `var' > p95_`var'
    
    // Ensure no zero values for log transformation
    replace `var' = 0.01 if `var' <= 0
    gen log_`var' = log(`var')
}

// Create bid position variable 
egen min_bid = min(bid_amount)
egen max_bid = max(bid_amount)
gen bid_position = (bid_amount - min_bid) / (max_bid - min_bid)
drop min_bid max_bid

// Create job value categories
gen job_value_category = .
replace job_value_category = 1 if bid_amount >= 0 & bid_amount < 100
replace job_value_category = 2 if bid_amount >= 100 & bid_amount < 500
replace job_value_category = 3 if bid_amount >= 500 & bid_amount < 5000
replace job_value_category = 4 if bid_amount >= 5000 & bid_amount < 25000
replace job_value_category = 5 if bid_amount >= 25000 | bid_amount == .

// Create award indicator
gen is_awarded = (award_status == "awarded")

// ---------------------------------------------------------------------------
// Main Analysis: Skill Metrics and Outcomes
// ---------------------------------------------------------------------------

// Generate dependent variables
gen skill_metric_ratio = skill_metric_after / skill_metric_before
gen log_skill_metric_ratio = log(skill_metric_ratio)

// Create experience/quality groups for heterogeneity analysis
gen high_quality = (rating >= 4.9)
label define quality_groups 0 "Low Quality" 1 "High Quality"
label values high_quality quality_groups

// Logistic regression for propensity score matching
logit treated log_rating completion_rate log_tenure_months ///
       log_bid_amount language_preference log_bid_order bid_position ///
       same_country client_experience is_awarded log_num_ratings ///
       avg_rating previous_experience category project_length ///
       currency_numeric if period == 1, cluster(user_id) 

// Generate propensity scores
predict pscore, pr
sum pscore, detail
scalar cal = r(sd) * 0.1

// Perform propensity score matching
psmatch2 treated, pscore(pscore) outcome(skill_metric_ratio) ///
        caliper(`=scalar(cal)') neighbor(1) noreplace

// Generate pair identifiers for matched observations
bysort user_id: gen pair = _id if _treat == 0 & period == 1
bysort user_id: egen paircount = count(pair) if period == 1
replace paircount = . if paircount < 2 & period == 1

// Fill in pair values across observations for the same user
xfill pair, i(user_id)
xfill paircount, i(user_id)

// Handle missing pair values
bysort user_id: egen first_pair = min(pair)
replace pair = first_pair if missing(pair)

// ---------------------------------------------------------------------------
// Regression Analysis: Main Effects
// ---------------------------------------------------------------------------

// Analysis 1: Main effect on skill metric
regress skill_metric_ratio treated##period log_rating completion_rate ///
        log_tenure_months language_preference log_bid_amount log_bid_order ///
        bid_position same_country project_description_length ///
        client_experience is_awarded log_num_ratings avg_rating ///
        previous_experience category project_length currency_numeric ///
        if period==1 | period==0 [fweight=_weight]

// ---------------------------------------------------------------------------
// Analysis 2: Effect on Success Rate
// ---------------------------------------------------------------------------

// Generate success variables
gen award_ratio = success_rate_after / success_rate_before
gen log_award_ratio = log(award_ratio + 1)

// Generate interaction terms
gen treated_skill_metric = treated * skill_metric_ratio

// Regression analysis
regress log_award_ratio treated skill_metric_ratio treated_skill_metric ///
        bid_count_after bid_count_before log_rating completion_rate ///
        log_tenure_months language_preference log_bid_amount log_bid_order ///
        bid_position same_country client_experience is_awarded ///
        log_num_ratings avg_rating previous_experience category ///
        project_length currency_numeric if period==1 | period==0 [fweight=_weight]

// ---------------------------------------------------------------------------
// Analysis 3: Effect on Quality Metrics (Rating)
// ---------------------------------------------------------------------------

// Prepare rating outcome variable
gen rating_outcome = rating

// Regression analysis
regress rating_outcome treated skill_metric_ratio treated_skill_metric ///
        bid_count_total log_rating completion_rate log_tenure_months ///
        language_preference log_bid_amount log_bid_order bid_position ///
        same_country client_experience is_awarded log_num_ratings ///
        avg_rating previous_experience category project_length ///
        currency_numeric if is_awarded == 1 [fweight=_weight]

// ---------------------------------------------------------------------------
// Analysis 4: Effect on Project Completion Time
// ---------------------------------------------------------------------------

// Generate completion time variable
gen completion_time = completion_days
gen log_completion_time = log(completion_time + 1)

// Generate interaction terms with squared components
gen treated_skill_metric2 = treated_skill_metric
gen inc_treated_skill_metric = (treated_skill_metric + 1)^2

// Regression analysis
regress log_completion_time treated skill_metric_ratio inc_treated_skill_metric ///
        bid_count_total log_rating completion_rate log_tenure_months ///
        language_preference log_bid_amount log_bid_order bid_position ///
        same_country client_experience is_awarded log_num_ratings ///
        avg_rating previous_experience category currency_numeric ///
        if is_awarded == 1 [fweight=_weight]

// ---------------------------------------------------------------------------
// Heterogeneity Analysis: High vs. Low Quality Users
// ---------------------------------------------------------------------------

// For low quality users
regress skill_metric_ratio treated##period log_rating completion_rate ///
        log_tenure_months language_preference log_bid_amount log_bid_order ///
        bid_position same_country client_experience is_awarded ///
        log_num_ratings avg_rating previous_experience category ///
        project_length currency_numeric if high_quality == 0 [fweight=_weight]

// For high quality users
regress skill_metric_ratio treated##period log_rating completion_rate ///
        log_tenure_months language_preference log_bid_amount log_bid_order ///
        bid_position same_country client_experience is_awarded ///
        log_num_ratings avg_rating previous_experience category ///
        project_length currency_numeric if high_quality == 1 [fweight=_weight]

// ---------------------------------------------------------------------------
// Robustness Checks: Alternative Specifications
// ---------------------------------------------------------------------------

// Robustness check 1: Using raw differences instead of ratios
gen skill_metric_diff = skill_metric_after - skill_metric_before

regress skill_metric_diff treated##period log_rating completion_rate ///
        log_tenure_months language_preference log_bid_amount log_bid_order ///
        bid_position same_country client_experience is_awarded ///
        log_num_ratings avg_rating previous_experience category ///
        project_length currency_numeric if period==1 | period==0 [fweight=_weight]

// Robustness check 2: Different caliper for matching
scalar cal_alt = r(sd) * 0.2
psmatch2 treated, pscore(pscore) outcome(skill_metric_ratio) ///
        caliper(`=scalar(cal_alt)') neighbor(1) noreplace

// Regenerate weights and matched pairs
bysort user_id: gen pair_alt = _id if _treat == 0 & period == 1
xfill pair_alt, i(user_id)

// Rerun main regression with alternative matching
regress skill_metric_ratio treated##period log_rating completion_rate ///
        log_tenure_months language_preference log_bid_amount log_bid_order ///
        bid_position same_country client_experience is_awarded ///
        log_num_ratings avg_rating previous_experience category ///
        project_length currency_numeric if period==1 | period==0 [fweight=_weight]