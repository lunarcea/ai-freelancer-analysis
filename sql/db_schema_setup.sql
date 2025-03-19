-- Database Schema Setup for Digital Labor Market Analysis
--
-- This script sets up the relational database schema for digital labor market data
-- including tables for projects, bids, users, and their relationships.
--
-- Tables include:
-- - projects: Contains project information
-- - bids: Contains bid information
-- - users: Contains user information
-- - project_skills: Contains skills required for each project

-- Create projects table
CREATE TABLE IF NOT EXISTS projects (
    project_id VARCHAR(255) PRIMARY KEY,
    owner_id VARCHAR(255),
    title VARCHAR(255),
    status VARCHAR(50),
    currency VARCHAR(10),
    description TEXT,
    type VARCHAR(50),
    bidperiod INT,
    bid_count INT,
    bid_avg DECIMAL(10, 2),
    budget DECIMAL(10, 2),
    featured BIT,
    sealed BIT,
    nonpublic BIT,
    fulltime BIT,
    urgent BIT,
    qualified BIT,
    NDA BIT,
    ip_contract BIT,
    recruiter BIT,
    skills TEXT,
    listed VARCHAR(50),
    language VARCHAR(50),
    owner_username VARCHAR(255),
    owner_name VARCHAR(255),
    owner_country VARCHAR(100),
    owner_rating DECIMAL(3, 2),
    owner_reviews INT,
    owner_email_verified BIT,
    owner_payment_verified BIT,
    owner_deposit_made BIT,
    owner_profile_complete BIT,
    owner_phone_verified BIT,
    owner_identity_verified BIT,
    owner_facebook_connected BIT,
    created_date DATETIME,
    CONSTRAINT FK_owner_id FOREIGN KEY (owner_id) REFERENCES users(user_id)
);

-- Create project budget table
CREATE TABLE IF NOT EXISTS project_budget (
    id INT IDENTITY(1,1) PRIMARY KEY,
    project_id VARCHAR(255),
    min_budget DECIMAL(10, 2),
    max_budget DECIMAL(10, 2),
    CONSTRAINT FK_ProjectBudget_project_id FOREIGN KEY (project_id) 
    REFERENCES projects (project_id)
);

-- Create project status table
CREATE TABLE IF NOT EXISTS project_status (
    id INT IDENTITY(1,1) PRIMARY KEY,
    project_id VARCHAR(255),
    status VARCHAR(50),
    updated_date DATETIME,
    CONSTRAINT FK_project_status_project_id FOREIGN KEY (project_id) 
    REFERENCES projects (project_id)
);

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(255) PRIMARY KEY,
    username VARCHAR(255),
    name VARCHAR(255),
    country VARCHAR(100),
    registration_date DATETIME,
    rating DECIMAL(3, 2),
    reviews INT,
    earnings_score DECIMAL(5, 2),
    completion_rate DECIMAL(5, 2)
);

-- Create bids table
CREATE TABLE IF NOT EXISTS bids (
    bid_id VARCHAR(255) PRIMARY KEY,
    project_id VARCHAR(255),
    bidder_id VARCHAR(255),
    bid_date DATETIME,
    amount DECIMAL(10, 2),
    period INT,
    description TEXT,
    status VARCHAR(50),
    award_status VARCHAR(50),
    sponsored BIT,
    highlighted BIT,
    CONSTRAINT FK_bids_project_id FOREIGN KEY (project_id) 
    REFERENCES projects (project_id),
    CONSTRAINT FK_bids_bidder_id FOREIGN KEY (bidder_id)
    REFERENCES users (user_id)
);

-- Create project skills table
CREATE TABLE IF NOT EXISTS project_skills (
    id INT IDENTITY(1,1) PRIMARY KEY,
    project_id VARCHAR(255),
    skill VARCHAR(255),
    CONSTRAINT FK_project_skills_project_id FOREIGN KEY (project_id) 
    REFERENCES projects (project_id)
);

-- Create user skills table
CREATE TABLE IF NOT EXISTS user_skills (
    id INT IDENTITY(1,1) PRIMARY KEY,
    user_id VARCHAR(255),
    skill VARCHAR(255),
    endorsements INT,
    CONSTRAINT FK_user_skills_user_id FOREIGN KEY (user_id) 
    REFERENCES users (user_id)
);

-- Create views for commonly used queries

-- View for projects with their average bid
CREATE VIEW vw_projects_with_avg_bid AS
SELECT 
    p.project_id,
    p.title,
    p.owner_id,
    p.status,
    p.currency,
    p.budget,
    COUNT(b.bid_id) AS bid_count,
    AVG(b.amount) AS average_bid_amount
FROM 
    projects p
LEFT JOIN 
    bids b ON p.project_id = b.project_id
GROUP BY 
    p.project_id, p.title, p.owner_id, p.status, p.currency, p.budget;

-- View for user activity
CREATE VIEW vw_user_activity AS
SELECT 
    u.user_id,
    u.username,
    COUNT(DISTINCT CASE WHEN p.project_id IS NOT NULL THEN p.project_id END) AS projects_posted,
    COUNT(DISTINCT CASE WHEN b.bid_id IS NOT NULL THEN b.project_id END) AS projects_bid_on,
    COUNT(DISTINCT CASE WHEN b.award_status = 'awarded' THEN b.project_id END) AS projects_won
FROM 
    users u
LEFT JOIN 
    projects p ON u.user_id = p.owner_id
LEFT JOIN 
    bids b ON u.user_id = b.bidder_id
GROUP BY 
    u.user_id, u.username;

-- Function to check if specific skills occur together frequently
CREATE FUNCTION fn_skill_co_occurrence (@skill1 VARCHAR(255), @skill2 VARCHAR(255))
RETURNS INT
AS
BEGIN
    DECLARE @count INT;
    
    SELECT @count = COUNT(*)
    FROM project_skills ps1
    JOIN project_skills ps2 ON ps1.project_id = ps2.project_id
    WHERE ps1.skill = @skill1 AND ps2.skill = @skill2;
    
    RETURN @count;
END;

CREATE TABLE