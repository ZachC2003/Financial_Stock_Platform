-- Financial Stock Platform - Supabase Database Setup
-- Run this entire script in your Supabase SQL Editor (https://app.supabase.io)

-- Create the analysis_results table if it doesn't exist
CREATE TABLE IF NOT EXISTS analysis_results (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  symbol TEXT NOT NULL,
  analysis_type TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  data JSONB NOT NULL
);

-- Add indexes for faster queries
CREATE INDEX IF NOT EXISTS analysis_results_symbol_idx ON analysis_results(symbol);
CREATE INDEX IF NOT EXISTS analysis_results_type_idx ON analysis_results(analysis_type);

-- Grant access to the table
ALTER TABLE analysis_results ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow read access to all users" ON analysis_results FOR SELECT USING (true);
CREATE POLICY "Allow insert access to all users" ON analysis_results FOR INSERT WITH CHECK (true);

-- Create institutional_holdings_history table for tracking ownership changes
CREATE TABLE IF NOT EXISTS institutional_holdings_history (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  symbol TEXT NOT NULL,
  institution_name TEXT NOT NULL, 
  date DATE NOT NULL,
  shares BIGINT,
  value NUMERIC,
  percentage NUMERIC,
  change_from_previous NUMERIC,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes for efficient querying
CREATE INDEX IF NOT EXISTS institutional_holdings_symbol_idx ON institutional_holdings_history(symbol);
CREATE INDEX IF NOT EXISTS institutional_holdings_date_idx ON institutional_holdings_history(date);
CREATE INDEX IF NOT EXISTS institutional_holdings_institution_idx ON institutional_holdings_history(institution_name);

-- Enable row level security 
ALTER TABLE institutional_holdings_history ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow read access to all users" ON institutional_holdings_history FOR SELECT USING (true);
CREATE POLICY "Allow insert access to all users" ON institutional_holdings_history FOR INSERT WITH CHECK (true);

-- Let's confirm it worked
SELECT 'Tables setup complete!' as result;
