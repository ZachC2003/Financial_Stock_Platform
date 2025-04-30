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

-- Let's confirm it worked
SELECT 'Table setup complete!' as result;
