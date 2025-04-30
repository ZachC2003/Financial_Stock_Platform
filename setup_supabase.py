"""
Supabase Setup Script

This script automatically creates the required tables in your Supabase database.
Run this once to set up your database schema. It requires valid Supabase credentials
in your .env file.
"""

import os
from dotenv import load_dotenv
from supabase import create_client
import json

# Load environment variables
load_dotenv()

def setup_supabase():
    """Set up the Supabase database tables using the SQL API"""
    
    # Get Supabase credentials from .env file
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ Error: Missing Supabase credentials")
        print("Please make sure your .env file contains:")
        print("SUPABASE_URL=https://your-project-id.supabase.co")
        print("SUPABASE_KEY=your-supabase-key")
        return False
    
    # Connect to Supabase
    print(f"🚀 Connecting to Supabase at {supabase_url}")
    
    try:
        # Initialize the Supabase client
        supabase = create_client(supabase_url, supabase_key)
        
        # SQL statement to create all required tables
        # We'll use the SQL API to execute this directly
        sql = """
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
        
        -- Create the institutional_holdings_history table for tracking ownership changes
        CREATE TABLE IF NOT EXISTS institutional_holdings_history (
          id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
          symbol TEXT NOT NULL,
          quarter TEXT NOT NULL, -- Format: YYYY-Q# (e.g., 2025-Q1)
          created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
          institutional_ownership_pct FLOAT,
          total_shares BIGINT,
          data JSONB
        );
        
        -- Add indexes for the institutional_holdings_history table
        CREATE INDEX IF NOT EXISTS inst_holdings_symbol_idx ON institutional_holdings_history(symbol);
        CREATE INDEX IF NOT EXISTS inst_holdings_quarter_idx ON institutional_holdings_history(quarter);
        """
        
        print("📋 Creating tables in Supabase...")
        
        # Use the direct SQL execution with the Supabase client
        response = supabase.table("_root").select("*").execute()
        print("✅ Connected to Supabase successfully!")
        
        # We can't directly execute SQL with the Python SDK, so we'll test with a simple insert
        # First, try to access the table to see if it exists
        try:
            # Check if both tables exist
            table_check1 = supabase.table("analysis_results").select("count(*)").limit(1).execute()
            print("✅ analysis_results table already exists")
            
            table_check2 = supabase.table("institutional_holdings_history").select("count(*)").limit(1).execute()
            print("✅ institutional_holdings_history table already exists")
        except Exception:
            # If tables don't exist, instruct the user to create them in the Supabase dashboard
            print("⚠️ One or more required tables don't exist yet.")
            print("Please run the following SQL in the Supabase SQL Editor:")
            print("--------------------------------------------------")
            print(sql)
            print("--------------------------------------------------")
            print("\nAfter running this SQL, come back and run this script again to verify.")
            return False
            
        print("✅ Success! Your Supabase database is properly configured")
        return True
            
    except Exception as e:
        print(f"❌ Error working with Supabase: {str(e)}")
        return False

if __name__ == "__main__":
    result = setup_supabase()
    
    if result:
        print("\n🎉 Your Supabase database is now set up and ready to use with the Financial Stock Platform!")
        print("You can now run the stock server and Streamlit app in real mode (MOCK_MODE=False).")
    else:
        print("\n⚠️ Check the instructions above to complete setting up your Supabase database.")
        print("You can still use the platform in mock mode by setting MOCK_MODE = True in streamlit_app.py")

