#!/usr/bin/env python3
"""
Manual Inactivity Status Updater

This script manually updates player inactivity status based on last match date.
Players with no matches in the last N weeks are marked as inactive.

Usage:
    python update_inactivity.py [--weeks 4]
"""

import argparse
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from elo_utils import update_inactivity_status

# Load environment variables
load_dotenv()

def main():
    """Main function to update inactivity status"""
    parser = argparse.ArgumentParser(
        description="Update player inactivity status based on last match date"
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=4,
        help="Number of weeks of inactivity before marking as inactive (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Initialize Supabase client
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            print("Error: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not found in environment variables")
            print("Please make sure your .env file contains these variables")
            return 1
        
        supabase_client: Client = create_client(supabase_url, supabase_key)
        print(f"Connected to Supabase")
        print(f"Updating inactivity status (threshold: {args.weeks} weeks)...")
        print("-" * 60)
        
        # Update inactivity status
        success = update_inactivity_status(supabase_client, inactivity_threshold_weeks=args.weeks)
        
        if success:
            print("-" * 60)
            print("Inactivity status update completed successfully!")
            return 0
        else:
            print("-" * 60)
            print("Error: Failed to update inactivity status")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

