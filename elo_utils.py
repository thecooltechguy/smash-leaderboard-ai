#!/usr/bin/env python3
"""
Shared ELO utilities for Smash Bros Leaderboard

This module contains all ELO calculation functions including rank ceiling logic
to avoid duplication across the codebase.
"""

import datetime
import pytz
from typing import Dict, Tuple, Optional
from supabase import Client
import json


# Rank ceiling implementation date - only apply ceiling after this date (California time)
RANK_CEILING_START_DATE = pytz.timezone('America/Los_Angeles').localize(datetime.datetime(2025, 8, 15, 0, 0, 0))

# Daily snapshot reset time (2 AM Pacific)
DAILY_RESET_HOUR = 2
PACIFIC_TZ = pytz.timezone('America/Los_Angeles')


def get_current_snapshot_date() -> str:
    """
    Get the current snapshot date based on 2 AM Pacific reset time.
    If it's before 2 AM, use the previous day as the snapshot date.
    
    Returns:
        str: Date string in YYYY-MM-DD format
    """
    now = datetime.datetime.now(PACIFIC_TZ)
    
    # If it's before 2 AM, use previous day
    if now.hour < DAILY_RESET_HOUR:
        snapshot_date = (now - datetime.timedelta(days=1)).date()
    else:
        snapshot_date = now.date()
    
    return snapshot_date.strftime('%Y-%m-%d')


def get_or_create_daily_snapshot(supabase_client: Client, current_elos: Dict[str, int]) -> Dict[str, int]:
    """
    Get or create the daily ELO snapshot for rank ceiling calculations.
    
    Args:
        supabase_client: Supabase client instance
        current_elos: Current ELO ratings of all players
    
    Returns:
        Dict[str, int]: Snapshot ELO ratings for rank ceiling calculations
    """
    snapshot_date = get_current_snapshot_date()
    
    try:
        # Try to get existing snapshot
        response = supabase_client.table("elo_snapshots")\
            .select("*")\
            .eq("snapshot_date", snapshot_date)\
            .execute()
        
        if response.data:
            # Snapshot exists, parse and return it
            snapshot_data = response.data[0]
            return json.loads(snapshot_data['elo_data'])
        else:
            # No snapshot exists, create one
            snapshot_elo_data = json.dumps(current_elos)
            
            # Insert new snapshot
            supabase_client.table("elo_snapshots").insert({
                "snapshot_date": snapshot_date,
                "elo_data": snapshot_elo_data,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }).execute()
            
            print(f"Created new daily ELO snapshot for {snapshot_date}")
            return current_elos.copy()
            
    except Exception as e:
        print(f"Error managing daily snapshot: {e}")
        print("Falling back to current ELOs for rank ceiling")
        return current_elos.copy()


def ensure_snapshot_table_exists(supabase_client: Client) -> bool:
    """
    Ensure the daily_elo_snapshots table exists in the database.
    This is for documentation - the table should be created manually.
    
    Required table schema:
    CREATE TABLE daily_elo_snapshots (
        id SERIAL PRIMARY KEY,
        snapshot_date DATE NOT NULL UNIQUE,
        elo_data JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    
    Returns:
        bool: True if table operations work, False otherwise
    """
    try:
        # Test if table exists by trying to select from it
        supabase_client.table("elo_snapshots").select("id").limit(1).execute()
        return True
    except Exception as e:
        print(f"Warning: daily_elo_snapshots table may not exist: {e}")
        print("Please create the table with this schema:")
        print("""
        CREATE TABLE daily_elo_snapshots (
            id SERIAL PRIMARY KEY,
            snapshot_date DATE NOT NULL UNIQUE,
            elo_data JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """)
        return False


def update_elo(rating_a: float, rating_b: float, winner: str, k: int = 32) -> tuple[int, int]:
    """
    Return the new (rating_a, rating_b) after one game.

    Parameters
    ----------
    rating_a : current Elo for Player A
    rating_b : current Elo for Player B
    winner   : 'A', 'B', or 'draw'
    k        : K-factor (default 32)

    Returns
    -------
    tuple[int, int] : New ratings as integers for (Player A, Player B)
    """
    # Expected scores (logistic curve)
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 - expected_a

    # Actual scores
    if winner.upper() == 'A':
        score_a, score_b = 1.0, 0.0
    elif winner.upper() == 'B':
        score_a, score_b = 0.0, 1.0
    elif winner.lower() in ('draw', 'tie', 'd'):
        score_a = score_b = 0.5
    else:
        raise ValueError("winner must be 'A', 'B', or 'draw'")

    # Rating updates
    new_rating_a = rating_a + k * (score_a - expected_a)
    new_rating_b = rating_b + k * (score_b - expected_b)

    # Return as integers
    return round(new_rating_a), round(new_rating_b)


def get_player_ranking(player_id: str, current_elos: Dict[str, int]) -> int:
    """Get the current ranking of a player based on ELO"""
    sorted_players = sorted(current_elos.items(), key=lambda x: x[1], reverse=True)
    for rank, (pid, elo) in enumerate(sorted_players, 1):
        if pid == player_id:
            return rank
    return len(sorted_players) + 1


def get_elo_ceiling(player_id: str, opponent_id: str, current_elos: Dict[str, int], 
                   snapshot_elos: Optional[Dict[str, int]] = None) -> int:
    """
    Calculate the ELO ceiling for a player based on rank ceiling rules.
    
    Rules:
    - A player cannot surpass someone ranked higher unless they beat that person
    - If playing against someone ranked lower, max ELO gain stops 1 point shy of next higher player
    - If playing against someone ranked higher, can gain full ELO (no ceiling)
    
    Args:
        player_id: ID of the player gaining ELO
        opponent_id: ID of the opponent
        current_elos: Current ELO ratings (for finding current ELO values)
        snapshot_elos: Daily snapshot ELO ratings (for ranking calculations)
    """
    # Use snapshot ELOs for ranking calculations if provided, otherwise fall back to current
    ranking_elos = snapshot_elos if snapshot_elos is not None else current_elos
    
    player_rank = get_player_ranking(player_id, ranking_elos)
    opponent_rank = get_player_ranking(opponent_id, ranking_elos)
    
    # If opponent is ranked higher or equal in the SNAPSHOT, no ceiling applies (can gain full ELO)
    if opponent_rank <= player_rank:
        return float('inf')
    
    # Opponent is ranked lower in SNAPSHOT, so apply ceiling
    # Find the ELO of the player ranked immediately above the current player in CURRENT rankings
    current_player_elo = current_elos.get(player_id, 1200)
    sorted_current_players = sorted(current_elos.items(), key=lambda x: x[1], reverse=True)
    
    # Find players with ELO higher than current player
    higher_players = [(pid, elo) for pid, elo in sorted_current_players if elo > current_player_elo]
    
    if not higher_players:
        # No one has higher ELO currently, no ceiling
        return float('inf')
    
    # Get the lowest ELO among higher players (the "next" player to surpass)
    next_higher_elo = min(elo for pid, elo in higher_players)
    
    # Ceiling is 1 point below the next higher player
    return next_higher_elo - 1


def get_inactive_player_last_match_dates_from_db(supabase_client: Client) -> Dict[str, datetime.datetime]:
    """Pre-compute last match dates for all inactive players (using Supabase queries)"""
    try:
        # Get all inactive players
        players_response = supabase_client.table("players").select("*").eq("inactive", True).execute()
        inactive_players = [p['id'] for p in players_response.data]
        
        if not inactive_players:
            return {}
        
        inactive_player_dates = {}
        
        for player_id in inactive_players:
            # Get all matches this player participated in - use a simpler approach
            # First get all participant records for this player
            participants_response = supabase_client.table("match_participants")\
                .select("match_id")\
                .eq("player", player_id)\
                .execute()
            
            if participants_response.data:
                # Get all match IDs
                match_ids = [p['match_id'] for p in participants_response.data]
                
                # Get the most recent match date from these matches (exclude archived)
                matches_response = supabase_client.table("matches")\
                    .select("created_at")\
                    .in_("id", match_ids)\
                    .eq("archived", False)\
                    .order("created_at", desc=True)\
                    .limit(1)\
                    .execute()
                
                if matches_response.data:
                    last_match_str = matches_response.data[0]['created_at']
                    # Parse and make timezone-aware
                    if last_match_str.endswith('Z'):
                        last_match_str = last_match_str.replace('Z', '+00:00')
                    elif '+' not in last_match_str and not last_match_str.endswith('+00:00'):
                        last_match_str += '+00:00'
                    
                    last_match_date = datetime.datetime.fromisoformat(last_match_str)
                    if last_match_date.tzinfo is None:
                        last_match_date = last_match_date.replace(tzinfo=datetime.timezone.utc)
                    
                    inactive_player_dates[player_id] = last_match_date
        
        return inactive_player_dates
    except Exception as e:
        print(f"Error getting inactive player dates: {e}")
        return {}


def get_inactive_player_last_match_dates_from_dataframes(players_df, matches_df, participants_df) -> Dict[str, datetime.datetime]:
    """Pre-compute last match dates for all inactive players (using pandas DataFrames - optimized for batch)"""
    try:
        import pandas as pd
        
        inactive_players = set(players_df[players_df['inactive'].fillna(False)]['id'].tolist())
        inactive_player_dates = {}
        
        for player_id in inactive_players:
            # Get all matches this player participated in
            player_matches = participants_df[participants_df['player'] == player_id]
            if len(player_matches) > 0:
                # Join with matches to get dates, using suffixes to avoid column name conflicts
                player_match_data = player_matches.merge(matches_df[['id', 'created_at']], 
                                                       left_on='match_id', right_on='id', 
                                                       how='inner', suffixes=('', '_match'))
                
                if len(player_match_data) > 0:
                    # Convert dates and get their last match date
                    player_match_data['match_datetime'] = pd.to_datetime(player_match_data['created_at_match'], utc=True)
                    last_match_date = player_match_data['match_datetime'].max()
                    inactive_player_dates[player_id] = last_match_date
        
        return inactive_player_dates
    except Exception as e:
        print(f"Error getting inactive player dates from dataframes: {e}")
        return {}


def update_elo_with_ceiling(rating_a: float,
                           rating_b: float,
                           winner: str,
                           player_a_id: str,
                           player_b_id: str,
                           current_elos: Dict[str, int],
                           inactive_player_dates: Dict[str, datetime.datetime],
                           match_date: datetime.datetime,
                           snapshot_elos: Optional[Dict[str, int]] = None,
                           k: int = 32) -> tuple[int, int]:
    """
    Return the new (rating_a, rating_b) after one game with rank ceiling applied.
    Excludes players who became inactive before this match date.
    """
    # Create active_elos dict excluding players who were inactive before this match
    active_elos = {}
    for player_id, elo in current_elos.items():
        if player_id in inactive_player_dates:
            # This player is marked inactive - check if they were inactive before this match
            last_match_date = inactive_player_dates[player_id]
            if last_match_date >= match_date:
                # Player was still active at this match date
                active_elos[player_id] = elo
        else:
            # Player is not marked inactive
            active_elos[player_id] = elo
    
    # Calculate normal ELO update first
    new_rating_a, new_rating_b = update_elo(rating_a, rating_b, winner, k)
    
    # Filter snapshot_elos to only active players if snapshot is provided
    active_snapshot_elos = None
    if snapshot_elos is not None:
        active_snapshot_elos = {pid: elo for pid, elo in snapshot_elos.items() 
                               if pid in active_elos}
    
    # Apply ceiling for Player A if they gained ELO (using active players only)
    if new_rating_a > rating_a:
        ceiling_a = get_elo_ceiling(player_a_id, player_b_id, active_elos, active_snapshot_elos)
        if new_rating_a > ceiling_a:
            new_rating_a = ceiling_a
            # Adjust Player B's rating accordingly to maintain zero-sum property
            actual_gain_a = new_rating_a - rating_a
            
            # Calculate what B's rating should be to maintain zero-sum
            if winner.upper() == 'A':
                # If A won but hit ceiling, B loses less
                rating_loss_b = actual_gain_a
                new_rating_b = rating_b - rating_loss_b
    
    # Apply ceiling for Player B if they gained ELO (using active players only)
    if new_rating_b > rating_b:
        # Use the same active_snapshot_elos as calculated above
        ceiling_b = get_elo_ceiling(player_b_id, player_a_id, active_elos, active_snapshot_elos)
        if new_rating_b > ceiling_b:
            new_rating_b = ceiling_b
            # Adjust Player A's rating accordingly to maintain zero-sum property
            actual_gain_b = new_rating_b - rating_b
            
            # Calculate what A's rating should be to maintain zero-sum
            if winner.upper() == 'B':
                # If B won but hit ceiling, A loses less
                rating_loss_a = actual_gain_b
                new_rating_a = rating_a - rating_loss_a
    
    return round(new_rating_a), round(new_rating_b)


def should_use_rank_ceiling() -> bool:
    """Check if rank ceiling should be applied based on current date"""
    current_time = datetime.datetime.now(RANK_CEILING_START_DATE.tzinfo)
    return current_time >= RANK_CEILING_START_DATE


def calculate_elo_update_for_streaming(rating_a: float, rating_b: float, winner: str,
                                     player_a_id: str, player_b_id: str,
                                     supabase_client: Client, k: int = 32) -> Tuple[int, int, bool]:
    """
    High-level function to calculate ELO update for streaming (real-time) processing.
    
    Returns:
        tuple[int, int, bool]: (new_elo_a, new_elo_b, ceiling_applied)
    """
    use_rank_ceiling = should_use_rank_ceiling()
    
    if use_rank_ceiling:
        # Get current ELOs for all players for rank ceiling calculation
        all_players_response = supabase_client.table("players").select("id, elo").execute()
        current_elos = {p['id']: p['elo'] for p in all_players_response.data}
        
        # Get or create daily snapshot for rank ceiling calculations
        snapshot_elos = get_or_create_daily_snapshot(supabase_client, current_elos)
        
        # Get inactive player dates for rank ceiling calculation (DB version)
        inactive_player_dates = get_inactive_player_last_match_dates_from_db(supabase_client)
        
        # Use current time as match date for real-time processing
        match_date = datetime.datetime.now(RANK_CEILING_START_DATE.tzinfo)
        
        # Calculate ELO with rank ceiling using snapshot
        new_elo_a, new_elo_b = update_elo_with_ceiling(
            rating_a, rating_b, winner,
            player_a_id, player_b_id, 
            current_elos, inactive_player_dates, match_date, snapshot_elos, k
        )
        
        # Check if ceiling was applied
        normal_elo_a, normal_elo_b = update_elo(rating_a, rating_b, winner, k)
        ceiling_applied = new_elo_a != normal_elo_a or new_elo_b != normal_elo_b
        
        return new_elo_a, new_elo_b, ceiling_applied
    else:
        # Use normal ELO calculation (pre-ceiling implementation)
        new_elo_a, new_elo_b = update_elo(rating_a, rating_b, winner, k)
        return new_elo_a, new_elo_b, False


def calculate_elo_update_for_batch(rating_a: float, rating_b: float, winner: str,
                                  player_a_id: str, player_b_id: str,
                                  current_elos: Dict[str, int], 
                                  inactive_player_dates: Dict[str, datetime.datetime],
                                  match_date: datetime.datetime,
                                  snapshot_elos: Optional[Dict[str, int]] = None,
                                  k: int = 32) -> Tuple[int, int, bool]:
    """
    High-level function to calculate ELO update for batch processing with pandas DataFrames.
    
    Returns:
        tuple[int, int, bool]: (new_elo_a, new_elo_b, ceiling_applied)
    """
    use_rank_ceiling = match_date >= RANK_CEILING_START_DATE
    
    if use_rank_ceiling:
        # Calculate ELO with rank ceiling
        new_elo_a, new_elo_b = update_elo_with_ceiling(
            rating_a, rating_b, winner,
            player_a_id, player_b_id, 
            current_elos, inactive_player_dates, match_date, snapshot_elos, k
        )
        
        # Check if ceiling was applied
        normal_elo_a, normal_elo_b = update_elo(rating_a, rating_b, winner, k)
        ceiling_applied = new_elo_a != normal_elo_a or new_elo_b != normal_elo_b
        
        return new_elo_a, new_elo_b, ceiling_applied
    else:
        # Use normal ELO calculation (pre-ceiling implementation)
        new_elo_a, new_elo_b = update_elo(rating_a, rating_b, winner, k)
        return new_elo_a, new_elo_b, False