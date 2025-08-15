#!/usr/bin/env python3
"""
Script to recompute all player ELO ratings from scratch based on historical match data.

This script:
1. Resets all player ELOs to the initial value (1200)
2. Processes all matches in chronological order (by created_at)
3. Recomputes ELO ratings for 1v1 matches only
4. Updates all players' final ELO values in the database
"""

import math
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime, timezone
import pytz

# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client (using environment variables)
try:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not found in environment variables")
    
    supabase_client: Client = create_client(supabase_url, supabase_key)
    print("Successfully connected to Supabase")
except Exception as e:
    print(f"Error: Failed to initialize Supabase client: {e}")
    exit(1)

# Rank ceiling implementation date - only apply ceiling after this date (California time)
pacific_tz = pytz.timezone('America/Los_Angeles')
RANK_CEILING_START_DATE = pacific_tz.localize(datetime(2025, 8, 15, 0, 0, 0))

def get_player_ranking(player_id: str, current_elos: Dict[str, int]) -> int:
    """Get the current ranking of a player based on ELO"""
    sorted_players = sorted(current_elos.items(), key=lambda x: x[1], reverse=True)
    for rank, (pid, elo) in enumerate(sorted_players, 1):
        if pid == player_id:
            return rank
    return len(sorted_players) + 1

def get_elo_ceiling(player_id: str, opponent_id: str, current_elos: Dict[str, int]) -> int:
    """
    Calculate the ELO ceiling for a player based on rank ceiling rules.
    
    Rules:
    - A player cannot surpass someone ranked higher unless they beat that person
    - If playing against someone ranked lower, max ELO gain stops 1 point shy of next higher player
    - If playing against someone ranked higher, can gain full ELO (no ceiling)
    """
    player_rank = get_player_ranking(player_id, current_elos)
    opponent_rank = get_player_ranking(opponent_id, current_elos)
    
    # If opponent is ranked higher or equal, no ceiling applies (can gain full ELO)
    if opponent_rank <= player_rank:
        return float('inf')
    
    # Opponent is ranked lower, so apply ceiling
    # Find the ELO of the player ranked immediately above the current player
    sorted_players = sorted(current_elos.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (pid, elo) in enumerate(sorted_players, 1):
        if rank == player_rank - 1:  # Player ranked immediately above
            return elo - 1  # Ceiling is 1 point below their ELO
    
    # If no one is ranked above (player is #1), no ceiling
    return float('inf')

def update_elo(rating_a: float,
               rating_b: float,
               winner: str,
               k: int = 32) -> tuple[int, int]:
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

    >>> update_elo(1400, 1000, 'A')
    (1403, 997)
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

def update_elo_with_ceiling(rating_a: float,
                           rating_b: float,
                           winner: str,
                           player_a_id: str,
                           player_b_id: str,
                           current_elos: Dict[str, int],
                           inactive_player_dates: Dict[str, datetime],
                           match_date: datetime,
                           k: int = 32) -> tuple[int, int]:
    """
    Return the new (rating_a, rating_b) after one game with rank ceiling applied.
    Excludes players who became inactive before this match date.

    Parameters
    ----------
    rating_a : current Elo for Player A
    rating_b : current Elo for Player B
    winner   : 'A', 'B', or 'draw'
    player_a_id : ID of Player A
    player_b_id : ID of Player B
    current_elos : Dict of all current player ELOs
    inactive_player_dates : Dict mapping inactive player IDs to their last match dates
    match_date : Date of the current match being processed
    k        : K-factor (default 32)

    Returns
    -------
    tuple[int, int] : New ratings as integers for (Player A, Player B)
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
    
    # Apply ceiling for Player A if they gained ELO (using active players only)
    if new_rating_a > rating_a:
        ceiling_a = get_elo_ceiling(player_a_id, player_b_id, active_elos)
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
        ceiling_b = get_elo_ceiling(player_b_id, player_a_id, active_elos)
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

def get_all_players() -> tuple[Dict[str, Dict], set]:
    """Get all players from the database and return original top 10"""
    try:
        response = supabase_client.table("players").select("*").execute()
        players = {}
        
        # First, build the full player data and collect original data
        original_players_with_ranking = []
        for player in response.data:
            players[player['id']] = {
                'id': player['id'],
                'name': player['display_name'],
                'elo': 1200,  # Reset to initial ELO
                'top_ten_played': player.get('top_ten_played', 0),  # Original value for ranking check
                'top_ten_faced_set': set()  # Track unique top 10 players faced
            }
            original_players_with_ranking.append({
                'id': player['id'],
                'elo': player.get('elo', 1200),  # Original ELO from database
                'top_ten_played': player.get('top_ten_played', 0)  # Original ranking status
            })
        
        # Get original top 10: first filter by ranked players (top_ten_played >= 3), then sort by ELO
        ranked_players = [p for p in original_players_with_ranking if p['top_ten_played'] >= 3]
        original_top_ten = sorted(ranked_players, key=lambda p: p['elo'], reverse=True)[:10]
        original_top_ten_ids = {player['id'] for player in original_top_ten}
        
        return players, original_top_ten_ids
    except Exception as e:
        print(f"Error fetching players: {e}")
        return {}, set()

def is_player_ranked(player_data: Dict) -> bool:
    """Check if a player is ranked (has top_ten_played >= 3)"""
    return player_data.get('top_ten_played', 0) >= 3

def get_current_top_ten_player_ids(players: Dict[str, Dict]) -> set:
    """Get the IDs of the current top 10 players by ELO"""
    sorted_players = sorted(players.values(), key=lambda p: p['elo'], reverse=True)
    return {player['id'] for player in sorted_players[:10]}

def update_top_ten_played(player1_id: str, player2_id: str, players: Dict[str, Dict], original_top_ten_ids: set):
    """Update top_ten_faced_set if a player faces someone from the original top 10"""
    
    # If player1 faces someone from original top 10, add them to player1's set
    if player2_id in original_top_ten_ids:
        players[player1_id]['top_ten_faced_set'].add(player2_id)
        print(f"    DEBUG: {players[player1_id]['name']} faced top 10 player {players[player2_id]['name']}")
    
    # If player2 faces someone from original top 10, add them to player2's set
    if player1_id in original_top_ten_ids:
        players[player2_id]['top_ten_faced_set'].add(player1_id)
        print(f"    DEBUG: {players[player2_id]['name']} faced top 10 player {players[player1_id]['name']}")

def get_all_matches_chronological() -> List[Dict]:
    """Get all matches ordered by created_at"""
    try:
        response = (
            supabase_client.table("matches")
            .select("*")
            .order("created_at", desc=False)
            .execute()
        )
        return response.data
    except Exception as e:
        print(f"Error fetching matches: {e}")
        return []

def get_match_participants(match_id: int) -> List[Dict]:
    """Get all participants for a specific match"""
    try:
        response = (
            supabase_client.table("match_participants")
            .select("*")
            .eq("match_id", match_id)
            .execute()
        )
        return response.data
    except Exception as e:
        print(f"Error fetching participants for match {match_id}: {e}")
        return []

def update_player_elo_in_db(player_id: str, elo: int):
    """Update a player's ELO in the database"""
    try:
        supabase_client.table("players").update({"elo": elo}).eq("id", player_id).execute()
    except Exception as e:
        print(f"Error updating ELO for player {player_id}: {e}")

def update_player_stats_in_db(player_id: str, elo: int, top_ten_played: int):
    """Update a player's ELO and top_ten_played in the database"""
    try:
        supabase_client.table("players").update({
            "elo": elo,
            "top_ten_played": top_ten_played
        }).eq("id", player_id).execute()
    except Exception as e:
        print(f"Error updating stats for player {player_id}: {e}")

def fetch_all_data_pandas() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch all data and return as pandas DataFrames"""
    print("Fetching all data...")
    
    # Fetch players
    players_response = supabase_client.table("players").select("*").execute()
    players_df = pd.DataFrame(players_response.data)
    
    # Fetch matches
    matches_response = supabase_client.table("matches").select("*").order("created_at", desc=False).execute()
    matches_df = pd.DataFrame(matches_response.data)
    
    # Fetch all match participants with pagination
    all_participants = []
    page_size = 1000
    start = 0
    
    while True:
        participants_response = (supabase_client.table("match_participants")
                               .select("*")
                               .range(start, start + page_size - 1)
                               .execute())
        
        if not participants_response.data:
            break
            
        all_participants.extend(participants_response.data)
        
        if len(participants_response.data) < page_size:
            break
            
        start += page_size
    
    participants_df = pd.DataFrame(all_participants)
    
    print(f"Loaded {len(players_df)} players, {len(matches_df)} matches, {len(participants_df)} participants")
    
    return players_df, matches_df, participants_df

def get_match_participants(match_id: int) -> List[Dict]:
    """Get all participants for a specific match (old method)"""
    try:
        response = (
            supabase_client.table("match_participants")
            .select("*")
            .eq("match_id", match_id)
            .execute()
        )
        return response.data
    except Exception as e:
        print(f"Error fetching participants for match {match_id}: {e}")
        return []

def get_original_top_ten_pandas(players_df: pd.DataFrame) -> set:
    """Get original top 10 player IDs based on database ELOs"""
    # Filter ranked players (top_ten_played >= 3)
    ranked_players = players_df[players_df['top_ten_played'] >= 3].copy()
    
    # Sort by ELO and take top 10
    original_top_ten = ranked_players.nlargest(10, 'elo')
    original_top_ten_ids = set(original_top_ten['id'].tolist())
    
    return original_top_ten_ids

def calculate_top_ten_played_pandas(matches_df: pd.DataFrame, participants_df: pd.DataFrame, 
                                   players_df: pd.DataFrame, original_top_ten_ids: set) -> pd.DataFrame:
    """First pass: Calculate top_ten_played for all players using pandas"""
    
    # Filter 1v1 matches only
    match_participant_counts = participants_df.groupby('match_id').size()
    valid_matches = match_participant_counts[match_participant_counts == 2].index
    valid_participants = participants_df[participants_df['match_id'].isin(valid_matches)].copy()
    
    # Create pairs of players for each match
    match_pairs = []
    for match_id in valid_matches:
        match_participants = valid_participants[valid_participants['match_id'] == match_id]
        if len(match_participants) == 2:
            players = match_participants['player'].tolist()
            match_pairs.extend([
                {'player': players[0], 'opponent': players[1], 'match_id': match_id},
                {'player': players[1], 'opponent': players[0], 'match_id': match_id}
            ])
    
    pairs_df = pd.DataFrame(match_pairs)
    
    # Filter pairs where opponent is in original top 10
    pairs_df['opponent_is_top_ten'] = pairs_df['opponent'].isin(original_top_ten_ids)
    top_ten_faced = pairs_df[pairs_df['opponent_is_top_ten']].copy()
    
    # Count unique top 10 players each player has faced
    top_ten_counts = (top_ten_faced.groupby('player')['opponent']
                     .nunique()
                     .reset_index()
                     .rename(columns={'opponent': 'top_ten_played_new'}))
    
    # Merge with players dataframe
    players_updated = players_df.merge(top_ten_counts, left_on='id', right_on='player', how='left')
    players_updated['top_ten_played_new'] = players_updated['top_ten_played_new'].fillna(0).astype(int)
    
    return players_updated

def get_inactive_player_last_match_dates(players_df: pd.DataFrame, matches_df: pd.DataFrame, 
                                       participants_df: pd.DataFrame) -> Dict[str, datetime]:
    """Pre-compute last match dates for all inactive players"""
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

def calculate_elos_pandas(matches_df: pd.DataFrame, participants_df: pd.DataFrame, 
                         players_updated: pd.DataFrame) -> pd.DataFrame:
    """Second pass: Calculate ELOs only for matches between ranked players with rank ceiling after Aug 15, 2025"""
    
    # Pre-compute last match dates for inactive players (optimization)
    inactive_player_dates = get_inactive_player_last_match_dates(players_updated, matches_df, participants_df)
    
    # Get ranked players (top_ten_played_new >= 3)
    ranked_player_ids = set(players_updated[players_updated['top_ten_played_new'] >= 3]['id'].tolist())
    
    # Initialize ELOs to 1200 for all players
    current_elos = {player_id: 1200 for player_id in players_updated['id']}
    
    # Filter 1v1 matches only
    match_participant_counts = participants_df.groupby('match_id').size()
    valid_matches = match_participant_counts[match_participant_counts == 2].index
    
    # Sort matches chronologically
    valid_matches_df = matches_df[matches_df['id'].isin(valid_matches)].sort_values('created_at')
    
    elo_updates = 0
    ceiling_applied = 0
    processed = 0
    
    for _, match in valid_matches_df.iterrows():
        match_id = match['id']
        match_participants = participants_df[participants_df['match_id'] == match_id]
        
        if len(match_participants) != 2:
            continue
            
        players = match_participants['player'].tolist()
        winners = match_participants[match_participants['has_won'] == True]['player'].tolist()
        
        if len(winners) != 1:  # Skip if no clear winner
            processed += 1
            continue
            
        player1_id, player2_id = players[0], players[1]
        
        # Only calculate ELO if both players are ranked
        if player1_id in ranked_player_ids and player2_id in ranked_player_ids:
            winner_id = winners[0]
            winner = 'A' if winner_id == player1_id else 'B'
            
            # Get current ELOs
            elo1 = current_elos[player1_id]
            elo2 = current_elos[player2_id]
            
            # Check if match is after rank ceiling implementation date
            match_date_str = match['created_at']
            if match_date_str.endswith('Z'):
                match_date_str = match_date_str.replace('Z', '+00:00')
            elif '+' not in match_date_str and not match_date_str.endswith('+00:00'):
                match_date_str += '+00:00'
            
            match_date = datetime.fromisoformat(match_date_str)
            # Ensure match_date is timezone-aware (UTC)
            if match_date.tzinfo is None:
                match_date = match_date.replace(tzinfo=timezone.utc)
            use_rank_ceiling = match_date >= RANK_CEILING_START_DATE
            
            if use_rank_ceiling:
                # Calculate new ELOs with rank ceiling
                new_elo1, new_elo2 = update_elo_with_ceiling(
                    elo1, elo2, winner, player1_id, player2_id, current_elos, 
                    inactive_player_dates, match_date
                )
                
                # Check if ceiling was applied
                normal_elo1, normal_elo2 = update_elo(elo1, elo2, winner)
                if new_elo1 != normal_elo1 or new_elo2 != normal_elo2:
                    ceiling_applied += 1
                    # Get player names for debugging
                    p1_name = players_updated[players_updated['id'] == player1_id]['name'].iloc[0]
                    p2_name = players_updated[players_updated['id'] == player2_id]['name'].iloc[0]
                    match_date_str = match_date.strftime('%Y-%m-%d')
                    print(f"    Ceiling applied ({match_date_str}): {p1_name} vs {p2_name} - Normal: ({normal_elo1}, {normal_elo2}) -> Ceiling: ({new_elo1}, {new_elo2})")
            else:
                # Calculate new ELOs without ceiling (pre-implementation)
                new_elo1, new_elo2 = update_elo(elo1, elo2, winner)
            
            # Update ELOs
            current_elos[player1_id] = new_elo1
            current_elos[player2_id] = new_elo2
            
            elo_updates += 1
        
        processed += 1
    
    print(f"    Processed {processed} matches, {elo_updates} ELO updates, {ceiling_applied} ceiling applications (post Aug 15, 2025)")
    
    # Update players dataframe with final ELOs
    players_final = players_updated.copy()
    players_final['elo_final'] = players_final['id'].map(current_elos)
    
    return players_final

def recompute_all_player_elos_old_method():
    """Old sequential method for comparison"""
    print("="*60)
    print("RECOMPUTING ALL PLAYER ELO RATINGS (OLD METHOD)")
    print("="*60)
    
    # Get all players and original top 10 (using old method)
    players_response = supabase_client.table("players").select("*").execute()
    players = {}
    original_players_with_ranking = []
    
    for player in players_response.data:
        players[player['id']] = {
            'id': player['id'],
            'name': player['name'],
            'elo': 1200,  # Reset to initial ELO
            'top_ten_played': player.get('top_ten_played', 0),
            'top_ten_faced_set': set()
        }
        original_players_with_ranking.append({
            'id': player['id'],
            'elo': player.get('elo', 1200),
            'top_ten_played': player.get('top_ten_played', 0)
        })
    
    # Get original top 10
    ranked_players = [p for p in original_players_with_ranking if p['top_ten_played'] >= 3]
    original_top_ten = sorted(ranked_players, key=lambda p: p['elo'], reverse=True)[:10]
    original_top_ten_ids = {player['id'] for player in original_top_ten}
    
    # Get matches
    matches_response = supabase_client.table("matches").select("*").order("created_at", desc=False).execute()
    matches = matches_response.data
    
    elo_updates = 0
    for match in matches:
        match_id = match['id']
        participants = get_match_participants(match_id)
        
        if len(participants) != 2:
            continue
            
        # Get players and winner
        player1, player2 = participants[0], participants[1]
        player1_id, player2_id = player1['player'], player2['player']
        
        if player1_id not in players or player2_id not in players:
            continue
            
        winner_participant = next((p for p in participants if p['has_won']), None)
        if not winner_participant:
            continue
            
        # Update top_ten_played (always)
        if player2_id in original_top_ten_ids:
            players[player1_id]['top_ten_faced_set'].add(player2_id)
        if player1_id in original_top_ten_ids:
            players[player2_id]['top_ten_faced_set'].add(player1_id)
            
        # Check if both ranked for ELO update
        player1_ranked = players[player1_id]['top_ten_played'] >= 3
        player2_ranked = players[player2_id]['top_ten_played'] >= 3
        
        if player1_ranked and player2_ranked:
            winner = 'A' if player1['has_won'] else 'B'
            elo1, elo2 = players[player1_id]['elo'], players[player2_id]['elo']
            new_elo1, new_elo2 = update_elo(elo1, elo2, winner)
            players[player1_id]['elo'] = new_elo1
            players[player2_id]['elo'] = new_elo2
            elo_updates += 1
    
    # Return final results
    results = {}
    for player in players.values():
        results[player['id']] = {
            'name': player['name'],
            'elo': player['elo'],
            'top_ten_played': len(player['top_ten_faced_set'])
        }
    
    return results

def calculate_elos_pandas_no_ceiling(matches_df: pd.DataFrame, participants_df: pd.DataFrame, 
                                    players_updated: pd.DataFrame) -> pd.DataFrame:
    """Calculate ELOs without rank ceiling for comparison"""
    
    # Get ranked players (top_ten_played_new >= 3)
    ranked_player_ids = set(players_updated[players_updated['top_ten_played_new'] >= 3]['id'].tolist())
    
    # Initialize ELOs to 1200 for all players
    current_elos = {player_id: 1200 for player_id in players_updated['id']}
    
    # Filter 1v1 matches only
    match_participant_counts = participants_df.groupby('match_id').size()
    valid_matches = match_participant_counts[match_participant_counts == 2].index
    
    # Sort matches chronologically
    valid_matches_df = matches_df[matches_df['id'].isin(valid_matches)].sort_values('created_at')
    
    elo_updates = 0
    processed = 0
    
    for _, match in valid_matches_df.iterrows():
        match_id = match['id']
        match_participants = participants_df[participants_df['match_id'] == match_id]
        
        if len(match_participants) != 2:
            continue
            
        players = match_participants['player'].tolist()
        winners = match_participants[match_participants['has_won'] == True]['player'].tolist()
        
        if len(winners) != 1:  # Skip if no clear winner
            processed += 1
            continue
            
        player1_id, player2_id = players[0], players[1]
        
        # Only calculate ELO if both players are ranked
        if player1_id in ranked_player_ids and player2_id in ranked_player_ids:
            winner_id = winners[0]
            winner = 'A' if winner_id == player1_id else 'B'
            
            # Get current ELOs
            elo1 = current_elos[player1_id]
            elo2 = current_elos[player2_id]
            
            # Calculate new ELOs without ceiling
            new_elo1, new_elo2 = update_elo(elo1, elo2, winner)
            
            # Update ELOs
            current_elos[player1_id] = new_elo1
            current_elos[player2_id] = new_elo2
            
            elo_updates += 1
        
        processed += 1
    
    # Update players dataframe with final ELOs
    players_final = players_updated.copy()
    players_final['elo_final'] = players_final['id'].map(current_elos)
    
    return players_final

def recompute_all_player_elos():
    """Main function to recompute all player ELO ratings using pandas with rank ceiling after Aug 15, 2025"""
    
    print("="*70)
    print("RECOMPUTING ALL PLAYER ELO RATINGS")
    print("(Rank ceiling applied only after August 15, 2025)")
    print("="*70)
    
    # Step 1: Fetch all data
    players_df, matches_df, participants_df = fetch_all_data_pandas()
    
    if len(players_df) == 0:
        print("No players found in database!")
        return
    
    # Step 2: Get original top 10 based on current database state
    original_top_ten_ids = get_original_top_ten_pandas(players_df)
    
    if len(original_top_ten_ids) == 0:
        print("No ranked players found in database!")
        return
    
    # Step 3: First pass - Calculate top_ten_played for all players
    players_with_top_ten = calculate_top_ten_played_pandas(matches_df, participants_df, players_df, original_top_ten_ids)
    
    # Step 4: Second pass - Calculate ELOs with rank ceiling
    print("\nCalculating ELOs with rank ceiling...")
    players_final_ceiling = calculate_elos_pandas(matches_df, participants_df, players_with_top_ten)
    
    # Step 5: Calculate ELOs without ceiling for comparison
    print("\nCalculating ELOs without rank ceiling for comparison...")
    players_final_no_ceiling = calculate_elos_pandas_no_ceiling(matches_df, participants_df, players_with_top_ten)
    
    # Step 6: Update database with ceiling-applied ELOs
    print("\nUpdating database with rank ceiling ELOs...")
    for _, player in players_final_ceiling.iterrows():
        try:
            update_player_stats_in_db(player['id'], int(player['elo_final']), int(player['top_ten_played_new']))
        except Exception as e:
            print(f"  Failed to update {player['name']}: {e}")
    
    # Step 7: Print comparison of rankings
    print("\n" + "="*80)
    print("RANK CEILING COMPARISON")
    print("="*80)
    
    # Sort both by final ELO
    final_rankings_ceiling = players_final_ceiling.sort_values('elo_final', ascending=False)
    
    # Create ranking comparison
    print(f"{'Rank':<4} {'Player':<20} {'With Ceiling':<12} {'Without Ceiling':<15} {'Difference':<10}")
    print("-" * 80)
    
    for rank, (_, player_ceiling) in enumerate(final_rankings_ceiling.iterrows(), 1):
        player_id = player_ceiling['id']
        player_no_ceiling = players_final_no_ceiling[players_final_no_ceiling['id'] == player_id].iloc[0]
        
        elo_ceiling = int(player_ceiling['elo_final'])
        elo_no_ceiling = int(player_no_ceiling['elo_final'])
        difference = elo_ceiling - elo_no_ceiling
        
        print(f"{rank:<4} {player_ceiling['display_name']:<20} {elo_ceiling:<12} {elo_no_ceiling:<15} {difference:+d}")
    
    print("\n" + "="*60)
    print("FINAL ELO RANKINGS (WITH RANK CEILING)")
    print("="*60)
    
    for rank, (_, player) in enumerate(final_rankings_ceiling.iterrows(), 1):
        print(f"{rank:2d}. {player['display_name']:<20} - {int(player['elo_final']):4d} ELO (top_ten_played: {int(player['top_ten_played_new'])})")
    
    print("="*60)
    print("ELO RECOMPUTATION WITH RANK CEILING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    try:
        recompute_all_player_elos()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
