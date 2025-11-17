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
from elo_utils import (
    update_elo, get_inactive_player_last_match_dates_from_dataframes
)

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
    """Get all matches ordered by created_at (exclude archived matches)"""
    try:
        all_matches = []
        page_size = 1000
        start = 0
        
        while True:
            response = (supabase_client.table("matches")
                       .select("*")
                       .eq("archived", False)
                       .order("created_at", desc=False)
                       .range(start, start + page_size - 1)
                       .execute())
            
            if not response.data:
                break
                
            all_matches.extend(response.data)
            
            if len(response.data) < page_size:
                break
                
            start += page_size
        
        return all_matches
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
    
    # Fetch all matches with pagination (exclude archived matches)
    all_matches = []
    page_size = 1000
    start = 0
    
    while True:
        matches_response = (supabase_client.table("matches")
                           .select("*")
                           .eq("archived", False)
                           .order("created_at", desc=False)
                           .range(start, start + page_size - 1)
                           .execute())
        
        if not matches_response.data:
            break
            
        all_matches.extend(matches_response.data)
        
        if len(matches_response.data) < page_size:
            break
            
        start += page_size
    
    matches_df = pd.DataFrame(all_matches)
    
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
    """Second pass: Calculate ELOs only for matches between ranked players"""
    
    # Get ranked players (top_ten_played_new >= 3)
    ranked_player_ids = set(players_updated[players_updated['top_ten_played_new'] >= 3]['id'].tolist())
    
    print(f"    Total ranked players: {len(ranked_player_ids)}")
    
    # Initialize ELOs to 1200 for all players
    current_elos = {player_id: 1200 for player_id in players_updated['id']}
    
    # Filter 1v1 matches only
    match_participant_counts = participants_df.groupby('match_id').size()
    valid_matches = match_participant_counts[match_participant_counts == 2].index
    
    print(f"    Total matches fetched: {len(matches_df)}")
    print(f"    1v1 matches: {len(valid_matches)}")
    
    # Sort matches chronologically
    valid_matches_df = matches_df[matches_df['id'].isin(valid_matches)].sort_values('created_at')
    
    elo_updates = 0
    processed = 0
    no_clear_winner = 0
    not_both_ranked = 0
    
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
            
            # Use normal ELO calculation
            new_elo1, new_elo2 = update_elo(elo1, elo2, winner)
            
            # Update ELOs
            current_elos[player1_id] = new_elo1
            current_elos[player2_id] = new_elo2
            
            elo_updates += 1
        else:
            not_both_ranked += 1
        
        processed += 1
    
    print(f"    Processed {processed} matches")
    print(f"    ELO updates applied: {elo_updates}")
    print(f"    Matches skipped (not both ranked): {not_both_ranked}")
    print(f"    Matches skipped (no clear winner): {no_clear_winner}")
    
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
    
    # Get all matches with pagination (exclude archived matches)
    all_matches = []
    page_size = 1000
    start = 0
    
    while True:
        matches_response = (supabase_client.table("matches")
                           .select("*")
                           .eq("archived", False)
                           .order("created_at", desc=False)
                           .range(start, start + page_size - 1)
                           .execute())
        
        if not matches_response.data:
            break
            
        all_matches.extend(matches_response.data)
        
        if len(matches_response.data) < page_size:
            break
            
        start += page_size
    
    matches = all_matches
    
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

def recompute_all_player_elos():
    """Main function to recompute all player ELO ratings using pandas"""
    
    print("="*70)
    print("RECOMPUTING ALL PLAYER ELO RATINGS")
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
    
    # Step 7: Print final rankings (exclude 1200 ELO unranked players)
    print("\n" + "="*60)
    print("FINAL ELO RANKINGS (WITH RANK CEILING)")
    print("="*60)
    
    # Sort by final ELO and filter out 1200 ELO players
    final_rankings_ceiling = players_final_ceiling.sort_values('elo_final', ascending=False)
    ranked_players = final_rankings_ceiling[final_rankings_ceiling['elo_final'] != 1200]
    
    for rank, (_, player) in enumerate(ranked_players.iterrows(), 1):
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
