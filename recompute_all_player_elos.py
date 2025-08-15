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
from typing import Dict, List, Optional
from supabase import create_client, Client
from dotenv import load_dotenv

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
                'name': player['name'],
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

def recompute_all_player_elos():
    """Main function to recompute all player ELO ratings"""
    
    print("="*60)
    print("RECOMPUTING ALL PLAYER ELO RATINGS")
    print("="*60)
    
    # Step 1: Get all players and reset their ELOs to 1200, get original top 10
    print("\n1. Loading players and resetting ELOs to 1200...")
    players, original_top_ten_ids = get_all_players()
    
    if not players:
        print("No players found in database!")
        return
    
    print(f"Original top 10 players (before recomputation): {len(original_top_ten_ids)} players")
    
    # Debug: Print the original top 10 player names
    original_top_ten_names = []
    for player_id in original_top_ten_ids:
        if player_id in players:
            original_top_ten_names.append(players[player_id]['name'])
    print(f"Original top 10: {', '.join(original_top_ten_names)}")
    
    print(f"Found {len(players)} players:")
    for player in players.values():
        print(f"  - {player['name']} (ID: {player['id']}) -> ELO: {player['elo']}")
    
    # Step 2: Get all matches in chronological order
    print("\n2. Loading matches in chronological order...")
    matches = get_all_matches_chronological()
    
    if not matches:
        print("No matches found in database!")
        return
    
    print(f"Found {len(matches)} matches to process")
    
    # Step 3: Process each match and update ELOs
    print("\n3. Processing matches and updating ELOs...")
    processed_matches = 0
    elo_updated_matches = 0
    
    for match in matches:
        match_id = match['id']
        created_at = match.get('created_at', 'Unknown')
        
        # Get participants for this match
        participants = get_match_participants(match_id)
        
        if len(participants) != 2:
            print(f"  Match {match_id} ({created_at}): Skipping (not 1v1, has {len(participants)} participants)")
            processed_matches += 1
            continue
        
        # Check if both participants are valid players and one has won
        player1 = None
        player2 = None
        winner_participant = None
        
        for participant in participants:
            player_id = participant['player']
            if player_id in players:
                if player1 is None:
                    player1 = participant
                else:
                    player2 = participant
                
                if participant['has_won']:
                    winner_participant = participant
        
        if not player1 or not player2:
            print(f"  Match {match_id} ({created_at}): Skipping (missing players)")
            processed_matches += 1
            continue
        
        if not winner_participant:
            print(f"  Match {match_id} ({created_at}): Skipping (no winner - no contest)")
            processed_matches += 1
            continue
        
        # Check if both players are ranked (top_ten_played >= 3)
        player1_id = player1['player']
        player2_id = player2['player']
        player1_ranked = is_player_ranked(players[player1_id])
        player2_ranked = is_player_ranked(players[player2_id])
        
        # Always update top_ten_played counters regardless of ranking status
        update_top_ten_played(player1_id, player2_id, players, original_top_ten_ids)
        
        if not (player1_ranked and player2_ranked):
            unranked_players = []
            if not player1_ranked:
                unranked_players.append(players[player1_id]['name'])
            if not player2_ranked:
                unranked_players.append(players[player2_id]['name'])
            print(f"  Match {match_id} ({created_at}): Skipping ELO update (unranked player(s): {', '.join(unranked_players)})")
            processed_matches += 1
            continue
        
        # Get current ELOs
        player1_name = players[player1_id]['name']
        player2_name = players[player2_id]['name']
        player1_current_elo = players[player1_id]['elo']
        player2_current_elo = players[player2_id]['elo']
        
        # Determine winner ('A' for player1, 'B' for player2)
        winner = 'A' if player1['has_won'] else 'B'
        winner_name = player1_name if winner == 'A' else player2_name
        
        # Calculate new ELOs
        new_elo_1, new_elo_2 = update_elo(player1_current_elo, player2_current_elo, winner)
        
        # Update ELOs in memory
        players[player1_id]['elo'] = new_elo_1
        players[player2_id]['elo'] = new_elo_2
        
        # Calculate changes
        elo_change_1 = new_elo_1 - player1_current_elo
        elo_change_2 = new_elo_2 - player2_current_elo
        
        print(f"  Match {match_id} ({created_at}): {winner_name} wins!")
        print(f"    {player1_name}: {player1_current_elo} → {new_elo_1} ({elo_change_1:+d})")
        print(f"    {player2_name}: {player2_current_elo} → {new_elo_2} ({elo_change_2:+d})")
        
        processed_matches += 1
        elo_updated_matches += 1
    
    print(f"\n4. Match processing complete:")
    print(f"   Total matches processed: {processed_matches}")
    print(f"   ELO updates applied: {elo_updated_matches}")
    
    # Step 4: Update all players' ELOs and top_ten_played in the database
    print("\n5. Updating final ELO ratings and top_ten_played in database...")
    
    for player in players.values():
        try:
            # Calculate final top_ten_played count from the set of unique players faced
            final_top_ten_played = len(player['top_ten_faced_set'])
            update_player_stats_in_db(player['id'], player['elo'], final_top_ten_played)
            print(f"  Updated {player['name']}: ELO = {player['elo']}, top_ten_played = {final_top_ten_played}")
        except Exception as e:
            print(f"  Failed to update {player['name']}: {e}")
    
    # Step 5: Print final rankings
    print("\n" + "="*60)
    print("FINAL ELO RANKINGS")
    print("="*60)
    
    # Sort players by ELO (highest first)
    sorted_players = sorted(players.values(), key=lambda p: p['elo'], reverse=True)
    
    for rank, player in enumerate(sorted_players, 1):
        print(f"{rank:2d}. {player['name']:<20} - {player['elo']:4d} ELO")
    
    print("="*60)
    print("ELO RECOMPUTATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    try:
        recompute_all_player_elos()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
