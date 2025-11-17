#!/usr/bin/env python3
"""
Advanced ELO System Manager for Smash Bros Leaderboard

This script provides utilities for managing the advanced ELO system including:
- Running inactivity penalties
- Showing current rankings
- Analyzing player matchup records
- Viewing daily head-to-head statistics
"""

import argparse
import datetime
from capture_card_processor import AdvancedEloSystem, supabase_client
from elo_utils import update_inactivity_status

def show_rankings(limit: int = 20):
    """Show current player rankings"""
    if not supabase_client:
        print("Error: Supabase client not available")
        return
    
    try:
        response = (
            supabase_client.table("players")
            .select("*")
            .order("elo", desc=True)
            .limit(limit)
            .execute()
        )
        
        if not response.data:
            print("No players found in database")
            return
        
        print(f"\n{'='*60}")
        print("CURRENT SMASH BROS LEADERBOARD")
        print(f"{'='*60}")
        print(f"{'Rank':<4} {'Player':<20} {'ELO':<6} {'Last Active':<12}")
        print("-" * 60)
        
        for i, player in enumerate(response.data, 1):
            # Get last activity
            last_match_response = (
                supabase_client.table("match_participants")
                .select("*, matches!inner(created_at)")
                .eq("player", player['id'])
                .order("matches.created_at", desc=True)
                .limit(1)
                .execute()
            )
            
            if last_match_response.data:
                last_active = datetime.datetime.fromisoformat(
                    last_match_response.data[0]['matches']['created_at']
                ).strftime("%Y-%m-%d")
            else:
                last_active = "Never"
            
            print(f"{i:<4} {player['name']:<20} {player['elo']:<6} {last_active:<12}")
        
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error showing rankings: {e}")

def show_player_stats(player_name: str):
    """Show detailed stats for a specific player"""
    if not supabase_client:
        print("Error: Supabase client not available")
        return
    
    try:
        # Get player
        player_response = (
            supabase_client.table("players")
            .select("*")
            .eq("name", player_name)
            .execute()
        )
        
        if not player_response.data:
            print(f"Player '{player_name}' not found")
            return
        
        player = player_response.data[0]
        player_id = player['id']
        
        # Initialize ELO system
        elo_system = AdvancedEloSystem(supabase_client)
        
        # Get beaten history
        beaten_history = elo_system.get_player_beaten_history(player_id)
        
        # Get all opponents
        all_opponents_response = (
            supabase_client.table("match_participants")
            .select("*, matches!inner(*)")
            .eq("player", player_id)
            .execute()
        )
        
        # Collect unique opponents
        opponents = set()
        for match_data in all_opponents_response.data:
            match_id = match_data['match_id']
            
            # Get other participants in same match
            other_participants_response = (
                supabase_client.table("match_participants")
                .select("*, players!inner(id, name, elo)")
                .eq("match_id", match_id)
                .neq("player", player_id)
                .execute()
            )
            
            for opponent_data in other_participants_response.data:
                opponents.add((opponent_data['player'], opponent_data['players']['name']))
        
        print(f"\n{'='*60}")
        print(f"PLAYER STATS: {player['name']}")
        print(f"{'='*60}")
        print(f"Current ELO: {player['elo']}")
        print(f"Players beaten: {len(beaten_history['beaten'])}")
        print(f"Beaten by: {len(beaten_history['beaten_by'])}")
        print(f"Total opponents: {len(opponents)}")
        
        if opponents:
            print(f"\nHead-to-Head Records:")
            print("-" * 40)
            
            for opponent_id, opponent_name in sorted(opponents, key=lambda x: x[1]):
                all_time_record = elo_system.get_all_time_matchup_record(player_id, opponent_id)
                daily_record = elo_system.get_daily_head_to_head_record(player_id, opponent_id)
                
                wins = all_time_record['player1_wins']
                losses = all_time_record['player2_wins']
                total = all_time_record['total_games']
                
                today_lead = daily_record['game_lead']
                today_status = f" (Today: {today_lead:+d})" if today_lead != 0 else " (Today: 0)"
                
                if total > 0:
                    win_rate = (wins / total) * 100
                    print(f"  vs {opponent_name:<15}: {wins}-{losses} ({win_rate:.1f}%){today_status}")
        
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error showing player stats: {e}")

def run_inactivity_penalties():
    """Run the inactivity penalty system"""
    if not supabase_client:
        print("Error: Supabase client not available")
        return
    
    elo_system = AdvancedEloSystem(supabase_client)
    elo_system.apply_inactivity_penalties()

def show_daily_stats(date: str = None):
    """Show daily matchup statistics"""
    if date is None:
        date = datetime.date.today().isoformat()
    
    if not supabase_client:
        print("Error: Supabase client not available")
        return
    
    try:
        # Get all matches for the date (exclude archived matches)
        response = (
            supabase_client.table("matches")
            .select("*, match_participants!inner(*, players!inner(name))")
            .eq("archived", False)
            .gte("created_at", date)
            .lt("created_at", (datetime.datetime.fromisoformat(date) + datetime.timedelta(days=1)).isoformat())
            .execute()
        )
        
        if not response.data:
            print(f"No matches found for {date}")
            return
        
        print(f"\n{'='*60}")
        print(f"DAILY STATS FOR {date}")
        print(f"{'='*60}")
        print(f"Total matches: {len(response.data)}")
        
        # Count unique players
        players_today = set()
        for match in response.data:
            for participant in match['match_participants']:
                players_today.add(participant['players']['name'])
        
        print(f"Unique players: {len(players_today)}")
        
        # Show matches
        print(f"\nMatches:")
        print("-" * 40)
        
        for match in response.data:
            participants = match['match_participants']
            if len(participants) == 2:  # 1v1 matches
                p1 = participants[0]
                p2 = participants[1]
                
                p1_name = p1['players']['name']
                p2_name = p2['players']['name']
                
                winner = p1_name if p1['has_won'] else p2_name if p2['has_won'] else "No Contest"
                
                match_time = datetime.datetime.fromisoformat(match['created_at']).strftime("%H:%M")
                print(f"  {match_time}: {p1_name} vs {p2_name} - Winner: {winner}")
        
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error showing daily stats: {e}")

def main():
    parser = argparse.ArgumentParser(description='Advanced ELO System Manager')
    parser.add_argument('--rankings', '-r', action='store_true', help='Show current rankings')
    parser.add_argument('--limit', type=int, default=20, help='Limit number of players to show in rankings (default: 20)')
    parser.add_argument('--player-stats', '-p', type=str, help='Show detailed stats for a specific player')
    parser.add_argument('--inactivity-penalties', '-i', action='store_true', help='Run inactivity penalty system')
    parser.add_argument('--update-inactivity', '-u', nargs='?', const=4, type=int, metavar='WEEKS', help='Update player inactivity status (default: 4 weeks threshold)')
    parser.add_argument('--daily-stats', '-d', type=str, help='Show daily stats for a specific date (YYYY-MM-DD format, default: today)')
    parser.add_argument('--today', action='store_true', help='Show today\'s stats')
    
    args = parser.parse_args()
    
    if args.rankings:
        show_rankings(args.limit)
    elif args.player_stats:
        show_player_stats(args.player_stats)
    elif args.inactivity_penalties:
        run_inactivity_penalties()
    elif args.update_inactivity is not None:
        # args.update_inactivity will be 4 (const) if flag provided without value, or the provided value
        weeks = args.update_inactivity
        print(f"Updating inactivity status (threshold: {weeks} weeks)...")
        if not supabase_client:
            print("Error: Supabase client not available")
            return
        update_inactivity_status(supabase_client, inactivity_threshold_weeks=weeks)
    elif args.daily_stats:
        show_daily_stats(args.daily_stats)
    elif args.today:
        show_daily_stats()
    else:
        print("No action specified. Use --help for available options.")
        print("\nQuick examples:")
        print("  python elo_manager.py --rankings")
        print("  python elo_manager.py --player-stats 'PlayerName'")
        print("  python elo_manager.py --inactivity-penalties")
        print("  python elo_manager.py --update-inactivity [WEEKS]")
        print("  python elo_manager.py --today")

if __name__ == "__main__":
    main() 