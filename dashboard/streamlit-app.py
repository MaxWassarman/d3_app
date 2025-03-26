import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import polars as pl
import requests
from bs4 import BeautifulSoup
import typing

# Set page config
st.set_page_config(
    page_title="Oberlin Baseball Stats Viewer",
    layout="wide"
)

# Data directory - change to relative path for GitHub deployment
DATA_DIR = "data/"

def get_available_years():
    """Get available years from data files"""
    years = set()
    try:
        for filename in os.listdir(DATA_DIR):
            if "_batting_" in filename or "_pitching_" in filename:
                year = filename.split("_")[-1].split(".")[0]
                if year.isdigit():
                    years.add(int(year))
    except:
        # If directory doesn't exist or can't be read, use current year
        years.add(datetime.now().year)
    
    return sorted(list(years))

def load_data(year):
    """Load batting and pitching data for a given year"""
    batting_file = os.path.join(DATA_DIR, f"d3_batting_{year}.csv")
    pitching_file = os.path.join(DATA_DIR, f"d3_pitching_{year}.csv")
    
    batting_data = None
    pitching_data = None
    batting_teams = []
    pitching_teams = []
    
    # Try to load batting data
    try:
        if os.path.exists(batting_file):
            batting_data = pd.read_csv(batting_file)
            # Get list of teams from batting data
            batting_teams = sorted(batting_data['team_name'].unique())
    except Exception as e:
        st.error(f"Error loading batting data: {e}")
    
    # Try to load pitching data
    try:
        if os.path.exists(pitching_file):
            pitching_data = pd.read_csv(pitching_file)
            # Get list of teams from pitching data
            pitching_teams = sorted(pitching_data['team_name'].unique())
    except Exception as e:
        st.error(f"Error loading pitching data: {e}")
    
    # Combine team lists and remove duplicates
    all_teams = sorted(list(set(batting_teams) | set(pitching_teams)))
    
    # Get file modification times for sidebar info
    batting_time = None
    pitching_time = None
    
    if os.path.exists(batting_file):
        batting_time = datetime.fromtimestamp(os.path.getmtime(batting_file))
    
    if os.path.exists(pitching_file):
        pitching_time = datetime.fromtimestamp(os.path.getmtime(pitching_file))
    
    return batting_data, pitching_data, all_teams, batting_time, pitching_time


# Process batting data function
def process_batting_data(batting_data, team, year):
    """Process batting data for a specific team"""
    if batting_data is None:
        return None, 0
    
    team_batting = batting_data[batting_data['team_name'] == team].copy()
    if team_batting.empty:
        return None, 0
    
    # Filter out players with less than 1 AB
    team_batting = team_batting[team_batting['AB'] >= 1]
    
    # Load wOBA weights from CSV file
    weights_file = os.path.join(DATA_DIR, f"d3_weights_{year}.csv")

    if not os.path.exists(weights_file) and year > 2024:
        weights_file = os.path.join(DATA_DIR, "d3_weights_2024.csv")

    if os.path.exists(weights_file):
        try:
            weights_df = pd.read_csv(weights_file)
            woba_weights = {}
            for _, row in weights_df.iterrows():
                woba_weights[row['events']] = row['normalized_weight']
            
            # Calculate singles (1B) if not already present
            if '1B' not in team_batting.columns:
                team_batting['1B'] = team_batting['H'] - team_batting['2B'] - team_batting['3B'] - team_batting['HR']
            
            # Calculate wOBA
            # Default values in case some weights are missing
            default_weights = {
                'BB': 0.82, 'HBP': 0.85, '1B': 0.95, 
                '2B': 1.27, '3B': 1.51, 'HR': 1.69
            }
            
            # Get weights from file or use defaults
            bb_weight = woba_weights.get('BB', default_weights['BB'])
            hbp_weight = woba_weights.get('HBP', default_weights['HBP'])
            single_weight = woba_weights.get('1B', default_weights['1B'])
            double_weight = woba_weights.get('2B', default_weights['2B'])
            triple_weight = woba_weights.get('3B', default_weights['3B'])
            hr_weight = woba_weights.get('HR', default_weights['HR'])
            
            # Calculate the numerator and denominator for wOBA
            numerator = (
                bb_weight * team_batting['BB'] +
                hbp_weight * team_batting['HBP'] +
                single_weight * team_batting['1B'] +
                double_weight * team_batting['2B'] +
                triple_weight * team_batting['3B'] +
                hr_weight * team_batting['HR']
            )
            
            denominator = (
                team_batting['AB'] + 
                team_batting['BB'] + 
                team_batting['HBP'] + 
                team_batting['SF']
            )
            
            # Calculate wOBA, handling division by zero
            team_batting['wOBA'] = numerator / denominator
            
            # Replace NaN values (from division by zero) with 0
            team_batting['wOBA'] = team_batting['wOBA'].fillna(0)
            
        except Exception as e:
            st.warning(f"Error calculating wOBA: {e}")
            # If there's an error, we'll continue without wOBA
            pass
    
    # Calculate the total number of team games
    team_games = team_batting['GP'].max() if 'GP' in team_batting.columns else 0
    
    # Calculate PA for each player (AB + BB + HBP + SF + SH)
    if all(col in team_batting.columns for col in ['AB', 'BB', 'HBP', 'SF', 'SH']):
        team_batting['PA'] = team_batting['AB'] + team_batting['BB'] + team_batting['HBP'] + team_batting['SF'] + team_batting['SH']
    else:
        # Fallback if some columns are missing
        team_batting['PA'] = team_batting['AB']
    
    # Add PA/G column
    team_batting['PA_per_G'] = team_batting['PA'] / team_batting['GP']
    
    # Calculate percent of games played
    team_batting['GP_pct'] = team_batting['GP'] / team_games if team_games > 0 else 0
    
    # Mark qualified players (≥2 PA/G and ≥75% of games)
    team_batting['qualified'] = (team_batting['PA_per_G'] >= 2) & (team_batting['GP_pct'] >= 0.75)
    
    # Sort by qualified first, then by BA
    if 'BA' in team_batting.columns:
        team_batting = team_batting.sort_values(['qualified', 'BA'], ascending=[False, False])
    
    # Format percentages
    for col in ["BA", "OBPct", "SlgPct", "wOBA"]:
        if col in team_batting.columns:
            team_batting[col] = team_batting[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    
    return team_batting, team_games

def process_pitching_data(pitching_data, team):
    """Process pitching data for a specific team"""
    if pitching_data is None:
        return None
    
    # First calculate league-wide stats for FIP constant
    if 'IP' in pitching_data.columns and 'ER' in pitching_data.columns:
        # League ERA calculation
        lg_era = (pitching_data['ER'].sum() * 9) / pitching_data['IP'].sum()
        
        # FIP components calculation
        hr_col = 'HR' if 'HR' in pitching_data.columns else 'HR-A'
        fo_col = 'FO' if 'FO' in pitching_data.columns else None
        
        fip_components = ((13 * pitching_data[hr_col].sum() + 
                          3 * (pitching_data['BB'].sum() + pitching_data['HB'].sum()) -
                          2 * pitching_data['SO'].sum()) / pitching_data['IP'].sum())
        
        # FIP constant
        f_constant = lg_era - fip_components
        
        # Calculate league HR/FB rate for xFIP if FO (fly outs) data is available
        lg_hr_fb_rate = None
        if fo_col is not None:
            fb_total = pitching_data[hr_col].sum() + pitching_data[fo_col].sum()
            if fb_total > 0:
                lg_hr_fb_rate = pitching_data[hr_col].sum() / fb_total
    else:
        # Default FIP constant if we can't calculate it
        f_constant = 4.81
        lg_hr_fb_rate = None
    
    # Filter for the selected team
    team_pitching = pitching_data[pitching_data['team_name'] == team].copy()
    if team_pitching.empty:
        return None
    
    # Calculate the total number of team games
    team_games = team_pitching['GS'].sum() if 'GS' in team_pitching.columns else 0
    
    # Add IP/G column
    team_pitching['IP_per_G'] = team_pitching['IP'] / team_games if team_games > 0 else 0
    
    # Mark qualified pitchers (≥1 IP/G)
    team_pitching['qualified'] = team_pitching['IP_per_G'] >= 1
    
    # Calculate Additional Stats
    def per_nine(numerator, ip):
        return np.where(ip > 0, (numerator / ip) * 9, 0)
    
    def percentage(numerator, denominator):
        return np.where(denominator > 0, (numerator / denominator) * 100, 0)
    
    # Calculate pitch mix percentages if BF column exists
    if 'BF' in team_pitching.columns:
        team_pitching['BB%'] = percentage(team_pitching['BB'], team_pitching['BF'])
        team_pitching['K%'] = percentage(team_pitching['SO'], team_pitching['BF'])
        team_pitching['K-BB%'] = team_pitching['K%'] - team_pitching['BB%']
    
    # Calculate FIP and xFIP
    if 'IP' in team_pitching.columns:
        # Identify which columns to use
        hr_col = 'HR' if 'HR' in team_pitching.columns else 'HR-A'
        
        # Calculate FIP for each pitcher
        team_pitching['FIP'] = np.where(
            team_pitching['IP'] > 0,
            f_constant + ((13 * team_pitching[hr_col] + 
                          3 * (team_pitching['BB'] + team_pitching['HB']) - 
                          2 * team_pitching['SO']) / team_pitching['IP']),
            float('nan')  # Use NaN for pitchers with 0 IP
        )
        
        # Calculate xFIP if we have the league HR/FB rate and fly out data
        if lg_hr_fb_rate is not None and fo_col is not None:
            team_pitching['xFIP'] = np.where(
                team_pitching['IP'] > 0,
                f_constant + ((13 * ((team_pitching[fo_col] + team_pitching[hr_col]) * lg_hr_fb_rate) +
                              3 * (team_pitching['BB'] + team_pitching['HB']) - 
                              2 * team_pitching['SO']) / team_pitching['IP']),
                float('nan')  # Use NaN for pitchers with 0 IP
            )
    
    # Handle ERA for pitchers with IP=0 and ER>0 (should be infinity)
    if all(col in team_pitching.columns for col in ["IP", "ER", "ERA"]):
        # For pitchers with IP=0 and ER>0, set ERA to infinity
        zero_ip_with_er = (team_pitching["IP"] == 0) & (team_pitching["ER"] > 0)
        team_pitching.loc[zero_ip_with_er, "ERA"] = np.inf
        
        # Also set FIP and xFIP to infinity for these pitchers
        if 'FIP' in team_pitching.columns:
            team_pitching.loc[zero_ip_with_er, "FIP"] = np.inf
        if 'xFIP' in team_pitching.columns:
            team_pitching.loc[zero_ip_with_er, "xFIP"] = np.inf
    
    # Sort by qualified first, then by ERA
    if 'ERA' in team_pitching.columns:
        team_pitching = team_pitching.sort_values(['qualified', 'ERA'], ascending=[False, True])
    
    return team_pitching

def display_batting_stats(team_batting, team_games, include_woba=True):
    """Display batting statistics"""
    if team_batting is None or team_batting.empty:
        return
    
    st.subheader("Batting Statistics")
    
    # Select columns to display
    batting_cols = [
        "player_name", "Yr", "Pos", "GP", "BA", "OBPct", "SlgPct", "wOBA",
        "AB", "R", "H", "2B", "3B", "HR",  "BB", "K", "RBI", "SB"
    ]
    
    # Only include columns that exist in the data
    batting_cols = [col for col in batting_cols if col in team_batting.columns]
    
    # Create a copy for display
    display_batting = team_batting[batting_cols].copy()
    
    # Create a formatted version for display only
    # This keeps the numeric columns as numbers for sorting
    formatted_batting = display_batting.copy()
    
    # Apply styling with a function that returns CSS for each row
    def batting_style(row):
        if not team_batting.loc[row.name, 'qualified']:
            return ['background-color: #f0f0f0'] * len(row)
        return [''] * len(row)
    
    # Apply the styling to the formatted dataframe
    styled_batting = formatted_batting.style.apply(batting_style, axis=1)
    
    # Display the styled table with the original dataframe for sorting
    # This allows proper numeric sorting while showing formatted values
    st.dataframe(
        styled_batting,
        hide_index=True,
        use_container_width=True
    )
    
    # Show qualification criteria
    st.caption(f"Players with white background: ≥2 PA/game and played in ≥75% of games ({int(team_games * 0.75)} games).")

def display_pitching_stats(team_pitching):
    """Display pitching statistics"""
    if team_pitching is None or team_pitching.empty:
        return
    
    st.subheader("Pitching Statistics")
    
    # Select columns to display
    pitching_cols = [
        "player_name", "Yr", "App", "GS", "ERA", "FIP", "xFIP", "IP", "W", "L", "BB%", "K%", "K-BB%",
        "H", "R", "ER", "BB", "SO"
    ]
    
    # Only include columns that exist in the data
    pitching_cols = [col for col in pitching_cols if col in team_pitching.columns]
    
    # Create a numeric dataframe for better sorting
    numeric_pitching = team_pitching.copy()
    
    # Convert string values to numeric (fixing any formatting done earlier)
    numeric_cols = ["ERA", "FIP", "xFIP", "IP", "BB%", "K%", "K-BB%"]
    for col in numeric_cols:
        if col in numeric_pitching.columns:
            # If the column has string values, convert back to numeric
            if numeric_pitching[col].dtype == 'object':
                # Handle 'Inf' strings and convert to float
                numeric_pitching[col] = numeric_pitching[col].replace('Inf', np.inf)
                numeric_pitching[col] = pd.to_numeric(numeric_pitching[col], errors='coerce')
    
    # Now create a display version with formatting 
    display_pitching = numeric_pitching[pitching_cols].copy()
    
    # Format values for display
    for col in numeric_cols:
        if col in display_pitching.columns:
            display_pitching[col] = numeric_pitching[col].apply(
                lambda x: "Inf" if x == np.inf else (f"{float(x):.2f}" if pd.notnull(x) else "")
            )
    
    # Apply styling for qualified status
    def highlight_unqualified(row):
        if not numeric_pitching.loc[row.name, 'qualified']:
            return ['background-color: #f0f0f0'] * len(display_pitching.columns)
        return [''] * len(display_pitching.columns)
    
    # Apply styling
    styled_pitching = display_pitching.style.apply(highlight_unqualified, axis=1)
    
    # Configure column types to ensure proper sorting
    column_config = {}
    for col in numeric_cols:
        if col in display_pitching.columns:
            # Tell Streamlit to treat these columns as numbers for sorting
            # But display the formatted values
            column_config[col] = st.column_config.NumberColumn(
                format="%.2f", 
            )
    
    # Display the dataframe with column configuration
    st.dataframe(
        styled_pitching, 
        column_config=column_config,
        hide_index=True,
        use_container_width=True
    )
    
    # Display caption
    st.caption(f"Players with white background: ≥1 IP/game.")

def scrape_data_from_qab() -> typing.List[typing.List[str]]:
    """Scrape data from Google Spreadsheet"""
    html = requests.get('https://docs.google.com/spreadsheets/d/1lltZw6rxNMv7np5sojYrmU5z03gmQBrNWGpreb2JnX8/gviz/tq?tqx=out:html&tq&gid=1').text
    soup = BeautifulSoup(html, 'html.parser')
    data = soup.find_all('table')[0]
    rows = [[td.text for td in row.find_all("td")] for row in data.find_all('tr')]
    return rows

def display_sidebar_info(batting_time, pitching_time):
    """Display data info in the sidebar"""
    st.sidebar.title("Data Info")
    
    if batting_time:
        st.sidebar.info(f"Batting data updated: {batting_time.strftime('%Y-%m-%d %H:%M')}")
    
    if pitching_time:
        st.sidebar.info(f"Pitching data updated: {pitching_time.strftime('%Y-%m-%d %H:%M')}")

def show_team_stats_page():
    """Display the team stats page"""
    st.title("Oberlin Baseball Team Stats Viewer")
    
    # Get available years
    years = get_available_years()
    selected_year = st.selectbox("Select Year", years, index=len(years)-1)
    
    # Load data
    batting_data, pitching_data, all_teams, batting_time, pitching_time = load_data(selected_year)
    
    # Find default index for Oberlin if it exists in the list
    default_team_index = 0  # Default to the first item (blank team)
    if "Oberlin" in all_teams:
        default_team_index = all_teams.index("Oberlin") + 1  # +1 because we added a blank team at the beginning
    
    # Team selection with Oberlin as default
    selected_team = st.selectbox("Select Team", [""] + all_teams, index=default_team_index)
    
    if selected_team:
        st.header(f"{selected_team} - {selected_year}")
        
        # Process and display batting stats - pass selected_year to use the right weights file
        team_batting, team_games = process_batting_data(batting_data, selected_team, selected_year)
        display_batting_stats(team_batting, team_games)
        
        # Process and display pitching stats
        team_pitching = process_pitching_data(pitching_data, selected_team)
        display_pitching_stats(team_pitching)
    else:
        st.warning(f"No team data available for {selected_year}")
    
    # Display sidebar info
    display_sidebar_info(batting_time, pitching_time)

def show_qab_data_page():
    """Display the QAB data page"""
    st.title("Hitter QAB Chart")
    
    try:
        # Show loading message
        with st.spinner("Loading data from spreadsheet..."):
            # Get data from spreadsheet
            rows = scrape_data_from_qab()
            
            # Use first row as column headers
            if rows and len(rows) > 0:
                headers = rows[0]
                data = rows[1:]
                
                # Limit to first 25 columns
                max_cols = 24
                if len(headers) > max_cols:
                    headers = headers[:max_cols]
                    data = [row[:max_cols] for row in data]
                
                df = pl.DataFrame(data=data, schema=headers)
            
            # Display data
            st.subheader("QAB Data")
            st.dataframe(df.head(24), use_container_width=True)
    except Exception as e:
        st.error(f"Error loading spreadsheet data: {e}")

def main():
    """Main function to run the Streamlit app"""
    # Add navigation to sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Team Stats", "QAB Data"])
    
    # Display the selected page
    if page == "Team Stats":
        show_team_stats_page()
    else:
        show_qab_data_page()

# Run the app
if __name__ == "__main__":
    main()