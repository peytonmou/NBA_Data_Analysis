import pandas as pd 
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

df = pd.read_csv('Seasons_Stats.csv')                    # Read the original data
pos_mapping = {'G-F': 'GF','F-G': 'GF', 'F-C': 'FC'}
df['Pos'] = df['Pos'].replace(pos_mapping)               # Use map to correct error entries of positions (although these rows will be filtered later by "Year" condition)

def split_positions(row):                                # Create a function to duplicate one more row for players who play two positions
    if pd.isna(row['Pos']):                              # Replace empty entry into 'na' to avoid error when split later
        positions = ['na']
    else:
        positions = str(row['Pos']).split('-')           # Convert entries into strings, split positions, return a list of two or one position elements
    rows = []
    for position in positions:                           # Iterate over all elements in the list of positions
        new_row = row.copy()                             # Copy the original row as the new row
        new_row['Pos'] = position                        # Update the position, if there are two positions in the original row, there will be two new rows
        rows.append(new_row)                             # Append the new row
    return pd.DataFrame(rows)

df_new = df.apply(split_positions,axis=1).reset_index(drop=True).drop(columns=['Unnamed: 0'], errors='ignore')
df_new = pd.concat(df_new.tolist(), ignore_index=True)   # Split positions, reset indexes, drop index column to avoid disorganized file, combine dataframes into one list

df_new= df_new[(df_new['Year'] > 1990) & (df_new['MP'] > 480) & (df_new['Pos'] != 'na')]              # Filter 'Year', 'MP', delete 'na'in 'Pos'
columns = ['Year','Player','Pos','Age','Tm','MP','G', 'PTS', 'AST', 'TRB', 'BLK', 'TS%', 'PER']       # Identify useful columns for computation later
df_new = df_new[columns].dropna()                                                                     # Filter useful columns and drop null value rows

# Average Performance by Age:
grouped1 = df_new.groupby(['Player','Year']).agg({             # DataFrame grouped by two labels (player and year)
    'Age':'mean',              # Age remains the same value
    'PTS': 'sum',              # Sum of points gained in one season
    'AST': 'sum',              # Sum of assits in one season
    'TRB': 'sum',              # Sum of rebounds in one season
    'BLK': 'sum',              # Sum of blocks in one season
    'G': 'sum',                # Sum of games played in one season
    'MP': 'sum',               # Sum of minutes played in one season 
    'TS%': lambda x: np.average(x, weights = df_new.loc[x.index, 'MP']),                  # Average shooting percentage in one season, weighted by minutes played
    'PER': lambda x: np.average(x, weights = df_new.loc[x.index, 'MP'])}).reset_index()   # Average efficiency rating in one season, weighted by minutes played

average_stats = grouped1.assign(                                                          # Calculate the averages of above attained four sums
    A_PTS = lambda x: x['PTS'] / x['G'],
    A_AST = lambda x: x['AST'] / x['G'],
    A_TRB = lambda x: x['TRB'] / x['G'],
    A_Blocks = lambda x: x['BLK'] / x['G'])
average_stats = average_stats.rename(columns={'TS%': 'A_TS','PER': 'A_PER'})              # Rename two columns' labels

grouped2 = average_stats.groupby('Age').agg({        # DataFrame grouped by one label (age) and calculate the means of six performance metrics            
    'Age': 'first',
    'A_TS': 'mean',
    'A_PER': 'mean',
    'A_PTS': 'mean',
    'A_AST': 'mean',
    'A_TRB': 'mean',
    'A_Blocks': 'mean'})

metrics = ['A_TS', 'A_PER', 'A_PTS', 'A_AST', 'A_TRB', 'A_Blocks']       # Create a list of six performance metrics
titles = {                                                               # Create a dictionary of six metrics as keys and six graph titles as values
    'A_TS': 'Average Shooting Percentage per Season',
    'A_PER': 'Average Efficiency Rating per Season',
    'A_PTS': 'Average Points per Game in a Season',
    'A_AST': 'Average Assists per Game in a Season',
    'A_TRB': 'Average Rebounds per Game in a Season',
    'A_Blocks': 'Average Blocks per Game in a Season'}
y_labels = {'A_TS': 'A_TS','A_PER': 'A_PER','A_PTS': 'A_PTS','A_AST': 'A_AST','A_TRB': 'A_TRB','A_Blocks': 'A_Blocks'}    # Create a dictionary of six metrics as keys and six y-labels as values

for metric in metrics:                                                   # Loop over the list of metrics to plot six line graphs
    plt.plot(grouped2['Age'], grouped2[metric], marker='o')
    plt.title(titles[metric])
    plt.xlabel('Age')
    plt.ylabel(y_labels[metric])
    plt.show()

# Average Performance by Age and Position:
grouped3 = df_new.groupby(['Pos','Age']).agg({
    'Pos': 'first',
    'Age': 'first',
    'G': 'sum',
    'MP': 'sum',
    'AST': 'sum', 
    'TRB': 'sum', 
    'PTS': 'sum',  
    'BLK': 'sum',   
    'TS%': lambda x: np.average(x, weights = df_new.loc[x.index,'MP']),
    'PER': lambda x: np.average(x, weights = df_new.loc[x.index, 'MP'])})

# Calculate four mean values
average_stats_pos = grouped3.assign(  
    A_PTS = lambda x: x['PTS'] / x['G'],
    A_AST = lambda x: x['AST'] / x['G'],
    A_TRB = lambda x: x['TRB'] / x['G'],
    A_Blocks = lambda x: x['BLK'] / x['G'])

# Rename two column labels and drop useless columns
average_stats_pos = average_stats_pos.rename(columns={'TS%': 'A_TS','PER': 'A_PER'}) 
columns = ['Pos','Age', 'A_PTS', 'A_AST', 'A_TRB', 'A_Blocks', 'A_TS', 'A_PER']       
average_stats_pos = average_stats_pos[columns].dropna()    

# Loop to plot 6 graphs (each has 5 different color lines for 5 positions)
positions = ['C','SF','SG','PF','PG']
colors = ['red','blue','green','yellow','black']
metrics = ['A_PTS','A_AST','A_Blocks','A_TRB','A_PER','A_TS']

for metric in metrics:
    # Plot one figure for each metric
    plt.figure()
    # Loop 5 times to draw 5 lines in each graph
    for position, color in zip(positions, colors):
        # Filter the relevant position's data
        pos_stats = average_stats_pos[average_stats_pos['Pos'] == position]
        # Plot the line, x=age, y=metric value, color as defined by list of colors, label as defined by list of positions
        plt.plot(pos_stats['Age'], pos_stats[metric], color=color, label=position)
    plt.title(f'{metric} by Age and Position')
    plt.xlabel('Age')
    plt.ylabel(metric)
    plt.legend(title='Position')
    plt.show()

# Overall Performance by Position:
grouped4 = df_new.groupby('Pos').agg({
    'Pos': 'first',
    'G': 'sum',
    'MP': 'sum',
    'AST': 'sum', 
    'TRB': 'sum', 
    'PTS': 'sum',  
    'BLK': 'sum',   
    'TS%': lambda x: np.average(x, weights = df_new.loc[x.index,'MP']),
    'PER': lambda x: np.average(x, weights = df_new.loc[x.index, 'MP'])})

overall_mean_pos = grouped4.assign(  
    O_PTS = lambda x: x['PTS'] / x['G'],
    O_AST = lambda x: x['AST'] / x['G'],
    O_TRB = lambda x: x['TRB'] / x['G'],
    O_Blocks = lambda x: x['BLK'] / x['G'])

overall_mean_pos = overall_mean_pos.rename(columns={'TS%': 'O_TS','PER': 'O_PER'})           
columns = ['Pos', 'G', 'MP', 'O_PTS', 'O_AST', 'O_TRB', 'O_Blocks', 'O_TS', 'O_PER']       
overall_mean_pos = overall_mean_pos[columns].dropna()   

def bootstrap_mean(data, g, num_samples):                                # g = total number of games played for each position
    rng = rand.default_rng()                                             # Generate a random number by default 
    mean_list = []
    for _ in range(num_samples):
        new_data = rng.choice(data, size=len(data), replace=True)        # Update data with one randoml number which is within data range
        mean = np.sum(new_data) / g                               
        mean_list += [mean]
    return np.percentile(np.array(mean_list), [2.5, 97.5])               # Return the 95% confidence interval of the mean_list

def bootstrap_weighted_mean(data, mp, num_samples):                      # mp = minutes played for each position, will be used as the weight
    rng = rand.default_rng()
    weighted_mean_list = []
    for _ in range(num_samples):
        indexs = rng.choice(len(data), size=len(data), replace=True)     # Get random numbers within data length range as indexs
        new_data = data.iloc[indexs]                                     # Find the values on specific indexs within data and MP
        new_weights = mp.iloc[indexs]
        weighted_mean = np.average(new_data, weights = new_weights)      # Calculate the weighted means
        weighted_mean_list += [weighted_mean]
    
    return np.percentile(np.array(weighted_mean_list), [2.5, 97.5])

ci = {}                                                                  # Initialise a dictionary for 95% ci of 5 positions' 6 metrics
positions = ['C','SF','SG','PF','PG']
O_metrics = ['O_PTS', 'O_AST', 'O_TRB', 'O_Blocks', 'O_TS', 'O_PER'] 
df_new = df_new.rename(columns={'PTS':'O_PTS', 'AST':'O_AST','BLK':'O_Blocks', 'TRB':'O_TRB','TS%':'O_TS','PER':'O_PER'})

for position in positions:
    ci[position] = {}                                                    # Create a dictionary respectively for each position

    for metric in O_metrics[:4]:                                         # [:4] indicates for O_PTS, O_AST, O_TRB and O_BLK
        data = df_new[df_new['Pos']==position][metric]
        g = overall_mean_pos[overall_mean_pos['Pos']==position]['G']
        x = bootstrap_mean(data, g, 10000)                               # Bootstrap sampling 10000 times and get 95% ci
        ci[position][metric] = x
    
    for metric in O_metrics[4:]:                                         # [4:] indicates for O_TS and O_PER                 
        data = df_new[df_new['Pos']==position][metric]
        mp = df_new[df_new['Pos']==position]['MP']
        x = bootstrap_weighted_mean(data, mp, 10000)                     # Bootstrap sampling 10000 times and get 95% ci (this mean is weighted computed)
        ci[position][metric] = x
    
for metric in O_metrics:                                                # Plot 6 bar graphs for overall metrics by position with 95% ci 
    plt.figure()                                                        # Respectively plot one graph for each metric
    heights = []
    y_err = []

    for position in positions:
        mean = overall_mean_pos.loc[overall_mean_pos['Pos']==position, metric].values[0]        # Extract the specific position's specific metric value
        heights += [mean]                                                                       # Append the value into list of heights
        a = ci[position][metric]                                                                # Extract the bootstrap 95% ci of this value
        y_err += [(a[1] - a[0]) / 2]                                                            # Calculate the half distance of error
        
    plt.bar(positions, heights, yerr = y_err, capsize=3)
    plt.title(f'{metric} Mean by Position with 95% Confidence Interval')
    plt.xlabel('Position')
    plt.ylabel(metric)
    plt.ylim(min(heights)*0.5, max(heights)*1.35)                                               # Adjust y-limit to clearly present difference among values
    plt.show()

# Westbrook's Triple Doubles:
df1 = pd.read_csv('games.csv')
df2 = pd.read_csv('games_details.csv', low_memory=False)          # low_memory=False, to read file in chunks and infer correct dtype
Westbrook_id = 201566

df_wb = df2[df2['PLAYER_ID']==Westbrook_id]                       # Filter Westbrook's game statistics
num_games = df_wb['GAME_ID'].nunique()                            # total number of games played by Westbrook
num_td = df_wb[(df_wb['AST'] >= 10) & (df_wb['PTS'] >= 10)        # number of Westbrook's triple doubles
               & (df_wb['REB'] >= 10)]['GAME_ID'].nunique()       
y = num_td / num_games                                            # Event Y = Westbrook got triple doubles, calculate its probability

df3 = pd.merge(df1, df_wb, on = 'GAME_ID')                        # Merge df1 and df2's Westbrook statistics based on GAME_ID 
x_y = (df3[(df3['HOME_TEAM_WINS']==1) & (df3['PTS'] >= 10)        # probability of both events X and Y happen (event X: team won) 
         & (df3['AST'] >= 10) & (df3['REB'] >= 10)]['GAME_ID'].nunique()) / num_games
x_given_y = x_y / y                                               # conditional probability of X given Y: P (X | Y) = P(X, Y) / P(Y)
x_not_y = (df3[(df3['HOME_TEAM_WINS']==1) & (df3['PTS'] < 10)     # probability of event X happen and Y not happen 
            & (df3['AST'] < 10) & (df3['REB'] < 10)]['GAME_ID'].nunique()) / num_games
x_given_not_y = x_not_y / (1-y)                                   # probability of X given not Y: P (X | not Y) = P(X, not Y) / P(not Y)

'''
Create a contingency table array with below representations:
          x_y: Team won and Westbrook got triple-double
          y - x_y: Westbrook got triple-double and team lost
          x_not_y: Westbrook did not get triple-double and team won
          (1 - y) - x_not_y: Westbrook did not get triple-double and team lost

Chi-squared test to evaluate whether there's a significant relationship between X and Y:
          chi2: measures the deviation from expected if X and Y are independent
          p: the probability of observing this data if the null hypothesis (X and Y are independence) is true (cannot reject null hypothesis when p value > 0.05)
          dof: degrees of freedom: (num_rows - 1) * (num_columns - 1), in this case dof == 1
          expected: expected counts if X and Y being independent
'''
contingency_table = np.array([[x_y, (y - x_y)], [x_not_y, ((1 - y) - x_not_y)]])
chi2, p, dof, expected = chi2_contingency(contingency_table)
