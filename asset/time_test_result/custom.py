input_path = "/kaggle/input/datasets/meownosaur/steam-17m-2years"
filename = "steam_reviews_incremental"

from scipy.sparse import save_npz
from scipy.sparse import csr_matrix
import os
import pandas as pd
import numpy as np

records = pd.read_csv(f"{input_path}/{filename}.csv")

records

records = records.sort_values('recommendation_id', ascending=False) \
       .drop_duplicates(subset=['steam_id', 'app_id'], keep='first')

records

# Set the optimal k-core thresholds
min_user_interact = 20
min_game_interact = 100

# Iteratively trim users and games until the dataset stabilizes
filtered_df = records.copy()

while True:
    start_len = len(filtered_df)
    
    # Trim Games
    game_counts = filtered_df['app_id'].value_counts()
    valid_games = game_counts[game_counts >= min_game_interact].index
    filtered_df = filtered_df[filtered_df['app_id'].isin(valid_games)]
    
    # Trim Users
    user_counts = filtered_df['steam_id'].value_counts()
    valid_users = user_counts[user_counts >= min_user_interact].index
    filtered_df = filtered_df[filtered_df['steam_id'].isin(valid_users)]
    
    # Break loop if no more rows were dropped
    if start_len == len(filtered_df):
        break

# Calculate and print final metrics
num_users = filtered_df['steam_id'].nunique()
num_games = filtered_df['app_id'].nunique()
num_actual_interact = len(filtered_df)
sparsity = round(1 - num_actual_interact / (num_users * num_games), 4)

print(f"Thresholds: Users >= {min_user_interact} | Games >= {min_game_interact}")
print(f"Users: {num_users:,}")
print(f"Games: {num_games:,}")
print(f"Interactions: {num_actual_interact:,}")
print(f"Sparsity: {sparsity}")

def split(df: pd.DataFrame, train_size: float, val_size: float, test_size: float, timestamp_sort=False):
    assert train_size + val_size + test_size == 10, "Sum of proportions must be 10"
    
    # Calculate cutoff percentages
    train_pct = train_size / 10.0
    val_pct = val_size / 10.0
    
    # 1. Sort the entire dataframe at once (Vectorized!)
    if timestamp_sort:
        temp_df = df.sort_values(["steam_id", "timestamp"]).copy()
    else:
        # If not sorting by time, shuffle the dataframe randomly first
        temp_df = df.sample(frac=1, random_state=1).sort_values("steam_id").copy()
        
    # 2. Calculate the chronological interaction number for each user
    temp_df['interact_rank'] = temp_df.groupby("steam_id").cumcount()
    temp_df['user_total'] = temp_df.groupby("steam_id")["steam_id"].transform('count')
    
    # 3. Calculate the cutoff boundaries for each user
    temp_df['train_cutoff'] = temp_df['user_total'] * train_pct
    temp_df['val_cutoff'] = temp_df['user_total'] * (train_pct + val_pct)
    
    # 4. Split the data using highly optimized Pandas masking
    df_train = temp_df[temp_df['interact_rank'] < temp_df['train_cutoff']]
    df_val = temp_df[(temp_df['interact_rank'] >= temp_df['train_cutoff']) & (temp_df['interact_rank'] < temp_df['val_cutoff'])]
    df_test = temp_df[temp_df['interact_rank'] >= temp_df['val_cutoff']]
    
    # 5. Clean up the temporary calculation columns
    drop_cols = ['interact_rank', 'user_total', 'train_cutoff', 'val_cutoff']
    df_train = df_train.drop(columns=drop_cols)
    df_val = df_val.drop(columns=drop_cols)
    df_test = df_test.drop(columns=drop_cols)
    
    return df_train, df_val, df_test

# raise KeyboardInterrupt

train, val, test = split(filtered_df, 7, 2, 1, timestamp_sort=True)

train

val

test

unique_users = filtered_df['steam_id'].unique()
unique_games = filtered_df['app_id'].unique()

user2idx = {user: idx for idx, user in enumerate(unique_users)}
game2idx = {game: idx for idx, game in enumerate(unique_games)}

num_users = len(unique_users)
num_games = len(unique_games)

def to_interaction_mat(df: pd.DataFrame, explicit_feedback=False):
    # Map the actual IDs to our global matrix indices
    user_indices = df['steam_id'].map(user2idx).values
    game_indices = df['app_id'].map(game2idx).values
    
    # 2. Build sparse matrix
    if explicit_feedback:
        # True becomes 1.0, False becomes -1.0
        data = np.where(df['voted_up'] == True, 1.0, -1.0).astype(np.float32)
    else:
        # Standard implicit feedback (everything is 1.0)
        data = np.ones(len(df), dtype=np.float32)

    R = csr_matrix((data, (user_indices, game_indices)),
                   shape=(num_users, num_games))
    
    return R

train_mat = to_interaction_mat(train, explicit_feedback=False)
val_mat = to_interaction_mat(val, explicit_feedback=False)
test_mat = to_interaction_mat(test, explicit_feedback=False)

train_mat

val_mat

test_mat

def save_sparse(sparse_mat, path, filename):
    os.makedirs(path, exist_ok=True)
    save_npz(f"{path}/{filename}.npz", sparse_mat)

save_sparse(train_mat, filename, 'train')
save_sparse(val_mat, filename, 'val')
save_sparse(test_mat, filename, 'test')