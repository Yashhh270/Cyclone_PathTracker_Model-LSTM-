import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM,Dense,Dropout
from tensorflow.keras.optimizers import Adam
import os
import geopandas as gpd
import matplotlib.pyplot as plt
#Model_path='tracker.keras'
def single_cyc(df):
     df=df.drop(columns=['CI No [or "T. No"]',
                        'Estimated Central Pressure (hPa) [or "E.C.P"]',
                        'Pressure Drop(hPa)[or"delta P"]','Grade (text)',
                        'Outermost closed isobar (hPa)',
                        'Diameter/Size of outermost closed isobar(in degree)',
                        'Unnamed: 14','Serial Number of system during year','Basin of origin'])
     df['Date'] = df['Date'].astype(str)
     df['Time'] = df['Time'].astype(str)
     df['dt']=pd.to_datetime(df['Date']+' '+df['Time'],
           format='%d/%m/%Y %H%M',dayfirst=True,errors='coerce')
     df=df.drop(columns=['Date','Time'])
     df['lat']=df['lat'].astype(float)
     df['lon']=df['lon'].astype(float)
     df['windspeed']=df['windspeed'].astype(float)
     return df
def load(dp):
    df=pd.read_csv(dp)
    df=df.drop(columns=['Name','CI No [or "T. No"]',
                        'Estimated Central Pressure (hPa) [or "E.C.P"]',
                        'Pressure Drop(hPa)[or"delta P"]','Grade (text)',
                        'Outermost closed isobar (hPa)',
                        'Diameter/Size of outermost closed isobar(in degree)',
                        'Unnamed: 14','Serial Number of system during year','Basin of origin'])
    df['dt']=pd.to_datetime(df['Date']+' '+df['Time'],
           format='%d/%m/%Y %H%M',dayfirst=True,errors='coerce')
    df=df.drop(columns=['Date','Time'])
    df['lat']=df['lat'].astype(float)
    df['lon']=df['lon'].astype(float)
    df['windspeed']=df['windspeed'].astype(float)
    df['td']=df['dt'].diff().dt.total_seconds()/3600
    df['cycid']=((df['td'] > 12) |
                (df['lat'].diff().abs() > 5) |
                (df['lon'].diff().abs() > 5)
                   ).cumsum()
    df['time-diff']=df.groupby('cycid')['dt'].diff().dt.total_seconds()/3600
    df['time-diff']=df['time-diff'].fillna(0)
    df=df.drop(columns=['td'])


    return df

def predict_track_until_landfall(initial_seq, model, scaler, max_steps=15, windspeed_threshold=0.2):
    current_seq = initial_seq.copy()
    predictions = []

    for _ in range(max_steps):
        pred_norm = model.predict(current_seq.reshape(1,12, 3), verbose=1)[0]
        pred = scaler.inverse_transform(pred_norm.reshape(1, -1))[0]
        predictions.append(pred)
        if pred[2] < windspeed_threshold:
            break

        norm_pred = scaler.transform([pred])[0]
        current_seq = np.vstack([current_seq[1:], norm_pred])#Stacking the Previous preictions

    return np.array(predictions)

def pre(df,seq_len=24):
    features=['lat','lon','windspeed']
    scaler=MinMaxScaler()
    df[features]=df[features].dropna()
    df[features]=scaler.fit_transform(df[features])
    seq=[]
    tar=[]
    cycids=[]
    for _,g in df.groupby('cycid'):
        d=g[features].values
        if len(d)<=seq_len:
            continue
        for i in range(len(g)-seq_len):
                seq_part = d[i:i+seq_len]
                tar_part = d[i+seq_len]
                if not np.isnan(seq_part).any() and not np.isnan(tar_part).any():
                    seq.append(seq_part)
                    tar.append(tar_part)
                    cycids.append(_)

    return np.array(seq),np.array(tar),scaler,np.array(cycids)

df=load('Book1.csv')
df_3h=df[df['time-diff'].isin([0.0,3.0])].copy()
print(df_3h.head())
X,y,scaler,cycid=pre(df_3h)
#print(X)
print(df_3h.tail())
print(y)
print(X.shape)
print(y.shape)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Initialize K-Fold
k = 5
kf = KFold(n_splits=k, shuffle=False)

# To collect metric
rmse_list = []
mae_list = []
mape_list = []


# Track best model
best_rmse = float('inf')
best_model = None
best_fold = -1

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"\n--- Fold {fold+1} ---")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Model definition
    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        Dropout(0.5),
        LSTM(32),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(3)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])

    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=25,
              batch_size=32,
              verbose=1)

    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test)

    # Metrics
    rmse = root_mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)

    rmse_list.append(rmse)
    mae_list.append(mae)
    mape_list.append(mape * 100)

    print(f"Fold {fold+1} RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape*100:.2f}%")

    # Save best model
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_fold = fold + 1

# Save best model after all folds
if best_model is not None:
    best_model.save('best_model_kfold.keras')
    print(f"\n✅ Best model saved from Fold {best_fold} with RMSE: {best_rmse:.2f}")
# Calculate averages
avg_rmse = np.mean(rmse_list)
avg_mae = np.mean(mae_list)
avg_mape = np.mean(mape_list)

# Plot
metrics = ['RMSE', 'MAE', 'MAPE']
values = [avg_rmse, avg_mae, avg_mape]

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, values, color=['skyblue', 'salmon', 'lightgreen'])

# Annotate bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.2f}', ha='center', fontsize=12)

plt.title('Average Performance Metrics Across K-Folds')
plt.ylabel('Metric Value')
plt.ylim(0, max(values) + 5)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()



#Prediction
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Load models
Model1 = load_model('best_model_kfold.keras')
Model = load_model('tracker.keras')

# Load and preprocess input
pre = pd.read_csv('predict.csv')
pre = single_cyc(pre)
f = ['lat', 'lon', 'windspeed']
seq = pre[f].values[:12]  # Initial sequence
act = pre[f].values      # Full actual values

# Normalize
nor = scaler.transform(seq)

# Predict until landfall
prediction = predict_track_until_landfall(nor, Model1, scaler)
predictions = np.vstack([seq, prediction])  # Combine initial + predicted

# Create DataFrames
df_pred = pd.DataFrame(predictions, columns=['lat', 'lon', 'windspeed'])
df_act = pd.DataFrame(act, columns=['lat', 'lon', 'windspeed'])
print(df_pred)
print(df_act)

# -------- Plot Track Map --------
world = gpd.read_file('ne_110m_admin_0_countries.shp')

fig, ax = plt.subplots(figsize=(10, 8))
world.plot(ax=ax, color='lightgrey', edgecolor='black')

ax.plot(df_pred['lon'], df_pred['lat'], 'r-', label='Predicted Track')
ax.plot(df_act['lon'], df_act['lat'], 'b-', label='Actual Track')

ax.set_xlim(63, 92)
ax.set_ylim(7, 25)
ax.set_title('Cyclone Track: Predicted vs Actual (Mandous)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# -------- Plot Bar Graph for Metrics --------
df_act_trimmed = df_act.iloc[:len(df_pred)]

# Flatten
y_pred_all = df_pred.values.flatten()
y_true_all = df_act_trimmed.values.flatten()

rmse_overall = root_mean_squared_error(y_true_all, y_pred_all)
mae_overall = mean_absolute_error(y_true_all, y_pred_all)
mape_overall = mean_absolute_percentage_error(y_true_all, y_pred_all) * 100

print(f"✅ Overall RMSE: {rmse_overall:.2f}")
print(f"✅ Overall MAE: {mae_overall:.2f}")
print(f"✅ Overall MAPE: {mape_overall:.2f}%")

metrics = ['RMSE', 'MAE', 'MAPE']
values = [rmse_overall, mae_overall, mape_overall]

plt.figure(figsize=(6, 5))
bars = plt.bar(metrics, values, color=['skyblue', 'salmon', 'lightgreen'])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}', ha='center', fontsize=12)

plt.title('Overall Performance Metrics')
plt.ylabel('Error Value')
plt.ylim(0, max(values) + 5)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
