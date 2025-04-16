from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

feature_names = ['Rain - mm', 'hour', 'Rain - mm lag1', 'Rain - mm lag2', 'Rain - mm lag3']

# Load the scaler
scaler_features = joblib.load('scaler_features.pkl')

# Load the scaler
scaler_labels = joblib.load('scaler_labels.pkl')

cumulative_15min_rainfall = [8.4, 14.5, 18.2, 22.2, 32.2, 23.4, 24.6]
now = datetime.now().replace(hour=2)

# Load the trained model

model = load_model('rainfall_mm_v4_with_cnn.h5', compile=False)
model.compile(loss=MeanSquaredError(), optimizer="adam")

print(model.input_shape)

# Prepare input data
input_data = np.array([
    [cumulative_15min_rainfall[-4], now.hour, cumulative_15min_rainfall[-3], cumulative_15min_rainfall[-2], cumulative_15min_rainfall[-1]],
    [cumulative_15min_rainfall[-5], now.hour, cumulative_15min_rainfall[-4], cumulative_15min_rainfall[-3], cumulative_15min_rainfall[-2]],
    [cumulative_15min_rainfall[-6], now.hour, cumulative_15min_rainfall[-5], cumulative_15min_rainfall[-4], cumulative_15min_rainfall[-3]],
    [cumulative_15min_rainfall[-7], now.hour, cumulative_15min_rainfall[-6], cumulative_15min_rainfall[-5], cumulative_15min_rainfall[-4]]
])

input_df = pd.DataFrame(input_data, columns=feature_names)

rain_mm_column = input_df[['Rain - mm']]
feature_columns = input_df.drop(columns=['Rain - mm'])

# Scale each part separately
rain_mm_scaled = scaler_labels.transform(rain_mm_column)  # Scale Rain - mm
feature_scaled = scaler_features.transform(feature_columns)  # Scale other features

# Combine the scaled parts back together
input_data_scaled = np.hstack((rain_mm_scaled, feature_scaled))

# Reshape to match LSTM expected input sha
input_data_scaled = input_data_scaled.reshape(1, 4, 5) 

predictions = model.predict(input_data_scaled)

# Scale the predictions

predictions_scaled = scaler_labels.inverse_transform(predictions)

predictions_scaled_rounded = np.round(predictions_scaled, 2)

total_predicted_rainfall = np.sum(predictions_scaled_rounded)

print("Predicted rainfall in the next hour:", predictions_scaled_rounded)

# Add classification and action logic here

