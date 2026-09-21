import joblib
import pandas as pd

model = joblib.load("digital_advertisement_performance_model.pkl")

sample = pd.DataFrame([{
    "platform": "Google",
    "ad_type": "Video",
    "budget": 85000,
    "impressions": 240000,
    "click_through_rate": 8.4,
    "engagement_rate": 16.5,
    "target_audience": "25-34",
    "campaign_duration_days": 14
}])

prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]

print("Predicted Advertisement Performance:", prediction)
print("\nClass probabilities:")
for label, probability in zip(model.classes_, probabilities):
    print(f"{label}: {probability:.2%}")
