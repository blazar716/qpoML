import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Generate synthetic light curve data
def generate_synthetic_light_curve(periodic=True, num_points=1000):
    time = np.linspace(0, 50, num_points)
    if periodic:
        amplitude = 1 + np.random.uniform(-0.1, 0.1)
        period = 10 * (1 + np.random.uniform(-0.1, 0.1))
        flux = amplitude * np.sin(2 * np.pi * time / period) + 0.1 * np.random.normal(size=num_points)
    else:
        flux = 0.5 * np.random.normal(size=num_points)
    return time, flux

num_samples = 1000
light_curves = []
labels = []

for _ in range(num_samples // 2):
    time, flux = generate_synthetic_light_curve(periodic=True)
    light_curves.append(flux)
    labels.append(1)

for _ in range(num_samples // 2):
    time, flux = generate_synthetic_light_curve(periodic=False)
    light_curves.append(flux)
    labels.append(0)

light_curves = np.array(light_curves)
labels = np.array(labels)

# Step 2: Extract features from the light curves
def extract_features(light_curve):
    mean_flux = np.mean(light_curve)
    std_flux = np.std(light_curve)
    max_flux = np.max(light_curve)
    min_flux = np.min(light_curve)
    return [mean_flux, std_flux, max_flux, min_flux]

X = np.array([extract_features(lc) for lc in light_curves])
y = labels

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train a machine learning model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

# Plotting an example of a periodic and non-periodic light curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(time, light_curves[0])
plt.title('Example Periodic Light Curve')
plt.xlabel('Time')
plt.ylabel('Flux')

plt.subplot(1, 2, 2)
plt.plot(time, light_curves[num_samples // 2])
plt.title('Example Non-Periodic Light Curve')
plt.xlabel('Time')
plt.ylabel('Flux')

plt.tight_layout()
plt.show()
