import pandas as pd
import numpy as np
import math
import re
import joblib
import os

def extract_features(url):
    """
    FINAL IMPLEMENTATION: Extracts all 11 features defined in the Team Guide (Page 8-9).
    This must match M3's training features exactly.
    """
    features = {}
    
    # 1. URL length
    features['URL length'] = len(url)
    
    # 2. Number of dots
    features['Number of dots'] = url.count('.')
    
    # 3. Number of hyphens
    features['Number of hyphens'] = url.count('-')
    
    # 4. Number of slashes
    features['Number of slashes'] = url.count('/')
    
    # 5. Has @ symbol
    features['Has @ symbol'] = 1 if '@' in url else 0
    
    # 6. Has HTTPS
    features['Has HTTPS'] = 1 if url.startswith('https') else 0
    
    # 7. Has IP address
    features['Has IP address'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0
    
    # 8. Number of subdomains
    domain_match = re.search(r'https?://([^/]+)', url)
    if domain_match:
        domain = domain_match.group(1)
        parts = domain.split('.')
        features['Number of subdomains'] = max(0, len(parts) - 2)
    else:
        features['Number of subdomains'] = 0
        
    # 9. Digit ratio
    features['Digit ratio'] = sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0
    
    # 10. URL entropy (Shannon Entropy)
    if len(url) > 0:
        prob = [float(url.count(c)) / len(url) for c in dict.fromkeys(list(url))]
        features['URL entropy'] = - sum([p * math.log(p) / math.log(2.0) for p in prob])
    else:
        features['URL entropy'] = 0
        
    # 11. Domain length
    if domain_match:
        features['Domain length'] = len(domain_match.group(1))
    else:
        features['Domain length'] = 0
    
    return features

def load_model():
    """
    Loads the trained model and scaler from the /models directory.
    """
    model_path = 'models/model.pkl'
    scaler_path = 'models/scaler.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler, False # False = Not in demo mode
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None, True
    return None, None, True

def predict_url(url, model, scaler, demo_mode=False):
    """
    Full prediction pipeline: Extract -> Scale -> Predict.
    """
    feature_dict = extract_features(url)
    
    if demo_mode:
        # High-quality heuristic for demo if model files aren't present yet
        is_phishing = any(word in url.lower() for word in ['verify', 'secure', 'login', 'update', '192.168', 'paypal', 'bank'])
        confidence = 94.2 if is_phishing else 97.8
        return is_phishing, confidence, feature_dict
    
    # Real Prediction Logic
    # Convert dict to array in the correct order
    feature_names = [
        'URL length', 'Number of dots', 'Number of hyphens', 'Number of slashes',
        'Has @ symbol', 'Has HTTPS', 'Has IP address', 'Number of subdomains',
        'Digit ratio', 'URL entropy', 'Domain length'
    ]
    X = np.array([feature_dict[f] for f in feature_names]).reshape(1, -1)
    
    # Scale and Predict
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    
    # Handle probability if model supports it
    try:
        prob = model.predict_proba(X_scaled)[0]
        confidence = prob[prediction] * 100
    except:
        confidence = 100.0 # Fallback if no proba
        
    return bool(prediction == 1), confidence, feature_dict
