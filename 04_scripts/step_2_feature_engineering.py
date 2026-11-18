from FraudDetection.DataPreparation.FeatureEngineering import feature_engineering

def main():
    """Lance le calcul et l’export des variables métier et historiques."""
    feature_engineering()
    
if __name__ == "__main__":
    main()