from FraudDetection.DataPreparation.DataCleaning import data_cleaning
from FraudDetection.DataPreparation.FeatureEngineering import feature_engineering
from FraudDetection.DataModelling.Modelling import modelling

def main():
    data_cleaning()
    feature_engineering()
    modelling()
    
if __name__ == "__main__":
    main()