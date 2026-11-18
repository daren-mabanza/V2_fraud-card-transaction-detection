from FraudDetection.DataPreparation.DataCleaning import data_cleaning
from FraudDetection.DataPreparation.FeatureEngineering import feature_engineering
from FraudDetection.DataModelling.Modelling import modelling
from FraudDetection.DataModelling.Export.Metrics import export_metrics
from FraudDetection.DataModelling.Export.TestData import export_test_data
from FraudDetection.DataModelling.Export.Shap import export_shap

def main():
    data_cleaning()
    feature_engineering()
    modelling()
    export_metrics()
    export_test_data()
    export_shap()

if __name__ == "__main__":
    main()
