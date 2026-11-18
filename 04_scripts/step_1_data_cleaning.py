from FraudDetection.DataPreparation.DataCleaning import data_cleaning

def main():
    """Lance le nettoyage et l’export des paiements pour l’EDA."""
    data_cleaning()
    
if __name__ == "__main__":
    main()