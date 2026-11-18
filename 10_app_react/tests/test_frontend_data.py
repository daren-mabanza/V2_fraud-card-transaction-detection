"""Vérifie la lecture directe des exports, sans entraînement ni artefacts modèle."""
from pathlib import Path
import json
import tempfile
import unittest

import pandas as pd
from FastAPITools.FrontendData import demo_data, business_metrics, TEST_DATA
from FraudDetection.DataModelling.ModellingParameters import features_selected

FIXTURES = Path(__file__).parent / 'fixtures'


class FrontendDataTests(unittest.TestCase):
    def setUp(self):
        """Prépare des exports temporaires à partir des jeux de test, sans charger le modèle."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / '11_configs').mkdir()
        (self.root / '11_configs/paths.yaml').write_text('echantillon:\n  type: Full\n')
        (self.root / '07_artifacts').mkdir()
        data = json.loads((FIXTURES / 'transactions.json').read_text())
        self.ids = [row['id'] for row in data['observations']]
        frame = pd.DataFrame([{
            **row['features'], 'id_paiement': row['id'],
            'date_paiement_utc': pd.Timestamp(row['date_utc']),
        } for row in data['observations']])
        (self.root / TEST_DATA).parent.mkdir(parents=True)
        frame.to_parquet(self.root / TEST_DATA, index=False)
        metrics = json.loads((FIXTURES / 'metrics.json').read_text())
        rows = [{
            'mois': row['period'], 'capacite_jour': row['capacity'], 'methode': row['method'],
            'fraudes_controlees_a_temps_pct': row['fraud_on_time'],
            'dossiers_pertinents_pct': row['relevant_cases'],
            'montant_potentiellement_recuperable_pct': row['recoverable_amount'],
            'n_revues': row['cases'],
        } for row in metrics['results'] if row['period'] != 'average']
        self.csv = self.root / '07_artifacts/Full_metriques_simulation_metier.csv'
        pd.DataFrame(rows).to_csv(self.csv, index=False)
        (self.root / '07_artifacts/Full_parametres_export_metier.json').write_text(json.dumps({'date_export_utc': metrics['exported_at']}))

    def test_parquet_to_response_preserves_features_and_missing_values(self):
        """Vérifie les identifiants, la journée, les variables et les valeurs manquantes renvoyés."""
        result = demo_data(self.root)
        self.assertEqual([row['id'] for row in result['observations']], self.ids)
        self.assertEqual(result['day'], '2025-04-21')
        self.assertEqual(set(result['observations'][0]['features']), set(features_selected()))
        self.assertIsNone(result['observations'][0]['features']['pays_ip'])
        json.dumps(result, allow_nan=False)

    def test_metrics_are_reread_after_an_export_changes(self):
        """Vérifie la moyenne mensuelle et la relecture d’un export modifié."""
        first = business_metrics(self.root)
        average = next(row for row in first['results'] if row['period'] == 'average' and row['capacity'] == 10 and row['method'] == 'ML')
        monthly = [row['fraud_on_time'] for row in first['results'] if row['period'] != 'average' and row['capacity'] == 10 and row['method'] == 'ML']
        self.assertAlmostEqual(average['fraud_on_time'], sum(monthly) / 6)
        frame = pd.read_csv(self.csv)
        frame.loc[frame.mois.eq('2025-01') & frame.capacite_jour.eq(10) & frame.methode.eq('ML'), 'fraudes_controlees_a_temps_pct'] = 12
        frame.to_csv(self.csv, index=False)
        updated = business_metrics(self.root)
        january = next(row for row in updated['results'] if row['period'] == '2025-01' and row['capacity'] == 10 and row['method'] == 'ML')
        self.assertEqual(january['fraud_on_time'], 12)

    def test_missing_and_incomplete_exports_are_not_replaced_by_old_data(self):
        """Vérifie que les exports absents ou incomplets provoquent une erreur."""
        (self.root / TEST_DATA).unlink()
        with self.assertRaises(FileNotFoundError):
            demo_data(self.root)
        frame = pd.read_csv(self.csv).iloc[1:]
        frame.to_csv(self.csv, index=False)
        with self.assertRaises(ValueError):
            business_metrics(self.root)


if __name__ == '__main__':
    unittest.main()
