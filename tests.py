import unittest
import os
import json
from app import app

class AutoMLTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.app = app.test_client()
        self.test_file_path = 'test_data_clf.csv'

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'AutoML', response.data)

    def test_upload_flow(self):
        with open(self.test_file_path, 'rb') as f:
            data = {
                'file': (f, 'test_data_clf.csv'),
                'target': 'Target'
            }
            response = self.app.post('/upload', data=data, content_type='multipart/form-data')
            
            self.assertEqual(response.status_code, 200)
            
            json_data = json.loads(response.data)
            self.assertIn('best_model', json_data)
            self.assertIn('leaderboard', json_data)
            self.assertEqual(json_data['problem_type'], 'Classification')
            
            print(f"\nBest Model: {json_data['best_model']}")
            print(f"Best Score: {json_data['best_score']}")

    def test_regression_flow(self):
        with open('test_data_reg.csv', 'rb') as f:
            data = {
                'file': (f, 'test_data_reg.csv'),
                'target': 'Price'
            }
            response = self.app.post('/upload', data=data, content_type='multipart/form-data')
            
            self.assertEqual(response.status_code, 200)
            
            json_data = json.loads(response.data)
            self.assertEqual(json_data['problem_type'], 'Regression')
            self.assertIn('best_model', json_data)
            
            print(f"\n[Regression] Best Model: {json_data['best_model']}")
            print(f"[Regression] Best Score (R2): {json_data['best_score']}")

if __name__ == '__main__':
    unittest.main()

