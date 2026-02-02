import unittest
from unittest.mock import MagicMock, patch
import logic
import health_logic
import drift_logic

class TestGeminiIntegration(unittest.TestCase):

    @patch('logic.genai')
    def test_logic_diagnose_failure(self, mock_genai):
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.return_value.text = "Diagnosis"

        diagnosis = logic.diagnose_failure({}, {}, [], api_key_input="test_key")
        
        mock_genai.configure.assert_called_with(api_key="test_key")
        mock_genai.GenerativeModel.assert_called_with('gemini-1.5-pro')
        mock_model.generate_content.assert_called()
        self.assertEqual(diagnosis, "Diagnosis")

    @patch('health_logic.genai')
    def test_health_generate_summary(self, mock_genai):
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.return_value.text = "Health Summary"

        summary = health_logic.generate_health_summary({'missing':{}, 'imbalance':{}, 'outliers':{}, 'stability':{}, 'summary': {'rows': 100, 'cols': 5, 'total_missing': 0}}, 100, "Healthy", api_key_input="test_key")
        
        mock_genai.configure.assert_called_with(api_key="test_key")
        mock_model.generate_content.assert_called()
        self.assertEqual(summary, "Health Summary")

    @patch('drift_logic.genai')
    def test_drift_explain(self, mock_genai):
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_model.generate_content.return_value.text = "Drift Explanation"

        report = {
            "summary": {"ref_shape": (100, 2), "curr_shape": (100, 2), "ref_missing": 0, "curr_missing": 0},
            "numerical_drift": [{"drift_detected": True, "feature": "f1"}],
            "categorical_drift": []
        }
        explanation = drift_logic.explain_drift(report, api_key_input="test_key")
        
        mock_genai.configure.assert_called_with(api_key="test_key")
        mock_model.generate_content.assert_called()
        self.assertEqual(explanation, "Drift Explanation")

if __name__ == '__main__':
    unittest.main()
