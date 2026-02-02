import unittest
from unittest.mock import MagicMock, patch
import logic
import health_logic
import drift_logic

class TestGroqIntegration(unittest.TestCase):

    @patch('logic.OpenAI')
    def test_logic_diagnose_failure(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices[0].message.content = "Groq Diagnosis"

        diagnosis = logic.diagnose_failure({}, {}, [], api_key_input="test_key", model_name="llama3-70b-8192")
        
        mock_openai.assert_called_with(base_url="https://api.groq.com/openai/v1", api_key="test_key")
        mock_client.chat.completions.create.assert_called()
        self.assertEqual(diagnosis, "Groq Diagnosis")

    @patch('health_logic.OpenAI')
    def test_health_generate_summary(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices[0].message.content = "Groq Health Summary"

        summary = health_logic.generate_health_summary({'missing':{}, 'imbalance':{}, 'outliers':{}, 'stability':{}, 'summary': {'rows': 100, 'cols': 5, 'total_missing': 0}}, 100, "Healthy", api_key_input="test_key")
        
        mock_openai.assert_called_with(base_url="https://api.groq.com/openai/v1", api_key="test_key")
        self.assertEqual(summary, "Groq Health Summary")

    @patch('drift_logic.OpenAI')
    def test_drift_explain(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices[0].message.content = "Groq Explanation"

        report = {
            "summary": {"ref_shape": (100, 2), "curr_shape": (100, 2), "ref_missing": 0, "curr_missing": 0},
            "numerical_drift": [{"drift_detected": True, "feature": "f1"}],
            "categorical_drift": []
        }
        explanation = drift_logic.explain_drift(report, api_key_input="test_key")
        
        mock_openai.assert_called_with(base_url="https://api.groq.com/openai/v1", api_key="test_key")
        self.assertEqual(explanation, "Groq Explanation")

if __name__ == '__main__':
    unittest.main()
