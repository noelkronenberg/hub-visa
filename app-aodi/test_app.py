import unittest
from streamlit.testing.v1 import AppTest

class TestApp(unittest.TestCase):

    def setUp(self):
        """
        Load the app script and create an AppTest object
        """

        self.app_test = AppTest.from_file('app/app.py', default_timeout=60)

    def test_app_runs(self):
        """
        Test if the app runs successfully.
        """

        result = self.app_test.run()
        self.assertTrue(result, "The app did not run successfully")

    def test_model_training(self):
        """
        Test if the model is trained successfully.
        """

        self.app_test.run()
        
        # simulate pressing the 'Update Model' button
        self.app_test.button[0].click().run()
        self.app_test.run()
        
        self.assertIn('rf_classifier', self.app_test.session_state, "Model was not saved in session state")
        self.assertIsNotNone(self.app_test.session_state.rf_classifier, "The model was not trained successfully")

if __name__ == '__main__':
    unittest.main()