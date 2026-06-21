import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add scripts directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

class TestGarmentIQPipeline(unittest.TestCase):
    
    @patch('run_garmentiq.giq')
    def test_tailor_agent_initialization(self, mock_giq):
        """
        Tests that the get_tailor_agent function correctly passes the required
        arguments to the GarmentIQ tailor agent even if the heavy models are not loaded.
        """
        from run_garmentiq import get_tailor_agent
        
        # Setup mock
        mock_agent = MagicMock()
        mock_giq.tailor.return_value = mock_agent
        
        # We need to mock the external dependencies that aren't available when garmentiq isn't installed
        with patch('run_garmentiq.garment_classes', {'shirt': 0, 'trouser': 1}), \
             patch('run_garmentiq.derivation_dict', {'shirt': {}}), \
             patch('run_garmentiq.tinyViT', MagicMock()), \
             patch('run_garmentiq.BiRefNet', MagicMock()), \
             patch('run_garmentiq.PoseHighResolutionNet', MagicMock()), \
             patch('run_garmentiq.load_birefnet_config', return_value={}):
             
            agent = get_tailor_agent(input_dir="test_in", model_dir="test_models", output_dir="test_out")
            
            # Verify the agent was created
            self.assertEqual(agent, mock_agent)
            
            # Verify tailor was called with our expected local directories
            mock_giq.tailor.assert_called_once()
            call_kwargs = mock_giq.tailor.call_args.kwargs
            
            self.assertEqual(call_kwargs['input_dir'], "test_in")
            self.assertEqual(call_kwargs['model_dir'], "test_models")
            self.assertEqual(call_kwargs['output_dir'], "test_out")
            
            # Verify some core configuration exists
            self.assertTrue(call_kwargs['do_derive'])
            self.assertTrue(call_kwargs['do_refine'])
            self.assertEqual(call_kwargs['classification_model_path'], "tiny_vit_inditex_finetuned.pt")

if __name__ == '__main__':
    unittest.main()
