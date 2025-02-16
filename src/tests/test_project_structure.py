# test_project_structure.py
import unittest
import tempfile
import os
import yaml
import logging
from project_structure import setup_logging, load_yaml_config, create_directory


class TestProjectStructure(unittest.TestCase):

    def test_setup_logging_console(self):
        config = {'level': 'DEBUG'}
        setup_logging(config)
        self.assertEqual(logging.getLogger().level, logging.DEBUG)

    def test_setup_logging_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, 'test.log')
            config = {'level': 'INFO', 'file': log_file}
            setup_logging(config)
            logging.info("Test message")
            with open(log_file, 'r') as f:
                self.assertIn("Test message", f.read())

    def test_setup_logging_invalid_level(self):
        config = {'level': 'INVALID'}
        setup_logging(config)
        self.assertEqual(logging.getLogger().level, logging.INFO)  # Should use default

    def test_load_yaml_config_valid(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as temp_file:
            yaml.dump({'key': 'value'}, temp_file)
            temp_file_path = temp_file.name
        config = load_yaml_config(temp_file_path)
        self.assertEqual(config, {'key': 'value'})
        os.remove(temp_file_path)

    def test_load_yaml_config_empty(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as temp_file:
            temp_file.write("")  # Write an empty file
            temp_file_path = temp_file.name
        config = load_yaml_config(temp_file_path)
        self.assertEqual(config, {})  # Should return an empty dict
        os.remove(temp_file_path)


    def test_load_yaml_config_invalid(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as temp_file:
            temp_file.write(":")  # Invalid YAML
            temp_file_path = temp_file.name
        config = load_yaml_config(temp_file_path)
        self.assertEqual(config, {})
        os.remove(temp_file_path)


    def test_load_yaml_config_not_found(self):
        config = load_yaml_config('nonexistent_file.yaml')
        self.assertEqual(config, {})

    def test_create_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = os.path.join(temp_dir, 'new_directory')
            create_directory(new_dir)
            self.assertTrue(os.path.exists(new_dir))

    def test_create_existing_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            create_directory(temp_dir)  # Already exists
            self.assertTrue(os.path.exists(temp_dir))



if __name__ == '__main__':
    unittest.main()