import unittest

if __name__ == '__main__':
    # Define a test suite using the default test loader's discovery method
    suite = unittest.defaultTestLoader.discover('.', pattern='test_*.py')
    
    # Create a test runner that will run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
