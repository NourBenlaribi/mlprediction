import unittest
import pandas as pd

from pre_processing import flag_failures, clean_data


class TestFlagFailures(unittest.TestCase):

    def test_flag_failures_when_no_failed_hdds(self):
        df = pd.DataFrame({'serial_number': ['HDD1', 'HDD2'], 'failure': [0, 0]})
        flag_failures(df)
        self.assertEqual(df.at[0, 'failure'], 0)
        self.assertEqual(df.at[0, 'serial_number'], 'HDD1')
        self.assertEqual(df.at[1, 'failure'], 0)
        self.assertEqual(df.at[1, 'serial_number'], 'HDD2')

    def test_flag_failures_when_failed_hdds(self):
        df = pd.DataFrame({'serial_number': ['HDD1', 'HDD1', 'HDD2'], 'failure': [1, 0, 0]})
        flag_failures(df, time_window_size=10)
        self.assertEqual(df.at[0, 'failure'], 1)
        self.assertEqual(df.at[0, 'serial_number'], 'HDD1')
        self.assertEqual(df.at[1, 'failure'], 1)
        self.assertEqual(df.at[1, 'serial_number'], 'HDD1')
        self.assertEqual(df.at[2, 'failure'], 0)
        self.assertEqual(df.at[2, 'serial_number'], 'HDD2')

    def test_clean_data(self):
        df = pd.DataFrame({'serial_number': ['HDD1', 'HDD1', 'HDD1', 'HDD2', 'HDD2', 'HDD2'], 'failure': [1, 0, 0, 0, 0, 0]})
        df = clean_data(df, to_keep_healthy=1, to_keep_failure=1)
        print(df)
        self.assertEqual(df.at[0, 'serial_number'], 'HDD1')
        self.assertEqual(df.at[1, 'serial_number'], 'HDD1')
        self.assertEqual(df.at[2, 'serial_number'], 'HDD2')
        self.assertEqual(df.at[3, 'serial_number'], 'HDD2')
        self.assertEqual(df.shape[0], 4)


if __name__ == '__main__':
    unittest.main()