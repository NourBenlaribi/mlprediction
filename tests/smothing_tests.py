import unittest
import pandas as pd

from smoothing_smart_param import smooth


class TestSmoothing(unittest.TestCase):

    def test_smoothing_with_one_column(self):
        df = pd.DataFrame({'serial_number': ['HDD1', 'HDD1', 'HDD1', 'HDD1', 'HDD1', 'HDD1', 'HDD1'],
                           'column1': [39.0, 44, 40, 45, 38, 43, 39]})
        smooth_houssem(df, columns=['column1'])
        self.assertAlmostEqual(39, df.at[0, 'column1'], places=2)
        self.assertAlmostEqual(39, df.at[1, 'column1'], places=2)
        self.assertAlmostEqual(40, df.at[2, 'column1'], places=2)
        self.assertAlmostEqual(40, df.at[3, 'column1'], places=2)
        self.assertAlmostEqual(41, df.at[4, 'column1'], places=2)
        self.assertAlmostEqual(40.4, df.at[5, 'column1'], places=2)
        self.assertAlmostEqual(40.92, df.at[6, 'column1'], places=2)

    def test_smoothing_with_one_column_and_two_hdds(self):
        df = pd.DataFrame({'serial_number': ['HDD1', 'HDD1', 'HDD1', 'HDD1', 'HDD1', 'HDD1', 'HDD1',
                                             'HDD2', 'HDD2', 'HDD2', 'HDD2', 'HDD2', 'HDD2', 'HDD2'],
                           'column1': [39.0, 44, 40, 45, 38, 43, 39,
                                       39.0, 44, 40, 45, 38, 43, 39]})
        smooth_houssem(df, columns=['column1'])
        self.assertAlmostEqual(39, df.at[0, 'column1'], places=2)
        self.assertAlmostEqual(39, df.at[1, 'column1'], places=2)
        self.assertAlmostEqual(40, df.at[2, 'column1'], places=2)
        self.assertAlmostEqual(40, df.at[3, 'column1'], places=2)
        self.assertAlmostEqual(41, df.at[4, 'column1'], places=2)
        self.assertAlmostEqual(40.4, df.at[5, 'column1'], places=2)
        self.assertAlmostEqual(40.92, df.at[6, 'column1'], places=2)

        self.assertAlmostEqual(39, df.at[7, 'column1'], places=2)
        self.assertAlmostEqual(39, df.at[8, 'column1'], places=2)
        self.assertAlmostEqual(40, df.at[9, 'column1'], places=2)
        self.assertAlmostEqual(40, df.at[10, 'column1'], places=2)
        self.assertAlmostEqual(41, df.at[11, 'column1'], places=2)
        self.assertAlmostEqual(40.4, df.at[12, 'column1'], places=2)
        self.assertAlmostEqual(40.92, df.at[13, 'column1'], places=2)

    def test_smoothing_with_two_columns(self):
        df = pd.DataFrame({'serial_number': ['HDD1', 'HDD1', 'HDD1', 'HDD1', 'HDD1', 'HDD1', 'HDD1'],
                           'column1': [39.0, 44, 40, 45, 38, 43, 39], 'column2': [39.0, 44, 40, 45, 38, 43, 39]})
        smooth_houssem(df, columns=['column1', 'column2'])
        self.assertAlmostEqual(39, df.at[0, 'column1'], places=2)
        self.assertAlmostEqual(39, df.at[1, 'column1'], places=2)
        self.assertAlmostEqual(40, df.at[2, 'column1'], places=2)
        self.assertAlmostEqual(40, df.at[3, 'column1'], places=2)
        self.assertAlmostEqual(41, df.at[4, 'column1'], places=2)
        self.assertAlmostEqual(40.4, df.at[5, 'column1'], places=2)
        self.assertAlmostEqual(40.92, df.at[6, 'column1'], places=2)

        self.assertAlmostEqual(39, df.at[0, 'column2'], places=2)
        self.assertAlmostEqual(39, df.at[1, 'column2'], places=2)
        self.assertAlmostEqual(40, df.at[2, 'column2'], places=2)
        self.assertAlmostEqual(40, df.at[3, 'column2'], places=2)
        self.assertAlmostEqual(41, df.at[4, 'column2'], places=2)
        self.assertAlmostEqual(40.4, df.at[5, 'column2'], places=2)
        self.assertAlmostEqual(40.92, df.at[6, 'column2'], places=2)

if __name__ == '__main__':
    unittest.main()