import numpy  as np
import pandas as pd

class DataFrameSimulator():
    def __init__(self, base_df:pd.DataFrame):
        self._base_df = base_df

    @property
    def base_df(self) -> pd.DataFrame:
        return self._base_df

    def characterize(self):
        self._column_models = {}

        for col_name in self.base_df.columns:

            col_series = self.base_df[col_name]
            col_type   = str(self.base_df.dtypes[col_name]).lower()

            print(f'{col_name}\t\t{col_type}')

            #FIXME: can't get type comparison to work, falling back to string compare type name
            if col_type.startswith('bool'):
                model = BoolModel(col_name, col_series)
            elif col_type.startswith('int'):
                model = IntModel(col_name, col_series)
            elif col_type.startswith('float'):
                model = FloatModel(col_name, col_series)
            elif col_type.startswith('str') or col_type.startswith('obj'):
                model = StrModel(col_name, col_series)
            else:
                model = DefaultModel(col_name, col_series)
            
            self._column_models[col_name] = model

    def simulate(self, rows) -> pd.DataFrame:
        empty_df = pd.DataFrame( {c : pd.Series(dtype=dt)
                                for c,dt in zip(self.base_df.columns, self.base_df.dtypes)
                                })

        sim_df = pd.DataFrame( { c: self._column_models[c].series(rows)
                                 for c in self.base_df.columns })

        return sim_df
    


class BoolModel():
    def __init__(self, name, base_series):
        self.name = name
        counts    = base_series.value_counts()
        t_cnt     = counts[True]
        f_cnt     = counts[False]

        self.probability_true = t_cnt/(t_cnt + f_cnt)

    def series(self, size=1) -> pd.Series:
        array = np.random.choice(a    = [True, False],
                                 p    = [self.probability_true,
                                         1-self.probability_true],
                                 size = size)
        ser = pd.Series(array, dtype=np.bool)
        ser.name = self.name

        return ser



class NumberModel():
    def __init__(self, name, base_series):
        self.name = name
        self.mean = base_series.mean()
        self.std  = base_series.std()

    def series(self, size=1) -> pd.Series:
        array = np.random.normal(loc  = self.mean,
                                 scale = self.std,
                                 size = size)
        ser = pd.Series(array, dtype=self.dtype)
        ser.name = self.name

        return ser



class FloatModel(NumberModel):
    def __init__(self, name, base_series):
        self.dtype = np.float64
        super().__init__(name, base_series)



class IntModel(NumberModel):
    def __init__(self, name, base_series):
        self.dtype = np.int64
        super().__init__(name, base_series)

    def series(self, size):
        ser = super().series(size)
        ser = np.ceil(ser).astype(self.dtype)

        return ser


class StrModel():
    def __init__(self, name, base_series):
        self.name      = name
        counts         = base_series.value_counts()
        total          = sum(counts)
        self.freq_dict = {entry: counts[entry]/total for entry in counts.keys()}

    def series(self, size=1) -> pd.Series:
        entries     = list(self.freq_dict.keys())
        frequencies = list(self.freq_dict.values())
        array       = np.random.choice( a    = entries,
                                        p    = frequencies,
                                        size = size)
        ser = pd.Series(array, dtype=str)
        ser.name = self.name

        return ser


class DefaultModel():
    def __init__(self, name, base_series):
        self.name        = name
        self.base_series = base_series

    def series(self, size=1) -> pd.Series:
        array = np.random.choice(a    = self.base_series,
                                 size = size)
        ser = pd.Series(array, dtype=self.base_series.dtype)
        ser.name = self.name

        return ser