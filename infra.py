import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def get_first_empty_line(filename: Path) -> int:
    i = 0
    with open(filename) as file:
        while line := file.readline():
            i += 1
            if line == '\n':
                break
    return i


def read_csv_skip_last_lines(filename: Path) -> pd.DataFrame:
    empty_line_number = get_first_empty_line(filename)
    raw_df = pd.read_csv(filename, sep=r'\s+', skiprows=1, header=None, nrows=empty_line_number - 2)
    return raw_df.rename(columns={0 : 'wavelenght', 1 : 'n', 2 : 'k'})


def plot_nk(filename: Path) -> None:
    silicate_dust = pd.read_csv(filename)
    sns.lineplot(data=silicate_dust, x='wavelenght', y='n', label='n')
    sns.lineplot(data=silicate_dust, x='wavelenght', y='k', label='k')
    plt.legend()
    plt.xscale('log')
