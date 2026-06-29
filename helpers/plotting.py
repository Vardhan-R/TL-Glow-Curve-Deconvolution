from uuid import uuid4
import altair as alt
import pandas as pd
import numpy as np


def plotLineGraph(x: np.ndarray, y: np.ndarray, colour: str, x_label: str = "x", y_label: str = "y") -> alt.Chart:
    source = pd.DataFrame({x_label: x, y_label: y})
    return alt.Chart(source).mark_line(color=colour).encode(x=x_label, y=y_label).interactive(name=str(uuid4()))


def plotScatterGraph(x: np.ndarray, y: np.ndarray, colour: str, x_label: str = "x", y_label: str = "y") -> alt.Chart:
    source = pd.DataFrame({x_label: x, y_label: y})
    return alt.Chart(source).mark_circle(color=colour, size=60).encode(x=x_label, y=y_label).interactive(name=str(uuid4()))
