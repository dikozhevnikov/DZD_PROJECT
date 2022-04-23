import pandas as pd

def get_histogram(clm, rule_id):
  target = clm.kwargs["target"]
  df = clm.kwargs["df"]
  keys = sorted(df[target].unique())
  values = clm.result["rules"][rule_id-1]["params"]["hist"]
  hist = pd.Series(values, keys)
  return hist
  
def plot_histogram(clm, rule_id):
  hist = get_histogram(clm, rule_id)
  hist.plot(kind = "bar")
