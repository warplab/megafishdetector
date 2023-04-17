import pandas as pd

def parse_viame_annotations_csv(csv_path):
    # Some Viame CSVs are improperly formatted, we ignore that data here by just reading the first 11 columns
    df = pd.read_csv(join(dir_path, "annotations.viame.csv"), usecols=list(range(11)))
    
