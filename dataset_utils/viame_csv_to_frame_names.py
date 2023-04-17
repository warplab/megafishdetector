import sys
import datetime
import pandas as pd

DATE_FORMAT_STR = "%H:%M:%S.%f"
FRAME_NUM_STR = "frame%06d.png"

viame_csv_file = sys.argv[1]
viame_output_csv_file = sys.argv[2]

# VIAME CSVs are poorly formatted, we throw out arbitrary comment lines at the end of rows
# TODO: attempt to maintain these comments rows for future species-specific data files
viame_csv = pd.read_csv(viame_csv_file, usecols=list(range(11)))

img_name_col = list(viame_csv.columns).index("2: Video or Image Identifier")

fps = int(viame_csv.iloc[0,img_name_col].strip("fps: "))

t0 = datetime.datetime.strptime("00:00:00.000000","%H:%M:%S.%f")

datetime_to_frame = [int((datetime.datetime.strptime(x, DATE_FORMAT_STR)-t0).total_seconds()*fps) for x in viame_csv.iloc[1:,img_name_col]]

frame_to_str = [FRAME_NUM_STR%(frame) for frame in datetime_to_frame]

viame_csv.iloc[1:,img_name_col] = frame_to_str
#viame_csv.to_csv("%s.frames.csv"%(viame_csv_file[:-4]))
viame_csv.to_csv(viame_output_csv_file, index=False)
