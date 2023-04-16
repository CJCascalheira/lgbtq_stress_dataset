"""
Last edited on: Thursday, Apr 15, 2023
@author: Kelsey Corro
"""

import pandas as pd
from os import chdir
# import re

chdir( r'C:\Users\kelse\OneDrive - New Mexico State University\Desktop\NMSU\Research\CMIMS' )


html_df = pd.read_csv( "03_html_data.csv" )
annotated_json_df = pd.read_csv( "06_annotated_json_data.csv" )

html_json_combined_rightjoin = html_df.merge(annotated_json_df, how = 'right')
html_json_combined_leftjoin = html_df.merge(annotated_json_df, how = 'left')

# Save the dataframe as a .csv file
html_json_combined_rightjoin.to_csv("07_CMIMS_html_annotated_json_data-no_null.csv",header=True, index=False, encoding='utf-8')
html_json_combined_leftjoin.to_csv("07_CMIMS_html_annotated_json_data-all_html.csv",header=True, index=False, encoding='utf-8')