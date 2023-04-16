"""
Last edited on: Thursday, Apr 13, 2023
@author: Kelsey Corro
"""

import pandas as pd
from os import chdir
import re

# This csv contains all the file paths and files that we are trying to extract data from
df = pd.read_csv(r'C:\Users\kelse\OneDrive - New Mexico State University\Desktop\NMSU\Research\CMIMS\01_html_files.csv')
directory = df[df.columns[0]]
filenames = df[df.columns[1]]


file_id_list = []
temp_id_list = []
text_list = []

for x in range( len(df) ):
    chdir( directory[x] )
    print( "Changed Directory to:", directory[x] )
    print()
    
    filename = filenames[x]
    a = re.split(".plain.html", filename)
    file_id = a[0]
    file_id_list.append( file_id )
    print( file_id )
    print()
    
    # Open the html file
    HTMLFile = open(filename, "r", encoding='utf-8')

    # Read the html file
    index = HTMLFile.read()

    # Extract the temp_id using regular expression
    # temp_id is the numbers between s1s1v1\"> and </pre>
    b = re.split("s1s1v1\">", index)
    temp_id = re.split("</pre>",b[1])[0]
    temp_id_list.append( temp_id )
    print(temp_id)
    print()

    # Extract the text using regular expression
    # text is the string of all characters between s1s2v1\"> and </pre>
    c = re.split("s1s2v1\">", index)
    text = re.split("</pre>",c[1])[0]
    text_list.append( text )
    print(text)
    print("-------------------------------------------------------------")
    print()

    HTMLFile.close()

# Create a dataframe with the data that has been extracted
html_data_df = pd.DataFrame(
    {'File_id': file_id_list,
     'Temp_id': temp_id_list,
     'Text': text_list
     })

# Change directory to where you want to save the .csv file
chdir( r'C:\Users\kelse\OneDrive - New Mexico State University\Desktop\NMSU\Research\CMIMS' )

# Save the dataframe as a .csv file
html_data_df.to_csv("03_html_data.csv",header=True, index=False, encoding='utf-8')