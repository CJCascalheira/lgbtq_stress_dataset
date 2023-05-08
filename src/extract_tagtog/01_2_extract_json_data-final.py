"""
Last edited on: Thursday, April 13, 2023
@author: Kelsey Corro
"""

import pandas as pd
from os import chdir
import re
import json

# This csv contains all the file paths and files that we are trying to extract data from
df = pd.read_csv(r'C:\Users\kelse\OneDrive - New Mexico State University\Desktop\NMSU\Research\CMIMS\02_json_files.csv')
directory = df[df.columns[0]]
filenames = df[df.columns[1]]

file_id_list = []
e_1_list = []
e_2_list = []
e_3_list = []
e_4_list = []
e_5_list = []
e_6_list = []
e_7_list = []
e_8_list = []
e_9_list = []

# Change 100 to len(df)
for x in range( len(df) ):
    chdir( directory[x] )
    print( "Changed Directory to:", directory[x] )
    print()
    
    filename = filenames[x]
    a = re.split(".ann.json", filename)
    file_id = a[0]
    file_id_list.append( file_id )
    print( file_id )
    print()
    
    # open the json file
    f = open( filename, encoding='utf-8' )
    
    # contents will be dictionary object
    data = json.load( f )
    
    # we are interested in labels contained in entities
    # we will convert the contents of the entities key into a string
    # so that it will be easier to search for labels
    string = json.dumps( data['entities'] )
    
    print( string )
    print()
    if "e_1" in string:
        print( "e_1 found\n")
        e_1_list.append( 1 )
    else:
        print( "e_1 not found\n")
        e_1_list.append( 0 )
        
    if "e_2" in string:
        print( "e_2 found\n")
        e_2_list.append( 1 )
    else:
        print( "e_2 not found\n")
        e_2_list.append( 0 )
        
    if "e_3" in string:
        print( "e_3 found\n")
        e_3_list.append( 1 )
    else:
        print( "e_3 not found\n")
        e_3_list.append( 0 )
        
    if "e_4" in string:
        print( "e_4 found\n")
        e_4_list.append( 1 )
    else:
        print( "e_4 not found\n")
        e_4_list.append( 0 )
        
    if "e_5" in string:
        print( "e_5 found\n")
        e_5_list.append( 1 )
    else:
        print( "e_5 not found\n")
        e_5_list.append( 0 )
        
    if "e_6" in string:
        print( "e_6 found\n")
        e_6_list.append( 1 )
    else:
        print( "e_6 not found\n")
        e_6_list.append( 0 )
        
    if "e_7" in string:
        print( "e_7 found\n")
        e_7_list.append( 1 )
    else:
        print( "e_7 not found\n")
        e_7_list.append( 0 )
        
    if "e_8" in string:
        print( "e_8 found\n")
        e_8_list.append( 1 )
    else:
        print( "e_8 not found\n")
        e_8_list.append( 0 )
    
    if "e_9" in string:
        print( "e_9 found\n")
        e_9_list.append( 1 )
    else:
        print( "e_9 not found\n")
        e_9_list.append( 0 )
        
    print("-----------------------------------------------------------------")
    print()
    
    f.close()

# Create a dataframe with the data that has been extracted
json_data_df = pd.DataFrame(
    {'File_id': file_id_list,
     'e_1': e_1_list,
     'e_2': e_2_list,
     'e_3': e_3_list,
     'e_4': e_4_list,
     'e_5': e_5_list,
     'e_6': e_6_list,
     'e_7': e_7_list,
     'e_8': e_8_list,
     'e_9': e_9_list
     })

# Create a new list that will contain minority stress label
minority_stress_list = []

# Iterate through each instance of the dataframe
# if any of the class ids e_3, e_4, e_5, or e_6 have a 1
# then assign 1 to minority_stress
for x in range ( len(df) ):
    if ( json_data_df['e_3'][x] == 1 or json_data_df['e_4'][x] == 1 or
         json_data_df['e_5'][x] == 1 or json_data_df['e_6'][x] == 1 ):
        minority_stress_list.append( 1 )
    else: 
        minority_stress_list.append( 0 )

# Create a dataframe with all information extracted and add minority_stress
json_data_df = pd.DataFrame(
    {'File_id': file_id_list,
     'e_1': e_1_list,
     'e_2': e_2_list,
     'e_3': e_3_list,
     'e_4': e_4_list,
     'e_5': e_5_list,
     'e_6': e_6_list,
     'e_7': e_7_list,
     'e_8': e_8_list,
     'e_9': e_9_list,
     'minority_stress': minority_stress_list
     })

# Change directory to where you want to save the .csv file
chdir( r'C:\Users\kelse\OneDrive - New Mexico State University\Desktop\NMSU\Research\CMIMS' )

# Save the dataframe as a .csv file
json_data_df.to_csv("04_json_data.csv",header=True, index=False, encoding='utf-8')