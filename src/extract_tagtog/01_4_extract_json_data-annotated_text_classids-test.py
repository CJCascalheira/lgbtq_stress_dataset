"""
Last edited on: Thursday, April 15, 2023
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
class_id_list = []
annotated_text_list = []

# directory[5] and filenames [5] will show sample Cory referred to
chdir( directory[5] )
print( "Changed Directory to:", directory[5] )
print()

filename = filenames[5]
a = re.split(".ann.json", filename)
file_id = a[0]
# do not append file_id to list unless we confirm there is text in file
# file_id_list.append( file_id )
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

# count how many times 'classId' appears
# count how many times 'text' appears
# verify that it is the same amount of times
# if no text appears next to classid, it will look like:
    # "text": "temp_id"

text_count = string.count('text')
print(text_count)
classId_count = string.count('classId')
print(classId_count)
text_tempid_count = string.count('text\": \"temp_id')
print(text_tempid_count)
print()

# use a counter for while loop to extract all ann
count = text_count - text_tempid_count

# if text_count == text_tempid_count, then we are not interested in the file
# because it does not have any annotated text
# if text_count != text_tempid_count, 
    # then we want to extract the annotated text
    
if ( text_count != text_tempid_count ):
    x = 1
    while ( count != 0 ):
        file_id_list.append( file_id )
        
        b = re.split("classId\": \"", string)
        class_id = re.split("\", \"part\"",b[x])[0]
        print( class_id )
        class_id_list.append( class_id )
        
        c = re.split("\"text\": \"", string)
        annotated_text = re.split("\"}],",c[x])[0]
        print( annotated_text, "\n" )
        annotated_text_list.append( annotated_text )
        x += 1
        count -= 1


# b = re.split("classId\": \"", string)
# class_id = re.split("\", \"part\"",b[1])[0]
# print( class_id )
# class_id_list.append( class_id )

# c = re.split("\"text\": \"", string)
# annotated_text = re.split("\"}],",c[1])[0]
# print( annotated_text )
# annotated_text_list.append( annotated_text )

''' We need to extract: 
    class_id = re.split("\", \"part\"",b[2])[0]
    annotated_text = re.split("\"}],",c[2])[0]
    class_id = re.split("\", \"part\"",b[3])[0]
    annotated_text = re.split("\"}],",c[3])[0]
    '''


#Change 100 to len(df)
# for x in range( 10 ):
#     chdir( directory[x] )
#     print( "Changed Directory to:", directory[x] )
#     print()
    
#     filename = filenames[x]
#     a = re.split(".ann.json", filename)
#     file_id = a[0]
#     file_id_list.append( file_id )
#     print( file_id )
#     print()
    
#     # open the json file
#     f = open( filename, encoding='utf-8' )
    
#     # contents will be dictionary object
#     data = json.load( f )
    
#     # we are interested in labels contained in entities
#     # we will convert the contents of the entities key into a string
#     # so that it will be easier to search for labels
#     string = json.dumps( data['entities'] )

#     print( string )
#     print()
