In order to extract all the desired data, FolderTree (which can be found from this website:
https://www.digitalcitizen.life/how-export-directory-tree-folder-windows/
and downloaded using the following link:
https://1drv.ms/u/s!AslOfY4IzwvrvPo3ThZMPv2M7D6Rwg?e=5vta1s/ ) was used to obtain all the file paths and filenames.

The executable file was placed in the folder containing all the .json files and all the .html files.
It was then run to extract .csv files initally named 'filetree.csv'.
This .csv files were renamed to 'json_files.csv' and 'html_files.csv' according to which folder they were extracted from.

Afterward, the python script '01_1_extract_html_data-final.py' was ran.
	
	Using regular expression, the file_id, temp_id, and text was extracted.

	file_id was extracted by taking the file name and extracting any characters before ".plain.html".

	temp_id was extracted by converting the contents of the html file into a string and obtaining
	the contents between the two strings:
		<pre id="s1s1v1">
		</pre>
		
		Example: <pre id="s1s1v1">28038</pre>
			28038 is the temp_id because it is between the strings specified above

	text was extracted by converting the contents of the html file into a string and obtaining
	the contents between the two strings:
		<pre id="s1s2v1">
		</pre>

		Example: <pre id="s1s2v1">Hi All,...</pre>
			Hi All,... is the text because it is between the strings specified above

	Once all file_ids, temp_ids, and texts were extracted from all instances,
	the contents were placed into a dataframe and converted into a .csv file called, "03_html_data.csv".

Afterward, the python script '01_2_extract_json_data-final' was ran.

	We were interested in seeing if the class ids e_1, e_2, e_3, e_4, e_5, e_6, e_7, e_8, and e_9 existed
	in the 'entities' section of the .json file.
	If the class id existed, it was assigned with a class label 1.
	If the class id did not exist, it was assigned with a class label 0.

	Once that information was extracted, an additional attribute 'minority_stress' was added.
	If an instance was assigned 1 for class ids e_3, e_4, e_5, or e_6, 
	minority_stress was assigned with the class label 1.
	
	After obtaining the desired information, the contents were placed into a dataframe
	and converted into a .csv file called, "04_json_data.csv".

Lastly, the python script '01_3_combine_html_json.py' was ran.

	This script combined the contents of "03_html_data.csv" and "04_json_data.csv".
	The two files were merged by matching the file_ids (the common attribute).

	The output are two files:
		- '05_CMIMS_html_json_data-all_html.csv'
			- There were more .html files than .json files
			- This file contains all .html files
				- For any file with no matching .json file, the contents of class ids e_1 to e_9
				  and minority_stress were assigned NaN or null
		- '05_CMIMS_html_json_data-no_null.csv'
			- This file only contains instances where a .html file had a matching .json file.