import os
from collections import defaultdict


path_1 = ['/Users/brenner/project_kojak/frames/drawings'] * 6
path_2 = ['/Users/brenner/project_kojak/frames/silhouettes'] * 6

prefixes = ['C_', 'fist_', 'L_', 'okay_', 'palm_', 'peace_']

def rename_strings(path, leading_string):
	os.chdir(path)
	counter = 0

	for f in os.listdir(path):
		if f.startswith(leading_string):
			counter += 1
			formatted_counter = str(counter).zfill(3)
			os.rename(f, f.replace(f, f'{leading_string}{formatted_counter}' + '_.jpg'))


	print(f'Counted {counter} files.')


def count_images(path):
	counts = defaultdict(int)
	for prefix in drawing_prefixes: #replace with prefixes or drawing_prefixes
		for f in os.listdir(path):
			if f.startswith(prefix):
				counts[prefix] += 1
	print(counts.items())


# # ***CALLS TO RENAME ALL FILES BELOW***
# for tup in zip(path_2, prefixes):
# 	rename_strings(tup[0], tup[1])
#
# for tup in zip(path_1, prefixes):
# 	rename_strings(tup[0], tup[1])

print(count_images(path_1[0]))
