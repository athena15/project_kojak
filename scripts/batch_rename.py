import os
from collections import defaultdict


path_1 = ['/Users/brenner/project_kojak/frames/drawings'] * 6
path_2 = ['/Users/brenner/project_kojak/frames/silhouettes'] * 6

prefixes = ['C_', 'fist_', 'L_', 'okay_', 'palm_', 'peace_']
drawing_prefixes = [f'drawing_{prefix}' for prefix in prefixes]
new_prefixes = {'drawing_C_': 'C_', 'drawing_fist_': 'fist_', 'drawing_L_':'L_', 'drawing_okay_':'okay_', 'drawing_palm_': 'palm_', 'drawing_peace_': 'peace_'}

def rename_strings(path, leading_string):
	os.chdir(path)
	counter = 0

	for f in os.listdir(path):
		if f.startswith(leading_string):
			counter += 1
			new_prefix = new_prefixes[leading_string]
			formatted_counter = str(counter).zfill(3)
			os.rename(f, f.replace(f, f'{new_prefix}{formatted_counter}' + '.jpg'))


	print(f'Counted {counter} files.')


def count_images(path):
	counts = defaultdict(int)
	for prefix in drawing_prefixes: #replace with prefixes or drawing_prefixes
		for f in os.listdir(path):
			if f.startswith(prefix):
				counts[prefix] += 1
	print(counts.items())


print(count_images(path_1[0]))

# # ***CALLS TO RENAME ALL FILES BELOW***
# for tup in zip(path_2, prefixes):
# 	rename_strings(tup[0], tup[1])
#
for tup in zip(path_1, drawing_prefixes):
	rename_strings(tup[0], tup[1])

print(count_images(path_1[0]))
