import re
from sys import argv
from os import walk
import gc

trace = '\n    std::cerr << "error" << std::endl;'


def recursive_remove_traces(path = '.'):
	for (dirpath, dirnames, filenames) in walk(path):
		for directory in dirnames:
			if directory[0] != '.':
				recursive_remove_traces(path + '/' + directory)
		for file in filenames:
			if file[-2:] == '.c' or file[-4:] == '.cpp':
				print(path + '/' + file)
				remove_traces(path + '/' + file)
		break

def remove_traces(path):
	file = open(path)
	code = file.read()
	output = open(path, 'w')

	trace_match = '\n.*std::cerr\s+<<\s+\"' + path.replace('.', '\.') + '.*\n'
	code = re.sub(trace_match, '', code)
	output.write(code)

	del code
	gc.collect()
	file.close()
	output.close()

def recursive_add_traces(path = '.'):
	for (dirpath, dirnames, filenames) in walk(path):
		for directory in dirnames:
			if directory[0] != '.':
				recursive_add_traces(path + '/' + directory)
		for file in filenames:
			if file[-2:] == '.c' or file[-4:] == '.cpp':
				print(path + '/' + file)
				add_traces(path + '/' + file)
		break

def add_traces(path, display = False):
	file = open(path)
	code = file.read()
	output = open(path, 'w')
	if 'include <iostream>' not in code: output.write('#include <iostream>\n')

	regex = re.compile(r'(\n.*\)\s*\{)')
	while 1:
		m = regex.search(code)
		if m:
			if display: print(m.group())
			position = m.end()
			str_out = path + ": " + m.group().replace("\"", "'").replace("\n", " ")
			output.write(code[:position] + trace.replace("error", str_out))
			code = code[position:]
			gc.collect()
		else:
			output.write(code)
			break
	del code
	gc.collect()
	file.close()
	output.close()


recursive_add_traces('./inference-engine/src/inference_engine')
recursive_add_traces('./inference-engine/src/mkldnn_plugin')
recursive_add_traces('./inference-engine/thirdparty/mkl-dnn/src')

#recursive_remove_traces('./inference-engine/src/inference_engine')
#recursive_remove_traces('./inference-engine/src/mkldnn_plugin')
#recursive_remove_traces('./inference-engine/thirdparty/mkl-dnn/src')

#example
#recursive_add_traces('./inference-engine/src/inference_engine/builders')
#add_traces('./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp')
#recursive_remove_traces('./inference-engine/src/inference_engine/builders')
#remove_traces('./inference-engine/src/inference_engine/builders/ie_concat_layer.cpp')

