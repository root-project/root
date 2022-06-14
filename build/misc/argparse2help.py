import sys
import importlib
import os

def getLongest():
	longestSize = 0
	for arg in listArgs:
		if (len(arg.option_strings)==0):
			size = len(arg.dest)
		else:
			size = len(", ".join(arg.option_strings))
		longestSize = max(longestSize, size)
	return longestSize

def write_header(parser, fileName):
	longestSize = getLongest()
	file= open(fileName, "w+")
	splitPath = sys.argv[2].split("/")
	file.write("#ifndef ROOT_{}\n".format(splitPath[len(splitPath)-1].partition(".")[0]))
	file.write("#define ROOT_{}\n".format(splitPath[len(splitPath)-1].partition(".")[0]))
	file.write("constexpr static const char kCommandLineOptionsHelp[] = R\"RAW(\n")
	file.write(parser.format_usage() + "\n")
	file.write("OPTIONS:\n")
	for arg in listArgs:
		options = ""
		help = arg.help
		if (len(arg.option_strings)==0):
			listOptions = [arg.dest]
		else:
			listOptions = arg.option_strings
		options = ", ".join(listOptions)
		spaces = " " * (12 + longestSize - len(options))
		if help != None:
			help = help.replace("\n", "\n  {}".format(" "*(len(options)) + spaces))
			file.write("  {}{}{}\n".format(options, spaces, help))
		else:
			file.write("  {}\n".format(options))
	file.write(")RAW\";\n")
	file.write("#endif\n")
	file.close()

def write_man(parser, fileName):
	file= open(fileName, "w+")
	file.write(".TH {} 1 \n".format(parser.prog))
	file.write(".SH SYNOPSIS\n")
	file.write(parser.format_usage() + "\n")
	file.write(".SH DESCRIPTION\n")
	file.write(parser.description + "\n")
	file.write(".SH OPTIONS\n")
	for arg in listArgs:
		options = ""
		help = arg.help
		if (len(arg.option_strings)==0):
			listOptions = [arg.dest]
		else:
			listOptions = arg.option_strings
		options = "\ ".join(listOptions)
		if help != None:
			file.write(".IP {}\n".format(options))
			file.write(help.replace("\n","\n.IP\n")+ "\n")
		else:
			file.write(".IP {}\n\n".format(options))
	file.close()

if __name__ == "__main__":
	path = os.path.expanduser(sys.argv[1])
	sys.path.insert(0, os.path.abspath(os.path.dirname(path)))
	i = importlib.import_module(os.path.splitext(os.path.basename(path))[0])
	parser = i.get_argparse()
	listArgs = parser._actions
	ext = sys.argv[2][-2:]
	if (ext == ".h"):
		write_header(parser, sys.argv[2])
	elif (ext == ".1"):
		write_man(parser, sys.argv[2])
