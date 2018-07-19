import sys
import importlib

def getLongest():
	longestSize = 0
	for arg in i.myAction.listArgs:
		if (len(arg.get("option_strings"))==0):
			size = len(arg.get("dest"))
		else:
			size = len(", ".join(arg.get("option_strings")))
		longestSize = max(longestSize, size)
	return longestSize
	
def write_header(parser, fileName):
	longestSize = getLongest() 
	file= open(fileName, "w+")
	file.write("#ifndef ROOT_{}\n".format(sys.argv[2].partition(".")[0]))
	file.write("#define ROOT_{}\n".format(sys.argv[2].partition(".")[0]))
	file.write("constexpr static const char kCommandLineOptionsHelp[] = R\"RAW(\n")
	file.write(parser.format_usage() + "\n")
	file.write("OPTIONS:\n")
	for arg in i.myAction.listArgs:
		options = ""
		help = arg.get("help") 
		if (len(arg.get("option_strings"))==0):
			listOptions = [arg.get("dest")]
		else:
			listOptions = arg.get("option_strings")
		options = ", ".join(listOptions)
		spaces = " " * (12 + longestSize - len(options))
		file.write("  {}{}{}\n".format(options, spaces, help))
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
	for arg in i.myAction.listArgs:
		options = ""
		help = arg.get("help") 
		if (len(arg.get("option_strings"))==0):
			listOptions = [arg.get("dest")]
		else:
			listOptions = arg.get("option_strings")
		options = "\ ".join(listOptions)
		file.write(".IP {}\n".format(options))
		file.write(help + "\n")
	file.close()

if __name__ == "__main__":
	i = importlib.import_module(sys.argv[1].partition(".")[0])
	parser = i.get_argparse()
	if (sys.argv[2].partition(".")[2] == "h"):
		write_header(parser, sys.argv[2])
	elif (sys.argv[2].partition(".")[2] == "1"):
		write_man(parser, sys.argv[2])
		