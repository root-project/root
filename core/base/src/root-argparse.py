import argparse

def get_argparse():
	parser = argparse.ArgumentParser(add_help=False, prog='root',
	description = """ROOTs Object-Oriented Technologies.\n
root is an interactive interpreter of C++ code. It uses the ROOT  framework.  For  more information on ROOT, please refer to\n
An extensive Users Guide is available from that site (see below).
""")
	parser.add_argument('-b', help='Run in batch mode without graphics')
	parser.add_argument('-x', help='Exit on exceptions')
	parser.add_argument('-e', help='Execute the command passed between single quotes')
	parser.add_argument('-n', help='Do not execute logon and logoff macros as specified in .rootrc')
	parser.add_argument('-t', help='Enable thread-safety and implicit multi-threading (IMT)')
	parser.add_argument('-q', help='Exit after processing command line macro files')
	parser.add_argument('-l', help='Do not show the ROOT banner')
	parser.add_argument('-a', help='Show the ROOT splash screen (Windows only)')
	parser.add_argument('-config', help='print ./configure options')
	parser.add_argument('-h','-?', '--help', help='Show summary of options')
	parser.add_argument('--version', help='Show the ROOT version')
	parser.add_argument('--notebook', help='Execute ROOT notebook')
	parser.add_argument('--web', help='Use web-based display for graphics, browser, geometry')
	parser.add_argument('--web=<type>', help='Use the specified web-based display such as chrome, firefox, qt6\nFor more options see the documentation of TROOT::SetWebDisplay()')
	parser.add_argument('--web=off', help='Disable any kind of web-based display')
	parser.add_argument('[dir]', help='if dir is a valid directory cd to it before executing')
	parser.add_argument('[data1.root...dataN.root]', help='Open the given ROOT files; remote protocols (such as http://) are supported')
	parser.add_argument('[file1.C...fileN.C]', help='Execute the ROOT macro file1.C ... fileN.C\nCompilation flags as well as macro arguments can be passed, see format in https://root.cern/manual/root_macros_and_shared_libraries/')
	parser.add_argument('[file1_C.so...fileN_C.so]', help='Load and execute file1_C.so ... fileN_C.so (or .dll if on Windows)\nThey should be already-compiled ROOT macros (shared libraries) or:\nregular user shared libraries e.g. userlib.so with a function userlib(args)')
	parser.add_argument('[anyfile1..anyfileN]', help='All other arguments pointing to existing files will be checked to see if they are ROOT Files (checking the MIME type inside the file) and if they are not they will be handled as a  ROOT macro file')
	return parser
