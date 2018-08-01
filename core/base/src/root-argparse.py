import argparse
import sys

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
	parser.add_argument('-l', help='Do not show splash screen')
	parser.add_argument('-config', help='print ./configure options')
	parser.add_argument('-memstat', help='run with memory usage monitoring')
	parser.add_argument('-h','-?', '--help', help='Show summary of options')
	parser.add_argument('--notebook', help='Execute ROOT notebook')
	parser.add_argument('--web', help='Display graphics in a web browser')
	parser.add_argument('dir', help='if dir is a valid directory cd to it before executing')
	return parser
