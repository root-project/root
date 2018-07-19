import argparse
import sys

class myAction(argparse.Action):
	listArgs = []
	def __init__(self, *args, **kwargs):
		super(myAction, self).__init__(*args, **kwargs)
		myAction.listArgs.append(kwargs)

def get_argparse():
	parser = argparse.ArgumentParser(add_help=False, prog='root',
	description = """ROOTs Object-Oriented Technologies.
	
	root is an interactive interpreter of C++ code. It uses the ROOT  framework.  For  more information on ROOT, please refer to
	
	An extensive Users Guide is available from that site (see below).
	""")
	parser.add_argument('-b', help='Run in batch mode without graphics', action=myAction)
	parser.add_argument('-x', help='Exit on exceptions', action=myAction)
	parser.add_argument('-e', help='Execute the command passed between single quotes', action=myAction)
	parser.add_argument('-n', help='Do not execute logon and logoff macros as specified in .rootrc', action=myAction)
	parser.add_argument('-t', help='Enable thread-safety and implicit multi-threading (IMT)', action=myAction)
	parser.add_argument('-q', help='Exit after processing command line macro files', action=myAction)
	parser.add_argument('-l', help='Do not show splash screen', action=myAction)
	parser.add_argument('-config', help='print ./configure options')
	parser.add_argument('-memstat', help='run with memory usage monitoring', action=myAction)
	parser.add_argument('-h','-?', '--help', help='Show summary of options', action=myAction)
	parser.add_argument('--notebook', help='Execute ROOT notebook', action=myAction)
	parser.add_argument('--web', help='Display graphics in a web browser', action=myAction)
	parser.add_argument('dir', help='if dir is a valid directory cd to it before executing', action=myAction)
	return parser
