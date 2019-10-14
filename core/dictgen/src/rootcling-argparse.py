import argparse
import sys

EPILOG = """
The options -p, -c, -l, -cint and -gccxml are deprecated and currently ignored.



IMPORTANT:
1) LinkDef.h must be the last argument on the rootcling command line.
2) Note that the LinkDef file name must contain the string:
   LinkDef.h, Linkdef.h or linkdef.h, i.e. NA49_LinkDef.h.

Before specifying the first header file one can also add include
file directories to be searched and preprocessor defines, like:
  -I$MYPROJECT/include -DDebug=1

NOTA BENE: the dictionaries that will be used within the same project must
have unique names



The (optional) file LinkDef.h looks like:

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class TAxis;
#pragma link C++ class TAttAxis-;
#pragma link C++ class TArrayC-!;
#pragma link C++ class AliEvent+;

#pragma link C++ function StrDup;
#pragma link C++ function operator+(const TString&,const TString&);

#pragma link C++ global gROOT;
#pragma link C++ global gEnv;

#pragma link C++ enum EMessageTypes;

#endif

This file tells rootcling which classes will be persisted on disk and what
entities will trigger automatic load of the shared library which contains
it. A trailing - in the class name tells rootcling to not generate the
Streamer() method. This is necessary for those classes that need a
customized Streamer() method. A trailing ! in the class name tells rootcling
to not generate the operator>>(TBuffer &b, MyClass *&obj) function. This is
necessary to be able to write pointers to objects of classes not inheriting
from TObject. See for an example the source of the TArrayF class.
If the class contains a ClassDef macro, a trailing + in the class
name tells rootcling to generate an automatic Streamer(), i.e. a
streamer that let ROOT do automatic schema evolution. Otherwise, a
trailing + in the class name tells rootcling to generate a ShowMember
function and a Shadow Class. The + option is mutually exclusive with
the - option. For legacy reasons it is not yet the default.
When the linkdef file is not specified a default version exporting
the classes with the names equal to the include files minus the .h
is generated.

The default constructor used by the ROOT I/O can be customized by
using the rootcling pragma:
   #pragma link C++ ioctortype UserClass;
For example, with this pragma and a class named MyClass,
this method will called the first of the following 3
constructors which exists and is public:
   MyClass(UserClass*);
   MyClass(TRootIOCtor*);
   MyClass(); // Or a constructor with all its arguments defaulted.

When more than one pragma ioctortype is used, the first seen has
priority.  For example with:
   #pragma link C++ ioctortype UserClass1;
   #pragma link C++ ioctortype UserClass2;

ROOT considers the constructors in this order:
   MyClass(UserClass1*);
   MyClass(UserClass2*);
   MyClass(TRootIOCtor*);
   MyClass(); // Or a constructor with all its arguments defaulted.
"""
def get_argparse():
	parser = argparse.ArgumentParser(add_help=False, prog='rootcling',
	description = 'This program generates the dictionaries needed for performing I/O of classes.',
	epilog = EPILOG
)
	parser.add_argument('-f', help='Overwrite an existing output file\nThe output file must have the .cxx, .C, .cpp, .cc or .cp extension.\n')
	parser.add_argument('-v', help='Display all messages')
	parser.add_argument('-v0', help='Display no messages at all')
	parser.add_argument('-v1', help='Display only error messages')
	parser.add_argument('-v2', help='Display error and warning messages (default).')
	parser.add_argument('-v3', help='Display error, warning and note messages')
	parser.add_argument('-v4', help='Display all messages\n')
	parser.add_argument('-m', help="""Specify absolute or relative path Clang pcm file to be loaded
The pcm file (module) produced by this invocation of rootcling
will not include any of the declarations already included in the
pcm files loaded via -m.  There can be more than one -m
""")
	parser.add_argument('-rmf', help="""Rootmap file name
Name of the rootmap file. In order to be picked up by ROOT it must
have .rootmap extension
""")
	parser.add_argument('-rml', help="""Rootmap library name
Specify the name of the library which contains the autoload keys. This
switch can be specified multiple times to autoload several libraries in
presence of a particular key
""")
	parser.add_argument('-split', help="""Split the dictionary
Split the dictionary in two, putting the ClassDef functions in a separate
file
""")
	parser.add_argument('-s', help="""Target library name
The flag -s must be followed by the name of the library that will
contain the object file corresponding to the dictionary produced by
this invocation of rootcling.
The name takes priority over the one specified for the rootmapfile.
The name influences the name of the created pcm:
   1) If it is not specified, the pcm is called libINPUTHEADER_rdict.pcm
   2) If it is specified, the pcm is called libTARGETLIBRARY_rdict.pcm
      Any "liblib" occurence is transformed in the expected "lib"
   3) If this is specified in conjunction with --multiDict, the output is
      libTARGETLIBRARY_DICTIONARY_rdict.pcm
""")
	parser.add_argument('-multiDict', help="""Enable support for multiple pcms in one library
Needs the -s flag. See its documentation.
""")
	parser.add_argument('-inlineInputHeader', help="""Add the argument header to the code of the dictionary
This allows the header to be inlined within the dictionary
""")
	parser.add_argument('-interpreteronly', help='No IO information in the dictionary\n')
	parser.add_argument('-noIncludePaths', help="""Do not store the headers' directories in the dictionary
Instead, rely on the environment variable $ROOT_INCLUDE_PATH at runtime
""")
	parser.add_argument('-excludePath', help="""Specify a path to be excluded from the include paths
specified for building this dictionary
""")
	parser.add_argument('--lib-list-prefix', help="""Specify libraries needed by the header files parsed
This feature is used by ACliC (the automatic library generator).
Rootcling will read the content of xxx.in for a list of rootmap files (see
rlibmap). Rootcling will read these files and use them to deduce a list of
libraries that are needed to properly link and load this dictionary. This
list of libraries is saved in the first line of the file xxx.out; the
remaining lines contains the list of classes for which this run of
rootcling produced a dictionary
""")
	return parser

