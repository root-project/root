File: $CINTSYSDIR/include/README

About this file:

 	This file contains information about $CINTSYSDIR/include directory 
	and files under the directory.


$CINTSYSDIR environment variable:

	Environment variable CINTSYSDIR must point to the directory where 
	you have installed cint. For example, if you installed cint into
	/usr/local/cint, 

	    ksh,bsh
		$ CINTSYSDIR=/usr/local/cint
		$ export CINTSYSDIR
	    csh
		% setenv CINTSYSDIR /usr/local/cint

	If you don't do this, you can not utilize files and class libraries
	uner $CINTSYSDIR/include.


$CINTSYSDIR/include directory:

	Cint searches for $CINTSYSDIR/include directory for standard include
	files which is specified as '#include <xxxx>'.  
	Files under $CINTSYSDIR/include are specially made for cint using
	original language extention. Do not try to include these files for
	compiler based C or C++ source code.

	There are 4 kinds of files under this directory.

	xxx.h   : Header file to be included as '#include <xxx.h>'
	xxx.sl  : Shared library binary to be linked as '#include <xxx.sl>'
	xxx.c   : Shared library source file
	xxx.o   : Shared library relocatably compiled object file

	You need to run HP-UX8.0 or later version to utilize these files.
	HP-UX7.0 and SunOS are not supported.

makefile:

	If you modify one of the xxx.c files, do 'make' in $CINTSYSDIR/include
	directory. It will update xxx.sl objects.
