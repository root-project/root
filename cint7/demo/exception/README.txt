demo/exception/README

 This is the simplest example of how you can use C++ exception handling
in Cint. Please understand that Cint's exception handling support has
some limitation. Small changes from this example may cause problems.

 To run this example, you must install Cint with G__STD_EXCEPTION flag. 
Please make sure stl/exception.dll is successfully created. (Refer to 
platform/README.txt)

 If you are interested, you can look into $CINTSYSDIR/src/Api.cxx, 
G__ExceptionWrapper function for how Cint handles C++ exception.

OUTLINE =====================================================================

# Files
 This directory contains

	README.txt   : This file
	ehdemo.h     : header for exception handling class
	ehdemo.cxx   : Example program
	setup	     : Shell script to setup and test 
	setup.bat    : WinNT batch file to setup and test , Visual C++
	setupbc.bat  : WinNT batch file to setup and test , Borland C++

# Running the demo

 Run setup or setup.bat script.  In this script, demo program runs 3 times,
 compiled, partly compiled by makecint  and  interpreted.  All 3 results
 should match except for Visual C++ which has limitation.

   UNIX:
     $ sh setup

   Windows Visual C++:
     C:\> setup.bat

   Windows Borland C++:
     C:\> setupbc.bat
