demo/makecint/exception/README

 This is the simplest example of how you can use C++ exception handling
in Cint. Please understand that Cint's exception handling support is
not mature and may cause problems. 

 To run this example, you must install Cint with G__STD_EXCEPTION flag. 
Please make sure stl/exception.dll is successfully created. (Refer to 
platform/README.txt)

OUTLINE =====================================================================

1) Files
 This directory contains

	README.txt   : This file
	eh.h         : header for exception handling class
	eh.cxx       : Example program
	setup	     : Shell script to setup and test 
	setup.bat    : WinNT batch file to setup and test , VC++
	setupbc.bat  : WinNT batch file to setup and test , C++Builder

