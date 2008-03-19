demo/makecint/ReadFile/README 

This example tries to encapsulate a simple file parsing class.

OUTLINE =====================================================================

1) Files
 This directory contains following files,

	README    : This file
	ReadFile.h: ReadFile class header
	ReadFile.C: ReadFile class implementation
	Common.h  : misc header
	test.C    : Test program
	setup	  : Shell script to setup and test 
	setupdll  : Shell script to setup and test DLL
	setup.bat : WinNT batch file to setup and test , VC++
	setupdll.bat : WinNT batch file to setup and test , VC++
	scsetup.bat : WinNT batch file to setup and test , SC++
	scdll.bat : WinNT batch file to setup and test , SC++

 You can interpret all of those files by cint as follows, by the way, and
you will get the same result with the precompiled version.

	$ cint ReadFile.C test.C


2) Pre-compile ReadFile class
 You can also pre-compile ReadFile.h/C and encapsulate it into a customized C++
interpreter named "ReadFile". First, create "Makefile" using the makecint.
Following command creates "Makefile" and several cint implementation 
dependent files whose file names start with G__. 

	$ makecint -mk Makefile -o ReadFile -H ReadFile.h -C++ ReadFile.C

You need to do above only once. After this, you can create the object 
"ReadFile" by make. This example is somewhat complicated. You may see
many warnings or even errors from the compiler.

	$ make


3) Using the "ReadFile"
 Created object "ReadFile" is a C/C++ interpreter which includes ReadFile.h/C 
as a pre-compiled class library. You can execute test.C as follows,

	$ ReadFile test.C

