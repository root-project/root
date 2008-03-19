demo/makecint/DArray/README 

 This is an example of simple array class.

OUTLINE =====================================================================

1) Files
 This directory contains following files,

	README    : This file
	DArray.h  : DArray class header file
	DArray.C  : DArray class instantiation
	test.C    : Test program
	setup	  : Shell script to setup and test 
	setupdll  : Shell script to setup and test DLL
	setup.bat : WinNT batch file to setup and test , VC++
	setupdll.bat : WinNT batch file to setup and test DLL , VC++
	scsetup.bat : WinNT batch file to setup and test , SC++
	scdll.bat : WinNT batch file to setup and test DLL , SC++

 You can interpret all of those files by cint as follows, by the way, and
you will get the same result with the precompiled version.

	$ cint DArray.C test.C


2) Pre-compile Array class and encapsulate it into the cint
 You can also pre-compile DArray.h/C and encapsulate it into a customized C++
interpreter named "DArray". First, create "Makefile" using the makecint.
Following command creates "Makefile" and several cint implementation 
dependent files whose file names start with G__. 

	$ makecint -mk Makefile -o DArray -H DArray.h -C++ DArray.C

You need to do above only once. After this, you can create the object 
"DArray" by make. This example is somewhat complicated. You may see
many warnings or even errors from the compiler.

	$ make


3) Using the "DArray"
 Created object "DArray" is a C/C++ interpreter which includes Array.h/C 
as a pre-compiled class library. You can execute test.C as follows,

	$ DArray test.C

