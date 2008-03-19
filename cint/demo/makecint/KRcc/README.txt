demo/makecint/KRcc/README 

 This example directory is setup for those who only have K&R C compiler.
Please look carefully into the Complex.h. There is a trick that you need to
play to make it work under K&R environment. 

OUTLINE =====================================================================

1) Files
 This directory contails following files,

	README    : This file
	Complex.h : Complex number class definition
	Complex.c : Complex number class implementation
	stub.h    : Stub function header file
	stub.c    : Stub function source file
	test.c    : Test program
	setup	  : Shell script to setup and test 
	setupdll  : Shell script to setup and test DLL
	setup.bat : WinNT batch file to setup and test , VC++
	setupdll.bat : WinNT batch file to setup and test , VC++
	scsetup.bat : WinNT batch file to setup and test , SC++
	scdll.bat : WinNT batch file to setup and test , SC++

 You can interpret all of those files by cint as follows, by the way, and
you will get the same result with the precompiled version.

	$ cint Complex.c stub.c test.c


2) Pre-compile Complex.h, Complex.c and encapsulate it into the cint
 You can also pre-compile Complex.h/c and encapsulate it into a customized C++
interpreter named "Complex". First, create "Makefile" using the makecint.
Following command creates "Makefile" and several cint implementation 
dependent files whose file names start with G__. 

	$ makecint -mk Makefile -o Complex -h Complex.h -C Complex.c -i stub.h

You need to do above only once. After this, you can create the object 
"Complex" simply by make.

	$ make


3) Using the "Complex"
 Created object "Complex" is a C/C++ interpreter which includes Complex.h/c 
as a pre-compiled class library. You can execute test.C as follows,

	$ Complex stub.c test.c



4) Making DLL(Dynamic Link Library) 
 If your operating system support dynamic linking feature and you have 
installed cint with DLL capability, you can precompile Complex.C as a DLL.
(In $CINTSYSDIR/MAKEINFO file, DLLPOST, CCDLLOPT, LDDLLOPT and LDOPT must 
be set appropriately. Refer to $CINTSYSDIR/platform/README file for detail 
of installation.)

	$ makecint -mk Makefile -dl Complex.dl -h Complex.h -C Complex.c -i stub.h
	$ make

 You can link Complex.dl to cint at run time.

	$ cint Complex.dl stub.c test.c
