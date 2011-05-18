demo/makecint/Complex/README 

 This is the simplest example of how you can use "makecint". It is recommended
to take time studying this example to understand how "makecint" works before 
you try encapsulating more complicated application.

OUTLINE =====================================================================

1) Files
 This directory contains following files,

	README    : This file
	Complex.h : Complex number class definition
	Complex.C : Complex number class implementation
	test.C    : Test program
	setup	  : Shell script to setup and test 
	setupdll  : Shell script to setup and test DLL
	setup.bat : WinNT batch file to setup and test , VC++
	setupdll.bat : WinNT batch file to setup and test , VC++
	scsetup.bat : WinNT batch file to setup and test , SC++
	scdll.bat : WinNT batch file to setup and test , SC++

 You can interpret all of those files by cint as follows, by the way, and
you will get the same result with the precompiled version.

	$ cint Complex.C test.C


2) Pre-compile Complex.h, Complex.C and encapsulate it into the cint
 You can also pre-compile Complex.h/C and encapsulate it into a customized C++
interpreter named "Complex". First, create "Makefile" using the makecint.
Following command creates "Makefile" and several cint implementation 
dependent files whose file names start with G__.  

	$ makecint -mk Makefile -o Complex -H Complex.h -C++ Complex.C

You need to do above only once. After this, you can create the object 
"Complex" simply by make.

   UNIX (and Windows using cygwin)
	$ make -f Makefile

   WinNT without cygwing is currently not supported.


3) Using the "Complex"
 Created object "Complex" is a C/C++ interpreter which includes Complex.h/C 
as a pre-compiled class library. You can execute test.C as follows,

	$ Complex test.C


4) Making DLL(Dynamic Link Library)
 If your operating system support dynamic linking feature and you have 
installed cint with DLL capability, you can precompile Complex.C as a DLL.
(In $CINTSYSDIR/MAKEINFO file, DLLPOST, CCDLLOPT, LDDLLOPT and LDOPT must 
be set appropriately. Refer to $CINTSYSDIR/platform/README file for detail 
of installation.)

   UNIX/Cygwin
	$ makecint -mk Makefile -dl Complex.dll -H Complex.h -C++ Complex.C
	$ make

 You can link Complex.dl to cint at run time.

	$ cint Complex.dll test.C
