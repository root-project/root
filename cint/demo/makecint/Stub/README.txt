demo/makecint/Stub/README

 This is the simplest example of how you can call interpreted function from
compiled code. Makecint enables you to use interpreted non-static global
function to be used in compiled code as if it were a compiled function.

OUTLINE =====================================================================

1) Files
 This directory contains

	README    : This file
	Src.h     : Compiled function header
	Src.C     : Compiled function body
	Stub.h    : Stub function header
	Stub.C    : Stub function body
	setup	  : Shell script to setup and test 
	setupdll  : Shell script to setup and test DLL
	setup.bat : WinNT batch file to setup and test , VC++
	setupdll.bat : WinNT batch file to setup and test , VC++
	scsetup.bat : WinNT batch file to setup and test , SC++
	scdll.bat : WinNT batch file to setup and test , SC++

 You can interpret all of those files by cint as follows, by the way, and
you will get the same result with the precompiled version.
	
	$ cint Src.C Stub.C

2) Precompile Src.C and register Stub function.
 You can compile Src.C and interpret Stub.C. Src.C uses function in Stub.C,
so, you need to register functions in Stub.C as Stub function. This can be
done by -i++ option (or -i option for C)

	$ makecint -mk Makefile -o Stub -H Src.h -i++ Stub.h -C++ Src.C

You need to do above only once. After this, you can create the object 
"Complex" simply by make.

	$ make

3) Using the "Stub" object
 Created object "Stub" is a C/C++ interpreter which includes Src.h/C
as a pre-compiled library with Stub.h registered as Stub. Interpreted 
function in Stub.C can be called from Src.C. 

	$ Stub Stub.C

I recommend to use -s(step into) option to watch what is going on.

	$ Stub -s Stub.C


4) Making DLL(Dynamic Link Library)
 If your operating system support dynamic linking feature and you have 
installed cint with DLL capability, you can precompile Complex.C as a DLL.
(In $CINTSYSDIR/MAKEINFO file, DLLPOST, CCDLLOPT, LDDLLOPT and LDOPT must 
be set appropriately. Refer to $CINTSYSDIR/platform/README file for detail 
of installation.)

	$ makecint -mk Makefile -dl Stub.dl -H Src.h -i++ Stub.h -C++ Src.C
	$ make
	$ cint Stub.dl Stub.C
