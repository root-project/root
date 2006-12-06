demo/makecint/Array/README 

CAUTION: EXAMPLE IN THIS DIRECTORY MAY NOT WORK ANY MORE. THIS WAS MADE FOR
         AN OLD C++ 3.0. TEMPLATE SPECIFICATION HAS BEEN CHNAGED SINCE THEN
         AND NO LONGER WORK IN CONTENPOLARY C++ COMPILER.

This is a bit advanced example using template. Some C++ compilers may not
accept this example due to compiler limitation. Please check if your C++
compiler handles template in very detail. 

OUTLINE =====================================================================

1) Files
 This directory contains following files,

	README    : This file
	Array.h : Array class template header file
	Array.C : Array class template instantiation
	Fundament.h : misc header
	Fundament.C : misc source
	test.C    : Test program
	setup	  : Shell script to setup and test , UNIX
        setupdll  : Shell script to setup and test as DLL , UNIX
	setup.bat : WinNT batch file to setup and test , VC++
        setupdll.bat : WinNT batch file to setup and test , VC++
	scsetup.bat : WinNT batch file to setup and test , SC++
        scdll.bat : WinNT batch file to setup and test , SC++

This example also uses following files,

	../Complex/Complex.h : Complex number class header
	../Complex/Complex.C : Complex number class source

 You can interpret all of those files by cint as follows, by the way, and
you will get the same result with the precompiled version.

	$ cint -I../Complex ../Complex/Complex.C Fundament.C Array.C test.C


2) Pre-compile Array template class and encapsulate it into the cint
 You can also pre-compile Array.h/C and encapsulate it into a customized C++
interpreter named "Array". First, create "Makefile" using the makecint.
Following command creates "Makefile" and several cint implementation 
dependent files whose file names start with G__. 

	$ makecint -mk Makefile -o Array -I../Complex -H Fundament.h  \
              Array.h -C++ Fundament.C Array.C ../Complex/Complex.C

 In case of WinNT, you must copy and rename source files as follows.

          Array.C               ->   Array.cxx
          Fundament.C           ->   Fundament.cxx
          ..\Complex\Complex.C  ->   Complex.cxx

You need to do above only once. After this, you can create the object 
"Array" by make. This example is somewhat complicated. You may see
many warnings or even errors from the compiler.

   UNIX
	$ make -f Makefile

   WinNT Symantec C++
	c:\> smake -f Makefile

   WinNT Visual C++
	c:\> nmake /F Makefile CFG="Array - Win32 Release"
	c:\> move Release\Complex.exe Complex.exe


3) Using the "Array"
 Created object "Array" is a C/C++ interpreter which includes Array.h/C 
as a pre-compiled class library. You can execute test.C as follows,

	$ Array test.C

