demo/makecint/UserMain/README 

 This is the simplest example of how you can embed Cint into your application.

OUTLINE =====================================================================

1) Files
 This directory contains following files,

	README       : This file
	UserMain.h   : Header file for user's main application
	UserMain.cxx : Source file for user's main application
	script.cxx   : Script to be interpreted by embedded Cint
	setup	     : Shell script to setup and test 
	setup.bat    : Windows batch file to setup and test , VC++
	setupbc.bat  : Windows batch file to setup and test , Borland C++

  To run this demo, simply run setup script.

       $ sh setup              UNIX

       c:\> setup.bat          Windows Visual C++

       c:\> setupbc.bat        Windows Borland C++



2) Compiling UserMain 

   UNIX
	$ makecint -mk Makefile -m -I$CINTSYSDIR -o UserMain -H UserMain.h -C++ UserMain.cxx

   Windows Visual C++
	$ makecint -mk Makefile -m -I%CINTSYSDIR% -o UserMain -H UserMain.h -C++ UserMain.cxx

   Windows Borland C++
	$ makecint -mk Makefile -m -Ic/cint -o UserMain -H UserMain.h -C++ UserMain.cxx

You need to do above only once. After this, you can create the object 
"Complex" simply by make.

   UNIX
	$ make -f Makefile

   WinNT Visual C++
        c:\> nmake /F Makefile CFG="UserMain - Win32 Release"
        c:\> move Release\UserMain.exe UserMain.exe

   WinNT Borland C++
        c:\> make.exe -f Makefile


3) Using the "Complex"
 Created object "UserMain" is a user appliation which invokes C/C++ 
interpreter. Refer to UserMain.cxx how it is implemented.

	$ UserMain
