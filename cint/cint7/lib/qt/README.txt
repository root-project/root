lib/qt/README.txt

 This directory includes build environment for Qt library access DLL. Run 
'sh setup' to build 'include/qtcint.dll'.  You need to modify setup.bat
script for setting Qt directory in your system.  By default, c:\qt is set.

Basic Qt library can be used in an interpreted code by including <qtcint.dll>

Platform support:
	 Windows	Visual C++ 6.0
    qtcint.dll hasn't been port to Linux/UNIX platforms. Your 
    contribution will be highly appreciated.

Files: 
       README.txt       : This file
       setup	        : qtcint.dll compile script
       qtcint.h	        : Header file for making qtcint.dll
       qtclasses.h	: Linkdef information for Qt classes
       qtfunctions.h	: Linkdef information for Qt functions
       qtglobals.h	: Linkdef information for Qt global variables
       qtdummy.h	: Qt macro emulation functions
       qtstatic.cxx	: Static object declaration to satisfy linker
       qconfig.h        : dummy file
       qmodules.h       : dummy file
       qplatformdefs.h  : dummy file

CAUTION:
  Qt-Cint is experimental. 
