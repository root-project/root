lib/win32api/readme.txt

ABSTRACT:

 This directory contains WIN32API DLL build environment using Visual C++ 4.0
 for CINT C++ interpreter.  Only part of Win32 API is embedded, so far.
 Please extend winfunc.h to extend capability.

FILES:

 This directory originally contains following files. Other files will
 be created after running setup.bat script, however, these can be
 recreated if you only keep following files.

   readme.txt : this file
   setup.bat  : setup batch script for Visual C++
   setupsc.bat: setup batch script for Symantec C++
   cintwin.h  : must use this header file to let cint read windows.h 
   winfunc.h  : windows.h dummy declaration for symbol registeration

BUILD:

 CINT must be properly installed using Visual C++ 4.0 or later version.
 CINT and VC++ related environment variables must be properly set.
 Open MS-DOS/cygwin prompt window, move to this directory and run setup.bat.

    cd %cintsysdir%\cint\lib\win32api
    sh setup

 If everything goes fine, following files will be created.

     %cintsysdir%\cint\include\win32api.dll
     %cintsysdir%\cint\lib\win32api\win32api.lib


HOW TO USE:

 Just include <windows.h> in your source file.


HOW TO ADD WIN32 API:

 You can add desired Win32 API functions to win32api.dll by modifying 
 winfunc.h in this directory. When you add an API, you need to add
 a pair of description as follows at least.  The first line defines
 Win32 API function prototype. Makecint (cint -c-2) reads this line
 to add symbol information. Second line turns on DLL linkage of the
 specified function.

    BOOL CopyFile(LPCTSTR lpExistingFileName,LPCTSTR lpNewFileName
	          ,BOOL bFailIfExists);
    #pragma link C func CopyFile;

 You may also need to add typedef statement if added API uses types
 that is not declared within makecint's scope.

 Run setup.bat script after modifying winfunc.h.

