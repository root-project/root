lib/posix/README.txt

ABSTRACT:

 This directory contains POSIX system call DLL build environment for UNIX 
 and Windows-NT/9x(emulation library).
 Only small subset of POSIX.1 API is supported. Please modify posix.h to 
 expand capability.

FILES:

 This directory originally contains following files. Other files will
 be created after running setup.bat script, however, these can be
 recreated if you only keep following files.

   README.txt : this file
 UNIX:
   mktypes.c  : generate $CINTSYSDIR/include/systypes.h
   setup      : setup shell script
   posix.h    : must use this header file to let cint read windows.h 
   exten.h    : Original Extension for convenience
   exten.c    : Original Extension for convenience
 WIN32
   setup.bat  : setup batch script for Visual C++
   setupsc.bat: setup batch script for Symantec C++
   winposix.h : POSIX system call emulation library for WIN32
   winposix.c : POSIX system call emulation library for WIN32

BUILD on UNIX:

 CINT must be properly installed. Move to this directory and run setup
 script.

    $ cd $CINTSYSDIR/lib/posix
    $ sh ./setup

 If everything goes fine, following file will be created.

     $CINTSYSDIR/include/posix.dll

BUILD on WIN32:

 CINT must be properly installed and include/win32api.dll must be created
 before this. include/win32api.dll can be build in lib/win32api directory.

    c:\> cd %CINTSYSDIR%\lib\posix
    c:\> setup

 If everything goes fine, following file will be created.

     $CINTSYSDIR/include/posix.dll


HOW TO USE:

 Just include <unistd.h> in your source file.


HOW TO ADD POSIX API:

 You can add desired POSIX API functions to posix.dll by modifying 
 posix.h in this directory. Run 'sh setup' script after modifying posix.h.

