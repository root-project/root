lib/longlong/README.txt

ABSTRACT:

 This directory contains 'long long' DLL build environment for cint.

FILES:

 This directory originally contains following files. Other files will
 be created after running setup.bat script, however, these can be
 recreated if you only keep following files.

   README.txt : this file
   setup      : setup shell script
   setup.bat  : setup batch script
   longlong.h : must use this header file to let cint read longlong definition

BUILD:

 CINT must be properly installed. Move to this directory and run setup
 script.

    $ cd $CINTSYSDIR/lib/longlong
    $ sh ./setup

       OR

    c:\> cd %CINTSYSDIR%\lib\longlong
    c:\> setup.bat

 $CINTSYSDIR/include/long.dll is the final product.


HOW TO USE:

 After building $CINTSYSDIR/include/long.dll , you can use 'long long' in
 your program. long.dll will be autumatically loaded.


CAUTION:

 Current version is only partly tested only on Linux and Win32.
