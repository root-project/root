lib/socket/README.txt

ABSTRACT:

 This directory contains TCP/IP library for cint.

FILES:

 This directory originally contains following files. Other files will
 be created after running setup.bat script, however, these can be
 recreated if you only keep following files.

   README.txt : this file
   setup      : setup shell script
   setup.bat  : setup batch script for Visual C++
   cintsock.h : Cint socket library header
   cintsock.c : Cint socket library dummy source

BUILD:

 CINT must be properly installed. Move to this directory and run setup
 script. If you use WIN32, include/win32api.dll must be created before this. 

    $ cd $CINTSYSDIR/cint/lib/socket
    $ sh ./setup

        OR

    c:\> cd %CINTSYSDIR%\cint\lib\socket
    c:\> setup.bat

 $CINTSYSDIR/include/socket.dll is the final product.


HOW TO USE:

  Include either of <winsock.h> or <socket.h>
