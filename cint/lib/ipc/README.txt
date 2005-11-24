lib/ipc/README.txt

ABSTRACT:
 This directory contains IPC DLL build environment for UNIX. This library
 includes Shared-Memory, Message and Semaphore API.

FILES:

 This directory originally contains following files. Other files will
 be created after running setup.bat script, however, these can be
 recreated if you only keep following files.

   README.txt : this file
 UNIX:
   README.txt : this file
   setup      : setup shell script
   ipcif.h    : must use this header file to let cint read sys/ipc.h

BUILD on UNIX:

 CINT must be properly installed. Move to this directory and run setup
 script.

    $ cd $CINTSYSDIR/lib/ipc
    $ sh ./setup

 If everything goes fine, following file will be created.

     $CINTSYSDIR/include/sys/ipc.dll


HOW TO USE:

 Include either <sys/ipc.h> , <sys/sem.h> , <sys/shm.h> , <sys/msg.h> 
 in your source file.

