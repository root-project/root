demo/mthread/README.txt

 This directory contains simple Multi-Thread programming example using CINT.

# General information
 Cint is not multithread safe. CINT must run in one thread. As long as you 
have only one CINT thread in a process, you can run other precompiled jobs 
in different threads. 


# Windows-NT/9x Visual C++ : Using Win32 API

  In Windows-NT/9x, cint supports CreateThread() Win32 API in lib/win32api 
 library. You can start threads with a precompiled function from the 
 interpreter. Please be careful about using CreateThread, a small mistake
 can crash MS-Windows.

 FILES: 
   README.txt   : This file
   mtlib.h      : Precompiled library source for multithread demo
   mtmain.cxx   : Interpreted source for multithread demo

 RUNNING DEMO:
   c:\cint\demo\mthread>  makecint -mk makemtlib -dl mtlib.dll -H mtlib.h
   c:\cint\demo\mthread>  make -f makemtlib
   c:\cint\demo\mthread>  cint mtmain.cxx


# UNIX : Using pthread

  In UNIX, multi thread program can be written using pthread library.
 ptlib.h and ptmain.cxx are pthread examples. Following example is only
 verified in Linux2.0.

 FILES: 
   README.txt   : This file
   ptlib.h      : Precompiled library source for multithread demo
   ptmain.cxx   : Interpreted source for multithread demo
   testall      : shell script to run pthread demo

 RUNNING DEMO:
   $ sh testall

         OR

   $ makecint -mk makeptlib -dl thread.dll -H ptlib.h
   $ make -f makeptlib
   $ cint ptmain.cxx


# UNIX : Using fork system calls

  In UNIX, child process can be easily created by fork system call.
 This is not a multi thread programming. However, I want to demonstrate 
 cint capability with fork

 FILES: 
   README.txt   : This file
   fork.cxx     : This is not a multi-thread program. Running background job
                  using fork system call on UNIX.

 RUNNING DEMO:
   $ cint fork.cxx

