demo/multilibcint/README.txt

 THIS EXAMPLE IS EXPERIMENTAL. NOT RECOMMENDED FOR ORDINARY USERS.

 This directory contains simple Multi-Thread programming example using CINT.
This solution is proposed by Christoph Bugel<chris@uxsrvc.tti-telecom.com>
and his collegue in TTI-telecom. 


# General information
 Cint is not multithread safe.  In this directory, we demonstrate a technique
to run multiple Cint cores in a process by copying libcint.so to multiple
libcintX.so.  Each shared library keeps static variables private. This 
emulates thread-safe operation.

# Caution:
 There is another multi-threading example in demo/mthread. Technique used 
here is different from that of demo/mthread.  Here, we copy multiple 
libcintX.so shared library and load them indivisually. Because each shared
library object keeps static variables private, we can run multiple cint 
cores in a process.
 
 If you use Linux, you MUST install Cint with platform/linux_RH6.2_mtso.  
Otherwise, this example will not work.
 If you use Windows VC++6.0, add G__MULTITHREADLIBCINT macro at the beginning
of G__ci.h and install Cint. 


 FILES: 
   README.txt : This file
   setup.bat  : Script to run the demo for Windows
   setup      : Script to run the demo for Linux
   mt.h       : header file
   mt.c       : multi-thread libcint library, source file
   main.cxx   : main program. On Windows, Cint can interpret this.
   test1.cxx  : test program to be interpreted by multi-thread libcint
   test2.cxx  : test program to be interpreted by multi-thread libcint


# Windows VC++6.0
 RUNNING DEMO:
   c:\> setup.bat

      OR

   $ makecint -mk Makefile -dl mt.dll -h mt.h -C mt.c
   $ make -f Makefile
   $ cint main.cxx


# Linux RedHat6.2
 RUNNING DEMO:
   $ sh setup

      OR

   $ g++ main.cxx mt.c -lpthread -ldl
   $ a.out


# Other platforms
  This example is not supported for other platforms at this moment.


# Issue
 There are problems associated with this technique. The problem occurs
when unloading or reloading shared library in Linux operating system.
This seems to work as long as libcintX.so's are not unloaded until the
very end of main process. The example program shows a way to workaround 
this problem. 

