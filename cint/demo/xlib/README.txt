demo/xlib/README.txt

 This directory contains xlib programming example using CINT.
Cint supports X11 library in lib/xlib directory. You need to build
include/X11/xlib.dll in order to run xlib demo program.

FILES: 
  README.txt   : This file
  test.c       : X11 example program

RUNNING THE DEMO:

  $  cint test.c

To compile the same program

  $  gcc -L/usr/X11R6/lib -lX11 test.c 
  $  a.out

 Hit ctl-C twice and type 'q' to terminate the demo.




