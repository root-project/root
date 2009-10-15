demo/gl/README.txt

 This directory contains gl programming example using CINT.
Cint supports OpenGL library in lib/gl directory. You need to build
include/GL/gl.dll in order to run xlib demo program.

 Please note that functions in display1.h and display2.h must be 
precompiled in order to run this demo correctly.


FILES: 
  README.txt   : This file
  testall      : Shell script to run demo

  test0.c      : openGL example program (interpreted)
  display0.h   : Source/header for precompiled function

  test1.c      : openGL example program (interpreted)
  display1.h   : Source/header for precompiled function

  test2.c      : openGL example program (interpreted)
  display2.h   : Source/header for precompiled function


DEMO0:
  Basic demo to demonstrate CINT/OpenGL capability.  Call back functions
 are precompiled as display0.dll before actually running the demo. If you
 change display() function, you must re-compile the display0.dll.


DEMO1:
  Basic demo to demonstrate CINT/OpenGL capability.  Call back functions
 are precompiled as display1.dll before actually running the demo. If you
 change display() or key() function, you must re-compile the display1.dll.

DEMO2:
  DEMO2 has an improvement over DEMO1. Body of call back functions are
 defined in test2.c which is interpreted.  In display2.h, wrapper function
 for calling interpreted displayBody() and keyBody() is defined. With this
 technique, you can interpret everything.


RUNNING THE DEMO:

  $  sh testall

        OR

  // DEMO0
  $  makecint -mk Makefile -dl display0.dll -h display0.h
  $  make clean
  $  make
  $  cint test0.c

  // DEMO1
  $  makecint -mk Makefile -dl display1.dll -h display1.h
  $  make clean
  $  make
  $  cint test1.c

  // DEMO2
  $  makecint -mk Makefile -dl display2.dll -h display2.h
  $  make clean
  $  make
  $  cint test2.c

