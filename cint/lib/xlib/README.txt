lib/xlib/README.txt

 This directory includes X11 shared library for CINT. Run 'sh setup' to
build 'include/X11/xlib.dll'.  You may need to modify setup script for
your system, especially include and library path.

Basic X11 API can be used in an interpreted code by including <X11/Xlib.h>. 
Example program is in demo/xlib directory.

Files: 
       README.txt    : This file
       setup	     : xlib.dll compile script
       TOP.h	     : Header file for making xlib.dll
       XLIB.h	     : Header file for making xlib.dll
       x11const.h    : Header file for making xlib.dll
       x11mfunc.h    : Header file for making xlib.dll

2001/Sep/29
 gcc-3.00 is supported from cint5.15.14, however, gl.dll can not be compiled.
The author is working to solve this problem.
