lib/gl/README.txt

 This directory includes opengl shared library for CINT. Run 'sh setup' to
build 'include/GL/gl.dll'.  You may need to modify setup script for
your system, especially include and library path.

Basic opengl API can be used in an interpreted code by including <GL/gl.h>,
<GL/glu.h>, <GL/glut.h> or <GL/xmesa.h>.

Files: 
       README.txt    : This file
       setup	     : gl.dll compile script
       TOP.h	     : Header file for making gl.dll
       GL.h	     : Header file for making gl.dll

2001/Sep/29
 gcc-3.00 is supported from cint5.15.14, however, gl.dll can not be compiled.
The author is working to solve this problem.
