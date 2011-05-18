lib/README.txt

 This directory contains several interesting precompiled library build
environments. 

lib/posix
   lib/posix directory contains build environment for include/posix.dl. 
  include/posix.dl contains subset of POSIX.1 system calls. Goto lib/posix 
  directory and do 'sh setup' if you use UNIX.
   This directory also contains POSIX system call emulation library for WIN32.
  Do 'setup.bat' if you use WIN32.

lib/win32api
   lib/win32api directory contains build environment for include/win32api.dll.
  include/win32api.dll contains subset of Win32 API functions. Goto 
  lib/win32api directory and do 'setup.bat' if you use WinNT/9x.

lib/socket
   lib/socket directory contains build environment for include/cintsock.dll.
  include/cintsock.dll contains small subset of TCP/IP library. Goto
  lib/socket directory and do 'sh setup' or 'setup.bat'.

lib/dll_stl
   lib/dll_stl directory contains build environment for precompiled STL
  containers. string, vector and map are currently supported. Goto lib/stl
  directory and do 'setup.bat'. 

lib/prec_stl
   lib/perc_stl directory contains dummy header files for precompiling STL
  containers.

lib/WildCard
lib/wintcldl
   lib/WildCard and lib/wintcldl directories contains build environment for
  WildC++ , a combination of CINT and Tcl/Tk. lib/WildCard is for UNIX and
  lib/wintcldl is for WinNT/9x. Goto one of those directory and read README
  file there.

lib/cintocx
   lib/cintocx directory contains build environment for cintocx.ocx. 
  Cint works with VisualBasic. This implementation is premature and not.
  recommended for use.

lib/xlib
   lib/xlib directory contains build environment for include/X11/xlib.dll.
  This DLL includes most of the basic X11R6 API.

lib/gl
   lib/gl directory contains build environment for include/gl.dll.
  This DLL includes basic openGL API.

lib/qt
   lib/qt directory contains build environment for include/qtcint.dll.
  This DLL includes basic Qt library API.


lib/stream
lib/cbstream
lib/bcstream
lib/vcstream
lib/stdstrct
   These directories are for cint built-in symbol table generation. Do not
  touch these directories unless you know very well about the details.



