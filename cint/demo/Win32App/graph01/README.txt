cint/demo/Win32App/graph01/README.txt


Description:
  MSVC++6.0 Win32 application. Cint script interprets Win32 API for
 graphics.


Files:

  README.txt     : This file

  graph01.dsw, graph01.dsp  : Microsoft Visual C++ 6.0 project file  
    This file (the project file) contains information at the project level and
    is used to build a single project or subproject. Other users can share the
    project (.dsp) file, but they should export the makefiles locally.

  G__clink.c     : Cint dictionary source file
  G__clink.h     : Cint dictionary header file
  Script.c       : Source file that is interpreted by Cint
  WinMain.c      : Main function for windows application
  WndProc.c      : Defines procedures of the main window
  WndProc.h      : Header file for WndProc.c
  CompiledLib.c  : Source file for precompiled library
  CompiledLib.h  : Header file for precompiled library

  Resource.rc    :
    This is a listing of all of the Microsoft Windows resources that the
    program uses.  It includes the icons, bitmaps, and cursors that are stored
    in the RES subdirectory.  This file can be directly edited in Microsoft
	Visual C++.
 App.ico         :
    This is an icon file, which is used as the application's icon (32x32).
    This icon is included by the main resource file TestApp.rc.

 small.ico
    %%This is an icon file, which contains a smaller version (16x16)
	of the application's icon. This icon is included by the main resource
	file TestApp.rc.

 StdAfx.h, StdAfx.cpp
    These files are used to build a precompiled header (PCH) file
    named TestApp.pch and a precompiled types file named StdAfx.obj.

 Resource.h
    This is the standard header file, which defines new resource IDs.
    Microsoft Visual C++ reads and updates this file.

How to build this application:

 Caution:
    Cint must be installed under c:\cint directory and this application
   must reside in c:\cint\demo\Win32app\graph01 directory.  Otherwise, you 
   may need to change project settings manually.

 1) Re-generate G__clink.c and G__clink.h
    Assuming that you have already installed Cint using VC++. 
   Run following command to create G__clink.c and G__clink.h

       c:\> cint -c-2 CompiledLib.h

 2) Build and Run application
    Double click graph01.dsw to invoke MS VC++. Build and run graph01.exe
   program.
