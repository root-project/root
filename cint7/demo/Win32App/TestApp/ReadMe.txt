cint/demo/Win32App/TestApp/ReadMe.txt


Description:
  MSVC++6.0 Win32 application which invokes cint from pull-down
 menu. 


Why this demo application is added:
  Cint can be embedded to any kind of C/C++ application. Doing so
 with a console project is supported by 'makecint -m'. However, 
 a way to do so with Win32 application (Window based appication)
 has not been clearly demonstrated.  This is a simple example that
 invokes cint as a script interpreter from pull-down menu. 


Preparation:
  1) Install cint in c:\cint using MSVC++6.0. Refer to cint 
     README.txt for details.
  2) Set following environment variables
     set CINTSYSDIR=c:\cint 
     set PATH=%PATH%;%CINTSYSDIR%


How to compile and run this application:
  1) Double click TestApp.dsw to start MSVC++ development studio.
  2) Build TestApp.exe by 'Build->Build all'
  3) Run TestApp.exe by 'Build->Execute TestApp.exe'
  4) In TestApp.exe, select pull-down menu 'File->cint' , 
     then a message box appears. This message box is invoked from
     an interpreted code in this directory 'script.cxx'. You can 
     change behavior of 'File->cint' by modifying script.cxx.


How this application was created:
 Here, I will explain how this application was created. This will give you
 a hint for adding Cint in your own project.

  1) Start MSVC++ development studio
  2) Create an Win32 application with 'File->New->Project->Win32 application'
     Select HelloWorld application.
  3) Modify TestApp.cxx 
       - Add '#include "G__ci.h"
       - Add a case in switch(wmid)
     Everything is marked as "CINT" in TestApp.cxx
  4) Add include path c:\cint. "Project->Setting" , C/C++ -> Preprocessor
     -> include path directory
  5) Add c:\cint\libcint.lib in the project by 
     "Project->Add to project->File". 
  6) Build TestApp.exe by "Build->Build all"

 Please refer also to following documentations.

   c:\cint\doc\makecint.txt
               ref.txt      (G__init_cint, G__calc, G__scratch_all API)



------ below, created by MSVC++ development studio ---------

========================================================================
       WIN32 APPLICATION : TestApp
========================================================================


AppWizard has created this TestApp application for you.  

This file contains a summary of what you will find in each of the files that
make up your TestApp application.

TestApp.cpp
    This is the main application source file.

TestApp.dsp
    This file (the project file) contains information at the project level and
    is used to build a single project or subproject. Other users can share the
    project (.dsp) file, but they should export the makefiles locally.
	

/////////////////////////////////////////////////////////////////////////////
AppWizard has created the following resources:

TestApp.rc
    This is a listing of all of the Microsoft Windows resources that the
    program uses.  It includes the icons, bitmaps, and cursors that are stored
    in the RES subdirectory.  This file can be directly edited in Microsoft
	Visual C++.

res\TestApp.ico
    This is an icon file, which is used as the application's icon (32x32).
    This icon is included by the main resource file TestApp.rc.

small.ico
    %%This is an icon file, which contains a smaller version (16x16)
	of the application's icon. This icon is included by the main resource
	file TestApp.rc.

/////////////////////////////////////////////////////////////////////////////
Other standard files:

StdAfx.h, StdAfx.cpp
    These files are used to build a precompiled header (PCH) file
    named TestApp.pch and a precompiled types file named StdAfx.obj.

Resource.h
    This is the standard header file, which defines new resource IDs.
    Microsoft Visual C++ reads and updates this file.

/////////////////////////////////////////////////////////////////////////////
Other notes:

AppWizard uses "TODO:" to indicate parts of the source code you
should add to or customize.


/////////////////////////////////////////////////////////////////////////////
