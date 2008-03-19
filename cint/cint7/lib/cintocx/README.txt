lib/cintocx/readme.txt

CINTOCX.ocx is an OLE control library to combine VisualBasic4.0 and CINT
C++ interpreter.  CINTOCX.ocx is an invisible OLE object which can be 
pasted on to VisualBasic GUI forms. 

# How to build on Visual C++ 4.0

 Double click cintocx.mdp from Explorer or File Manager and build the 
 project. cintocx.ocx object is built and automatically registered.

 [If you use later version of Visual C++, Visual Studio will ask you
 if you want to translate it to a new format. Answer 'yes' to create
 cintocx.dsw and cintocx.dsp.  After the translation, choose Release
 (non unicode version) by 'Build->Set Active Configuration' from the
 pulldown menu.]


# cintocx.ocx manual registeration

 (Note: Location of cintocx.ocx has been changed 
  from %cintsysdir%\cintocx.ocx to %cintsysdir%\lib\cintocx\cintocx.ocx)

 cintocx.ocx is included in CINT Windows-NT/95 binary package. Install 
 CINT binary package by setup script. You'll find 

         %CINTSYSDIR%\lib\cintocx\cintocx.ocx
         %CINTSYSDIR%\regsvr32.exe

     C:\> %CINTSYSDIR%\REGSVR32.EXE %CINTSYSDIR%\lib\cintocx\cintocx.ocx

 Start VisualBasic4.0. You will find "cintocx.ocx custom control" by 
 opening OLE control selection diaglog from Tool->Custom Control menu bar.


# CINTOCX.ocx specification

 Methods:
   Init(String CintCommand)
   Eval(String C++Expression)
   Terminate
   Stepmode
   Interrupt
 Property:
   String Result
 Event:
   EvalDone


Architecture:

  +-----------------+                                      +-----------+
  | VisualBasic     |                                      |CINTOCX.OCX|
  |                 |                                      |           |
  |                 | call                                 |           |
  |                 | ---Cintocx1.Init("cint Src.cxx")---> |           |
  |                 |                                      |           |
  |                 |                          Fire event  |           |
  |Cintocx1_EvalDone| <-----------EvalDone---------------- |           |
  |                 |                                      |           |
  |StrRslt =        | Read property                        |           |
  |  Cintocx1.Result| ............Result.................. |           |
  |                 |                                      |           |
  |                 |                                      |           |
  |                 |                                      |           |
  |                 |                                      |           |
  |                 |                                      |           |
  |                 | call                                 |           |
  |                 | ---Cintocx1.Eval("doit(3.14)")-----> |           |
  |                 |                                      |           |
  |                 |                          Fire event  |           |
  |Cintocx1_EvalDone| <-----------EvalDone---------------- |           |
  |                 |                                      |           |
  |StrRslt =        | Read property                        |           |
  |  Cintocx1.Result| ............Result.................. |           |
  |                 |                                      |           |
  |                 |                                      |           |
  |                 |                                      |           |
  |                 |                                      |           |
  |                 |                                      |           |
  |                 | call                                 |           |
  |                 | ---Cintocx1.Stepmode---------------> |           |
  |                 |                                      |           |
  |                 |                                      |           |
  |                 | call                                 |           |
  |                 | ---Cintocx1.Interrupt--------------> |           |
  |                 |                                      |           |
  |                 |                                      |           |
  |                 |                                      |           |
  |                 |                                      |           |
  |                 | call                                 |           |
  |                 | ---Cintocx1.Terminate--------------> |           |
  |                 |                                      |           |
  |                 |                                      |           |
  +-----------------+                                      +-----------+

Initialization:
 CINT and VisualBasic work in separate threads. First, you need to initialize
 CINT and its' thread by Init() method. Init method takes one String argument.
 Format of the Init string argument is "cint [options] [sourcefiles]". If
 there is a main() function in the given source file, it will be evaluated.

Evaluating C/C++ expression:
 C/C++ expression can be evaluated by Eval() method. Eval() method returns 
 immediately after scheduling the given expression to a que. When evaluation
 is finished, CINTOCX fires EvalDone event. You can refer to Result property 
 to get evaluated value of the experession.

Thread Termination:
 You MUST call Terminate method before leaving the session. Terminate method
 kills CINT thread. Otherwides, you will find strange behavior.

Debugging and Interruption:
 Stepmode method sets step execution flag. You can debug C/C++ source code
 from character console. Interrupt method is similar but breaks the CINT
 thread immediately.

Character console:
 Character console can be created and destructed by following command.
   Cintocx1.Eval("G__AllocConsole()")
   Cintocx1.Eval("G__FreeConsole()")

Multiple CINTOCX objects:
 You can put as many CINTOCX.ocx objects on the form , but there
 is always only one CINT object.  Multiple CINTOCX objects communicates to
 CINT core by Event-Que. Requests are synchronized and aligned by the Que 
 and evaluated sequencially.


========================================================================
		OLE Control DLL : CINTOCX
========================================================================

ControlWizard has created this project for your CINTOCX OLE Control DLL,
which contains 1 control.

This skeleton project not only demonstrates the basics of writing an OLE
Control, but is also a starting point for writing the specific features
of your control.

This file contains a summary of what you will find in each of the files
that make up your CINTOCX OLE Control DLL.

cintocx.mak
	The Visual C++ project makefile for building your OLE Control.

cintocx.h
	This is the main include file for the OLE Control DLL.  It
	includes other project-specific includes such as resource.h.

cintocx.cpp
	This is the main source file that contains code for DLL initialization,
	termination and other bookkeeping.

cintocx.rc
	This is a listing of the Microsoft Windows resources that the project
	uses.  This file can be directly edited with the Visual C++ resource
	editor.

cintocx.def
	This file contains information about the OLE Control DLL that
	must be provided to run with Microsoft Windows.

cintocx.clw
	This file contains information used by ClassWizard to edit existing
	classes or add new classes.  ClassWizard also uses this file to store
	information needed to generate and edit message maps and dialog data
	maps and to generate prototype member functions.

cintocx.odl
	This file contains the Object Description Language source code for the
	type library of your control.

/////////////////////////////////////////////////////////////////////////////
Cintocx control:

CintocxCtl.h
	This file contains the declaration of the CCintocxCtrl C++ class.

CintocxCtl.cpp
	This file contains the implementation of the CCintocxCtrl C++ class.

CintocxPpg.h
	This file contains the declaration of the CCintocxPropPage C++ class.

CintocxPpg.cpp
	This file contains the implementation of the CCintocxPropPage C++ class.

CintocxCtl.bmp
	This file contains a bitmap that a container will use to represent the
	CCintocxCtrl control when it appears on a tool palette.  This bitmap
	is included by the main resource file cintocx.rc.

/////////////////////////////////////////////////////////////////////////////
Other standard files:

stdafx.h, stdafx.cpp
	These files are used to build a precompiled header (PCH) file
	named stdafx.pch and a precompiled types (PCT) file named stdafx.obj.

resource.h
	This is the standard header file, which defines new resource IDs.
	The Visual C++ resource editor reads and updates this file.

/////////////////////////////////////////////////////////////////////////////
Other notes:

ControlWizard uses "TODO:" to indicate parts of the source code you
should add to or customize.

/////////////////////////////////////////////////////////////////////////////
