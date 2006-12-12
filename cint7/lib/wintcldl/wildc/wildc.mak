# Microsoft Developer Studio Generated NMAKE File, Format Version 4.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Dynamic-Link Library" 0x0102

!IF "$(CFG)" == ""
CFG=wildc - Win32 Debug
!MESSAGE No configuration specified.  Defaulting to wildc - Win32 Debug.
!ENDIF 

!IF "$(CFG)" != "wildc - Win32 Release" && "$(CFG)" != "wildc - Win32 Debug"
!MESSAGE Invalid configuration "$(CFG)" specified.
!MESSAGE You can specify a configuration when running NMAKE on this makefile
!MESSAGE by defining the macro CFG on the command line.  For example:
!MESSAGE 
!MESSAGE NMAKE /f "wildc.mak" CFG="wildc - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "wildc - Win32 Release" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "wildc - Win32 Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE 
!ERROR An invalid configuration is specified.
!ENDIF 

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE 
NULL=nul
!ENDIF 
################################################################################
# Begin Project
# PROP Target_Last_Scanned "wildc - Win32 Debug"
MTL=mktyplib.exe
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "wildc - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Target_Dir ""
OUTDIR=.\Release
INTDIR=.\Release

ALL : "$(OUTDIR)\wildc.dll"

CLEAN : 
	-@erase ".\Release\wildc.dll"
	-@erase ".\Release\CINTLIB.OBJ"
	-@erase ".\Release\WILDCIF.OBJ"
	-@erase ".\Release\wildc.lib"
	-@erase ".\Release\wildc.exp"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

# ADD BASE CPP /nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /YX /c
# ADD CPP /nologo /MT /w /W0 /GX /O2 /I "c:\cint" /I "c:\cint\src" /I "c:\cint\lib\vcstream" /I "c:\tcl\include" /I "c:\cint\lib\stdstrct" /D "NDEBUG" /D "WIN32" /D "_WINDOWS" /D "G__WILDC" /D "G__WIN32" /D "G__VISUAL" /D "G__SHAREDLIB" /D "G__REDIRECTIO" /D "G__SPECIALSTDIO" /YX /c
CPP_PROJ=/nologo /MT /w /W0 /GX /O2 /I "c:\cint" /I "c:\cint\src" /I\
 "c:\cint\lib\vcstream" /I "c:\tcl\include" /I "c:\cint\lib\stdstrct" /D\
 "NDEBUG" /D "WIN32" /D "_WINDOWS" /D "G__WILDC" /D "G__WIN32" /D "G__VISUAL" /D\
 "G__SHAREDLIB" /D "G__REDIRECTIO" /D "G__SPECIALSTDIO" /Fp"$(INTDIR)/wildc.pch"\
 /YX /Fo"$(INTDIR)/" /c 
CPP_OBJS=.\Release/
CPP_SBRS=
# ADD BASE MTL /nologo /D "NDEBUG" /win32
# ADD MTL /nologo /D "NDEBUG" /win32
MTL_PROJ=/nologo /D "NDEBUG" /win32 
# ADD BASE RSC /l 0x411 /d "NDEBUG"
# ADD RSC /l 0x411 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
BSC32_FLAGS=/nologo /o"$(OUTDIR)/wildc.bsc" 
BSC32_SBRS=
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib /nologo /subsystem:windows /dll /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib /nologo /subsystem:windows /dll /machine:I386
LINK32_FLAGS=kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib\
 advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib /nologo\
 /subsystem:windows /dll /incremental:no /pdb:"$(OUTDIR)/wildc.pdb"\
 /machine:I386 /def:"\CINT\LIB\WINTCLDL\WILDC.DEF" /out:"$(OUTDIR)/wildc.dll"\
 /implib:"$(OUTDIR)/wildc.lib" 
DEF_FILE= \
	"..\WILDC.DEF"
LINK32_OBJS= \
	".\Release\CINTLIB.OBJ" \
	".\Release\WILDCIF.OBJ" \
	"..\..\..\Libcint.lib" \
	"..\..\..\..\TCL\bin\tk41.lib" \
	"..\..\..\..\Tcl\bin\tcl75.lib"

"$(OUTDIR)\wildc.dll" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ELSEIF  "$(CFG)" == "wildc - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Target_Dir ""
OUTDIR=.\Debug
INTDIR=.\Debug

ALL : "$(OUTDIR)\wildc.dll"

CLEAN : 
	-@erase ".\Debug\vc40.pdb"
	-@erase ".\Debug\vc40.idb"
	-@erase ".\Debug\wildc.dll"
	-@erase ".\Debug\CINTLIB.OBJ"
	-@erase ".\Debug\WILDCIF.OBJ"
	-@erase ".\Debug\wildc.ilk"
	-@erase ".\Debug\wildc.lib"
	-@erase ".\Debug\wildc.exp"
	-@erase ".\Debug\wildc.pdb"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

# ADD BASE CPP /nologo /MTd /W3 /Gm /GX /Zi /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /YX /c
# ADD CPP /nologo /MTd /w /W0 /Gm /GX /Zi /Od /I "c:\cint" /I "c:\cint\src" /I "c:\cint\lib\vcstream" /I "c:\tcl\include" /I "c:\cint\lib\stdstrct" /D "_DEBUG" /D "WIN32" /D "_WINDOWS" /D "G__WILDC" /D "G__WIN32" /D "G__VISUAL" /D "G__SHAREDLIB" /D "G__REDIRECTIO" /D "G__SPECIALSTDIO" /YX /c
CPP_PROJ=/nologo /MTd /w /W0 /Gm /GX /Zi /Od /I "c:\cint" /I "c:\cint\src" /I\
 "c:\cint\lib\vcstream" /I "c:\tcl\include" /I "c:\cint\lib\stdstrct" /D\
 "_DEBUG" /D "WIN32" /D "_WINDOWS" /D "G__WILDC" /D "G__WIN32" /D "G__VISUAL" /D\
 "G__SHAREDLIB" /D "G__REDIRECTIO" /D "G__SPECIALSTDIO" /Fp"$(INTDIR)/wildc.pch"\
 /YX /Fo"$(INTDIR)/" /Fd"$(INTDIR)/" /c 
CPP_OBJS=.\Debug/
CPP_SBRS=
# ADD BASE MTL /nologo /D "_DEBUG" /win32
# ADD MTL /nologo /D "_DEBUG" /win32
MTL_PROJ=/nologo /D "_DEBUG" /win32 
# ADD BASE RSC /l 0x411 /d "_DEBUG"
# ADD RSC /l 0x411 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
BSC32_FLAGS=/nologo /o"$(OUTDIR)/wildc.bsc" 
BSC32_SBRS=
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib /nologo /subsystem:windows /dll /debug /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib /nologo /subsystem:windows /dll /debug /machine:I386
LINK32_FLAGS=kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib\
 advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib /nologo\
 /subsystem:windows /dll /incremental:yes /pdb:"$(OUTDIR)/wildc.pdb" /debug\
 /machine:I386 /def:"\CINT\LIB\WINTCLDL\WILDC.DEF" /out:"$(OUTDIR)/wildc.dll"\
 /implib:"$(OUTDIR)/wildc.lib" 
DEF_FILE= \
	"..\WILDC.DEF"
LINK32_OBJS= \
	".\Debug\CINTLIB.OBJ" \
	".\Debug\WILDCIF.OBJ" \
	"..\..\..\Libcint.lib" \
	"..\..\..\..\TCL\bin\tk41.lib" \
	"..\..\..\..\Tcl\bin\tcl75.lib"

"$(OUTDIR)\wildc.dll" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

!ENDIF 

.c{$(CPP_OBJS)}.obj:
   $(CPP) $(CPP_PROJ) $<  

.cpp{$(CPP_OBJS)}.obj:
   $(CPP) $(CPP_PROJ) $<  

.cxx{$(CPP_OBJS)}.obj:
   $(CPP) $(CPP_PROJ) $<  

.c{$(CPP_SBRS)}.sbr:
   $(CPP) $(CPP_PROJ) $<  

.cpp{$(CPP_SBRS)}.sbr:
   $(CPP) $(CPP_PROJ) $<  

.cxx{$(CPP_SBRS)}.sbr:
   $(CPP) $(CPP_PROJ) $<  

################################################################################
# Begin Target

# Name "wildc - Win32 Release"
# Name "wildc - Win32 Debug"

!IF  "$(CFG)" == "wildc - Win32 Release"

!ELSEIF  "$(CFG)" == "wildc - Win32 Debug"

!ENDIF 

################################################################################
# Begin Source File

SOURCE=\CINT\LIB\WINTCLDL\CINTLIB.C

!IF  "$(CFG)" == "wildc - Win32 Release"

DEP_CPP_CINTL=\
	"c:\tcl\include\tcl.h"\
	"c:\cint\G__ci.h"\
	".\..\..\..\src\sunos.h"\
	".\..\..\..\src\newsos.h"\
	".\..\..\..\src\memtest.h"\
	

"$(INTDIR)\CINTLIB.OBJ" : $(SOURCE) $(DEP_CPP_CINTL) "$(INTDIR)"
   $(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "wildc - Win32 Debug"

DEP_CPP_CINTL=\
	"c:\tcl\include\tcl.h"\
	"c:\cint\G__ci.h"\
	

"$(INTDIR)\CINTLIB.OBJ" : $(SOURCE) $(DEP_CPP_CINTL) "$(INTDIR)"
   $(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=\CINT\LIB\WINTCLDL\WILDCIF.C

!IF  "$(CFG)" == "wildc - Win32 Release"

DEP_CPP_WILDC=\
	".\..\wildcIF.h"\
	"c:\cint\G__ci.h"\
	".\..\TOP.h"\
	".\..\..\..\src\sunos.h"\
	".\..\..\..\src\newsos.h"\
	".\..\..\..\src\memtest.h"\
	

"$(INTDIR)\WILDCIF.OBJ" : $(SOURCE) $(DEP_CPP_WILDC) "$(INTDIR)"
   $(CPP) $(CPP_PROJ) $(SOURCE)


!ELSEIF  "$(CFG)" == "wildc - Win32 Debug"

DEP_CPP_WILDC=\
	".\..\wildcIF.h"\
	"c:\cint\G__ci.h"\
	".\..\TOP.h"\
	".\..\TCLTK.h"\
	"c:\tcl\include\tk.h"\
	

"$(INTDIR)\WILDCIF.OBJ" : $(SOURCE) $(DEP_CPP_WILDC) "$(INTDIR)"
   $(CPP) $(CPP_PROJ) $(SOURCE)


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=\CINT\LIB\WINTCLDL\WILDC.DEF

!IF  "$(CFG)" == "wildc - Win32 Release"

!ELSEIF  "$(CFG)" == "wildc - Win32 Debug"

!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=\TCL\bin\tk41.lib

!IF  "$(CFG)" == "wildc - Win32 Release"

!ELSEIF  "$(CFG)" == "wildc - Win32 Debug"

!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=\cint\Libcint.lib

!IF  "$(CFG)" == "wildc - Win32 Release"

!ELSEIF  "$(CFG)" == "wildc - Win32 Debug"

!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=\Tcl\bin\tcl75.lib

!IF  "$(CFG)" == "wildc - Win32 Release"

!ELSEIF  "$(CFG)" == "wildc - Win32 Debug"

!ENDIF 

# End Source File
# End Target
# End Project
################################################################################
