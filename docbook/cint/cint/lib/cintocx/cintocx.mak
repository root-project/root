# Microsoft Developer Studio Generated NMAKE File, Format Version 4.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Dynamic-Link Library" 0x0102

!IF "$(CFG)" == ""
CFG=cintocx - Win32 Debug
!MESSAGE No configuration specified.  Defaulting to cintocx - Win32 Debug.
!ENDIF 

!IF "$(CFG)" != "cintocx - Win32 Release" && "$(CFG)" !=\
 "cintocx - Win32 Debug" && "$(CFG)" != "cintocx - Win32 Unicode Debug" &&\
 "$(CFG)" != "cintocx - Win32 Unicode Release"
!MESSAGE Invalid configuration "$(CFG)" specified.
!MESSAGE You can specify a configuration when running NMAKE on this makefile
!MESSAGE by defining the macro CFG on the command line.  For example:
!MESSAGE 
!MESSAGE NMAKE /f "cintocx.mak" CFG="cintocx - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "cintocx - Win32 Release" (based on\
 "Win32 (x86) Dynamic-Link Library")
!MESSAGE "cintocx - Win32 Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "cintocx - Win32 Unicode Debug" (based on\
 "Win32 (x86) Dynamic-Link Library")
!MESSAGE "cintocx - Win32 Unicode Release" (based on\
 "Win32 (x86) Dynamic-Link Library")
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
# PROP Target_Last_Scanned "cintocx - Win32 Debug"
MTL=mktyplib.exe
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "cintocx - Win32 Release"

# PROP BASE Use_MFC 2
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP BASE Target_Ext "ocx"
# PROP Use_MFC 2
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Target_Dir ""
# PROP Target_Ext "ocx"
OUTDIR=.\Release
INTDIR=.\Release
# Begin Custom Macros
OutDir=.\Release
# End Custom Macros

ALL : "$(OUTDIR)\cintocx.ocx" "$(OUTDIR)\regsvr32.trg"

CLEAN : 
	-@erase ".\Release\cintocx.lib"
	-@erase ".\Release\cintocx.obj"
	-@erase ".\Release\cintocx.pch"
	-@erase ".\Release\CintocxPpg.obj"
	-@erase ".\Release\CintocxCtl.obj"
	-@erase ".\Release\StdAfx.obj"
	-@erase ".\Release\cintocx.res"
	-@erase ".\Release\cintocx.tlb"
	-@erase ".\Release\cintocx.exp"
	-@erase ".\Release\regsvr32.trg"
	-@erase ".\Release\cintocx.ocx"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

# ADD BASE CPP /nologo /MD /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_WINDLL" /D "_AFXDLL" /D "_MBCS" /Yu"stdafx.h" /c
# ADD CPP /nologo /MD /W3 /GX /O2 /I "c:\cint" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_WINDLL" /D "_AFXDLL" /D "_MBCS" /D "_USRDLL" /Yu"stdafx.h" /c
CPP_PROJ=/nologo /MD /W3 /GX /O2 /I "c:\cint" /D "WIN32" /D "NDEBUG" /D\
 "_WINDOWS" /D "_WINDLL" /D "_AFXDLL" /D "_MBCS" /D "_USRDLL"\
 /Fp"$(INTDIR)/cintocx.pch" /Yu"stdafx.h" /Fo"$(INTDIR)/" /c 
CPP_OBJS=.\Release/
CPP_SBRS=
# ADD BASE MTL /nologo /D "NDEBUG" /win32
# ADD MTL /nologo /D "NDEBUG" /win32
MTL_PROJ=/nologo /D "NDEBUG" /win32 
# ADD BASE RSC /l 0x411 /d "NDEBUG" /d "_AFXDLL"
# ADD RSC /l 0x411 /d "NDEBUG" /d "_AFXDLL"
RSC_PROJ=/l 0x411 /fo"$(INTDIR)/cintocx.res" /d "NDEBUG" /d "_AFXDLL" 
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
BSC32_FLAGS=/nologo /o"$(OUTDIR)/cintocx.bsc" 
BSC32_SBRS=
LINK32=link.exe
# ADD BASE LINK32 /nologo /subsystem:windows /dll /machine:I386
# ADD LINK32 /nologo /subsystem:windows /dll /machine:I386
LINK32_FLAGS=/nologo /subsystem:windows /dll /incremental:no\
 /pdb:"$(OUTDIR)/cintocx.pdb" /machine:I386 /def:".\cintocx.def"\
 /out:"$(OUTDIR)/cintocx.ocx" /implib:"$(OUTDIR)/cintocx.lib" 
DEF_FILE= \
	".\cintocx.def"
LINK32_OBJS= \
	"$(INTDIR)/cintocx.obj" \
	"$(INTDIR)/CintocxPpg.obj" \
	"$(INTDIR)/CintocxCtl.obj" \
	"$(INTDIR)/StdAfx.obj" \
	"$(INTDIR)/cintocx.res" \
	"..\..\libcint.lib"

"$(OUTDIR)\cintocx.ocx" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

# Begin Custom Build - Registering OLE control...
OutDir=.\Release
TargetPath=.\Release\cintocx.ocx
InputPath=.\Release\cintocx.ocx
SOURCE=$(InputPath)

"$(OutDir)\regsvr32.trg" : $(SOURCE) "$(INTDIR)" "$(OUTDIR)"
   regsvr32 /s /c "$(TargetPath)"
   echo regsvr32 exec. time > "$(OutDir)\regsvr32.trg"

# End Custom Build

!ELSEIF  "$(CFG)" == "cintocx - Win32 Debug"

# PROP BASE Use_MFC 2
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP BASE Target_Ext "ocx"
# PROP Use_MFC 2
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Target_Dir ""
# PROP Target_Ext "ocx"
OUTDIR=.\Debug
INTDIR=.\Debug
# Begin Custom Macros
OutDir=.\Debug
# End Custom Macros

ALL : "$(OUTDIR)\cintocx.ocx" "$(OUTDIR)\regsvr32.trg"

CLEAN : 
	-@erase ".\Debug\vc40.pdb"
	-@erase ".\Debug\cintocx.pch"
	-@erase ".\Debug\vc40.idb"
	-@erase ".\Debug\cintocx.ilk"
	-@erase ".\Debug\cintocx.obj"
	-@erase ".\Debug\CintocxPpg.obj"
	-@erase ".\Debug\CintocxCtl.obj"
	-@erase ".\Debug\StdAfx.obj"
	-@erase ".\Debug\cintocx.res"
	-@erase ".\Debug\cintocx.tlb"
	-@erase ".\Debug\cintocx.lib"
	-@erase ".\Debug\cintocx.exp"
	-@erase ".\Debug\cintocx.pdb"
	-@erase ".\Debug\regsvr32.trg"
	-@erase ".\Debug\cintocx.ocx"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

# ADD BASE CPP /nologo /MDd /W3 /Gm /GX /Zi /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_WINDLL" /D "_AFXDLL" /D "_MBCS" /Yu"stdafx.h" /c
# ADD CPP /nologo /MDd /W3 /Gm /GX /Zi /Od /I "c:\cint" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_WINDLL" /D "_AFXDLL" /D "_MBCS" /D "_USRDLL" /Yu"stdafx.h" /c
CPP_PROJ=/nologo /MDd /W3 /Gm /GX /Zi /Od /I "c:\cint" /D "WIN32" /D "_DEBUG"\
 /D "_WINDOWS" /D "_WINDLL" /D "_AFXDLL" /D "_MBCS" /D "_USRDLL"\
 /Fp"$(INTDIR)/cintocx.pch" /Yu"stdafx.h" /Fo"$(INTDIR)/" /Fd"$(INTDIR)/" /c 
CPP_OBJS=.\Debug/
CPP_SBRS=
# ADD BASE MTL /nologo /D "_DEBUG" /win32
# ADD MTL /nologo /D "_DEBUG" /win32
MTL_PROJ=/nologo /D "_DEBUG" /win32 
# ADD BASE RSC /l 0x411 /d "_DEBUG" /d "_AFXDLL"
# ADD RSC /l 0x411 /d "_DEBUG" /d "_AFXDLL"
RSC_PROJ=/l 0x411 /fo"$(INTDIR)/cintocx.res" /d "_DEBUG" /d "_AFXDLL" 
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
BSC32_FLAGS=/nologo /o"$(OUTDIR)/cintocx.bsc" 
BSC32_SBRS=
LINK32=link.exe
# ADD BASE LINK32 /nologo /subsystem:windows /dll /debug /machine:I386
# ADD LINK32 /nologo /subsystem:windows /dll /debug /machine:I386
LINK32_FLAGS=/nologo /subsystem:windows /dll /incremental:yes\
 /pdb:"$(OUTDIR)/cintocx.pdb" /debug /machine:I386 /def:".\cintocx.def"\
 /out:"$(OUTDIR)/cintocx.ocx" /implib:"$(OUTDIR)/cintocx.lib" 
DEF_FILE= \
	".\cintocx.def"
LINK32_OBJS= \
	"$(INTDIR)/cintocx.obj" \
	"$(INTDIR)/CintocxPpg.obj" \
	"$(INTDIR)/CintocxCtl.obj" \
	"$(INTDIR)/StdAfx.obj" \
	"$(INTDIR)/cintocx.res" \
	"..\..\libcint.lib"

"$(OUTDIR)\cintocx.ocx" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

# Begin Custom Build - Registering OLE control...
OutDir=.\Debug
TargetPath=.\Debug\cintocx.ocx
InputPath=.\Debug\cintocx.ocx
SOURCE=$(InputPath)

"$(OutDir)\regsvr32.trg" : $(SOURCE) "$(INTDIR)" "$(OUTDIR)"
   regsvr32 /s /c "$(TargetPath)"
   echo regsvr32 exec. time > "$(OutDir)\regsvr32.trg"

# End Custom Build

!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Debug"

# PROP BASE Use_MFC 2
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "DebugU"
# PROP BASE Intermediate_Dir "DebugU"
# PROP BASE Target_Dir ""
# PROP BASE Target_Ext "ocx"
# PROP Use_MFC 2
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "DebugU"
# PROP Intermediate_Dir "DebugU"
# PROP Target_Dir ""
# PROP Target_Ext "ocx"
OUTDIR=.\DebugU
INTDIR=.\DebugU
# Begin Custom Macros
OutDir=.\DebugU
# End Custom Macros

ALL : "$(OUTDIR)\cintocx.ocx" "$(OUTDIR)\regsvr32.trg"

CLEAN : 
	-@erase ".\DebugU\vc40.pdb"
	-@erase ".\DebugU\cintocx.pch"
	-@erase ".\DebugU\vc40.idb"
	-@erase ".\DebugU\cintocx.ilk"
	-@erase ".\DebugU\cintocx.obj"
	-@erase ".\DebugU\CintocxPpg.obj"
	-@erase ".\DebugU\CintocxCtl.obj"
	-@erase ".\DebugU\StdAfx.obj"
	-@erase ".\DebugU\cintocx.res"
	-@erase ".\DebugU\cintocx.tlb"
	-@erase ".\DebugU\cintocx.lib"
	-@erase ".\DebugU\cintocx.exp"
	-@erase ".\DebugU\cintocx.pdb"
	-@erase ".\DebugU\regsvr32.trg"
	-@erase ".\DebugU\cintocx.ocx"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

# ADD BASE CPP /nologo /MDd /W3 /Gm /GX /Zi /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_WINDLL" /D "_AFXDLL" /D "_MBCS" /D "_USRDLL" /Yu"stdafx.h" /c
# ADD CPP /nologo /MDd /W3 /Gm /GX /Zi /Od /I "c:\cint" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_WINDLL" /D "_AFXDLL" /D "_USRDLL" /D "_UNICODE" /Yu"stdafx.h" /c
CPP_PROJ=/nologo /MDd /W3 /Gm /GX /Zi /Od /I "c:\cint" /D "WIN32" /D "_DEBUG"\
 /D "_WINDOWS" /D "_WINDLL" /D "_AFXDLL" /D "_USRDLL" /D "_UNICODE"\
 /Fp"$(INTDIR)/cintocx.pch" /Yu"stdafx.h" /Fo"$(INTDIR)/" /Fd"$(INTDIR)/" /c 
CPP_OBJS=.\DebugU/
CPP_SBRS=
# ADD BASE MTL /nologo /D "_DEBUG" /win32
# ADD MTL /nologo /D "_DEBUG" /win32
MTL_PROJ=/nologo /D "_DEBUG" /win32 
# ADD BASE RSC /l 0x411 /d "_DEBUG" /d "_AFXDLL"
# ADD RSC /l 0x411 /d "_DEBUG" /d "_AFXDLL"
RSC_PROJ=/l 0x411 /fo"$(INTDIR)/cintocx.res" /d "_DEBUG" /d "_AFXDLL" 
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
BSC32_FLAGS=/nologo /o"$(OUTDIR)/cintocx.bsc" 
BSC32_SBRS=
LINK32=link.exe
# ADD BASE LINK32 /nologo /subsystem:windows /dll /debug /machine:I386
# ADD LINK32 /nologo /subsystem:windows /dll /debug /machine:I386
LINK32_FLAGS=/nologo /subsystem:windows /dll /incremental:yes\
 /pdb:"$(OUTDIR)/cintocx.pdb" /debug /machine:I386 /def:".\cintocx.def"\
 /out:"$(OUTDIR)/cintocx.ocx" /implib:"$(OUTDIR)/cintocx.lib" 
DEF_FILE= \
	".\cintocx.def"
LINK32_OBJS= \
	"$(INTDIR)/cintocx.obj" \
	"$(INTDIR)/CintocxPpg.obj" \
	"$(INTDIR)/CintocxCtl.obj" \
	"$(INTDIR)/StdAfx.obj" \
	"$(INTDIR)/cintocx.res" \
	"..\..\libcint.lib"

"$(OUTDIR)\cintocx.ocx" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

# Begin Custom Build - Registering OLE control...
OutDir=.\DebugU
TargetPath=.\DebugU\cintocx.ocx
InputPath=.\DebugU\cintocx.ocx
SOURCE=$(InputPath)

"$(OutDir)\regsvr32.trg" : $(SOURCE) "$(INTDIR)" "$(OUTDIR)"
   regsvr32 /s /c "$(TargetPath)"
   echo regsvr32 exec. time > "$(OutDir)\regsvr32.trg"

# End Custom Build

!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Release"

# PROP BASE Use_MFC 2
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "ReleaseU"
# PROP BASE Intermediate_Dir "ReleaseU"
# PROP BASE Target_Dir ""
# PROP BASE Target_Ext "ocx"
# PROP Use_MFC 2
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "ReleaseU"
# PROP Intermediate_Dir "ReleaseU"
# PROP Target_Dir ""
# PROP Target_Ext "ocx"
OUTDIR=.\ReleaseU
INTDIR=.\ReleaseU
# Begin Custom Macros
OutDir=.\ReleaseU
# End Custom Macros

ALL : "$(OUTDIR)\cintocx.ocx" "$(OUTDIR)\regsvr32.trg"

CLEAN : 
	-@erase ".\ReleaseU\cintocx.lib"
	-@erase ".\ReleaseU\cintocx.obj"
	-@erase ".\ReleaseU\cintocx.pch"
	-@erase ".\ReleaseU\StdAfx.obj"
	-@erase ".\ReleaseU\CintocxPpg.obj"
	-@erase ".\ReleaseU\CintocxCtl.obj"
	-@erase ".\ReleaseU\cintocx.res"
	-@erase ".\ReleaseU\cintocx.tlb"
	-@erase ".\ReleaseU\cintocx.exp"
	-@erase ".\ReleaseU\regsvr32.trg"
	-@erase ".\ReleaseU\cintocx.ocx"

"$(OUTDIR)" :
    if not exist "$(OUTDIR)/$(NULL)" mkdir "$(OUTDIR)"

# ADD BASE CPP /nologo /MD /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_WINDLL" /D "_AFXDLL" /D "_MBCS" /D "_USRDLL" /Yu"stdafx.h" /c
# ADD CPP /nologo /MD /W3 /GX /O2 /I "c:\cint" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_WINDLL" /D "_AFXDLL" /D "_USRDLL" /D "_UNICODE" /Yu"stdafx.h" /c
CPP_PROJ=/nologo /MD /W3 /GX /O2 /I "c:\cint" /D "WIN32" /D "NDEBUG" /D\
 "_WINDOWS" /D "_WINDLL" /D "_AFXDLL" /D "_USRDLL" /D "_UNICODE"\
 /Fp"$(INTDIR)/cintocx.pch" /Yu"stdafx.h" /Fo"$(INTDIR)/" /c 
CPP_OBJS=.\ReleaseU/
CPP_SBRS=
# ADD BASE MTL /nologo /D "NDEBUG" /win32
# ADD MTL /nologo /D "NDEBUG" /win32
MTL_PROJ=/nologo /D "NDEBUG" /win32 
# ADD BASE RSC /l 0x411 /d "NDEBUG" /d "_AFXDLL"
# ADD RSC /l 0x411 /d "NDEBUG" /d "_AFXDLL"
RSC_PROJ=/l 0x411 /fo"$(INTDIR)/cintocx.res" /d "NDEBUG" /d "_AFXDLL" 
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
BSC32_FLAGS=/nologo /o"$(OUTDIR)/cintocx.bsc" 
BSC32_SBRS=
LINK32=link.exe
# ADD BASE LINK32 /nologo /subsystem:windows /dll /machine:I386
# ADD LINK32 /nologo /subsystem:windows /dll /machine:I386
LINK32_FLAGS=/nologo /subsystem:windows /dll /incremental:no\
 /pdb:"$(OUTDIR)/cintocx.pdb" /machine:I386 /def:".\cintocx.def"\
 /out:"$(OUTDIR)/cintocx.ocx" /implib:"$(OUTDIR)/cintocx.lib" 
DEF_FILE= \
	".\cintocx.def"
LINK32_OBJS= \
	"$(INTDIR)/cintocx.obj" \
	"$(INTDIR)/StdAfx.obj" \
	"$(INTDIR)/CintocxPpg.obj" \
	"$(INTDIR)/CintocxCtl.obj" \
	"$(INTDIR)/cintocx.res" \
	"..\..\libcint.lib"

"$(OUTDIR)\cintocx.ocx" : "$(OUTDIR)" $(DEF_FILE) $(LINK32_OBJS)
    $(LINK32) @<<
  $(LINK32_FLAGS) $(LINK32_OBJS)
<<

# Begin Custom Build - Registering OLE control...
OutDir=.\ReleaseU
TargetPath=.\ReleaseU\cintocx.ocx
InputPath=.\ReleaseU\cintocx.ocx
SOURCE=$(InputPath)

"$(OutDir)\regsvr32.trg" : $(SOURCE) "$(INTDIR)" "$(OUTDIR)"
   regsvr32 /s /c "$(TargetPath)"
   echo regsvr32 exec. time > "$(OutDir)\regsvr32.trg"

# End Custom Build

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

# Name "cintocx - Win32 Release"
# Name "cintocx - Win32 Debug"
# Name "cintocx - Win32 Unicode Debug"
# Name "cintocx - Win32 Unicode Release"

!IF  "$(CFG)" == "cintocx - Win32 Release"

!ELSEIF  "$(CFG)" == "cintocx - Win32 Debug"

!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Debug"

!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Release"

!ENDIF 

################################################################################
# Begin Source File

SOURCE=.\ReadMe.txt

!IF  "$(CFG)" == "cintocx - Win32 Release"

!ELSEIF  "$(CFG)" == "cintocx - Win32 Debug"

!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Debug"

!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Release"

!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\StdAfx.cpp
DEP_CPP_STDAF=\
	".\StdAfx.h"\
	

!IF  "$(CFG)" == "cintocx - Win32 Release"

# ADD CPP /Yc"stdafx.h"

BuildCmds= \
	$(CPP) /nologo /MD /W3 /GX /O2 /I "c:\cint" /D "WIN32" /D "NDEBUG" /D\
 "_WINDOWS" /D "_WINDLL" /D "_AFXDLL" /D "_MBCS" /D "_USRDLL"\
 /Fp"$(INTDIR)/cintocx.pch" /Yc"stdafx.h" /Fo"$(INTDIR)/" /c $(SOURCE) \
	

"$(INTDIR)\StdAfx.obj" : $(SOURCE) $(DEP_CPP_STDAF) "$(INTDIR)"
   $(BuildCmds)

"$(INTDIR)\cintocx.pch" : $(SOURCE) $(DEP_CPP_STDAF) "$(INTDIR)"
   $(BuildCmds)

!ELSEIF  "$(CFG)" == "cintocx - Win32 Debug"

# ADD CPP /Yc"stdafx.h"

BuildCmds= \
	$(CPP) /nologo /MDd /W3 /Gm /GX /Zi /Od /I "c:\cint" /D "WIN32" /D "_DEBUG" /D\
 "_WINDOWS" /D "_WINDLL" /D "_AFXDLL" /D "_MBCS" /D "_USRDLL"\
 /Fp"$(INTDIR)/cintocx.pch" /Yc"stdafx.h" /Fo"$(INTDIR)/" /Fd"$(INTDIR)/" /c\
 $(SOURCE) \
	

"$(INTDIR)\StdAfx.obj" : $(SOURCE) $(DEP_CPP_STDAF) "$(INTDIR)"
   $(BuildCmds)

"$(INTDIR)\cintocx.pch" : $(SOURCE) $(DEP_CPP_STDAF) "$(INTDIR)"
   $(BuildCmds)

!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Debug"

# ADD BASE CPP /Yc"stdafx.h"
# ADD CPP /Yc"stdafx.h"

BuildCmds= \
	$(CPP) /nologo /MDd /W3 /Gm /GX /Zi /Od /I "c:\cint" /D "WIN32" /D "_DEBUG" /D\
 "_WINDOWS" /D "_WINDLL" /D "_AFXDLL" /D "_USRDLL" /D "_UNICODE"\
 /Fp"$(INTDIR)/cintocx.pch" /Yc"stdafx.h" /Fo"$(INTDIR)/" /Fd"$(INTDIR)/" /c\
 $(SOURCE) \
	

"$(INTDIR)\StdAfx.obj" : $(SOURCE) $(DEP_CPP_STDAF) "$(INTDIR)"
   $(BuildCmds)

"$(INTDIR)\cintocx.pch" : $(SOURCE) $(DEP_CPP_STDAF) "$(INTDIR)"
   $(BuildCmds)

!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Release"

# ADD BASE CPP /Yc"stdafx.h"
# ADD CPP /Yc"stdafx.h"

BuildCmds= \
	$(CPP) /nologo /MD /W3 /GX /O2 /I "c:\cint" /D "WIN32" /D "NDEBUG" /D\
 "_WINDOWS" /D "_WINDLL" /D "_AFXDLL" /D "_USRDLL" /D "_UNICODE"\
 /Fp"$(INTDIR)/cintocx.pch" /Yc"stdafx.h" /Fo"$(INTDIR)/" /c $(SOURCE) \
	

"$(INTDIR)\StdAfx.obj" : $(SOURCE) $(DEP_CPP_STDAF) "$(INTDIR)"
   $(BuildCmds)

"$(INTDIR)\cintocx.pch" : $(SOURCE) $(DEP_CPP_STDAF) "$(INTDIR)"
   $(BuildCmds)

!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\cintocx.cpp
DEP_CPP_CINTO=\
	".\StdAfx.h"\
	".\cintocx.h"\
	

!IF  "$(CFG)" == "cintocx - Win32 Release"


"$(INTDIR)\cintocx.obj" : $(SOURCE) $(DEP_CPP_CINTO) "$(INTDIR)"\
 "$(INTDIR)\cintocx.pch"


!ELSEIF  "$(CFG)" == "cintocx - Win32 Debug"


"$(INTDIR)\cintocx.obj" : $(SOURCE) $(DEP_CPP_CINTO) "$(INTDIR)"\
 "$(INTDIR)\cintocx.pch"


!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Debug"


"$(INTDIR)\cintocx.obj" : $(SOURCE) $(DEP_CPP_CINTO) "$(INTDIR)"\
 "$(INTDIR)\cintocx.pch"


!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Release"


"$(INTDIR)\cintocx.obj" : $(SOURCE) $(DEP_CPP_CINTO) "$(INTDIR)"\
 "$(INTDIR)\cintocx.pch"


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\cintocx.def

!IF  "$(CFG)" == "cintocx - Win32 Release"

!ELSEIF  "$(CFG)" == "cintocx - Win32 Debug"

!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Debug"

!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Release"

!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\cintocx.rc

!IF  "$(CFG)" == "cintocx - Win32 Release"

DEP_RSC_CINTOC=\
	".\cintocx.ico"\
	".\CintocxCtl.bmp"\
	
NODEP_RSC_CINTOC=\
	".\Release\cintocx.tlb"\
	

"$(INTDIR)\cintocx.res" : $(SOURCE) $(DEP_RSC_CINTOC) "$(INTDIR)"\
 "$(INTDIR)\cintocx.tlb"
   $(RSC) /l 0x411 /fo"$(INTDIR)/cintocx.res" /i "Release" /d "NDEBUG" /d\
 "_AFXDLL" $(SOURCE)


!ELSEIF  "$(CFG)" == "cintocx - Win32 Debug"

DEP_RSC_CINTOC=\
	".\cintocx.ico"\
	".\CintocxCtl.bmp"\
	
NODEP_RSC_CINTOC=\
	".\Debug\cintocx.tlb"\
	

"$(INTDIR)\cintocx.res" : $(SOURCE) $(DEP_RSC_CINTOC) "$(INTDIR)"\
 "$(INTDIR)\cintocx.tlb"
   $(RSC) /l 0x411 /fo"$(INTDIR)/cintocx.res" /i "Debug" /d "_DEBUG" /d\
 "_AFXDLL" $(SOURCE)


!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Debug"

DEP_RSC_CINTOC=\
	".\cintocx.ico"\
	".\CintocxCtl.bmp"\
	
NODEP_RSC_CINTOC=\
	".\DebugU\cintocx.tlb"\
	

"$(INTDIR)\cintocx.res" : $(SOURCE) $(DEP_RSC_CINTOC) "$(INTDIR)"\
 "$(INTDIR)\cintocx.tlb"
   $(RSC) /l 0x411 /fo"$(INTDIR)/cintocx.res" /i "DebugU" /d "_DEBUG" /d\
 "_AFXDLL" $(SOURCE)


!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Release"

DEP_RSC_CINTOC=\
	".\cintocx.ico"\
	".\CintocxCtl.bmp"\
	
NODEP_RSC_CINTOC=\
	".\ReleaseU\cintocx.tlb"\
	

"$(INTDIR)\cintocx.res" : $(SOURCE) $(DEP_RSC_CINTOC) "$(INTDIR)"\
 "$(INTDIR)\cintocx.tlb"
   $(RSC) /l 0x411 /fo"$(INTDIR)/cintocx.res" /i "ReleaseU" /d "NDEBUG" /d\
 "_AFXDLL" $(SOURCE)


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\cintocx.odl

!IF  "$(CFG)" == "cintocx - Win32 Release"


"$(OUTDIR)\cintocx.tlb" : $(SOURCE) "$(OUTDIR)"
   $(MTL) /nologo /D "NDEBUG" /tlb "$(OUTDIR)/cintocx.tlb" /win32 $(SOURCE)


!ELSEIF  "$(CFG)" == "cintocx - Win32 Debug"


"$(OUTDIR)\cintocx.tlb" : $(SOURCE) "$(OUTDIR)"
   $(MTL) /nologo /D "_DEBUG" /tlb "$(OUTDIR)/cintocx.tlb" /win32 $(SOURCE)


!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Debug"


"$(OUTDIR)\cintocx.tlb" : $(SOURCE) "$(OUTDIR)"
   $(MTL) /nologo /D "_DEBUG" /tlb "$(OUTDIR)/cintocx.tlb" /win32 $(SOURCE)


!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Release"


"$(OUTDIR)\cintocx.tlb" : $(SOURCE) "$(OUTDIR)"
   $(MTL) /nologo /D "NDEBUG" /tlb "$(OUTDIR)/cintocx.tlb" /win32 $(SOURCE)


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\CintocxCtl.cpp
DEP_CPP_CINTOCX=\
	".\StdAfx.h"\
	".\cintocx.h"\
	".\CintocxCtl.h"\
	".\CintocxPpg.h"\
	"c:\cint\G__ci.h"\
	"..\..\src\sunos.h"\
	"..\..\src\newsos.h"\
	"..\..\src\memtest.h"\
	

!IF  "$(CFG)" == "cintocx - Win32 Release"


"$(INTDIR)\CintocxCtl.obj" : $(SOURCE) $(DEP_CPP_CINTOCX) "$(INTDIR)"\
 "$(INTDIR)\cintocx.pch"


!ELSEIF  "$(CFG)" == "cintocx - Win32 Debug"


"$(INTDIR)\CintocxCtl.obj" : $(SOURCE) $(DEP_CPP_CINTOCX) "$(INTDIR)"\
 "$(INTDIR)\cintocx.pch"


!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Debug"


"$(INTDIR)\CintocxCtl.obj" : $(SOURCE) $(DEP_CPP_CINTOCX) "$(INTDIR)"\
 "$(INTDIR)\cintocx.pch"


!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Release"


"$(INTDIR)\CintocxCtl.obj" : $(SOURCE) $(DEP_CPP_CINTOCX) "$(INTDIR)"\
 "$(INTDIR)\cintocx.pch"


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=.\CintocxPpg.cpp
DEP_CPP_CINTOCXP=\
	".\StdAfx.h"\
	".\cintocx.h"\
	".\CintocxPpg.h"\
	

!IF  "$(CFG)" == "cintocx - Win32 Release"


"$(INTDIR)\CintocxPpg.obj" : $(SOURCE) $(DEP_CPP_CINTOCXP) "$(INTDIR)"\
 "$(INTDIR)\cintocx.pch"


!ELSEIF  "$(CFG)" == "cintocx - Win32 Debug"


"$(INTDIR)\CintocxPpg.obj" : $(SOURCE) $(DEP_CPP_CINTOCXP) "$(INTDIR)"\
 "$(INTDIR)\cintocx.pch"


!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Debug"


"$(INTDIR)\CintocxPpg.obj" : $(SOURCE) $(DEP_CPP_CINTOCXP) "$(INTDIR)"\
 "$(INTDIR)\cintocx.pch"


!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Release"


"$(INTDIR)\CintocxPpg.obj" : $(SOURCE) $(DEP_CPP_CINTOCXP) "$(INTDIR)"\
 "$(INTDIR)\cintocx.pch"


!ENDIF 

# End Source File
################################################################################
# Begin Source File

SOURCE=\cint\libcint.lib

!IF  "$(CFG)" == "cintocx - Win32 Release"

!ELSEIF  "$(CFG)" == "cintocx - Win32 Debug"

!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Debug"

!ELSEIF  "$(CFG)" == "cintocx - Win32 Unicode Release"

!ENDIF 

# End Source File
# End Target
# End Project
################################################################################
