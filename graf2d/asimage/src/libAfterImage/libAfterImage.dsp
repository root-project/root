# Microsoft Developer Studio Project File - Name="libAfterImage" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Static Library" 0x0104

CFG=libAfterImage - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "libAfterImage.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "libAfterImage.mak" CFG="libAfterImage - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "libAfterImage - Win32 Release" (based on "Win32 (x86) Static Library")
!MESSAGE "libAfterImage - Win32 Debug" (based on "Win32 (x86) Static Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "libAfterImage - Win32 Release"

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
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /YX /FD /c
# ADD CPP /nologo /MD /W3 /GX /O2 /D "WIN32" /D "NO_DEBUG_OUTPUT" /D "NDEBUG" /D "_LIB" /YX /FD /c
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo /out:"libAfterImage.lib"

!ELSEIF  "$(CFG)" == "libAfterImage - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir ""
# PROP Intermediate_Dir "win32\Debug"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /YX /FD /GZ /c
# ADD CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /FD /GZ /c
# SUBTRACT CPP /YX
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo

!ENDIF 

# Begin Target

# Name "libAfterImage - Win32 Release"
# Name "libAfterImage - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Group "PNG Files"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\libpng\png.c
# End Source File
# Begin Source File

SOURCE=.\libpng\pngerror.c
# End Source File
# Begin Source File

SOURCE=.\libpng\pngget.c
# End Source File
# Begin Source File

SOURCE=.\libpng\pngmem.c
# End Source File
# Begin Source File

SOURCE=.\libpng\pngpread.c
# End Source File
# Begin Source File

SOURCE=.\libpng\pngread.c
# End Source File
# Begin Source File

SOURCE=.\libpng\pngrio.c
# End Source File
# Begin Source File

SOURCE=.\libpng\pngrtran.c
# End Source File
# Begin Source File

SOURCE=.\libpng\pngrutil.c
# End Source File
# Begin Source File

SOURCE=.\libpng\pngset.c
# End Source File
# Begin Source File

SOURCE=.\libpng\pngtrans.c
# End Source File
# Begin Source File

SOURCE=.\libpng\pngwio.c
# End Source File
# Begin Source File

SOURCE=.\libpng\pngwrite.c
# End Source File
# Begin Source File

SOURCE=.\libpng\pngwtran.c
# End Source File
# Begin Source File

SOURCE=.\libpng\pngwutil.c
# End Source File
# End Group
# Begin Group "JPG Files"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\libjpeg\jcapimin.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jcapistd.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jccoefct.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jccolor.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jcdctmgr.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jchuff.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jcinit.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jcmainct.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jcmarker.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jcmaster.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jcomapi.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jcparam.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jcphuff.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jcprepct.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jcsample.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jctrans.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jdapimin.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jdapistd.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jdatadst.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jdatasrc.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jdcoefct.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jdcolor.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\transupp.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jaricom.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jdarith.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jcarith.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jddctmgr.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jdhuff.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jdinput.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jdmainct.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jdmarker.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jdmaster.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jdmerge.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jdpostct.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jdsample.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jdtrans.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jerror.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jfdctflt.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jfdctfst.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jfdctint.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jidctflt.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jidctfst.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jidctint.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jmemmgr.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jmemnobs.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jquant1.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jquant2.c
# End Source File
# Begin Source File

SOURCE=.\libjpeg\jutils.c
# End Source File
# End Group
# Begin Group "ZLIB Files"

# PROP Default_Filter ""
# Begin Source File

SOURCE=.\zlib\adler32.c
# End Source File
# Begin Source File

SOURCE=.\zlib\compress.c
# End Source File
# Begin Source File

SOURCE=.\zlib\crc32.c
# End Source File
# Begin Source File

SOURCE=.\zlib\deflate.c
# End Source File
# Begin Source File

SOURCE=.\zlib\gzio.c
# End Source File
# Begin Source File

SOURCE=.\zlib\infback.c
# End Source File
# Begin Source File

SOURCE=.\zlib\inffast.c
# End Source File
# Begin Source File

SOURCE=.\zlib\inflate.c
# End Source File
# Begin Source File

SOURCE=.\zlib\inftrees.c
# End Source File
# Begin Source File

SOURCE=.\zlib\trees.c
# End Source File
# Begin Source File

SOURCE=.\zlib\uncompr.c
# End Source File
# Begin Source File

SOURCE=.\zlib\zutil.c
# End Source File
# End Group
# Begin Group "UNGIF Files"

# PROP Default_Filter "*.c"
# Begin Source File

SOURCE=.\libungif\dgif_lib.c
# End Source File
# Begin Source File

SOURCE=.\libungif\egif_lib.c
# End Source File
# Begin Source File

SOURCE=.\libungif\gif_err.c
# End Source File
# Begin Source File

SOURCE=.\libungif\gifalloc.c
# End Source File
# End Group
# Begin Source File

SOURCE=.\afterbase.c
# End Source File
# Begin Source File

SOURCE=.\ascmap.c
# End Source File
# Begin Source File

SOURCE=.\asfont.c
# End Source File
# Begin Source File

SOURCE=.\asimage.c
# End Source File
# Begin Source File

SOURCE=.\asimagexml.c
# End Source File
# Begin Source File

SOURCE=.\asstorage.c
# End Source File
# Begin Source File

SOURCE=.\asvisual.c
# End Source File
# Begin Source File

SOURCE=.\blender.c
# End Source File
# Begin Source File

SOURCE=.\bmp.c
# End Source File
# Begin Source File

SOURCE=.\char2uni.c
# End Source File
# Begin Source File

SOURCE=.\export.c
# End Source File
# Begin Source File

SOURCE=.\imencdec.c
# End Source File
# Begin Source File

SOURCE=.\import.c
# End Source File
# Begin Source File

SOURCE=.\pixmap.c
# End Source File
# Begin Source File

SOURCE=.\transform.c
# End Source File
# Begin Source File

SOURCE=.\ungif.c
# End Source File
# Begin Source File

SOURCE=.\xcf.c
# End Source File
# Begin Source File

SOURCE=.\ximage.c
# End Source File
# Begin Source File

SOURCE=.\xpm.c
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\win32\afterbase.h
# End Source File
# Begin Source File

SOURCE=.\afterimage.h
# End Source File
# Begin Source File

SOURCE=.\ascmap.h
# End Source File
# Begin Source File

SOURCE=.\asfont.h
# End Source File
# Begin Source File

SOURCE=.\asim_afterbase.h
# End Source File
# Begin Source File

SOURCE=.\asimage.h
# End Source File
# Begin Source File

SOURCE=.\asimagexml.h
# End Source File
# Begin Source File

SOURCE=.\asvisual.h
# End Source File
# Begin Source File

SOURCE=.\blender.h
# End Source File
# Begin Source File

SOURCE=.\bmp.h
# End Source File
# Begin Source File

SOURCE=.\char2uni.h
# End Source File
# Begin Source File

SOURCE=.\win32\config.h
# End Source File
# Begin Source File

SOURCE=.\export.h
# End Source File
# Begin Source File

SOURCE=.\imencdec.h
# End Source File
# Begin Source File

SOURCE=.\import.h
# End Source File
# Begin Source File

SOURCE=.\pixmap.h
# End Source File
# Begin Source File

SOURCE=.\transform.h
# End Source File
# Begin Source File

SOURCE=.\ungif.h
# End Source File
# Begin Source File

SOURCE=.\xcf.h
# End Source File
# Begin Source File

SOURCE=.\ximage.h
# End Source File
# Begin Source File

SOURCE=.\xpm.h
# End Source File
# Begin Source File

SOURCE=.\xwrap.h
# End Source File
# End Group
# End Target
# End Project
