cint 5.16.16 / 6.1.16         ( CINT is pronounced "C-int" )
     | |  |
     | |  +- Patch level (changed almost weekly at each release)
     | +- Minor version  (changed at DLL binary incompatibility)
     +- Major version    (major architecture change)

 Author                 Masaharu Goto
 Copyright(c) 1995~2005 Masaharu Goto (gotom@hanno.jp)
 Mailing list           cint@root.cern.ch


Note: Search 'Installation' for installation instruction.


====== CINT ABSTRACT =======================================================

 "cint" is a C/C++ interpreter which has following features.

 * Support K&R-C, ANSI-C, ANSI-C++
    Cint has 80-90% coverage on K&R-C, ANSI-C and C++ language constructs. 
   (Multiple inheritance, virtual function, function overloading, operator 
   overloading, default parameter, template, etc..)  Cint is solid enough 
   to interpret its own source code. 
    Cint is not aimed to be a 100% ANSI/ISO compliant C++ language processor.
   It rather is a portable script language environment which is close enough 
   to the standard C++.

 * Handling Huge C/C++ source code
    Cint can handle huge C/C++ source code. This has been a problem for other
   C++ interpreter. Cint is quick in loading source files. Cint can interpret
   its own over 60,000 lines source code.

 * Interpretation & Native Code Execution can be mixed
    Depending on speed and interactiveness requirement, you can mix Native
   Code execution and interpretation. "makecint" makes it possible to 
   encapsulate arbitrary C/C++ object as a precompiled library. Precompiled
   library can be configured as a Dynamic Link Library. Access between
   interpreted code and precompiled code can be done seamlessly in both
   direction. 

 * Single-Language solution
    Cint/makecint is a Single-Language environment. It works with any 
   ANSI-C/C++ compiler to provide the interpreter environment on top of it.

 * Bridge between serious programmers and other professionals
    Cint is meant to be a bridging tool between software and non-software
   professionals. C++ looks rather easy under the interpreter environment.
   It helps non-software professionals to talk in the same language to
   their software counterpart. Today's System-On-Silicon evolution demands
   integration and standardization of design tools in software, hardware, IC
   and system design processes. Cint is a key enabling technology to this
   critical issue.

 * Dynamic C++
    Cint is dynamic. It can process C++ statements from command line,
   dynamically define/erase class definition and functions, load/unload 
   source files and Dynamic Link Library.  Extended Run Time Type 
   Identification mechanism is provided. This will allow you to explore
   unthinkable way of using C++.

 * Built-in Debugger, class browser
    Cint has a built-in debugger with an extensive capability to debug
   complex C++ execution flow. Text base class browser is a part of the 
   debugger capability.

 * Portability
    CINT works on number of Operating Systems.
   HP-UX, Linux, SunOS, Solaris, AIX, Alpha-OSF, IRIX, FreeBSD, NetBSD, 
   NEC EWS4800, NewsOS, BeBox, Windows-NT ,Windows-9x, MS-DOS, MacOS, VMS,
   NextStep, Convex. Porting should be easy. Refer to platform/README. 
   What about OS2, VxWorks, etc...?

 * CINT users spread world wide
    Many people world-wide are using Cint. Many C/C++ libraries have 
   been encapsulated. CERN and Fermi-Lab choose Cint as front-end command 
   processor and script interpreter for "ROOT" Object Oriented Software 
   Framework. The ROOT/CINT framework will be used in Large Hadron 
   Collider(LHC) research project beyond 2020.

 * More applications
   We have done, so far, following integration.
   ROOT/CINT framework: Next generation C++ Object Oriented Framework
   WildC++ interpreter: CINT + Tcl/Tk 
   CINTOCX            : CINT + VisualBasic

   There are unlimited opportunity of CINT integration. To list up a few,
   Cint3D             : CINT, openGL, DirectModel, VRML integration for 3D
   VeriCint           : CINT + Verilog-XL simulator connected by PLI and TCP/IP
   MathCint           : CINT + Math library + Digital Filter Design tool, etc
   CintSQL            : CINT + Database connection
   CintWin32          : CINT + Win32 API
   Your contribution will be greatly appreciated.


License: ===================================================================

 See file COPYING in this directory for the licensing terms.

Getting the latest package =================================================

 The latest source code package is available from following sources.

 Static Version : from Cint web site
   http://root.cern.ch/root/Cint.html

 Aggressive Version
   ftp://root.cern.ch/root/cintX.X.tar.gz 
                                (X.X means whatever the latest revision number)

 CINT mailing list
   cint@root.cern.ch
     Send request to 'Majordomo@pcroot.cern.ch' containing following line for
     subscription.

       subscribe cint [preferred mail address]

 Archive for CINT mailing list can be accessed as follows.
   http://root.cern.ch/root/cinttalk/AboutCint.html


Windows-NT/95/98/Me/2000/XP Installation: ===================================

 - Getting source package

  Source package is distributed as cint5.14.tar.gz from the Cint web page.
  Please download this file.


 - Unpack the package

  You must install CINT in c:\cint directory. Make c:\cint directory, copy 
  cint source package as cint.tgz and unpack it by gzip+tar.

      c:\> mkdir c:\cint
      c:\> cd c:\cint
      c:\cint> copy [where_you_saved_package]\cint5.14.tar.gz cint.tgz
      c:\cint> gzip -vd cint.tgz
      c:\cint> tar xvf cint.tar

  Alternatively, You can unpack the package by using Winzip twice. Assuming
  Winzip is already installed in your machine,

      c:\> mkdir c:\cint
      c:\> cd c:\cint
      c:\cint> copy [where_you_saved_package]\cint5.14.tar.gz cint.tgz
      ### click cint.tgz   -> cint.tar 
      ### click cint.tar   -> unpacking cint source files

  For Windows, you can install Cint by  (a) using a binary distribution
  or  (b-f) compiling the source. Choose one of following options.


 a) Using binary distribution

  Binary package is distributed as cintwin.tar.gz from the Cint web page.
  Please download this file.  Note that cintwin.ta.gz binary package does 
  not work alone. It must be always installed on top of the source. Do
  following after unpacking the source package.

  You must install CINT in c:\cint directory. Copy cint binary package 
  as cintwin.tgz and unpack it by gzip+tar.

      c:\> cd c:\cint
      c:\cint> copy [where_you_saved_package]\cintwin.tar.gz cintwin.tgz
      c:\cint> gzip -vd cintwin.tgz
      c:\cint> tar xvf cintwin.tar

  Alternatively, You can unpack the package by using Winzip twice. Assuming
  Winzip is already installed in your machine,

      c:\> mkdir c:\cint
      c:\> cd c:\cint
      c:\cint> copy [where_you_saved_package]\cintwin.tar.gz cintwin.tgz
      ### click cintwin.tgz   -> cintwin.tar 
      ### click cintwin.tar   -> unpacking cint binary files

  You need to set environment variables manually.  Add following lines
  in c:\autoexec.bat and reboot.

      set CINTSYSDIR=c:\cint
      set PATH=%PATH%;%CINTSYSDIR%

  Limitation: 
    With the binary distribution, you can only use cint C++ interpreter.
    In order to use makecint, you must compile cint from source. See
    b), c), d).

 b) Using Visual C++ 4.0, 5.0, 6.0   (See b' for VC++ 7.0)
   If you use Visual C++ 5.0 or later , you need to run vcvars32.bat
   to set VC++ environment variables. vcvars32.bat exists in Visual 
   C++ binary directory. Before running this, please make sure that 
   your MSDOS Command Window has sufficient environment variable space. 
   Edit MSDOS Command Window property and increase environment variable 
   space. You may also need to comment out following line from vcvars32.bat.

        rem set vcsource=L:\DEVSTUDIO

   Then run the script

        ### Change directory to either of following directory, depending on
        ### version of the compiler.
        ###     "\Program Files\DevStudio\VC\bin"
        ###     "\Program Files\Microsoft Visual Studio\VC98\bin"
        ### Then
        c:\Program Files\Microsoft Visual Studio\VC98\bin> vcvars32.bat

   You do not need to do above if you use Visual C++ 4.0. 
 
   Go to c:\cint\platform\visual , c:\cint\platform\visualCpp6 or
   c:\cint\platform\visualCpp7 directory according to version of compiler
   you have. And run setup.bat. This script updates c:\autoexec.bat. 
   Reboot your PC after installation. 
   Read c:\cint\platform\visualXXX\readme.txt if you find problems.

        c:\cint> cd c:\cint\platform\visual            : VC++4.0/5.0/6.0
                          OR
        c:\cint> cd c:\cint\platform\visualCpp6        : VC++6.0
                          OR
        c:\cint> cd c:\cint\platform\visualCpp7        : VC++7.0

   Depending on operation system and compiler version, run following
   setup script.

     Windows-XP , VC++ 7.0
        c:\cint\platform\visualCpp7> setup.bat

     Windows-95/98/Me , VC++ 6.0
        c:\cint\platform\visual> setup.bat

     Windows-2000 , VC++ 6.0
      It looks like Windows-2000 has problem setting environment variablle in
      batch script. Set environment variable CINTSYSDIR=c:\cint and 
      PATH=%CINTSYSDIR%;%PATH% first. Then run following command.
        c:\cint\platform\visual> setup2000.bat

     Windows-NT , VC++ 6.0
        c:\cint\platform\visualCpp6> setup.bat

     Windows-95/98/2000/Me , VC++ 5.0 or older
        c:\cint\platform\visual> setupold.bat

     Windows-NT , VC++ 5.0 or older
        c:\cint\platform\visual> setupNTold.bat

   Then reboot system.
        ### REBOOT SYSTEM ###


 c) Using Borland C++ Compiler version 5.5
   Free version of Borland BCC32.exe comiler is available from 
   http://www.borland.com.  BCC32.exe version 5.5 is supported from
   cint5.15.53.  Go to c:\cint\platform\borlandcc5 directory and run
   setup.bat.  This script updates c:\autoexec.bat. Reboot your PC after 
   installation.  Read c:\cint\platform\borlandcc5\README.txt if you find
   problems.

        c:\cint> cd c:\cint\platform\borlandcc5
        c:\cint\platform\borlandcc5> setup.bat
        ### REBOOT SYSTEM ###

 d) Using Borland C++ Builder 3.0
   Go to c:\cint\platform\borland directory and run setup.bat. This script
   updates c:\autoexec.bat. Reboot your PC after installation. 
   Read c:\cint\platform\borland\README.txt if you find problems.

        c:\cint> cd c:\cint\platform\borland
        c:\cint\platform\borland> setup.bat
        ### REBOOT SYSTEM ###

   Note: This process may not work under Borland C++ Builder 5.0. Please 
   refer to platform/README.txt


 e) Using Symantec C++ 7.2
   Go to c:\cint\platform\symantec directory and run setup.bat. This script
   updates c:\autoexec.bat. Reboot your PC after installation.
   Read c:\cint\platform\symantec\README.txt if you find problems.

        c:\cint> cd c:\cint\platform\symantec
        c:\cint\platform\symantec> setup.bat
        ### REBOOT SYSTEM ###


 f) Using DJGPP
   For compilation using DGJPP, read  platform/README.txt


Note:
  About 30Mbyte free disc space is needed for compiling Cint core system. 
  Setup script continues to compile optional library components which
  require ~90Mbyte free disc space.


UNIX/Linux/Cygwin Installation: ============================================

 - Getting source package

   Source package is distributed as cint5.14.tar.gz from the Cint web page.
   Please download this file.


 - Unpack the package

   At any event of failing installation please refer to platform/README.
   You need to get either of 'cint5.14.tar.gz', 'cint.tar.gz' or 'cint.tgz'. 
   The file includes all of the Cint related source files.
   Make a directory:(for example /usr/local/cint)

	$ mkdir /usr/local/cint
	$ cd /usr/local/cint
        $ cp [where_you_saved_package]/cint5.14.tar.gz cint.tar.gz

   Copy the archive file to that directory and do the following to unpack 
   the package.

	$ gunzip -c cint.tar.gz | tar xvf -
	       or
        $ gzip -vd cint.tar.gz
        $ tar xvf cint.tar

   At this moment, C++ source files have .cxx file name extension. If you
   have a C++ compiler which does not accept .cxx extension, cxx2C script
   will rename them. This is an optional procedure. If you run cxx2C by
   mistake, you can recover by using C2cxx script.

        $ sh ./cxx2C                    (optional)


 - Compile Cint and optional library

   Then, look into platform directory to find an appropriate platform 
   dependency file for your environment. You need to give that file as an
   argument to setup script.

	$ sh ./setup platform/[machine]

   If you can not find any file that is close to your system, you need to
   create one by yourself. If you have gcc/g++, platform/gcc_min or gcc_max
   may be good.  There can be platform dependent restriction. Please read 
   platform/README file carefully.


Note1:
  About 30Mbyte free disc space is needed for compiling Cint core system. 
  Setup script continues to compile optional library components which
  require ~90Mbyte free disc space.

Note2: Linux installation requirements:
  When installing Linux package, you need to add following packages in
  order to build Cint. Basically, you need to install developper environment.
	binutils (d),  gcc (d), gcc-g++ (d), make (d), kernel-headers (d),
	readline (l), ncurses (l), glibc (l)


Setting up environment variable:(UNIX) ======================================

   After installation you need to set environment variables. If you installed 
   cint under '/usr/local/cint', for example

	ksh/bsh/bash 	CINTSYSDIR=/usr/local/cint
			PATH=$PATH:$CINTSYSDIR
			MANPATH=$MANPATH:$CINTSYSDIR/doc
			export PATH CINTSYSDIR MANPATH

	csh		setenv CINTSYSDIR /usr/local/cint
			setenv PATH ($PATH $CINTSYSDIR)
			setenv MANPATH ($MANPATH $CINTSYSDIR/doc)

   You may need to set following variable too

	bash		LD_LIBRARY_PATH=.:$CINTSYSDIR:$LD_LIBRARY_PATH
			LD_ELF_LIBRARY_PATH=$LD_LIBRARY_PATH
			export LD_LIBRARY_PATH LD_ELF_LIBRARY_PATH

   Now you can use 'cint'(interpreter itself) and 'makecint'(compile utility).

	$ cint <options> <[file1].c [file2].c [shl].sl ...> [main].c
	$ makecint <-o|-dl> <[obj]|[shl].sl> -H [src].h -C++ [src].C 


Caution ====================================================================

   Cint creates temp-files in /tmp, /usr/tmp, \temp or windows\temp
   directory. Those files are automatically removed in normal situation but
   sometimes left unremoved by accident. Check /tmp and /usr/tmp regulary 
   to remove garbage.  Temp-files are named *_cint.

Documentation ==============================================================

	$CINTSYSDIR/doc/cint.txt     # cint manual
	$CINTSYSDIR/doc/makecint.txt # makecint manual
	$CINTSYSDIR/doc/limitati.txt # cint syntax limitation
	$CINTSYSDIR/doc/limitnum.txt # cint quantitive limitation
	$CINTSYSDIR/doc/ref.txt      # cint special feature, ERTTI API
	$CINTSYSDIR/doc/cintapi.txt  # cint API
	$CINTSYSDIR/doc/bytecode.txt # cint incremental bytecode compiler


Demo =======================================================================

    You can find demonstration programs in $CINTSYSDIR/demo/ directory.
   These programs can be referred as programming examples. Please read README
   file under there. Makecint examples are found in $CINTSYSDIR/demo/makecint
   directory.

    It is highly recommended to start with those demo programs. 


Contributers ==============================================================

  The Author would like to thank Fons Rademakers and Rene Brun in CERN
 for jointly develop ping ROOT-CINT framework system. Their invaluable advice 
 has been improving cint. They ported CINT on 10 different UNIX platforms.

  Philippe Canal in Fermi-Lab contributes to many enhancements and bug fixes.

  Axel Naumann from CERN contributes to the build system and the update
to use the Reflex in memory database and many other contributions.

  Stefan Roiser from CERN is the main developer of the Reflex package.

  Scott Snyder provided the initial implementaton of template support.

  Osamu Kotanigawa is giving great advice on Windows-NT Visual C++ and 
 Borland C++ porting. He created src/vcstrm.cxx and src/bcstrm.cxx. His
 contribution improved CINT in many aspects.

  Jacek Holeczek refined DLL schemes on IBM AIX. He wrote platform/aix*
 and platform/aixMakefileBaseAdd.

  Dibyendu Majumdar in U.K. ported CINT on MS-DOS using DJGPP.

  Kiyoshi Yamamoto, in CQ publishing company, is giving me opportunity to 
 write a book and articles about CINT.

  Masa Habu in HP Cupertino suggested usefulness of C++ interpreter in 1992
 when CINT was only a C interpreter.

  Junichi Mizoguchi in Agilent Technologies Japan created a base concept 
 of makecint by his automated software test execution environment.

  And many thanks to people in Agilent Technologies Japan Hachioji-site.

The Author ===============================================================

 Masaharu Goto

  Please contact the author for bugs and requests by sending an e-mail to
 cint@pcroot.cern.ch. You can also send message to 'rootdev@pcroot.cern.ch' 
 or 'roottalk[pcroot.cern.ch' for generic questions.

  Bugs are tend to be fixed very quickly. (Normally 1-2 days if I wasn't 
 gone for vacation or business trip.) 
