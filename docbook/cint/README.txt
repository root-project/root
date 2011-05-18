cint 5.18.00         (CINT is pronounced "C-int")
     | |  |
     | |  +- Patch level (changed almost weekly at each release)
     | +- Minor version  (changed at DLL binary incompatibility)
     +- Major version    (major architecture change)

 Author                 Masaharu Goto
 Copyright(c) 1995~2010 Masaharu Goto (gotom@hanno.jp)
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
   NEC EWS4800, NewsOS, BeBox, Windows-NT ,Windows-9x, MS-DOS, MacOS,
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

 The source code package is available, 
 see http://root.cern.ch/twiki/bin/view/ROOT/CINT

CINT mailing list ==========================================================
   cint@root.cern.ch
     Send request to 'Majordomo@pcroot.cern.ch' containing following line for
     subscription.

       subscribe cint [preferred mail address]

 Archive for CINT mailing list can be accessed as follows.
   http://root.cern.ch/root/cinttalk/AboutCint.html


Installation: ==============================================================

 See http://root.cern.ch/twiki/bin/view/ROOT/CINT

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
			export LD_LIBRARY_PATH

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

  Stefan Roiser from CERN was the initial developer of the Reflex package.

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
 cint@root.cern.ch. You can also send message to 'rootdev@root.cern.ch' 
 or 'roottalk@root.cern.ch' for generic questions.

  Bugs are tend to be fixed very quickly. (Normally 1-2 days.) 
