src/README.txt

# Cint 6.x Major re-engineering

 Re-engineering source file names start with "bc_"

    bc_autoobj.cxx/.h
    bc_cfunc.cxx/.h
    bc_inst.cxx/.h
    bc_parse.cxx/.h
    bc_reader.cxx/.h
    bc_type.cxx/.h

    bc_dict.cxx/.h  : Legacy Cint dictionary for debugging purpose


NOTE: 
  PLEASE DO NOT TAKE CINT SOURCE CODE AS PROGRAMMING EXAMPLE OR REFERENCEE
 OF AN IMTERPRETER IMPLEMENTATION. CINT IS A CHUNK OF LEGACY CODE WHICH IS 
 SUBJECT TO REENGINEERING. IT IS SAFER TO USE CINT ONLY FROM OUTSIDE.


 This directory contains CINT - C++ interpreter core source code.
Source files are categorized as follows.

# ERTTI API source files
  ERTTI (Extensive Run Time Type Identification) API provides means of 
 inspecting symbol table and class structure of interpreted source code.
 Cint symbol table is encapsureated in 9 C++ classes.  Refer to following
 source code and corresponding header files for ERTTI specs. Also, refer
 to doc/ref.txt.

     Api.cxx
     BaseCls.cxx
     CallFunc.cxx
     Class.cxx
     DataMbr.cxx
     Method.cxx
     MethodAr.cxx
     Token.cxx
     Type.cxx
     Typedf.cxx

  Apiif.cxx contains interface method for ERTTI APIs. This file is generated
 by 'cint -c-1'.

     Apiif.cxx 


# CINT - C++ interpreter core
  Following source files are the core of the interpreter. Those files are
 written in K&R style C for portability reason.  
  Please do not regard those source files as programming example or 
 implementation reference. I started to write Cint when I wasn't 
 knowledgeble to modern programming methodology. 
  All of the global names start with 'G__' to avoid name conflict. Macro
 G__OLDIMPLEMENTATION??? are used for revision control. G__OLDIMPLEMENTATION???
 are usually not defined.

     auxu.cxx
     cast.cxx
     debug.cxx
     decl.cxx
     disp.cxx
     dump.cxx
     end.cxx
     error.cxx
     expr.cxx
     fread.cxx
     func.cxx
     g__cfunc.cxx
     gcoll.cxx
     global1.cxx
     global2.cxx
     ifunc.cxx
     inherit.cxx
     init.cxx
     input.cxx
     intrpt.cxx
     loadfile.cxx
     macro.cxx
     malloc.cxx
     memtest.cxx
     new.cxx
     newlink.cxx
     oldlink.cxx
     opr.cxx
     parse.cxx
     pause.cxx
     pcode.cxx
     pragma.cxx
     quote.cxx
     scrupto.cxx
     shl.cxx
     sizeof.cxx
     struct.cxx
     stub.cxx
     tmplt.cxx
     typedef.cxx
     val2a.cxx
     value.cxx
     var.cxx

  Files named after specific computer platform are only needed for 
 corresponding environment. 

     macos.cxx
     new.os.c
     sunos.cxx
     winnt.cxx

# Revision history
  HISTORY file contains source code modification history. Macro
 G__OLDIMPLEMENTATION??? are used to track modification history within
 single set of source files.

     HISTORY

# C++ iostream library interface method
  Following source files include C++ iostream library interface method.
 In most cases, libstrm.cxx works, however, special version is needed
 for Visual C++ and Borland C++. If non of following files work, use 
 fakestrm.cxx.  dmystrm.cxx is provided for C-compiler-only installation.
 Refer platform/README.txt for more detail.

     libstrm.cxx     # HP-UX, Linux, Solaris, AIX, IRIX, gcc 2.x, etc,...
     gcc3strm.cxx    # gcc 3.x
     sun5strm.cxx    # Solaris, Sun C++ Compiler 5.x
     vcstrm.cxx      # Visual C++ 4.0/5.0/6.0
     cbstrm.cpp      # Borland C++ Builder 3.0, Borland C++ compiler 5.5
     bcstrm.cxx      # Borland C++ (older version)
     kccstrm.cxx     # KAI C++ compiler
     iccstrm.cxx     # Intel C++ compiler
     fakestrm.cxx    # Dummy for C++ compiler
     dmystrm.cxx       # Dummy for C compiler

# ANSI C standard library struct interface method
  stdstrct.cxx is needed to build cint. It contains inferface method for
 defining struct.

     stdstrct.cxx
     dmystrct.cxx

