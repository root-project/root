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

     v6_auxu.cxx
     v6_cast.cxx
     v6_debug.cxx
     v6_decl.cxx
     v6_disp.cxx
     v6_dump.cxx
     v6_end.cxx
     v6_error.cxx
     v6_expr.cxx
     v6_fread.cxx
     v6_func.cxx
     g__cfunc.cxx
     v6_gcoll.cxx
     v6_global1.cxx
     v6_global2.cxx
     v6_ifunc.cxx
     v6_inherit.cxx
     v6_init.cxx
     v6_input.cxx
     v6_intrpt.cxx
     v6_loadfile.cxx
     v6_macro.cxx
     v6_malloc.cxx
     v6_memtest.cxx
     v6_new.cxx
     v6_newlink.cxx
     v6_oldlink.cxx
     v6_opr.cxx
     v6_parse.cxx
     v6_pause.cxx
     v6_pcode.cxx
     v6_pragma.cxx
     v6_quote.cxx
     v6_scrupto.cxx
     v6_shl.cxx
     v6_sizeof.cxx
     v6_struct.cxx
     v6_stub.cxx
     v6_tmplt.cxx
     v6_typedef.cxx
     v6_val2a.cxx
     v6_value.cxx
     v6_var.cxx

  Files named after specific computer platform are only needed for 
 corresponding environment. 

     v6_macos.cxx
     v6_new.os.c
     v6_sunos.cxx
     v6_winnt.cxx

# Revision history
  HISTORY file contains source code modification history. Macro
 G__OLDIMPLEMENTATION??? are used to track modification history within
 single set of source files.

     HISTORY

# C++ iostream library interface method
  Following source files include C++ iostream library interface method.
 In most cases, libstrm.cxx works, however, special version is needed
 for Visual C++ and Borland C++. If non of following files work, use 
 fakestrm.cxx.  v6_dmystrm.cxx is provided for C-compiler-only installation.
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
     v6_dmystrm.cxx       # Dummy for C compiler

# ANSI C standard library struct interface method
  v6_stdstrct.cxx is needed to build cint. It contains inferface method for
 defining struct.

     v6_stdstrct.cxx
     v6_dmystrct.cxx

