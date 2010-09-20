/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/**************************************************************************
* cint (C++ Interpreter)
************************************************************************
* Source file testmain.c
************************************************************************
* Description:
*  Test version of G__main.c. With testmain.c, you can interpret
* cint source code by cint itself and it interprets another C program.
************************************************************************
* To execute
*  $ cint G__ci.c G__cfunc.c G__setup.c G__testmain.c <source file>
*
* For example,
*  $ cint -I$CINTSYSDIR -I$CINTSYSDIR/src +P G__testmain.c simple.c
*                                            ------------- -----------
*                                            this part is  this part is
*                                            interpreted   interpreted
*                                            by compiled   by interpreted
*                                            cint.         cint
*  
************************************************************************
*
* (C) Copyright 1991,1992,1993  Yokogawa Hewlett Packard HSTD R&D
* Author                        Masaharu Goto (gotom@tky.hp.com)
*
*  Refer to README file for conditions of using, copying and distributing
* CINT.
*
**************************************************************************/

#define G__TESTMAIN
#undef G__SHAREDLIB

#undef G__MEMTEST

#define G__POSIX_H

#include "src/v6_auxu.cxx"
#include "src/v6_cast.cxx"
#include "src/v6_debug.cxx"
#include "src/v6_decl.cxx"
#include "src/v6_disp.cxx"
#include "src/v6_dump.cxx"
#include "src/v6_end.cxx"
#include "src/v6_error.cxx"
#include "src/v6_expr.cxx"
#include "src/v6_fread.cxx"
#include "src/v6_func.cxx"
#include "src/v6_gcoll.cxx"
#include "src/v6_global1.cxx"
#include "src/v6_global2.cxx"
#include "src/g__cfunc.c"
#include "src/v6_ifunc.cxx"
#include "src/v6_inherit.cxx"
#include "src/v6_init.cxx"
#include "src/v6_input.cxx"
#include "src/v6_intrpt.cxx"
#include "src/v6_loadfile.cxx"
#include "src/v6_macro.cxx"
#include "src/v6_malloc.cxx"
/* #include "src/v6_memtest.cxx" */
#include "src/v6_new.cxx"
#include "src/v6_newlink.cxx"
#include "src/v6_opr.cxx"
#include "src/v6_parse.cxx"
#include "src/v6_pause.cxx"
#include "src/v6_pcode.cxx"
#include "src/v6_pragma.cxx"
#include "src/v6_quote.cxx"
#include "src/v6_scrupto.cxx"
#include "src/v6_shl.cxx"
#include "src/v6_sizeof.cxx"
#include "src/v6_struct.cxx"
#include "src/v6_stub.cxx"
#include "src/v6_sunos.cxx"
#include "src/v6_tmplt.cxx"
#include "src/v6_typedef.cxx"
#include "src/v6_val2a.cxx"
#include "src/v6_value.cxx"
#include "src/v6_var.cxx"
#include "src/v6_dmystrm.cxx"

#ifndef G__XREF
#include "src/Api.cxx"
#include "src/Class.cxx"
#include "src/BaseCls.cxx"
#include "src/CallFunc.cxx"
#include "src/DataMbr.cxx"
#include "src/Method.cxx"
#include "src/MethodAr.cxx"
#include "src/Token.cxx"
#include "src/Type.cxx"
#include "src/Typedf.cxx"
#endif

#include "main/G__setup.c"

extern short G__othermain;

main(argc,argv)
int argc;
char *argv[];
{
	G__othermain=0;
	return(G__main(argc,argv));
}


#ifdef G__NEVER
signal(signal,func)
int signal;
void *func;  // void (*func)();
{
}
#endif

alarm(int time) // dummy
{
}

void *rl_attempted_completion_function;
char *rl_basic_word_break_characters;


G__ASSERT(int x) // dummy
{
}

#ifndef G__CHECK
G__CHECK(x,y,z)
int x;
int y;
int z;
{
}
#endif
