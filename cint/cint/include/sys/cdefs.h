/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef __CINT_INTERNAL_CPP__

/* Using external C/C++ preprocessor with -p or +P option */
#if defined(__GNUC__) || defined(G__GNUC)
#include_next "sys/cdefs.h"
#else
#include "/usr/include/sys/cdefs.h"
#endif

#else /*  __CINT_INTERNAL_CPP__ */

/* Using Cint's internal preprocessor which has limitation */
/* nothing */

#include "platform.h"

#define __BEGIN_DECLS 
#define __END_DECLS 

#undef __attribute__
#define __attribute__(X) 

#endif

