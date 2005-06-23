/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file bc_parse.h
 ************************************************************************
 * Description:
 *  block scope parser and compiler
 ************************************************************************
 * Copyright(c) 2004~2004  Masaharu Goto 
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/
#ifndef BC_ASSIGN_H
#define BC_ASSIGN_H

#if !defined(__sun) && (!defined(_MSC_VER) || _MSC_VER > 1200) && !(defined(__xlC__) || defined(__xlc__))
extern "C" {
#ifdef __CINT__
#include "../G__ci.h"
#else
#include "common.h"
#endif
}
#else
#include "G__ci.h"
#include "common.h"
#endif

#include "bc_parse.h"
#include "bc_reader.h"
#include "bc_inst.h"

////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////


#endif


