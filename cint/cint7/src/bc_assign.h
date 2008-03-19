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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef BC_ASSIGN_H
#define BC_ASSIGN_H

#include "G__ci.h"

struct G__var_array;
namespace Reflex {
   class Member;
}

////////////////////////////////////////////////////////////////////
namespace Cint {
   namespace Internal {
int G__bc_objassignment (G__value *plresult ,G__value *prresult);
int G__bc_assignment(const Reflex::Member& var,int lparan
                                ,int lvar_type,G__value *prresult
                                ,long struct_offset,long store_struct_offset
                                ,G__value *ppara);

////////////////////////////////////////////////////////////////////
   } // namespace Internal
} // namespace Cint

#endif


