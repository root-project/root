/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file bc_eh.h
 ************************************************************************
 * Description:
 *  Cint exception handler
 ************************************************************************
 * Copyright(c) 2004~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef BC_EH_H
#define BC_EH_H

#if !defined(__sun) && (!defined(_MSC_VER) || _MSC_VER > 1200) && !(defined(__xlC__) || defined(__xlc__))
//extern "C" {
#ifdef __CINT__
#include "../G__ci.h"
#else
#include "common.h"
#endif
//}
#else
#include "G__ci.h"
#include "common.h"
#endif

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
class G__bc_exception {
  G__value m_buf;
 public:
   G__bc_exception(G__value& buf) { m_buf=buf; }
   G__value& Get() { return(m_buf); }
   //void dealloc() ; 
};

//////////////////////////////////////////////////////////////////////



#endif

/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:2
 * c-continued-statement-offset:2
 * c-brace-offset:-2
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-2
 * compile-command:"make -k"
 * End:
 */
