/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file bc_vtbl.h
 ************************************************************************
 * Description:
 *  virtual table generator
 ************************************************************************
 * Copyright(c) 2004~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef BC_VTBL_H
#define BC_VTBL_H

#include "G__ci.h"
#include "common.h"

#include "bc_cfunc.h"
#include <vector>
#include <utility>

namespace Cint {
   namespace Bytecode {

/***********************************************************************
* G__Vtabledata
***********************************************************************/
class G__Vtabledata {
  // function id
  Reflex::Member m_ifn; // the function

  // object offset
  int m_offset;
  
 public:
    G__Vtabledata(const Reflex::Member& ifn,int offset)
  : m_ifn(ifn), m_offset(offset) { }

  Reflex::Member GetFunction() const { return(m_ifn); }
  int GetOffset() const { return(m_offset); }
  //void SetIfunc(struct G__ifunc_table *ifunc) { m_ifunc=ifunc; }
  void SetIfn(const Reflex::Member& ifn) { m_ifn=ifn; }
  void SetOffset(int offset) { m_offset=offset; }

  void disp(FILE* fp) ;
};


/***********************************************************************
* G__Vtableoffset
***********************************************************************/
struct G__Vtbloffset {
   Reflex::Scope m_basetagnum;
   int m_vtbloffset;

   void disp(FILE* fp) ;
};

/***********************************************************************
* G__Vtable
***********************************************************************/
class G__Vtable {
 public:
  std::vector<G__Vtabledata> m_vtbl;
  std::vector<G__Vtbloffset> m_vtbloffset; 
  void addvfunc(const Reflex::Member& func,int offset) {
    m_vtbl.push_back(G__Vtabledata(func,offset));
  }
  int addbase(const Reflex::Scope& basetagnum,int vtbloffset) ;
  G__Vtabledata* resolve(int index,const Reflex::Scope& basetagnum) ;

  void disp(FILE* fp) ;
};

/***********************************************************************
* internal functions
***********************************************************************/
void G__bc_make_vtbl(const Reflex::Scope& tagnum) ;
void G__bc_make_defaultctor(const Reflex::Scope& tagnum) ;
void G__bc_make_assignopr(const Reflex::Scope& tagnum) ;
void G__bc_make_dtor(const Reflex::Scope& tagnum) ;

/***********************************************************************
* G__bc_struct() 
***********************************************************************/
void G__bc_struct(const Reflex::Scope& tagnum) ;

/***********************************************************************
* G__bc_delete_vtbl() 
***********************************************************************/
void G__bc_delete_vtbl(const Reflex::Scope& tagnum);

void G__bc_disp_vtbl(FILE* fp,const Reflex::Scope& tagnum);

   } // namespace Bytecode
} // namespace Cint

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
