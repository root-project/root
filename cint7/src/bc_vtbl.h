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
  struct G__ifunc_table* m_ifunc;
  int m_ifn;

  // object offset
  int m_offset;
  
 public:
  G__Vtabledata(struct G__ifunc_table* ifunc,int ifn,int offset)
  : m_ifunc(ifunc) , m_ifn(ifn), m_offset(offset) { }

  struct G__ifunc_table* GetIfunc() const { return(m_ifunc); }
  int GetIfn() const { return(m_ifn); }
  int GetOffset() const { return(m_offset); }
  void SetIfunc(struct G__ifunc_table *ifunc) { m_ifunc=ifunc; }
  void SetIfn(int ifn) { m_ifn=ifn; }
  void SetOffset(int offset) { m_offset=offset; }

  void disp(FILE* fp) ;
};


/***********************************************************************
* G__Vtableoffset
***********************************************************************/
struct G__Vtbloffset {
  short m_basetagnum;
  short m_vtbloffset;

  void disp(FILE* fp) ;
};

/***********************************************************************
* G__Vtable
***********************************************************************/
class G__Vtable {
 public:
  std::vector<G__Vtabledata> m_vtbl;
  std::vector<G__Vtbloffset> m_vtbloffset; 
  void addvfunc(G__ifunc_table* ifunc,int ifn,int offset) {
    m_vtbl.push_back(G__Vtabledata(ifunc,ifn,offset));
  }
  int addbase(int basetagnum,int vtbloffset) ;
  G__Vtabledata* resolve(int index,int basetagnum) ;

  void disp(FILE* fp) ;
};

/***********************************************************************
* internal functions
***********************************************************************/
void G__bc_make_vtbl(int tagnum) ;
void G__bc_make_defaultctor(int tagnum) ;
void G__bc_make_assignopr(int tagnum) ;
void G__bc_make_dtor(int tagnum) ;

/***********************************************************************
* G__function_signature_match()
***********************************************************************/
int G__function_signature_match(struct G__ifunc_table* ifunc1
					,int ifn1
					,struct G__ifunc_table* ifunc2
					,int ifn2
					,int mask
					,int matchmode) ;

/***********************************************************************
* G__bc_struct() 
***********************************************************************/
void G__bc_struct(int tagnum) ;

/***********************************************************************
* G__bc_delete_vtbl() 
***********************************************************************/
void G__bc_delete_vtbl(int tagnum);

void G__bc_disp_vtbl(FILE* fp,int tagnum);

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
