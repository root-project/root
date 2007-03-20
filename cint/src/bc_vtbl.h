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

#include "bc_cfunc.h"
#include <vector>
#include <utility>
using namespace std;

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
  vector<G__Vtabledata> m_vtbl;
  vector<G__Vtbloffset> m_vtbloffset; 
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
extern "C" int G__function_signature_match(struct G__ifunc_table* ifunc1
					,int ifn1
					,struct G__ifunc_table* ifunc2
					,int ifn2
					,int mask
					,int matchmode) ;

/***********************************************************************
* G__bc_struct() 
***********************************************************************/
extern "C" void G__bc_struct(int tagnum) ;

/***********************************************************************
* G__bc_delete_vtbl() 
***********************************************************************/
extern "C" void G__bc_delete_vtbl(int tagnum);

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
