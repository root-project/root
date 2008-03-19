/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file typeif.h
 ************************************************************************
 * Description:
 *  C/C++ type abstraction
 ************************************************************************
 * Copyright(c) 2004~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef BC_TYPE_H
#define BC_TYPE_H

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "G__ci.h"
#include "common.h"
#ifdef G__DEBUG
#undef G__MEMTEST_H
#undef G__MEMTEST
#include "memtest.h"
#endif


#include "Api.h"
#include "Reflex/Builder/TypeBuilder.h"
#include <string>

namespace Cint{
   namespace Bytecode {

template<class T> T bc_min(T a,T b) { return((a<b)?a:b); }
template<class T> T bc_max(T a,T b) { return((a>b)?a:b); }

/*************************************************************************
 * class G__TypeReader
 *************************************************************************/
// FIXME: G__TypeReader should have a G__TypeInfo data member instead of using inheritence.
class G__TypeReader : public G__TypeInfo {
  int m_static;
  int m_type;  // just for error check

 public:
  G__TypeReader(const char *typenamein) : G__TypeInfo(typenamein) 
     { m_type=0; m_static=0; }
  G__TypeReader() : G__TypeInfo() { clear(); }
  G__TypeReader(G__value& buf) : G__TypeInfo(buf) { m_type=0;m_static=0; }

  void clear();

  int append(const std::string& token,int c);

 public:
  void append_static() { m_static=1; }
  void append_const() { typenum = Reflex::ConstBuilder(typenum); }
 private:
  void append_unsigned() ;
  void append_long() ;
  void append_int() ;
  void append_short() ;
  void append_char() ;
  void append_double() ;
  void append_float() ;
  void append_void() ;
  void append_FILE() ;
  void append_bool() ;

 public:
  void incplevel() ;
  void decplevel() ;
  void increflevel() ;
  void decreflevel() ;

  // not needed anymore
  // void setisconst(int isconstin) { isconst=isconstin; }
#pragma message(FIXME("Is setreftype really needed? Can we at least use Reflex API?"))
  void setreftype(int reftypein) { /*reftype=reftypein;*/ }
  void setstatic(int isstatic) { m_static=isstatic; }

  void nextdecl() ;

 public:
  long Property();
  int Isfundamental() { return(Property()&G__BIT_ISFUNDAMENTAL); }
  int Isstatic() const { return(m_static); }
  int Ispointer() const;
  int Isreference() const;

  void Init(G__value& x);
  void Init(G__TypeInfo& x);
  void Init(const Reflex::Member& var) {
     G__TypeInfo::Init((struct G__var_array*)var.Id(), -2);
  }
  G__value GetValue() const; 

};

   } // namespace Bytecode
} // namespace Cint

#endif
