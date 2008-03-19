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

#if !defined(__sun) && (!defined(_MSC_VER) || _MSC_VER > 1200) && !(defined(__xlC__) || defined(__xlc__))
//extern "C" {
#ifdef __CINT__
#include "../G__ci.h"
#else
#include "common.h"
#endif
#ifdef G__DEBUG
#undef G__MEMTEST_H
#undef G__MEMTEST
#include "memtest.h"
#endif
//}
#else
#include "G__ci.h"
#include "common.h"
#ifdef G__DEBUG
#undef G__MEMTEST_H
#undef G__MEMTEST
#include "memtest.h"
#endif
#endif


#include "Api.h"
#include <string>
using namespace std;

template<class T> T bc_min(T a,T b) { return((a<b)?a:b); }
template<class T> T bc_max(T a,T b) { return((a>b)?a:b); }

/*************************************************************************
 * class G__TypeReader
 *************************************************************************/
class G__TypeReader : public G__TypeInfo {
  int m_static;
  int m_type;  // just for error check

 public:
  G__TypeReader(const char *typenamein) : G__TypeInfo(typenamein) 
     { m_type=0; m_static=0; }
  G__TypeReader() : G__TypeInfo() { clear(); }
  G__TypeReader(G__value& buf) : G__TypeInfo(buf) { m_type=0;m_static=0; }

  void clear();

  int append(const string& token,int c);

 public:
  void append_static() { m_static=1; }
  void append_const() { isconst |= Ispointer()?G__PCONSTVAR:G__CONSTVAR; }
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

  void setisconst(int isconstin) { isconst=isconstin; }
  void setreftype(int reftypein) { reftype=reftypein; }
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
  void Init(struct G__var_array *var,int ig15) {G__TypeInfo::Init(var,ig15);}
  G__value GetValue() const; 

};

#endif
