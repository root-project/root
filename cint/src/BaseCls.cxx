/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file BaseCls.cxx
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Author                  Masaharu Goto 
 * Copyright(c) 1995~1999  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation. The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#include "Api.h"
#include "common.h"

/*********************************************************************
* class G__BaseClassInfo
*
* Rene says OK
* 
*********************************************************************/

///////////////////////////////////////////////////////////////////////////

G__BaseClassInfo::G__BaseClassInfo(G__ClassInfo &a) : G__ClassInfo()
{
  Init(a);
}

///////////////////////////////////////////////////////////////////////////

void G__BaseClassInfo::Init(G__ClassInfo &a)
{
  derivedtagnum = a.Tagnum();
  basep = -1;
}

///////////////////////////////////////////////////////////////////////////

long G__BaseClassInfo::Offset()
{
  if(IsValid()) {
    return((long)G__struct.baseclass[derivedtagnum]->baseoffset[basep]);
  }
  else {
    return(-1);
  }
}

///////////////////////////////////////////////////////////////////////////

long G__BaseClassInfo::Property()
{
  long property;
  if(IsValid()) {
    property = G__ClassInfo::Property();
    if(G__struct.baseclass[derivedtagnum]->property[basep]&G__ISVIRTUALBASE)
#ifndef G__OLDIMPLEMENTATION2148
      property|=G__BIT_ISVIRTUALBASE;
#else
      property|=G__BIT_ISVIRTUAL;
#endif
#ifndef G__OLDIMPLEMENTATION1888
    if(G__struct.baseclass[derivedtagnum]->property[basep]&G__ISDIRECTINHERIT)
      property|=G__BIT_ISDIRECTINHERIT;
#endif
    switch(G__struct.baseclass[derivedtagnum]->baseaccess[basep]) {
    case G__PUBLIC:
      return(property|G__BIT_ISPUBLIC);
    case G__PROTECTED:
      return(property|G__BIT_ISPROTECTED);
    case G__PRIVATE:
      return(property|G__BIT_ISPRIVATE);
    default:
      return(property);
    }
  }
  else {
    return(0);
  }
}

///////////////////////////////////////////////////////////////////////////

int G__BaseClassInfo::IsValid()
{
  if(0<=derivedtagnum && derivedtagnum < G__struct.alltag &&
     0<=basep && basep<G__struct.baseclass[derivedtagnum]->basen) {
    return(1);
  }
  else {
    return(0);
  }
}

///////////////////////////////////////////////////////////////////////////
int G__BaseClassInfo::Next() {
  return(Next(1)); 
}

///////////////////////////////////////////////////////////////////////////
int G__BaseClassInfo::Next(int onlydirect)
{
  ++basep;
#ifndef G__FONS56
  if(onlydirect) {
    while (IsValid() &&
     !(G__struct.baseclass[derivedtagnum]->property[basep]&G__ISDIRECTINHERIT))
      ++basep;
  }
  // initialize base class so we can get name of baseclass
  if (IsValid()) {
     G__ClassInfo::Init(G__struct.baseclass[derivedtagnum]->basetagnum[basep]);
     return 1;
  }
#endif
  return(IsValid());
}

///////////////////////////////////////////////////////////////////////////
int G__BaseClassInfo::Prev() {
  return(Next(1)); 
}

///////////////////////////////////////////////////////////////////////////
int G__BaseClassInfo::Prev(int onlydirect)
{
  if(-1==basep) basep = G__struct.baseclass[derivedtagnum]->basen-1;
  else --basep;
#ifndef G__FONS56
  if(onlydirect) {
    while (IsValid() &&
     !(G__struct.baseclass[derivedtagnum]->property[basep]&G__ISDIRECTINHERIT))
      --basep;
  }
  // initialize base class so we can get name of baseclass
  if (IsValid()) {
     G__ClassInfo::Init(G__struct.baseclass[derivedtagnum]->basetagnum[basep]);
     return 1;
  }
#endif
  return(IsValid());
}
///////////////////////////////////////////////////////////////////////////
