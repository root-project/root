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
 * Copyright(c) 1995~1999  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
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
Cint::G__BaseClassInfo::G__BaseClassInfo(G__ClassInfo& a)
: G__ClassInfo()
{
  Init(a);
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__BaseClassInfo::Init(G__ClassInfo& a)
{
  derivedtagnum = a.Tagnum();
  basep = -1;
}

///////////////////////////////////////////////////////////////////////////
long Cint::G__BaseClassInfo::Offset()
{
  if (IsValid()) {
    return (long) G__struct.baseclass[derivedtagnum]->herit[basep]->baseoffset;
  }
  else {
    return -1;
  }
}

///////////////////////////////////////////////////////////////////////////
long Cint::G__BaseClassInfo::Property()
{
  if (IsValid()) {
    long property = G__ClassInfo::Property();
    if (G__struct.baseclass[derivedtagnum]->herit[basep]->property & G__ISVIRTUALBASE) {
      property |= G__BIT_ISVIRTUALBASE;
    }
    if (G__struct.baseclass[derivedtagnum]->herit[basep]->property&G__ISDIRECTINHERIT) {
      property |= G__BIT_ISDIRECTINHERIT;
    }
    switch (G__struct.baseclass[derivedtagnum]->herit[basep]->baseaccess) {
    case G__PUBLIC:
      property |= G__BIT_ISPUBLIC;
      break;
    case G__PROTECTED:
      property |= G__BIT_ISPROTECTED;
      break;
    case G__PRIVATE:
      property |= G__BIT_ISPRIVATE;
      break;
    default:
      break;
    }
    return property;
  }
  else {
    return 0;
  }
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__BaseClassInfo::IsValid()
{
  if (
      (0 <= derivedtagnum) &&
      (derivedtagnum < G__struct.alltag) &&
      (0 <= basep) &&
      (basep < G__struct.baseclass[derivedtagnum]->basen)
  ) {
    return 1;
  }
  else {
    return 0;
  }
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__BaseClassInfo::Next()
{
  return Next(1); 
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__BaseClassInfo::Next(int onlydirect)
{
  ++basep;
  if (onlydirect) {
    while (IsValid() && !(G__struct.baseclass[derivedtagnum]->herit[basep]->property&G__ISDIRECTINHERIT)) {
      ++basep;
    }
  }
  // initialize base class so we can get name of baseclass
  if (IsValid()) {
    G__ClassInfo::Init(G__struct.baseclass[derivedtagnum]->herit[basep]->basetagnum);
    return 1;
  }
  return 0;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__BaseClassInfo::Prev()
{
  return Prev(1); 
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__BaseClassInfo::Prev(int onlydirect)
{
  if (basep == -1) {
    basep = G__struct.baseclass[derivedtagnum]->basen - 1;
  }
  else {
    --basep;
  }
  if(onlydirect) {
    while (IsValid() && !(G__struct.baseclass[derivedtagnum]->herit[basep]->property & G__ISDIRECTINHERIT)) {
      --basep;
    }
  }
  // initialize base class so we can get name of baseclass
  if (IsValid()) {
     G__ClassInfo::Init(G__struct.baseclass[derivedtagnum]->herit[basep]->basetagnum);
     return 1;
  }
  return 0;
}

