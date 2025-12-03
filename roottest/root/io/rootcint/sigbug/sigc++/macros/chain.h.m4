// -*- c++ -*-
dnl  chain.h.m4 - adaptor for excution of sequential slots.
dnl 
//   Copyright 2001, Karl Einar Nelson
dnl
dnl  This library is free software; you can redistribute it and/or
dnl  modify it under the terms of the GNU Lesser General Public
dnl  License as published by the Free Software Foundation; either
dnl  version 2 of the License, or (at your option) any later version.
dnl
dnl  This library is distributed in the hope that it will be useful,
dnl  but WITHOUT ANY WARRANTY; without even the implied warranty of
dnl  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
dnl  Lesser General Public License for more details.
dnl
dnl  You should have received a copy of the GNU Lesser General Public
dnl  License along with this library; if not, write to the Free Software
dnl  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA
dnl
include(template.macros.m4)
#ifndef   __header__
#define   __header__
#include <sigc++/slot.h>
#include <sigc++/bind.h>

// FIXME this is a quick hack - needs to become proper adaptor class
//  with handling of notify from both slots.  Needs stuff for void 
//  return broken compilers

/*
  SigC::chain
  -------------
  chain() binds two slots as a unified call.  Chain takes two
  slots and returns a third.  The second slot is called the
  getter.  It will receive all of the parameters of the resulting
  slot.  The first slot, the setter, will receive the return value 
  from the getter and its return will be the return value of the 
  combined slot.  An arbitrary number of chains can be set up taking 
  the return of one slot and passing it to the parameters of the next.

  Simple Sample usage:

    float get(int i)  {return i+1;}
    double set(float) {return i+2;}

    Slot1<double,int>   s1=chain(slot(&set),slot(&get)); 
    s1(1);  // set(get(1));

*/

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif


define([__CHAINF__],[[chain]eval(NUM($*)-2)_<LIST($*)>])
dnl
dnl ADAPTOR_CHAIN([P1..PN])
dnl
define([ADAPTOR_CHAIN],[dnl
template <LIST(class R,class C,ARG_CLASS($1))> 
R [chain]NUM($1)_(LIST(ARG_BOTH($1),__SLOT__(R,C) setter,__SLOT__(C,$1) getter))
  { return setter(getter(ARG_NAME($1))); } 
template <LIST(class R,class C,ARG_CLASS($1))> 
__SLOT__(R,$1) chain(const __SLOT__(R,C)& setter,const __SLOT__(C,$1)& getter) 
  { 
    typedef R (*Func)(LIST(ARG_TYPE($1),__SLOT__(R,C),__SLOT__(C,$1)));
    Func func=&__CHAINF__(R,C,$1);
    return bind(slot(func),setter,getter);
  }

])

ADAPTOR_CHAIN(ARGS(P,0))
ADAPTOR_CHAIN(ARGS(P,1))
ADAPTOR_CHAIN(ARGS(P,2))
ADAPTOR_CHAIN(ARGS(P,3))
ADAPTOR_CHAIN(ARGS(P,4))
ADAPTOR_CHAIN(ARGS(P,5))

#ifdef SIGC_CXX_NAMESPACES
}
#endif

#endif // __header__
