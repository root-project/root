// -*- c++ -*-
dnl  object_slot.h.m4 - adaptor for changing argument types
dnl 
//   Copyright 2000, Karl Einar Nelson
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

#ifndef   SIGC_CLOSURE
#define   SIGC_CLOSURE
#include <sigc++/slot.h>
#include <sigc++/object.h>

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

class Object;
/**************************************************************/
// These are internal classes used to represent function varients of slots

// (internal) 
template <class T_object>
struct Closure_ : public SlotNode
  {
    typedef void (Object::*Method)(void);
    T_object  object_;
    Method     method_;
    
    template <class T,class T2>
    Closure_(FuncPtr proxy,const T_object &object,T2 method)
      : SlotNode(proxy), object_(object), 
        method(reinterpret_cast<Closure_::Method>(method))
      {}
    virtual ~Closure_()
      {}
  };

dnl
dnl CLOSURE(ARGS)
dnl
define([CLOSURE],[dnl
template <LIST(class R,class O1,class O2,ARG_CLASS($1))>
struct Closure[]NUM($1)_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(LIST(ARG_REF($1),void * s)) 
      { 
        typedef RType (Obj::*Method)(ARG_TYPE($1));
        Closure_<O1>* os=(Closure_<O1>*)s;
        return (os->object_).*(reinterpret_cast<Method>(os->method_))(ARG_NAME($1)); 
      }
  };

template <LIST(class R,ARG_CLASS($1),class O1,class O2)>
__SLOT__(R,$1)
  closure(const O1& obj,R (O2::*method)(ARG_TYPE($1)))
  { 
    typedef Closure[]NUM($1)_<LIST(R,ARG_TYPE($1),O2)> SType;
    return new Closure_<O1>((FuncPtr)(&SType::proxy),
                            obj, 
                            method);
  }

template <LIST(class R,ARG_CLASS($1),class O1,class O2)>
__SLOT__(R,$1)
  closure(const O1& obj,R (O2::*method)(ARG_TYPE($1)) const)
  {
    typedef Closure[]NUM($1)_<LIST(R,ARG_TYPE($1),O2)> SType;
    return new Closure_<O1>((FuncPtr)(&SType::proxy),
                            obj,
                            method);
  }

])

// These do not derive from Closure, they merely are extended
// ctor wrappers.  They introduce how to deal with the proxy.
CLOSURE(ARGS(P,0))
CLOSURE(ARGS(P,1))
CLOSURE(ARGS(P,2))
CLOSURE(ARGS(P,3))
CLOSURE(ARGS(P,4))
CLOSURE(ARGS(P,5))

#ifdef SIGC_CXX_NAMESPACES
}
#endif

#endif // SIGC_SLOT
