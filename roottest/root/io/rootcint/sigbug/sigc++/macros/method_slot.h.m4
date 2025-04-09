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

#ifndef   SIGC_METHOD_SLOT
#define   SIGC_METHOD_SLOT
#include <sigc++/slot.h>

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

/**************************************************************/
// These are internal classes used to represent function varients of slots

class Object;
// (internal) 
struct LIBSIGC_API MethodSlotNode : public SlotNode
  {
#ifdef _MSC_VER
private:
	/** the sole purpose of this declaration is to introduce a new type that is
	guaranteed not to be related to any other type. (Ab)using class SigC::Object
	for this lead to some faulty conversions taking place with MSVC6. */
	class GenericObject;
    typedef void* (GenericObject::*Method)(void*);
public:
#else
    typedef void* (Object::*Method)(void*);
#endif
    Method     method_;
    
    template <class T2>
    MethodSlotNode(FuncPtr proxy,T2 method)
      : SlotNode(proxy)
      { init(reinterpret_cast<Method&>(method)); }
    void init(Method method);
    virtual ~MethodSlotNode();
  };

struct LIBSIGC_API ConstMethodSlotNode : public SlotNode
  {
#ifdef _MSC_VER
private:
	/** the sole purpose of this declaration is to introduce a new type that is
	guaranteed not to be related to any other type. (Ab)using class SigC::Object
	for this lead to some faulty conversions taking place with MSVC6. */
	class GenericObject;
    typedef void* (GenericObject::*Method)(void*) const;
public:
#else
    typedef void* (Object::*Method)(void*) const;
#endif
    Method     method_;
    
    template <class T2>
    ConstMethodSlotNode(FuncPtr proxy,T2 method)
      : SlotNode(proxy)
      { init(reinterpret_cast<Method&>(method)); }
    void init(Method method);
    virtual ~ConstMethodSlotNode();
  };

dnl
dnl METHOD_SLOT(ARGS)
dnl
define([METHOD_SLOT],[dnl
template <LIST(class R,class Obj,ARG_CLASS($1))>
struct MethodSlot[]NUM($1)_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(LIST(Obj& obj, ARG_REF($1),void * s)) 
      { 
        typedef RType (Obj::*Method)(ARG_TYPE($1));
        MethodSlotNode* os = (MethodSlotNode*)s;
        return ((Obj*)(&obj)
           ->*(reinterpret_cast<Method&>(os->method_)))(ARG_NAME($1));
      }
  };

template <LIST(class R,class Obj,ARG_CLASS($1))>
struct ConstMethodSlot[]NUM($1)_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(LIST(Obj& obj, ARG_REF($1),void * s)) 
      { 
        typedef RType (Obj::*Method)(ARG_TYPE($1)) const;
        ConstMethodSlotNode* os = (ConstMethodSlotNode*)s;
        return ((Obj*)(&obj)
           ->*(reinterpret_cast<Method&>(os->method_)))(ARG_NAME($1));
      }
  };

template <LIST(class R,class Obj,ARG_CLASS($1))>
__SLOT__(R,Obj&,$1)
  slot(R (Obj::*method)(ARG_TYPE($1)))
  { 
    typedef MethodSlot[]NUM($1)_<LIST(R,Obj,ARG_TYPE($1))> SType;
    return new MethodSlotNode((FuncPtr)(&SType::proxy),
                            method);
  }

template <LIST(class R,class Obj,ARG_CLASS($1))>
__SLOT__(R,const Obj&,$1)
  slot(R (Obj::*method)(ARG_TYPE($1)) const)
  {
    typedef ConstMethodSlot[]NUM($1)_<LIST(R,Obj,ARG_TYPE($1))> SType;
    return new ConstMethodSlotNode((FuncPtr)(&SType::proxy),
                            method);
  }

])

// These do not derive from MethodSlot, they merely are extended
// ctor wrappers.  They introduce how to deal with the proxy.
METHOD_SLOT(ARGS(P,0))
METHOD_SLOT(ARGS(P,1))
METHOD_SLOT(ARGS(P,2))
METHOD_SLOT(ARGS(P,3))
METHOD_SLOT(ARGS(P,4))
METHOD_SLOT(ARGS(P,5))

#ifdef SIGC_CXX_NAMESPACES
}
#endif


#endif // SIGC_SLOT
