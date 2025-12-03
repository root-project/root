// -*- c++ -*-
dnl  class_slot.h.m4 - constructs slots for non-complient classes
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

#ifndef   SIGC_CLASS_SLOT
#define   SIGC_CLASS_SLOT
#include <sigc++/slot.h>

/*
  SigC::slot_class() (class)
  -----------------------
  slot_class() can be applied to a class method to form a Slot with a
  profile equivalent to the method.  At the same time an instance
  of that class must be specified.  This is an unsafe interface.

  This does NOT require that the class be derived from SigC::Object.
  However, the object should be static with regards to the signal system.
  (allocated within the global scope.)  If it is not and a connected
  slot is call it will result in a segfault.  If the object must
  be destroyed before the connected slots, all connections must
  be disconnected by hand.

  Sample usage:

    struct A
      {
       void foo(int, int);
      } a;

    Slot2<void,int,int> s = slot_class(a, &A::foo);

*/


#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif


/**************************************************************/
// These are internal classes used to represent function varients of slots

// (internal) 
struct LIBSIGC_API ClassSlotNode : public SlotNode
  {
#ifdef _MSC_VER
private:
	/** the sole purpose of this declaration is to introduce a new type that is
	guaranteed not to be related to any other type. (Ab)using class SigC::Object
	for this lead to some faulty conversions taking place with MSVC6. */
	class GenericObject;
    typedef void (GenericObject::*Method)(void);
public:
#else
    typedef void (SlotNode::*Method)(void);
#endif
    void    *object_;
    Method   method_;
 
    template <class T1, class T2>
    ClassSlotNode(FuncPtr proxy,T1* obj,T2 method)
        : SlotNode(proxy), object_(obj), method_(reinterpret_cast<Method&>(method))
        {}

    virtual ~ClassSlotNode();
  };

dnl
dnl CLASS_SLOT(ARGS)
dnl
define([CLASS_SLOT],[dnl
template <LIST(class R,ARG_CLASS($1),class Obj)>
struct ClassSlot[]NUM($1)_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(LIST(ARG_REF($1),void *s)) 
      { 
        typedef RType (Obj::*Method)(ARG_TYPE($1));
        ClassSlotNode* os = (ClassSlotNode*)(s);
        return ((Obj*)(os->object_)
           ->*(reinterpret_cast<Method&>(os->method_)))(ARG_NAME($1)); 
      }
  };

template <LIST(class R,ARG_CLASS($1),class Obj)>
  Slot[]NUM($1)<LIST(R,ARG_TYPE($1))> 
    slot_class(Obj& obj,R (Obj::*method)(ARG_TYPE($1)))
  { 
    typedef ClassSlot[]NUM($1)_<LIST(R,ARG_TYPE($1),Obj)> SType;
    return new ClassSlotNode((FuncPtr)(&SType::proxy),&obj,method);
  }

template <LIST(class R, ARG_CLASS($1), class Obj)>
  Slot[]NUM($1)<LIST(R, ARG_TYPE($1))>
    slot_class(Obj& obj, R (Obj::*method)(ARG_TYPE($1)) const)
  {
    typedef ClassSlot[]NUM($1)_<LIST(R,ARG_TYPE($1), Obj)> SType;
    return new ClassSlotNode((FuncPtr)(&SType::proxy), &obj, method);
  }

])

// These do not derive from ClassSlot, they merely are extended
// ctor wrappers.  They introduce how to deal with the proxy.
CLASS_SLOT(ARGS(P,0))
CLASS_SLOT(ARGS(P,1))
CLASS_SLOT(ARGS(P,2))
CLASS_SLOT(ARGS(P,3))
CLASS_SLOT(ARGS(P,4))
CLASS_SLOT(ARGS(P,5))
CLASS_SLOT(ARGS(P,6))

#ifdef SIGC_CXX_NAMESPACES
}
#endif

#endif /* SIGC_CLASS_SLOT */

