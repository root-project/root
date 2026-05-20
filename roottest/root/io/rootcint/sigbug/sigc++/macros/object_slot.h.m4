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

#ifndef   SIGC_OBJECT_SLOT
#define   SIGC_OBJECT_SLOT
#include <sigc++/slot.h>
#include <sigc++/object.h>

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

/**************************************************************/
// These are internal classes used to represent function varients of slots

// (internal) 
struct LIBSIGC_API ObjectSlotNode : public SlotNode
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
    typedef void (Object::*Method)(void);
#endif
    Control_  *control_;
    void      *object_;
    Method     method_;
    Link       link_;
    
    // Can be a dependency 
    virtual Link* link();
    virtual void notify(bool from_child);

    template <class T,class T2>
    ObjectSlotNode(FuncPtr proxy,T* control,void *object,T2 method)
      : SlotNode(proxy)
      { init(control,object,reinterpret_cast<Method&>(method)); }
    void init(Object* control, void* object, Method method);
    virtual ~ObjectSlotNode();
  };

dnl
dnl OBJECT_SLOT(ARGS)
dnl
define([OBJECT_SLOT],[dnl
template <LIST(class R,ARG_CLASS($1),class Obj)>
struct ObjectSlot[]NUM($1)_ 
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(LIST(ARG_REF($1),void * s)) 
      { 
        typedef RType (Obj::*Method)(ARG_TYPE($1));
        ObjectSlotNode* os = (ObjectSlotNode*)s;
        return ((Obj*)(os->object_)
           ->*(reinterpret_cast<Method&>(os->method_)))(ARG_NAME($1)); 
      }
  };

template <LIST(class R,ARG_CLASS($1),class O1,class O2)>
__SLOT__(R,$1)
  slot(O1& obj,R (O2::*method)(ARG_TYPE($1)))
  { 
    typedef ObjectSlot[]NUM($1)_<LIST(R,ARG_TYPE($1),O2)> SType;
    O2& obj_of_method = obj;
    return new ObjectSlotNode((FuncPtr)(&SType::proxy),
                            &obj,
                            &obj_of_method,
                            method);
  }

template <LIST(class R,ARG_CLASS($1),class O1,class O2)>
__SLOT__(R,$1)
  slot(O1& obj,R (O2::*method)(ARG_TYPE($1)) const)
  {
    typedef ObjectSlot[]NUM($1)_<LIST(R,ARG_TYPE($1),O2)> SType;
    O2& obj_of_method = obj;
    return new ObjectSlotNode((FuncPtr)(&SType::proxy),
                            &obj,
                            &obj_of_method,
                            method);
  }

])

// These do not derive from ObjectSlot, they merely are extended
// ctor wrappers.  They introduce how to deal with the proxy.
OBJECT_SLOT(ARGS(P,0))
OBJECT_SLOT(ARGS(P,1))
OBJECT_SLOT(ARGS(P,2))
OBJECT_SLOT(ARGS(P,3))
OBJECT_SLOT(ARGS(P,4))
OBJECT_SLOT(ARGS(P,5))
OBJECT_SLOT(ARGS(P,6))

#ifdef SIGC_CXX_NAMESPACES
}
#endif


#endif /* SIGC_OBJECT_SLOT */

