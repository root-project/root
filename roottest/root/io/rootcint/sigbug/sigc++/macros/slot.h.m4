// -*- c++ -*-
dnl  slot.h.m4 - slot class for sigc++
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
#ifndef   __header__
#define   __header__

/* FIXME where is the docs buddy! */

/** @defgroup Slots
 * Slots are type-safe representations of callback methods and functions.
 * A Slot can be constructed from any function, regardless of whether it is
 * a global function, a member method, static, or virtual.
 *
 * Use the SigC::slot() template function to get a SigC::Slot, like so:
 * @code
 * SigC::Slot1<void, int> slot = SigC::slot(someobj, &SomeClass::somemethod);
 * @endcode
 * or
 * @code
 * m_Button.signal_clicked().connect( SigC::slot(*this, &MyWindow::on_button_clicked) );
 * @endcode
 * The compiler will complain if SomeClass::somemethod has the wrong signature.
 *
 * You can also pass slots as method parameters where you might normally pass a function pointer.
 */

#include <sigc++/node.h>
#include <sigc++/trait.h>

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

// this is a dummy type which we will cast to any static function
typedef void (*FuncPtr)(void*);

// (internal) All Slot types derive from this.
class LIBSIGC_API SlotNode: public NodeBase
  {
    public:
      virtual ~SlotNode()=0;

      // node must be dynamic
      virtual void add_dependency(NodeBase*);
      virtual void remove_dependency(NodeBase*);

      // message from child that it has died and we should start
      // our shut down.  If from_child is true, we do not need 
      // to clean up the child links.
      virtual void notify(bool from_child);

      SlotNode(FuncPtr proxy);

      FuncPtr proxy_;
      NodeBase *dep_;
  };

/**************************************************************/
// These are internal classes used to represent function varients of slots

// (internal) 
struct LIBSIGC_API FuncSlotNode : public SlotNode
  {
    FuncPtr func_;
    FuncSlotNode(FuncPtr proxy,FuncPtr func);
    virtual ~FuncSlotNode();
  };

define([__FUNC_SLOT__],[[FuncSlot]eval(NUM($*)-1)_<LIST($*)>])dnl
dnl
dnl FUNC_SLOT(ARGS)
dnl
define([FUNC_SLOT],[dnl
template <LIST(class R,ARG_CLASS($1))>
struct FuncSlot[]NUM($1)_
  {
    typedef typename Trait<R>::type RType;
    typedef RType (*Callback)($1);
    static RType proxy(LIST(ARG_REF($1),void *s)) 
      {   
        return ((Callback)(((FuncSlotNode*)s)->func_))(ARG_NAME($1)); 
      }
  };
])

// These do not derive from FuncSlot, they merely hold typedefs and
// static functions on how to deal with the proxy.
FUNC_SLOT(ARGS(P,0))
FUNC_SLOT(ARGS(P,1))
FUNC_SLOT(ARGS(P,2))
FUNC_SLOT(ARGS(P,3))
FUNC_SLOT(ARGS(P,4))
FUNC_SLOT(ARGS(P,5))
FUNC_SLOT(ARGS(P,6))

/**************************************************************/
// Slot# is a holder to a SlotNode, its type is held by type of the 
// pointer and not the SlotNode itself.  This reduces the
// number and size of type objects.

/// (internal) Typeless Slot base class.
class LIBSIGC_API SlotBase : public Node
  {
    public:
      // For backwards compatiblity
      bool connected() const { return valid(); }

      // (internal) 
      SlotNode* operator->() { return static_cast<SlotNode*>(node_); }
    protected:
      // users don't use slots so we will protect the methods
      SlotBase() : Node() {}
      SlotBase(const SlotBase& s) : Node(s) {}
      ~SlotBase() {}

      SlotBase& operator =(const SlotBase& s)
        { Node::operator=(s); return *this; }
  };

dnl
dnl  SLOT([P1...PN])
dnl
dnl  Notes, 
dnl   - changed call to valid() to node_->notified to cut emission time in
dnl     half
dnl
define([SLOT],[dnl
/// @ingroup Slots
template <LIST(class R,ARG_CLASS($1))>
class Slot[]NUM($1) : public SlotBase
  {
    public:
      typedef typename Trait<R>::type RType;
      typedef R (*Callback)(ARG_TYPE($1));
      typedef RType (*Proxy)(LIST(ARG_REF($1),void*));
      RType operator ()(ARG_REF($1))
        {
          if (!node_) return RType();
          if (node_->notified_)
            { clear(); return RType(); }
          return ((Proxy)(static_cast<SlotNode*>(node_)->proxy_))
            (LIST(ARG_NAME($1),node_));
        }
  
      Slot[]NUM($1)& operator= (const Slot[]NUM($1) &s)
        {
          SlotBase::operator=(s);
          return *this;
        }
  
      Slot[]NUM($1)() 
        : SlotBase() {} 
      Slot[]NUM($1)(const Slot[]NUM($1)& s) 
        : SlotBase(s) 
        {}
      Slot[]NUM($1)(SlotNode* node)
        : SlotBase()
        { assign(node); }
      Slot[]NUM($1)(Callback callback)
        : SlotBase()
        { 
          typedef __FUNC_SLOT__(R,$1) Proxy_;
          assign(new FuncSlotNode((FuncPtr)&Proxy_::proxy,(FuncPtr)callback));
        }
      ~Slot[]NUM($1)() {}
  };

template <LIST(class R,ARG_CLASS($1))>
__SLOT__(R,$1) slot(R (*func)(ARG_TYPE($1)))
  { return func; }
])

// these represent the callable structure of a slot with various types.
// they are fundimentally just wrappers.  They have no differences other
// that the types they cast data to.
SLOT(ARGS(P,0))
SLOT(ARGS(P,1))
SLOT(ARGS(P,2))
SLOT(ARGS(P,3))
SLOT(ARGS(P,4))
SLOT(ARGS(P,5))
SLOT(ARGS(P,6))

#ifdef SIGC_CXX_NAMESPACES
}
#endif

#endif // SIGC_SLOT
