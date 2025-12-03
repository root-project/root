// -*- c++ -*-
dnl  signal.h.m4 - signal class for sigc++
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
#ifndef __header__
#define __header__
#include <sigc++/slot.h>
#include <sigc++/connection.h>
#include <sigc++/marshal.h>

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

/** @defgroup Signals
 * Use connect() with SigC::slot to connect a method or function with a Signal.
 *
 * @code
 * signal_clicked.connect( SigC::slot(*this, &MyWindow::on_clicked) );
 * @endcode
 *
 * When the signal is emitted your method will be called.
 *
 * connect() returns a Connection, which you can later use to disconnect your method.
 *
 * When Signals are copied they share the underlying information,
 * so you can have a protected/private SigC::Signal member and a public accessor method.
 */

class SignalConnectionNode;
class SignalExec_;

class LIBSIGC_API SignalNode : public SlotNode
  {
    public:
      int exec_count_; // atomic
      SignalConnectionNode *begin_,*end_;

      SignalNode();
      ~SignalNode();

      // must be inline to avoid emission slowdowns.
      void exec_reference()
        { 
          reference();
          exec_count_ += 1;
        }

      // must be inline to avoid emission slowdowns.
      void exec_unreference()
        {
          exec_count_ -= 1;

          if (defered_ && !exec_count_)
            cleanup();

          unreference();
        }

      SlotNode* create_slot(FuncPtr proxy); // nothrow
      ConnectionNode* push_front(const SlotBase& s);
      ConnectionNode* push_back(const SlotBase& s);

      virtual void remove(SignalConnectionNode* c);
      bool empty();
      void clear();
      void cleanup(); // nothrow
  };

class LIBSIGC_API SignalBase
  {
      friend class SignalConnectionNode;
    private:
      SignalBase& operator= (const SignalBase&); // no copy

    protected:
      typedef SignalExec_ Exec;

      mutable SignalNode *impl_;

      SlotNode* create_slot(FuncPtr c) const
        { return impl()->create_slot(c); }

      ConnectionNode* push_front(const SlotBase& s)
        { return impl()->push_front(s); }

      ConnectionNode* push_back(const SlotBase& s)
        { return impl()->push_back(s); }

      SignalBase();
      SignalBase(const SignalBase& s);
      SignalBase(SignalNode* s);
      ~SignalBase();

    public:
      bool empty() const
        { return !impl_ || impl()->empty(); }

      void clear()
        {
          if(impl_)
            impl()->clear();
        }

      SignalNode* impl() const;
  };

class LIBSIGC_API SignalConnectionNode : public ConnectionNode
  {
    public:
      virtual void notify(bool from_child);

      virtual ~SignalConnectionNode();
      SignalConnectionNode(SlotNode*);
  
      SignalNode *parent_;
      SignalConnectionNode *next_,*prev_;
      
      SlotNode* dest() { return (SlotNode*)(slot().impl()); }
  };

// Exeception-safe class for tracking signals.
class LIBSIGC_API SignalExec_
  {
  public:
    SignalNode* signal_;

    SignalExec_(SignalNode* signal) :signal_(signal)
      { signal_->exec_reference(); }

    ~SignalExec_()
      { signal_->exec_unreference(); }
  };
    

/********************************************************/

define([__SIGNAL__],[[Signal]eval(NUM($*)-2)<LIST($*)>])dnl
dnl
dnl  SIGNAL([P1..PN], R)
dnl
define([SIGNAL],[dnl
define([_R_],ifelse($2,void, void, R))
ifelse($2,void, [dnl
/// @ingroup Signals
template <LIST(ARG_CLASS($1), class Marsh)>
class __SIGNAL__(void, $1, Marsh) : public SignalBase
],[dnl
/// @ingroup Signals
template <LIST(class R,ARG_CLASS($1),class Marsh=Marshal<R>) >
class Signal[]NUM($1) : public SignalBase
])dnl
  {
    public:
      typedef Slot[]NUM($1)<LIST(_R_, [$1])> InSlotType;
ifelse($2,void, [dnl
      typedef InSlotType OutSlotType;
      typedef void OutType;
],[dnl
      typedef Slot[]NUM($1)<LIST(typename Marsh::OutType, [$1])> OutSlotType;
      typedef typename Trait<typename Marsh::OutType>::type OutType;
])dnl
 
    private:
      // Used for both emit and proxy.
      static OutType emit_(LIST(ARG_REF($1), void* data));

    public:
      OutSlotType slot() const
        { return create_slot((FuncPtr)(&emit_)); }

      operator OutSlotType() const
        { return create_slot((FuncPtr)(&emit_)); }

      /// You can call Connection::disconnect() later.
      Connection connect(const InSlotType& s)
        { return Connection(push_back(s)); }

      /// Call all the connected methods.
      OutType emit(ARG_REF($1))
        { ifelse($2,void, ,return) emit_(LIST(ARG_NAME($1), impl_)); }

      /// See emit()
      OutType operator()(ARG_REF($1))
        { ifelse($2,void, ,return) emit_(LIST(ARG_NAME($1), impl_)); }
 
      Signal[]NUM($1)() 
        : SignalBase() 
        {}

      Signal[]NUM($1)(const InSlotType& s)
        : SignalBase() 
        { connect(s); }

      ~Signal[]NUM($1)() {}
  };


// emit
ifelse($2,void, [dnl
template <LIST(ARG_CLASS($1),class Marsh)>
void __SIGNAL__(void, $1, Marsh)::emit_(LIST(ARG_REF($1), void* data))
  {
    SignalNode* impl = static_cast<SignalNode*>(data);

    if (!impl||!impl->begin_)
      return;

    Exec exec(impl);
    SlotNode* s = 0;
    for (SignalConnectionNode* i = impl->begin_; i; i = i->next_)
      {
        if (i->blocked())
          continue;

        s = i->dest();
        ((typename __SLOT__(void, $1)::Proxy)(s->proxy_))(LIST(ARG_NAME($1), s));
      }
    return;
  }
],[dnl
template <LIST(class R, ARG_CLASS($1), class Marsh)>
typename __SIGNAL__(_R_, $1, Marsh)::OutType
__SIGNAL__(_R_, $1, Marsh)::emit_(LIST(ARG_REF($1), void* data))
  {
    SignalNode* impl = static_cast<SignalNode*>(data);

    if (!impl || !impl->begin_)
      return Marsh::default_value();

    Exec exec(impl);
    Marsh rc;
    SlotNode* s = 0;

    for (SignalConnectionNode* i = impl->begin_; i; i=i->next_)
      {
        if (i->blocked()) continue;
        s = i->dest();
        if (rc.marshal(((typename __SLOT__(R,$1)::Proxy)(s->proxy_))(LIST(ARG_NAME($1), s))))
          return rc.value();
      }
    return rc.value();
  }
])dnl

])

SIGNAL(ARGS(P,0))
SIGNAL(ARGS(P,1))
SIGNAL(ARGS(P,2))
SIGNAL(ARGS(P,3))
SIGNAL(ARGS(P,4))
SIGNAL(ARGS(P,5))

#ifdef SIGC_CXX_PARTIAL_SPEC
SIGNAL(ARGS(P,0),void)
SIGNAL(ARGS(P,1),void)
SIGNAL(ARGS(P,2),void)
SIGNAL(ARGS(P,3),void)
SIGNAL(ARGS(P,4),void)
SIGNAL(ARGS(P,5),void)
#endif


#ifdef SIGC_CXX_NAMESPACES
}
#endif

#endif // __header__
