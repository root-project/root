// -*- c++ -*-
dnl  retype_return.h.m4 - adaptor for changing return type
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
#include <sigc++/adaptor.h>

/*
  SigC::auto_disconnect
  -------------
  auto_disconnect() causes a slot to disconnect automatically after a
  fixed number of calls.

  Simple Sample usage:

    void f(int);

    Slot1<void,int>   s1=auto_disconnect(slot(&f),2); 
    s1();
    s1(); // slot disconnects
    s1(); 

*/

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

// (internal) 
struct AdaptorAutoDisconnectSlotNode : public AdaptorSlotNode
  {
    // struct to post decrement the disconnect count
    struct Dec 
      {
        AdaptorAutoDisconnectSlotNode* node_;
        Dec(AdaptorAutoDisconnectSlotNode& node): node_(&node) {}
        ~Dec() { if (!--node_->count_) node_->notify(false); }
      };
    int count_;
    AdaptorAutoDisconnectSlotNode(FuncPtr proxy,const Node& s,int count);
    virtual ~AdaptorAutoDisconnectSlotNode();
  };


define([__AAD_SLOT__],[[AdaptorAutoDisconnectSlot]eval(NUM($*)-1)_<LIST($*)>])dnl
dnl
dnl ADAPTOR_AUTO_DISCONNECT_SLOT([P1..PN])
dnl
define([ADAPTOR_AUTO_DISCONNECT_SLOT],[dnl
/****************************************************************
***** Adaptor Auto Disconnect Slot, NUM($1) arguments
****************************************************************/
template <LIST(class R,ARG_CLASS($1))>
struct [AdaptorAutoDisconnectSlot]NUM($1)_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(LIST(ARG_REF($1),void *data)) 
      { 
        AdaptorAutoDisconnectSlotNode& node=
          *static_cast<AdaptorAutoDisconnectSlotNode*>(data);
        AdaptorAutoDisconnectSlotNode::Dec dec(node); 
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return ((__SLOT__(R,$1)::Proxy)(slot->proxy_))
          (LIST(ARG_NAME($1),slot));
      }
  };

template <LIST(class R,ARG_CLASS($1))>
__SLOT__(R,$1)
  auto_disconnect(const __SLOT__(R,$1) &s,int count=1)
  { 
    return new AdaptorAutoDisconnectSlotNode((FuncPtr)(&__AAD_SLOT__(R,$1)::proxy),s,count);
  }

])

ADAPTOR_AUTO_DISCONNECT_SLOT(ARGS(P,0))
ADAPTOR_AUTO_DISCONNECT_SLOT(ARGS(P,1))
ADAPTOR_AUTO_DISCONNECT_SLOT(ARGS(P,2))
ADAPTOR_AUTO_DISCONNECT_SLOT(ARGS(P,3))
ADAPTOR_AUTO_DISCONNECT_SLOT(ARGS(P,4))
ADAPTOR_AUTO_DISCONNECT_SLOT(ARGS(P,5))

#ifdef SIGC_CXX_NAMESPACES
}
#endif

#endif // __header__
