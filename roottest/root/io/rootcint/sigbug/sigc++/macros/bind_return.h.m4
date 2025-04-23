// -*- c++ -*-
dnl  bind_return.h.m4 - adaptor for fixing the return value of a slot.
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
#include <sigc++/bind.h>

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

define([__ARB_SLOT__],[[AdaptorBindReturnSlot]eval(NUM($*)-2)_<LIST($*)>])dnl
dnl
dnl ADAPTOR_BIND_RETURN_SLOT([P1..PN])
dnl
define([ADAPTOR_BIND_RETURN_SLOT],[dnl
/****************************************************************
***** Adaptor Return Bind Slot NUM($1) arguments
****************************************************************/
template <LIST(class R1,class R2,ARG_CLASS($1))>
struct [AdaptorBindReturnSlot]NUM($1)_
  {
    typedef AdaptorBindData1_<R1> Data;
    typedef typename __SLOT__(R2,$1)::Proxy Proxy;
    static R1 proxy(LIST(ARG_REF($1),void *data)) 
      { 
        Data& node = *reinterpret_cast<Data*>(data);
        SlotNode* slot = static_cast<SlotNode*>(node.adaptor.slot_.impl());
        ((Proxy)(slot->proxy_))
          (LIST(ARG_NAME($1),slot));
        return node.c1_; 
      }
  };

/// @ingroup bind
template <LIST(class R1, class R2,ARG_CLASS($1))>
__SLOT__(R1,$1)
  bind_return(const __SLOT__(R2,$1) &s,R1 ret)
  { 
    typedef AdaptorBindData1_<R1> Data;
    typedef __ARB_SLOT__(R1,R2,$1) Adaptor;
    return reinterpret_cast<SlotNode*>(
       new Data((FuncPtr)(&Adaptor::proxy),s,
                (FuncPtr)(&Data::dtor),ret));
  }

])

ADAPTOR_BIND_RETURN_SLOT(ARGS(P,0))
ADAPTOR_BIND_RETURN_SLOT(ARGS(P,1))
ADAPTOR_BIND_RETURN_SLOT(ARGS(P,2))
ADAPTOR_BIND_RETURN_SLOT(ARGS(P,3))
ADAPTOR_BIND_RETURN_SLOT(ARGS(P,4))
ADAPTOR_BIND_RETURN_SLOT(ARGS(P,5))

#ifdef SIGC_CXX_NAMESPACES
}
#endif

#endif // __header__
