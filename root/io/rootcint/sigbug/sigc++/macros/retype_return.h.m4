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

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

define([__ART_SLOT__],[[AdaptorRetypeReturnSlot]eval(NUM($*)-2)_<LIST($*)>])dnl
dnl
dnl ADAPTOR_RETYPE_RETURN_SLOT([P1..PN],R)
dnl
define([ADAPTOR_RETYPE_RETURN_SLOT],[dnl
/****************************************************************
***** Adaptor Return Type Slot, NUM($1) arguments
****************************************************************/
template <LIST(ifelse($2,void,,class R2),class R1,ARG_CLASS($1))>
struct [AdaptorRetypeReturnSlot]NUM($1)_ ifelse($2,void,[<LIST(void,R1,ARG_TYPE($1))>])
  {
    typedef typename __SLOT__(R1,$1)::Proxy Proxy;
ifelse($2,void,[dnl
    static void proxy(LIST(ARG_REF($1),void *data)) 
],[dnl
    typedef typename Trait<R2>::type RType;
    static RType proxy(LIST(ARG_REF($1),void *data)) 
])dnl
      { 
        AdaptorSlotNode& node=*static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
ifelse($2,void,[dnl
        ((Proxy)(slot->proxy_))
          (LIST(ARG_NAME($1),slot));
],[dnl
        return RType(((Proxy)(slot->proxy_))
          (LIST(ARG_NAME($1),slot)));
])dnl
      }
  };

ifelse($2,void,,[dnl
/// @ingroup retype
template <LIST(class R2, class R,ARG_CLASS($1))>
__SLOT__(R2,$1)
  retype_return(const __SLOT__(R,$1) &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&__ART_SLOT__(R2,R,$1)::proxy),s);
  }

/// @ingroup hide
template <LIST(class R, ARG_CLASS($1))>
__SLOT__(void,$1)
  hide_return(const __SLOT__(R,$1) &s)
  {
    return retype_return<void>(s);
  }
])dnl

])

ADAPTOR_RETYPE_RETURN_SLOT(ARGS(P,0))
ADAPTOR_RETYPE_RETURN_SLOT(ARGS(P,1))
ADAPTOR_RETYPE_RETURN_SLOT(ARGS(P,2))
ADAPTOR_RETYPE_RETURN_SLOT(ARGS(P,3))
ADAPTOR_RETYPE_RETURN_SLOT(ARGS(P,4))
ADAPTOR_RETYPE_RETURN_SLOT(ARGS(P,5))
ADAPTOR_RETYPE_RETURN_SLOT(ARGS(P,6))

#if !defined(SIGC_CXX_VOID_CAST_RETURN) && defined(SIGC_CXX_PARTIAL_SPEC)
ADAPTOR_RETYPE_RETURN_SLOT(ARGS(P,0),void)
ADAPTOR_RETYPE_RETURN_SLOT(ARGS(P,1),void)
ADAPTOR_RETYPE_RETURN_SLOT(ARGS(P,2),void)
ADAPTOR_RETYPE_RETURN_SLOT(ARGS(P,3),void)
ADAPTOR_RETYPE_RETURN_SLOT(ARGS(P,4),void)
ADAPTOR_RETYPE_RETURN_SLOT(ARGS(P,5),void)
ADAPTOR_RETYPE_RETURN_SLOT(ARGS(P,6),void)
#endif

#ifdef SIGC_CXX_NAMESPACES
}
#endif

#endif // __header__
