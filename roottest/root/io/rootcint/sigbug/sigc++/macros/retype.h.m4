// -*- c++ -*-
dnl  retype.h.m4 - adaptor for changing argument types
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

/** @defgroup retype
 * retype() alters a Slot to change arguments and return type.
 *
 * Only allowed conversions or conversions to void can properly
 * be implemented.  The types of the return and all arguments must always
 * be specified as a template parameters.
 *
 * Simple sample usage:
 *
 * @code
 *  float f(float,float);
 *
 *  SigC::Slot1<int, int, int>  s1 = SigC::retype<int, int, int>(slot(&f));
 *  @endcode
 *
 *
 * SigC::retype_return() alters a Slot by changing the return type.
 *
 * Only allowed conversions or conversions to void can properly
 * be implemented.  The type must always be specified as a
 * template parameter.
 *
 * Simple sample usage:
 *
 * @code
 * int f(int);
 *
 * SigC::Slot1<void, int> s1 = SigC::retype_return<void>(slot(&f));
 * SigC::Slot1<float, int> s2 = SigC::retype_return<float>(slot(&f));
 * @endcode
 *
 * SigC::hide_return() is an easy-to-use variant that converts the Slot by
 * dropping its return value, thus converting it to a void slot.
 *
 * Simple Sample usage:
 *
 * @code
 * int f(int);
 * SigC::Slot1<void, int> s = SigC::hide_return( slot(&f) );
 * @endcode
 */

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

define([__RETYPE_SLOT__],[[AdaptorRetypeSlot]eval(NUM($*)/2-1)_<LIST($*)>])dnl
dnl
dnl ADAPTOR_RETYPE_SLOT([P1..PN],[Q1..Q2],R)
dnl
define([ADAPTOR_RETYPE_SLOT],[dnl
template <LIST(ifelse($3,void,,class R1), ARG_CLASS($1),class R2,ARG_CLASS($2))>
struct [AdaptorRetypeSlot]NUM($1)_ ifelse($3, void,[<LIST(void,ARG_TYPE($1), R2,ARG_TYPE($2))>])
  {
    typedef typename __SLOT__(R2,$2)::Proxy Proxy;
ifelse($3,void,[dnl
    static void proxy(LIST(ARG_REF($1), void *data))
],[dnl
    typedef typename Trait<R1>::type RType;
    static RType proxy(LIST(ARG_REF($1), void *data))
])dnl
      { 
        AdaptorSlotNode& node = *static_cast<AdaptorSlotNode*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
ifelse($3,void,[dnl
        ((Proxy)(slot->proxy_))(LIST(ARG_NAME($1), slot));
],[dnl
        return RType(((Proxy)(slot->proxy_))
          (LIST(ARG_NAME($1),slot)));
])dnl
      }
  };

ifelse($3,void,,[dnl
/// @ingroup retype
template <LIST(class R1,ARG_CLASS($1),class R2,ARG_CLASS($2))>
__SLOT__(R1,$1)
  retype(const __SLOT__(R2,$2) &s)
  { 
    return new AdaptorSlotNode((FuncPtr)(&__RETYPE_SLOT__(R1,$1,R2,$2)::proxy),s);
  }

template <LIST(class R1,ARG_CLASS($1),class R2,ARG_CLASS($2))>
__SLOT__(R1,$1)
  [retype]NUM($1)[(]const __SLOT__(R2,$2) &s[)]
  { 
    return new AdaptorSlotNode((FuncPtr)(&__RETYPE_SLOT__(R1,$1,R2,$2)::proxy),s);
  }

])
])

ADAPTOR_RETYPE_SLOT(ARGS(P,0),ARGS(C,0))
ADAPTOR_RETYPE_SLOT(ARGS(P,1),ARGS(C,1))
ADAPTOR_RETYPE_SLOT(ARGS(P,2),ARGS(C,2))
ADAPTOR_RETYPE_SLOT(ARGS(P,3),ARGS(C,3))
ADAPTOR_RETYPE_SLOT(ARGS(P,4),ARGS(C,4))
ADAPTOR_RETYPE_SLOT(ARGS(P,5),ARGS(C,5))
ADAPTOR_RETYPE_SLOT(ARGS(P,6),ARGS(C,6))

#if !defined(SIGC_CXX_VOID_CAST_RETURN) && defined(SIGC_CXX_PARTIAL_SPEC)
ADAPTOR_RETYPE_SLOT(ARGS(P,0),ARGS(C,0),void)
ADAPTOR_RETYPE_SLOT(ARGS(P,1),ARGS(C,1),void)
ADAPTOR_RETYPE_SLOT(ARGS(P,2),ARGS(C,2),void)
ADAPTOR_RETYPE_SLOT(ARGS(P,3),ARGS(C,3),void)
ADAPTOR_RETYPE_SLOT(ARGS(P,4),ARGS(C,4),void)
ADAPTOR_RETYPE_SLOT(ARGS(P,5),ARGS(C,5),void)
ADAPTOR_RETYPE_SLOT(ARGS(P,6),ARGS(C,6),void)
#endif

#ifdef SIGC_CXX_NAMESPACES
} // namespace
#endif

#endif // __header__
