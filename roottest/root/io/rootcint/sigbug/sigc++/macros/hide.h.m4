// -*- c++ -*-
dnl  hide.h.m4  -  hide one or two of the signal's arguments
dnl  
//   Copyright 2000, Martin Schulze <MHL.Schulze@t-online.de>
//   Copyright 2001, Karl Einar Nelson
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

/** @defgroup hide
 * SigC::hide() alters a Slot in that it adds one or two parameters
 * whose values are ignored on invocation of the Slot.
 * Thus you can discard one or more of the arguments of a Signal.
 * You have to specify the type of the parameters to ignore as
 * template arguments as in
 * @code
 * SigC::Slot1<void, int> slot1;
 * SigC::Slot0<void> slot2 = SigC::hide<int>(slot1);
 * @endcode
 *
 * SigC::hide_return() alters the Slot by
 * dropping its return value, thus converting it to a void Slot.
 *
 * Simple sample usage:
 *
 * @code
 * int f(int);
 * SigC::Slot1<void, int> s = SigC::hide_return( slot(&f) );
 * @endcode
 */

#include <sigc++/adaptor.h>

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

dnl
dnl ADAPTOR_HIDE(R,[P1..PN],[H1..HM])
dnl
define([ADAPTOR_HIDE],[dnl
template <LIST(class R,ARG_CLASS($1),ARG_CLASS($2))>
struct [AdaptorHide]NUM($1)[_]NUM($2)_
  {
    typedef typename Trait<R>::type RType;
    static RType proxy(LIST(ARG_REF($1),ARG_REFTYPE($2),[void *data])) 
      {
        AdaptorSlotNode& node = *(AdaptorSlotNode*)(data);
        SlotNode* slot=static_cast<SlotNode*>(node.slot_.impl());
        return ((typename __SLOT__(R,$1)::Proxy)(slot->proxy_))(LIST(ARG_NAME($1),slot));
      }
  };

/// @ingroup hide
template <LIST(ARG_CLASS($2), class R, ARG_CLASS($1))>
__SLOT__(R,[$1],[$2])
hide(const __SLOT__(R,$1)& s)
  {
    return new AdaptorSlotNode( (FuncPtr)(&[AdaptorHide]NUM($1)[_]NUM($2)_<LIST(R,ARG_TYPE($1),ARG_TYPE($2))>::proxy), s );
  }

])dnl ADAPTOR_HIDE

ADAPTOR_HIDE(ARGS(P,0),ARGS(H,1))
ADAPTOR_HIDE(ARGS(P,0),ARGS(H,2))
ADAPTOR_HIDE(ARGS(P,1),ARGS(H,1))
ADAPTOR_HIDE(ARGS(P,1),ARGS(H,2))
ADAPTOR_HIDE(ARGS(P,2),ARGS(H,1))
ADAPTOR_HIDE(ARGS(P,2),ARGS(H,2))
ADAPTOR_HIDE(ARGS(P,3),ARGS(H,1))
ADAPTOR_HIDE(ARGS(P,3),ARGS(H,2))
ADAPTOR_HIDE(ARGS(P,4),ARGS(H,1))

#ifdef SIGC_CXX_NAMESPACES
}  // namespace SigC
#endif

#endif  // __header__
