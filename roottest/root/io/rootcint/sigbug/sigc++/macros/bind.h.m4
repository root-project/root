// -*- c++ -*-
dnl  bind.h.m4 - adaptor to fix arguments to a value
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
dnl
dnl  Implementation notes:
dnl    In order to avoid creating type records and vtables for 
dnl  the various bind type slots, I have factored out the dtor
dnl  of the data items.  It creates a data node containing 
dnl  both the adaptor and the extra data which needs to be added.
dnl  Then when the adaptor is destroyed it, it calls the dtor 
dnl  procedure which takes out the extra data.  
dnl
dnl  It is possible this technique is non-portable though given
dnl  the necessary interactions with C code it seems unlikely 
dnl  to break all but the most exotic of C++ compilers.
dnl  (watch Karl eat hat)
dnl
include(template.macros.m4)
#ifndef   __header__
#define   __header__
#include <sigc++/adaptor.h>

/** @defgroup bind
 *
 * SigC::bind() alters a SigC::Slot by fixing arguments to certain values.
 *
 * Argument fixing starts from the last argument.
 * Up to two arguments can be bound at a time.
 *
 * Simple sample usage:
 * @code
 * void f(int, int);
 * SigC:Slot2<void, int, int> s1 = SigC::slot(f);
  *
 * SigC::Slot1<void, int>  s2 = SigC::bind(s1,1);
 * s2(2);  // call f with arguments 2,1
 * @endcode
 *
 *  Multibinding usage:
 *
 * @code
 *  void f(int,int);
 *  SigC::Slot2<void, int, int> s1 = SigC::slot(f);
  *
 *  SigC::Slot0<void>  s2 = SigC::bind(s1, 1, 2);
 *  s2();  // call f with arguments 1, 2
 * @endcode
 *
 *  Type specified usage:
 *
 *  @code
 *  class A {};
 *  class B : public A {};
 *  B* b;
 *  SigC::Slot0<void, A*> s1;
 *
 *  SigC::Slot0<void> s2 = SIgC::bind(s1, b);  // B* converted to A*
 * @endcode
 *
 *
 * SigC::bind_return() alters a Slot by fixing the return value to certain values
 *
 * Return value fixing ignores any slot return value.  The slot is
 * destroyed in the process and a new one is created, so references
 * to the slot will no longer be valid.
 *
 * Typecasting may be necessary to match arguments between the
 * slot and the bound return value.  Types must be an exact match.
 * To ensure the proper type, the type can be explicitly specified
 * on template instantation.
 *
 * Simple sample usage:
 * @code
 * void f(int, int);
 * SigC::Slot1<int, int, int>  s1 = SigC::bind_return(slot(&f), 1);
 * std::cout << "s2: " << s1(2, 1) << std::endl;
 * @endcode
 *
 * Type specified usage:
 * @code
 * class A {};
 * class B : public A {};
 * B* b;
 * SigC::Slot1<void> s1;
 *
 * SigC::Slot0<A*> s2 = SigC::bind_return<A*>(s1, b);  // B* must be told to match A*
 * @endcode
 *
 */

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

define([FORMAT_ARG_CBINIT],[LOWER([$1])_(LOWER([$1]))])dnl
define([FORMAT_ARG_CBNAME],[node.LOWER([$1])_])dnl
define([FORMAT_ARG_CBDTOR],[node.LOWER([$1])_.~[$1]();])dnl
define([FORMAT_ARG_CBBIND],[[$1] LOWER([$1])_;])dnl
dnl
define([ARG_CBINIT],[PROT(ARG_LOOP([FORMAT_ARG_CBINIT],[[,]],$*))])dnl
define([ARG_CBNAME],[PROT(ARG_LOOP([FORMAT_ARG_CBNAME],[[,]],$*))])dnl
define([ARG_CBDTOR],[PROT(ARG_LOOP([FORMAT_ARG_CBDTOR],[[
        ]],$*))])dnl
define([ARG_CBBIND],[PROT(ARG_LOOP([FORMAT_ARG_CBBIND],[[
        ]],$*))])dnl
dnl

/**************************************************************/
// These are internal classes used to represent function varients of slots

// (internal) 
struct LIBSIGC_API AdaptorBindSlotNode : public AdaptorSlotNode
  {
    FuncPtr dtor_;

    AdaptorBindSlotNode(FuncPtr proxy, const Node& s, FuncPtr dtor);

    virtual ~AdaptorBindSlotNode();
  };



dnl
dnl ADAPTOR_BIND_DATA([C0..CM])
dnl
define([ADAPTOR_BIND_DATA],[dnl
template <ARG_CLASS($1)>
struct [AdaptorBindData]NUM($1)_
  {
    typedef [AdaptorBindData]NUM($1)_ Self;
    AdaptorBindSlotNode adaptor;
    ARG_CBBIND($1)
    AdaptorBindData[]NUM($1)_(FuncPtr p, const Node& s ,FuncPtr d,
      ARG_BOTH($1))
    : adaptor(p, s, d), ARG_CBINIT($1)
      {}

    static void dtor(void* data)
      {
        Self& node = *reinterpret_cast<Self*>(data);
        ARG_CBDTOR($1)
      }
  }; 

])

dnl
dnl ADAPTOR_BIND_SLOT([P1..PN],[C0..CM],[A0..AM])
dnl
define([ADAPTOR_BIND_SLOT],[dnl
template <LIST(class R,ARG_CLASS($1),ARG_CLASS($2))>
struct [AdaptorBindSlot]NUM($1)[_]NUM($2)_
  {
    typedef typename Trait<R>::type RType;
    typedef typename __SLOT__(R,$1,$2)::Proxy Proxy;
    static RType proxy(LIST(ARG_REF($1),void *data)) 
      { 
        typedef [AdaptorBindData]NUM($2)_<ARG_TYPE($2)> Data;
        Data& node=*reinterpret_cast<Data*>(data);
        SlotNode* slot=static_cast<SlotNode*>(node.adaptor.slot_.impl());
        return ((Proxy)(slot->proxy_))
          (LIST(ARG_NAME($1),ARG_CBNAME($2)),slot);
      }

  };

/// @ingroup bind
template <LIST(ARG_CLASS($3),class R,ARG_CLASS($1),ARG_CLASS($2))>
__SLOT__(R,$1)
  bind(const __SLOT__(R,$1,$2)& s,
       ARG_BOTH($3))
  { 
    typedef [AdaptorBindData]NUM($2)[_]<ARG_TYPE($2)> Data;
    typedef [AdaptorBindSlot]NUM($1)[_]NUM($2)_<LIST(R,ARG_TYPE($1),ARG_TYPE($2))> Adaptor;
    return reinterpret_cast<SlotNode*>(
       new Data((FuncPtr)(&Adaptor::proxy),s,
                (FuncPtr)(&Data::dtor),ARG_NAME($3)));
  }

])

ADAPTOR_BIND_DATA(ARGS(C,1))
ADAPTOR_BIND_DATA(ARGS(C,2))

ADAPTOR_BIND_SLOT(ARGS(P,0),ARGS(C,1),ARGS(A,1))
ADAPTOR_BIND_SLOT(ARGS(P,1),ARGS(C,1),ARGS(A,1))
ADAPTOR_BIND_SLOT(ARGS(P,2),ARGS(C,1),ARGS(A,1))
ADAPTOR_BIND_SLOT(ARGS(P,3),ARGS(C,1),ARGS(A,1))
ADAPTOR_BIND_SLOT(ARGS(P,4),ARGS(C,1),ARGS(A,1))

ADAPTOR_BIND_SLOT(ARGS(P,0),ARGS(C,2),ARGS(A,2))
ADAPTOR_BIND_SLOT(ARGS(P,1),ARGS(C,2),ARGS(A,2))
ADAPTOR_BIND_SLOT(ARGS(P,2),ARGS(C,2),ARGS(A,2))
ADAPTOR_BIND_SLOT(ARGS(P,3),ARGS(C,2),ARGS(A,2))

#ifdef SIGC_CXX_NAMESPACES
} // namespace
#endif

#endif // __header__
