// -*- c++ -*-
dnl  convert.h.m4 - adaptor for changing a slot
dnl 
dnl  Copyright 2000, Karl Einar Nelson
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
  SigC::convert
  -------------
  convert() alters a Slot by assigning a conversion function 
  which can completely alter the parameter types of a slot. 

  Only convert functions for changing with same number of
  arguments is compiled by default.  See examples/custom_convert.h.m4 
  for details on how to build non standard ones.

  Sample usage:
    int my_string_to_char(Slot2<int,const char*> &d,const string &s)
    int f(const char*);
    string s=hello;


    Slot1<int,const string &>  s2=convert(slot(&f),my_string_to_char);
    s2(s);  

*/

#ifdef SIGC_CXX_NAMESPACES
namespace SigC
{
#endif

// (internal)
struct AdaptorConvertSlotNode : public AdaptorSlotNode
  {
    FuncPtr convert_func_;

    AdaptorConvertSlotNode(FuncPtr proxy,const Node& s,FuncPtr dtor);

    virtual ~AdaptorConvertSlotNode();
  };



define([__CRT_SLOT__],[[AdaptorConvertSlot]eval(NUM($*)-2)_<LIST($*)>])dnl
dnl
dnl ADAPTOR_CONVERT_SLOT([P1..PN])
dnl
define([ADAPTOR_CONVERT_SLOT],[dnl
template <LIST(class R,ARG_CLASS($1),class T)>
struct [AdaptorConvertSlot]NUM($1)_
  {
    typedef typename Trait<R>::type RType;
    typedef R (*ConvertFunc)(LIST(T&,$1));
    static RType proxy(LIST(ARG_REF($1),void *data)) 
      { 
        AdaptorConvertSlotNode& node=*(AdaptorConvertSlotNode*)(data);
        T &slot_=(T&)(node.slot_);
        return ((ConvertFunc)(node.convert_func_))
          (LIST(slot_,ARG_NAME($1)));
      }
  };

template <LIST(class R,ARG_CLASS($1),class T)>
__SLOT__(R,$1)
  convert(const T& slot_, R (*convert_func)(LIST(T&,$1)))
  { 
    return new AdaptorConvertSlotNode((FuncPtr)(&__CRT_SLOT__(R,$1,T)::proxy),
                                    slot_,
                                    (FuncPtr)(convert_func));
  }

])

ADAPTOR_CONVERT_SLOT(ARGS(P,0))
ADAPTOR_CONVERT_SLOT(ARGS(P,1))
ADAPTOR_CONVERT_SLOT(ARGS(P,2))
ADAPTOR_CONVERT_SLOT(ARGS(P,3))
ADAPTOR_CONVERT_SLOT(ARGS(P,4))
ADAPTOR_CONVERT_SLOT(ARGS(P,5))

#ifdef SIGC_CXX_NAMESPACES
}
#endif



#endif // __header__
