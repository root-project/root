// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
# define REFLEX_BUILD
#endif

#include "Reflex/Any.h"
#include <string>
#include <iostream>

//-------------------------------------------------------------------------------
std::ostream&
Reflex::operator <<(std::ostream& o,
                    const Any& any) {
//-------------------------------------------------------------------------------
// Puts the different any objects on the ostream.
   if (any.TypeInfo() == typeid(char)) {
      o << any_cast<char>(any);
   } else if (any.TypeInfo() == typeid(int)) {
      o << any_cast<int>(any);
   } else if (any.TypeInfo() == typeid(short)) {
      o << any_cast<short>(any);
   } else if (any.TypeInfo() == typeid(long)) {
      o << any_cast<long>(any);
   } else if (any.TypeInfo() == typeid(float)) {
      o << any_cast<float>(any);
   } else if (any.TypeInfo() == typeid(double)) {
      o << any_cast<double>(any);
   } else if (any.TypeInfo() == typeid(const char*)) {
      o << any_cast<const char*>(any);
   } else if (any.TypeInfo() == typeid(std::string)) {
      o << any_cast<std::string>(any);
   } else { o << "Any object at " << std::hex << &static_cast<Any::Holder<int>*>(any.fContent)->fHeld << std::dec; }
   return o;
} // <<
