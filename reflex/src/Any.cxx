// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Any.h"
#include <string>
#include <iostream>

using namespace ROOT::Reflex;

//-------------------------------------------------------------------------------
std::ostream & ROOT::Reflex::operator << ( std::ostream& o, 
                                           const Any& any) {
//-------------------------------------------------------------------------------
  if      ( any.TypeGet() == typeid(char) )   o << any_cast<char>(any);
  else if ( any.TypeGet() == typeid(int) )    o << any_cast<int>(any);
  else if ( any.TypeGet() == typeid(short) )  o << any_cast<short>(any);
  else if ( any.TypeGet() == typeid(long) )   o << any_cast<long>(any);
  else if ( any.TypeGet() == typeid(float) )  o << any_cast<float>(any);
  else if ( any.TypeGet() == typeid(double) ) o << any_cast<double>(any);
  else if ( any.TypeGet() == typeid(const char*) ) o << any_cast<const char*>(any);
  else if ( any.TypeGet() == typeid(std::string) ) o << any_cast<std::string>(any); 
  else o << "Any object at " << std::hex << &static_cast<Any::holder<int>*>(any.content)->held;
  return o;
}
