// @(#)root/cintex:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Cintex_CINTdefs
#define ROOT_Cintex_CINTdefs

#include "Reflex/Type.h"

#include <utility>
#include <stdexcept>
#include <iostream>
#include <map>


namespace ROOT {
namespace Cintex {

typedef std::pair<char, std::string> CintTypeDesc;

ROOT::Reflex::Type CleanType(const ROOT::Reflex::Type& t);
CintTypeDesc CintType(const ROOT::Reflex::Type&);
void CintType(const ROOT::Reflex::Type&, int& typenum, int& tagnum);
std::string CintName(const std::string&);
std::string CintName(const ROOT::Reflex::Type&);
int CintTag(const std::string&);
bool IsSTLinternal(const std::string& nam);
bool IsSTL(const std::string& nam);
bool IsSTLext(const std::string& nam);

} // namespace Cintex
} // namespace ROOT
#endif // ROOT_Cintex_CINTdefs
