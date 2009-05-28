// @(#)root/cintex:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "CINTdefs.h"

#include "Reflex/Reflex.h"
#include "Reflex/Type.h"
#include "Api.h"

#include <string>

using namespace ROOT::Reflex;
using namespace std;

namespace ROOT {
namespace Cintex {

Type CleanType(const Type& typ)
{
   if (!typ) {
      return typ;
   }
   Type t = typ;
   for (; t.IsTypedef(); t = CleanType(t.ToType())) {}
   for (; t.IsPointer(); t = CleanType(t.ToType())) {}
   for (; t.IsArray(); t = CleanType(t.ToType())) {}
   return t;
}

CintTypeDesc CintType(const ROOT::Reflex::Type& typ)
{
   Type t = CleanType(typ);
   string nam = t.Name(SCOPED);
   if (nam == "void") {
      return CintTypeDesc('y', "-");
   }
   if (nam == "bool") {
      return CintTypeDesc('g', "-");
   }
   if (nam == "char") {
      return CintTypeDesc('c', "-");
   }
   if (nam == "signed char") {
      return CintTypeDesc('c', "-");
   }
   if (nam == "unsigned char") {
      return CintTypeDesc('b', "-");
   }
   if (nam == "short") {
      return CintTypeDesc('s', "-");
   }
   if (nam == "short int") {
      return CintTypeDesc('s', "-");
   }
   if (nam == "signed short") {
      return CintTypeDesc('s', "-");
   }
   if (nam == "signed short int") {
      return CintTypeDesc('s', "-");
   }
   if (nam == "unsigned short") {
      return CintTypeDesc('r', "-");
   }
   if (nam == "short unsigned") {
      return CintTypeDesc('r', "-");
   }
   if (nam == "unsigned short int") {
      return CintTypeDesc('r', "-");
   }
   if (nam == "short unsigned int") {
      return CintTypeDesc('r', "-");
   }
   if (nam == "int") {
      return CintTypeDesc('i', "-");
   }
   if (nam == "signed") {
      return CintTypeDesc('i', "-");
   }
   if (nam == "signed int") {
      return CintTypeDesc('i', "-");
   }
   if (nam == "unsigned int") {
      return CintTypeDesc('h', "-");
   }
   if (nam == "unsigned") {
      return CintTypeDesc('h', "-");
   }
   if (nam == "long") {
      return CintTypeDesc('l', "-");
   }
   if (nam == "long int") {
      return CintTypeDesc('l', "-");
   }
   if (nam == "signed long") {
      return CintTypeDesc('l', "-");
   }
   if (nam == "signed long int") {
      return CintTypeDesc('l', "-");
   }
   if (nam == "long signed") {
      return CintTypeDesc('l', "-");
   }
   if (nam == "long signed int") {
      return CintTypeDesc('l', "-");
   }
   if (nam == "unsigned long") {
      return CintTypeDesc('k', "-");
   }
   if (nam == "unsigned long int") {
      return CintTypeDesc('k', "-");
   }
   if (nam == "long unsigned int") {
      return CintTypeDesc('k', "-");
   }
   if (nam == "longlong") {
      return CintTypeDesc('n', "-");
   }
   if (nam == "long long") {
      return CintTypeDesc('n', "-");
   }
   if (nam == "long long int") {
      return CintTypeDesc('n', "-");
   }
   if (nam == "long long signed int") {
      return CintTypeDesc('n', "-");
   }
   if (nam == "long long signed") {
      return CintTypeDesc('n', "-");
   }
   if (nam == "signed long long int") {
      return CintTypeDesc('n', "-");
   }
   if (nam == "ulonglong") {
      return CintTypeDesc('m', "-");
   }
   if (nam == "unsigned long long") {
      return CintTypeDesc('m', "-");
   }
   if (nam == "long long unsigned") {
      return CintTypeDesc('m', "-");
   }
   if (nam == "long long unsigned int") {
      return CintTypeDesc('m', "-");
   }
   if (nam == "unsigned long long int") {
      return CintTypeDesc('m', "-");
   }
   if (nam == "long double") {
      return CintTypeDesc('q', "-");
   }
   if (nam == "double") {
      return CintTypeDesc('d', "-");
   }
   if (nam == "double32") {
      return CintTypeDesc('d', "-");
   }
   if (nam == "float") {
      return CintTypeDesc('f', "-");
   }
   if (t.IsEnum()) {
      return CintTypeDesc('i', "-");
   }
   if (t.IsFunction()) {
      return CintTypeDesc('y', "-");
   }
   if (!t.IsFundamental()) {
      return CintTypeDesc('u', CintName(t));
   }
   return CintTypeDesc('-', CintName(t));
}

void CintType(const Type& typ, int& typenum, int& tagnum)
{
   Type t(typ);
   int indir = 0;
   for (; t.IsTypedef(); t = t.ToType()) {}
   for (; t.IsPointer(); t = t.ToType()) {
      ++indir;
   }
   CintTypeDesc dsc = CintType(t);
   typenum  = dsc.first + ((indir > 0) ? ('A' - 'a') : 0);
   tagnum = -1;
   if (dsc.first == 'u') {
      tagnum = G__defined_tagname(dsc.second.c_str(), 2);
      if (tagnum == -1) {
         G__linked_taginfo taginfo;
         taginfo.tagnum = -1;
         if (t.IsClass() || t.IsStruct()) {
            taginfo.tagtype = 'c';
         }
         else {
            taginfo.tagtype = 'a';
         }
         taginfo.tagname = dsc.second.c_str();
         G__get_linked_tagnum(&taginfo);
         tagnum = taginfo.tagnum;
      }
   }
}

// Typename normalization table
// Note: order is important!
static const char* s_normalize[][2] =  {
     {"  ", " " }
   , {", ", "," }
   , {" signed ", " " }
   , {",signed ", "," }
   , {"<signed ", "<" }
   , {"(signed ", "(" }
   , {"ulonglong", "unsigned long long" }
   , {"longlong", "long long" }
   , {"long unsigned", "unsigned long" }
   , {"short unsigned", "unsigned short" }
   , {"short int", "short" }
   , {"long int", "long" }
   , {"basic_string<char> ", "string"}
   , {"basic_string<char>", "string"}
   , {"basic_string<char,allocator<char> > ", "string"}
   , {"basic_string<char,allocator<char> >", "string"}
   , {"basic_string<char,char_traits<char>,allocator<char> > ", "string"}
   , {"basic_string<char,char_traits<char>,allocator<char> >", "string"}
};

std::string CintName(const std::string& full_nam)
{
   size_t occ = 0;
   std::string s = (full_nam.substr(0, 2) == "::") ? full_nam.substr(2) : full_nam;
   // Completely ignore namespace "std::"
   occ = s.find("std::");
   while (occ != std::string::npos) {
      s.replace(occ, 5, "");
      occ = s.find("std::");
   }
   // Remove spaces after commas.
   occ = s.find(", ");
   while (occ != std::string::npos) {
      s.replace(occ, 2, ",");
      occ = s.find(", ");
   }
   // Transform "* const" to "*const".
   occ = s.find("* const");
   while (occ != std::string::npos)    {
      if (!isalnum(s[occ+7])) {
         s.replace(occ, 7, "*const");
      }
      occ = s.find("* const");
   }
   // Transform "& const" to "&const".
   occ = s.find("& const");
   while (occ != std::string::npos)    {
      if (!isalnum(s[occ+7])) {
         s.replace(occ, 7, "&const");
      }
      occ = s.find("& const");
   }
   //
   // Perform naming normalization for primitives
   // since GCC-XML just generates anything.
   //
   for (size_t i = 0; i < sizeof(s_normalize) / sizeof(s_normalize[0]); ++i) {
      occ = s.find(s_normalize[i][0]);
      while (occ != std::string::npos)    {
         s.replace(occ, strlen(s_normalize[i][0]), s_normalize[i][1]);
         occ = s.find(s_normalize[i][0]);
      }
   }
   //
   //  Remove any array dimensions.
   //
   if (s.find('[') != std::string::npos) {
      s = s.substr(0, s.find('['));
   }
   return s;
}

std::string CintName(const Type& typ)
{
   Type t = CleanType(typ);
   return CintName(t.Name(SCOPED));
}

int CintTag(const std::string& Name)
{
   std::string n = CintName(Name);
   if (n == "-") {
      return -1;
   }
   return G__search_tagname(n.c_str(), 'c');
}

bool IsSTLinternal(const std::string& nam)
{
   if (
      nam.empty() ||
      (nam.substr(0, 6) == "std::_") ||
      (nam.substr(0, 9) == "stdext::_") ||
      (nam.substr(0, 12) == "__gnu_cxx::_")
   ) {
      return true;
   }
   return false;
}

bool IsSTL(const std::string& nam)
{
   if (IsSTLinternal(nam))  {
      return true;
   }
   std::string sub = nam.substr(0, 8);
   bool stl = 
      (nam.substr(0, 17) == "std::basic_string") ||
      (sub == "std::str") ||
      (sub == "std::vec") ||
      (sub == "std::lis") ||
      (sub == "std::set") ||
      (sub == "std::deq") ||
      (sub == "std::map") ||
      (sub == "std::mul") ||
      (sub == "stdext::") ||
      (sub == "__gnu_cx");
   return stl;
}

bool IsSTLext(const std::string& nam)
{
   std::string sub = nam.substr(0, 8);
   return (sub == "stdext::") || (sub == "__gnu_cx");
}

} // namespace Cintex
} // namespace ROOT
