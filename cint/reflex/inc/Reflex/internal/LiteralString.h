// @(#)root/reflex:$Id$
// Author: Axel Naumann, 2010

// Copyright CERN, CH-1211 Geneva 23, 2004-2010, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_LiteralString
#define Reflex_LiteralString

#include <set>
#include <cstring>
#include <string>
#include "Reflex/Kernel.h"

namespace Reflex {
   class RFLX_API LiteralString {
   public:
      LiteralString(): fLiteral(0), fAllocSize(0) {}
      LiteralString(const char* s);
      LiteralString(const LiteralString& other);

      ~LiteralString();

      static void Add(const char* s);
      static void Remove(const char* s);

      LiteralString& operator=(const LiteralString& other);

      const char* c_str() const { return fLiteral; }
      const char** key() const { return (const char**) &fLiteral; }
      size_t length() const { return strlen(fLiteral); }
      void erase(size_t i);

      void ToHeap();
      bool IsLiteral() const { return !fAllocSize; }

      bool operator<(const LiteralString& other) const {
         return strcmp(fLiteral, other.fLiteral) < 0; }
      bool operator>(const LiteralString& other) const {
         return strcmp(fLiteral, other.fLiteral) > 0; }
      bool operator==(const LiteralString& other) const {
         return strcmp(fLiteral, other.fLiteral) == 0; }
      bool operator!=(const LiteralString& other) const {
         return strcmp(fLiteral, other.fLiteral) != 0; }

      bool operator<(const char* other) const {
         return strcmp(fLiteral, other) < 0; }
      bool operator>(const char* other) const {
         return strcmp(fLiteral, other) > 0; }
      bool operator==(const char* other) const {
         return strcmp(fLiteral, other) == 0; }
      bool operator!=(const char* other) const {
         return strcmp(fLiteral, other) != 0; }

      bool operator<(const std::string& other) const {
         return other.compare(fLiteral) < 0; }
      bool operator>(const std::string& other) const {
         return other.compare(fLiteral) > 0; }
      bool operator==(const std::string& other) const {
         return other.compare(fLiteral) == 0; }
      bool operator!=(const std::string& other) const {
         return other.compare(fLiteral) != 0; }
      LiteralString& operator+=(const LiteralString& other);
      LiteralString& operator+=(const std::string& other);
      LiteralString& operator+=(const char* other);
      char operator[](size_t i) const { return fBuf[i]; }

      // operator const std::string&()

   private:
      void Reserve(size_t size);
      void StrDup(const char* s);
      void StrCat(const char* s);

      union {
         const char* fLiteral;
         char* fBuf;
      };
      size_t fAllocSize;
   };
}

#endif // Reflex_LiteralString
