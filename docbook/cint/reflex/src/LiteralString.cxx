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

#include "Reflex/internal/LiteralString.h"

#include <cstdlib>

namespace {
   class LiteralStringSet {
   public:
      static LiteralStringSet& Instance();
      void Add(const char* s) { fLiterals.insert((void*)s); }
      void Remove(const char* s) { fLiterals.erase((void*)s); }
      bool IsLiteral(const char* s) { return fLiterals.find(s) != fLiterals.end();}
   private:
      LiteralStringSet() {}

      std::set<const void*> fLiterals;
   };

   //-------------------------------------------------------------------------------
   LiteralStringSet&
   LiteralStringSet::Instance() {
      //-------------------------------------------------------------------------------
      // Return static instance of LiteralStringSet
      static LiteralStringSet s;
      return s;
   }
}

//-------------------------------------------------------------------------------
void
Reflex::LiteralString::Add(const char* s) {
//-------------------------------------------------------------------------------
// Add s to set of string literals
   LiteralStringSet::Instance().Add(s);
}

//-------------------------------------------------------------------------------
void
Reflex::LiteralString::Remove(const char* s) {
//-------------------------------------------------------------------------------
// Add s to set of string literals
   LiteralStringSet::Instance().Remove(s);
}

//-------------------------------------------------------------------------------
Reflex::LiteralString::LiteralString(const char* s):
   fLiteral(s), fAllocSize(0) {
//-------------------------------------------------------------------------------
// Create a LiteralString
   if (!LiteralStringSet::Instance().IsLiteral(s))
      StrDup(s);
}

//-------------------------------------------------------------------------------
Reflex::LiteralString::LiteralString(const LiteralString& other):
   fLiteral(other.fLiteral), fAllocSize(0) {
//-------------------------------------------------------------------------------
// Copy construct a LiteralString
   if (other.fAllocSize)
      StrDup(other.fBuf);
}

//-------------------------------------------------------------------------------
Reflex::LiteralString&
Reflex::LiteralString::operator=(const LiteralString& other) {
//-------------------------------------------------------------------------------
// Assign a LiteralString
   this->~LiteralString();

   fLiteral = other.fLiteral;
   fAllocSize = 0;
   if (other.fAllocSize)
      StrDup(other.fBuf);
   return *this;
}

//-------------------------------------------------------------------------------
Reflex::LiteralString::~LiteralString() {
//-------------------------------------------------------------------------------
// Destruct a LiteralString
   if (fAllocSize)
      free(fBuf);
}

//-------------------------------------------------------------------------------
void
Reflex::LiteralString::Reserve(size_t size) {
//-------------------------------------------------------------------------------
// Force the literal to be on the heap, capable of storing at least "size"
// characters.

   if (fAllocSize < size) {
      // need to (re)allocate

      if (fAllocSize) {
         fBuf = (char*)realloc(fBuf, size);
      } else {
         char* buf = (char*)malloc(size);
         memcpy(buf, fLiteral, strlen(fLiteral) + 1);
         fBuf = buf;
      }
      fAllocSize = size;
   }
}

//-------------------------------------------------------------------------------
void
Reflex::LiteralString::StrDup(const char* s) {
//-------------------------------------------------------------------------------
// Create a heap representation of "s".

   size_t len = strlen(s);
   Reserve(len + 1);
   strncpy(fBuf, s, len + 1);
}

//-------------------------------------------------------------------------------
Reflex::LiteralString&
Reflex::LiteralString::operator +=(const char* s) {
//-------------------------------------------------------------------------------
// Add a string, moving this to the heap.
   size_t len = strlen(s);
   Reserve(len +  (fLiteral ? strlen(fLiteral) : 0) + 1);
   strncat(fBuf, s, len);
   return *this;
}

//-------------------------------------------------------------------------------
Reflex::LiteralString&
Reflex::LiteralString::operator +=(const std::string& s) {
//-------------------------------------------------------------------------------
// Add a string, moving this to the heap.
   size_t len = s.length();
   Reserve(len +  (fLiteral ? strlen(fLiteral) : 0) + 1);
   strncat(fBuf, s.c_str(), len);
   return *this;
}

//-------------------------------------------------------------------------------
Reflex::LiteralString&
Reflex::LiteralString::operator +=(const LiteralString& s) {
//-------------------------------------------------------------------------------
// Add a string literal, moving this to the heap.
   size_t len = s.length();
   Reserve(len +  (fLiteral ? strlen(fLiteral) : 0) + 1);
   strncat(fBuf, s.c_str(), len);
   return *this;
}

//-------------------------------------------------------------------------------
void
Reflex::LiteralString::erase(size_t i) {
//-------------------------------------------------------------------------------
// Erase index i to end
   if (!fAllocSize) {
      const char* literal = fLiteral;
      Reserve(i + 1);
      memcpy(fBuf, literal, i);
   }
   fBuf[i] = 0;
}


//-------------------------------------------------------------------------------
void
Reflex::LiteralString::ToHeap() {
//-------------------------------------------------------------------------------
// Move the string literal to the heap; no-op if not a string literal.
   if (!fAllocSize) {
      const char* literal = fLiteral;
      size_t len = strlen(literal);
      Reserve(len + 1);
      memcpy(fBuf, literal, len + 1);
   }
}
