/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef VSTRING_H
#define VSTRING_H

#include "VType.h"

class VString
{
   friend int strcmp(const VString& a, const char* b);
   friend int strcmp(const char* b, const VString& a);

public:

   VString()
   : len(0)
   , str(0)
   {
   }

   VString(const char* strIn);
   VString(const VString& kstrIn);
   VString& operator=(const VString& obj);
   VString& operator=(const char* s);

   ~VString()
   {
      delete[] str;
   }

   int operator==(const VString& x)
   {
      if (
         (!len && !x.len) ||
         ((len == x.len) && !strcmp(str, x.str))
      ) {
         return MATCH;
      }
      return UNMATCH;
   }

   int Write(FILE* fp);
   int Read(FILE* fp);

   void append(const VString& s);
   void append(const char* s);

   int Length()
   {
      return len;
   }

   const char* String()
   {
      if (str) {
         return str;
      }
      return "";
   }

private:
   int len;
   char* str;
};

#endif // VSTRING_H
