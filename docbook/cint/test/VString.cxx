/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "VString.h"

using namespace std;

VString::VString(const char* strIn)
: len(0)
, str(0)
{
   if (strIn && strIn[0]) {
      len = strlen(strIn);
      str = new char[len+1];
      strcpy(str, strIn);
   }
}

VString::VString(const VString& kstrIn)
: len(0)
, str(0)
{
   if (kstrIn.str) {
      len = kstrIn.len;
      str = new char[len+1];
      strcpy(str, kstrIn.str);
   }
}

VString& VString::operator=(const VString& obj)
{
   if (this != &obj) {
      len = 0;
      delete[] str;
      str = 0;
      if (obj.str) {
         len = obj.len;
         str = new char[len+1];
         strcpy(str, obj.str);
      }
   }
   return *this;
}

VString& VString::operator=(const char* s)
{
   len = 0;
   delete[] str;
   str = 0;
   if (s && s[0]) {
      len = strlen(s);
      str = new char[len+1];
      strcpy(str, s);
   }
   return *this;
}

void VString::append(const VString& s)
{
   if (!s.len) {
      return;
   }
   append(s.str);
}

void VString::append(const char* s)
{
   if (!s) {
      return;
   }
   if (str) {
      len = len + strlen(s);
      char* p = new char[len+1];
      sprintf(p, "%s%s", str, s);
      delete[] str;
      str = p;
   }
   else {
      *this = s;
   }
}

int VString::Write(FILE* fp)
{
   fwrite(&len, sizeof(len), 1, fp);
   if (len) {
      fwrite(str, len + 1, 1, fp);
   }
   return SUCCESS;
}

int VString::Read(FILE* fp)
{
   len = 0;
   delete[] str;
   str = 0;
   fread(&len, sizeof(len), 1, fp);
   if (len) {
      str = new char[len+1];
      fread(str, len + 1, 1, fp);
   }
   return SUCCESS;
}

int Debug = 0;

int strcmp(const VString& a, const char* b)
{
   if (!a.len && !strlen(b)) {
      return 0;
   }
   else if (a.len && strlen(b)) {
      return strcmp(a.str, b);
   }
   return 1;
}

int strcmp(const char* b, const VString& a)
{
   if (!a.len && !strlen(b)) {
      return 0;
   }
   else if (a.len && strlen(b)) {
      return strcmp(a.str, b);
   }
   return 1;
}

