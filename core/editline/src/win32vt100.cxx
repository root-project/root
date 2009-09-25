// @(#)root/editline:$Id$
// Author: Axel Naumann, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "win32vt100.h"
#include <set>
#include <cstring>
#include <stdarg.h>
#include <stdio.h>
#include <strings.h>

char*
tigetstr(const char* cap) {
   // We support reset, set foreground, set bold, reset to default.
   // The escape sequence is \027 plus the capability string
   // plus ",parameter" for setf.
   static const char opts[] = "\027oc\000\027setaf%d\000\027bold\000\027sgr0\000";
   const char* o = opts;
   const char* p = 0;

   while (*o && !(p = strstr(o + 1, cap))) {
      o += strlen(o) + 1;
   }

   if (*o && p) {
      return (char*) p - 1;
   }
   return 0;
} // tigetstr


typedef int (*PutcFunc_t)(int);
int
tputs(const char* what, int, PutcFunc_t myputc) {
   if (!what || !(*what)) {
      return 1;
   }
   const char* c = what;

   while (*c)
      (*myputc)(*(c++));
   return c - what;
}


char*
tparm(const char* termstr, ...) {
   if (termstr[0] != '\027') {
      return 0;
   }

   switch (termstr[2]) {
   case 'c': {
      // SetTermToNormal
      break;
   }
   case 'e': {
      // Set color
      va_list vl;
      va_start(vl, termstr);

      va_arg(vl, int);
      va_end(vl);
      // set color i
      break;
   }
   case 'o': {
      // bold
      break;
   }
   case 'g': {
      // reset
      break;
   }
   } // switch
   return (char*) "";
} // tparm


int
setupterm(const char* /*term*/, int /*fd*/, int* errcode) {
   if (errcode) {
      *errcode = 0;
   }
   return !ERR;
}


int
win32vt100_putc(int c) {
   return putc(c, stdout);
}
