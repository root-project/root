// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////
// This file is part of the liblineedit code. See el.fH for the
// full license (BSD).
////////////////////////////////////////////////////////////////////////

#include <string.h>

size_t
el_strlcpy(char* dst, const char* src, size_t size) {
   if (size) {
      strncpy(dst, src, size - 1);
      dst[size - 1] = '\0';
   } else {
      dst[0] = '\0';
   }
   return strlen(src);
}


size_t
el_strlcat(char* dst, const char* src, size_t size) {
   int dl = strlen(dst);
   int sz = size - dl - 1;

   if (sz >= 0) {
      strncat(dst, src, sz);
      dst[sz] = '\0';
   }

   return dl + strlen(src);
}
