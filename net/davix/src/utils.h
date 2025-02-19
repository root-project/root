/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <string>

inline bool isno(const char *str)
{
   if (!str)
      return false;

   if (!strcmp(str, "n") || !strcmp(str, "no") || !strcmp(str, "0") || !strcmp(str, "false"))
      return true;

   return false;
}
