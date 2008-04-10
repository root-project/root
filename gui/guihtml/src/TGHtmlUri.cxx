// $Id$
// Author:  Valeriy Onuchin   03/05/2007

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun, Fons Rademakers and Reiner Rohlfs *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**************************************************************************

    HTML widget for xclass. Based on tkhtml 1.28
    Copyright (C) 1997-2000 D. Richard Hipp <drh@acm.org>
    Copyright (C) 2002-2003 Hector Peraza.

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public
    License along with this library; if not, write to the Free
    Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

**************************************************************************/

// Routines for processing URLs.

#include <ctype.h>
#include <string.h>
#include <stdlib.h>

#include "TGHtml.h"
#include "TGHtmlUri.h"


static char *StrNDup(const char *z, int n);
static void ReplaceStr(char **pzDest, const char *zSrc);
#if 0  // not used
static char *Trim(char *z);
#endif


//______________________________________________________________________________
TGHtmlUri::TGHtmlUri(const char *zUri) 
{
   // Parse a text URI into an HtmlUri structure.
   
   int n;

   zScheme = zAuthority = zPath = zQuery = zFragment = (char *) 0;

   if (zUri && *zUri) {
      while (isspace(*zUri)) ++zUri;
      n = ComponentLength(zUri, "", ":/?# ");
      if (n > 0 && zUri[n] == ':') {
         zScheme = StrNDup(zUri, n);
         zUri += n+1;
      }
      n = ComponentLength(zUri, "//", "/?# ");
      if (n > 0) {
         zAuthority = StrNDup(&zUri[2], n-2);
         zUri += n;
      }
      n = ComponentLength(zUri, "", "?# ");
      if (n > 0) {
         zPath = StrNDup(zUri, n);
         zUri += n;
      }
      n = ComponentLength(zUri, "?", "# ");
      if (n > 0) {
         zQuery = StrNDup(&zUri[1], n-1);
         zUri += n;
      }
      n = ComponentLength(zUri, "#", " ");
      if (n > 0) {
         zFragment = StrNDup(&zUri[1], n-1);
      }
   }
}

//______________________________________________________________________________
TGHtmlUri::TGHtmlUri(const TGHtmlUri *uri)
{

   zScheme = zAuthority = zPath = zQuery = zFragment = (char *) 0;

   if (uri) {
      if (uri->zScheme)    zScheme    = StrDup(uri->zScheme);
      if (uri->zAuthority) zAuthority = StrDup(uri->zAuthority);
      if (uri->zPath)      zPath      = StrDup(uri->zPath);
      if (uri->zQuery)     zQuery     = StrDup(uri->zQuery);
      if (uri->zFragment)  zFragment  = StrDup(uri->zFragment);
   }
}

//______________________________________________________________________________
TGHtmlUri::~TGHtmlUri()
{

   if (zScheme) delete[] zScheme;
   if (zAuthority) delete[] zAuthority;
   if (zPath) delete[] zPath;
   if (zQuery) delete[] zQuery;
   if (zFragment) delete[] zFragment;
}

//______________________________________________________________________________
int TGHtmlUri::EqualsUri(const TGHtmlUri *uri, int field_mask)
{

   if (!uri) return 0;

   if (field_mask & URI_SCHEME_MASK) {
      if (uri->zScheme && zScheme) {
         if (strcmp(uri->zScheme, zScheme) != 0) return 0;
      } else if (uri->zScheme != zScheme) {  // one of them null?
         return 0;
      }
   }

   if (field_mask & URI_AUTH_MASK) {
      if (uri->zAuthority && zAuthority) {
         if (strcmp(uri->zAuthority, zAuthority) != 0) return 0;
      } else if (uri->zAuthority != zAuthority) {
         return 0;
      }
   }

   if (field_mask & URI_PATH_MASK) {
      if (uri->zPath && zPath) {
         if (strcmp(uri->zPath, zPath) != 0) return 0;
      } else if (uri->zPath != zPath) {
         return 0;
      }
   }

   if (field_mask & URI_QUERY_MASK) {
      if (uri->zQuery && zQuery) {
         if (strcmp(uri->zQuery, zQuery) != 0) return 0;
      } else if (uri->zQuery != zQuery) {
         return 0;
      }
   }

   if (field_mask & URI_FRAGMENT_MASK) {
      if (uri->zFragment && zFragment) {
         if (strcmp(uri->zFragment, zFragment) != 0) return 0;
      } else if (uri->zFragment != zFragment) {
         return 0;
      }
   }

   return 1;
}

//______________________________________________________________________________
int TGHtmlUri::ComponentLength(const char *z, const char *zInit, const char *zTerm)
{
   // Return the length of the next component of the URL in z[] given
   // that the component starts at z[0].  The initial sequence of the
   // component must be zInit[].  The component is terminated by any
   // character in zTerm[].  The length returned is 0 if the component
   // doesn't exist.  The length includes the zInit[] string, but not
   // the termination character.
   //
   //        Component        zInit      zTerm
   //        ----------       -------    -------
   //        scheme           ""         ":/?#"
   //        authority        "//"       "/?#"
   //        path             "/"        "?#"
   //        query            "?"        "#"
   //        fragment         "#"        ""
   int i, n;

   for (n = 0; zInit[n]; ++n) {
      if (zInit[n] != z[n]) return 0;
   }
   while (z[n]) {
      for (i = 0; zTerm[i]; ++i) {
         if (z[n] == zTerm[i]) return n;
      }
      ++n;
   }

   return n;
}

//______________________________________________________________________________
char *TGHtmlUri::BuildUri()
{
   // Create a string to hold the given URI. Memory to hold the string is
   // allocated with new[] and must be freed by the calling function.
  
   int n = 1;
   char *z;

   if (zScheme)    n += strlen(zScheme) + 1;
   if (zAuthority) n += strlen(zAuthority) + 3;
   if (zPath)      n += strlen(zPath) + 1;
   if (zQuery)     n += strlen(zQuery) + 1;
   if (zFragment)  n += strlen(zFragment) + 1;
   z = new char[n];
   if (z == 0) return 0;
   n = 0;
   if (zScheme) {
      sprintf(z, "%s:", zScheme);
      n = strlen(z);
   }
   if (zAuthority) {
      sprintf(&z[n], "//%s", zAuthority);
      n += strlen(&z[n]);
   }
   if (zAuthority && zAuthority[strlen(zAuthority)-1] != '/' &&
      !(zPath && zPath[0] == '/')) {
      strcat(z, "/");
      ++n;
   }
   if (zPath) {
      sprintf(&z[n], "%s", zPath);
      n += strlen(&z[n]);
   }
   if (zQuery) {
      sprintf(&z[n], "?%s", zQuery);
      n += strlen(&z[n]);
   }
   if (zFragment) {
      sprintf(&z[n], "#%s", zFragment);
   } else {
      z[n] = 0;
   }

   return z;
}


//______________________________________________________________________________
static char *StrNDup(const char *z, int n) 
{

   // Duplicate a string of length n.
   char *zResult;

   if (n <= 0) n = strlen(z);
   zResult = new char[n + 1];
   if (zResult) {
      memcpy(zResult, z, n);
      zResult[n] = 0;
   }
   return zResult;
}

//______________________________________________________________________________
static void ReplaceStr(char **pzDest, const char *zSrc)
{
   // Replace the string in *pzDest with the string in zSrc
   
   if (*pzDest != 0) delete[] *pzDest;
   if (zSrc == 0) {
      *pzDest = 0;
   } else {
      *pzDest = StrNDup(zSrc, -1);
   }
}

#if 0  // not used
static char *Trim(char *z) 
{
   // Remove leading and trailing spaces from the given string. Return
   // a new string allocated with new[].
   int i;
   char *zNew;

   while (isspace(*z)) z++;
   i = strlen(z);
   zNew = new char[i + 1];
   if (zNew == 0) return 0;
   strcpy(zNew, z);
   while (i > 0 && isspace(zNew[i-1])) zNew[--i] = 0;

   return zNew;
}
#endif

//______________________________________________________________________________
char *TGHtml::ResolveUri(const char *zUri)
{
   // This function resolves the specified URI and returns the result in
   // a newly allocated string. The resolver algorithm specified in section
   // 5.2 of RFC 2396 is used.

   char *result = 0;
   TGHtmlUri *base, *term;

   if (zUri == 0 || *zUri == 0) return 0;

   if (zBaseHref && *zBaseHref) {
      base = new TGHtmlUri(zBaseHref);
   } else {
      base = new TGHtmlUri(zBase);
   }

   term = new TGHtmlUri(zUri);

   if (term->zScheme == 0 &&
       term->zAuthority == 0 &&
       term->zPath == 0 &&
       term->zQuery == 0 &&
       term->zFragment) {
      ReplaceStr(&base->zFragment, term->zFragment);
   } else if (term->zScheme) {
      TGHtmlUri *temp;
      temp = term;
      term = base;
      base = temp;
   } else if (term->zAuthority) {
      ReplaceStr(&base->zAuthority, term->zAuthority);
      ReplaceStr(&base->zPath, term->zPath);
      ReplaceStr(&base->zQuery, term->zQuery);
      ReplaceStr(&base->zFragment, term->zFragment);
   } else if (term->zPath && (term->zPath[0] == '/' || base->zPath == 0)) {
      ReplaceStr(&base->zPath, term->zPath);
      ReplaceStr(&base->zQuery, term->zQuery);
      ReplaceStr(&base->zFragment, term->zFragment);
   } else if (term->zPath && base->zPath) {
      char *zBuf;
      int i, j;
      zBuf = new char[strlen(base->zPath) + strlen(term->zPath) + 2];
      if (zBuf) {
         sprintf(zBuf, "%s", base->zPath);
         for (i = strlen(zBuf) - 1; i >= 0 && zBuf[i] != '/'; --i) {
            zBuf[i] = 0;
         }
         strcat(zBuf, term->zPath);
         for (i = 0; zBuf[i]; i++) {
            if (zBuf[i] == '/' && zBuf[i+1] == '.' && zBuf[i+2] == '/') {
               strcpy(&zBuf[i+1], &zBuf[i+3]);
               --i;
               continue;
            }
            if (zBuf[i] == '/' && zBuf[i+1] == '.' && zBuf[i+2] == 0) {
               zBuf[i+1] = 0;
               continue;
            }
            if (i > 0 && zBuf[i] == '/' && zBuf[i+1] == '.' && 
                zBuf[i+2] == '.' && (zBuf[i+3] == '/' || zBuf[i+3] == 0)) {
               for (j = i - 1; j >= 0 && zBuf[j] != '/'; --j) {}
               if (zBuf[i+3]) {
                  strcpy(&zBuf[j+1], &zBuf[i+4]);
               } else {
                  zBuf[j+1] = 0;
               }
               i = j - 1;
               if (i < -1) i = -1;
               continue;
            }
         }
         delete[] base->zPath;
         base->zPath = zBuf;
      }
      ReplaceStr(&base->zQuery, term->zQuery);
      ReplaceStr(&base->zFragment, term->zFragment);
   }
   delete term;
  
   result = base->BuildUri();
   delete base;

   return result;
}
