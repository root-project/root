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


////////////////////////////////////////////////////////////////////////////////
/// Parse a text URI into an HtmlUri structure.

TGHtmlUri::TGHtmlUri(const char *zUri)
{
   int n;

   fZScheme = fZAuthority = fZPath = fZQuery = fZFragment = (char *) 0;

   if (zUri && *zUri) {
      while (isspace(*zUri)) ++zUri;
      n = ComponentLength(zUri, "", ":/?# ");
      if (n > 0 && zUri[n] == ':') {
         fZScheme = StrNDup(zUri, n);
         zUri += n+1;
      }
      n = ComponentLength(zUri, "//", "/?# ");
      if (n > 0) {
         fZAuthority = StrNDup(&zUri[2], n-2);
         zUri += n;
      }
      n = ComponentLength(zUri, "", "?# ");
      if (n > 0) {
         fZPath = StrNDup(zUri, n);
         zUri += n;
      }
      n = ComponentLength(zUri, "?", "# ");
      if (n > 0) {
         fZQuery = StrNDup(&zUri[1], n-1);
         zUri += n;
      }
      n = ComponentLength(zUri, "#", " ");
      if (n > 0) {
         fZFragment = StrNDup(&zUri[1], n-1);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Html uri copy constructor.

TGHtmlUri::TGHtmlUri(const TGHtmlUri *uri)
{
   fZScheme = fZAuthority = fZPath = fZQuery = fZFragment = (char *) 0;

   if (uri) {
      if (uri->fZScheme)    fZScheme    = StrDup(uri->fZScheme);
      if (uri->fZAuthority) fZAuthority = StrDup(uri->fZAuthority);
      if (uri->fZPath)      fZPath      = StrDup(uri->fZPath);
      if (uri->fZQuery)     fZQuery     = StrDup(uri->fZQuery);
      if (uri->fZFragment)  fZFragment  = StrDup(uri->fZFragment);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Html uri destructor.

TGHtmlUri::~TGHtmlUri()
{
   if (fZScheme) delete[] fZScheme;
   if (fZAuthority) delete[] fZAuthority;
   if (fZPath) delete[] fZPath;
   if (fZQuery) delete[] fZQuery;
   if (fZFragment) delete[] fZFragment;
}

////////////////////////////////////////////////////////////////////////////////
/// Compare another uri with given field mask.

int TGHtmlUri::EqualsUri(const TGHtmlUri *uri, int field_mask)
{
   if (!uri) return 0;

   if (field_mask & URI_SCHEME_MASK) {
      if (uri->fZScheme && fZScheme) {
         if (strcmp(uri->fZScheme, fZScheme) != 0) return 0;
      } else if (uri->fZScheme != fZScheme) {  // one of them null?
         return 0;
      }
   }

   if (field_mask & URI_AUTH_MASK) {
      if (uri->fZAuthority && fZAuthority) {
         if (strcmp(uri->fZAuthority, fZAuthority) != 0) return 0;
      } else if (uri->fZAuthority != fZAuthority) {
         return 0;
      }
   }

   if (field_mask & URI_PATH_MASK) {
      if (uri->fZPath && fZPath) {
         if (strcmp(uri->fZPath, fZPath) != 0) return 0;
      } else if (uri->fZPath != fZPath) {
         return 0;
      }
   }

   if (field_mask & URI_QUERY_MASK) {
      if (uri->fZQuery && fZQuery) {
         if (strcmp(uri->fZQuery, fZQuery) != 0) return 0;
      } else if (uri->fZQuery != fZQuery) {
         return 0;
      }
   }

   if (field_mask & URI_FRAGMENT_MASK) {
      if (uri->fZFragment && fZFragment) {
         if (strcmp(uri->fZFragment, fZFragment) != 0) return 0;
      } else if (uri->fZFragment != fZFragment) {
         return 0;
      }
   }

   return 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the length of the next component of the URL in z[] given
/// that the component starts at z[0].  The initial sequence of the
/// component must be zInit[].  The component is terminated by any
/// character in zTerm[].  The length returned is 0 if the component
/// doesn't exist.  The length includes the zInit[] string, but not
/// the termination character.
///
///        Component        zInit      zTerm
///        ----------       -------    -------
///        scheme           ""         ":/?#"
///        authority        "//"       "/?#"
///        path             "/"        "?#"
///        query            "?"        "#"
///        fragment         "#"        ""

int TGHtmlUri::ComponentLength(const char *z, const char *zInit, const char *zTerm)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Create a string to hold the given URI. Memory to hold the string is
/// allocated with new[] and must be freed by the calling function.

char *TGHtmlUri::BuildUri()
{
   int n = 1;
   char *z;

   if (fZScheme)    n += strlen(fZScheme) + 1;
   if (fZAuthority) n += strlen(fZAuthority) + 3;
   if (fZPath)      n += strlen(fZPath) + 1;
   if (fZQuery)     n += strlen(fZQuery) + 1;
   if (fZFragment)  n += strlen(fZFragment) + 1;
   z = new char[n];
   if (z == 0) return 0;
   n = 0;
   if (fZScheme) {
      // coverity[secure_coding]
      sprintf(z, "%s:", fZScheme);
      n = strlen(z);
   }
   if (fZAuthority) {
      // coverity[secure_coding]
      sprintf(&z[n], "//%s", fZAuthority);
      n += strlen(&z[n]);
   }
   if (fZAuthority && fZAuthority[strlen(fZAuthority)-1] != '/' &&
      !(fZPath && fZPath[0] == '/')) {
      // coverity[secure_coding]
      strcat(z, "/"); // NOLINT
      ++n;
   }
   if (fZPath) {
      // coverity[secure_coding]
      sprintf(&z[n], "%s", fZPath);
      n += strlen(&z[n]);
   }
   if (fZQuery) {
      // coverity[secure_coding]
      sprintf(&z[n], "?%s", fZQuery);
      n += strlen(&z[n]);
   }
   if (fZFragment) {
      // coverity[secure_coding]
      sprintf(&z[n], "#%s", fZFragment);
   } else {
      z[n] = 0;
   }

   return z;
}


////////////////////////////////////////////////////////////////////////////////
/// Duplicate a string of length n.

static char *StrNDup(const char *z, int n)
{
   char *zResult;

   if (n <= 0) n = strlen(z);
   zResult = new char[n + 1];
   if (zResult) {
      memcpy(zResult, z, n);
      zResult[n] = 0;
   }
   return zResult;
}

////////////////////////////////////////////////////////////////////////////////
/// Replace the string in *pzDest with the string in zSrc

static void ReplaceStr(char **pzDest, const char *zSrc)
{
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

////////////////////////////////////////////////////////////////////////////////
/// This function resolves the specified URI and returns the result in
/// a newly allocated string. The resolver algorithm specified in section
/// 5.2 of RFC 2396 is used.

char *TGHtml::ResolveUri(const char *zUri)
{
   char *result = 0;
   TGHtmlUri *base, *term;

   if (zUri == 0 || *zUri == 0) return 0;

   if (fZBaseHref && *fZBaseHref) {
      base = new TGHtmlUri(fZBaseHref);
   } else {
      base = new TGHtmlUri(fZBase);
   }

   term = new TGHtmlUri(zUri);

   if (term->fZScheme == 0 &&
       term->fZAuthority == 0 &&
       term->fZPath == 0 &&
       term->fZQuery == 0 &&
       term->fZFragment) {
      ReplaceStr(&base->fZFragment, term->fZFragment);
   } else if (term->fZScheme) {
      TGHtmlUri *temp;
      temp = term;
      term = base;
      base = temp;
   } else if (term->fZAuthority) {
      ReplaceStr(&base->fZAuthority, term->fZAuthority);
      ReplaceStr(&base->fZPath, term->fZPath);
      ReplaceStr(&base->fZQuery, term->fZQuery);
      ReplaceStr(&base->fZFragment, term->fZFragment);
   } else if (term->fZPath && (term->fZPath[0] == '/' || base->fZPath == 0)) {
      ReplaceStr(&base->fZPath, term->fZPath);
      ReplaceStr(&base->fZQuery, term->fZQuery);
      ReplaceStr(&base->fZFragment, term->fZFragment);
   } else if (term->fZPath && base->fZPath) {
      char *zBuf;
      int i, j;
      zBuf = new char[strlen(base->fZPath) + strlen(term->fZPath) + 2];
      if (zBuf) {
         // coverity[secure_coding]
         sprintf(zBuf, "%s", base->fZPath); // NOLINT
         for (i = strlen(zBuf) - 1; i >= 0 && zBuf[i] != '/'; --i) {
            zBuf[i] = 0;
         }
         // coverity[secure_coding]
         strlcat(zBuf, term->fZPath, strlen(base->fZPath) + strlen(term->fZPath) + 2);
         for (i = 0; zBuf[i]; i++) {
            if (zBuf[i] == '/' && zBuf[i+1] == '.' && zBuf[i+2] == '/') {
               // coverity[secure_coding]
               strcpy(&zBuf[i+1], &zBuf[i+3]); // NOLINT
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
                  // coverity[secure_coding]
                  strcpy(&zBuf[j+1], &zBuf[i+4]); // NOLINT
               } else {
                  zBuf[j+1] = 0;
               }
               i = j - 1;
               if (i < -1) i = -1;
               continue;
            }
         }
         delete[] base->fZPath;
         base->fZPath = zBuf;
      }
      ReplaceStr(&base->fZQuery, term->fZQuery);
      ReplaceStr(&base->fZFragment, term->fZFragment);
   }
   delete term;

   result = base->BuildUri();
   delete base;

   return result;
}
