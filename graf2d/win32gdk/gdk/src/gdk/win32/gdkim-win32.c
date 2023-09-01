/* GDK - The GIMP Drawing Kit
 * Copyright (C) 1995-1997 Peter Mattis, Spencer Kimball and Josh MacDonald
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/*
 * Modified by the GTK+ Team and others 1997-1999.  See the AUTHORS
 * file for a list of people on the GTK+ Team.  See the ChangeLog
 * files for a list of changes.  These files are distributed with
 * GTK+ at ftp://ftp.gtk.org/pub/gtk/. 
 */

#if HAVE_CONFIG_H
#  include <config.h>
#endif

#include <stdlib.h>
#include <string.h>

#include "gdkim.h"
#include "gdkpixmap.h"
#include "gdkprivate.h"
#include "gdki18n.h"
#include "gdkwin32.h"

/*
 *--------------------------------------------------------------
 * gdk_set_locale
 *
 * Arguments:
 *
 * Results:
 *
 * Side effects:
 *
 *--------------------------------------------------------------
 */

gchar *gdk_set_locale(void)
{
   if (!setlocale(LC_ALL, ""))
      g_warning("locale not supported by C library");

   return g_win32_getlocale();
}

void gdk_im_begin(GdkIC * ic, GdkWindow * window)
{
}

void gdk_im_end(void)
{
}

GdkIMStyle gdk_im_decide_style(GdkIMStyle supported_style)
{
   return GDK_IM_PREEDIT_NONE | GDK_IM_STATUS_NONE;
}

GdkIMStyle gdk_im_set_best_style(GdkIMStyle style)
{
   return GDK_IM_PREEDIT_NONE | GDK_IM_STATUS_NONE;
}

gint gdk_im_ready(void)
{
   return FALSE;
}

GdkIC *gdk_ic_new(GdkICAttr * attr, GdkICAttributesType mask)
{
   return NULL;
}

void gdk_ic_destroy(GdkIC * ic)
{
}

GdkIMStyle gdk_ic_get_style(GdkIC * ic)
{
   return GDK_IM_PREEDIT_NONE | GDK_IM_STATUS_NONE;
}

void gdk_ic_set_values(GdkIC * ic, ...)
{
}

void gdk_ic_get_values(GdkIC * ic, ...)
{
}

GdkICAttributesType
gdk_ic_set_attr(GdkIC * ic, GdkICAttr * attr, GdkICAttributesType mask)
{
   return 0;
}

GdkICAttributesType
gdk_ic_get_attr(GdkIC * ic, GdkICAttr * attr, GdkICAttributesType mask)
{
   return 0;
}

GdkEventMask gdk_ic_get_events(GdkIC * ic)
{
   return 0;
}

/*
 * gdk_wcstombs 
 *
 * Returns a multi-byte string converted from the specified array
 * of wide characters. The string is newly allocated. The array of
 * wide characters must be null-terminated. If the conversion is
 * failed, it returns NULL.
 *
 * On Win32, we always use UTF-8.
 */
gchar *gdk_wcstombs(const GdkWChar * src)
{
   gint len;
   const GdkWChar *wcp;
   guchar *mbstr, *bp;

   wcp = src;
   len = 0;
   while (*wcp) {
      const GdkWChar c = *wcp++;

      if (c < 0x80)
         len += 1;
      else if (c < 0x800)
         len += 2;
      else if (c < 0x10000)
         len += 3;
      else if (c < 0x200000)
         len += 4;
      else if (c < 0x4000000)
         len += 5;
      else
         len += 6;
   }

   mbstr = g_malloc(len + 1);

   wcp = src;
   bp = mbstr;
   while (*wcp) {
      int first;
      GdkWChar c = *wcp++;

      if (c < 0x80) {
         first = 0;
         len = 1;
      } else if (c < 0x800) {
         first = 0xc0;
         len = 2;
      } else if (c < 0x10000) {
         first = 0xe0;
         len = 3;
      } else if (c < 0x200000) {
         first = 0xf0;
         len = 4;
      } else if (c < 0x4000000) {
         first = 0xf8;
         len = 5;
      } else {
         first = 0xfc;
         len = 6;
      }

      /* Woo-hoo! */
      switch (len) {
      case 6:
         bp[5] = (c & 0x3f) | 0x80;
         c >>= 6;               /* Fall through */
      case 5:
         bp[4] = (c & 0x3f) | 0x80;
         c >>= 6;               /* Fall through */
      case 4:
         bp[3] = (c & 0x3f) | 0x80;
         c >>= 6;               /* Fall through */
      case 3:
         bp[2] = (c & 0x3f) | 0x80;
         c >>= 6;               /* Fall through */
      case 2:
         bp[1] = (c & 0x3f) | 0x80;
         c >>= 6;               /* Fall through */
      case 1:
         bp[0] = c | first;
      }

      bp += len;
   }
   *bp = 0;
   return mbstr;
}


/*
 * gdk_mbstowcs
 *
 * Converts the specified string into GDK wide characters, and,
 * returns the number of wide characters written. The string 'src'
 * must be null-terminated. If the conversion is failed, it returns
 * -1.
 *
 * On Win32, the string is assumed to be in UTF-8.  Also note that
 * GdkWChar is 32 bits, while wchar_t, and the wide characters the
 * Windows API uses, are 16 bits!
 */

/* First a helper function for not zero-terminated strings */
gint
gdk_nmbstowcs(GdkWChar * dest,
              const gchar * src, gint src_len, gint dest_max)
{
   guchar *cp, *end;
   gint n;

   cp = (guchar *) src;
   end = cp + src_len;
   n = 0;
   while (cp != end && dest != dest + dest_max) {
      gint i, mask = 0, len;
      guchar c = *cp;

      if (c < 0x80) {
         len = 1;
         mask = 0x7f;
      } else if ((c & 0xe0) == 0xc0) {
         len = 2;
         mask = 0x1f;
      } else if ((c & 0xf0) == 0xe0) {
         len = 3;
         mask = 0x0f;
      } else if ((c & 0xf8) == 0xf0) {
         len = 4;
         mask = 0x07;
      } else if ((c & 0xfc) == 0xf8) {
         len = 5;
         mask = 0x03;
      } else if ((c & 0xfc) == 0xfc) {
         len = 6;
         mask = 0x01;
      } else
         return -1;

      if (cp + len > end)
         return -1;

      *dest = (cp[0] & mask);
      for (i = 1; i < len; i++) {
         if ((cp[i] & 0xc0) != 0x80)
            return -1;
         *dest <<= 6;
         *dest |= (cp[i] & 0x3f);
      }
      if (*dest == -1)
         return -1;

      cp += len;
      dest++;
      n++;
   }
   if (cp != end)
      return -1;

   return n;
}

gint gdk_mbstowcs(GdkWChar * dest, const gchar * src, gint dest_max)
{
   return gdk_nmbstowcs(dest, src, strlen(src), dest_max);
}


/* A version that converts to wchar_t wide chars */

gint
gdk_nmbstowchar_ts(wchar_t * dest,
                   const gchar * src, gint src_len, gint dest_max)
{
#if 1
   return mbstowcs(dest, src, src_len);
#else
   wchar_t *wcp;
   guchar *cp, *end;
   gint n;

   wcp = dest;
   cp = (guchar *) src;
   end = cp + src_len;
   n = 0;
   while (cp != end && wcp != dest + dest_max) {
      gint i, mask = 0, len;
      guchar c = *cp;

      if (c < 0x80) {
         len = 1;
         mask = 0x7f;
      } else if ((c & 0xe0) == 0xc0) {
         len = 2;
         mask = 0x1f;
      } else if ((c & 0xf0) == 0xe0) {
         len = 3;
         mask = 0x0f;
      } else                    /* Other lengths are not possible with 16-bit wchar_t! */
         return -1;

      if (cp + len > end)
         return -1;

      *wcp = (cp[0] & mask);
      for (i = 1; i < len; i++) {
         if ((cp[i] & 0xc0) != 0x80)
            return -1;
         *wcp <<= 6;
         *wcp |= (cp[i] & 0x3f);
      }
      if (*wcp == 0xFFFF)
         return -1;

      cp += len;
      wcp++;
      n++;
   }
   if (cp != end)
      return -1;

   return n;
#endif
}
