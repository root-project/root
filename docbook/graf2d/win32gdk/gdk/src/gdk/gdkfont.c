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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
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

#include "gdkfont.h"
#include "gdkprivate.h"

GdkFont *gdk_font_ref(GdkFont * font)
{
   GdkFontPrivate *private;

   g_return_val_if_fail(font != NULL, NULL);

   private = (GdkFontPrivate *) font;
   private->ref_count += 1;
   return font;
}

void gdk_font_unref(GdkFont * font)
{
   GdkFontPrivate *private;
   private = (GdkFontPrivate *) font;

   g_return_if_fail(font != NULL);
   g_return_if_fail(private->ref_count > 0);

   private->ref_count -= 1;
   if (private->ref_count == 0)
      _gdk_font_destroy(font);
}

gint gdk_string_width(GdkFont * font, const gchar * string)
{
   g_return_val_if_fail(font != NULL, -1);
   g_return_val_if_fail(string != NULL, -1);

   return gdk_text_width(font, string, _gdk_font_strlen(font, string));
}

gint gdk_char_width(GdkFont * font, gchar character)
{
   g_return_val_if_fail(font != NULL, -1);

   return gdk_text_width(font, &character, 1);
}

gint gdk_char_width_wc(GdkFont * font, GdkWChar character)
{
   g_return_val_if_fail(font != NULL, -1);

   return gdk_text_width_wc(font, &character, 1);
}

gint gdk_string_measure(GdkFont * font, const gchar * string)
{
   g_return_val_if_fail(font != NULL, -1);
   g_return_val_if_fail(string != NULL, -1);

   return gdk_text_measure(font, string, _gdk_font_strlen(font, string));
}

void
gdk_string_extents(GdkFont * font,
                   const gchar * string,
                   gint * lbearing,
                   gint * rbearing,
                   gint * width, gint * ascent, gint * descent)
{
   g_return_if_fail(font != NULL);
   g_return_if_fail(string != NULL);

   gdk_text_extents(font, string, _gdk_font_strlen(font, string),
                    lbearing, rbearing, width, ascent, descent);
}


gint gdk_text_measure(GdkFont * font, const gchar * text, gint text_length)
{
   gint rbearing;

   g_return_val_if_fail(font != NULL, -1);
   g_return_val_if_fail(text != NULL, -1);

   gdk_text_extents(font, text, text_length, NULL, &rbearing, NULL, NULL,
                    NULL);
   return rbearing;
}

gint gdk_char_measure(GdkFont * font, gchar character)
{
   g_return_val_if_fail(font != NULL, -1);

   return gdk_text_measure(font, &character, 1);
}

gint gdk_string_height(GdkFont * font, const gchar * string)
{
   g_return_val_if_fail(font != NULL, -1);
   g_return_val_if_fail(string != NULL, -1);

   return gdk_text_height(font, string, _gdk_font_strlen(font, string));
}

gint gdk_text_height(GdkFont * font, const gchar * text, gint text_length)
{
   gint ascent, descent;

   g_return_val_if_fail(font != NULL, -1);
   g_return_val_if_fail(text != NULL, -1);

   gdk_text_extents(font, text, text_length, NULL, NULL, NULL, &ascent,
                    &descent);
   return ascent + descent;
}

gint gdk_char_height(GdkFont * font, gchar character)
{
   g_return_val_if_fail(font != NULL, -1);

   return gdk_text_height(font, &character, 1);
}
