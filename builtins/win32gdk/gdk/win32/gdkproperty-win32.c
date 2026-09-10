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

#include "config.h"

#include <string.h>

#include "gdkproperty.h"
#include "gdkselection.h"
#include "gdkprivate.h"
#include "gdkwin32.h"

GdkAtom gdk_atom_intern(const gchar * atom_name, gint only_if_exists)
{
   GdkAtom retval;
   static GHashTable *atom_hash = NULL;

   if (!atom_hash)
      atom_hash = g_hash_table_new(g_str_hash, g_str_equal);

   retval = GPOINTER_TO_UINT(g_hash_table_lookup(atom_hash, atom_name));
   if (!retval) {
      if (strcmp(atom_name, "PRIMARY") == 0)
         retval = GDK_SELECTION_PRIMARY;
      else if (strcmp(atom_name, "SECONDARY") == 0)
         retval = GDK_SELECTION_SECONDARY;
      else if (strcmp(atom_name, "ATOM") == 0)
         retval = GDK_SELECTION_TYPE_ATOM;
      else if (strcmp(atom_name, "BITMAP") == 0)
         retval = GDK_SELECTION_TYPE_BITMAP;
      else if (strcmp(atom_name, "COLORMAP") == 0)
         retval = GDK_SELECTION_TYPE_COLORMAP;
      else if (strcmp(atom_name, "DRAWABLE") == 0)
         retval = GDK_SELECTION_TYPE_DRAWABLE;
      else if (strcmp(atom_name, "INTEGER") == 0)
         retval = GDK_SELECTION_TYPE_INTEGER;
      else if (strcmp(atom_name, "PIXMAP") == 0)
         retval = GDK_SELECTION_TYPE_PIXMAP;
      else if (strcmp(atom_name, "WINDOW") == 0)
         retval = GDK_SELECTION_TYPE_WINDOW;
      else if (strcmp(atom_name, "STRING") == 0)
         retval = GDK_SELECTION_TYPE_STRING;
      else {
         retval = GlobalFindAtom(atom_name);
         if (only_if_exists && retval == 0)
            retval = 0;
         else
            retval = GlobalAddAtom(atom_name);
      }
      g_hash_table_insert(atom_hash,
                          g_strdup(atom_name), GUINT_TO_POINTER(retval));
   }

   return retval;
}

gchar *gdk_atom_name(GdkAtom atom)
{
   gchar name[256];

   switch (atom) {
   case GDK_SELECTION_PRIMARY:
      return g_strdup("PRIMARY");
   case GDK_SELECTION_SECONDARY:
      return g_strdup("SECONDARY");
   case GDK_SELECTION_TYPE_ATOM:
      return g_strdup("ATOM");
   case GDK_SELECTION_TYPE_BITMAP:
      return g_strdup("BITMAP");
   case GDK_SELECTION_TYPE_COLORMAP:
      return g_strdup("COLORMAP");
   case GDK_SELECTION_TYPE_DRAWABLE:
      return g_strdup("DRAWABLE");
   case GDK_SELECTION_TYPE_INTEGER:
      return g_strdup("INTEGER");
   case GDK_SELECTION_TYPE_PIXMAP:
      return g_strdup("PIXMAP");
   case GDK_SELECTION_TYPE_WINDOW:
      return g_strdup("WINDOW");
   case GDK_SELECTION_TYPE_STRING:
      return g_strdup("STRING");
   }
   if (atom < 0xC000)
      return g_strdup_printf("#%x", atom);
   else if (GlobalGetAtomName(atom, name, sizeof(name)) == 0)
      return NULL;
   return g_strdup(name);
}

gint
gdk_property_get(GdkWindow * window,
                 GdkAtom property,
                 GdkAtom type,
                 gulong offset,
                 gulong length,
                 gint pdelete,
                 GdkAtom * actual_property_type,
                 gint * actual_format_type,
                 gint * actual_length, guchar ** data)
{
   g_return_val_if_fail(window != NULL, FALSE);
   g_return_val_if_fail(GDK_IS_WINDOW(window), FALSE);

   if (GDK_DRAWABLE_DESTROYED(window))
      return FALSE;

   g_warning("gdk_property_get: Not implemented");

   return FALSE;
}

void
gdk_property_change(GdkWindow * window,
                    GdkAtom property,
                    GdkAtom type,
                    gint format,
                    GdkPropMode mode, const guchar * data, gint nelements)
{
   HGLOBAL hdata;
   gint i, length;
   gchar *prop_name, *type_name;
   guchar *ptr;

   g_return_if_fail(window != NULL);
   g_return_if_fail(GDK_IS_WINDOW(window));

   if (GDK_DRAWABLE_DESTROYED(window))
      return;

   GDK_NOTE(DND,
            (prop_name = gdk_atom_name(property),
             type_name = gdk_atom_name(type),
             g_print
             ("gdk_property_change: %#x %#x (%s) %#x (%s) %s %d*%d bytes %.10s\n",
              GDK_DRAWABLE_XID(window), property, prop_name, type,
              type_name,
              (mode ==
               GDK_PROP_MODE_REPLACE ? "REPLACE" : (mode ==
                                                    GDK_PROP_MODE_PREPEND ?
                                                    "PREPEND" : (mode ==
                                                                 GDK_PROP_MODE_APPEND
                                                                 ? "APPEND"
                                                                 :
                                                                 "???"))),
              format, nelements, data), g_free(prop_name),
             g_free(type_name)));

   if (property == gdk_selection_property
       && type == GDK_TARGET_STRING
       && format == 8 && mode == GDK_PROP_MODE_REPLACE) {
      length = nelements;
      for (i = 0; i < nelements; i++)
         if (data[i] == '\n')
            length++;
#if 1
      GDK_NOTE(DND, g_print("...OpenClipboard(%#x)\n",
                            GDK_DRAWABLE_XID(window)));
      if (!OpenClipboard(GDK_DRAWABLE_XID(window))) {
         WIN32_API_FAILED("OpenClipboard");
         return;
      }
#endif
      hdata = GlobalAlloc(GMEM_MOVEABLE | GMEM_DDESHARE, length + 1);
      ptr = GlobalLock(hdata);
      GDK_NOTE(DND, g_print("...hdata=%#x, ptr=%#x\n", hdata, ptr));

      for (i = 0; i < nelements; i++) {
         if (*data == '\n')
            *ptr++ = '\r';
         *ptr++ = *data++;
      }
      *ptr++ = '\0';
      GlobalUnlock(hdata);
      GDK_NOTE(DND, g_print("...SetClipboardData(CF_TEXT, %#x)\n", hdata));
      if (!SetClipboardData(CF_TEXT, hdata))
         WIN32_API_FAILED("SetClipboardData");
#if 1
      GDK_NOTE(DND, g_print("...CloseClipboard()\n"));
      if (!CloseClipboard())
         WIN32_API_FAILED("CloseClipboard");
#endif
   } else
      g_warning("gdk_property_change: General case not implemented");
}

void gdk_property_delete(GdkWindow * window, GdkAtom property)
{
   gchar *prop_name;
   extern void gdk_selection_property_delete(GdkWindow *);

   g_return_if_fail(window != NULL);
   g_return_if_fail(GDK_IS_WINDOW(window));

   GDK_NOTE(DND,
            (prop_name = gdk_atom_name(property),
             g_print("gdk_property_delete: %#x %#x (%s)\n",
                     (window ? GDK_DRAWABLE_XID(window) : 0),
                     property, prop_name), g_free(prop_name)));

   if (property == gdk_selection_property)
      gdk_selection_property_delete(window);
   else
      g_warning("gdk_property_delete: General case not implemented");
}
