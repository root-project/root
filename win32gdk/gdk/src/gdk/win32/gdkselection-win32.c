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

#include <gdk/gdk.h>
#include "gdkwin32.h"

/* We emulate the GDK_SELECTION window properties by storing
 * it's data in a per-window hashtable.
 */

typedef struct {
   guchar *data;
   gint length;
   gint format;
   GdkAtom type;
} GdkSelProp;

static GHashTable *sel_prop_table = NULL;

void gdk_win32_selection_init(void)
{
   if (sel_prop_table == NULL)
      sel_prop_table = g_hash_table_new(g_int_hash, g_int_equal);
}

void
gdk_sel_prop_store(GdkWindow * owner,
                   GdkAtom type, gint format, guchar * data, gint length)
{
   GdkSelProp *prop;

   prop = g_hash_table_lookup(sel_prop_table, &GDK_DRAWABLE_XID(owner));
   if (prop != NULL) {
      g_free(prop->data);
      g_hash_table_remove(sel_prop_table, &GDK_DRAWABLE_XID(owner));
   }
   prop = g_new(GdkSelProp, 1);
   prop->data = data;
   prop->length = length;
   prop->format = format;
   prop->type = type;
   g_hash_table_insert(sel_prop_table, &GDK_DRAWABLE_XID(owner), prop);
}

gint
gdk_selection_owner_set(GdkWindow * owner,
                        GdkAtom selection, guint32 time, gint send_event)
{
   gchar *sel_name;
   HWND xwindow;

   GDK_NOTE(DND,
            (sel_name = gdk_atom_name(selection),
             g_print("gdk_selection_owner_set: %#x %#x (%s)\n",
                     (owner ? GDK_DRAWABLE_XID(owner) : 0),
                     selection, sel_name), g_free(sel_name)));

   if (selection != gdk_clipboard_atom)
      return FALSE;

   if (owner != NULL)
      xwindow = GDK_DRAWABLE_XID(owner);
   else
      xwindow = NULL;

   GDK_NOTE(DND, g_print("...OpenClipboard(%#x)\n", xwindow));
   if (!OpenClipboard(xwindow)) {
      WIN32_API_FAILED("OpenClipboard");
      return FALSE;
   }
   GDK_NOTE(DND, g_print("...EmptyClipboard()\n"));
   if (!EmptyClipboard()) {
      WIN32_API_FAILED("EmptyClipboard");
      CloseClipboard();
      return FALSE;
   }
#if 0
   /* No delayed rendering */
   if (xwindow != NULL)
      SetClipboardData(CF_TEXT, NULL);
#endif
   GDK_NOTE(DND, g_print("...CloseClipboard()\n"));
   if (!CloseClipboard()) {
      WIN32_API_FAILED("CloseClipboard");
      return FALSE;
   }
   if (owner != NULL) {
      /* Send ourselves an ersatz selection request message so that
       * gdk_property_change will be called to store the clipboard data.
       */
      SendMessage(xwindow, gdk_selection_request_msg, selection, 0);
   }

   return TRUE;
}

GdkWindow *gdk_selection_owner_get(GdkAtom selection)
{
   GdkWindow *window;
   gchar *sel_name;

#if 0
   /* XXX Hmm, gtk selections seem to work best with this. This causes
    * gtk to always get the clipboard contents from Windows, and not
    * from the editable's own stashed-away copy.
    */
   return NULL;
#else
   if (selection != gdk_clipboard_atom)
      window = NULL;
   else {
      window = gdk_window_lookup(GetClipboardOwner());
      if (window == NULL)
         window = (GdkWindow *)GetClipboardOwner();
   }

#endif

   GDK_NOTE(DND,
            (sel_name = gdk_atom_name(selection),
             g_print("gdk_selection_owner_get: %#x (%s) = %#x\n",
                     selection, sel_name,
                     (window ? GDK_DRAWABLE_XID(window) : 0)),
             g_free(sel_name)));

   return window;
}

void
gdk_selection_convert(GdkWindow * requestor,
                      GdkAtom selection, GdkAtom target, guint32 time)
{
   HGLOBAL hdata;
   GdkSelProp *prop;
   guchar *ptr, *data, *datap, *p;
   guint i, length, slength;
   gchar *sel_name, *tgt_name;

   g_return_if_fail(requestor != NULL);
   if (GDK_DRAWABLE_DESTROYED(requestor))
      return;

   GDK_NOTE(DND,
            (sel_name = gdk_atom_name(selection),
             tgt_name = gdk_atom_name(target),
             g_print("gdk_selection_convert: %#x %#x (%s) %#x (%s)\n",
                     GDK_DRAWABLE_XID(requestor), selection, sel_name,
                     target, tgt_name), g_free(sel_name),
             g_free(tgt_name)));

   if (selection == gdk_clipboard_atom) {
      /* Converting the CLIPBOARD selection means he wants the
       * contents of the clipboard. Get the clipboard data,
       * and store it for later.
       */
      GDK_NOTE(DND, g_print("...OpenClipboard(%#x)\n",
                            GDK_DRAWABLE_XID(requestor)));
      if (!OpenClipboard(GDK_DRAWABLE_XID(requestor))) {
         WIN32_API_FAILED("OpenClipboard");
         return;
      }

      GDK_NOTE(DND, g_print("...GetClipboardData(CF_TEXT)\n"));
      if ((hdata = GetClipboardData(CF_TEXT)) != NULL) {
         if ((ptr = GlobalLock(hdata)) != NULL) {
            length = GlobalSize(hdata);

            GDK_NOTE(DND, g_print("...got data: %d bytes: %.10s\n",
                                  length, ptr));

            slength = 0;
            p = ptr;
            for (i = 0; i < length; i++) {
               if (*p == '\0')
                  break;
               else if (*p != '\r')
                  slength++;
               p++;
            }

            data = datap = g_malloc(slength + 1);
            p = ptr;
            for (i = 0; i < length; i++) {
               if (*p == '\0')
                  break;
               else if (*p != '\r')
                  *datap++ = *p;
               p++;
            }
            *datap++ = '\0';
            gdk_sel_prop_store(requestor, GDK_TARGET_STRING, 8,
                               data, strlen(data) + 1);

            GlobalUnlock(hdata);
         }
      }
      GDK_NOTE(DND, g_print("...CloseClipboard()\n"));
      CloseClipboard();


      /* Send ourselves an ersatz selection notify message so that we actually
       * fetch the data.
       */
      SendMessage(GDK_DRAWABLE_XID(requestor), gdk_selection_notify_msg,
                  selection, target);
   } else if (selection == gdk_win32_dropfiles_atom) {
      /* This means he wants the names of the dropped files.
       * gdk_dropfiles_filter already has stored the text/uri-list
       * data, tempoarily on gdk_root_parent's selection "property".
       */
      GdkSelProp *prop;

      prop = g_hash_table_lookup(sel_prop_table,
                                 &GDK_DRAWABLE_XID(gdk_parent_root));

      if (prop != NULL) {
         g_hash_table_remove(sel_prop_table,
                             &GDK_DRAWABLE_XID(gdk_parent_root));
         gdk_sel_prop_store(requestor, prop->type, prop->format,
                            prop->data, prop->length);
         g_free(prop);
         SendMessage(GDK_DRAWABLE_XID(requestor), gdk_selection_notify_msg,
                     selection, target);
      }
   } else {
      g_warning("gdk_selection_convert: General case not implemented");
   }
}

gint
gdk_selection_property_get(GdkWindow * requestor,
                           guchar ** data,
                           GdkAtom * ret_type, gint * ret_format)
{
   GdkSelProp *prop;

   g_return_val_if_fail(requestor != NULL, 0);
   g_return_val_if_fail(GDK_IS_WINDOW(requestor), 0);

   if (GDK_DRAWABLE_DESTROYED(requestor))
      return 0;

   GDK_NOTE(DND, g_print("gdk_selection_property_get: %#x",
                         GDK_DRAWABLE_XID(requestor)));

   prop =
       g_hash_table_lookup(sel_prop_table, &GDK_DRAWABLE_XID(requestor));

   if (prop == NULL) {
      GDK_NOTE(DND, g_print(": NULL\n"));
      *data = NULL;
      return 0;
   }
   GDK_NOTE(DND, g_print(": %d bytes\n", prop->length));
   *data = g_malloc(prop->length);
   if (prop->length > 0)
      memmove(*data, prop->data, prop->length);
   if (ret_type)
      *ret_type = prop->type;
   if (ret_format)
      *ret_format = prop->format;

   return prop->length;
}

void gdk_selection_property_delete(GdkWindow * window)
{
   GdkSelProp *prop;

   GDK_NOTE(DND, g_print("gdk_selection_property_delete: %#x",
                         GDK_DRAWABLE_XID(window)));

   prop = g_hash_table_lookup(sel_prop_table, &GDK_DRAWABLE_XID(window));
   if (prop != NULL) {
      g_free(prop->data);
      g_hash_table_remove(sel_prop_table, &GDK_DRAWABLE_XID(window));
   } else
      g_warning("gdk_selection_property_delete: not found");
}

void
gdk_selection_send_notify(guint32 requestor,
                          GdkAtom selection,
                          GdkAtom target, GdkAtom property, guint32 time)
{
   gchar *sel_name, *tgt_name, *prop_name;

   GDK_NOTE(DND,
            (sel_name = gdk_atom_name(selection),
             tgt_name = gdk_atom_name(target),
             prop_name = gdk_atom_name(property),
             g_print
             ("gdk_selection_send_notify: %#x %#x (%s) %#x (%s) %#x (%s)\n",
              requestor, selection, sel_name, target, tgt_name, property,
              prop_name), g_free(sel_name), g_free(tgt_name),
             g_free(prop_name)));

   /* Send ourselves a selection clear message so that gtk thinks we don't
    * have the selection, and will claim it anew when needed, and
    * we thus get a chance to store data in the Windows clipboard.
    * Otherwise, if a gtkeditable does a copy to clipboard several times
    * only the first one actually gets copied to the Windows clipboard,
    * as only he first one causes a call to gdk_property_change.
    *
    * Hmm, there is something fishy with this. Cut and paste inside the
    * same app didn't work, the gtkeditable immediately forgot the
    * clipboard contents in gtk_editable_selection_clear as a result of
    * this message. OTOH, when I changed gdk_selection_owner_get to
    * always return NULL, it works. Sigh.
    */

   SendMessage((HWND) requestor, gdk_selection_clear_msg, selection, 0);
}

gint
gdk_text_property_to_text_list(GdkAtom encoding,
                               gint format,
                               const guchar * text,
                               gint length, gchar *** list)
{
   GDK_NOTE(DND,
            g_print("gdk_text_property_to_text_list not implemented\n"));

   return 0;
}

void gdk_free_text_list(gchar ** list)
{
   g_return_if_fail(list != NULL);

   /* ??? */
}

gint
gdk_string_to_compound_text(const gchar * str,
                            GdkAtom * encoding,
                            gint * format, guchar ** ctext, gint * length)
{
   g_warning("gdk_string_to_compound_text: Not implemented");

   return 0;
}

void gdk_free_compound_text(guchar * ctext)
{
   g_warning("gdk_free_compound_text: Not implemented");
}
