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

#include "gdkwindow.h"
#include "gdkprivate.h"

GdkWindow *_gdk_window_alloc(void)
{
   GdkWindowPrivate *private = g_new(GdkWindowPrivate, 1);
   GdkWindow *window = (GdkWindow *) private;

   window->user_data = NULL;

   private->drawable.ref_count = 1;
   private->drawable.destroyed = FALSE;
   private->drawable.klass = NULL;
   private->drawable.klass_data = NULL;
   private->drawable.window_type = GDK_WINDOW_CHILD;

   private->drawable.width = 1;
   private->drawable.height = 1;

   private->drawable.colormap = NULL;

   private->parent = NULL;
   private->x = 0;
   private->y = 0;
   private->resize_count = 0;

   private->mapped = FALSE;
   private->guffaw_gravity = FALSE;
   private->extension_events = FALSE;

   private->filters = NULL;
   private->children = NULL;

   return window;
}

void gdk_window_set_user_data(GdkWindow * window, gpointer user_data)
{
   g_return_if_fail(window != NULL);

   window->user_data = user_data;
}

void gdk_window_get_user_data(GdkWindow * window, gpointer * data)
{
   g_return_if_fail(window != NULL);

   *data = window->user_data;
}

void gdk_window_get_position(GdkWindow * window, gint * x, gint * y)
{
   GdkWindowPrivate *window_private;

   g_return_if_fail(window != NULL);
   g_return_if_fail(GDK_IS_WINDOW(window));

   window_private = (GdkWindowPrivate *) window;

   if (x)
      *x = window_private->x;
   if (y)
      *y = window_private->y;
}

GdkWindow *gdk_window_get_parent(GdkWindow * window)
{
   g_return_val_if_fail(window != NULL, NULL);
   g_return_val_if_fail(GDK_IS_WINDOW(window), NULL);

   return ((GdkWindowPrivate *) window)->parent;
}

GdkWindow *gdk_window_get_toplevel(GdkWindow * window)
{
   GdkWindowPrivate *private;

   g_return_val_if_fail(window != NULL, NULL);
   g_return_val_if_fail(GDK_IS_WINDOW(window), NULL);

   private = (GdkWindowPrivate *) window;
   while (GDK_DRAWABLE_TYPE(private) == GDK_WINDOW_CHILD)
      private = (GdkWindowPrivate *) private->parent;

   return (GdkWindow *) window;
}

void
gdk_window_add_filter(GdkWindow * window,
                      GdkFilterFunc function, gpointer data)
{
   GdkWindowPrivate *private;
   GList *tmp_list;
   GdkEventFilter *filter;

   g_return_if_fail(window != NULL);
   g_return_if_fail(GDK_IS_WINDOW(window));

   private = (GdkWindowPrivate *) window;
   if (private && GDK_DRAWABLE_DESTROYED(window))
      return;

   if (private)
      tmp_list = private->filters;
   else
      tmp_list = gdk_default_filters;

   while (tmp_list) {
      filter = (GdkEventFilter *) tmp_list->data;
      if ((filter->function == function) && (filter->data == data))
         return;
      tmp_list = tmp_list->next;
   }

   filter = g_new(GdkEventFilter, 1);
   filter->function = function;
   filter->data = data;

   if (private)
      private->filters = g_list_append(private->filters, filter);
   else
      gdk_default_filters = g_list_append(gdk_default_filters, filter);
}

void
gdk_window_remove_filter(GdkWindow * window,
                         GdkFilterFunc function, gpointer data)
{
   GdkWindowPrivate *private;
   GList *tmp_list, *node;
   GdkEventFilter *filter;

   g_return_if_fail(window != NULL);
   g_return_if_fail(GDK_IS_WINDOW(window));

   private = (GdkWindowPrivate *) window;

   if (private)
      tmp_list = private->filters;
   else
      tmp_list = gdk_default_filters;

   while (tmp_list) {
      filter = (GdkEventFilter *) tmp_list->data;
      node = tmp_list;
      tmp_list = tmp_list->next;

      if ((filter->function == function) && (filter->data == data)) {
         if (private)
            private->filters = g_list_remove_link(private->filters, node);
         else
            gdk_default_filters =
                g_list_remove_link(gdk_default_filters, node);
         g_list_free_1(node);
         g_free(filter);

         return;
      }
   }
}

GList *gdk_window_get_toplevels(void)
{
   GList *new_list = NULL;
   GList *tmp_list;

   tmp_list = ((GdkWindowPrivate *) gdk_parent_root)->children;
   while (tmp_list) {
      new_list = g_list_prepend(new_list, tmp_list->data);
      tmp_list = tmp_list->next;
   }

   return new_list;
}

/*************************************************************
 * gdk_window_is_visible:
 *     Check if the given window is mapped.
 *   arguments:
 *     window: 
 *   results:
 *     is the window mapped
 *************************************************************/

gboolean gdk_window_is_visible(GdkWindow * window)
{
   GdkWindowPrivate *private = (GdkWindowPrivate *) window;

   g_return_val_if_fail(window != NULL, FALSE);
   g_return_val_if_fail(GDK_IS_WINDOW(window), FALSE);

   return private->mapped;
}

/*************************************************************
 * gdk_window_is_viewable:
 *     Check if the window and all ancestors of the window
 *     are mapped. (This is not necessarily "viewable" in
 *     the X sense, since we only check as far as we have
 *     GDK window parents, not to the root window)
 *   arguments:
 *     window:
 *   results:
 *     is the window viewable
 *************************************************************/

gboolean gdk_window_is_viewable(GdkWindow * window)
{
   GdkWindowPrivate *private = (GdkWindowPrivate *) window;

   g_return_val_if_fail(window != NULL, FALSE);
   g_return_val_if_fail(GDK_IS_WINDOW(window), FALSE);

   while (private &&
          (private != (GdkWindowPrivate *) gdk_parent_root) &&
          (private->drawable.window_type != GDK_WINDOW_FOREIGN)) {
      if (!private->mapped)
         return FALSE;

      private = (GdkWindowPrivate *) private->parent;
   }

   return TRUE;
}
