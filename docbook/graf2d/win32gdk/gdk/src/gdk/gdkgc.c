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

#include <string.h>

#include "gdkgc.h"
#include "gdkprivate.h"

GdkGC *gdk_gc_alloc(void)
{
   GdkGCPrivate *private;

   private = g_new(GdkGCPrivate, 1);
   private->ref_count = 1;
   private->klass = NULL;
   private->klass_data = NULL;

   return (GdkGC *) private;
}

GdkGC *gdk_gc_new(GdkDrawable * drawable)
{
   g_return_val_if_fail(drawable != NULL, NULL);

   if (GDK_DRAWABLE_DESTROYED(drawable))
      return NULL;

   return gdk_gc_new_with_values(drawable, NULL, 0);
}

GdkGC *gdk_gc_new_with_values(GdkDrawable * drawable,
                              GdkGCValues * values,
                              GdkGCValuesMask values_mask)
{
   g_return_val_if_fail(drawable != NULL, NULL);

   if (GDK_DRAWABLE_DESTROYED(drawable))
      return NULL;

   return ((GdkDrawablePrivate *) drawable)->klass->create_gc(drawable,
                                                              values,
                                                              values_mask);
}

GdkGC *gdk_gc_ref(GdkGC * gc)
{
   GdkGCPrivate *private = (GdkGCPrivate *) gc;

   g_return_val_if_fail(gc != NULL, NULL);
   private->ref_count += 1;

   return gc;
}

void gdk_gc_unref(GdkGC * gc)
{
   GdkGCPrivate *private = (GdkGCPrivate *) gc;

   g_return_if_fail(gc != NULL);
   g_return_if_fail(private->ref_count > 0);

   private->ref_count--;

   if (private->ref_count == 0)
      private->klass->destroy(gc);
}

void gdk_gc_get_values(GdkGC * gc, GdkGCValues * values)
{
   g_return_if_fail(gc != NULL);
   g_return_if_fail(values != NULL);

   ((GdkGCPrivate *) gc)->klass->get_values(gc, values);
}

void
gdk_gc_set_values(GdkGC * gc,
                  GdkGCValues * values, GdkGCValuesMask values_mask)
{
   g_return_if_fail(gc != NULL);
   g_return_if_fail(values != NULL);

   ((GdkGCPrivate *) gc)->klass->set_values(gc, values, values_mask);
}

void gdk_gc_set_foreground(GdkGC * gc, GdkColor * color)
{
   GdkGCValues values;

   g_return_if_fail(gc != NULL);
   g_return_if_fail(color != NULL);

   values.foreground = *color;
   gdk_gc_set_values(gc, &values, GDK_GC_FOREGROUND);
}

void gdk_gc_set_background(GdkGC * gc, GdkColor * color)
{
   GdkGCValues values;

   g_return_if_fail(gc != NULL);
   g_return_if_fail(color != NULL);

   values.background = *color;
   gdk_gc_set_values(gc, &values, GDK_GC_BACKGROUND);
}

void gdk_gc_set_font(GdkGC * gc, GdkFont * font)
{
   GdkGCValues values;

   g_return_if_fail(gc != NULL);
   g_return_if_fail(font != NULL);

   values.font = font;
   gdk_gc_set_values(gc, &values, GDK_GC_FONT);
}

void gdk_gc_set_function(GdkGC * gc, GdkFunction function)
{
   GdkGCValues values;

   g_return_if_fail(gc != NULL);

   values.function = function;
   gdk_gc_set_values(gc, &values, GDK_GC_FUNCTION);
}

void gdk_gc_set_fill(GdkGC * gc, GdkFill fill)
{
   GdkGCValues values;

   g_return_if_fail(gc != NULL);

   values.fill = fill;
   gdk_gc_set_values(gc, &values, GDK_GC_FILL);
}

void gdk_gc_set_tile(GdkGC * gc, GdkPixmap * tile)
{
   GdkGCValues values;

   g_return_if_fail(gc != NULL);

   values.tile = tile;
   gdk_gc_set_values(gc, &values, GDK_GC_TILE);
}

void gdk_gc_set_stipple(GdkGC * gc, GdkPixmap * stipple)
{
   GdkGCValues values;

   g_return_if_fail(gc != NULL);

   values.stipple = stipple;
   gdk_gc_set_values(gc, &values, GDK_GC_STIPPLE);
}

void gdk_gc_set_ts_origin(GdkGC * gc, gint x, gint y)
{
   GdkGCValues values;

   g_return_if_fail(gc != NULL);

   values.ts_x_origin = x;
   values.ts_y_origin = y;

   gdk_gc_set_values(gc, &values, GDK_GC_TS_X_ORIGIN | GDK_GC_TS_Y_ORIGIN);
}

void gdk_gc_set_clip_origin(GdkGC * gc, gint x, gint y)
{
   GdkGCValues values;

   g_return_if_fail(gc != NULL);

   values.clip_x_origin = x;
   values.clip_y_origin = y;

   gdk_gc_set_values(gc, &values,
                     GDK_GC_CLIP_X_ORIGIN | GDK_GC_CLIP_Y_ORIGIN);
}

void gdk_gc_set_clip_mask(GdkGC * gc, GdkBitmap * mask)
{
   GdkGCValues values;

   g_return_if_fail(gc != NULL);

   values.clip_mask = mask;
   gdk_gc_set_values(gc, &values, GDK_GC_CLIP_MASK);
}


void gdk_gc_set_subwindow(GdkGC * gc, GdkSubwindowMode mode)
{
   GdkGCValues values;

   g_return_if_fail(gc != NULL);

   values.subwindow_mode = mode;
   gdk_gc_set_values(gc, &values, GDK_GC_SUBWINDOW);
}

void gdk_gc_set_exposures(GdkGC * gc, gboolean exposures)
{
   GdkGCValues values;

   g_return_if_fail(gc != NULL);

   values.graphics_exposures = exposures;
   gdk_gc_set_values(gc, &values, GDK_GC_EXPOSURES);
}

void
gdk_gc_set_line_attributes(GdkGC * gc,
                           gint line_width,
                           GdkLineStyle line_style,
                           GdkCapStyle cap_style, GdkJoinStyle join_style)
{
   GdkGCValues values;

   values.line_width = line_width;
   values.line_style = line_style;
   values.cap_style = cap_style;
   values.join_style = join_style;

   gdk_gc_set_values(gc, &values,
                     GDK_GC_LINE_WIDTH |
                     GDK_GC_LINE_STYLE |
                     GDK_GC_CAP_STYLE | GDK_GC_JOIN_STYLE);
}

void
gdk_gc_set_dashes(GdkGC * gc, gint dash_offset, gint8 dash_list[], gint n)
{
   g_return_if_fail(gc != NULL);
   g_return_if_fail(dash_list != NULL);

   ((GdkGCPrivate *) gc)->klass->set_dashes(gc, dash_offset, dash_list, n);
}
