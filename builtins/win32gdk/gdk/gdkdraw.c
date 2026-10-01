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

#include "gdkdrawable.h"
#include "gdkprivate.h"
#include "gdkwindow.h"

/* Manipulation of drawables
 */
GdkDrawable *gdk_drawable_alloc(void)
{
   GdkDrawablePrivate *private = g_new(GdkDrawablePrivate, 1);
   GdkDrawable *drawable = (GdkDrawable *) private;

   drawable->user_data = NULL;

   private->ref_count = 1;
   private->destroyed = FALSE;
   private->klass = NULL;
   private->klass_data = NULL;
   private->window_type = GDK_WINDOW_CHILD;

   private->width = 1;
   private->height = 1;

   private->colormap = NULL;

   return drawable;
}

void
gdk_drawable_set_data(GdkDrawable * drawable,
                      const gchar * key,
                      gpointer data, GDestroyNotify destroy_func)
{
   g_dataset_set_data_full(drawable, key, data, destroy_func);
}

void gdk_drawable_get_data(GdkDrawable * drawable, const gchar * key)
{
   g_dataset_get_data(drawable, key);
}

GdkDrawableType gdk_drawable_get_type(GdkDrawable * drawable)
{
   g_return_val_if_fail(drawable != NULL, (GdkDrawableType) - 1);

   return GDK_DRAWABLE_TYPE(drawable);
}

void
gdk_drawable_get_size(GdkDrawable * drawable, gint * width, gint * height)
{
   GdkDrawablePrivate *drawable_private;

   g_return_if_fail(drawable != NULL);

   drawable_private = (GdkDrawablePrivate *) drawable;

   if (width)
      *width = drawable_private->width;
   if (height)
      *height = drawable_private->height;
}

GdkVisual *gdk_drawable_get_visual(GdkDrawable * drawable)
{
   GdkColormap *colormap;

   g_return_val_if_fail(drawable != NULL, NULL);

   colormap = gdk_drawable_get_colormap(drawable);
   return colormap ? gdk_colormap_get_visual(colormap) : NULL;
}

GdkDrawable *gdk_drawable_ref(GdkDrawable * drawable)
{
   GdkDrawablePrivate *private = (GdkDrawablePrivate *) drawable;
   g_return_val_if_fail(drawable != NULL, NULL);

   private->ref_count += 1;
   return drawable;
}

void gdk_drawable_unref(GdkDrawable * drawable)
{
   GdkDrawablePrivate *private = (GdkDrawablePrivate *) drawable;

   if ((drawable == NULL) || (private->ref_count <= 0))
      return;
   g_return_if_fail(drawable != NULL);
   g_return_if_fail(private->ref_count > 0);

   private->ref_count -= 1;
   if (private->ref_count == 0) {
      GDK_NOTE(MISC, g_print("calling class destroy method\n"));
      private->klass->destroy(drawable);
      g_dataset_destroy(drawable);
      g_free(drawable);
   }
}

/* Drawing
 */
void gdk_draw_point(GdkDrawable * drawable, GdkGC * gc, gint x, gint y)
{
   GdkGCPrivate *gc_private;
   GdkPoint point;

   g_return_if_fail(drawable != NULL);
   g_return_if_fail(gc != NULL);

   if (GDK_DRAWABLE_DESTROYED(drawable))
      return;
   gc_private = (GdkGCPrivate *) gc;

   point.x = x;
   point.y = y;

   ((GdkDrawablePrivate *) drawable)->klass->draw_points(drawable, gc,
                                                         &point, 1);
}

void
gdk_draw_line(GdkDrawable * drawable,
              GdkGC * gc, gint x1, gint y1, gint x2, gint y2)
{
   GdkGCPrivate *gc_private;
   GdkSegment segment;

   g_return_if_fail(drawable != NULL);
   g_return_if_fail(gc != NULL);

   if (GDK_DRAWABLE_DESTROYED(drawable))
      return;
   gc_private = (GdkGCPrivate *) gc;

   segment.x1 = x1;
   segment.y1 = y1;
   segment.x2 = x2;
   segment.y2 = y2;
   ((GdkDrawablePrivate *) drawable)->klass->draw_segments(drawable, gc,
                                                           &segment, 1);
}

void
gdk_draw_rectangle(GdkDrawable * drawable,
                   GdkGC * gc,
                   gint filled, gint x, gint y, gint width, gint height)
{
   GdkDrawablePrivate *drawable_private;

   g_return_if_fail(drawable != NULL);
   g_return_if_fail(gc != NULL);

   drawable_private = (GdkDrawablePrivate *) drawable;
   if (GDK_DRAWABLE_DESTROYED(drawable))
      return;

   if (width < 0)
      width = drawable_private->width;
   if (height < 0)
      height = drawable_private->height;

   ((GdkDrawablePrivate *) drawable)->klass->draw_rectangle(drawable, gc,
                                                            filled, x, y,
                                                            width, height);
}

void
gdk_draw_arc(GdkDrawable * drawable,
             GdkGC * gc,
             gint filled,
             gint x,
             gint y, gint width, gint height, gint angle1, gint angle2)
{
   GdkDrawablePrivate *drawable_private;
   GdkGCPrivate *gc_private;

   g_return_if_fail(drawable != NULL);
   g_return_if_fail(gc != NULL);

   drawable_private = (GdkDrawablePrivate *) drawable;
   if (GDK_DRAWABLE_DESTROYED(drawable))
      return;
   gc_private = (GdkGCPrivate *) gc;

   if (width < 0)
      width = drawable_private->width;
   if (height < 0)
      height = drawable_private->height;

   ((GdkDrawablePrivate *) drawable)->klass->draw_arc(drawable, gc, filled,
                                                      x, y, width, height,
                                                      angle1, angle2);
}

void
gdk_draw_polygon(GdkDrawable * drawable,
                 GdkGC * gc, gint filled, GdkPoint * points, gint npoints)
{
   g_return_if_fail(drawable != NULL);
   g_return_if_fail(gc != NULL);

   if (GDK_DRAWABLE_DESTROYED(drawable))
      return;

   ((GdkDrawablePrivate *) drawable)->klass->draw_polygon(drawable, gc,
                                                          filled, points,
                                                          npoints);
}

/* gdk_draw_string
 *
 * Modified by Li-Da Lho to draw 16 bits and Multibyte strings
 *
 * Interface changed: add "GdkFont *font" to specify font or fontset explicitly
 */
void
gdk_draw_string(GdkDrawable * drawable,
                GdkFont * font,
                GdkGC * gc, gint x, gint y, const gchar * string)
{
   gdk_draw_text(drawable, font, gc, x, y, string,
                 _gdk_font_strlen(font, string));
}

/* gdk_draw_text
 *
 * Modified by Li-Da Lho to draw 16 bits and Multibyte strings
 *
 * Interface changed: add "GdkFont *font" to specify font or fontset explicitly
 */
void
gdk_draw_text(GdkDrawable * drawable,
              GdkFont * font,
              GdkGC * gc,
              gint x, gint y, const gchar * text, gint text_length)
{
   g_return_if_fail(drawable != NULL);
   g_return_if_fail(font != NULL);
   g_return_if_fail(gc != NULL);
   g_return_if_fail(text != NULL);

   ((GdkDrawablePrivate *) drawable)->klass->draw_text(drawable, font, gc,
                                                       x, y, text,
                                                       text_length);
}

void
gdk_draw_text_wc(GdkDrawable * drawable,
                 GdkFont * font,
                 GdkGC * gc,
                 gint x, gint y, const GdkWChar * text, gint text_length)
{
   g_return_if_fail(drawable != NULL);
   g_return_if_fail(font != NULL);
   g_return_if_fail(gc != NULL);
   g_return_if_fail(text != NULL);

   ((GdkDrawablePrivate *) drawable)->klass->draw_text_wc(drawable, font,
                                                          gc, x, y, text,
                                                          text_length);
}

void
gdk_draw_drawable(GdkDrawable * drawable,
                  GdkGC * gc,
                  GdkDrawable * src,
                  gint xsrc,
                  gint ysrc,
                  gint xdest, gint ydest, gint width, gint height)
{
   g_return_if_fail(drawable != NULL);
   g_return_if_fail(src != NULL);
   g_return_if_fail(gc != NULL);

   if (GDK_DRAWABLE_DESTROYED(drawable) || GDK_DRAWABLE_DESTROYED(src))
      return;

   if (width == -1)
      width = ((GdkDrawablePrivate *) src)->width;
   if (height == -1)
      height = ((GdkDrawablePrivate *) src)->height;

   ((GdkDrawablePrivate *) drawable)->klass->draw_drawable(drawable, gc,
                                                           src, xsrc, ysrc,
                                                           xdest, ydest,
                                                           width, height);
}

void
gdk_draw_image(GdkDrawable * drawable,
               GdkGC * gc,
               GdkImage * image,
               gint xsrc,
               gint ysrc, gint xdest, gint ydest, gint width, gint height)
{
   GdkImagePrivate *image_private;

   g_return_if_fail(drawable != NULL);
   g_return_if_fail(image != NULL);
   g_return_if_fail(gc != NULL);

   image_private = (GdkImagePrivate *) image;

   if (width == -1)
      width = image->width;
   if (height == -1)
      height = image->height;


   image_private->klass->image_put(image, drawable, gc, xsrc, ysrc,
                                   xdest, ydest, width, height);
}

void
gdk_draw_points(GdkDrawable * drawable,
                GdkGC * gc, GdkPoint * points, gint npoints)
{
   g_return_if_fail(drawable != NULL);
   g_return_if_fail((points != NULL) && (npoints > 0));
   g_return_if_fail(gc != NULL);
   g_return_if_fail(npoints >= 0);

   if (npoints == 0)
      return;

   if (GDK_DRAWABLE_DESTROYED(drawable))
      return;

   ((GdkDrawablePrivate *) drawable)->klass->draw_points(drawable, gc,
                                                         points, npoints);
}

void
gdk_draw_segments(GdkDrawable * drawable,
                  GdkGC * gc, GdkSegment * segs, gint nsegs)
{
   g_return_if_fail(drawable != NULL);

   if (nsegs == 0)
      return;

   g_return_if_fail(segs != NULL);
   g_return_if_fail(gc != NULL);
   g_return_if_fail(nsegs >= 0);

   if (GDK_DRAWABLE_DESTROYED(drawable))
      return;

   ((GdkDrawablePrivate *) drawable)->klass->draw_segments(drawable, gc,
                                                           segs, nsegs);
}

void
gdk_draw_lines(GdkDrawable * drawable,
               GdkGC * gc, GdkPoint * points, gint npoints)
{

   g_return_if_fail(drawable != NULL);
   g_return_if_fail(points != NULL);
   g_return_if_fail(gc != NULL);
   g_return_if_fail(npoints >= 0);

   if (npoints == 0)
      return;

   if (GDK_DRAWABLE_DESTROYED(drawable))
      return;

   ((GdkDrawablePrivate *) drawable)->klass->draw_lines(drawable, gc,
                                                        points, npoints);
}
