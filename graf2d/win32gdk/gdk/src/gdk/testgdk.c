/* testgdk -- validation program for GDK
 * Copyright (C) 2000 Tor Lillqvist
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

/* This program is intended to be used to validate the correctness of
 * the basic graphics operations in a GDK backend. The results of the
 * operations are compared against those produced by a correctly
 * functioning X11 backend (and X11 server).
 *
 * Obviously, only the most basic operations reasonably be expected to
 * produce pixel-by-pixel identical results as the X11 backend. We
 * don't even try to test the correctness of ellipses, tiles or
 * stipples. Not to mention fonts.
 *
 * But, for those operations we do test, we should try to test quite
 * many combinations of parameters.
 *
 * This is just a quick hack, and could be improved a lot. There are
 * copy-pasted code snippets all over that need to be factored out
 * into separate functions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <gdk/gdk.h>

/* CQTESTF -- "Conditionally Quiet TEST with Fail message"
 * macro that prints PASS or FAIL messages
 * parms:	quiet:	if TRUE, only print FAIL messages
 *		expr:	the expression to test
 *		failfmt:if expr is FALSE, print this message
 *			(both format and args)
 */

#define CQTESTF(quiet, expr, failfmt) \
  (tmpb = (expr), \
   (tmpb ? (quiet ? 0 : printf ("PASS: %d %s\n", __LINE__, #expr)) \
         : (printf ("FAIL: %d %s", __LINE__, #expr), \
         printf failfmt, \
         printf ("\n"), \
         retval = FALSE, \
         error (), \
         return_value++)), \
   tmpb)

/* Variations with less parms */

#define CQTEST(quiet, expr) \
  CQTESTF (quiet, expr, (""))

#define TEST(expr) \
  CQTEST (FALSE, expr)

#define QTEST(expr) \
  CQTEST (TRUE, expr)

#define TESTF(expr, failfmt) \
  CQTESTF (FALSE, expr, failfmt)

#define QTESTF(expr, failfmt) \
  CQTESTF (TRUE, expr, failfmt)

#define ASSERT(expr) \
  do { \
    if (!QTEST (expr)) \
      printf ("That is fatal. Goodbye\n"), exit (1);\
  } while (0)

#define N(a) (sizeof(a)/sizeof(*a))

static int return_value = 0;

static gboolean retval;
static gboolean tmpb;

static GdkVisual *system_visual;
static GdkVisual *best_visual;
static GdkWindow *w;

static GdkColormap *system_colourmap;

static GdkColor white, black, red, green, blue, rand1_colour, rand2_colour;

static GdkGC *black_gc, *white_gc, *red_gc, *rand1_gc, *rand2_gc;
static GdkGC *gcs[5];
static GdkGC *black_bitmap_gc;

static int error(void)
{
   /* Place breakpoint here to catch failures */
   return 0;
}

static gboolean test_visual_coherency(gboolean quiet, GdkVisual * visual)
{
   gboolean retval = TRUE;

   CQTEST(quiet, visual->type >= GDK_VISUAL_STATIC_GRAY &&
          visual->type <= GDK_VISUAL_DIRECT_COLOR);
   CQTEST(quiet, visual->depth >= 1 && visual->depth <= 32);
   CQTEST(quiet, visual->byte_order == GDK_LSB_FIRST
          || visual->byte_order == GDK_MSB_FIRST);

   return retval;
}

/* Test visuals
 */
static void test_visuals(void)
{
   GdkVisual *visual;
   GList *visuals;

   system_visual = gdk_visual_get_system();
   ASSERT(system_visual != NULL);
   TEST(test_visual_coherency(FALSE, system_visual));

   best_visual = gdk_visual_get_best();
   if (best_visual != system_visual)
      TEST(test_visual_coherency(TRUE, best_visual));

   visuals = gdk_list_visuals();
   while (visuals) {
      visual = visuals->data;
      TEST(test_visual_coherency(TRUE, visual));
      visuals = visuals->next;
   }
}

/* Create a top-level window used by other tests
 */
static void test_first_window(void)
{
   GdkWindowAttr wa;

   wa.width = 100;
   wa.height = 100;
   wa.window_type = GDK_WINDOW_TOPLEVEL;
   wa.wclass = GDK_INPUT_OUTPUT;

   w = gdk_window_new(NULL, &wa, 0);

   ASSERT(w != NULL);
}

/* Test colourmaps.
 */
static void test_colourmaps(void)
{
   system_colourmap = gdk_colormap_get_system();
   ASSERT(system_colourmap != NULL);
}

/* Test colours.
 */
static void test_colours(void)
{
   ASSERT(gdk_color_white(system_colourmap, &white));
   ASSERT(gdk_color_black(system_colourmap, &black));
   red.red = 65535;
   red.green = red.blue = 0;
   TEST(gdk_colormap_alloc_color(system_colourmap, &red, FALSE, TRUE));

   rand1_colour.red = rand() % 65536;
   rand1_colour.green = rand() % 65536;
   rand1_colour.blue = rand() % 65536;
   TEST(gdk_colormap_alloc_color(system_colourmap, &rand1_colour, FALSE,
                                 TRUE));

   rand2_colour.red = rand() % 65536;
   rand2_colour.green = rand() % 65536;
   rand2_colour.blue = rand() % 65536;
   TEST(gdk_colormap_alloc_color(system_colourmap, &rand2_colour, FALSE,
                                 TRUE));
}

static gboolean test_default_gc(GdkGCValues * gcvalues, gboolean quiet)
{
   gboolean retval = TRUE;

   CQTEST(quiet, gcvalues->foreground.pixel == 0);
   CQTEST(quiet, gcvalues->background.pixel == 1);
   CQTEST(quiet, gcvalues->function == GDK_COPY);
   CQTEST(quiet, gcvalues->fill == GDK_SOLID);
   CQTEST(quiet, gcvalues->tile == NULL);
   CQTEST(quiet, gcvalues->stipple == NULL);
   CQTEST(quiet, gcvalues->clip_mask == NULL);
   CQTEST(quiet, gcvalues->subwindow_mode == GDK_CLIP_BY_CHILDREN);
   CQTEST(quiet, gcvalues->line_width == 0);
   CQTEST(quiet, gcvalues->line_style == GDK_LINE_SOLID);
   CQTEST(quiet, gcvalues->cap_style == GDK_CAP_BUTT);
   CQTEST(quiet, gcvalues->join_style == GDK_JOIN_MITER);

   return retval;
}

/* Create GdkGCs with various values,
 * check that gdk_gc_get_values returns the same,
 * or something reasonably close.
 */
static void test_gcs(void)
{
   GdkPixmap *pixmap;
   GdkGC *gc;
   GdkGCValues gcvalues;
   GdkColor colour;
   GdkFunction function;
   GdkFill fill;
   gboolean retval;

   gc = gdk_gc_new(w);
   gdk_gc_get_values(gc, &gcvalues);
   test_default_gc(&gcvalues, FALSE);

   colour.pixel = 1234;
   gdk_gc_set_foreground(gc, &colour);
   gdk_gc_get_values(gc, &gcvalues);
   TEST(gcvalues.foreground.pixel == 1234);

   colour.pixel = 0;
   gdk_gc_set_foreground(gc, &colour);
   gdk_gc_get_values(gc, &gcvalues);
   TEST(test_default_gc(&gcvalues, TRUE));

   colour.pixel = 5678;
   gdk_gc_set_background(gc, &colour);
   gdk_gc_get_values(gc, &gcvalues);
   TEST(gcvalues.background.pixel == 5678);

   colour.pixel = 1;
   gdk_gc_set_background(gc, &colour);
   gdk_gc_get_values(gc, &gcvalues);
   TEST(test_default_gc(&gcvalues, TRUE));

   retval = TRUE;
   for (function = GDK_COPY; function <= GDK_SET; function++) {
      gdk_gc_set_function(gc, function);
      gdk_gc_get_values(gc, &gcvalues);
      QTEST(gcvalues.function == function);
      gdk_gc_set_function(gc, GDK_COPY);
      gdk_gc_get_values(gc, &gcvalues);
      QTEST(test_default_gc(&gcvalues, TRUE));
   }
   TEST(retval);

   retval = TRUE;
   for (fill = GDK_SOLID; fill <= GDK_OPAQUE_STIPPLED; fill++) {
      gdk_gc_set_fill(gc, fill);
      gdk_gc_get_values(gc, &gcvalues);
      QTEST(gcvalues.fill == fill);
      gdk_gc_set_fill(gc, GDK_SOLID);
      gdk_gc_get_values(gc, &gcvalues);
      QTEST(test_default_gc(&gcvalues, TRUE));
   }
   TEST(retval);

   black_gc = gdk_gc_new(w);
   gdk_gc_copy(black_gc, gc);
   gdk_gc_get_values(black_gc, &gcvalues);
   TEST(test_default_gc(&gcvalues, TRUE));
   gdk_gc_unref(gc);

   gdk_gc_set_foreground(black_gc, &black);
   gdk_gc_get_values(black_gc, &gcvalues);
   TEST(gcvalues.foreground.pixel == black.pixel);

   white_gc = gdk_gc_new(w);

   gdk_gc_set_foreground(white_gc, &white);
   gdk_gc_get_values(white_gc, &gcvalues);
   TEST(gcvalues.foreground.pixel == white.pixel);

   red_gc = gdk_gc_new(w);
   gdk_gc_set_foreground(red_gc, &red);
   gdk_gc_get_values(red_gc, &gcvalues);
   TEST(gcvalues.foreground.pixel == red.pixel);

   rand1_gc = gdk_gc_new(w);
   gdk_gc_set_foreground(rand1_gc, &rand1_colour);
   gdk_gc_get_values(rand1_gc, &gcvalues);
   TESTF(gcvalues.foreground.pixel == rand1_colour.pixel,
         (" %#06x != %#06x", gcvalues.foreground.pixel,
          rand1_colour.pixel));

   rand2_gc = gdk_gc_new(w);
   gdk_gc_set_foreground(rand2_gc, &rand2_colour);
   gdk_gc_get_values(rand2_gc, &gcvalues);
   TESTF(gcvalues.foreground.pixel == rand2_colour.pixel,
         (" %#06x != %#06x", gcvalues.foreground.pixel,
          rand2_colour.pixel));

   gcs[0] = black_gc;
   gcs[1] = white_gc;
   gcs[2] = red_gc;
   gcs[3] = rand1_gc;
   gcs[4] = rand2_gc;

   pixmap = gdk_pixmap_new(NULL, 1, 1, 1);
   black_bitmap_gc = gdk_gc_new(pixmap);
   gdk_pixmap_unref(pixmap);
}

/* Create pixmaps, check that properties are as expected.
 * No graphic operations tested yet.
 */
static void test_pixmaps(gint depth)
{
   GdkPixmap *pixmap;
   GdkImage *image;
   GdkGC *gc;
   gint width, height;
   gint w, h;
   gboolean retval = TRUE;

   for (width = 1; width <= 64; width += 2)
      for (height = 1; height <= 32; height += 3) {
         pixmap = gdk_pixmap_new(NULL, width, height, depth);
         ASSERT(pixmap != NULL);
         gdk_window_get_size(pixmap, &w, &h);
         QTESTF(w == width, (" w:%d", w));
         QTESTF(h == height, (" h:%d", h));
         image = gdk_image_get(pixmap, 0, 0, w, h);
         QTEST(image != NULL);
         QTEST(image->width == width);
         QTEST(image->height == height);
         QTEST(image->depth == depth);
         gdk_image_destroy(image);
         gdk_pixmap_unref(pixmap);
      }
   TEST(retval);
}

/* Ditto for images.
 */
static void test_images(void)
{
   GdkImage *image;
   GdkImageType image_type;
   gint width, height;
   gboolean retval = TRUE;

   for (width = 1; width <= 64; width += 3)
      for (height = 1; height <= 32; height += 7)
         for (image_type = GDK_IMAGE_NORMAL;
              image_type <= GDK_IMAGE_FASTEST; image_type++) {
            image =
                gdk_image_new(image_type, system_visual, width, height);
            if (image == NULL && image_type == GDK_IMAGE_SHARED)
               /* Ignore failure to create shared image,
                * display might not be local.
                */
               ;
            else {
               ASSERT(image != NULL);
               QTEST(image->width == width);
               QTEST(image->height == height);
               QTEST(image->depth == system_visual->depth);
               QTEST(image->bpp >= (image->depth - 1) / 8 + 1);
               QTEST(image->mem != NULL);
               gdk_image_destroy(image);
            }
         }
   TEST(retval);
}

/* Test creating temp windows.
 */
static void test_temp_windows(void)
{
   GdkWindow *window;
   GdkWindowAttr wa;
   GdkVisual *visual;
   gint width, height;
   gint w, h, x, y, d;
   gboolean retval = TRUE;

   wa.window_type = GDK_WINDOW_TEMP;
   wa.wclass = GDK_INPUT_OUTPUT;

   for (width = 1; width <= 64; width += 4)
      for (height = 1; height <= 32; height += 7) {
         wa.width = width;
         wa.height = height;
         window = gdk_window_new(NULL, &wa, 0);
         ASSERT(window != NULL);
         gdk_window_get_geometry(window, &x, &y, &w, &h, &d);
         QTESTF(w == width, ("w:%d", w));
         QTESTF(h == height, ("h:%d", h));
         gdk_window_show(window);
         gdk_window_get_geometry(window, &x, &y, &w, &h, &d);
         QTESTF(w == width, ("w:%d", w));
         QTESTF(h == height, ("h:%d", h));
         gdk_window_resize(window, 37, 19);
         gdk_window_get_geometry(window, &x, &y, &w, &h, &d);
         QTESTF(w == 37, ("w:%d", w));
         QTESTF(h == 19, ("h:%d", h));
         visual = gdk_window_get_visual(window);
         QTEST(visual == system_visual);
         gdk_window_hide(window);
         gdk_window_unref(window);
      }
   TEST(retval);
}

static void
test_gc_function(GdkFunction function,
                 guint32 oldpixel,
                 guint32 newpixel, guint32 foreground, guint32 mask)
{
   switch (function) {
   case GDK_COPY:
      QTEST(newpixel == (foreground & mask));
      break;
   case GDK_INVERT:
      QTEST(newpixel == ((~oldpixel) & mask));
      break;
   case GDK_XOR:
      QTEST(newpixel == ((oldpixel ^ foreground) & mask));
      break;
   case GDK_CLEAR:
      QTEST(newpixel == 0);
      break;
   case GDK_AND:
      QTEST(newpixel == ((oldpixel & foreground) & mask));
      break;
   case GDK_AND_REVERSE:
      QTEST(newpixel == (((~oldpixel) & foreground) & mask));
      break;
   case GDK_AND_INVERT:
      QTEST(newpixel == ((oldpixel & (~foreground)) & mask));
      break;
   case GDK_NOOP:
      QTEST(newpixel == (oldpixel & mask));
      break;
   case GDK_OR:
      QTEST(newpixel == ((oldpixel | foreground) & mask));
      break;
   case GDK_EQUIV:
      QTEST(newpixel == ((oldpixel ^ (~foreground)) & mask));
      break;
   case GDK_OR_REVERSE:
      QTEST(newpixel == (((~oldpixel) | foreground) & mask));
      break;
   case GDK_COPY_INVERT:
      QTEST(newpixel == ((~foreground) & mask));
      break;
   case GDK_OR_INVERT:
      QTEST(newpixel == ((oldpixel | (~foreground)) & mask));
      break;
   case GDK_NAND:
      QTEST(newpixel == (((~oldpixel) | (~foreground)) & mask));
      break;
   case GDK_SET:
      QTEST(newpixel == ((~0) & mask));
      break;
   default:
      ASSERT(FALSE);
   }
}

static void
test_one_point_on_drawable(GdkDrawable * drawable, GdkGC * gc, int depth)
{
   GdkImage *image;
   GdkGCValues gcvalues;
   gint xoff, yoff;
   guint32 oldpixels[3][3], newpixel, mask;
   const gint x = 4;
   const gint y = 5;

   gdk_gc_get_values(gc, &gcvalues);

   image = gdk_image_get(drawable, x + -1, y + -1, 3, 3);
   QTEST(image != NULL);
   for (xoff = -1; xoff <= 1; xoff++)
      for (yoff = -1; yoff <= 1; yoff++) {
         oldpixels[xoff + 1][yoff + 1] =
             gdk_image_get_pixel(image, xoff + 1, yoff + 1);
      }
   gdk_image_destroy(image);

   if (depth == 32)
      mask = 0xFFFFFFFF;
   else
      mask = (1 << depth) - 1;

   gdk_draw_point(drawable, gc, x, y);

   image = gdk_image_get(drawable, x - 1, y - 1, 3, 3);
   QTEST(image != NULL);
   for (xoff = -1; xoff <= 1; xoff++)
      for (yoff = -1; yoff <= 1; yoff++) {
         newpixel = gdk_image_get_pixel(image, xoff + 1, yoff + 1);
         if (xoff == 0 && yoff == 0)
            test_gc_function(gcvalues.function, oldpixels[1][1], newpixel,
                             gcvalues.foreground.pixel, mask);
         else
            QTEST(newpixel == oldpixels[xoff + 1][yoff + 1]);
      }
   gdk_image_destroy(image);
}


/* Test drawing points.
 */
static void test_points(void)
{
   GdkPixmap *pixmap;
   GdkWindow *window;
   GdkFunction function;
   gint width, height;
   int i, j;

   width = 8;
   height = 8;
   pixmap = gdk_pixmap_new(w, width, height, -1);

   for (i = 0; i < N(gcs); i++)
      for (j = 0; j < N(gcs); j++)
         for (function = GDK_COPY; function <= GDK_SET; function++) {
            gdk_draw_rectangle(pixmap, gcs[i], TRUE, 0, 0, width, height);
            gdk_gc_set_function(gcs[j], function);
            test_one_point_on_drawable(pixmap, gcs[j],
                                       system_visual->depth);
            gdk_gc_set_function(gcs[j], GDK_COPY);
         }

   gdk_pixmap_unref(pixmap);

   pixmap = gdk_pixmap_new(w, width, height, 1);
   test_one_point_on_drawable(pixmap, black_bitmap_gc, 1);
   for (function = GDK_COPY; function <= GDK_SET; function++) {
      gdk_gc_set_function(black_bitmap_gc, function);
      test_one_point_on_drawable(pixmap, black_bitmap_gc, 1);
   }

   gdk_pixmap_unref(pixmap);
}

static void
test_one_line_on_drawable(GdkDrawable * drawable,
                          GdkGC * gc, int depth, gboolean horisontal)
{
   GdkImage *oldimage, *newimage;
   GdkGCValues gcvalues;
   gint line_width;
   gint w, h;
   gint w_up, w_down, w_left, w_right;
   gint x, y;
   guint32 oldpixel, newpixel, mask;

   gdk_gc_get_values(gc, &gcvalues);
   line_width = gcvalues.line_width > 0 ? gcvalues.line_width : 1;
   w_up = w_left = line_width / 2;
   w_down = w_right =
       (line_width & 1) ? line_width / 2 : line_width / 2 - 1;
   gdk_window_get_size(drawable, &w, &h);
   oldimage = gdk_image_get(drawable, 0, 0, w, h);

   if (depth == 32)
      mask = 0xFFFFFFFF;
   else
      mask = (1 << depth) - 1;

   if (horisontal) {
      const gint x1 = 10;
      const gint y1 = 10;
      const gint x2 = 13;
      const gint y2 = y1;

      gdk_draw_line(drawable, gc, x1, y1, x2, y2);
      newimage = gdk_image_get(drawable, 0, 0, w, h);
      for (x = x1 - 1; x <= x2 + 1; x++)
         for (y = y1 - w_up - 1; y <= y1 + w_down + 1; y++) {
            oldpixel = gdk_image_get_pixel(oldimage, x, y);
            newpixel = gdk_image_get_pixel(newimage, x, y);
            if (x >= x1 && x < x2 && y >= y1 - w_up && y <= y1 + w_down)
               test_gc_function(gcvalues.function, oldpixel, newpixel,
                                gcvalues.foreground.pixel, mask);
            else
               QTEST(oldpixel == newpixel);
         }
   } else {                     /* vertical */

      const gint x1 = 10;
      const gint y1 = 10;
      const gint x2 = 10;
      const gint y2 = 13;

      gdk_draw_line(drawable, gc, x1, y1, x2, y2);
      newimage = gdk_image_get(drawable, 0, 0, w, h);
      for (x = x1 - w_left - 1; x <= x1 + w_right + 1; x++)
         for (y = y1 - 1; y <= y2 + 1; y++) {
            oldpixel = gdk_image_get_pixel(oldimage, x, y);
            newpixel = gdk_image_get_pixel(newimage, x, y);
            if (x >= x1 - w_left && x <= x1 + w_right && y >= y1 && y < y2)
               test_gc_function(gcvalues.function, oldpixel, newpixel,
                                gcvalues.foreground.pixel, mask);
            else
               QTEST(oldpixel == newpixel);
         }
   }

   gdk_image_destroy(oldimage);
   gdk_image_destroy(newimage);
}

/* Test drawing lines.
 */
static void test_lines(void)
{
   GdkPixmap *pixmap;
   GdkFunction function;
   gint width;
   int i, j;
   gboolean horisontal = TRUE;

   pixmap = gdk_pixmap_new(w, 30, 30, -1);

   for (i = 0; i < N(gcs); i++)
      for (j = 0; j < N(gcs); j++)
         for (function = GDK_COPY; function <= GDK_SET; function++)
            for (width = 1; width <= 4; width++) {
               gdk_draw_rectangle(pixmap, gcs[i], TRUE, 0, 0, 30, 30);
               gdk_gc_set_function(gcs[j], function);
               gdk_gc_set_line_attributes(gcs[j], width,
                                          GDK_LINE_SOLID, GDK_CAP_BUTT,
                                          GDK_JOIN_MITER);
               test_one_line_on_drawable(pixmap, gcs[j],
                                         system_visual->depth, horisontal);
               /* Toggle between horisontal and vertical... */
               horisontal = !horisontal;
               gdk_gc_set_function(gcs[j], GDK_COPY);
            }

   gdk_pixmap_unref(pixmap);
}

static void
test_one_rectangle_on_drawable(GdkDrawable * drawable,
                               GdkGC * gc, int depth, gboolean filled)
{
   GdkImage *oldimage, *newimage;
   GdkGCValues gcvalues;
   gint line_width;
   gint w, h;
   gint w_up, w_down, w_left, w_right;
   gint x, y;
   guint32 oldpixel, newpixel, mask;
   const gint x0 = 10;
   const gint y0 = 13;
   const gint width = 7;
   const gint height = 9;

   gdk_gc_get_values(gc, &gcvalues);

   if (!filled) {
      line_width = gcvalues.line_width > 0 ? gcvalues.line_width : 1;
      w_up = w_left = line_width / 2;
      w_down = w_right =
          (line_width & 1) ? line_width / 2 : line_width / 2 - 1;
   }

   gdk_window_get_size(drawable, &w, &h);
   oldimage = gdk_image_get(drawable, 0, 0, w, h);

   if (depth == 32)
      mask = 0xFFFFFFFF;
   else
      mask = (1 << depth) - 1;

   gdk_draw_rectangle(drawable, gc, filled, x0, y0, width, height);
   newimage = gdk_image_get(drawable, 0, 0, w, h);

   for (x = x0 - 1; x <= x0 + width + 1; x++)
      for (y = y0 - 1; y < y0 + height + 1; y++) {
         oldpixel = gdk_image_get_pixel(oldimage, x, y);
         newpixel = gdk_image_get_pixel(newimage, x, y);

         if (filled) {
            if (x >= x0 && x < x0 + width && y >= y0 && y < y0 + height)
               test_gc_function(gcvalues.function, oldpixel, newpixel,
                                gcvalues.foreground.pixel, mask);
            else
               QTEST(oldpixel == newpixel);
         } else {
            if ((x >= x0 - w_left && x <= x0 + width + w_right &&
                 y >= y0 - w_up && y <= y0 + w_down) ||
                (x >= x0 - w_left && x <= x0 + width + w_right &&
                 y >= y0 + height - w_up && y <= y0 + height + w_down) ||
                (x >= x0 - w_left && x <= x0 + w_right &&
                 y >= y0 - w_up && y <= y0 + height + w_down) ||
                (x >= x0 + width - w_left && x <= x0 + width + w_right &&
                 y >= y0 - w_up && y <= y0 + height + w_down))
               test_gc_function(gcvalues.function, oldpixel, newpixel,
                                gcvalues.foreground.pixel, mask);
            else
               QTEST(oldpixel == newpixel);
         }
      }

   gdk_image_destroy(oldimage);
   gdk_image_destroy(newimage);
}

/* Test drawing rectangles.
 */
static void test_rectangles(void)
{
   GdkPixmap *pixmap;
   GdkFunction function;
   gint width;
   int i, j;
   gboolean filled = FALSE;

   pixmap = gdk_pixmap_new(w, 30, 30, -1);

   for (i = 0; i < N(gcs); i++)
      for (j = 0; j < N(gcs); j++)
         for (function = GDK_COPY; function <= GDK_SET; function++)
            for (width = 1; width <= 4; width++) {
               gdk_draw_rectangle(pixmap, gcs[i], TRUE, 0, 0, 30, 30);
               gdk_gc_set_function(gcs[j], function);
               gdk_gc_set_line_attributes(gcs[j], width,
                                          GDK_LINE_SOLID, GDK_CAP_BUTT,
                                          GDK_JOIN_MITER);
               test_one_rectangle_on_drawable(pixmap, gcs[j],
                                              system_visual->depth,
                                              filled);
               filled = !filled;
               gdk_gc_set_function(gcs[j], GDK_COPY);
            }

   gdk_pixmap_unref(pixmap);
}

static void
test_some_arcs_on_drawable(GdkDrawable * drawable,
                           GdkGC * gc, int depth, gboolean filled)
{
   GdkImage *oldimage, *newimage;
   GdkGCValues gcvalues;
   gint line_width;
   gint w, h;
   gint w_up, w_down, w_left, w_right;
   gint x, y;
   guint32 oldpixel, newpixel, mask;
   /* XXX */
   const gint x0 = 10;
   const gint y0 = 13;
   const gint width = 7;
   const gint height = 9;

   gdk_gc_get_values(gc, &gcvalues);

   if (!filled) {
      line_width = gcvalues.line_width > 0 ? gcvalues.line_width : 1;
      w_up = w_left = line_width / 2;
      w_down = w_right =
          (line_width & 1) ? line_width / 2 : line_width / 2 - 1;
   }

   gdk_window_get_size(drawable, &w, &h);
   oldimage = gdk_image_get(drawable, 0, 0, w, h);

   if (depth == 32)
      mask = 0xFFFFFFFF;
   else
      mask = (1 << depth) - 1;

   /* XXX */
   newimage = gdk_image_get(drawable, 0, 0, w, h);

   for (x = x0 - 1; x <= x0 + width + 1; x++)
      for (y = y0 - 1; y < y0 + height + 1; y++) {
         oldpixel = gdk_image_get_pixel(oldimage, x, y);
         newpixel = gdk_image_get_pixel(newimage, x, y);

         if (filled) {
            /* XXX */
         } else {
            /* XXX */
         }
      }

   gdk_image_destroy(oldimage);
   gdk_image_destroy(newimage);
}

/* Test drawing arcs. Results don't have to be exactly as on X11,
 * but "close".
 */
static void test_arcs(void)
{
   GdkPixmap *pixmap;
   GdkFunction function;
   gint width;
   int i, j;
   gboolean filled = FALSE;

   pixmap = gdk_pixmap_new(w, 30, 30, -1);

   for (i = 0; i < N(gcs); i++)
      for (j = 0; j < N(gcs); j++)
         for (function = GDK_COPY; function <= GDK_SET; function++)
            for (width = 1; width <= 4; width++) {
               gdk_draw_rectangle(pixmap, gcs[i], TRUE, 0, 0, 30, 30);
               gdk_gc_set_function(gcs[j], function);
               gdk_gc_set_line_attributes(gcs[j], width,
                                          GDK_LINE_SOLID, GDK_CAP_BUTT,
                                          GDK_JOIN_MITER);
               test_some_arcs_on_drawable(pixmap, gcs[j],
                                          system_visual->depth, filled);
               filled = !filled;
               gdk_gc_set_function(gcs[j], GDK_COPY);
            }

   gdk_pixmap_unref(pixmap);
}

/* Test region operations.
 */
static void test_regions(void)
{
}

static void tests(void)
{
   srand(time(NULL));

   test_visuals();
   test_first_window();
   test_colourmaps();
   test_colours();
   test_gcs();
   test_pixmaps(1);
   test_pixmaps(system_visual->depth);
   if (best_visual->depth != system_visual->depth)
      test_pixmaps(best_visual->depth);
   test_images();
   test_temp_windows();
   test_points();
   test_lines();
   test_rectangles();
   test_arcs();
   test_regions();
}

int main(int argc, char **argv)
{
   gdk_init(&argc, &argv);

   tests();

   return return_value;
}
