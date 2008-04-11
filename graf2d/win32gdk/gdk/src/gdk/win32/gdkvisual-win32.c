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

#include "gdkvisual.h"
#include "gdkprivate-win32.h"

static void gdk_visual_decompose_mask(gulong mask,
                                      gint * shift, gint * prec);

static GdkVisualPrivate *system_visual;

static gint available_depths[1];

static GdkVisualType available_types[1];

#ifdef G_ENABLE_DEBUG

static const gchar *visual_names[] = {
   "static gray",
   "grayscale",
   "static color",
   "pseudo color",
   "true color",
   "direct color",
};

#endif                          /* G_ENABLE_DEBUG */

void gdk_visual_init(void)
{
   struct {
      BITMAPINFOHEADER bi;
      union {
         RGBQUAD colors[256];
         DWORD fields[256];
      } u;
   } bmi;
   HBITMAP hbm;

   int rastercaps, numcolors, sizepalette, bitspixel;

   system_visual = g_new(GdkVisualPrivate, 1);

   bitspixel = GetDeviceCaps(gdk_DC, BITSPIXEL);
   rastercaps = GetDeviceCaps(gdk_DC, RASTERCAPS);
   system_visual->xvisual = g_new(Visual, 1);
   system_visual->xvisual->visualid = 0;
   system_visual->xvisual->bitspixel = bitspixel;

   // temporary solves a strange bug in root in 16 bit mode:
   // axis labels background is not the same color than the 
   // pad background...
   if (bitspixel == 16) bitspixel = 24; 

   if (rastercaps & RC_PALETTE) {
      g_error
          ("Palettized display (%d-colour) mode not supported on Windows.",
           GetDeviceCaps(gdk_DC, SIZEPALETTE));
      system_visual->visual.type = GDK_VISUAL_PSEUDO_COLOR;
      numcolors = GetDeviceCaps(gdk_DC, NUMCOLORS);
      sizepalette = GetDeviceCaps(gdk_DC, SIZEPALETTE);
      system_visual->xvisual->map_entries = sizepalette;
   } else if (bitspixel == 1) {
      system_visual->visual.type = GDK_VISUAL_STATIC_GRAY;
      system_visual->xvisual->map_entries = 2;
   } else if (bitspixel == 4) {
      system_visual->visual.type = GDK_VISUAL_STATIC_COLOR;
      system_visual->xvisual->map_entries = 16;
   } else if (bitspixel == 8) {
      system_visual->visual.type = GDK_VISUAL_STATIC_COLOR;
      system_visual->xvisual->map_entries = 256;
   } else if (bitspixel == 16) {
      system_visual->visual.type = GDK_VISUAL_TRUE_COLOR;
#if 1
      /* This code by Mike Enright,
       * see http://www.users.cts.com/sd/m/menright/display.html
       */
      memset(&bmi, 0, sizeof(bmi));
      bmi.bi.biSize = sizeof(bmi.bi);

      hbm = CreateCompatibleBitmap(gdk_DC, 1, 1);
      GetDIBits(gdk_DC, hbm, 0, 1, NULL,
                (BITMAPINFO *) & bmi, DIB_RGB_COLORS);
      GetDIBits(gdk_DC, hbm, 0, 1, NULL,
                (BITMAPINFO *) & bmi, DIB_RGB_COLORS);
      DeleteObject(hbm);

      if (bmi.bi.biCompression != BI_BITFIELDS) {
         /* Either BI_RGB or BI_RLE_something
          * .... or perhaps (!!) something else.
          * Theoretically biCompression might be
          * mmioFourCC('c','v','i','d') but I doubt it.
          */
         if (bmi.bi.biCompression == BI_RGB) {
            /* It's 555 */
            bitspixel = 15;
            system_visual->visual.red_mask = 0x00007C00;
            system_visual->visual.green_mask = 0x000003E0;
            system_visual->visual.blue_mask = 0x0000001F;
         } else {
            g_assert_not_reached();
         }
      } else {
         DWORD allmasks =
             bmi.u.fields[0] | bmi.u.fields[1] | bmi.u.fields[2];
         int k = 0;
         while (allmasks) {
            if (allmasks & 1)
               k++;
            allmasks /= 2;
         }
         bitspixel = k;
         system_visual->visual.red_mask = bmi.u.fields[0];
         system_visual->visual.green_mask = bmi.u.fields[1];
         system_visual->visual.blue_mask = bmi.u.fields[2];
      }
#else
      /* Old, incorrect (but still working) code. */
#if 0
      system_visual->visual.red_mask = 0x0000F800;
      system_visual->visual.green_mask = 0x000007E0;
      system_visual->visual.blue_mask = 0x0000001F;
#else
      system_visual->visual.red_mask = 0x00007C00;
      system_visual->visual.green_mask = 0x000003E0;
      system_visual->visual.blue_mask = 0x0000001F;
#endif
#endif
   } else if (bitspixel == 24 || bitspixel == 32) {
//      bitspixel = 24;
      system_visual->visual.type = GDK_VISUAL_TRUE_COLOR;
      system_visual->visual.red_mask = 0x00FF0000;
      system_visual->visual.green_mask = 0x0000FF00;
      system_visual->visual.blue_mask = 0x000000FF;
   } else
      g_error("gdk_visual_init: unsupported BITSPIXEL: %d\n", bitspixel);

   system_visual->visual.depth = bitspixel;
   system_visual->visual.byte_order = GDK_LSB_FIRST;
   system_visual->visual.bits_per_rgb = 42;	/* Not used? */

   if ((system_visual->visual.type == GDK_VISUAL_TRUE_COLOR) ||
       (system_visual->visual.type == GDK_VISUAL_DIRECT_COLOR)) {
      gdk_visual_decompose_mask(system_visual->visual.red_mask,
                                &system_visual->visual.red_shift,
                                &system_visual->visual.red_prec);

      gdk_visual_decompose_mask(system_visual->visual.green_mask,
                                &system_visual->visual.green_shift,
                                &system_visual->visual.green_prec);

      gdk_visual_decompose_mask(system_visual->visual.blue_mask,
                                &system_visual->visual.blue_shift,
                                &system_visual->visual.blue_prec);
      system_visual->xvisual->map_entries =
          1 << (MAX(system_visual->visual.red_prec,
                    MAX(system_visual->visual.green_prec,
                        system_visual->visual.blue_prec)));
   } else {
      system_visual->visual.red_mask = 0;
      system_visual->visual.red_shift = 0;
      system_visual->visual.red_prec = 0;

      system_visual->visual.green_mask = 0;
      system_visual->visual.green_shift = 0;
      system_visual->visual.green_prec = 0;

      system_visual->visual.blue_mask = 0;
      system_visual->visual.blue_shift = 0;
      system_visual->visual.blue_prec = 0;
   }
   system_visual->visual.colormap_size =
       system_visual->xvisual->map_entries;

   available_depths[0] = system_visual->visual.depth;
   available_types[0] = system_visual->visual.type;
}

GdkVisual *gdk_visual_ref(GdkVisual * visual)
{
   return visual;
}

void gdk_visual_unref(GdkVisual * visual)
{
   return;
}

gint gdk_visual_get_best_depth(void)
{
   return available_depths[0];
}

GdkVisualType gdk_visual_get_best_type(void)
{
   return available_types[0];
}

GdkVisual *gdk_visual_get_system(void)
{
   return ((GdkVisual *) system_visual);
}

GdkVisual *gdk_visual_get_best(void)
{
   return ((GdkVisual *) system_visual);
}

GdkVisual *gdk_visual_get_best_with_depth(gint depth)
{
   if (depth == system_visual->visual.depth)
      return (GdkVisual *) system_visual;
   else
      return NULL;
}

GdkVisual *gdk_visual_get_best_with_type(GdkVisualType visual_type)
{
   if (visual_type == system_visual->visual.type)
      return (GdkVisual *) system_visual;
   else
      return NULL;
}

GdkVisual *gdk_visual_get_best_with_both(gint depth,
                                         GdkVisualType visual_type)
{
   if ((depth == system_visual->visual.depth) &&
       (visual_type == system_visual->visual.type))
      return (GdkVisual *) system_visual;
   else
      return NULL;
}

void gdk_query_depths(gint ** depths, gint * count)
{
   *count = 1;
   *depths = available_depths;
}

void gdk_query_visual_types(GdkVisualType ** visual_types, gint * count)
{
   *count = 1;
   *visual_types = available_types;
}

GList *gdk_list_visuals(void)
{
   return g_list_append(NULL, (gpointer) system_visual);
}

static void
gdk_visual_decompose_mask(gulong mask, gint * shift, gint * prec)
{
   *shift = 0;
   *prec = 0;

   while (!(mask & 0x1)) {
      (*shift)++;
      mask >>= 1;
   }

   while (mask & 0x1) {
      (*prec)++;
      mask >>= 1;
   }
}
