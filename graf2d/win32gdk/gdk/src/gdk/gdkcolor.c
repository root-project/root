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

#include <time.h>

#include "gdkcolor.h"
#include "gdkprivate.h"

GdkColormap *gdk_colormap_ref(GdkColormap * cmap)
{
   GdkColormapPrivate *private = (GdkColormapPrivate *) cmap;

   g_return_val_if_fail(cmap != NULL, NULL);

   private->ref_count += 1;
   return cmap;
}

void gdk_colormap_unref(GdkColormap * cmap)
{
   GdkColormapPrivate *private = (GdkColormapPrivate *) cmap;

   g_return_if_fail(cmap != NULL);
   g_return_if_fail(private->ref_count > 0);

   private->ref_count -= 1;
   if (private->ref_count == 0)
      _gdk_colormap_real_destroy(cmap);
}

GdkVisual *gdk_colormap_get_visual(GdkColormap * colormap)
{
   GdkColormapPrivate *private;

   g_return_val_if_fail(colormap != NULL, NULL);

   private = (GdkColormapPrivate *) colormap;

   return private->visual;
}

void
gdk_colors_store(GdkColormap * colormap, GdkColor * colors, gint ncolors)
{
   gint i;

   for (i = 0; i < ncolors; i++) {
      colormap->colors[i].pixel = colors[i].pixel;
      colormap->colors[i].red = colors[i].red;
      colormap->colors[i].green = colors[i].green;
      colormap->colors[i].blue = colors[i].blue;
   }

   gdk_colormap_change(colormap, ncolors);
}

/*
 *--------------------------------------------------------------
 * gdk_color_copy
 *
 *   Copy a color structure into new storage.
 *
 * Arguments:
 *   "color" is the color struct to copy.
 *
 * Results:
 *   A new color structure.  Free it with gdk_color_free.
 *
 *--------------------------------------------------------------
 */

static GMemChunk *color_chunk;

GdkColor *gdk_color_copy(const GdkColor * color)
{
   GdkColor *new_color;

   g_return_val_if_fail(color != NULL, NULL);

   if (color_chunk == NULL)
      color_chunk = g_mem_chunk_new("colors",
                                    sizeof(GdkColor),
                                    4096, G_ALLOC_AND_FREE);

   new_color = g_chunk_new(GdkColor, color_chunk);
   *new_color = *color;
   return new_color;
}

/*
 *--------------------------------------------------------------
 * gdk_color_free
 *
 *   Free a color structure obtained from gdk_color_copy.  Do not use
 *   with other color structures.
 *
 * Arguments:
 *   "color" is the color struct to free.
 *
 *-------------------------------------------------------------- */

void gdk_color_free(GdkColor * color)
{
   g_assert(color_chunk != NULL);
   g_return_if_fail(color != NULL);

   g_mem_chunk_free(color_chunk, color);
}

gboolean gdk_color_white(GdkColormap * colormap, GdkColor * color)
{
   gint return_val;

   g_return_val_if_fail(colormap != NULL, FALSE);

   if (color) {
      color->red = 65535;
      color->green = 65535;
      color->blue = 65535;

      return_val = gdk_color_alloc(colormap, color);
   } else
      return_val = FALSE;

   return return_val;
}

gboolean gdk_color_black(GdkColormap * colormap, GdkColor * color)
{
   gint return_val;

   g_return_val_if_fail(colormap != NULL, FALSE);

   if (color) {
      color->red = 0;
      color->green = 0;
      color->blue = 0;

      return_val = gdk_color_alloc(colormap, color);
   } else
      return_val = FALSE;

   return return_val;
}

/********************
 * Color allocation *
 ********************/

gboolean
gdk_colormap_alloc_color(GdkColormap * colormap,
                         GdkColor * color,
                         gboolean writeable, gboolean best_match)
{
   gboolean success;

   gdk_colormap_alloc_colors(colormap, color, 1, writeable, best_match,
                             &success);

   return success;
}

gboolean gdk_color_alloc(GdkColormap * colormap, GdkColor * color)
{
   gboolean success;

   gdk_colormap_alloc_colors(colormap, color, 1, FALSE, TRUE, &success);

   return success;
}

guint gdk_color_hash(const GdkColor * colora)
{
   return ((colora->red) +
           (colora->green << 11) +
           (colora->blue << 22) + (colora->blue >> 6));
}

gboolean gdk_color_equal(const GdkColor * colora, const GdkColor * colorb)
{
   g_return_val_if_fail(colora != NULL, FALSE);
   g_return_val_if_fail(colorb != NULL, FALSE);

   return ((colora->red == colorb->red) &&
           (colora->green == colorb->green) &&
           (colora->blue == colorb->blue));
}
