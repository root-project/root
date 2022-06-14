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

/* Color Context module
 * Copyright 1994,1995 John L. Cwikla
 * Copyright (C) 1997 by Ripley Software Development
 * Copyright (C) 1997 by Federico Mena (port to Gtk/Gdk)
 */

/* Copyright 1994,1995 John L. Cwikla
 *
 * Permission to use, copy, modify, distribute, and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appears in all copies and that
 * both that copyright notice and this permission notice appear in
 * supporting documentation, and that the name of John L. Cwikla or
 * Wolfram Research, Inc not be used in advertising or publicity
 * pertaining to distribution of the software without specific, written
 * prior permission.  John L. Cwikla and Wolfram Research, Inc make no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 *
 * John L. Cwikla and Wolfram Research, Inc disclaim all warranties with
 * regard to this software, including all implied warranties of
 * merchantability and fitness, in no event shall John L. Cwikla or
 * Wolfram Research, Inc be liable for any special, indirect or
 * consequential damages or any damages whatsoever resulting from loss of
 * use, data or profits, whether in an action of contract, negligence or
 * other tortious action, arising out of or in connection with the use or
 * performance of this software.
 *
 * Author:
 *  John L. Cwikla
 *  X Programmer
 *  Wolfram Research Inc.
 *
 *  cwikla@wri.com
 */

/*
 * Modified by the GTK+ Team and others 1997-1999.  See the AUTHORS
 * file for a list of people on the GTK+ Team.  See the ChangeLog
 * files for a list of changes.  These files are distributed with
 * GTK+ at ftp://ftp.gtk.org/pub/gtk/. 
 */

#include "config.h"

#include <stdlib.h>
#include <string.h>

#include "gdkcc.h"
#include "gdkcolor.h"
#include "gdkwin32.h"

#define MAX_IMAGE_COLORS 256

typedef struct _GdkColorContextPrivate GdkColorContextPrivate;

struct _GdkColorContextPrivate {
   GdkColorContext color_context;
   XStandardColormap std_cmap;
};

static guint hash_color(gconstpointer key)
{
   const GdkColor *color = key;

   return (color->red * 33023 + color->green * 30013 +
           color->blue * 27011);
}

static gint compare_colors(gconstpointer a, gconstpointer b)
{
   const GdkColor *aa = a;
   const GdkColor *bb = b;

   return ((aa->red == bb->red) && (aa->green == bb->green)
           && (aa->blue == bb->blue));
}

static void
free_hash_entry(gpointer key, gpointer value, gpointer user_data)
{
   g_free(key);                 /* key and value are the same GdkColor */
}

static int pixel_sort(const void *a, const void *b)
{
   return ((GdkColor *) a)->pixel - ((GdkColor *) b)->pixel;
}

static void
my_x_query_colors(GdkColormap * colormap, GdkColor * colors, gint ncolors)
{
   gint i;

   for (i = 0; i < ncolors; i++) {
      PALETTEENTRY palentry;

      GetPaletteEntries(GDK_COLORMAP_WIN32COLORMAP(colormap)->palette,
                        colors[i].pixel, 1, &palentry);
      colors[i].red = (palentry.peRed * 65535) / 255;
      colors[i].green = (palentry.peGreen * 65535) / 255;
      colors[i].blue = (palentry.peBlue * 65535) / 255;
   }
}

static void query_colors(GdkColorContext * cc)
{
   gint i;
   GdkColorContextPrivate *ccp = (GdkColorContextPrivate *) cc;
   cc->cmap = g_new(GdkColor, cc->num_colors);

   for (i = 0; i < cc->num_colors; i++)
      cc->cmap[i].pixel =
          cc->clut ? cc->clut[i] : ccp->std_cmap.base_pixel + i;

   my_x_query_colors(cc->colormap, cc->cmap, cc->num_colors);

   qsort(cc->cmap, cc->num_colors, sizeof(GdkColor), pixel_sort);
}

static void init_bw(GdkColorContext * cc)
{
   GdkColor color;

   g_warning
       ("init_bw: failed to allocate colors, falling back to black and white");

   cc->mode = GDK_CC_MODE_BW;

   color.red = color.green = color.blue = 0;

   if (!gdk_color_alloc(cc->colormap, &color))
      cc->black_pixel = 0;
   else
      cc->black_pixel = color.pixel;

   color.red = color.green = color.blue = 0xffff;

   if (!gdk_color_alloc(cc->colormap, &color))
      cc->white_pixel = cc->black_pixel ? 0 : 1;
   else
      cc->white_pixel = color.pixel;

   cc->num_colors = 2;
}

static void init_gray(GdkColorContext * cc)
{
   GdkColorContextPrivate *ccp = (GdkColorContextPrivate *) cc;
   GdkColor *clrs, *cstart;
   gint i;
   gdouble dinc;

   cc->num_colors = 256;        /* Bogus, but will never get here anyway? */

   cc->clut = g_new(unsigned long, cc->num_colors);
   cstart = g_new(GdkColor, cc->num_colors);

 retrygray:

   dinc = 65535.0 / (cc->num_colors - 1);

   clrs = cstart;

   for (i = 0; i < cc->num_colors; i++) {
      clrs->red = clrs->green = clrs->blue = dinc * i;

      if (!gdk_color_alloc(cc->colormap, clrs)) {
         gdk_colors_free(cc->colormap, cc->clut, i, 0);

         cc->num_colors /= 2;

         if (cc->num_colors > 1)
            goto retrygray;
         else {
            g_free(cc->clut);
            cc->clut = NULL;
            init_bw(cc);
            g_free(cstart);
            return;
         }
      }

      cc->clut[i] = clrs++->pixel;
   }

   g_free(cstart);

   /* XXX: is this the right thing to do? */
   ccp->std_cmap.colormap = GDK_COLORMAP_WIN32COLORMAP(cc->colormap);
   ccp->std_cmap.base_pixel = 0;
   ccp->std_cmap.red_max = cc->num_colors - 1;
   ccp->std_cmap.green_max = 0;
   ccp->std_cmap.blue_max = 0;
   ccp->std_cmap.red_mult = 1;
   ccp->std_cmap.green_mult = 0;
   ccp->std_cmap.blue_mult = 0;

   cc->white_pixel = 255;
   cc->black_pixel = 0;

   query_colors(cc);

   cc->mode = GDK_CC_MODE_MY_GRAY;
}

static void init_color(GdkColorContext * cc)
{
   GdkColorContextPrivate *ccp = (GdkColorContextPrivate *) cc;
   gint cubeval;

   cubeval = 1;
   while ((cubeval * cubeval * cubeval) <
          GDK_VISUAL_XVISUAL(cc->visual)->map_entries)
      cubeval++;
   cubeval--;

   cc->num_colors = cubeval * cubeval * cubeval;

   ccp->std_cmap.red_max = cubeval - 1;
   ccp->std_cmap.green_max = cubeval - 1;
   ccp->std_cmap.blue_max = cubeval - 1;
   ccp->std_cmap.red_mult = cubeval * cubeval;
   ccp->std_cmap.green_mult = cubeval;
   ccp->std_cmap.blue_mult = 1;
   ccp->std_cmap.base_pixel = 0;

   cc->white_pixel = 255;       /* ??? */
   cc->black_pixel = 0;         /* ??? */

   /* a CLUT for storing allocated pixel indices */

   cc->max_colors = cc->num_colors;
   cc->clut = g_new(unsigned long, cc->max_colors);

   for (cubeval = 0; cubeval < cc->max_colors; cubeval++)
      cc->clut[cubeval] = cubeval;

   query_colors(cc);

   cc->mode = GDK_CC_MODE_STD_CMAP;
}


static void init_true_color(GdkColorContext * cc)
{
   GdkColorContextPrivate *ccp = (GdkColorContextPrivate *) cc;
   unsigned long rmask, gmask, bmask;

   cc->mode = GDK_CC_MODE_TRUE;

   /* Red */

   rmask = cc->masks.red = cc->visual->red_mask;

   cc->shifts.red = 0;
   cc->bits.red = 0;

   while (!(rmask & 1)) {
      rmask >>= 1;
      cc->shifts.red++;
   }

   while (rmask & 1) {
      rmask >>= 1;
      cc->bits.red++;
   }

   /* Green */

   gmask = cc->masks.green = cc->visual->green_mask;

   cc->shifts.green = 0;
   cc->bits.green = 0;

   while (!(gmask & 1)) {
      gmask >>= 1;
      cc->shifts.green++;
   }

   while (gmask & 1) {
      gmask >>= 1;
      cc->bits.green++;
   }

   /* Blue */

   bmask = cc->masks.blue = cc->visual->blue_mask;

   cc->shifts.blue = 0;
   cc->bits.blue = 0;

   while (!(bmask & 1)) {
      bmask >>= 1;
      cc->shifts.blue++;
   }

   while (bmask & 1) {
      bmask >>= 1;
      cc->bits.blue++;
   }

   cc->num_colors =
       (cc->visual->red_mask | cc->visual->green_mask | cc->visual->
        blue_mask) + 1;

   cc->white_pixel = 0xffffff;
   cc->black_pixel = 0;
}

static void init_palette(GdkColorContext * cc)
{
   /* restore correct mode for this cc */

   switch (cc->visual->type) {
   case GDK_VISUAL_STATIC_GRAY:
   case GDK_VISUAL_GRAYSCALE:
      if (GDK_VISUAL_XVISUAL(cc->visual)->map_entries == 2)
         cc->mode = GDK_CC_MODE_BW;
      else
         cc->mode = GDK_CC_MODE_MY_GRAY;
      break;

   case GDK_VISUAL_TRUE_COLOR:
      cc->mode = GDK_CC_MODE_TRUE;
      break;

   case GDK_VISUAL_STATIC_COLOR:
   case GDK_VISUAL_PSEUDO_COLOR:
      cc->mode = GDK_CC_MODE_STD_CMAP;
      break;

   default:
      cc->mode = GDK_CC_MODE_UNDEFINED;
      break;
   }

   /* previous palette */

   if (cc->num_palette)
      g_free(cc->palette);

   if (cc->fast_dither)
      g_free(cc->fast_dither);

   /* clear hash table if present */

   if (cc->color_hash) {
      g_hash_table_foreach(cc->color_hash, free_hash_entry, NULL);
      g_hash_table_destroy(cc->color_hash);
      cc->color_hash = g_hash_table_new(hash_color, compare_colors);
   }

   cc->palette = NULL;
   cc->num_palette = 0;
   cc->fast_dither = NULL;
}

GdkColorContext *gdk_color_context_new(GdkVisual * visual,
                                       GdkColormap * colormap)
{
   GdkColorContextPrivate *ccp;
   gint use_private_colormap = FALSE;	/* XXX: maybe restore full functionality later? */
   GdkColorContext *cc;
   gint retry_count;
   GdkColormap *default_colormap;

   g_assert(visual != NULL);
   g_assert(colormap != NULL);

   ccp = g_new(GdkColorContextPrivate, 1);
   cc = (GdkColorContext *) ccp;
   cc->visual = visual;
   cc->colormap = colormap;
   cc->clut = NULL;
   cc->cmap = NULL;
   cc->mode = GDK_CC_MODE_UNDEFINED;
   cc->need_to_free_colormap = FALSE;

   cc->color_hash = NULL;
   cc->palette = NULL;
   cc->num_palette = 0;
   cc->fast_dither = NULL;

   default_colormap = gdk_colormap_get_system();

   retry_count = 0;

   while (retry_count < 2) {
      /* Only create a private colormap if the visual found isn't equal
       * to the default visual and we don't have a private colormap,
       * -or- if we are instructed to create a private colormap (which
       * never is the case for XmHTML).
       */

      if (use_private_colormap || ((cc->visual != gdk_visual_get_system())	/* default visual? */
                                   &&(GDK_COLORMAP_WIN32COLORMAP(colormap)
                                      ==
                                      GDK_COLORMAP_WIN32COLORMAP
                                      (default_colormap)))) {
         g_warning("gdk_color_context_new: non-default visual detected, "
                   "using private colormap");

         cc->colormap = gdk_colormap_new(cc->visual, FALSE);

         cc->need_to_free_colormap = (GDK_COLORMAP_WIN32COLORMAP(colormap)
                                      !=
                                      GDK_COLORMAP_WIN32COLORMAP
                                      (default_colormap));
      }

      switch (visual->type) {
      case GDK_VISUAL_STATIC_GRAY:
      case GDK_VISUAL_GRAYSCALE:
         GDK_NOTE(COLOR_CONTEXT,
                  g_message("gdk_color_context_new: visual class is %s\n",
                            (visual->type == GDK_VISUAL_STATIC_GRAY) ?
                            "GDK_VISUAL_STATIC_GRAY" :
                            "GDK_VISUAL_GRAYSCALE"));

         if (GDK_VISUAL_XVISUAL(cc->visual)->map_entries == 2)
            init_bw(cc);
         else
            init_gray(cc);

         break;

      case GDK_VISUAL_TRUE_COLOR:	/* shifts */
         GDK_NOTE(COLOR_CONTEXT,
                  g_message
                  ("gdk_color_context_new: visual class is GDK_VISUAL_TRUE_COLOR\n"));

         init_true_color(cc);
         break;

      case GDK_VISUAL_STATIC_COLOR:
      case GDK_VISUAL_PSEUDO_COLOR:
         GDK_NOTE(COLOR_CONTEXT,
                  g_message("gdk_color_context_new: visual class is %s\n",
                            (visual->type == GDK_VISUAL_STATIC_COLOR) ?
                            "GDK_VISUAL_STATIC_COLOR" :
                            "GDK_VISUAL_PSEUDO_COLOR"));

         init_color(cc);
         break;

      default:
         g_assert_not_reached();
      }

      if ((cc->mode == GDK_CC_MODE_BW) && (cc->visual->depth > 1)) {
         use_private_colormap = TRUE;
         retry_count++;
      } else
         break;
   }

   /* no. of colors allocated yet */

   cc->num_allocated = 0;

   GDK_NOTE(COLOR_CONTEXT,
            g_message
            ("gdk_color_context_new: screen depth is %i, no. of colors is %i\n",
             cc->visual->depth, cc->num_colors));

   return (GdkColorContext *) cc;
}

GdkColorContext *gdk_color_context_new_mono(GdkVisual * visual,
                                            GdkColormap * colormap)
{
   GdkColorContextPrivate *ccp;
   GdkColorContext *cc;

   g_assert(visual != NULL);
   g_assert(colormap != NULL);

   cc = g_new(GdkColorContext, 1);
   ccp = (GdkColorContextPrivate *) cc;
   cc->visual = visual;
   cc->colormap = colormap;
   cc->clut = NULL;
   cc->cmap = NULL;
   cc->mode = GDK_CC_MODE_UNDEFINED;
   cc->need_to_free_colormap = FALSE;

   init_bw(cc);

   return (GdkColorContext *) cc;
}

/* This doesn't currently free black/white, hmm... */

void gdk_color_context_free(GdkColorContext * cc)
{
   g_assert(cc != NULL);

   if ((cc->visual->type == GDK_VISUAL_STATIC_COLOR)
       || (cc->visual->type == GDK_VISUAL_PSEUDO_COLOR)) {
      gdk_colors_free(cc->colormap, cc->clut, cc->num_allocated, 0);
      g_free(cc->clut);
   } else if (cc->clut != NULL) {
      gdk_colors_free(cc->colormap, cc->clut, cc->num_colors, 0);
      g_free(cc->clut);
   }

   if (cc->cmap != NULL)
      g_free(cc->cmap);

   if (cc->need_to_free_colormap)
      gdk_colormap_unref(cc->colormap);

   /* free any palette that has been associated with this GdkColorContext */

   init_palette(cc);

   g_free(cc);
}

unsigned long
gdk_color_context_get_pixel(GdkColorContext * cc,
                            gushort red,
                            gushort green, gushort blue, gint * failed)
{
   GdkColorContextPrivate *ccp = (GdkColorContextPrivate *) cc;
   g_assert(cc != NULL);
   g_assert(failed != NULL);

   *failed = FALSE;

   switch (cc->mode) {
   case GDK_CC_MODE_BW:
      {
         gdouble value;

         value = (red / 65535.0 * 0.30
                  + green / 65535.0 * 0.59 + blue / 65535.0 * 0.11);

         if (value > 0.5)
            return cc->white_pixel;

         return cc->black_pixel;
      }

   case GDK_CC_MODE_MY_GRAY:
      {
         unsigned long ired, igreen, iblue;

         red = red * 0.30 + green * 0.59 + blue * 0.11;
         green = 0;
         blue = 0;

         if ((ired =
              red * (ccp->std_cmap.red_max + 1) / 0xffff) >
             ccp->std_cmap.red_max)
            ired = ccp->std_cmap.red_max;

         ired *= ccp->std_cmap.red_mult;

         if ((igreen =
              green * (ccp->std_cmap.green_max + 1) / 0xffff) >
             ccp->std_cmap.green_max)
            igreen = ccp->std_cmap.green_max;

         igreen *= ccp->std_cmap.green_mult;

         if ((iblue =
              blue * (ccp->std_cmap.blue_max + 1) / 0xffff) >
             ccp->std_cmap.blue_max)
            iblue = ccp->std_cmap.blue_max;

         iblue *= ccp->std_cmap.blue_mult;

         if (cc->clut != NULL)
            return cc->clut[ccp->std_cmap.base_pixel + ired + igreen +
                            iblue];

         return ccp->std_cmap.base_pixel + ired + igreen + iblue;
      }

   case GDK_CC_MODE_TRUE:
      {
         unsigned long ired, igreen, iblue;

         if (cc->clut == NULL) {
            red >>= 16 - cc->bits.red;
            green >>= 16 - cc->bits.green;
            blue >>= 16 - cc->bits.blue;

            ired = (red << cc->shifts.red) & cc->masks.red;
            igreen = (green << cc->shifts.green) & cc->masks.green;
            iblue = (blue << cc->shifts.blue) & cc->masks.blue;

            return ired | igreen | iblue;
         }

         ired = cc->clut[red * cc->max_entry / 65535] & cc->masks.red;
         igreen =
             cc->clut[green * cc->max_entry / 65535] & cc->masks.green;
         iblue = cc->clut[blue * cc->max_entry / 65535] & cc->masks.blue;

         return ired | igreen | iblue;
      }

   case GDK_CC_MODE_PALETTE:
      return gdk_color_context_get_pixel_from_palette(cc, &red, &green,
                                                      &blue, failed);

   case GDK_CC_MODE_STD_CMAP:
   default:
      {
         GdkColor color;
         GdkColor *result = NULL;

         color.red = red;
         color.green = green;
         color.blue = blue;

         if (cc->color_hash)
            result = g_hash_table_lookup(cc->color_hash, &color);

         if (!result) {
            color.red = red;
            color.green = green;
            color.blue = blue;
            color.pixel = 0;

            if (!gdk_color_alloc(cc->colormap, &color))
               *failed = TRUE;
            else {
               GdkColor *cnew;

               /* XXX: the following comment comes directly from
                * XCC.c.  I don't know if it is relevant for
                * gdk_color_alloc() as it is for XAllocColor()
                * - Federico
                */
               /*
                * I can't figure this out entirely, but it *is* possible
                * that XAllocColor succeeds, even if the number of
                * allocations we've made exceeds the number of available
                * colors in the current colormap. And therefore it
                * might be necessary for us to resize the CLUT.
                */

               if (cc->num_allocated == cc->max_colors) {
                  cc->max_colors *= 2;

                  GDK_NOTE(COLOR_CONTEXT,
                           g_message("gdk_color_context_get_pixel: "
                                     "resizing CLUT to %i entries\n",
                                     cc->max_colors));

                  cc->clut = g_realloc(cc->clut,
                                       cc->max_colors * sizeof(unsigned long));
               }

               /* Key and value are the same color structure */

               cnew = g_new(GdkColor, 1);
               *cnew = color;

               if (!cc->color_hash)
                  cc->color_hash =
                      g_hash_table_new(hash_color, compare_colors);
               g_hash_table_insert(cc->color_hash, cnew, cnew);

               cc->clut[cc->num_allocated] = color.pixel;
               cc->num_allocated++;
               return color.pixel;
            }
         }

         return result->pixel;
      }
   }
}

void
gdk_color_context_get_pixels(GdkColorContext * cc,
                             gushort * reds,
                             gushort * greens,
                             gushort * blues,
                             gint ncolors,
                             unsigned long * colors, gint * nallocated)
{
   gint i, k, idx;
   gint cmapsize, ncols = 0, nopen = 0, counter = 0;
   gint bad_alloc = FALSE;
   gint failed[MAX_IMAGE_COLORS], allocated[MAX_IMAGE_COLORS];
   GdkColor defs[MAX_IMAGE_COLORS], cmap[MAX_IMAGE_COLORS];
#ifdef G_ENABLE_DEBUG
   gint exact_col = 0, subst_col = 0, close_col = 0, black_col = 0;
#endif
   g_assert(cc != NULL);
   g_assert(reds != NULL);
   g_assert(greens != NULL);
   g_assert(blues != NULL);
   g_assert(colors != NULL);
   g_assert(nallocated != NULL);

   memset(defs, 0, MAX_IMAGE_COLORS * sizeof(GdkColor));
   memset(failed, 0, MAX_IMAGE_COLORS * sizeof(gint));
   memset(allocated, 0, MAX_IMAGE_COLORS * sizeof(gint));

   /* Will only have a value if used by the progressive image loader */

   ncols = *nallocated;

   *nallocated = 0;

   /* First allocate all pixels */

   for (i = 0; i < ncolors; i++) {
      /* colors[i] is only zero if the pixel at that location hasn't
       * been allocated yet.  This is a sanity check required for proper
       * color allocation by the progressive image loader
       */

      if (colors[i] == 0) {
         defs[i].red = reds[i];
         defs[i].green = greens[i];
         defs[i].blue = blues[i];

         colors[i] =
             gdk_color_context_get_pixel(cc, reds[i], greens[i], blues[i],
                                         &bad_alloc);

         /* successfully allocated, store it */

         if (!bad_alloc) {
            defs[i].pixel = colors[i];
            allocated[ncols++] = colors[i];
         } else
            failed[nopen++] = i;
      }
   }

   *nallocated = ncols;

   /* all colors available, all done */

   if ((ncols == ncolors) || (nopen == 0)) {
      GDK_NOTE(COLOR_CONTEXT,
               g_message
               ("gdk_color_context_get_pixels: got all %i colors; "
                "(%i colors allocated so far)\n", ncolors,
                cc->num_allocated));

      return;
   }

   /* The fun part.  We now try to allocate the colors we couldn't allocate
    * directly.  The first step will map a color onto its nearest color
    * that has been allocated (either by us or someone else).  If any colors
    * remain unallocated, we map these onto the colors that we have allocated
    * ourselves.
    */

   /* read up to MAX_IMAGE_COLORS colors of the current colormap */

   cmapsize = MIN(cc->num_colors, MAX_IMAGE_COLORS);

   /* see if the colormap has any colors to read */

   if (cmapsize < 0) {
      g_warning
          ("gdk_color_context_get_pixels: oops!  no colors available, "
           "your images will look *really* ugly.");

      return;
   }
#ifdef G_ENABLE_DEBUG
   exact_col = ncols;
#endif

   /* initialize pixels */

   for (i = 0; i < cmapsize; i++) {
      cmap[i].pixel = i;
      cmap[i].red = cmap[i].green = cmap[i].blue = 0;
   }

   /* read the colormap */

   my_x_query_colors(cc->colormap, cmap, cmapsize);

   /* get a close match for any unallocated colors */

   counter = nopen;
   nopen = 0;
   idx = 0;

   do {
      gint d, j, mdist, close, ri, gi, bi;
      gint rd, gd, bd;

      i = failed[idx];

      mdist = 0x1000000;
      close = -1;

      /* Store these vals.  Small performance increase as this skips three
       * indexing operations in the loop code.
       */

      ri = reds[i];
      gi = greens[i];
      bi = blues[i];

      /* Walk all colors in the colormap and see which one is the
       * closest.  Uses plain least squares.
       */

      for (j = 0; (j < cmapsize) && (mdist != 0); j++) {
         /* Don't replace these by shifts; the sign may get clobbered */

         rd = (ri - cmap[j].red) / 256;
         gd = (gi - cmap[j].green) / 256;
         bd = (bi - cmap[j].blue) / 256;

         d = rd * rd + gd * gd + bd * bd;

         if (d < mdist) {
            close = j;
            mdist = d;
         }
      }

      if (close != -1) {
         rd = cmap[close].red;
         gd = cmap[close].green;
         bd = cmap[close].blue;

         /* allocate */

         colors[i] =
             gdk_color_context_get_pixel(cc, rd, gd, bd, &bad_alloc);

         /* store */

         if (!bad_alloc) {
            defs[i] = cmap[close];
            defs[i].pixel = colors[i];
            allocated[ncols++] = colors[i];
#ifdef G_ENABLE_DEBUG
            close_col++;
#endif
         } else
            failed[nopen++] = i;
      } else
         failed[nopen++] = i;
      /* deal with in next stage if allocation failed */
   }
   while (++idx < counter);

   *nallocated = ncols;

   /* This is the maximum no. of allocated colors.  See also the nopen == 0
    * note above.
    */

   if ((ncols == ncolors) || (nopen == 0)) {
      GDK_NOTE(COLOR_CONTEXT,
               g_message
               ("gdk_color_context_get_pixels: got %i colors, %i exact and "
                "%i close (%i colors allocated so far)\n", ncolors,
                exact_col, close_col, cc->num_allocated));

      return;
   }

   /* Now map any remaining unallocated pixels into the colors we did get */

   idx = 0;

   do {
      gint d, mdist, close, ri, gi, bi;
      gint j, rd, gd, bd;

      i = failed[idx];

      mdist = 0x1000000;
      close = -1;

      /* store */

      ri = reds[i];
      gi = greens[i];
      bi = blues[i];

      /* search allocated colors */

      for (j = 0; (j < ncols) && (mdist != 0); j++) {
         k = allocated[j];

         /* Don't replace these by shifts; the sign may get clobbered */

         rd = (ri - defs[k].red) / 256;
         gd = (gi - defs[k].green) / 256;
         bd = (bi - defs[k].blue) / 256;

         d = rd * rd + gd * gd + bd * bd;

         if (d < mdist) {
            close = k;
            mdist = d;
         }
      }

      if (close < 0) {
         /* too bad, map to black */

         defs[i].pixel = cc->black_pixel;
         defs[i].red = defs[i].green = defs[i].blue = 0;
#ifdef G_ENABLE_DEBUG
         black_col++;
#endif
      } else {
         defs[i] = defs[close];
#ifdef G_ENABLE_DEBUG
         subst_col++;
#endif
      }

      colors[i] = defs[i].pixel;
   }
   while (++idx < nopen);

   GDK_NOTE(COLOR_CONTEXT,
            g_message
            ("gdk_color_context_get_pixels: got %i colors, %i exact, %i close, "
             "%i substituted, %i to black (%i colors allocated so far)\n",
             ncolors, exact_col, close_col, subst_col, black_col,
             cc->num_allocated));
}

void
gdk_color_context_get_pixels_incremental(GdkColorContext * cc,
                                         gushort * reds,
                                         gushort * greens,
                                         gushort * blues,
                                         gint ncolors,
                                         gint * used,
                                         unsigned long * colors,
                                         gint * nallocated)
{
   gint i, k, idx;
   gint cmapsize, ncols = 0, nopen = 0, counter = 0;
   gint bad_alloc = FALSE;
   gint failed[MAX_IMAGE_COLORS], allocated[MAX_IMAGE_COLORS];
   GdkColor defs[MAX_IMAGE_COLORS], cmap[MAX_IMAGE_COLORS];
#ifdef G_ENABLE_DEBUG
   gint exact_col = 0, subst_col = 0, close_col = 0, black_col = 0;
#endif

   g_assert(cc != NULL);
   g_assert(reds != NULL);
   g_assert(greens != NULL);
   g_assert(blues != NULL);
   g_assert(used != NULL);
   g_assert(colors != NULL);
   g_assert(nallocated != NULL);

   memset(defs, 0, MAX_IMAGE_COLORS * sizeof(GdkColor));
   memset(failed, 0, MAX_IMAGE_COLORS * sizeof(gint));
   memset(allocated, 0, MAX_IMAGE_COLORS * sizeof(gint));

   /* Will only have a value if used by the progressive image loader */

   ncols = *nallocated;

   *nallocated = 0;

   /* First allocate all pixels */

   for (i = 0; i < ncolors; i++) {
      /* used[i] is only -1 if the pixel at that location hasn't
       * been allocated yet.  This is a sanity check required for proper
       * color allocation by the progressive image loader.
       * When colors[i] == 0 it indicates the slot is available for
       * allocation.
       */

      if (used[i] != FALSE) {
         if (colors[i] == 0) {
            defs[i].red = reds[i];
            defs[i].green = greens[i];
            defs[i].blue = blues[i];

            colors[i] =
                gdk_color_context_get_pixel(cc, reds[i], greens[i],
                                            blues[i], &bad_alloc);

            /* successfully allocated, store it */

            if (!bad_alloc) {
               defs[i].pixel = colors[i];
               allocated[ncols++] = colors[i];
            } else
               failed[nopen++] = i;
         }
#ifdef DEBUG
         else
            GDK_NOTE(COLOR_CONTEXT,
                     g_message("gdk_color_context_get_pixels_incremental: "
                               "pixel at slot %i already allocated, skipping\n",
                               i));
#endif
      }
   }

   *nallocated = ncols;

   if ((ncols == ncolors) || (nopen == 0)) {
      GDK_NOTE(COLOR_CONTEXT,
               g_message
               ("gdk_color_context_get_pixels_incremental: got all %i colors "
                "(%i colors allocated so far)\n", ncolors,
                cc->num_allocated));

      return;
   }

   cmapsize = MIN(cc->num_colors, MAX_IMAGE_COLORS);

   if (cmapsize < 0) {
      g_warning("gdk_color_context_get_pixels_incremental: oops!  "
                "No colors available images will look *really* ugly.");
      return;
   }
#ifdef G_ENABLE_DEBUG
   exact_col = ncols;
#endif

   /* initialize pixels */

   for (i = 0; i < cmapsize; i++) {
      cmap[i].pixel = i;
      cmap[i].red = cmap[i].green = cmap[i].blue = 0;
   }

   /* read */

   my_x_query_colors(cc->colormap, cmap, cmapsize);

   /* now match any unallocated colors */

   counter = nopen;
   nopen = 0;
   idx = 0;

   do {
      gint d, j, mdist, close, ri, gi, bi;
      gint rd, gd, bd;

      i = failed[idx];

      mdist = 0x1000000;
      close = -1;

      /* store */

      ri = reds[i];
      gi = greens[i];
      bi = blues[i];

      for (j = 0; (j < cmapsize) && (mdist != 0); j++) {
         /* Don't replace these by shifts; the sign may get clobbered */

         rd = (ri - cmap[j].red) / 256;
         gd = (gi - cmap[j].green) / 256;
         bd = (bi - cmap[j].blue) / 256;

         d = rd * rd + gd * gd + bd * bd;

         if (d < mdist) {
            close = j;
            mdist = d;
         }
      }

      if (close != -1) {
         rd = cmap[close].red;
         gd = cmap[close].green;
         bd = cmap[close].blue;

         /* allocate */

         colors[i] =
             gdk_color_context_get_pixel(cc, rd, gd, bd, &bad_alloc);

         /* store */

         if (!bad_alloc) {
            defs[i] = cmap[close];
            defs[i].pixel = colors[i];
            allocated[ncols++] = colors[i];
#ifdef G_ENABLE_DEBUG
            close_col++;
#endif
         } else
            failed[nopen++] = i;
      } else
         failed[nopen++] = i;
      /* deal with in next stage if allocation failed */
   }
   while (++idx < counter);

   *nallocated = ncols;

   if ((ncols == ncolors) || (nopen == 0)) {
      GDK_NOTE(COLOR_CONTEXT,
               g_message("gdk_color_context_get_pixels_incremental: "
                         "got %i colors, %i exact and %i close "
                         "(%i colors allocated so far)\n",
                         ncolors, exact_col, close_col,
                         cc->num_allocated));

      return;
   }

   /* map remaining unallocated pixels into colors we did get */

   idx = 0;

   do {
      gint d, mdist, close, ri, gi, bi;
      gint j, rd, gd, bd;

      i = failed[idx];

      mdist = 0x1000000;
      close = -1;

      ri = reds[i];
      gi = greens[i];
      bi = blues[i];

      /* search allocated colors */

      for (j = 0; (j < ncols) && (mdist != 0); j++) {
         k = allocated[j];

         /* downscale */
         /* Don't replace these by shifts; the sign may get clobbered */

         rd = (ri - defs[k].red) / 256;
         gd = (gi - defs[k].green) / 256;
         bd = (bi - defs[k].blue) / 256;

         d = rd * rd + gd * gd + bd * bd;

         if (d < mdist) {
            close = k;
            mdist = d;
         }
      }

      if (close < 0) {
         /* too bad, map to black */

         defs[i].pixel = cc->black_pixel;
         defs[i].red = defs[i].green = defs[i].blue = 0;
#ifdef G_ENABLE_DEBUG
         black_col++;
#endif
      } else {
         defs[i] = defs[close];
#ifdef G_ENABLE_DEBUG
         subst_col++;
#endif
      }

      colors[i] = defs[i].pixel;
   }
   while (++idx < nopen);

   GDK_NOTE(COLOR_CONTEXT,
            g_message("gdk_color_context_get_pixels_incremental: "
                      "got %i colors, %i exact, %i close, %i substituted, %i to black "
                      "(%i colors allocated so far)\n",
                      ncolors, exact_col, close_col, subst_col, black_col,
                      cc->num_allocated));
}

gint gdk_color_context_query_color(GdkColorContext * cc, GdkColor * color)
{
   return gdk_color_context_query_colors(cc, color, 1);
}

gint
gdk_color_context_query_colors(GdkColorContext * cc,
                               GdkColor * colors, gint num_colors)
{
   gint i;
   GdkColor *tc;

   g_assert(cc != NULL);
   g_assert(colors != NULL);

   switch (cc->mode) {
   case GDK_CC_MODE_BW:
      for (i = 0, tc = colors; i < num_colors; i++, tc++) {
         if (tc->pixel == cc->white_pixel)
            tc->red = tc->green = tc->blue = 65535;
         else
            tc->red = tc->green = tc->blue = 0;
      }
      break;

   case GDK_CC_MODE_TRUE:
      if (cc->clut == NULL)
         for (i = 0, tc = colors; i < num_colors; i++, tc++) {
#if 0 // bb change
            tc->red =
                ((tc->pixel & cc->masks.red) >> cc->shifts.red) << (16 -
                                                                    cc->
                                                                    bits.
                                                                    red);
            tc->green =
                ((tc->pixel & cc->masks.green) >> cc->shifts.
                 green) << (16 - cc->bits.green);
            tc->blue =
                ((tc->pixel & cc->masks.blue) >> cc->shifts.blue) << (16 -
                                                                      cc->
                                                                      bits.
                                                                      blue);
#else
            tc->red = 65535. * (double)((tc->pixel & cc->masks.red) 
                >> cc->shifts.red) / ((1 << cc->visual->red_prec) - 1);
            tc->green = 65535. * (double)((tc->pixel & cc->masks.green) 
                >> cc->shifts.green) / ((1 << cc->visual->green_prec) - 1);
            tc->blue = 65535. * (double)((tc->pixel & cc->masks.blue) 
                >> cc->shifts.blue) / ((1 << cc->visual->blue_prec) - 1);
#endif
      } else {
         my_x_query_colors(cc->colormap, colors, num_colors);
         return 1;
      }
      break;

   case GDK_CC_MODE_STD_CMAP:
   default:
      if (cc->cmap == NULL) {
         my_x_query_colors(cc->colormap, colors, num_colors);
         return 1;
      } else {
         gint first, last, half;
         unsigned long half_pixel;

         for (i = 0, tc = colors; i < num_colors; i++) {
            first = 0;
            last = cc->num_colors - 1;

            while (first <= last) {
               half = (first + last) / 2;
               half_pixel = cc->cmap[half].pixel;

               if (tc->pixel == half_pixel) {
                  tc->red = cc->cmap[half].red;
                  tc->green = cc->cmap[half].green;
                  tc->blue = cc->cmap[half].blue;
                  first = last + 1;	/* false break */
               } else {
                  if (tc->pixel > half_pixel)
                     first = half + 1;
                  else
                     last = half - 1;
               }
            }
         }
         return 1;
      }
      break;
   }
   return 1;
}

gint
gdk_color_context_add_palette(GdkColorContext * cc,
                              GdkColor * palette, gint num_palette)
{
   gint i, j, erg;
   gushort r, g, b;
   unsigned long pixel[1];

   g_assert(cc != NULL);

   /* initialize this palette (will also erase previous palette as well) */

   init_palette(cc);

   /* restore previous mode if we aren't adding a new palette */

   if (num_palette == 0)
      return 0;

   /* copy incoming palette */

   cc->palette = g_new0(GdkColor, num_palette);

   j = 0;

   for (i = 0; i < num_palette; i++) {
      erg = 0;
      pixel[0] = 0;

      /* try to allocate this color */

      r = palette[i].red;
      g = palette[i].green;
      b = palette[i].blue;

      gdk_color_context_get_pixels(cc, &r, &g, &b, 1, pixel, &erg);

      /* only store if we succeed */

      if (erg) {
         /* store in palette */

         cc->palette[j].red = r;
         cc->palette[j].green = g;
         cc->palette[j].blue = b;
         cc->palette[j].pixel = pixel[0];

         /* move to next slot */

         j++;
      }
   }

   /* resize to fit */

   if (j != num_palette)
      cc->palette = g_realloc(cc->palette, j * sizeof(GdkColor));

   /* clear the hash table, we don't use it when dithering */

   if (cc->color_hash) {
      g_hash_table_foreach(cc->color_hash, free_hash_entry, NULL);
      g_hash_table_destroy(cc->color_hash);
      cc->color_hash = NULL;
   }

   /* store real palette size */

   cc->num_palette = j;

   /* switch to palette mode */

   cc->mode = GDK_CC_MODE_PALETTE;

   /* sort palette */

   qsort(cc->palette, cc->num_palette, sizeof(GdkColor), pixel_sort);

   cc->fast_dither = NULL;

   return j;
}

void gdk_color_context_init_dither(GdkColorContext * cc)
{
   gint rr, gg, bb, err, erg, erb;
   gint success = FALSE;

   g_assert(cc != NULL);

   /* now we can initialize the fast dither matrix */

   if (cc->fast_dither == NULL)
      cc->fast_dither = g_new(GdkColorContextDither, 1);

   /* Fill it.  We ignore unsuccessful allocations, they are just mapped
    * to black instead */

   for (rr = 0; rr < 32; rr++)
      for (gg = 0; gg < 32; gg++)
         for (bb = 0; bb < 32; bb++) {
            err = (rr << 3) | (rr >> 2);
            erg = (gg << 3) | (gg >> 2);
            erb = (bb << 3) | (bb >> 2);

            cc->fast_dither->fast_rgb[rr][gg][bb] =
                gdk_color_context_get_index_from_palette(cc, &err, &erg,
                                                         &erb, &success);
            cc->fast_dither->fast_err[rr][gg][bb] = err;
            cc->fast_dither->fast_erg[rr][gg][bb] = erg;
            cc->fast_dither->fast_erb[rr][gg][bb] = erb;
         }
}

void gdk_color_context_free_dither(GdkColorContext * cc)
{
   g_assert(cc != NULL);

   if (cc->fast_dither)
      g_free(cc->fast_dither);

   cc->fast_dither = NULL;
}

unsigned long
gdk_color_context_get_pixel_from_palette(GdkColorContext * cc,
                                         gushort * red,
                                         gushort * green,
                                         gushort * blue, gint * failed)
{
   unsigned long pixel = 0;
   gint dif, dr, dg, db, j = -1;
   gint mindif = 0x7fffffff;
   gint err = 0, erg = 0, erb = 0;
   gint i;

   g_assert(cc != NULL);
   g_assert(red != NULL);
   g_assert(green != NULL);
   g_assert(blue != NULL);
   g_assert(failed != NULL);

   *failed = FALSE;

   for (i = 0; i < cc->num_palette; i++) {
      dr = *red - cc->palette[i].red;
      dg = *green - cc->palette[i].green;
      db = *blue - cc->palette[i].blue;

      dif = dr * dr + dg * dg + db * db;

      if (dif < mindif) {
         mindif = dif;
         j = i;
         pixel = cc->palette[i].pixel;
         err = dr;
         erg = dg;
         erb = db;

         if (mindif == 0)
            break;
      }
   }

   /* we failed to map onto a color */

   if (j == -1)
      *failed = TRUE;
   else {
      *red = ABS(err);
      *green = ABS(erg);
      *blue = ABS(erb);
   }

   return pixel;
}

guchar
gdk_color_context_get_index_from_palette(GdkColorContext * cc,
                                         gint * red,
                                         gint * green,
                                         gint * blue, gint * failed)
{
   gint dif, dr, dg, db, j = -1;
   gint mindif = 0x7fffffff;
   gint err = 0, erg = 0, erb = 0;
   gint i;

   g_assert(cc != NULL);
   g_assert(red != NULL);
   g_assert(green != NULL);
   g_assert(blue != NULL);
   g_assert(failed != NULL);

   *failed = FALSE;

   for (i = 0; i < cc->num_palette; i++) {
      dr = *red - cc->palette[i].red;
      dg = *green - cc->palette[i].green;
      db = *blue - cc->palette[i].blue;

      dif = dr * dr + dg * dg + db * db;

      if (dif < mindif) {
         mindif = dif;
         j = i;
         err = dr;
         erg = dg;
         erb = db;

         if (mindif == 0)
            break;
      }
   }

   /* we failed to map onto a color */

   if (j == -1) {
      *failed = TRUE;
      j = 0;
   } else {
      /* return error fractions */

      *red = err;
      *green = erg;
      *blue = erb;
   }

   return j;
}
