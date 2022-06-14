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

#include "gdk.h"
#include "gdkprivate-win32.h"


GdkRegion *gdk_region_new(void)
{
   GdkRegionPrivate *private;
   GdkRegion *region;
   HRGN xregion;
   RECT emptyRect;

   /* Create an empty region */
   SetRectEmpty(&emptyRect);
   xregion = CreateRectRgnIndirect(&emptyRect);
   private = g_new(GdkRegionPrivate, 1);
   private->xregion = xregion;
   region = (GdkRegion *) private;
   region->user_data = NULL;

   return region;
}

void gdk_region_destroy(GdkRegion * region)
{
   GdkRegionPrivate *private;

   g_return_if_fail(region != NULL);

   private = (GdkRegionPrivate *) region;
   DeleteObject(private->xregion);
   g_free(private);
}

gboolean gdk_region_empty(GdkRegion * region)
{
   GdkRegionPrivate *private;
   RECT rect;

   g_return_val_if_fail(region != NULL, 0);

   private = (GdkRegionPrivate *) region;

   return (GetRgnBox(private->xregion, &rect) == NULLREGION);
}

gboolean gdk_region_equal(GdkRegion * region1, GdkRegion * region2)
{
   GdkRegionPrivate *private1;
   GdkRegionPrivate *private2;

   g_return_val_if_fail(region1 != NULL, 0);
   g_return_val_if_fail(region2 != NULL, 0);

   private1 = (GdkRegionPrivate *) region1;
   private2 = (GdkRegionPrivate *) region2;

   return EqualRgn(private1->xregion, private2->xregion);
}

void gdk_region_get_clipbox(GdkRegion * region, GdkRectangle * rectangle)
{
   GdkRegionPrivate *rp;
   RECT r;

   g_return_if_fail(region != NULL);
   g_return_if_fail(rectangle != NULL);

   rp = (GdkRegionPrivate *) region;

   GetRgnBox(rp->xregion, &r);
   rectangle->x = r.left;
   rectangle->y = r.top;
   rectangle->width = r.right - r.left;
   rectangle->height = r.bottom - r.top;
}

gboolean gdk_region_point_in(GdkRegion * region, gint x, gint y)
{
   GdkRegionPrivate *private;

   g_return_val_if_fail(region != NULL, 0);

   private = (GdkRegionPrivate *) region;

   return PtInRegion(private->xregion, x, y);
}

GdkOverlapType gdk_region_rect_in(GdkRegion * region, GdkRectangle * rect)
{
   GdkRegionPrivate *private;
   RECT r;

   g_return_val_if_fail(region != NULL, 0);

   private = (GdkRegionPrivate *) region;

   r.left = rect->x;
   r.top = rect->y;
   r.right = rect->x + rect->width;
   r.bottom = rect->y + rect->height;

   if (RectInRegion(private->xregion, &r))
      return GDK_OVERLAP_RECTANGLE_PART;

   return GDK_OVERLAP_RECTANGLE_OUT;	/*what else ? */
}

GdkRegion *gdk_region_polygon(GdkPoint * points,
                              gint npoints, GdkFillRule fill_rule)
{
   GdkRegionPrivate *private;
   GdkRegion *region;
   HRGN xregion;
   POINT *pts;
   gint xfill_rule = ALTERNATE;
   gint i;

   g_return_val_if_fail(points != NULL, NULL);
   g_return_val_if_fail(npoints != 0, NULL);	/* maybe we should check for at least three points */

   switch (fill_rule) {
   case GDK_EVEN_ODD_RULE:
      xfill_rule = ALTERNATE;
      break;

   case GDK_WINDING_RULE:
      xfill_rule = WINDING;
      break;
   }

   pts = g_malloc(npoints * sizeof(*pts));
   for (i = 0; i < npoints; i++) {
      pts[i].x = points[i].x;
      pts[i].y = points[i].y;
   }
   xregion = CreatePolygonRgn(pts, npoints, xfill_rule);
   g_free(pts);

   private = g_new(GdkRegionPrivate, 1);
   private->xregion = xregion;
   region = (GdkRegion *) private;
   region->user_data = NULL;

   return region;
}

void gdk_region_offset(GdkRegion * region, gint dx, gint dy)
{
   GdkRegionPrivate *private;

   g_return_if_fail(region != NULL);

   private = (GdkRegionPrivate *) region;

   OffsetRgn(private->xregion, dx, dy);
}

void gdk_region_shrink(GdkRegion * region, gint dx, gint dy)
{
   GdkRegionPrivate *private;
   HRGN shrunken_bbox;
   RECT r;

   g_return_if_fail(region != NULL);

   private = (GdkRegionPrivate *) region;

   if (dx > 0 || dy > 0) {
      /* We want to shrink it in one or both dimensions.
       * Is it correct just to intersect it with a smaller bounding box?
       * XXX
       */
      GetRgnBox(private->xregion, &r);
      if (dx > 0) {
         r.left += dx - dx / 2;
         r.right -= dx / 2;
      }
      if (dy > 0) {
         r.top += dy - dy / 2;
         r.bottom -= dy / 2;
      }

      shrunken_bbox = CreateRectRgnIndirect(&r);
      CombineRgn(private->xregion, private->xregion,
                 shrunken_bbox, RGN_AND);
      DeleteObject(shrunken_bbox);
   } else {
      /* Do nothing if the regions is expanded? XXX */
   }
}

GdkRegion *gdk_region_union_with_rect(GdkRegion * region,
                                      GdkRectangle * rect)
{
   GdkRegionPrivate *private;
   GdkRegion *res;
   GdkRegionPrivate *res_private;
   RECT xrect;
   HRGN rectangle;

   g_return_val_if_fail(region != NULL, NULL);

   private = (GdkRegionPrivate *) region;

   xrect.left = rect->x;
   xrect.top = rect->y;
   xrect.right = rect->x + rect->width;
   xrect.bottom = rect->y + rect->height;

   res = gdk_region_new();
   res_private = (GdkRegionPrivate *) res;

   rectangle = CreateRectRgnIndirect(&xrect);
   CombineRgn(res_private->xregion, private->xregion, rectangle, RGN_OR);
   DeleteObject(rectangle);
   return res;
}

static GdkRegion *gdk_regions_op(GdkRegion * source1,
                                 GdkRegion * source2, guint op)
{
   GdkRegionPrivate *private1;
   GdkRegionPrivate *private2;
   GdkRegion *res;
   GdkRegionPrivate *res_private;

   g_return_val_if_fail(source1 != NULL, NULL);
   g_return_val_if_fail(source2 != NULL, NULL);

   private1 = (GdkRegionPrivate *) source1;
   private2 = (GdkRegionPrivate *) source2;

   res = gdk_region_new();
   res_private = (GdkRegionPrivate *) res;

   CombineRgn(res_private->xregion, private1->xregion, private2->xregion,
              op);
   return res;
}

GdkRegion *gdk_regions_intersect(GdkRegion * source1, GdkRegion * source2)
{
   return gdk_regions_op(source1, source2, RGN_AND);
}

GdkRegion *gdk_regions_union(GdkRegion * source1, GdkRegion * source2)
{
   return gdk_regions_op(source1, source2, RGN_OR);
}

GdkRegion *gdk_regions_subtract(GdkRegion * source1, GdkRegion * source2)
{
   return gdk_regions_op(source1, source2, RGN_DIFF);
}

GdkRegion *gdk_regions_xor(GdkRegion * source1, GdkRegion * source2)
{
   return gdk_regions_op(source1, source2, RGN_XOR);
}
