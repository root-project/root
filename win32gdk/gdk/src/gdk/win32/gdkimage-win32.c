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

#include "gdk.h"                /* For gdk_error_trap_* / gdk_flush_* */
#include "gdkimage.h"
#include "gdkprivate.h"
#include "gdkwin32.h"

static void gdk_win32_image_destroy(GdkImage * image);
static void gdk_image_put(GdkImage * image,
                          GdkDrawable * drawable,
                          GdkGC * gc,
                          gint xsrc,
                          gint ysrc,
                          gint xdest, gint ydest, gint width, gint height);

static GdkImageClass image_class = {
   gdk_win32_image_destroy,
   gdk_image_put
};

static GList *image_list = NULL;

void gdk_image_exit(void)
{
   GdkImage *image;

   while (image_list) {
      image = image_list->data;
      gdk_win32_image_destroy(image);
   }
}

GdkImage *gdk_image_new_bitmap(GdkVisual * visual, gpointer data, gint w,
                               gint h)
/*
 * Desc: create a new bitmap image
 */
{
   Visual *xvisual;
   GdkImage *image;
   GdkImagePrivateWin32 *private;
   struct {
      BITMAPINFOHEADER bmiHeader;
      union {
         WORD bmiIndices[2];
         RGBQUAD bmiColors[2];
      } u;
   } bmi;
   char *bits;
   int bpl = (w - 1) / 8 + 1;
   int bpl32 = ((w - 1) / 32 + 1) * 4;

   private = g_new(GdkImagePrivateWin32, 1);
   image = (GdkImage *) private;
   private->base.ref_count = 1;
   private->base.klass = &image_class;

   image->type = GDK_IMAGE_SHARED;
   image->visual = visual;
   image->width = w;
   image->height = h;
   image->depth = 1;
   xvisual = ((GdkVisualPrivate *) visual)->xvisual;

   GDK_NOTE(MISC, g_print("gdk_image_new_bitmap: %dx%d\n", w, h));

   bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
   bmi.bmiHeader.biWidth = w;
   bmi.bmiHeader.biHeight = -h;
   bmi.bmiHeader.biPlanes = 1;
   bmi.bmiHeader.biBitCount = 1;
   bmi.bmiHeader.biCompression = BI_RGB;
   bmi.bmiHeader.biSizeImage = 0;
   bmi.bmiHeader.biXPelsPerMeter = bmi.bmiHeader.biYPelsPerMeter = 0;
   bmi.bmiHeader.biClrUsed = 0;
   bmi.bmiHeader.biClrImportant = 0;

   bmi.u.bmiColors[0].rgbBlue =
       bmi.u.bmiColors[0].rgbGreen = bmi.u.bmiColors[0].rgbRed = 0x00;
   bmi.u.bmiColors[0].rgbReserved = 0x00;

   bmi.u.bmiColors[1].rgbBlue =
       bmi.u.bmiColors[1].rgbGreen = bmi.u.bmiColors[1].rgbRed = 0xFF;
   bmi.u.bmiColors[1].rgbReserved = 0x00;

   private->ximage = CreateDIBSection(gdk_DC, (BITMAPINFO *) & bmi,
                                      DIB_RGB_COLORS, &bits, NULL, 0);
   if (bpl != bpl32) {
      /* Win32 expects scanlines in DIBs to be 32 bit aligned */
      int i;
      for (i = 0; i < h; i++)
         memmove(bits + i * bpl32, ((char *) data) + i * bpl, bpl);
   } else
      memmove(bits, data, bpl * h);
   image->mem = bits;
   image->bpl = bpl32;
   image->byte_order = GDK_MSB_FIRST;

   image->bpp = 1;
   return (image);
}                               /* gdk_image_new_bitmap() */

void gdk_image_init(void)
{
}

static GdkImage *gdk_image_new_with_depth(GdkImageType type,
                                          GdkVisual * visual,
                                          gint width,
                                          gint height, gint depth)
{
   GdkImage *image;
   GdkImagePrivateWin32 *private;
   Visual *xvisual;
   struct {
      BITMAPINFOHEADER bmiHeader;
      union {
         WORD bmiIndices[256];
         DWORD bmiMasks[3];
         RGBQUAD bmiColors[256];
      } u;
   } bmi;
   UINT iUsage;
   int i;

   if (type == GDK_IMAGE_FASTEST || type == GDK_IMAGE_NORMAL)
      type = GDK_IMAGE_SHARED;

   GDK_NOTE(MISC, g_print("gdk_image_new_with_depth: %dx%dx%d %s\n",
                          width, height, depth,
                          (type == GDK_IMAGE_SHARED ? "shared" :
                           (type ==
                            GDK_IMAGE_SHARED_PIXMAP ? "shared_pixmap" :
                            "???"))));

   private = g_new(GdkImagePrivateWin32, 1);
   image = (GdkImage *) private;

   private->base.ref_count = 1;
   private->base.klass = &image_class;

   image->type = type;
   image->visual = visual;
   image->width = width;
   image->height = height;
   image->depth = depth;

   xvisual = ((GdkVisualPrivate *) visual)->xvisual;

   bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
   bmi.bmiHeader.biWidth = width;
   bmi.bmiHeader.biHeight = -height;
   bmi.bmiHeader.biPlanes = 1;
   if (depth == 15)
      bmi.bmiHeader.biBitCount = 16;
   else
      bmi.bmiHeader.biBitCount = depth;
#if 1
   if (depth == 16)
      bmi.bmiHeader.biCompression = BI_BITFIELDS;
   else
#endif
      bmi.bmiHeader.biCompression = BI_RGB;
   bmi.bmiHeader.biSizeImage = 0;
   bmi.bmiHeader.biXPelsPerMeter = bmi.bmiHeader.biYPelsPerMeter = 0;
   bmi.bmiHeader.biClrUsed = 0;
   bmi.bmiHeader.biClrImportant = 0;

   if (image->visual->type == GDK_VISUAL_PSEUDO_COLOR) {
      iUsage = DIB_PAL_COLORS;
      for (i = 0; i < 256; i++)
         bmi.u.bmiIndices[i] = i;
   } else {
      if (depth == 1) {
         bmi.u.bmiColors[0].rgbBlue =
             bmi.u.bmiColors[0].rgbGreen =
             bmi.u.bmiColors[0].rgbRed = 0x00;
         bmi.u.bmiColors[0].rgbReserved = 0x00;

         bmi.u.bmiColors[1].rgbBlue =
             bmi.u.bmiColors[1].rgbGreen =
             bmi.u.bmiColors[1].rgbRed = 0xFF;
         bmi.u.bmiColors[1].rgbReserved = 0x00;

      }
#if 1
      else if (depth == 16) {
         bmi.u.bmiMasks[0] = visual->red_mask;
         bmi.u.bmiMasks[1] = visual->green_mask;
         bmi.u.bmiMasks[2] = visual->blue_mask;
      }
#endif
      iUsage = DIB_RGB_COLORS;
   }

   private->ximage =
       CreateDIBSection(gdk_DC, (BITMAPINFO *) & bmi, iUsage,
                        &image->mem, NULL, 0);

   if (private->ximage == NULL) {
      WIN32_GDI_FAILED("CreateDIBSection");
      g_free(image);
      return NULL;
   }

   switch (depth) {
   case 1:
   case 8:
      image->bpp = 1;
      break;
   case 15:
   case 16:
      image->bpp = 2;
      break;
   case 24:
      image->bpp = 3;
      break;
   case 32:
      image->bpp = 4;
      break;
   default:
      g_warning("gdk_image_new_with_depth: depth = %d", depth);
      g_assert_not_reached();
   }
   image->byte_order = GDK_LSB_FIRST;
   if (depth == 1)
      image->bpl = ((width - 1) / 32 + 1) * 4;
   else
      image->bpl = ((width * image->bpp - 1) / 4 + 1) * 4;

   GDK_NOTE(MISC, g_print("... = %#x mem = %#x, bpl = %d\n",
                          private->ximage, image->mem, image->bpl));

   return image;
}

GdkImage *gdk_image_new(GdkImageType type,
                        GdkVisual * visual, gint width, gint height)
{
   GdkVisualPrivate *visual_private = (GdkVisualPrivate *) visual;
   return gdk_image_new_with_depth(type, visual, width, height,
#if 0
                                   visual_private->xvisual->bitspixel);
#else
                                   visual->depth);
#endif
}

GdkImage *gdk_image_bitmap_new(GdkImageType type,
                               GdkVisual * visual, gint width, gint height)
{
   return gdk_image_new_with_depth(type, visual, width, height, 1);
}

GdkImage *gdk_image_get(GdkWindow * window,
                        gint x, gint y, gint width, gint height)
{
   GdkImage *image;
   GdkImagePrivateWin32 *private;
   HDC hdc, memdc;
   struct {
      BITMAPINFOHEADER bmiHeader;
      union {
         WORD bmiIndices[256];
         DWORD bmiMasks[3];
         RGBQUAD bmiColors[256];
      } u;
   } bmi;
   HGDIOBJ oldbitmap1, oldbitmap2;
   UINT iUsage;
   BITMAP bm;
   int i;

   g_return_val_if_fail(window != NULL, NULL);

   if (GDK_DRAWABLE_DESTROYED(window))
      return NULL;

   GDK_NOTE(MISC, g_print("gdk_image_get: %#x %dx%d@+%d+%d\n",
                          GDK_DRAWABLE_XID(window), width, height, x, y));

   private = g_new(GdkImagePrivateWin32, 1);
   image = (GdkImage *) private;

   private->base.ref_count = 1;
   private->base.klass = &image_class;

   image->type = GDK_IMAGE_SHARED;
   image->visual = gdk_window_get_visual(window);
   image->width = width;
   image->height = height;

   /* This function is called both to blit from a window and from
    * a pixmap.
    */
   if (GDK_DRAWABLE_TYPE(window) == GDK_DRAWABLE_PIXMAP) {
      if ((hdc = CreateCompatibleDC(NULL)) == NULL) {
         WIN32_GDI_FAILED("CreateCompatibleDC");
         g_free(image);
         return NULL;
      }
      if ((oldbitmap1 =
           SelectObject(hdc, GDK_DRAWABLE_XID(window))) == NULL) {
         WIN32_GDI_FAILED("SelectObject");
         DeleteDC(hdc);
         g_free(image);
         return NULL;
      }
      GetObject(GDK_DRAWABLE_XID(window), sizeof(BITMAP), &bm);
      GDK_NOTE(MISC,
               g_print
               ("gdk_image_get: bmWidth = %d, bmHeight = %d, bmWidthBytes = %d, bmBitsPixel = %d\n",
                bm.bmWidth, bm.bmHeight, bm.bmWidthBytes, bm.bmBitsPixel));
      image->depth = bm.bmBitsPixel;
      if (image->depth <= 8) {
         iUsage = DIB_PAL_COLORS;
         for (i = 0; i < 256; i++)
            bmi.u.bmiIndices[i] = i;
      } else
         iUsage = DIB_RGB_COLORS;
   } else {
      if ((hdc = GetDC(GDK_DRAWABLE_XID(window))) == NULL) {
         WIN32_GDI_FAILED("GetDC");
         g_free(image);
         return NULL;
      }
      image->depth = gdk_visual_get_system()->depth;
      if (image->visual->type == GDK_VISUAL_PSEUDO_COLOR) {
         iUsage = DIB_PAL_COLORS;
         for (i = 0; i < 256; i++)
            bmi.u.bmiIndices[i] = i;
      } else
         iUsage = DIB_RGB_COLORS;
   }

   if ((memdc = CreateCompatibleDC(hdc)) == NULL) {
      WIN32_GDI_FAILED("CreateCompatibleDC");
      if (GDK_DRAWABLE_TYPE(window) == GDK_DRAWABLE_PIXMAP) {
         SelectObject(hdc, oldbitmap1);
         DeleteDC(hdc);
      } else {
         ReleaseDC(GDK_DRAWABLE_XID(window), hdc);
      }
      g_free(image);
      return NULL;
   }

   bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
   bmi.bmiHeader.biWidth = width;
   bmi.bmiHeader.biHeight = -height;
   bmi.bmiHeader.biPlanes = 1;
   bmi.bmiHeader.biBitCount = image->depth;
   if (image->depth == 16) {
      bmi.bmiHeader.biCompression = BI_BITFIELDS;
      if (image->visual == NULL) {
         /* XXX ??? Is it always this if depth==16 and a pixmap? Guess so. */
         bmi.u.bmiMasks[0] = 0xf800;
         bmi.u.bmiMasks[1] = 0x07e0;
         bmi.u.bmiMasks[2] = 0x001f;
      } else {
         bmi.u.bmiMasks[0] = image->visual->red_mask;
         bmi.u.bmiMasks[1] = image->visual->green_mask;
         bmi.u.bmiMasks[2] = image->visual->blue_mask;
      }
   } else
      bmi.bmiHeader.biCompression = BI_RGB;
   bmi.bmiHeader.biSizeImage = 0;
   bmi.bmiHeader.biXPelsPerMeter = bmi.bmiHeader.biYPelsPerMeter = 0;
   bmi.bmiHeader.biClrUsed = 0;
   bmi.bmiHeader.biClrImportant = 0;

   if ((private->ximage =
        CreateDIBSection(hdc, (BITMAPINFO *) & bmi, iUsage,
                         &image->mem, NULL, 0)) == NULL) {
      WIN32_GDI_FAILED("CreateDIBSection");
      DeleteDC(memdc);
      if (GDK_DRAWABLE_TYPE(window) == GDK_DRAWABLE_PIXMAP) {
         SelectObject(hdc, oldbitmap1);
         DeleteDC(hdc);
      } else {
         ReleaseDC(GDK_DRAWABLE_XID(window), hdc);
      }
      g_free(image);
      return NULL;
   }

   if ((oldbitmap2 = SelectObject(memdc, private->ximage)) == NULL) {
      WIN32_GDI_FAILED("SelectObject");
      DeleteObject(private->ximage);
      DeleteDC(memdc);
      if (GDK_DRAWABLE_TYPE(window) == GDK_DRAWABLE_PIXMAP) {
         SelectObject(hdc, oldbitmap1);
         DeleteDC(hdc);
      } else {
         ReleaseDC(GDK_DRAWABLE_XID(window), hdc);
      }
      g_free(image);
      return NULL;
   }

   if (!BitBlt(memdc, 0, 0, width, height, hdc, x, y, SRCCOPY)) {
      WIN32_GDI_FAILED("BitBlt");
      SelectObject(memdc, oldbitmap2);
      DeleteObject(private->ximage);
      DeleteDC(memdc);
      if (GDK_DRAWABLE_TYPE(window) == GDK_DRAWABLE_PIXMAP) {
         SelectObject(hdc, oldbitmap1);
         DeleteDC(hdc);
      } else {
         ReleaseDC(GDK_DRAWABLE_XID(window), hdc);
      }
      g_free(image);
      return NULL;
   }

   if (SelectObject(memdc, oldbitmap2) == NULL)
      WIN32_GDI_FAILED("SelectObject");

   if (!DeleteDC(memdc))
      WIN32_GDI_FAILED("DeleteDC");

   if (GDK_DRAWABLE_TYPE(window) == GDK_DRAWABLE_PIXMAP) {
      SelectObject(hdc, oldbitmap1);
      DeleteDC(hdc);
   } else {
      ReleaseDC(GDK_DRAWABLE_XID(window), hdc);
   }

   switch (image->depth) {
   case 1:
   case 8:
      image->bpp = 1;
      break;
   case 15:
   case 16:
      image->bpp = 2;
      break;
   case 24:
      image->bpp = 3;
      break;
   case 32:
      image->bpp = 4;
      break;
   default:
      g_warning("gdk_image_get: image->depth = %d", image->depth);
      g_assert_not_reached();
   }
   image->byte_order = GDK_LSB_FIRST;
   if (image->depth == 1)
      image->bpl = ((width - 1) / 32 + 1) * 4;
   else
      image->bpl = ((width * image->bpp - 1) / 4 + 1) * 4;

   GDK_NOTE(MISC, g_print("... = %#x mem = %#x, bpl = %d\n",
                          private->ximage, image->mem, image->bpl));

   return image;
}

guint32 gdk_image_get_pixel(GdkImage * image, gint x, gint y)
{
   guint32 pixel;

   g_return_val_if_fail(image != NULL, 0);

   g_return_val_if_fail(x >= 0 && x < image->width
                        && y >= 0 && y < image->height, 0);

   if (image->depth == 1)
      pixel =
          (((char *) image->mem)[y * image->bpl +
                                 (x >> 3)] & (1 << (7 - (x & 0x7)))) != 0;
   else {
      guchar *pixelp =
          (guchar *) image->mem + y * image->bpl + x * image->bpp;

      switch (image->bpp) {
      case 1:
         pixel = *pixelp;
         break;

         /* Windows is always LSB, no need to check image->byte_order. */
      case 2:
         pixel = pixelp[0] | (pixelp[1] << 8);
         break;

      case 3:
         pixel = pixelp[0] | (pixelp[1] << 8) | (pixelp[2] << 16);
         break;

      case 4:
         pixel = pixelp[0] | (pixelp[1] << 8) | (pixelp[2] << 16);
         break;
      }
   }

   return pixel;
}

void gdk_image_put_pixel(GdkImage * image, gint x, gint y, guint32 pixel)
{
   g_return_if_fail(image != NULL);

   g_return_if_fail(x >= 0 && x < image->width && y >= 0
                    && y < image->height);

   if (image->depth == 1)
      if (pixel & 1)
         ((guchar *) image->mem)[y * image->bpl + (x >> 3)] |=
             (1 << (7 - (x & 0x7)));
      else
         ((guchar *) image->mem)[y * image->bpl + (x >> 3)] &=
             ~(1 << (7 - (x & 0x7)));
   else {
      guchar *pixelp =
          (guchar *) image->mem + y * image->bpl + x * image->bpp;

      /* Windows is always LSB, no need to check image->byte_order. */
      switch (image->bpp) {
      case 4:
         pixelp[3] = 0;
      case 3:
         pixelp[2] = ((pixel >> 16) & 0xFF);
      case 2:
         pixelp[1] = ((pixel >> 8) & 0xFF);
      case 1:
         pixelp[0] = (pixel & 0xFF);
      }
   }
}

static void gdk_win32_image_destroy(GdkImage * image)
{
   GdkImagePrivateWin32 *private;

   g_return_if_fail(image != NULL);

   private = (GdkImagePrivateWin32 *) image;

   GDK_NOTE(MISC, g_print("gdk_win32_image_destroy: %#x%s\n",
                          private->ximage,
                          (image->type == GDK_IMAGE_SHARED_PIXMAP ?
                           " (shared pixmap)" : "")));

   switch (image->type) {
   case GDK_IMAGE_SHARED_PIXMAP:
      break;                    /* The Windows bitmap has already been
                                 * (or will be) deleted when freeing
                                 * the corresponding pixmap.
                                 */

   case GDK_IMAGE_SHARED:
      if (!DeleteObject(private->ximage))
         WIN32_GDI_FAILED("DeleteObject");
      break;

   default:
      g_assert_not_reached();
   }

   g_free(image);
}

static void
gdk_image_put(GdkImage * image,
              GdkDrawable * drawable,
              GdkGC * gc,
              gint xsrc,
              gint ysrc, gint xdest, gint ydest, gint width, gint height)
{
   GdkDrawablePrivate *drawable_private;
   GdkImagePrivateWin32 *image_private;
   GdkGCPrivate *gc_private;
   HDC hdc;
   GdkColormapPrivateWin32 *colormap_private;

   g_return_if_fail(drawable != NULL);
   g_return_if_fail(image != NULL);
   g_return_if_fail(gc != NULL);

   if (GDK_DRAWABLE_DESTROYED(drawable))
      return;
   image_private = (GdkImagePrivateWin32 *) image;
   drawable_private = (GdkDrawablePrivate *) drawable;
   gc_private = (GdkGCPrivate *) gc;

   hdc = gdk_gc_predraw(drawable, gc_private, 0);
   colormap_private =
       (GdkColormapPrivateWin32 *) drawable_private->colormap;
   if (colormap_private && colormap_private->xcolormap->rc_palette) {
      DIBSECTION ds;
      static struct {
         BITMAPINFOHEADER bmiHeader;
         WORD bmiIndices[256];
      } bmi;
      static gboolean bmi_inited = FALSE;
      int i;

      if (!bmi_inited) {
         for (i = 0; i < 256; i++)
            bmi.bmiIndices[i] = i;
         bmi_inited = TRUE;
      }

      if (GetObject(image_private->ximage, sizeof(DIBSECTION),
                    &ds) != sizeof(DIBSECTION)) {
         WIN32_GDI_FAILED("GetObject");
      }
#if 0
      g_print
          ("xdest = %d, ydest = %d, xsrc = %d, ysrc = %d, width = %d, height = %d\n",
           xdest, ydest, xsrc, ysrc, width, height);
      g_print
          ("bmWidth = %d, bmHeight = %d, bmBitsPixel = %d, bmBits = %p\n",
           ds.dsBm.bmWidth, ds.dsBm.bmHeight, ds.dsBm.bmBitsPixel,
           ds.dsBm.bmBits);
      g_print
          ("biWidth = %d, biHeight = %d, biBitCount = %d, biClrUsed = %d\n",
           ds.dsBmih.biWidth, ds.dsBmih.biHeight, ds.dsBmih.biBitCount,
           ds.dsBmih.biClrUsed);
#endif
      bmi.bmiHeader = ds.dsBmih;
      /* I have spent hours on getting the parameters to
       * SetDIBitsToDevice right. I wonder what drugs the guys in
       * Redmond were on when they designed this API.
       */
      if (SetDIBitsToDevice(hdc,
                            xdest, ydest,
                            width, height,
                            xsrc, (-ds.dsBmih.biHeight) - height - ysrc,
                            0, -ds.dsBmih.biHeight,
                            ds.dsBm.bmBits,
                            (CONST BITMAPINFO *) & bmi,
                            DIB_PAL_COLORS) == 0)
         WIN32_GDI_FAILED("SetDIBitsToDevice");
   } else {
      HDC memdc;
      HGDIOBJ oldbitmap;

      if ((memdc = CreateCompatibleDC(hdc)) == NULL) {
         WIN32_GDI_FAILED("CreateCompatibleDC");
         gdk_gc_postdraw(drawable, gc_private, 0);
         return;
      }

      if ((oldbitmap = SelectObject(memdc, image_private->ximage)) == NULL) {
         WIN32_GDI_FAILED("SelectObject");
         gdk_gc_postdraw(drawable, gc_private, 0);
         return;
      }
      if (!BitBlt(hdc, xdest, ydest, width, height,
                  memdc, xsrc, ysrc, SRCCOPY))
         WIN32_GDI_FAILED("BitBlt");

      if (SelectObject(memdc, oldbitmap) == NULL)
         WIN32_GDI_FAILED("SelectObject");

      if (!DeleteDC(memdc))
         WIN32_GDI_FAILED("DeleteDC");
   }
   gdk_gc_postdraw(drawable, gc_private, 0);
}
