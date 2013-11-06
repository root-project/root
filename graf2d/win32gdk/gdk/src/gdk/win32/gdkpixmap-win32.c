/* GDK - The GIMP Drawing Kit
 * Copyright (C) 1995-1997 Peter Mattis, Spencer Kimball and Josh MacDonald
 * Copyright (C) 1998-1999 Tor Lillqvist
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "gdkpixmap.h"
#include "gdkprivate.h"
#include "gdkwin32.h"

typedef struct {
   gchar *color_string;
   GdkColor color;
   gint transparent;
} _GdkPixmapColor;

typedef struct {
   guint ncolors;
   GdkColormap *colormap;
   gulong pixels[1];
} _GdkPixmapInfo;

static void gdk_win32_pixmap_destroy(GdkPixmap * pixmap)
{
   GdkDrawablePrivate *private = (GdkDrawablePrivate *) pixmap;

   GDK_NOTE(MISC, g_print("gdk_win32_pixmap_destroy: %#x\n",
                          GDK_DRAWABLE_XID(pixmap)));

   if (!DeleteObject(GDK_DRAWABLE_XID(pixmap)))
      WIN32_GDI_FAILED("DeleteObject");

   gdk_xid_table_remove(GDK_DRAWABLE_XID(pixmap));

   g_free(GDK_DRAWABLE_WIN32DATA(pixmap));
}

static GdkDrawable *gdk_win32_pixmap_alloc(void)
{
   GdkDrawable *drawable;
   GdkDrawablePrivate *private;

   static GdkDrawableClass klass;
   static gboolean initialized = FALSE;

   if (!initialized) {
      initialized = TRUE;

      klass = _gdk_win32_drawable_class;
      klass.destroy = gdk_win32_pixmap_destroy;
   }

   drawable = gdk_drawable_alloc();
   private = (GdkDrawablePrivate *) drawable;

   private->klass = &klass;
   private->klass_data = g_new(GdkDrawableWin32Data, 1);
   private->window_type = GDK_DRAWABLE_PIXMAP;

   return drawable;
}

GdkPixmap *gdk_pixmap_new(GdkWindow * window,
                          gint width, gint height, gint depth)
{
   GdkPixmap *pixmap;
   GdkDrawablePrivate *private;
   struct {
      BITMAPINFOHEADER bmiHeader;
      union {
         WORD bmiIndices[256];
         DWORD bmiMasks[3];
         RGBQUAD bmiColors[256];
      } u;
   } bmi;
   UINT iUsage;
   HDC hdc;
   GdkVisual *visual;
   guchar *bits;
   gint i;

   g_return_val_if_fail(window == NULL || GDK_IS_WINDOW(window), NULL);
   g_return_val_if_fail((window != NULL) || (depth != -1), NULL);
   g_return_val_if_fail((width != 0) && (height != 0), NULL);

   if (!window)
      window = gdk_parent_root;

   if (GDK_DRAWABLE_DESTROYED(window))
      return NULL;

   if (depth == -1)
      depth = gdk_drawable_get_visual(window)->depth;

   GDK_NOTE(MISC, g_print("gdk_pixmap_new: %dx%dx%d\n",
                          width, height, depth));

   pixmap = gdk_win32_pixmap_alloc();
   private = (GdkDrawablePrivate *) pixmap;

   visual = gdk_drawable_get_visual(window);

   if ((hdc = GetDC(GDK_DRAWABLE_XID(window))) == NULL) {
      WIN32_GDI_FAILED("GetDC");
      g_free(private);
      return NULL;
   }

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

   iUsage = DIB_RGB_COLORS;
   if (depth == 1) {
      bmi.u.bmiColors[0].rgbBlue =
          bmi.u.bmiColors[0].rgbGreen = bmi.u.bmiColors[0].rgbRed = 0x00;
      bmi.u.bmiColors[0].rgbReserved = 0x00;

      bmi.u.bmiColors[1].rgbBlue =
          bmi.u.bmiColors[1].rgbGreen = bmi.u.bmiColors[1].rgbRed = 0xFF;
      bmi.u.bmiColors[1].rgbReserved = 0x00;
      private->colormap = NULL;
   } else {
      private->colormap = ((GdkWindowPrivate *) window)->drawable.colormap;
      if (private->colormap == NULL)
         private->colormap = gdk_colormap_get_system();

      if (depth == 8) {
         iUsage = DIB_PAL_COLORS;
         for (i = 0; i < 256; i++)
            bmi.u.bmiIndices[i] = i;
      } else {
         if (depth != visual->depth)
            g_warning
                ("gdk_pixmap_new: depth %d doesn't match display depth %d",
                 depth, visual->depth);
#if 1
         if (depth == 16) {
            bmi.u.bmiMasks[0] = visual->red_mask;
            bmi.u.bmiMasks[1] = visual->green_mask;
            bmi.u.bmiMasks[2] = visual->blue_mask;
         }
#endif
      }
   }
   if ((GDK_DRAWABLE_WIN32DATA(pixmap)->xid =
        CreateDIBSection(hdc, (BITMAPINFO *) & bmi,
                         iUsage, (PVOID *) & bits, NULL, 0)) == NULL) {
      WIN32_GDI_FAILED("CreateDIBSection");
      ReleaseDC(GDK_DRAWABLE_XID(window), hdc);
      g_free(pixmap);
      return NULL;
   }
   ReleaseDC(GDK_DRAWABLE_XID(window), hdc);

   GDK_NOTE(MISC, g_print("... = %#x\n", GDK_DRAWABLE_XID(pixmap)));

   private->width = width;
   private->height = height;

   gdk_xid_table_insert(&GDK_DRAWABLE_XID(pixmap), pixmap);

   return pixmap;
}

GdkPixmap *gdk_pixmap_create_on_shared_image(GdkImage ** image_return,
                                             GdkWindow * window,
                                             GdkVisual * visual,
                                             gint width,
                                             gint height, gint depth)
{
   GdkPixmap *pixmap;
   GdkDrawablePrivate *private;

   g_return_val_if_fail(window != NULL, NULL);


   if (depth == 1)
      *image_return =
          gdk_image_bitmap_new(GDK_IMAGE_SHARED_PIXMAP, visual, width,
                               height);
   else {
      g_return_val_if_fail(depth == visual->depth, NULL);
      *image_return =
          gdk_image_new(GDK_IMAGE_SHARED_PIXMAP, visual, width, height);
   }

   g_return_val_if_fail(*image_return != NULL, NULL);

   pixmap = gdk_win32_pixmap_alloc();
   private = (GdkDrawablePrivate *) pixmap;

   GDK_DRAWABLE_WIN32DATA(pixmap)->xid =
       ((GdkImagePrivateWin32 *) * image_return)->ximage;
   private->colormap = ((GdkWindowPrivate *) window)->drawable.colormap;
   private->width = width;
   private->height = height;

   gdk_xid_table_insert(&GDK_DRAWABLE_XID(pixmap), pixmap);

   GDK_NOTE(MISC,
            g_print("gdk_pixmap_create_on_shared_image: %dx%dx%d = %#x\n",
                    width, height, depth, GDK_DRAWABLE_XID(pixmap)));

   return pixmap;
}

static unsigned char mirror[256] = {
   0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0,
   0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0, 0x70, 0xf0,
   0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8,
   0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8,
   0x04, 0x84, 0x44, 0xc4, 0x24, 0xa4, 0x64, 0xe4,
   0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4,
   0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec,
   0x1c, 0x9c, 0x5c, 0xdc, 0x3c, 0xbc, 0x7c, 0xfc,
   0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2,
   0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2,
   0x0a, 0x8a, 0x4a, 0xca, 0x2a, 0xaa, 0x6a, 0xea,
   0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa,
   0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6,
   0x16, 0x96, 0x56, 0xd6, 0x36, 0xb6, 0x76, 0xf6,
   0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee,
   0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe,
   0x01, 0x81, 0x41, 0xc1, 0x21, 0xa1, 0x61, 0xe1,
   0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
   0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9,
   0x19, 0x99, 0x59, 0xd9, 0x39, 0xb9, 0x79, 0xf9,
   0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5,
   0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5,
   0x0d, 0x8d, 0x4d, 0xcd, 0x2d, 0xad, 0x6d, 0xed,
   0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd,
   0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3,
   0x13, 0x93, 0x53, 0xd3, 0x33, 0xb3, 0x73, 0xf3,
   0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb,
   0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb,
   0x07, 0x87, 0x47, 0xc7, 0x27, 0xa7, 0x67, 0xe7,
   0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7,
   0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef,
   0x1f, 0x9f, 0x5f, 0xdf, 0x3f, 0xbf, 0x7f, 0xff
};

GdkPixmap *gdk_bitmap_create_from_data(GdkWindow * window,
                                       const gchar * data,
                                       gint width, gint height)
{
   GdkPixmap *pixmap;
   GdkDrawablePrivate *private;
   gint i, j, bpl, aligned_bpl;
   guchar *bits;

   g_return_val_if_fail(data != NULL, NULL);
   g_return_val_if_fail((width != 0) && (height != 0), NULL);
   g_return_val_if_fail(window == NULL || GDK_IS_WINDOW(window), NULL);

   if (!window)
      window = gdk_parent_root;

   if (GDK_DRAWABLE_DESTROYED(window))
      return NULL;

   pixmap = gdk_win32_pixmap_alloc();
   private = (GdkDrawablePrivate *) pixmap;

   private->width = width;
   private->height = height;

   bpl = ((width - 1) / 8 + 1);
   aligned_bpl = ((bpl - 1) / 2 + 1) * 2;
   bits = g_malloc(aligned_bpl * height);
   for (i = 0; i < height; i++)
      for (j = 0; j < bpl; j++)
         bits[i * aligned_bpl + j] = mirror[(guchar) data[i * bpl + j]];

   GDK_DRAWABLE_WIN32DATA(pixmap)->xid =
       CreateBitmap(width, height, 1, 1, bits);

   GDK_NOTE(MISC, g_print("gdk_bitmap_create_from_data: %dx%d = %#x\n",
                          width, height, GDK_DRAWABLE_XID(pixmap)));

   g_free(bits);

   private->colormap = NULL;
   gdk_xid_table_insert(&GDK_DRAWABLE_XID(pixmap), pixmap);

   return pixmap;
}

GdkPixmap *gdk_pixmap_create_from_data(GdkWindow * window,
                                       const gchar * data,
                                       gint width,
                                       gint height,
                                       gint depth,
                                       GdkColor * fg, GdkColor * bg)
{
   /* Oh wow. I struggled with dozens of lines of code trying to get
    * this right using a monochrome Win32 bitmap created from data, and
    * a colour DIB section as the result, trying setting pens,
    * background colors, whatnot and BitBlt:ing.  Nope. Then finally I
    * realized it's much easier to do it using gdk...:
    */

   GdkPixmap *result = gdk_pixmap_new(window, width, height, depth);
   GdkPixmap *source =
       gdk_bitmap_create_from_data(window, data, width, height);
   GdkGC *gc = gdk_gc_new(result);
   gdk_gc_set_foreground(gc, fg);
   gdk_gc_set_background(gc, bg);
   gdk_draw_drawable(result, gc, source, 0, 0, 0, 0, width, height);
   gdk_drawable_unref(source);
   gdk_gc_unref(gc);

   GDK_NOTE(MISC, g_print("gdk_pixmap_create_from_data: %dx%dx%d = %#x\n",
                          width, height, depth, GDK_DRAWABLE_XID(result)));
   return result;
}

static gint
gdk_pixmap_seek_string(FILE * infile,
                       const gchar * str, gint skip_comments)
{
   char instr[1024];

   while (!feof(infile)) {
      fscanf(infile, "%1023s", instr);
      if (skip_comments == TRUE && strcmp(instr, "/*") == 0) {
         fscanf(infile, "%1023s", instr);
         while (!feof(infile) && strcmp(instr, "*/") != 0)
            fscanf(infile, "%1023s", instr);
         fscanf(infile, "%1023s", instr);
      }
      if (strcmp(instr, str) == 0)
         return TRUE;
   }

   return FALSE;
}

static gint gdk_pixmap_seek_char(FILE * infile, gchar c)
{
   gint b, oldb;

   while ((b = getc(infile)) != EOF) {
      if (c != b && b == '/') {
         b = getc(infile);
         if (b == EOF)
            return FALSE;
         else if (b == '*') {   /* we have a comment */
            b = -1;
            do {
               oldb = b;
               b = getc(infile);
               if (b == EOF)
                  return FALSE;
            }
            while (!(oldb == '*' && b == '/'));
         }
      } else if (c == b)
         return TRUE;
   }
   return FALSE;
}

static gint
gdk_pixmap_read_string(FILE * infile, gchar ** buffer, guint * buffer_size)
{
   gint c;
   guint cnt = 0, bufsiz, ret = FALSE;
   gchar *buf;

   buf = *buffer;
   bufsiz = *buffer_size;
   if (buf == NULL) {
      bufsiz = 10 * sizeof(gchar);
      buf = g_new(gchar, bufsiz);
   }

   do
      c = getc(infile);
   while (c != EOF && c != '"');

   if (c != '"')
      goto out;

   while ((c = getc(infile)) != EOF) {
      if (cnt == bufsiz) {
         guint new_size = bufsiz * 2;
         if (new_size > bufsiz)
            bufsiz = new_size;
         else
            goto out;

         buf = (gchar *) g_realloc(buf, bufsiz);
         buf[bufsiz - 1] = '\0';
      }

      if (c != '"')
         buf[cnt++] = c;
      else {
         buf[cnt] = 0;
         ret = TRUE;
         break;
      }
   }

 out:
   buf[bufsiz - 1] = '\0';      /* ensure null termination for errors */
   *buffer = buf;
   *buffer_size = bufsiz;
   return ret;
}

static gchar *gdk_pixmap_skip_whitespaces(gchar * buffer)
{
   gint32 index = 0;

   while (buffer[index] != 0
          && (buffer[index] == 0x20 || buffer[index] == 0x09))
      index++;

   return &buffer[index];
}

static gchar *gdk_pixmap_skip_string(gchar * buffer)
{
   gint32 index = 0;

   while (buffer[index] != 0 && buffer[index] != 0x20
          && buffer[index] != 0x09)
      index++;

   return &buffer[index];
}

#define MAX_COLOR_LEN 120

static gchar *gdk_pixmap_extract_color(gchar * buffer)
{
   gint counter, numnames;
   gchar *ptr = NULL, ch, temp[128];
   gchar color[MAX_COLOR_LEN], *retcol;
   gint space;

   counter = 0;
   while (ptr == NULL) {
      if (buffer[counter] == 'c') {
         ch = buffer[counter + 1];
         if (ch == 0x20 || ch == 0x09)
            ptr = &buffer[counter + 1];
      } else if (buffer[counter] == 0)
         return NULL;

      counter++;
   }

   ptr = gdk_pixmap_skip_whitespaces(ptr);

   if (ptr[0] == 0)
      return NULL;
   else if (ptr[0] == '#') {
      counter = 1;
      while (ptr[counter] != 0 &&
             ((ptr[counter] >= '0' && ptr[counter] <= '9') ||
              (ptr[counter] >= 'a' && ptr[counter] <= 'f') ||
              (ptr[counter] >= 'A' && ptr[counter] <= 'F')))
         counter++;

      retcol = g_new(gchar, counter + 1);
      strncpy(retcol, ptr, counter);

      retcol[counter] = 0;

      return retcol;
   }

   color[0] = 0;
   numnames = 0;

   space = MAX_COLOR_LEN - 1;
   while (space > 0) {
      sscanf(ptr, "%127s", temp);

      if (((gint) ptr[0] == 0) ||
          (strcmp("s", temp) == 0) || (strcmp("m", temp) == 0) ||
          (strcmp("g", temp) == 0) || (strcmp("g4", temp) == 0)) {
         break;
      } else {
         if (numnames > 0) {
            space -= 1;
            strcat(color, " ");
         }
         strncat(color, temp, space);
         space -= MIN(space, (gint)strlen(temp));
         ptr = gdk_pixmap_skip_string(ptr);
         ptr = gdk_pixmap_skip_whitespaces(ptr);
         numnames++;
      }
   }

   retcol = g_strdup(color);
   return retcol;
}


enum buffer_op {
   op_header,
   op_cmap,
   op_body
};


static void gdk_xpm_destroy_notify(gpointer data)
{
   _GdkPixmapInfo *info = (_GdkPixmapInfo *) data;
   GdkColor color;
   guint i;

   for (i = 0; i < info->ncolors; i++) {
      color.pixel = info->pixels[i];
      gdk_colormap_free_colors(info->colormap, &color, 1);
   }

   gdk_colormap_unref(info->colormap);
   g_free(info);
}

static GdkPixmap *_gdk_pixmap_create_from_xpm(GdkWindow * window,
                                              GdkColormap * colormap,
                                              GdkBitmap ** mask,
                                              GdkColor * transparent_color,
                                              gchar *
                                              (*get_buf) (enum buffer_op
                                                          op,
                                                          gpointer handle),
                                              gpointer handle)
{
   GdkPixmap *pixmap = NULL;
   GdkImage *image = NULL;
   GdkVisual *visual;
   GdkGC *gc = NULL;
   GdkColor tmp_color;
   gint width, height, num_cols, cpp, n, ns, cnt, xcnt, ycnt, wbytes;
   gchar *buffer, pixel_str[32];
   gchar *name_buf;
   _GdkPixmapColor *color = NULL, *fallbackcolor = NULL;
   _GdkPixmapColor *colors = NULL;
   gulong index;
   GHashTable *color_hash = NULL;
   _GdkPixmapInfo *color_info = NULL;

   if ((window == NULL) && (colormap == NULL))
      g_warning("Creating pixmap from xpm with NULL window and colormap");

   if (window == NULL)
      window = gdk_parent_root;

   if (colormap == NULL) {
      colormap = gdk_drawable_get_colormap(window);
      visual = gdk_drawable_get_visual(window);
   } else
      visual = ((GdkColormapPrivate *) colormap)->visual;

   buffer = (*get_buf) (op_header, handle);
   if (buffer == NULL)
      return NULL;

   sscanf(buffer, "%d %d %d %d", &width, &height, &num_cols, &cpp);
   if (cpp >= 32) {
      g_warning("Pixmap has more than 31 characters per color");
      return NULL;
   }

   color_hash = g_hash_table_new(g_str_hash, g_str_equal);

   if (transparent_color == NULL) {
      gdk_color_white(colormap, &tmp_color);
      transparent_color = &tmp_color;
   }

   /* For pseudo-color and grayscale visuals, we have to remember
    * the colors we allocated, so we can free them later.
    */
   if ((visual->type == GDK_VISUAL_PSEUDO_COLOR) ||
       (visual->type == GDK_VISUAL_GRAYSCALE)) {
      color_info = g_malloc(sizeof(_GdkPixmapInfo) +
                            sizeof(gulong) * (num_cols - 1));
      color_info->ncolors = num_cols;
      color_info->colormap = colormap;
      gdk_colormap_ref(colormap);
   }

   name_buf = g_new(gchar, num_cols * (cpp + 1));
   colors = g_new(_GdkPixmapColor, num_cols);

   for (cnt = 0; cnt < num_cols; cnt++) {
      gchar *color_name;

      buffer = (*get_buf) (op_cmap, handle);
      if (buffer == NULL)
         goto error;

      color = &colors[cnt];
      color->color_string = &name_buf[cnt * (cpp + 1)];
      strncpy(color->color_string, buffer, cpp);
      color->color_string[cpp] = 0;
      buffer += strlen(color->color_string);
      color->transparent = FALSE;

      color_name = gdk_pixmap_extract_color(buffer);

      if (color_name == NULL ||
          gdk_color_parse(color_name, &color->color) == FALSE) {
         color->color = *transparent_color;
         color->transparent = TRUE;
      }

      g_free(color_name);

      /* FIXME: The remaining slowness appears to happen in this
         function. */
      gdk_color_alloc(colormap, &color->color);

      if (color_info)
         color_info->pixels[cnt] = color->color.pixel;

      g_hash_table_insert(color_hash, color->color_string, color);
      if (cnt == 0)
         fallbackcolor = color;
   }

   index = 0;
   image = gdk_image_new(GDK_IMAGE_FASTEST, visual, width, height);

   if (mask) {
      /* The pixmap mask is just a bits pattern.
       * Color 0 is used for background and 1 for foreground.
       * We don't care about the colormap, we just need 0 and 1.
       */
      GdkColor mask_pattern;

      *mask = gdk_pixmap_new(window, width, height, 1);
      gc = gdk_gc_new(*mask);

      mask_pattern.pixel = 0;
      gdk_gc_set_foreground(gc, &mask_pattern);
      gdk_draw_rectangle(*mask, gc, TRUE, 0, 0, -1, -1);

      mask_pattern.pixel = 1;
      gdk_gc_set_foreground(gc, &mask_pattern);
   }

   wbytes = width * cpp;
   for (ycnt = 0; ycnt < height; ycnt++) {
      buffer = (*get_buf) (op_body, handle);

      /* FIXME: this slows things down a little - it could be
       * integrated into the strncpy below, perhaps. OTOH, strlen
       * is fast.
       */
      if ((buffer == NULL) || (gint)strlen(buffer) < wbytes)
         continue;

      for (n = 0, cnt = 0, xcnt = 0; n < wbytes; n += cpp, xcnt++) {
         strncpy(pixel_str, &buffer[n], cpp);
         pixel_str[cpp] = 0;
         ns = 0;

         color = g_hash_table_lookup(color_hash, pixel_str);

         if (!color)            /* screwed up XPM file */
            color = fallbackcolor;

         gdk_image_put_pixel(image, xcnt, ycnt, color->color.pixel);

         if (mask && color->transparent) {
            if (cnt < xcnt)
               gdk_draw_line(*mask, gc, cnt, ycnt, xcnt - 1, ycnt);
            cnt = xcnt + 1;
         }
      }

      if (mask && (cnt < xcnt))
         gdk_draw_line(*mask, gc, cnt, ycnt, xcnt - 1, ycnt);
   }

 error:

   if (mask)
      gdk_gc_unref(gc);

   if (image != NULL) {
      pixmap = gdk_pixmap_new(window, width, height, visual->depth);

      if (color_info)
         gdk_drawable_set_data(pixmap, "gdk-xpm", color_info,
                               gdk_xpm_destroy_notify);

      gc = gdk_gc_new(pixmap);
      gdk_gc_set_foreground(gc, transparent_color);
      gdk_draw_image(pixmap, gc, image, 0, 0, 0, 0, image->width,
                     image->height);
      gdk_gc_unref(gc);
      gdk_image_unref(image);
   } else if (color_info)
      gdk_xpm_destroy_notify(color_info);

   if (color_hash != NULL)
      g_hash_table_destroy(color_hash);

   if (colors != NULL)
      g_free(colors);

   if (name_buf != NULL)
      g_free(name_buf);

   return pixmap;
}


struct file_handle {
   FILE *infile;
   gchar *buffer;
   guint buffer_size;
};


static gchar *file_buffer(enum buffer_op op, gpointer handle)
{
   struct file_handle *h = handle;

   switch (op) {
   case op_header:
      if (gdk_pixmap_seek_string(h->infile, "XPM", FALSE) != TRUE)
         break;

      if (gdk_pixmap_seek_char(h->infile, '{') != TRUE)
         break;
      /* Fall through to the next gdk_pixmap_seek_char. */

   case op_cmap:
      gdk_pixmap_seek_char(h->infile, '"');
      fseek(h->infile, -1, SEEK_CUR);
      /* Fall through to the gdk_pixmap_read_string. */

   case op_body:
      gdk_pixmap_read_string(h->infile, &h->buffer, &h->buffer_size);
      return h->buffer;
   }
   return 0;
}

GdkPixmap *gdk_pixmap_colormap_create_from_xpm(GdkWindow * window,
                                               GdkColormap * colormap,
                                               GdkBitmap ** mask,
                                               GdkColor *
                                               transparent_color,
                                               const gchar * filename)
{
   struct file_handle h;
   GdkPixmap *pixmap = NULL;

   memset(&h, 0, sizeof(h));
   h.infile = fopen(filename, "rb");
   if (h.infile != NULL) {
      pixmap = _gdk_pixmap_create_from_xpm(window, colormap, mask,
                                           transparent_color,
                                           file_buffer, &h);
      fclose(h.infile);
      g_free(h.buffer);
   }

   return pixmap;
}

GdkPixmap *gdk_pixmap_create_from_xpm(GdkWindow * window,
                                      GdkBitmap ** mask,
                                      GdkColor * transparent_color,
                                      const gchar * filename)
{
   return gdk_pixmap_colormap_create_from_xpm(window, NULL, mask,
                                              transparent_color, filename);
}

struct mem_handle {
   gchar **data;
   int offset;
};


static gchar *mem_buffer(enum buffer_op op, gpointer handle)
{
   struct mem_handle *h = handle;
   switch (op) {
   case op_header:
   case op_cmap:
   case op_body:
      if (h->data[h->offset])
         return h->data[h->offset++];
   }
   return 0;
}

GdkPixmap *gdk_pixmap_colormap_create_from_xpm_d(GdkWindow * window,
                                                 GdkColormap * colormap,
                                                 GdkBitmap ** mask,
                                                 GdkColor *
                                                 transparent_color,
                                                 gchar ** data)
{
   struct mem_handle h;
   GdkPixmap *pixmap = NULL;

   memset(&h, 0, sizeof(h));
   h.data = data;
   pixmap = _gdk_pixmap_create_from_xpm(window, colormap, mask,
                                        transparent_color, mem_buffer, &h);
   return pixmap;
}

GdkPixmap *gdk_pixmap_create_from_xpm_d(GdkWindow * window,
                                        GdkBitmap ** mask,
                                        GdkColor * transparent_color,
                                        gchar ** data)
{
   return gdk_pixmap_colormap_create_from_xpm_d(window, NULL, mask,
                                                transparent_color, data);
}

GdkPixmap *gdk_pixmap_foreign_new(guint32 anid)
{
   GdkPixmap *pixmap;
   GdkDrawablePrivate *private;
   HBITMAP xpixmap;
   SIZE size;
   unsigned int w_ret, h_ret;

   /* check to make sure we were passed something at
      least a little sane */
   g_return_val_if_fail((anid != 0), NULL);

   /* set the pixmap to the passed in value */
   xpixmap = (HBITMAP) anid;

   /* get information about the BITMAP to fill in the structure for
      the gdk window */
   GetBitmapDimensionEx(xpixmap, &size);
   w_ret = size.cx;
   h_ret = size.cy;

   /* allocate a new gdk pixmap */
   pixmap = gdk_win32_pixmap_alloc();
   private = (GdkDrawablePrivate *) pixmap;

   GDK_DRAWABLE_WIN32DATA(pixmap)->xid = xpixmap;
   private->colormap = NULL;
   private->width = w_ret;
   private->height = h_ret;

   gdk_xid_table_insert(&GDK_DRAWABLE_XID(pixmap), pixmap);

   return pixmap;
}
