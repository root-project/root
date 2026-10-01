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

#ifndef __GDK_WIN32_H__
#define __GDK_WIN32_H__

#include <gdk/win32/gdkprivate-win32.h>

#include <time.h>
#include <locale.h>

#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */

#define GDK_ROOT_WINDOW()             ((gulong) HWND_DESKTOP)
#define GDK_ROOT_PARENT()             ((GdkWindow *) gdk_parent_root)
#define GDK_DISPLAY()                 NULL
#define GDK_DRAWABLE_XID(win)         (GDK_DRAWABLE_WIN32DATA(win)->xid)
#define GDK_IMAGE_XIMAGE(image)       (((GdkImagePrivate *) image)->ximage)
#define GDK_COLORMAP_XDISPLAY(cmap)   NULL
#define GDK_COLORMAP_WIN32COLORMAP(cmap)(((GdkColormapPrivateWin32 *) cmap)->xcolormap)
#define GDK_CURSOR_XID(cursor)        (((GdkCursorPrivate*) cursor)->xcursor)
#define GDK_VISUAL_XVISUAL(vis)       (((GdkVisualPrivate *) vis)->xvisual)

#define GDK_WINDOW_XDISPLAY	      GDK_DRAWABLE_XDISPLAY
#define GDK_WINDOW_XWINDOW	      GDK_DRAWABLE_XID
#define GDK_FONT_XFONT(font)          (((GdkWin32SingleFont *)((GdkFontPrivateWin32 *)font)->fonts->data)->xfont)

/* Functions to create GDK pixmaps and windows from their native equivalents */
   GdkPixmap *gdk_pixmap_foreign_new(gulong anid);
   GdkWindow *gdk_window_foreign_new(gulong anid);

/* Return a device context to draw in a drawable, given a GDK GC,
 * and a mask indicating which GC values might be used (for efficiency,
 * no need to muck around with text-related stuff if we aren't going
 * to output text, for instance).
 */
   HDC gdk_win32_hdc_get(GdkDrawable * drawable,
                         GdkGC * gc, GdkGCValuesMask usage);


/* Each HDC returned from gdk_win32_hdc_get must be released with
 * this function
 */
   void gdk_win32_hdc_release(GdkDrawable * drawable,
                              GdkGC * gc, GdkGCValuesMask usage);

#ifdef __cplusplus
}
#endif                          /* __cplusplus */
#endif                          /* __GDK_WIN32_H__ */
