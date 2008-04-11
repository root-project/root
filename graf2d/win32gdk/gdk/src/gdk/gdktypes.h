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

#ifndef __GDK_TYPES_H__
#define __GDK_TYPES_H__

/* GDK uses "glib". (And so does GTK).
 */
#include <glib.h>

#ifdef G_OS_WIN32
#  ifdef GDK_COMPILATION
#    define GDKVAR __declspec(dllexport)
#  else
#    define GDKVAR extern __declspec(dllimport)
#  endif
#else
#  define GDKVAR extern
#endif

/* The system specific file gdkconfig.h contains such configuration
 * settings that are needed not only when compiling GDK (or GTK)
 * itself, but also occasionally when compiling programs that use GDK
 * (or GTK). One such setting is what windowing API backend is in use.
 */
#include <gdkconfig.h>

/* some common magic values */
#define GDK_NONE	     0L
#define GDK_CURRENT_TIME     0L
#define GDK_PARENT_RELATIVE  1L

/* special deviceid for core pointer events */
#define GDK_CORE_POINTER 0xfedc


#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */


/* Type definitions for the basic structures.
 */
   typedef struct _GdkPoint GdkPoint;
   typedef struct _GdkRectangle GdkRectangle;
   typedef struct _GdkSegment GdkSegment;

/*
 * Note that on some platforms the wchar_t type
 * is not the same as GdkWChar. For instance
 * on Win32, wchar_t is unsigned short.
 */
   typedef guint32 GdkWChar;
   typedef gulong GdkAtom;

/* Forward declarations of commonly used types
 */
   typedef struct _GdkColor GdkColor;
   typedef struct _GdkColormap GdkColormap;
   typedef struct _GdkCursor GdkCursor;
   typedef struct _GdkFont GdkFont;
   typedef struct _GdkGC GdkGC;
   typedef struct _GdkImage GdkImage;
   typedef struct _GdkRegion GdkRegion;
   typedef struct _GdkVisual GdkVisual;

   typedef struct _GdkDrawable GdkDrawable;
   typedef struct _GdkDrawable GdkBitmap;
   typedef struct _GdkDrawable GdkPixmap;
   typedef struct _GdkDrawable GdkWindow;

   typedef enum {
      GDK_LSB_FIRST,
      GDK_MSB_FIRST
   } GdkByteOrder;

/* Types of modifiers.
 */
   typedef enum {
      GDK_SHIFT_MASK = 1 << 0,
      GDK_LOCK_MASK = 1 << 1,
      GDK_CONTROL_MASK = 1 << 2,
      GDK_MOD1_MASK = 1 << 3,
      GDK_MOD2_MASK = 1 << 4,
      GDK_MOD3_MASK = 1 << 5,
      GDK_MOD4_MASK = 1 << 6,
      GDK_MOD5_MASK = 1 << 7,
      GDK_BUTTON1_MASK = 1 << 8,
      GDK_BUTTON2_MASK = 1 << 9,
      GDK_BUTTON3_MASK = 1 << 10,
      GDK_BUTTON4_MASK = 1 << 11,
      GDK_BUTTON5_MASK = 1 << 12,
      GDK_RELEASE_MASK = 1 << 13,
      GDK_MODIFIER_MASK = 0x3fff
   } GdkModifierType;

   typedef enum {
      GDK_INPUT_READ = 1 << 0,
      GDK_INPUT_WRITE = 1 << 1,
      GDK_INPUT_EXCEPTION = 1 << 2
   } GdkInputCondition;

   typedef enum {
      GDK_OK = 0,
      GDK_ERROR = -1,
      GDK_ERROR_PARAM = -2,
      GDK_ERROR_FILE = -3,
      GDK_ERROR_MEM = -4
   } GdkStatus;

   typedef void (*GdkInputFunction) (gpointer data,
                                     gint source,
                                     GdkInputCondition condition);

   typedef void (*GdkDestroyNotify) (gpointer data);

   struct _GdkPoint {
      gint16 x;
      gint16 y;
   };

   struct _GdkRectangle {
      gint16 x;
      gint16 y;
      guint16 width;
      guint16 height;
   };

   struct _GdkSegment {
      gint16 x1;
      gint16 y1;
      gint16 x2;
      gint16 y2;
   };


#ifdef __cplusplus
}
#endif                          /* __cplusplus */
#endif                          /* __GDK_TYPES_H__ */
