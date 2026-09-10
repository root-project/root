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

#ifndef __GDK_PRIVATE_H__
#define __GDK_PRIVATE_H__

#include <gdk/gdktypes.h>
#include <gdk/gdkevents.h>
#include <gdk/gdkfont.h>
#include <gdk/gdkgc.h>
#include <gdk/gdkim.h>
#include <gdk/gdkimage.h>
#include <gdk/gdkregion.h>
#include <gdk/gdkvisual.h>
#include <gdk/gdkwindow.h>

#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */

#define GDK_DRAWABLE_TYPE(d) (((GdkDrawablePrivate *)d)->window_type)
#define GDK_IS_WINDOW(d) (GDK_DRAWABLE_TYPE(d) <= GDK_WINDOW_TEMP || \
                          GDK_DRAWABLE_TYPE(d) == GDK_WINDOW_FOREIGN)
#define GDK_IS_PIXMAP(d) (GDK_DRAWABLE_TYPE(d) == GDK_DRAWABLE_PIXMAP)
#define GDK_DRAWABLE_DESTROYED(d) (((GdkDrawablePrivate *)d)->destroyed)

#define gdk_window_lookup(xid)	   ((GdkWindow*) gdk_xid_table_lookup (xid))
#define gdk_pixmap_lookup(xid)	   ((GdkPixmap*) gdk_xid_table_lookup (xid))
#define gdk_font_lookup(xid)	   ((GdkFont*) gdk_xid_table_lookup (xid))

   typedef struct _GdkDrawablePrivate GdkDrawablePrivate;
/* typedef struct _GdkDrawablePrivate     GdkPixmapPrivate; */
   typedef struct _GdkWindowPrivate GdkWindowPrivate;
   typedef struct _GdkImageClass GdkImageClass;
   typedef struct _GdkImagePrivate GdkImagePrivate;
   typedef struct _GdkGCPrivate GdkGCPrivate;
   typedef struct _GdkColormapPrivate GdkColormapPrivate;
   typedef struct _GdkColorInfo GdkColorInfo;
   typedef struct _GdkFontPrivate GdkFontPrivate;
   typedef struct _GdkEventFilter GdkEventFilter;
   typedef struct _GdkClientFilter GdkClientFilter;

   struct _GdkDrawablePrivate {
      GdkDrawable drawable;
      GdkDrawableClass *klass;
      gpointer klass_data;

      guint8 window_type;
      guint ref_count;

      guint16 width;
      guint16 height;

      GdkColormap *colormap;

      guint destroyed:2;
   };

   struct _GdkWindowPrivate {
      GdkDrawablePrivate drawable;

      GdkWindow *parent;
      gint16 x;
      gint16 y;
      guint8 resize_count;
      guint mapped:1;
      guint guffaw_gravity:1;

      gint extension_events;

      GList *filters;
      GList *children;
   };

   struct _GdkImageClass {
      void (*destroy) (GdkImage * image);
      void (*image_put) (GdkImage * image,
                         GdkDrawable * window,
                         GdkGC * gc,
                         gint xsrc,
                         gint ysrc,
                         gint xdest, gint ydest, gint width, gint height);
   };

   struct _GdkImagePrivate {
      GdkImage image;

      guint ref_count;
      GdkImageClass *klass;
   };

   struct _GdkFontPrivate {
      GdkFont font;
      guint ref_count;
   };

   struct _GdkGCPrivate {
      guint ref_count;
      GdkGCClass *klass;
      gpointer klass_data;
   };

   typedef enum {
      GDK_COLOR_WRITEABLE = 1 << 0
   } GdkColorInfoFlags;

   struct _GdkColorInfo {
      GdkColorInfoFlags flags;
      guint ref_count;
   };

   struct _GdkColormapPrivate {
      GdkColormap colormap;
      GdkVisual *visual;

      guint ref_count;
   };

   struct _GdkEventFilter {
      GdkFilterFunc function;
      gpointer data;
   };

   struct _GdkClientFilter {
      GdkAtom type;
      GdkFilterFunc function;
      gpointer data;
   };

   typedef enum {
      GDK_ARG_STRING,
      GDK_ARG_INT,
      GDK_ARG_BOOL,
      GDK_ARG_NOBOOL,
      GDK_ARG_CALLBACK
   } GdkArgType;


   typedef struct _GdkArgContext GdkArgContext;
   typedef struct _GdkArgDesc GdkArgDesc;

   typedef void (*GdkArgFunc) (const char *name, const char *arg,
                               gpointer data);

   struct _GdkArgContext {
      GPtrArray *tables;
      gpointer cb_data;
   };

   struct _GdkArgDesc {
      const char *name;
      GdkArgType type;
      gpointer location;
      GdkArgFunc callback;
   };


   typedef enum {
      GDK_DEBUG_MISC = 1 << 0,
      GDK_DEBUG_EVENTS = 1 << 1,
      GDK_DEBUG_DND = 1 << 2,
      GDK_DEBUG_COLOR_CONTEXT = 1 << 3,
      GDK_DEBUG_XIM = 1 << 4
   } GdkDebugFlag;

   void gdk_event_button_generate(GdkEvent * event);

/* FIFO's for event queue, and for events put back using
 * gdk_event_put().
 */
   extern GList *gdk_queued_events;
   extern GList *gdk_queued_tail;

   extern GdkEventFunc gdk_event_func;	/* Callback for events */
   extern gpointer gdk_event_data;
   extern GDestroyNotify gdk_event_notify;

   GdkEvent *gdk_event_new(void);

   void gdk_events_init(void);
   void gdk_events_queue(void);
   GdkEvent *gdk_event_unqueue(void);

   GList *gdk_event_queue_find_first(void);
   void gdk_event_queue_remove_link(GList * node);
   void gdk_event_queue_append(GdkEvent * event);

   void gdk_window_init(void);
   void gdk_visual_init(void);
   void gdk_dnd_init(void);

   void gdk_image_init(void);
   void gdk_image_exit(void);

   void gdk_input_init(void);
   void gdk_input_exit(void);

   void gdk_windowing_exit(void);

   void gdk_window_add_colormap_windows(GdkWindow * window);
   void gdk_window_destroy_notify(GdkWindow * window);

/* If you pass x = y = -1, it queries the pointer
   to find out where it currently is.
   If you pass x = y = -2, it does anything necessary
   to know that the drag is ending.
*/
   void gdk_dnd_display_drag_cursor(gint x,
                                    gint y,
                                    gboolean drag_ok,
                                    gboolean change_made);

   extern gint gdk_debug_level;
   extern gboolean gdk_show_events;
   extern gint gdk_screen;
   GDKVAR GdkWindow *gdk_parent_root;
   GDKVAR gint gdk_error_code;
   GDKVAR gint gdk_error_warnings;
   extern GList *gdk_default_filters;

   GdkWindow *_gdk_window_alloc(void);

/* Font/string functions implemented in module-specific code */
   gint _gdk_font_strlen(GdkFont * font, const char *str);
   void _gdk_font_destroy(GdkFont * font);

   void _gdk_colormap_real_destroy(GdkColormap * colormap);

   void _gdk_cursor_destroy(GdkCursor * cursor);

/* Initialization */

   extern GdkArgDesc _gdk_windowing_args[];
   gboolean _gdk_windowing_init_check(int argc, char **argv);

#ifdef USE_XIM
/* XIM support */
   gint gdk_im_open(void);
   void gdk_im_close(void);
   void gdk_ic_cleanup(void);
#endif                          /* USE_XIM */

/* Debugging support */

#ifdef G_ENABLE_DEBUG

#define GDK_NOTE(type,action)		     G_STMT_START { \
    if (gdk_debug_flags & GDK_DEBUG_##type)		    \
       { action; };			     } G_STMT_END

#else                           /* !G_ENABLE_DEBUG */

#define GDK_NOTE(type,action)

#endif                          /* G_ENABLE_DEBUG */

   GDKVAR guint gdk_debug_flags;


#ifdef __cplusplus
}
#endif                          /* __cplusplus */
#endif                          /* __GDK_PRIVATE_H__ */
