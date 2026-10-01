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

#ifndef __GDK_H__
#define __GDK_H__

#include <gdk/gdkcc.h>
#include <gdk/gdkcolor.h>
#include <gdk/gdkcursor.h>
#include <gdk/gdkdnd.h>
#include <gdk/gdkdrawable.h>
#include <gdk/gdkevents.h>
#include <gdk/gdkfont.h>
#include <gdk/gdkgc.h>
#include <gdk/gdkim.h>
#include <gdk/gdkimage.h>
#include <gdk/gdkinput.h>
#include <gdk/gdkpixmap.h>
#include <gdk/gdkproperty.h>
#include <gdk/gdkregion.h>
#include <gdk/gdkrgb.h>
#include <gdk/gdkselection.h>
#include <gdk/gdktypes.h>
#include <gdk/gdkvisual.h>
#include <gdk/gdkwindow.h>

#include <gdk/gdkcompat.h>

#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */


/* Initialization, exit and events
 */
#define	  GDK_PRIORITY_EVENTS		(G_PRIORITY_DEFAULT)
   void gdk_init(gint * argc, gchar *** argv);
   gboolean gdk_init_check(gint * argc, gchar *** argv);
   void gdk_exit(gint error_code);
   gchar *gdk_set_locale(void);

/* Push and pop error handlers for X errors
 */
   void gdk_error_trap_push(void);
   gint gdk_error_trap_pop(void);


   void gdk_set_use_xshm(gboolean use_xshm);

   gboolean gdk_get_use_xshm(void);
   gchar *gdk_get_display(void);

   gint gdk_input_add_full(gint source,
                           GdkInputCondition condition,
                           GdkInputFunction function,
                           gpointer data, GdkDestroyNotify destroy);
   gint gdk_input_add(gint source,
                      GdkInputCondition condition,
                      GdkInputFunction function, gpointer data);
   void gdk_input_remove(gint tag);

   gint gdk_pointer_grab(GdkWindow * window,
                         gboolean owner_events,
                         GdkEventMask event_mask,
                         GdkWindow * confine_to,
                         GdkCursor * cursor, guint32 time);
   void gdk_pointer_ungrab(guint32 time);
   gint gdk_keyboard_grab(GdkWindow * window,
                          gboolean owner_events, guint32 time);
   void gdk_keyboard_ungrab(guint32 time);
   gboolean gdk_pointer_is_grabbed(void);
   gint gdk_button_grab(gint button, gint mod, GdkWindow * window,
                        gboolean owner_events,
                        GdkEventMask event_mask,
                        GdkWindow * confine_to,
                        GdkCursor * cursor);
   void gdk_button_ungrab(gint button, gint mod, GdkWindow * window);
   gint gdk_key_grab(gint keycode, gint mod, GdkWindow * window);
   void gdk_key_ungrab(gint keycode, gint mod, GdkWindow * window);

   gint gdk_screen_width(void);
   gint gdk_screen_height(void);

   gint gdk_screen_width_mm(void);
   gint gdk_screen_height_mm(void);

   void gdk_flush(void);
   void gdk_beep(void);

   void gdk_key_repeat_disable(void);
   void gdk_key_repeat_restore(void);

/* Rectangle utilities
 */
   gboolean gdk_rectangle_intersect(GdkRectangle * src1,
                                    GdkRectangle * src2,
                                    GdkRectangle * dest);
   void gdk_rectangle_union(GdkRectangle * src1,
                            GdkRectangle * src2, GdkRectangle * dest);

/* Conversion functions between wide char and multibyte strings. 
 */
   gchar *gdk_wcstombs(const GdkWChar * src);
   gint gdk_mbstowcs(GdkWChar * dest, const gchar * src, gint dest_max);

/* Miscellaneous */
   void gdk_event_send_clientmessage_toall(GdkEvent * event);
   gboolean gdk_event_send_client_message(GdkEvent * event, guint32 xid);

/* Key values
 */
   gchar *gdk_keyval_name(guint keyval);
   guint gdk_keyval_from_name(const gchar * keyval_name);
   void gdk_keyval_convert_case(guint symbol,
                                guint * lower, guint * upper);
   guint gdk_keyval_to_upper(guint keyval);
   guint gdk_keyval_to_lower(guint keyval);
   gboolean gdk_keyval_is_upper(guint keyval);
   gboolean gdk_keyval_is_lower(guint keyval);


/* Threading
 */

   GDKVAR GMutex *gdk_threads_mutex;

   void gdk_threads_enter(void);
   void gdk_threads_leave(void);

#ifdef	G_THREADS_ENABLED
#  define GDK_THREADS_ENTER()	G_STMT_START {	\
      if (gdk_threads_mutex)                 	\
        g_mutex_lock (gdk_threads_mutex);   	\
   } G_STMT_END
#  define GDK_THREADS_LEAVE()	G_STMT_START { 	\
      if (gdk_threads_mutex)                 	\
        g_mutex_unlock (gdk_threads_mutex); 	\
   } G_STMT_END
#else                           /* !G_THREADS_ENABLED */
#  define GDK_THREADS_ENTER()
#  define GDK_THREADS_LEAVE()
#endif                          /* !G_THREADS_ENABLED */

#ifdef __cplusplus
}
#endif                          /* __cplusplus */
#endif                          /* __GDK_H__ */
