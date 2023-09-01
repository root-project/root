/* gmain.h - the GLib Main loop
 * Copyright (C) 1998-2000 Red Hat, Inc.
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

#ifndef __G_MAIN_H__
#define __G_MAIN_H__

#include <glib/gslist.h>
#include <glib/gthread.h>

G_BEGIN_DECLS

typedef struct _GMainContext	        GMainContext;	/* Opaque */
typedef struct _GMainLoop	        GMainLoop;	/* Opaque */
typedef struct _GSource	                GSource;
typedef struct _GSourceCallbackFuncs	GSourceCallbackFuncs;
typedef struct _GSourceFuncs	        GSourceFuncs;

typedef gboolean (*GSourceFunc)       (gpointer data);

struct _GSource
{
  /*< private >*/
  gpointer callback_data;
  GSourceCallbackFuncs *callback_funcs;

  GSourceFuncs *source_funcs;
  guint ref_count;

  GMainContext *context;

  gint priority;
  guint flags;
  guint id;

  GSList *poll_fds;
  
  GSource *prev;
  GSource *next;
};

struct _GSourceCallbackFuncs
{
  void (*ref)   (gpointer     cb_data);
  void (*unref) (gpointer     cb_data);
  void (*get)   (gpointer     cb_data,
		 GSource     *source, 
		 GSourceFunc *func,
		 gpointer    *data);
};

typedef void (*GSourceDummyMarshal) (void);

struct _GSourceFuncs
{
  gboolean (*prepare)  (GSource    *source,
			gint       *timeout);
  gboolean (*check)    (GSource    *source);
  gboolean (*dispatch) (GSource    *source,
			GSourceFunc callback,
			gpointer    user_data);
  void     (*finalize) (GSource    *source); /* Can be NULL */

  /* For use by g_source_set_closure */
  GSourceFunc     closure_callback;	   
  GSourceDummyMarshal closure_marshal; /* Really is of type GClosureMarshal */
};

/* Any definitions using GPollFD or GPollFunc are primarily
 * for Unix and not guaranteed to be the compatible on all
 * operating systems on which GLib runs. Right now, the
 * GLib does use these functions on Win32 as well, but interprets
 * them in a fairly different way than on Unix. If you use
 * these definitions, you are should be prepared to recode
 * for different operating systems.
 *
 *
 * On Win32, the fd in a GPollFD should be Win32 HANDLE (*not* a file
 * descriptor as provided by the C runtime) that can be used by
 * MsgWaitForMultipleObjects. This does *not* include file handles
 * from CreateFile, SOCKETs, nor pipe handles. (But you can use
 * WSAEventSelect to signal events when a SOCKET is readable).
 *
 * On Win32, fd can also be the special value G_WIN32_MSG_HANDLE to
 * indicate polling for messages. These message queue GPollFDs should
 * be added with the g_main_poll_win32_msg_add function.
 *
 * But note that G_WIN32_MSG_HANDLE GPollFDs should not be used by GDK
 * (GTK) programs, as GDK itself wants to read messages and convert them
 * to GDK events.
 *
 * So, unless you really know what you are doing, it's best not to try
 * to use the main loop polling stuff for your own needs on
 * Win32. It's really only written for the GIMP's needs so
 * far.
 */
typedef struct _GPollFD GPollFD;
typedef gint	(*GPollFunc)	(GPollFD *ufds,
				 guint	  nfsd,
				 gint     timeout);

struct _GPollFD
{
  glong		fd;
  gushort 	events;
  gushort 	revents;
};

/* Standard priorities */

#define G_PRIORITY_HIGH            -100
#define G_PRIORITY_DEFAULT          0
#define G_PRIORITY_HIGH_IDLE        100
#define G_PRIORITY_DEFAULT_IDLE     200
#define G_PRIORITY_LOW	            300

/* GMainContext: */

GMainContext *g_main_context_new       (void);
void          g_main_context_ref       (GMainContext *context);
void          g_main_context_unref     (GMainContext *context);
GMainContext *g_main_context_default   (void);

gboolean      g_main_context_iteration (GMainContext *context,
					gboolean      may_block);
gboolean      g_main_context_pending   (GMainContext *context);

/* For implementation of legacy interfaces
 */
GSource      *g_main_context_find_source_by_id              (GMainContext *context,
							     guint         id);
GSource      *g_main_context_find_source_by_user_data       (GMainContext *context,
							     gpointer      user_data);
GSource      *g_main_context_find_source_by_funcs_user_data (GMainContext *context,
 							     GSourceFuncs *funcs,
							     gpointer      user_data);

/* Low level functions for implementing custom main loops.
 */
void     g_main_context_wakeup  (GMainContext *context);
gboolean g_main_context_acquire (GMainContext *context);
void     g_main_context_release (GMainContext *context);
gboolean g_main_context_wait    (GMainContext *context,
				 GCond        *cond,
				 GMutex       *mutex);

gboolean g_main_context_prepare  (GMainContext *context,
				  gint         *priority);
gint     g_main_context_query    (GMainContext *context,
				  gint          max_priority,
				  gint         *timeout,
				  GPollFD      *fds,
				  gint          n_fds);
gint     g_main_context_check    (GMainContext *context,
				  gint          max_priority,
				  GPollFD      *fds,
				  gint          n_fds);
void     g_main_context_dispatch (GMainContext *context);

void      g_main_context_set_poll_func (GMainContext *context,
					GPollFunc     func);
GPollFunc g_main_context_get_poll_func (GMainContext *context);

/* Low level functions for use by source implementations
 */
void g_main_context_add_poll      (GMainContext *context,
				   GPollFD      *fd,
				   gint          priority);
void g_main_context_remove_poll   (GMainContext *context,
				   GPollFD      *fd);

/* GMainLoop: */

GMainLoop *g_main_loop_new        (GMainContext *context,
			    	   gboolean      is_running);
void       g_main_loop_run        (GMainLoop    *loop);
void       g_main_loop_quit       (GMainLoop    *loop);
GMainLoop *g_main_loop_ref        (GMainLoop    *loop);
void       g_main_loop_unref      (GMainLoop    *loop);
gboolean   g_main_loop_is_running (GMainLoop    *loop);
GMainContext *g_main_loop_get_context (GMainLoop    *loop);

/* GSource: */

GSource *g_source_new             (GSourceFuncs   *source_funcs,
				   guint           struct_size);
GSource *g_source_ref             (GSource        *source);
void     g_source_unref           (GSource        *source);

guint    g_source_attach          (GSource        *source,
				   GMainContext   *context);
void     g_source_destroy         (GSource        *source);

void     g_source_set_priority    (GSource        *source,
				   gint            priority);
gint     g_source_get_priority    (GSource        *source);
void     g_source_set_can_recurse (GSource        *source,
				   gboolean        can_recurse);
gboolean g_source_get_can_recurse (GSource        *source);
guint    g_source_get_id          (GSource        *source);

GMainContext *g_source_get_context (GSource       *source);

void g_source_set_callback          (GSource              *source,
				     GSourceFunc           func,
				     gpointer              data,
				     GDestroyNotify        notify);


/* Used to implement g_source_connect_closure and internally*/
void g_source_set_callback_indirect (GSource              *source,
				     gpointer              callback_data,
				     GSourceCallbackFuncs *callback_funcs);

void     g_source_add_poll         (GSource        *source,
				    GPollFD        *fd);
void     g_source_remove_poll      (GSource        *source,
				    GPollFD        *fd);

void     g_source_get_current_time (GSource        *source,
				    GTimeVal       *timeval);

 /* void g_source_connect_closure (GSource        *source,
                                  GClosure       *closure);
 */

/* Specific source types
 */
GSource *g_idle_source_new    (void);
GSource *g_timeout_source_new (guint         interval);

/* Miscellaneous functions
 */
void g_get_current_time		        (GTimeVal	*result);

/* ============== Compat main loop stuff ================== */

#ifndef G_DISABLE_DEPRECATED

/* Legacy names for GMainLoop functions
 */
#define 	g_main_new(is_running)	g_main_loop_new (NULL, is_running);
#define         g_main_run(loop)        g_main_loop_run(loop)
#define         g_main_quit(loop)       g_main_loop_quit(loop)
#define         g_main_destroy(loop)    g_main_loop_unref(loop)
#define         g_main_is_running(loop) g_main_loop_is_running(loop)

/* Functions to manipulate the default main loop
 */

#define	g_main_iteration(may_block) g_main_context_iteration      (NULL, may_block)
#define g_main_pending()            g_main_context_pending        (NULL)

#define g_main_set_poll_func(func)   g_main_context_set_poll_func (NULL, func)

#endif /* G_DISABLE_DEPRECATED */

/* Source manipulation by ID */
gboolean g_source_remove                     (guint          tag);
gboolean g_source_remove_by_user_data        (gpointer       user_data);
gboolean g_source_remove_by_funcs_user_data  (GSourceFuncs  *funcs,
					      gpointer       user_data);

/* Idles and timeouts */
guint		g_timeout_add_full	(gint           priority,
					 guint          interval, 
					 GSourceFunc    function,
					 gpointer       data,
					 GDestroyNotify notify);
guint		g_timeout_add		(guint          interval,
					 GSourceFunc    function,
					 gpointer       data);
guint		g_idle_add	   	(GSourceFunc	function,
					 gpointer	data);
guint	   	g_idle_add_full		(gint   	priority,
					 GSourceFunc	function,
					 gpointer	data,
					 GDestroyNotify notify);
gboolean	g_idle_remove_by_data	(gpointer	data);

/* Hook for GClosure / GSource integration. Don't touch */
GLIB_VAR GSourceFuncs g_timeout_funcs;
GLIB_VAR GSourceFuncs g_idle_funcs;

#ifdef G_OS_WIN32

/* This is used to add polling for Windows messages. GDK (GTK+) programs
 * should *not* use this.
 */
void        g_main_poll_win32_msg_add (gint        priority,
				       GPollFD    *fd,
				       guint       hwnd);
#endif /* G_OS_WIN32 */

G_END_DECLS

#endif /* __G_MAIN_H__ */
