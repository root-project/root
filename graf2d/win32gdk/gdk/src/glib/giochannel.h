/* GLIB - Library of useful routines for C programming
 * Copyright (C) 1995-1997  Peter Mattis, Spencer Kimball and Josh MacDonald
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/*
 * Modified by the GLib Team and others 1997-2000.  See the AUTHORS
 * file for a list of people on the GLib Team.  See the ChangeLog
 * files for a list of changes.  These files are distributed with
 * GLib at ftp://ftp.gtk.org/pub/gtk/. 
 */

#ifndef __G_IOCHANNEL_H__
#define __G_IOCHANNEL_H__

#include <gmain.h>
#include <g_types.h>

G_BEGIN_DECLS

/* GIOChannel
 */

typedef struct _GIOChannel	GIOChannel;
typedef struct _GIOFuncs        GIOFuncs;
typedef enum
{
  G_IO_ERROR_NONE,
  G_IO_ERROR_AGAIN,
  G_IO_ERROR_INVAL,
  G_IO_ERROR_UNKNOWN
} GIOError;
typedef enum
{
  G_SEEK_CUR,
  G_SEEK_SET,
  G_SEEK_END
} GSeekType;
typedef enum
{
  G_IO_IN	GLIB_SYSDEF_POLLIN,
  G_IO_OUT	GLIB_SYSDEF_POLLOUT,
  G_IO_PRI	GLIB_SYSDEF_POLLPRI,
  G_IO_ERR	GLIB_SYSDEF_POLLERR,
  G_IO_HUP	GLIB_SYSDEF_POLLHUP,
  G_IO_NVAL	GLIB_SYSDEF_POLLNVAL
} GIOCondition;

struct _GIOChannel
{
  guint channel_flags;
  guint ref_count;
  GIOFuncs *funcs;
};

typedef gboolean (*GIOFunc) (GIOChannel   *source,
			     GIOCondition  condition,
			     gpointer      data);
struct _GIOFuncs
{
  GIOError  (*io_read)         (GIOChannel   *channel, 
				gchar        *buf, 
				guint         count,
				guint        *bytes_read);
  GIOError  (*io_write)        (GIOChannel   *channel, 
				gchar        *buf, 
				guint         count,
				guint        *bytes_written);
  GIOError  (*io_seek)         (GIOChannel   *channel, 
				gint          offset, 
				GSeekType     type);
  void      (*io_close)        (GIOChannel   *channel);
  GSource * (*io_create_watch) (GIOChannel   *channel,
				GIOCondition  condition);
  void      (*io_free)         (GIOChannel   *channel);
};

void        g_io_channel_init   (GIOChannel    *channel);
void        g_io_channel_ref    (GIOChannel    *channel);
void        g_io_channel_unref  (GIOChannel    *channel);
GIOError    g_io_channel_read   (GIOChannel    *channel, 
			         gchar         *buf, 
			         guint          count,
			         guint         *bytes_read);
GIOError  g_io_channel_write    (GIOChannel    *channel, 
			         gchar         *buf, 
			         guint          count,
			         guint         *bytes_written);
GIOError  g_io_channel_seek     (GIOChannel    *channel,
			         gint           offset, 
			         GSeekType      type);
void      g_io_channel_close    (GIOChannel    *channel);
guint     g_io_add_watch_full   (GIOChannel    *channel,
			         gint           priority,
			         GIOCondition   condition,
			         GIOFunc        func,
			         gpointer       user_data,
			         GDestroyNotify notify);
GSource *g_io_create_watch      (GIOChannel   *channel,
			         GIOCondition  condition);
guint    g_io_add_watch         (GIOChannel    *channel,
			         GIOCondition   condition,
			         GIOFunc        func,
			         gpointer       user_data);

/* On Unix, IO channels created with this function for any file
 * descriptor or socket.
 *
 * On Win32, use this only for files opened with the MSVCRT (the
 * Microsoft run-time C library) _open() or _pipe, including file
 * descriptors 0, 1 and 2 (corresponding to stdin, stdout and stderr).
 *
 * The term file descriptor as used in the context of Win32 refers to
 * the emulated Unix-like file descriptors MSVCRT provides. The native
 * corresponding concept is file HANDLE. There isn't as of yet a way to
 * get GIOChannels for file HANDLEs.
 */
GIOChannel* g_io_channel_unix_new    (int         fd);
gint        g_io_channel_unix_get_fd (GIOChannel *channel);

#ifdef G_OS_WIN32

#define G_WIN32_MSG_HANDLE 19981206

/* Use this to get a GPollFD from a GIOChannel, so that you can call
 * g_io_channel_win32_poll(). After calling this you should only use
 * g_io_channel_read() to read from the GIOChannel, i.e. never read()
 * or recv() from the underlying file descriptor or SOCKET.
 */
void        g_io_channel_win32_make_pollfd (GIOChannel   *channel,
					    GIOCondition  condition,
					    GPollFD      *fd);

/* This can be used to wait a until at least one of the channels is readable.
 * On Unix you would do a select() on the file descriptors of the channels.
 * This should probably be available for all platforms?
 */
gint        g_io_channel_win32_poll   (GPollFD    *fds,
				       gint        n_fds,
				       gint        timeout);

/* This is used to add polling for Windows messages. GDK (GTk+) programs
 * should *not* use this.
 */
void        g_main_poll_win32_msg_add (gint        priority,
				       GPollFD    *fd,
				       guint       hwnd);

/* An IO channel for Windows messages for window handle hwnd. */
GIOChannel *g_io_channel_win32_new_messages (guint hwnd);

/* An IO channel for C runtime (emulated Unix-like) file
 * descriptors. Identical to g_io_channel_unix_new above.
 * After calling g_io_add_watch() on a IO channel returned
 * by this function, you shouldn't call read() on the file
 * descriptor.
 */
GIOChannel* g_io_channel_win32_new_fd (int         fd);

/* Get the C runtime file descriptor of a channel. */
gint        g_io_channel_win32_get_fd (GIOChannel *channel);

/* An IO channel for a SOCK_STREAM winsock socket. The parameter
 * should be a SOCKET. After calling g_io_add_watch() on a IO channel
 * returned by this function, you shouldn't call recv() on the SOCKET.
 */
GIOChannel *g_io_channel_win32_new_stream_socket (int socket);

#endif

G_END_DECLS

#endif /* __G_IOCHANNEL_H__ */
