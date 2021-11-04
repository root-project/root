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

#include <glib/gconvert.h>
#include <glib/gmain.h>
#include <glib/gstring.h>

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

#define G_IO_CHANNEL_ERROR g_io_channel_error_quark()

typedef enum
{
  /* Derived from errno */
  G_IO_CHANNEL_ERROR_FBIG,
  G_IO_CHANNEL_ERROR_INVAL,
  G_IO_CHANNEL_ERROR_IO,
  G_IO_CHANNEL_ERROR_ISDIR,
  G_IO_CHANNEL_ERROR_NOSPC,
  G_IO_CHANNEL_ERROR_NXIO,
  G_IO_CHANNEL_ERROR_OVERFLOW,
  G_IO_CHANNEL_ERROR_PIPE,
  /* Other */
  G_IO_CHANNEL_ERROR_FAILED
} GIOChannelError;

typedef enum
{
  G_IO_STATUS_ERROR,
  G_IO_STATUS_NORMAL,
  G_IO_STATUS_EOF,
  G_IO_STATUS_AGAIN
} GIOStatus;

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

typedef enum
{
  G_IO_FLAG_APPEND = 1 << 0,
  G_IO_FLAG_NONBLOCK = 1 << 1,
  G_IO_FLAG_IS_READABLE = 1 << 2,	/* Read only flag */
  G_IO_FLAG_IS_WRITEABLE = 1 << 3,	/* Read only flag */
  G_IO_FLAG_IS_SEEKABLE = 1 << 4,	/* Read only flag */
  G_IO_FLAG_MASK = (1 << 5) - 1,
  G_IO_FLAG_GET_MASK = G_IO_FLAG_MASK,
  G_IO_FLAG_SET_MASK = G_IO_FLAG_APPEND | G_IO_FLAG_NONBLOCK
} GIOFlags;

struct _GIOChannel
{
  /*< private >*/
  guint ref_count;
  GIOFuncs *funcs;

  gchar *encoding;
  GIConv read_cd;
  GIConv write_cd;
  gchar *line_term;		/* String which indicates the end of a line of text */
  guint line_term_len;		/* So we can have null in the line term */

  gsize buf_size;
  GString *read_buf;		/* Raw data from the channel */
  GString *encoded_read_buf;    /* Channel data converted to UTF-8 */
  GString *write_buf;		/* Data ready to be written to the file */
  gchar partial_write_buf[6];	/* UTF-8 partial characters, null terminated */

  /* Group the flags together, immediately after partial_write_buf, to save memory */

  guint use_buffer     : 1;	/* The encoding uses the buffers */
  guint do_encode      : 1;	/* The encoding uses the GIConv coverters */
  guint close_on_unref : 1;	/* Close the channel on final unref */
  guint is_readable    : 1;	/* Cached GIOFlag */
  guint is_writeable   : 1;	/* ditto */
  guint is_seekable    : 1;	/* ditto */
};

typedef gboolean (*GIOFunc) (GIOChannel   *source,
			     GIOCondition  condition,
			     gpointer      data);
struct _GIOFuncs
{
  GIOStatus (*io_read)           (GIOChannel   *channel, 
			          gchar        *buf, 
				  gsize         count,
				  gsize        *bytes_read,
				  GError      **err);
  GIOStatus (*io_write)          (GIOChannel   *channel, 
				  const gchar  *buf, 
				  gsize         count,
				  gsize        *bytes_written,
				  GError      **err);
  GIOStatus (*io_seek)           (GIOChannel   *channel, 
				  glong         offset, 
				  GSeekType     type,
				  GError      **err);
  GIOStatus  (*io_close)         (GIOChannel   *channel,
				  GError      **err);
  GSource*   (*io_create_watch)  (GIOChannel   *channel,
				  GIOCondition  condition);
  void       (*io_free)          (GIOChannel   *channel);
  GIOStatus  (*io_set_flags)     (GIOChannel   *channel,
                                  GIOFlags      flags,
				  GError      **err);
  GIOFlags   (*io_get_flags)     (GIOChannel   *channel);
};

void        g_io_channel_init   (GIOChannel    *channel);
void        g_io_channel_ref    (GIOChannel    *channel);
void        g_io_channel_unref  (GIOChannel    *channel);

#ifndef G_DISABLE_DEPRECATED
GIOError    g_io_channel_read   (GIOChannel    *channel, 
			         gchar         *buf, 
			         gsize          count,
			         gsize         *bytes_read);
GIOError  g_io_channel_write    (GIOChannel    *channel, 
			         const gchar   *buf, 
			         gsize          count,
			         gsize         *bytes_written);
GIOError  g_io_channel_seek     (GIOChannel    *channel,
			         glong          offset, 
			         GSeekType      type);
void      g_io_channel_close    (GIOChannel    *channel);
#endif /* G_DISABLE_DEPRECATED */

GIOStatus g_io_channel_shutdown (GIOChannel      *channel,
				 gboolean         flush,
				 GError         **err);
guint     g_io_add_watch_full   (GIOChannel      *channel,
				 gint             priority,
				 GIOCondition     condition,
				 GIOFunc          func,
				 gpointer         user_data,
				 GDestroyNotify   notify);
GSource * g_io_create_watch     (GIOChannel      *channel,
				 GIOCondition     condition);
guint     g_io_add_watch        (GIOChannel      *channel,
				 GIOCondition     condition,
				 GIOFunc          func,
				 gpointer         user_data);

/* character encoding conversion involved functions.
 */

void                  g_io_channel_set_buffer_size      (GIOChannel   *channel,
							 gsize         size);
gsize                 g_io_channel_get_buffer_size      (GIOChannel   *channel);
GIOCondition          g_io_channel_get_buffer_condition (GIOChannel   *channel);
GIOStatus             g_io_channel_set_flags            (GIOChannel   *channel,
							 GIOFlags      flags,
							 GError      **error);
GIOFlags              g_io_channel_get_flags            (GIOChannel   *channel);
void                  g_io_channel_set_line_term        (GIOChannel   *channel,
							 const gchar  *line_term,
							 gint          length);
G_CONST_RETURN gchar* g_io_channel_get_line_term        (GIOChannel   *channel,
							 gint         *length);
void		      g_io_channel_set_buffered		(GIOChannel   *channel,
							 gboolean      buffered);
gboolean	      g_io_channel_get_buffered		(GIOChannel   *channel);
GIOStatus             g_io_channel_set_encoding         (GIOChannel   *channel,
							 const gchar  *encoding,
							 GError      **error);
G_CONST_RETURN gchar* g_io_channel_get_encoding         (GIOChannel   *channel);
void                  g_io_channel_set_close_on_unref	(GIOChannel   *channel,
							 gboolean      do_close);
gboolean              g_io_channel_get_close_on_unref	(GIOChannel   *channel);


GIOStatus   g_io_channel_flush            (GIOChannel   *channel,
					   GError      **error);
GIOStatus   g_io_channel_read_line        (GIOChannel   *channel,
					   gchar       **str_return,
					   gsize        *length,
					   gsize        *terminator_pos,
					   GError      **error);
GIOStatus   g_io_channel_read_line_string (GIOChannel   *channel,
					   GString      *buffer,
					   gsize        *terminator_pos,
					   GError      **error);
GIOStatus   g_io_channel_read_to_end      (GIOChannel   *channel,
					   gchar       **str_return,
					   gsize        *length,
					   GError      **error);
GIOStatus   g_io_channel_read_chars       (GIOChannel   *channel,
					   gchar        *buf,
					   gsize         count,
					   gsize        *bytes_read,
					   GError      **error);
GIOStatus   g_io_channel_read_unichar     (GIOChannel   *channel,
					   gunichar     *thechar,
					   GError      **error);
GIOStatus   g_io_channel_write_chars      (GIOChannel   *channel,
					   const gchar  *buf,
					   gssize        count,
					   gsize        *bytes_written,
					   GError      **error);
GIOStatus   g_io_channel_write_unichar    (GIOChannel   *channel,
					   gunichar      thechar,
					   GError      **error);
GIOStatus   g_io_channel_seek_position    (GIOChannel   *channel,
					   glong         offset,
					   GSeekType     type,
					   GError      **error);
GIOChannel* g_io_channel_new_file         (const gchar  *filename,
					   const gchar  *mode,
					   GError      **error);

/* Error handling */

GQuark          g_io_channel_error_quark      (void);
GIOChannelError g_io_channel_error_from_errno (gint en);

/* On Unix, IO channels created with this function for any file
 * descriptor or socket.
 *
 * On Win32, this can be used either for files opened with the MSVCRT
 * (the Microsoft run-time C library) _open() or _pipe, including file
 * descriptors 0, 1 and 2 (corresponding to stdin, stdout and stderr),
 * or for Winsock SOCKETs. If the parameter is a legal file
 * descriptor, it is assumed to be such, otherwise it should be a
 * SOCKET. This relies on SOCKETs and file descriptors not
 * overlapping. If you want to be certain, call either
 * g_io_channel_win32_new_fd() or g_io_channel_win32_new_socket()
 * instead as appropriate.
 *
 * The term file descriptor as used in the context of Win32 refers to
 * the emulated Unix-like file descriptors MSVCRT provides. The native
 * corresponding concept is file HANDLE. There isn't as of yet a way to
 * get GIOChannels for Win32 file HANDLEs.
 */
GIOChannel* g_io_channel_unix_new    (int         fd);
gint        g_io_channel_unix_get_fd (GIOChannel *channel);


/* Hook for GClosure / GSource integration. Don't touch */
GLIB_VAR GSourceFuncs g_io_watch_funcs;

#ifdef G_OS_WIN32

#define G_WIN32_MSG_HANDLE 19981206

/* Use this to get a GPollFD from a GIOChannel, so that you can call
 * g_io_channel_win32_poll(). After calling this you should only use
 * g_io_channel_read() to read from the GIOChannel, i.e. never read()
 * from the underlying file descriptor. For SOCKETs, it is possible to call
 * recv().
 */
void        g_io_channel_win32_make_pollfd (GIOChannel   *channel,
					    GIOCondition  condition,
					    GPollFD      *fd);

/* This can be used to wait a until at least one of the channels is readable.
 * On Unix you would do a select() on the file descriptors of the channels.
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

/* Create an IO channel for Windows messages for window handle hwnd. */
GIOChannel *g_io_channel_win32_new_messages (gulong hwnd);

/* Create an IO channel for C runtime (emulated Unix-like) file
 * descriptors. After calling g_io_add_watch() on a IO channel
 * returned by this function, you shouldn't call read() on the file
 * descriptor. This is because adding polling for a file descriptor is
 * implemented on Win32 by starting a thread that sits blocked in a
 * read() from the file descriptor most of the time. All reads from
 * the file descriptor should be done by this internal GLib
 * thread. Your code should call only g_io_channel_read().
 */
GIOChannel* g_io_channel_win32_new_fd (gint         fd);

/* Get the C runtime file descriptor of a channel. */
gint        g_io_channel_win32_get_fd (GIOChannel *channel);

/* Create an IO channel for a winsock socket. The parameter should be
 * a SOCKET. Contrary to IO channels for file descriptors (on *Win32),
 * you can use normal recv() or recvfrom() on sockets even if GLib
 * is polling them.
 */
GIOChannel *g_io_channel_win32_new_socket (gint socket);

#endif

G_END_DECLS

#endif /* __G_IOCHANNEL_H__ */
