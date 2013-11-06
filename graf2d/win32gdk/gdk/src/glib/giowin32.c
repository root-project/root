/* GLIB - Library of useful routines for C programming
 * Copyright (C) 1995-1997  Peter Mattis, Spencer Kimball and Josh MacDonald
 *
 * giowin32.c: IO Channels for Win32.
 * Copyright 1998 Owen Taylor and Tor Lillqvist
 * Copyright 1999-2000 Tor Lillqvist and Craig Setera
 * Copyright 2001 Andrew Lanoix
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

/* Define this to get (very) verbose logging of all channels */
/* #define G_IO_WIN32_DEBUG */

#include "glib.h"

#include <stdlib.h>
#include <windows.h>
#include <winsock.h>          /* Not everybody has winsock2 */
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <errno.h>
#include <sys/stat.h>

#include "glibintl.h"

typedef struct _GIOWin32Channel GIOWin32Channel;
typedef struct _GIOWin32Watch GIOWin32Watch;

#define BUFFER_SIZE 4096

typedef enum {
  G_IO_WIN32_WINDOWS_MESSAGES,	/* Windows messages */
  G_IO_WIN32_FILE_DESC,		/* Unix-like file descriptors from
				 * _open() or _pipe(). Read with read().
				 * Have to create separate thread to read.
				 */
  G_IO_WIN32_SOCKET		/* Sockets. A separate thread is blocked
				 * in select() most of the time.
				 */
} GIOWin32ChannelType;

struct _GIOWin32Channel {
  GIOChannel channel;
  gint fd;			/* Either a Unix-like file handle as provided
				 * by the Microsoft C runtime, or a SOCKET
				 * as provided by WinSock.
				 */
  GIOWin32ChannelType type;
  
  gboolean debug;

  CRITICAL_SECTION mutex;

  /* This is used by G_IO_WIN32_WINDOWS_MESSAGES channels */
  HWND hwnd;			/* handle of window, or NULL */
  
  /* Following fields are used by both fd and socket channels. */
  gboolean running;		/* Is reader thread running. FALSE if
				 * EOF has been reached.
				 */
  gboolean needs_close;		/* If the channel has been closed while
				 * the reader thread was still running.
				 */
  guint thread_id;		/* If non-NULL has a reader thread, or has
				 * had.*/
  HANDLE thread_handle;
  HANDLE data_avail_event;

  gushort revents;

  /* Following fields used by fd channels for input */
  
  /* Data is kept in a circular buffer. To be able to distinguish between
   * empty and full buffer, we cannot fill it completely, but have to
   * leave a one character gap.
   *
   * Data available is between indexes rdp and wrp-1 (modulo BUFFER_SIZE).
   *
   * Empty:    wrp == rdp
   * Full:     (wrp + 1) % BUFFER_SIZE == rdp
   * Partial:  otherwise
   */
  guchar *buffer;		/* (Circular) buffer */
  gint wrp, rdp;		/* Buffer indices for writing and reading */
  HANDLE space_avail_event;

  /* Following fields used by socket channels */
  GSList *watches;
  HANDLE data_avail_noticed_event;
};

#define LOCK(mutex) EnterCriticalSection (&mutex)
#define UNLOCK(mutex) LeaveCriticalSection (&mutex)

struct _GIOWin32Watch {
  GSource       source;
  GPollFD       pollfd;
  GIOChannel   *channel;
  GIOCondition  condition;
  GIOFunc       callback;
};

static void
g_io_channel_win32_init (GIOWin32Channel *channel)
{
#ifdef G_IO_WIN32_DEBUG
  channel->debug = TRUE;
#else
  if (getenv ("G_IO_WIN32_DEBUG") != NULL)
    channel->debug = TRUE;
  else
    channel->debug = FALSE;
#endif
  channel->buffer = NULL;
  channel->running = FALSE;
  channel->needs_close = FALSE;
  channel->thread_id = 0;
  channel->data_avail_event = NULL;
  channel->revents = 0;
  channel->space_avail_event = NULL;
  channel->data_avail_noticed_event = NULL;
  channel->watches = NULL;
  InitializeCriticalSection (&channel->mutex);
}

static void
create_events (GIOWin32Channel *channel)
{
  SECURITY_ATTRIBUTES sec_attrs;
  
  sec_attrs.nLength = sizeof(SECURITY_ATTRIBUTES);
  sec_attrs.lpSecurityDescriptor = NULL;
  sec_attrs.bInheritHandle = FALSE;

  /* The data available event is manual reset, the space available event
   * is automatic reset.
   */
  if (!(channel->data_avail_event = CreateEvent (&sec_attrs, TRUE, FALSE, NULL))
      || !(channel->space_avail_event = CreateEvent (&sec_attrs, FALSE, FALSE, NULL))
      || !(channel->data_avail_noticed_event = CreateEvent (&sec_attrs, FALSE, FALSE, NULL)))
    {
      gchar *msg = g_win32_error_message (GetLastError ());
      g_error ("Error creating event: %s", msg);
    }
}

static unsigned __stdcall
read_thread (void *parameter)
{
  GIOWin32Channel *channel = parameter;
  guchar *buffer;
  guint nbytes;

  g_io_channel_ref ((GIOChannel *)channel);

  if (channel->debug)
    g_print ("read_thread %#x: start fd:%d, data_avail:%#x, space_avail:%#x\n",
	     channel->thread_id,
	     channel->fd,
	     (guint) channel->data_avail_event,
	     (guint) channel->space_avail_event);
  
  channel->buffer = g_malloc (BUFFER_SIZE);
  channel->rdp = channel->wrp = 0;
  channel->running = TRUE;

  SetEvent (channel->space_avail_event);
  
  while (channel->running)
    {
      LOCK (channel->mutex);
      if (channel->debug)
	g_print ("read_thread %#x: rdp=%d, wrp=%d\n",
		 channel->thread_id, channel->rdp, channel->wrp);
      if ((channel->wrp + 1) % BUFFER_SIZE == channel->rdp)
	{
	  /* Buffer is full */
	  if (channel->debug)
	    g_print ("read_thread %#x: resetting space_avail\n",
		     channel->thread_id);
	  ResetEvent (channel->space_avail_event);
	  if (channel->debug)
	    g_print ("read_thread %#x: waiting for space\n",
		     channel->thread_id);
	  UNLOCK (channel->mutex);
	  WaitForSingleObject (channel->space_avail_event, INFINITE);
	  LOCK (channel->mutex);
	  if (channel->debug)
	    g_print ("read_thread %#x: rdp=%d, wrp=%d\n",
		     channel->thread_id, channel->rdp, channel->wrp);
	}
      
      buffer = channel->buffer + channel->wrp;
      
      /* Always leave at least one byte unused gap to be able to
       * distinguish between the full and empty condition...
       */
      nbytes = MIN ((channel->rdp + BUFFER_SIZE - channel->wrp - 1) % BUFFER_SIZE,
		    BUFFER_SIZE - channel->wrp);

      if (channel->debug)
	g_print ("read_thread %#x: calling read() for %d bytes\n",
		 channel->thread_id, nbytes);

      UNLOCK (channel->mutex);

      nbytes = read (channel->fd, buffer, nbytes);
      
      LOCK (channel->mutex);

      channel->revents = G_IO_IN;
      if (nbytes == 0)
	channel->revents |= G_IO_HUP;
      else if (nbytes < 0)
	channel->revents |= G_IO_ERR;

      if (channel->debug)
	g_print ("read_thread %#x: read() returned %d, rdp=%d, wrp=%d\n",
		 channel->thread_id, nbytes, channel->rdp, channel->wrp);

      if (nbytes <= 0)
	break;

      channel->wrp = (channel->wrp + nbytes) % BUFFER_SIZE;
      if (channel->debug)
	g_print ("read_thread %#x: rdp=%d, wrp=%d, setting data_avail\n",
		 channel->thread_id, channel->rdp, channel->wrp);
      SetEvent (channel->data_avail_event);
      UNLOCK (channel->mutex);
    }
  
  channel->running = FALSE;
  if (channel->needs_close)
    {
      if (channel->debug)
	g_print ("read_thread %#x: channel fd %d needs closing\n",
		 channel->thread_id, channel->fd);
      close (channel->fd);
      channel->fd = -1;
    }

  if (channel->debug)
    g_print ("read_thread %#x: EOF, rdp=%d, wrp=%d, setting data_avail\n",
	     channel->thread_id, channel->rdp, channel->wrp);
  SetEvent (channel->data_avail_event);
  UNLOCK (channel->mutex);
  
  g_io_channel_unref((GIOChannel *)channel);
  
  /* No need to call _endthreadex(), the actual thread starter routine
   * in MSVCRT (see crt/src/threadex.c:_threadstartex) calls
   * _endthreadex() for us.
   */

  CloseHandle (channel->thread_handle);

  return 0;
}

static void
create_thread (GIOWin32Channel     *channel,
	       GIOCondition         condition,
	       unsigned (__stdcall *thread) (void *parameter))
{
  channel->thread_handle =
    (HANDLE) _beginthreadex (NULL, 0, thread, channel, 0,
			     &channel->thread_id);
  if (channel->thread_handle == 0)
    g_warning (G_STRLOC ": Error creating reader thread: %s",
	       strerror (errno));
  WaitForSingleObject (channel->space_avail_event, INFINITE);
}

static GIOStatus
buffer_read (GIOWin32Channel *channel,
	     guchar          *dest,
	     gsize            count,
	     gsize           *bytes_read,
	     GError         **err)
{
  guint nbytes;
  guint left = count;
  
  LOCK (channel->mutex);
  if (channel->debug)
    g_print ("reading from thread %#x %d bytes, rdp=%d, wrp=%d\n",
	     channel->thread_id, count, channel->rdp, channel->wrp);
  
  if (channel->wrp == channel->rdp)
    {
      UNLOCK (channel->mutex);
      if (channel->debug)
	g_print ("waiting for data from thread %#x\n", channel->thread_id);
      WaitForSingleObject (channel->data_avail_event, INFINITE);
      if (channel->debug)
	g_print ("done waiting for data from thread %#x\n", channel->thread_id);
      LOCK (channel->mutex);
      if (channel->wrp == channel->rdp && !channel->running)
	{
	  UNLOCK (channel->mutex);
          *bytes_read = 0;
	  return G_IO_STATUS_NORMAL; /* as before, normal case ? */
	}
    }
  
  if (channel->rdp < channel->wrp)
    nbytes = channel->wrp - channel->rdp;
  else
    nbytes = BUFFER_SIZE - channel->rdp;
  UNLOCK (channel->mutex);
  nbytes = MIN (left, nbytes);
  if (channel->debug)
    g_print ("moving %d bytes from thread %#x\n",
	     nbytes, channel->thread_id);
  memcpy (dest, channel->buffer + channel->rdp, nbytes);
  dest += nbytes;
  left -= nbytes;
  LOCK (channel->mutex);
  channel->rdp = (channel->rdp + nbytes) % BUFFER_SIZE;
  if (channel->debug)
    g_print ("setting space_avail for thread %#x\n", channel->thread_id);
  SetEvent (channel->space_avail_event);
  if (channel->debug)
    g_print ("for thread %#x: rdp=%d, wrp=%d\n",
	     channel->thread_id, channel->rdp, channel->wrp);
  if (channel->running && channel->wrp == channel->rdp)
    {
      if (channel->debug)
	g_print ("resetting data_avail of thread %#x\n",
		 channel->thread_id);
      ResetEvent (channel->data_avail_event);
    };
  UNLOCK (channel->mutex);
  
  /* We have no way to indicate any errors form the actual
   * read() or recv() call in the reader thread. Should we have?
   */
  *bytes_read = count - left;
  return (*bytes_read > 0) ? G_IO_STATUS_NORMAL : G_IO_STATUS_EOF;
}

static unsigned __stdcall
select_thread (void *parameter)
{
  GIOWin32Channel *channel = parameter;
  fd_set read_fds, write_fds, except_fds;
  GSList *tmp;
  int n;

  g_io_channel_ref ((GIOChannel *)channel);

  if (channel->debug)
    g_print ("select_thread %#x: start fd:%d,\n\tdata_avail:%#x, data_avail_noticed:%#x\n",
	     channel->thread_id,
	     channel->fd,
	     (guint) channel->data_avail_event,
	     (guint) channel->data_avail_noticed_event);
  
  channel->rdp = channel->wrp = 0;
  channel->running = TRUE;

  SetEvent (channel->space_avail_event);
  
  while (channel->running)
    {
      FD_ZERO (&read_fds);
      FD_ZERO (&write_fds);
      FD_ZERO (&except_fds);

      tmp = channel->watches;
      while (tmp)
	{
	  GIOWin32Watch *watch = (GIOWin32Watch *)tmp->data;

	  if (watch->condition & (G_IO_IN | G_IO_HUP))
	    FD_SET (channel->fd, &read_fds);
	  if (watch->condition & G_IO_OUT)
	    FD_SET (channel->fd, &write_fds);
	  if (watch->condition & G_IO_ERR)
	    FD_SET (channel->fd, &except_fds);
	  
	  tmp = tmp->next;
	}
      if (channel->debug)
	g_print ("select_thread %#x: calling select() for%s%s%s\n",
		 channel->thread_id,
		 (FD_ISSET (channel->fd, &read_fds) ? " IN" : ""),
		 (FD_ISSET (channel->fd, &write_fds) ? " OUT" : ""),
		 (FD_ISSET (channel->fd, &except_fds) ? " ERR" : ""));

      n = select (1, &read_fds, &write_fds, &except_fds, NULL);
      
      if (n == SOCKET_ERROR)
	{
	  if (channel->debug)
	    g_print ("select_thread %#x: select returned SOCKET_ERROR\n",
		     channel->thread_id);
	  break;
	}

      if (channel->debug)
	g_print ("select_thread %#x: got%s%s%s\n",
		 channel->thread_id,
		 (FD_ISSET (channel->fd, &read_fds) ? " IN" : ""),
		 (FD_ISSET (channel->fd, &write_fds) ? " OUT" : ""),
		 (FD_ISSET (channel->fd, &except_fds) ? " ERR" : ""));

      if (FD_ISSET (channel->fd, &read_fds))
	channel->revents |= G_IO_IN;
      if (FD_ISSET (channel->fd, &write_fds))
	channel->revents |= G_IO_OUT;
      if (FD_ISSET (channel->fd, &except_fds))
	channel->revents |= G_IO_ERR;

      if (channel->debug)
	g_print ("select_thread %#x: resetting data_avail_noticed,\n"
		 "\tsetting data_avail\n",
		 channel->thread_id);
      ResetEvent (channel->data_avail_noticed_event);
      SetEvent (channel->data_avail_event);

      LOCK (channel->mutex);
      if (channel->needs_close)
	{
	  UNLOCK (channel->mutex);
	  break;
	}
      UNLOCK (channel->mutex);

      if (channel->debug)
	g_print ("select_thread %#x: waiting for data_avail_noticed\n",
		 channel->thread_id);

      WaitForSingleObject (channel->data_avail_noticed_event, INFINITE);
      if (channel->debug)
	g_print ("select_thread %#x: got data_avail_noticed\n",
		 channel->thread_id);
    }
  
  channel->running = FALSE;
  LOCK (channel->mutex);
  if (channel->fd != -1)
    {
      /* DO NOT close the fd here */
      channel->fd = -1;
    }

  if (channel->debug)
    g_print ("select_thread %#x: got error, setting data_avail\n",
	     channel->thread_id);
  SetEvent (channel->data_avail_event);
  UNLOCK (channel->mutex);
  
  g_io_channel_unref((GIOChannel *)channel);
  
  /* No need to call _endthreadex(), the actual thread starter routine
   * in MSVCRT (see crt/src/threadex.c:_threadstartex) calls
   * _endthreadex() for us.
   */

  CloseHandle (channel->thread_handle);

  return 0;
}

static gboolean
g_io_win32_prepare (GSource *source,
		    gint    *timeout)
{
  GIOWin32Watch *watch = (GIOWin32Watch *)source;
  GIOWin32Channel *channel = (GIOWin32Channel *)watch->channel;
  
  *timeout = -1;
  
  if (channel->type == G_IO_WIN32_FILE_DESC)
    {
      LOCK (channel->mutex);
      if (channel->running && channel->wrp == channel->rdp)
	channel->revents = 0;
      UNLOCK (channel->mutex);
    }
  else if (channel->type == G_IO_WIN32_SOCKET)
    {
      channel->revents = 0;

      if (channel->debug)
	g_print ("g_io_win32_prepare: thread %#x, setting data_avail_noticed\n",
		 channel->thread_id);
      SetEvent (channel->data_avail_noticed_event);
      if (channel->debug)
	g_print ("g_io_win32_prepare: thread %#x, there.\n",
		 channel->thread_id);
    }

  return FALSE;
  /* XXX: why should we want to do this ? */
  watch->condition = g_io_channel_get_buffer_condition (watch->channel);

  return (watch->pollfd.revents & (G_IO_IN | G_IO_OUT)) == watch->condition;
}

static gboolean
g_io_win32_check (GSource *source)
{
	MSG msg;
  GIOWin32Watch *watch = (GIOWin32Watch *)source;
  GIOWin32Channel *channel = (GIOWin32Channel *)watch->channel;
  GIOCondition buffer_condition = g_io_channel_get_buffer_condition (watch->channel);

  
  if (channel->debug)
    g_print ("g_io_win32_check: for thread %#x:\n"
	     "\twatch->pollfd.events:%#x, watch->pollfd.revents:%#x, channel->revents:%#x\n",
	     channel->thread_id,
	     watch->pollfd.events, watch->pollfd.revents, channel->revents);

  if (channel->type != G_IO_WIN32_WINDOWS_MESSAGES)
	{
     watch->pollfd.revents = (watch->pollfd.events & channel->revents);
	}
	else
	{
    return (PeekMessage (&msg, channel->hwnd, 0, 0, PM_NOREMOVE));
	}

  if (channel->type == G_IO_WIN32_SOCKET)
    {
      if (channel->debug)
	g_print ("g_io_win32_check: thread %#x, resetting data_avail\n",
		 channel->thread_id);
      ResetEvent (channel->data_avail_event);
      if (channel->debug)
	g_print ("g_io_win32_check: thread %#x, there.\n",
		 channel->thread_id);
    }

  return (watch->pollfd.revents & watch->condition);
}

static gboolean
g_io_win32_dispatch (GSource     *source,
		     GSourceFunc  callback,
		     gpointer     user_data)
{
  GIOFunc func = (GIOFunc)callback;
  GIOWin32Watch *watch = (GIOWin32Watch *)source;
  
  if (!func)
    {
      g_warning (G_STRLOC ": GIOWin32Watch dispatched without callback\n"
		 "You must call g_source_connect().");
      return FALSE;
    }
  
  return (*func) (watch->channel,
		  watch->pollfd.revents & watch->condition,
		  user_data);
}

static void
g_io_win32_finalize (GSource *source)
{
  GIOWin32Watch *watch = (GIOWin32Watch *)source;
  GIOWin32Channel *channel = (GIOWin32Channel *)watch->channel;
  
  if (channel->debug)
    g_print ("g_io_win32_finalize: channel with thread %#x\n",
	     channel->thread_id);

  channel->watches = g_slist_remove (channel->watches, watch);

  SetEvent (channel->data_avail_noticed_event);
  g_io_channel_unref (watch->channel);
}

#if defined(G_PLATFORM_WIN32) && defined(__GNUC__)
__declspec(dllexport)
#endif
GSourceFuncs g_io_watch_funcs = {
  g_io_win32_prepare,
  g_io_win32_check,
  g_io_win32_dispatch,
  g_io_win32_finalize
};

static GSource *
g_io_win32_create_watch (GIOChannel    *channel,
			 GIOCondition   condition,
			 unsigned (__stdcall *thread) (void *parameter))
{
  GIOWin32Channel *win32_channel = (GIOWin32Channel *)channel;
  GIOWin32Watch *watch;
  GSource *source;

  source = g_source_new (&g_io_watch_funcs, sizeof (GIOWin32Watch));
  watch = (GIOWin32Watch *)source;
  
  watch->channel = channel;
  g_io_channel_ref (channel);
  
  watch->condition = condition;
  
  if (win32_channel->data_avail_event == NULL)
    create_events (win32_channel);

  watch->pollfd.fd = (gint) win32_channel->data_avail_event;
  watch->pollfd.events = condition;
  
  if (win32_channel->debug)
    g_print ("g_io_win32_create_watch: fd:%d condition:%#x handle:%#x\n",
	     win32_channel->fd, condition, watch->pollfd.fd);
  
  win32_channel->watches = g_slist_append (win32_channel->watches, watch);

  if (win32_channel->thread_id == 0)
    create_thread (win32_channel, condition, thread);

  g_source_add_poll (source, &watch->pollfd);
  
  return source;
}

static GIOStatus
g_io_win32_msg_read (GIOChannel *channel,
		     gchar      *buf,
		     gsize       count,
		     gsize      *bytes_read,
		     GError    **err)
{
  GIOWin32Channel *win32_channel = (GIOWin32Channel *)channel;
  MSG msg;               /* In case of alignment problems */
  
  if (count < sizeof (MSG))
    {
      g_set_error(err, G_IO_CHANNEL_ERROR, G_IO_CHANNEL_ERROR_INVAL,
        _("Incorrect message size")); /* Correct error message? FIXME */
      return G_IO_STATUS_ERROR;
    }
  
  if (win32_channel->debug)
    g_print ("g_io_win32_msg_read: for %#x\n",
	     win32_channel->hwnd);
  if (!PeekMessage (&msg, win32_channel->hwnd, 0, 0, PM_REMOVE))
    return G_IO_STATUS_AGAIN;

  memmove (buf, &msg, sizeof (MSG));
  *bytes_read = sizeof (MSG);

  return (*bytes_read > 0) ? G_IO_STATUS_NORMAL : G_IO_STATUS_EOF;
}

static GIOStatus
g_io_win32_msg_write (GIOChannel  *channel,
		      const gchar *buf,
		      gsize        count,
		      gsize       *bytes_written,
		      GError     **err)
{
  GIOWin32Channel *win32_channel = (GIOWin32Channel *)channel;
  MSG msg;
  
  if (count != sizeof (MSG))
    {
      g_set_error(err, G_IO_CHANNEL_ERROR, G_IO_CHANNEL_ERROR_INVAL,
        _("Incorrect message size")); /* Correct error message? FIXME */
      return G_IO_STATUS_ERROR;
    }
  
  /* In case of alignment problems */
  memmove (&msg, buf, sizeof (MSG));
  if (!PostMessage (win32_channel->hwnd, msg.message, msg.wParam, msg.lParam))
    {
      g_set_error(err, G_IO_CHANNEL_ERROR, G_IO_CHANNEL_ERROR_FAILED,
        _("Unknown error")); /* Correct error message? FIXME */
      return G_IO_STATUS_ERROR;
    }

  *bytes_written = sizeof (MSG);

  return G_IO_STATUS_NORMAL;
}

static GIOStatus
g_io_win32_no_seek (GIOChannel *channel,
		    glong       offset,
		    GSeekType   type,
		    GError     **err)
{
  g_assert_not_reached ();

  return G_IO_STATUS_ERROR;
}

static GIOStatus
g_io_win32_msg_close (GIOChannel *channel,
		      GError    **err)
{
  /* Nothing to be done. Or should we set hwnd to some invalid value? */

  return G_IO_STATUS_NORMAL;
}

static void
g_io_win32_free (GIOChannel *channel)
{
  GIOWin32Channel *win32_channel = (GIOWin32Channel *)channel;
  
  if (win32_channel->debug)
    g_print ("thread %#x: freeing channel, fd: %d\n",
	     win32_channel->thread_id,
	     win32_channel->fd);

  if (win32_channel->data_avail_event)
    CloseHandle (win32_channel->data_avail_event);
  if (win32_channel->space_avail_event)
    CloseHandle (win32_channel->space_avail_event);
  if (win32_channel->data_avail_noticed_event)
    CloseHandle (win32_channel->data_avail_noticed_event);
  DeleteCriticalSection (&win32_channel->mutex);

  g_free (win32_channel->buffer);
  g_slist_free (win32_channel->watches);
  g_free (win32_channel);
}

static GSource *
g_io_win32_msg_create_watch (GIOChannel    *channel,
			     GIOCondition   condition)
{
  GIOWin32Watch *watch;
  GSource *source;

  source = g_source_new (&g_io_watch_funcs, sizeof (GIOWin32Watch));
  watch = (GIOWin32Watch *)source;
  
  watch->channel = channel;
  g_io_channel_ref (channel);
  
  watch->condition = condition;
  
  watch->pollfd.fd = G_WIN32_MSG_HANDLE;
  watch->pollfd.events = condition;
  
  g_source_add_poll (source, &watch->pollfd);
  
  return source;
}

static GIOStatus
g_io_win32_fd_read (GIOChannel *channel,
		    gchar      *buf,
		    gsize       count,
		    gsize      *bytes_read,
		    GError    **err)
{
  GIOWin32Channel *win32_channel = (GIOWin32Channel *)channel;
  gint result;
  
  if (win32_channel->debug)
    g_print ("g_io_win32_fd_read: fd:%d count:%d\n",
	     win32_channel->fd, count);
  
  if (win32_channel->thread_id)
    {
      return buffer_read (win32_channel, buf, count, bytes_read, err);
    }

  result = read (win32_channel->fd, buf, count);

  if (result < 0)
    {
      *bytes_read = 0;

      switch(errno)
        {
#ifdef EAGAIN
          case EAGAIN:
            return G_IO_STATUS_AGAIN;
#endif
          default:
            g_set_error (err, G_IO_CHANNEL_ERROR,
                         g_io_channel_error_from_errno (errno),
                         strerror (errno));
            return G_IO_STATUS_ERROR;
        }
    }

  *bytes_read = result;

  return G_IO_STATUS_NORMAL; /* XXX: 0 byte read an error ?? */
  return (result > 0) ? G_IO_STATUS_NORMAL : G_IO_STATUS_EOF;
}

static GIOStatus
g_io_win32_fd_write (GIOChannel  *channel,
		     const gchar *buf,
		     gsize        count,
		     gsize       *bytes_written,
		     GError     **err)
{
  GIOWin32Channel *win32_channel = (GIOWin32Channel *)channel;
  gint result;
  
  result = write (win32_channel->fd, buf, count);
  if (win32_channel->debug)
    g_print ("g_io_win32_fd_write: fd:%d count:%d = %d\n",
	     win32_channel->fd, count, result);

  if (result < 0)
    {
      *bytes_written = 0;

      switch(errno)
        {
#ifdef EAGAIN
          case EAGAIN:
            return G_IO_STATUS_AGAIN;
#endif
          default:
            g_set_error (err, G_IO_CHANNEL_ERROR,
                         g_io_channel_error_from_errno (errno),
                         strerror (errno));
            return G_IO_STATUS_ERROR;
        }
    }

  *bytes_written = result;

  return G_IO_STATUS_NORMAL;
}

static GIOStatus
g_io_win32_fd_seek (GIOChannel *channel,
		    glong       offset,
		    GSeekType   type,
		    GError    **err)
{
  GIOWin32Channel *win32_channel = (GIOWin32Channel *)channel;
  int whence;
  off_t result;
  
  switch (type)
    {
    case G_SEEK_SET:
      whence = SEEK_SET;
      break;
    case G_SEEK_CUR:
      whence = SEEK_CUR;
      break;
    case G_SEEK_END:
      whence = SEEK_END;
      break;
    default:
      whence = -1; /* Keep the compiler quiet */
      g_assert_not_reached();
    }
  
  result = lseek (win32_channel->fd, offset, whence);
  
  if (result < 0)
    {
      g_set_error (err, G_IO_CHANNEL_ERROR,
		   g_io_channel_error_from_errno (errno),
		   strerror (errno));
      return G_IO_STATUS_ERROR;
    }

  return G_IO_STATUS_NORMAL;
}

static GIOStatus
g_io_win32_fd_close (GIOChannel *channel,
	             GError    **err)
{
  GIOWin32Channel *win32_channel = (GIOWin32Channel *)channel;
  
  if (win32_channel->debug)
    g_print ("thread %#x: closing fd %d\n",
	     win32_channel->thread_id,
	     win32_channel->fd);
  LOCK (win32_channel->mutex);
  if (win32_channel->running)
    {
      if (win32_channel->debug)
	g_print ("thread %#x: running, marking fd %d for later close\n",
		 win32_channel->thread_id, win32_channel->fd);
      win32_channel->running = FALSE;
      win32_channel->needs_close = TRUE;
      SetEvent (win32_channel->data_avail_event);
    }
  else
    {
      if (win32_channel->debug)
	g_print ("closing fd %d\n", win32_channel->fd);
      close (win32_channel->fd);
      if (win32_channel->debug)
	g_print ("closed fd %d, setting to -1\n",
		 win32_channel->fd);
      win32_channel->fd = -1;
    }
  UNLOCK (win32_channel->mutex);

  /* FIXME error detection? */

  return G_IO_STATUS_NORMAL;
}

static GSource *
g_io_win32_fd_create_watch (GIOChannel    *channel,
			    GIOCondition   condition)
{
  return g_io_win32_create_watch (channel, condition, read_thread);
}

static GIOStatus
g_io_win32_sock_read (GIOChannel *channel,
		      gchar      *buf,
		      gsize       count,
		      gsize      *bytes_read,
		      GError    **err)
{
  GIOWin32Channel *win32_channel = (GIOWin32Channel *)channel;
  gint result;
  GIOChannelError error;

  if (win32_channel->debug)
    g_print ("g_io_win32_sock_read: sockfd:%d count:%d\n",
	     win32_channel->fd, count);
#ifdef WE_NEED_TO_HANDLE_WSAEINTR /* not anymore with wsock2 ? */
repeat: 
#endif
  result = recv (win32_channel->fd, buf, count, 0);

  if (win32_channel->debug)
    g_print ("g_io_win32_sock_read: recv:%d\n", result);
  
  if (result == SOCKET_ERROR)
    {
      *bytes_read = 0;

      switch (WSAGetLastError ())
	{
	case WSAEINVAL:
          error = G_IO_CHANNEL_ERROR_INVAL;
          break;
	case WSAEWOULDBLOCK:
          return G_IO_STATUS_AGAIN;
#ifdef WE_NEED_TO_HANDLE_WSAEINTR /* not anymore with wsock2 ? */
	case WSAEINTR:
          goto repeat;
#endif
	default:
	  error = G_IO_CHANNEL_ERROR_FAILED;
          break;
	}
      g_set_error(err, G_IO_CHANNEL_ERROR, error, _("Socket error"));
      return G_IO_STATUS_ERROR;
      /* FIXME get all errors, better error messages */
    }
  else
    {
      *bytes_read = result;

      return G_IO_STATUS_NORMAL; /* XXX: 0 byte read an error ?? */
      return (result > 0) ? G_IO_STATUS_NORMAL : G_IO_STATUS_EOF;
    }
}

static GIOStatus
g_io_win32_sock_write (GIOChannel  *channel,
		       const gchar *buf,
		       gsize        count,
		       gsize       *bytes_written,
		       GError     **err)
{
  GIOWin32Channel *win32_channel = (GIOWin32Channel *)channel;
  gint result;
  GIOChannelError error;
  
  if (win32_channel->debug)
    g_print ("g_io_win32_sock_write: sockfd:%d count:%d\n",
	     win32_channel->fd, count);
#ifdef WE_NEED_TO_HANDLE_WSAEINTR /* not anymore with wsock2 ? */
repeat:
#endif
  result = send (win32_channel->fd, buf, count, 0);
  
  if (win32_channel->debug)
    g_print ("g_io_win32_sock_write: send:%d\n", result);
  
  if (result == SOCKET_ERROR)
    {
      *bytes_written = 0;

      switch (WSAGetLastError ())
	{
	case WSAEINVAL:
	  error = G_IO_CHANNEL_ERROR_INVAL;
          break;
	case WSAEWOULDBLOCK:
          return G_IO_STATUS_AGAIN;
#ifdef WE_NEED_TO_HANDLE_WSAEINTR /* not anymore with wsock2 ? */
	case WSAEINTR:
          goto repeat;
#endif
	default:
	  error = G_IO_CHANNEL_ERROR_FAILED;
          break;
	}
      g_set_error(err, G_IO_CHANNEL_ERROR, error, _("Socket error"));
      return G_IO_STATUS_ERROR;
      /* FIXME get all errors, better error messages */
    }
  else
    {
      *bytes_written = result;

      return G_IO_STATUS_NORMAL;
    }
}

static GIOStatus
g_io_win32_sock_close (GIOChannel *channel,
		       GError    **err)
{
  GIOWin32Channel *win32_channel = (GIOWin32Channel *)channel;

  LOCK(win32_channel->mutex);
  if (win32_channel->running)
  {
    if (win32_channel->debug)
	g_print ("thread %#x: running, marking for later close\n",
		 win32_channel->thread_id);
    win32_channel->running = FALSE;
    win32_channel->needs_close = TRUE;
    SetEvent(win32_channel->data_avail_noticed_event);
  }
  if (win32_channel->fd != -1)
  {
    if (win32_channel->debug)
       g_print ("thread %#x: closing socket %d\n",
	     win32_channel->thread_id,
	     win32_channel->fd);
  
    closesocket (win32_channel->fd);
    win32_channel->fd = -1;
  }
  UNLOCK(win32_channel->mutex);

  /* FIXME error detection? */

  return G_IO_STATUS_NORMAL;
}

static GSource *
g_io_win32_sock_create_watch (GIOChannel    *channel,
			      GIOCondition   condition)
{
  return g_io_win32_create_watch (channel, condition, select_thread);
}

GIOChannel *
g_io_channel_new_file (const gchar  *filename,
                       const gchar  *mode,
                       GError      **error)
{
  int fid, flags, pmode;
  GIOChannel *channel;

  enum { /* Cheesy hack */
    MODE_R = 1 << 0,
    MODE_W = 1 << 1,
    MODE_A = 1 << 2,
    MODE_PLUS = 1 << 3,
  } mode_num;

  g_return_val_if_fail (filename != NULL, NULL);
  g_return_val_if_fail (mode != NULL, NULL);
  g_return_val_if_fail ((error == NULL) || (*error == NULL), NULL);

  switch (mode[0])
    {
      case 'r':
        mode_num = MODE_R;
        break;
      case 'w':
        mode_num = MODE_W;
        break;
      case 'a':
        mode_num = MODE_A;
        break;
      default:
        g_warning (G_STRLOC ": Invalid GIOFileMode %s.\n", mode);
        return NULL;
    }

  switch (mode[1])
    {
      case '\0':
        break;
      case '+':
        if (mode[2] == '\0')
          {
            mode_num |= MODE_PLUS;
            break;
          }
        /* Fall through */
      default:
        g_warning (G_STRLOC ": Invalid GIOFileMode %s.\n", mode);
        return NULL;
    }

  switch (mode_num)
    {
      case MODE_R:
        flags = O_RDONLY;
        pmode = _S_IREAD;
        break;
      case MODE_W:
        flags = O_WRONLY | O_TRUNC | O_CREAT;
        pmode = _S_IWRITE;
        break;
      case MODE_A:
        flags = O_WRONLY | O_APPEND | O_CREAT;
        pmode = _S_IWRITE;
        break;
      case MODE_R | MODE_PLUS:
        flags = O_RDWR;
        pmode = _S_IREAD | _S_IWRITE;
        break;
      case MODE_W | MODE_PLUS:
        flags = O_RDWR | O_TRUNC | O_CREAT;
        pmode = _S_IREAD | _S_IWRITE;
        break;
      case MODE_A | MODE_PLUS:
        flags = O_RDWR | O_APPEND | O_CREAT;
        pmode = _S_IREAD | _S_IWRITE;
        break;
      default:
        g_assert_not_reached ();
        flags = 0;
        pmode = 0;
    }


  /* always open 'untranslated' */
  fid = open (filename, flags | _O_BINARY, pmode);
  if (fid < 0)
    {
      g_set_error (error, G_FILE_ERROR,
                   g_file_error_from_errno (errno),
                   strerror (errno));
      return (GIOChannel *)NULL;
    }

  channel = g_io_channel_win32_new_fd (fid);

  /* XXX: move this to g_io_channel_win32_new_fd () */
  channel->close_on_unref = TRUE;
  channel->is_seekable = TRUE;

  switch (mode_num)
    {
      case MODE_R:
        channel->is_readable = TRUE;
        channel->is_writeable = FALSE;
        break;
      case MODE_W:
      case MODE_A:
        channel->is_readable = FALSE;
        channel->is_writeable = TRUE;
        break;
      case MODE_R | MODE_PLUS:
      case MODE_W | MODE_PLUS:
      case MODE_A | MODE_PLUS:
        channel->is_readable = TRUE;
        channel->is_writeable = TRUE;
        break;
      default:
        g_assert_not_reached ();
    }

  if (((GIOWin32Channel *)channel)->debug)
    g_print ("g_io_channel_win32_new_file: fd = %ud\n", fid);

  return channel;
}

GIOStatus
g_io_win32_set_flags (GIOChannel     *channel,
                      GIOFlags        flags,
                      GError        **err)
{
  GIOWin32Channel *win32_channel = (GIOWin32Channel *)channel;

  g_set_error (err, 
               G_IO_CHANNEL_ERROR, 
               g_file_error_from_errno (EACCES), 
               _("Channel set flags unsupported"));
  return G_IO_STATUS_ERROR;
}

GIOFlags
g_io_win32_fd_get_flags (GIOChannel     *channel)
{
  GIOFlags flags = 0;
  struct _stat st;
  GIOWin32Channel *win32_channel = (GIOWin32Channel *)channel;

  g_return_val_if_fail (win32_channel != NULL, 0);
  g_return_val_if_fail (win32_channel->type == G_IO_WIN32_FILE_DESC, 0);

  if (0 == _fstat (win32_channel->fd, &st))
    {
       /* XXX: G_IO_FLAG_APPEND */
       /* XXX: G_IO_FLAG_NONBLOCK */
       if (st.st_mode & _S_IREAD)    flags |= G_IO_FLAG_IS_READABLE;
       if (st.st_mode & _S_IWRITE)   flags |= G_IO_FLAG_IS_WRITEABLE;
       /* XXX: */
       if (!(st.st_mode & _S_IFIFO)) flags |= G_IO_FLAG_IS_SEEKABLE;
    }

  return flags;
}

/*
 * Generic implementation, just translating createion flags
 */
GIOFlags
g_io_win32_get_flags (GIOChannel     *channel)
{
  GIOFlags flags;

  flags =   (channel->is_readable ? G_IO_FLAG_IS_READABLE : 0)
          | (channel->is_writeable ? G_IO_FLAG_IS_READABLE : 0)
          | (channel->is_seekable ? G_IO_FLAG_IS_SEEKABLE : 0);

  return flags;
}

static GIOFuncs win32_channel_msg_funcs = {
  g_io_win32_msg_read,
  g_io_win32_msg_write,
  g_io_win32_no_seek,
  g_io_win32_msg_close,
  g_io_win32_msg_create_watch,
  g_io_win32_free,
  g_io_win32_set_flags,
  g_io_win32_get_flags,
};

static GIOFuncs win32_channel_fd_funcs = {
  g_io_win32_fd_read,
  g_io_win32_fd_write,
  g_io_win32_fd_seek,
  g_io_win32_fd_close,
  g_io_win32_fd_create_watch,
  g_io_win32_free,
  g_io_win32_set_flags,
  g_io_win32_fd_get_flags,
};

static GIOFuncs win32_channel_sock_funcs = {
  g_io_win32_sock_read,
  g_io_win32_sock_write,
  g_io_win32_no_seek,
  g_io_win32_sock_close,
  g_io_win32_sock_create_watch,
  g_io_win32_free,
  g_io_win32_set_flags,
  g_io_win32_get_flags,
};

GIOChannel *
g_io_channel_win32_new_messages (guint hwnd)
{
  GIOWin32Channel *win32_channel = g_new (GIOWin32Channel, 1);
  GIOChannel *channel = (GIOChannel *)win32_channel;

  g_io_channel_init (channel);
  g_io_channel_win32_init (win32_channel);
  if (win32_channel->debug)
    g_print ("g_io_channel_win32_new_messages: hwnd = %ud\n", hwnd);
  channel->funcs = &win32_channel_msg_funcs;
  win32_channel->type = G_IO_WIN32_WINDOWS_MESSAGES;
  win32_channel->hwnd = (HWND) hwnd;

  /* XXX: check this. */
  channel->is_readable = IsWindow (win32_channel->hwnd);
  channel->is_writeable = IsWindow (win32_channel->hwnd);

  channel->is_seekable = FALSE;

  return channel;
}

GIOChannel *
g_io_channel_win32_new_fd (gint fd)
{
  GIOWin32Channel *win32_channel;
  GIOChannel *channel;
  struct stat st;

  if (fstat (fd, &st) == -1)
    {
      g_warning (G_STRLOC ": %d isn't a (emulated) file descriptor", fd);
      return NULL;
    }

  win32_channel = g_new (GIOWin32Channel, 1);
  channel = (GIOChannel *)win32_channel;

  g_io_channel_init (channel);
  g_io_channel_win32_init (win32_channel);
  if (win32_channel->debug)
    g_print ("g_io_channel_win32_new_fd: fd = %d\n", fd);
  channel->funcs = &win32_channel_fd_funcs;
  win32_channel->type = G_IO_WIN32_FILE_DESC;
  win32_channel->fd = fd;


  /* fstat doesn't deliver senseful values, but
   * fcntl isn't available, so guess ...
   */
  if (st.st_mode & _S_IFIFO)
    {
      channel->is_readable  = TRUE;
      channel->is_writeable = TRUE;
      channel->is_seekable  = FALSE;
    }
  else
    {
      channel->is_readable  = !!(st.st_mode & _S_IREAD);
      channel->is_writeable = !!(st.st_mode & _S_IWRITE);
      /* XXX: pipes aren't seeakable, are they ? */
      channel->is_seekable = !(st.st_mode & _S_IFIFO);
    }

  return channel;
}

gint
g_io_channel_win32_get_fd (GIOChannel *channel)
{
  GIOWin32Channel *win32_channel = (GIOWin32Channel *)channel;

  return win32_channel->fd;
}

GIOChannel *
g_io_channel_win32_new_socket (int socket)
{
  GIOWin32Channel *win32_channel = g_new (GIOWin32Channel, 1);
  GIOChannel *channel = (GIOChannel *)win32_channel;

  g_io_channel_init (channel);
  g_io_channel_win32_init (win32_channel);
  if (win32_channel->debug)
    g_print ("g_io_channel_win32_new_socket: sockfd:%d\n", socket);
  channel->funcs = &win32_channel_sock_funcs;
  win32_channel->type = G_IO_WIN32_SOCKET;
  win32_channel->fd = socket;

  /* XXX: check this */
  channel->is_readable = TRUE;
  channel->is_writeable = TRUE;
  channel->is_seekable = FALSE;

  return channel;
}

GIOChannel *
g_io_channel_unix_new (gint fd)
{
  struct stat st;

  if (fstat (fd, &st) == 0)
    return g_io_channel_win32_new_fd (fd);
  
  if (getsockopt (fd, SOL_SOCKET, SO_TYPE, NULL, NULL) != SO_ERROR)
    return g_io_channel_win32_new_socket(fd);

  g_warning (G_STRLOC ": %d is neither a file descriptor or a socket", fd);
  return NULL;
}

gint
g_io_channel_unix_get_fd (GIOChannel *channel)
{
  return g_io_channel_win32_get_fd (channel);
}

void
g_io_channel_win32_set_debug (GIOChannel *channel,
			      gboolean    flag)
{
  GIOWin32Channel *win32_channel = (GIOWin32Channel *)channel;

  win32_channel->debug = flag;
}

gint
g_io_channel_win32_poll (GPollFD *fds,
			 gint     n_fds,
			 gint     timeout)
{
  int result;

  g_return_val_if_fail (n_fds >= 0, 0);

  result = (*g_main_context_get_poll_func (NULL)) (fds, n_fds, timeout);

  return result;
}

void
g_io_channel_win32_make_pollfd (GIOChannel   *channel,
				GIOCondition  condition,
				GPollFD      *fd)
{
  GIOWin32Channel *win32_channel = (GIOWin32Channel *)channel;

  if (win32_channel->data_avail_event == NULL)
    create_events (win32_channel);
  
  fd->fd = (gint) win32_channel->data_avail_event;
  fd->events = condition;

  if (win32_channel->thread_id == 0)
    if ((condition & G_IO_IN) && win32_channel->type == G_IO_WIN32_FILE_DESC)
      create_thread (win32_channel, condition, read_thread);
    else if (win32_channel->type == G_IO_WIN32_SOCKET)
      create_thread (win32_channel, condition, select_thread);
}

/* Binary compatibility */
GIOChannel *
g_io_channel_win32_new_stream_socket (int socket)
{
  return g_io_channel_win32_new_socket (socket);
}
