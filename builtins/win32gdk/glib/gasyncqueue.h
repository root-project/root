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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
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

#ifndef __G_ASYNCQUEUE_H__
#define __G_ASYNCQUEUE_H__

#include <glib/gthread.h>

G_BEGIN_DECLS

typedef struct _GAsyncQueue     GAsyncQueue;

/* Asyncronous Queues, can be used to communicate between threads
 */

/* Get a new GAsyncQueue with the ref_count 1 */
GAsyncQueue*  g_async_queue_new                (void);

/* Lock and unlock an GAsyncQueue, all functions lock the queue for
 * themselves, but in certain cirumstances you want to hold the lock longer,
 * thus you lock the queue, call the *_unlocked functions and unlock it again
 */
void          g_async_queue_lock               (GAsyncQueue *queue);
void          g_async_queue_unlock             (GAsyncQueue *queue);

/* Ref and unref the GAsyncQueue. g_async_queue_unref_unlocked makes
 * no sense, as after the unreffing the Queue might be gone and can't
 * be unlocked. So you have a function to call, if you don't hold the
 * lock (g_async_queue_unref) and one to call, when you already hold
 * the lock (g_async_queue_unref_and_unlock). After that however, you
 * don't hold the lock anymore and the Queue might in fact be
 * destroyed, if you unrefed to zero */
void          g_async_queue_ref                (GAsyncQueue *queue);
void          g_async_queue_ref_unlocked       (GAsyncQueue *queue);
void          g_async_queue_unref              (GAsyncQueue *queue);
void          g_async_queue_unref_and_unlock   (GAsyncQueue *queue);

/* Push data into the async queue. Must not be NULL */
void          g_async_queue_push               (GAsyncQueue *queue,
                                                gpointer     data);
void          g_async_queue_push_unlocked      (GAsyncQueue *queue,
                                                gpointer     data);

/* Pop data from the async queue, when no data is there, the thread is blocked
 * until data arrives */
gpointer      g_async_queue_pop                (GAsyncQueue *queue);
gpointer      g_async_queue_pop_unlocked       (GAsyncQueue *queue);

/* Try to pop data, NULL is returned in case of empty queue */
gpointer      g_async_queue_try_pop            (GAsyncQueue *queue);
gpointer      g_async_queue_try_pop_unlocked   (GAsyncQueue *queue);

/* Wait for data until at maximum until end_time is reached, NULL is returned
 * in case of empty queue*/
gpointer      g_async_queue_timed_pop          (GAsyncQueue *queue,
                                                GTimeVal    *end_time);
gpointer      g_async_queue_timed_pop_unlocked (GAsyncQueue *queue,
                                                GTimeVal    *end_time);

/* Return the length of the queue, negative values mean, that threads
 * are waiting, positve values mean, that there are entries in the
 * queue. Actually this function returns the length of the queue minus
 * the number of waiting threads, g_async_queue_length == 0 could also
 * mean 'n' entries in the queue and 'n' thread waiting, such can
 * happen due to locking of the queue or due to scheduling. */
gint          g_async_queue_length             (GAsyncQueue *queue);
gint          g_async_queue_length_unlocked    (GAsyncQueue *queue);

G_END_DECLS

#endif /* __G_ASYNCQUEUE_H__ */

