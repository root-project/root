/* GLIB - Library of useful routines for C programming
 * Copyright (C) 1995-1997  Peter Mattis, Spencer Kimball and Josh MacDonald
 *
 * GAsyncQueue: asynchronous queue implementation, based on Gqueue.
 * Copyright (C) 2000 Sebastian Wilhelmi; University of Karlsruhe
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
 * MT safe
 */

#include "glib.h"

struct _GAsyncQueue
{
  GMutex *mutex;
  GCond *cond;
  GQueue *queue;
  guint waiting_threads;
  guint ref_count;
};

/**
 * g_async_queue_new:
 * 
 * Creates a new asynchronous queue with the initial reference count of 1.
 * 
 * Return value: the new #GAsyncQueue.
 **/
GAsyncQueue*
g_async_queue_new ()
{
  GAsyncQueue* retval = g_new (GAsyncQueue, 1);
  retval->mutex = g_mutex_new ();
  retval->cond = g_cond_new ();
  retval->queue = g_queue_new ();
  retval->waiting_threads = 0;
  retval->ref_count = 1;
  return retval;
}

/**
 * g_async_queue_ref:
 * @queue: a #GAsyncQueue.
 *
 * Increases the reference count of the asynchronous @queue by 1.
 **/
void 
g_async_queue_ref (GAsyncQueue *queue)
{
  g_return_if_fail (queue);
  g_return_if_fail (queue->ref_count > 0);
  
  g_mutex_lock (queue->mutex);
  queue->ref_count++;
  g_mutex_unlock (queue->mutex);
}

/**
 * g_async_queue_ref_unlocked:
 * @queue: a #GAsyncQueue.
 * 
 * Increases the reference count of the asynchronous @queue by 1. This
 * function must be called while holding the @queue's lock.
 **/
void 
g_async_queue_ref_unlocked (GAsyncQueue *queue)
{
  g_return_if_fail (queue);
  g_return_if_fail (queue->ref_count > 0);
  
  queue->ref_count++;
}

/**
 * g_async_queue_unref_and_unlock:
 * @queue: a #GAsyncQueue.
 * 
 * Decreases the reference count of the asynchronous @queue by 1 and
 * releases the lock. This function must be called while holding the
 * @queue's lock. If the reference count went to 0, the @queue will be
 * destroyed and the memory allocated will be freed. So you are not
 * allowed to use the @queue afterwards, as it might have disappeared.
 * The obvious asymmetry (it is not named
 * g_async_queue_unref_unlocked) is because the queue can't be
 * unlocked after dereffing it, as it might already have disappeared.
 **/
void 
g_async_queue_unref_and_unlock (GAsyncQueue *queue)
{
  gboolean stop;

  g_return_if_fail (queue);
  g_return_if_fail (queue->ref_count > 0);

  queue->ref_count--;
  stop = (queue->ref_count == 0);
  g_mutex_unlock (queue->mutex);
  
  if (stop)
    {
      g_return_if_fail (queue->waiting_threads == 0);
      g_mutex_free (queue->mutex);
      g_cond_free (queue->cond);
      g_queue_free (queue->queue);
      g_free (queue);
    }
}

/**
 * g_async_queue_unref:
 * @queue: a #GAsyncQueue.
 * 
 * Decreases the reference count of the asynchronous @queue by 1. If
 * the reference count went to 0, the @queue will be destroyed and the
 * memory allocated will be freed. So you are not allowed to use the
 * @queue afterwards, as it might have disappeared.
 **/
void 
g_async_queue_unref (GAsyncQueue *queue)
{
  g_return_if_fail (queue);
  g_return_if_fail (queue->ref_count > 0);

  g_mutex_lock (queue->mutex);
  g_async_queue_unref_and_unlock (queue);
}

/**
 * g_async_queue_lock:
 * @queue: a #GAsyncQueue.
 * 
 * Acquire the @queue's lock. After that you can only call the
 * g_async_queue_*_unlocked function variants on that
 * @queue. Otherwise it will deadlock.
 **/
void
g_async_queue_lock (GAsyncQueue *queue)
{
  g_return_if_fail (queue);
  g_return_if_fail (queue->ref_count > 0);

  g_mutex_lock (queue->mutex);
}

/**
 * g_async_queue_unlock:
 * @queue: a #GAsyncQueue.
 * 
 * Release the queue's lock.
 **/
void 
g_async_queue_unlock (GAsyncQueue *queue)
{
  g_return_if_fail (queue);
  g_return_if_fail (queue->ref_count > 0);

  g_mutex_unlock (queue->mutex);
}

/**
 * g_async_queue_push:
 * @queue: a #GAsyncQueue.
 * @data: @data to push into the @queue.
 *
 * Push the @data into the @queue. @data must not be #NULL.
 **/
void
g_async_queue_push (GAsyncQueue* queue, gpointer data)
{
  g_return_if_fail (queue);
  g_return_if_fail (queue->ref_count > 0);
  g_return_if_fail (data);

  g_mutex_lock (queue->mutex);
  g_async_queue_push_unlocked (queue, data);
  g_mutex_unlock (queue->mutex);
}

/**
 * g_async_queue_push_unlocked:
 * @queue: a #GAsyncQueue.
 * @data: @data to push into the @queue.
 * 
 * Push the @data into the @queue. @data must not be #NULL. This
 * function must be called while holding the @queue's lock.
 **/
void
g_async_queue_push_unlocked (GAsyncQueue* queue, gpointer data)
{
  g_return_if_fail (queue);
  g_return_if_fail (queue->ref_count > 0);
  g_return_if_fail (data);

  g_queue_push_head (queue->queue, data);
  g_cond_signal (queue->cond);
}

static gpointer
g_async_queue_pop_intern_unlocked (GAsyncQueue* queue, gboolean tryit, 
				   GTimeVal *end_time)
{
  gpointer retval;

  if (!g_queue_peek_tail (queue->queue))
    {
      if (tryit)
	return NULL;
      if (!end_time)
        {
          queue->waiting_threads++;
	  while (!g_queue_peek_tail (queue->queue))
            g_cond_wait(queue->cond, queue->mutex);
          queue->waiting_threads--;
        }
      else
        {
          queue->waiting_threads++;
          while (!g_queue_peek_tail (queue->queue))
            if (!g_cond_timed_wait (queue->cond, queue->mutex, end_time))
              break;
          queue->waiting_threads--;
          if (!g_queue_peek_tail (queue->queue))
	    return NULL;
        }
    }

  retval = g_queue_pop_tail (queue->queue);

  g_assert (retval);

  return retval;
}

/**
 * g_async_queue_pop:
 * @queue: a #GAsyncQueue.
 * 
 * Pop data from the @queue. This function blocks until data become
 * available.
 *
 * Return value: data from the queue.
 **/
gpointer
g_async_queue_pop (GAsyncQueue* queue)
{
  gpointer retval;

  g_return_val_if_fail (queue, NULL);
  g_return_val_if_fail (queue->ref_count > 0, NULL);

  g_mutex_lock (queue->mutex);
  retval = g_async_queue_pop_intern_unlocked (queue, FALSE, NULL);
  g_mutex_unlock (queue->mutex);

  return retval;
}

/**
 * g_async_queue_pop_unlocked:
 * @queue: a #GAsyncQueue.
 * 
 * Pop data from the @queue. This function blocks until data become
 * available. This function must be called while holding the @queue's
 * lock.
 *
 * Return value: data from the queue.
 **/
gpointer
g_async_queue_pop_unlocked (GAsyncQueue* queue)
{
  g_return_val_if_fail (queue, NULL);
  g_return_val_if_fail (queue->ref_count > 0, NULL);

  return g_async_queue_pop_intern_unlocked (queue, FALSE, NULL);
}

/**
 * g_async_queue_try_pop:
 * @queue: a #GAsyncQueue.
 * 
 * Try to pop data from the @queue. If no data is available, #NULL is
 * returned.
 *
 * Return value: data from the queue or #NULL, when no data is
 * available immediately.
 **/
gpointer
g_async_queue_try_pop (GAsyncQueue* queue)
{
  gpointer retval;

  g_return_val_if_fail (queue, NULL);
  g_return_val_if_fail (queue->ref_count > 0, NULL);

  g_mutex_lock (queue->mutex);
  retval = g_async_queue_pop_intern_unlocked (queue, TRUE, NULL);
  g_mutex_unlock (queue->mutex);

  return retval;
}

/**
 * g_async_queue_try_pop_unlocked:
 * @queue: a #GAsyncQueue.
 * 
 * Try to pop data from the @queue. If no data is available, #NULL is
 * returned. This function must be called while holding the @queue's
 * lock.
 *
 * Return value: data from the queue or #NULL, when no data is
 * available immediately.
 **/
gpointer
g_async_queue_try_pop_unlocked (GAsyncQueue* queue)
{
  g_return_val_if_fail (queue, NULL);
  g_return_val_if_fail (queue->ref_count > 0, NULL);

  return g_async_queue_pop_intern_unlocked (queue, TRUE, NULL);
}

/**
 * g_async_queue_timed_pop:
 * @queue: a #GAsyncQueue.
 * @end_time: a #GTimeVal, determining the final time.
 *
 * Pop data from the @queue. If no data is received before @end_time,
 * #NULL is returned.
 *
 * To easily calculate @end_time a combination of g_get_current_time()
 * and g_time_val_add() can be used.
 *
 * Return value: data from the queue or #NULL, when no data is
 * received before @end_time.
 **/
gpointer
g_async_queue_timed_pop (GAsyncQueue* queue, GTimeVal *end_time)
{
  gpointer retval;

  g_return_val_if_fail (queue, NULL);
  g_return_val_if_fail (queue->ref_count > 0, NULL);

  g_mutex_lock (queue->mutex);
  retval = g_async_queue_pop_intern_unlocked (queue, FALSE, end_time);
  g_mutex_unlock (queue->mutex);

  return retval;  
}

/**
 * g_async_queue_timed_pop_unlocked:
 * @queue: a #GAsyncQueue.
 * @end_time: a #GTimeVal, determining the final time.
 *
 * Pop data from the @queue. If no data is received before @end_time,
 * #NULL is returned. This function must be called while holding the
 * @queue's lock.
 *
 * To easily calculate @end_time a combination of g_get_current_time()
 * and g_time_val_add() can be used.
 *
 * Return value: data from the queue or #NULL, when no data is
 * received before @end_time.
 **/
gpointer
g_async_queue_timed_pop_unlocked (GAsyncQueue* queue, GTimeVal *end_time)
{
  g_return_val_if_fail (queue, NULL);
  g_return_val_if_fail (queue->ref_count > 0, NULL);

  return g_async_queue_pop_intern_unlocked (queue, FALSE, end_time);
}

/**
 * g_async_queue_length:
 * @queue: a #GAsyncQueue.
 * 
 * Returns the length of the queue, negative values mean waiting
 * threads, positive values mean available entries in the
 * @queue. Actually this function returns the number of data items in
 * the queue minus the number of waiting threads. Thus a return value
 * of 0 could mean 'n' entries in the queue and 'n' thread waiting.
 * That can happen due to locking of the queue or due to
 * scheduling.  
 *
 * Return value: the length of the @queue.
 **/
gint
g_async_queue_length (GAsyncQueue* queue)
{
  gint retval;

  g_return_val_if_fail (queue, 0);
  g_return_val_if_fail (queue->ref_count > 0, 0);

  g_mutex_lock (queue->mutex);
  retval = queue->queue->length - queue->waiting_threads;
  g_mutex_unlock (queue->mutex);

  return retval;
}

/**
 * g_async_queue_length_unlocked:
 * @queue: a #GAsyncQueue.
 * 
 * Returns the length of the queue, negative values mean waiting
 * threads, positive values mean available entries in the
 * @queue. Actually this function returns the number of data items in
 * the queue minus the number of waiting threads. Thus a return value
 * of 0 could mean 'n' entries in the queue and 'n' thread waiting.
 * That can happen due to locking of the queue or due to
 * scheduling. This function must be called while holding the @queue's
 * lock.
 *
 * Return value: the length of the @queue.
 **/
gint
g_async_queue_length_unlocked (GAsyncQueue* queue)
{
  g_return_val_if_fail (queue, 0);
  g_return_val_if_fail (queue->ref_count > 0, 0);

  return queue->queue->length - queue->waiting_threads;
}

