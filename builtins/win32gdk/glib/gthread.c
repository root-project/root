/* GLIB - Library of useful routines for C programming
 * Copyright (C) 1995-1997  Peter Mattis, Spencer Kimball and Josh MacDonald
 *
 * gmutex.c: MT safety related functions
 * Copyright 1998 Sebastian Wilhelmi; University of Karlsruhe
 *                Owen Taylor
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

/* 
 * MT safe
 */

#include "config.h"
#include "glib.h"

#ifdef G_THREAD_USE_PID_SURROGATE
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <errno.h>
#endif /* G_THREAD_USE_PID_SURROGATE */

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include <string.h>

#if GLIB_SIZEOF_SYSTEM_THREAD == SIZEOF_VOID_P
# define g_system_thread_equal_simple(thread1, thread2)			\
   ((thread1).dummy_pointer == (thread2).dummy_pointer)
# define g_system_thread_assign(dest, src)				\
   ((dest).dummy_pointer = (src).dummy_pointer)
#else /* GLIB_SIZEOF_SYSTEM_THREAD != SIZEOF_VOID_P */
# define g_system_thread_equal_simple(thread1, thread2)			\
   (memcmp (&(thread1), &(thread2), GLIB_SIZEOF_SYSTEM_THREAD) == 0)
# define g_system_thread_assign(dest, src)				\
   (memcpy (&(dest), &(src), GLIB_SIZEOF_SYSTEM_THREAD))
#endif /* GLIB_SIZEOF_SYSTEM_THREAD == SIZEOF_VOID_P */

#define g_system_thread_equal(thread1, thread2)				\
  (g_thread_functions_for_glib_use.thread_equal ? 			\
   g_thread_functions_for_glib_use.thread_equal (&(thread1), &(thread2)) :\
   g_system_thread_equal_simple((thread1), (thread2)))

GQuark 
g_thread_error_quark (void)
{
  static GQuark quark;
  if (!quark)
    quark = g_quark_from_static_string ("g_thread_error");
  return quark;
}

/* Keep this in sync with GRealThread in gmain.c! */
typedef struct _GRealThread GRealThread;
struct  _GRealThread
{
  GThread thread;
  gpointer private_data;
  gpointer retval;
  GSystemThread system_thread;
#ifdef G_THREAD_USE_PID_SURROGATE
  pid_t pid;
#endif /* G_THREAD_USE_PID_SURROGATE */
};

#ifdef G_THREAD_USE_PID_SURROGATE
static gint priority_map[] = { 15, 0, -15, -20 };
static gboolean prio_warned = FALSE;
# define SET_PRIO(pid, prio) G_STMT_START{				\
  gint error = setpriority (PRIO_PROCESS, (pid), priority_map[prio]);	\
  if (error == -1 && errno == EACCES && !prio_warned)			\
    {									\
      prio_warned = TRUE;						\
      g_warning ("Priorities can only be increased by root.");		\
    }									\
  }G_STMT_END
#endif /* G_THREAD_USE_PID_SURROGATE */

typedef struct _GStaticPrivateNode GStaticPrivateNode;
struct _GStaticPrivateNode
{
  gpointer       data;
  GDestroyNotify destroy;
};

static void g_thread_cleanup (gpointer data);
static void g_thread_fail (void);

/* Global variables */

static GSystemThread zero_thread; /* This is initialized to all zero */
gboolean g_thread_use_default_impl = TRUE;
gboolean g_threads_got_initialized = FALSE;

#if defined(G_PLATFORM_WIN32) && defined(__GNUC__)
__declspec(dllexport)
#endif
GThreadFunctions g_thread_functions_for_glib_use = {
  (GMutex*(*)(void))g_thread_fail,                 /* mutex_new */
  NULL,                                        /* mutex_lock */
  NULL,                                        /* mutex_trylock */
  NULL,                                        /* mutex_unlock */
  NULL,                                        /* mutex_free */
  (GCond*(*)(void))g_thread_fail,                  /* cond_new */
  NULL,                                        /* cond_signal */
  NULL,                                        /* cond_broadcast */
  NULL,                                        /* cond_wait */
  NULL,                                        /* cond_timed_wait  */
  NULL,                                        /* cond_free */
  (GPrivate*(*)(GDestroyNotify))g_thread_fail, /* private_new */
  NULL,                                        /* private_get */
  NULL,                                        /* private_set */
  (void(*)(GThreadFunc, gpointer, gulong, 
	   gboolean, gboolean, GThreadPriority, 
	   gpointer, GError**))g_thread_fail,  /* thread_create */
  NULL,                                        /* thread_yield */
  NULL,                                        /* thread_join */
  NULL,                                        /* thread_exit */
  NULL,                                        /* thread_set_priority */
  NULL                                         /* thread_self */
}; 

/* Local data */

static GMutex   *g_mutex_protect_static_mutex_allocation = NULL;
static GPrivate *g_thread_specific_private = NULL;
static GSList   *g_thread_all_threads = NULL;
static GSList   *g_thread_free_indeces = NULL;

G_LOCK_DEFINE_STATIC (g_thread);

/* This must be called only once, before any threads are created.
 * It will only be called from g_thread_init() in -lgthread.
 */
void
g_mutex_init (void)
{
  GRealThread* main_thread;
 
  /* We let the main thread (the one that calls g_thread_init) inherit
   * the data, that it set before calling g_thread_init
   */
  main_thread = (GRealThread*) g_thread_self ();

  g_thread_specific_private = g_private_new (g_thread_cleanup);
  G_THREAD_UF (private_set, (g_thread_specific_private, main_thread));
  G_THREAD_UF (thread_self, (&main_thread->system_thread));

  g_mutex_protect_static_mutex_allocation = g_mutex_new ();
}

void 
g_static_mutex_init (GStaticMutex *mutex)
{
  static GStaticMutex init_mutex = G_STATIC_MUTEX_INIT;

  g_return_if_fail (mutex);

  *mutex = init_mutex;
}

GMutex *
g_static_mutex_get_mutex_impl (GMutex** mutex)
{
  if (!g_thread_supported ())
    return NULL;

  g_assert (g_mutex_protect_static_mutex_allocation);

  g_mutex_lock (g_mutex_protect_static_mutex_allocation);

  if (!(*mutex)) 
    *mutex = g_mutex_new (); 

  g_mutex_unlock (g_mutex_protect_static_mutex_allocation);
  
  return *mutex;
}

void
g_static_mutex_free (GStaticMutex* mutex)
{
  GMutex **runtime_mutex;
  
  g_return_if_fail (mutex);

  /* The runtime_mutex is the first (or only) member of GStaticMutex,
   * see both versions (of glibconfig.h) in configure.in */
  runtime_mutex = ((GMutex**)mutex);
  
  if (*runtime_mutex)
    g_mutex_free (*runtime_mutex);

  *runtime_mutex = NULL;
}

void     
g_static_rec_mutex_init (GStaticRecMutex *mutex)
{
  static GStaticRecMutex init_mutex = G_STATIC_REC_MUTEX_INIT;
  
  g_return_if_fail (mutex);

  *mutex = init_mutex;
}

void
g_static_rec_mutex_lock (GStaticRecMutex* mutex)
{
  GSystemThread self;

  g_return_if_fail (mutex);

  if (!g_thread_supported ())
    return;

  G_THREAD_UF (thread_self, (&self));

  if (g_system_thread_equal (self, mutex->owner))
    {
      mutex->depth++;
      return;
    }
  g_static_mutex_lock (&mutex->mutex);
  g_system_thread_assign (mutex->owner, self);
  mutex->depth = 1;
}

gboolean
g_static_rec_mutex_trylock (GStaticRecMutex* mutex)
{
  GSystemThread self;

  g_return_val_if_fail (mutex, FALSE);

  if (!g_thread_supported ())
    return TRUE;

  G_THREAD_UF (thread_self, (&self));

  if (g_system_thread_equal (self, mutex->owner))
    {
      mutex->depth++;
      return TRUE;
    }

  if (!g_static_mutex_trylock (&mutex->mutex))
    return FALSE;

  g_system_thread_assign (mutex->owner, self);
  mutex->depth = 1;
  return TRUE;
}

void
g_static_rec_mutex_unlock (GStaticRecMutex* mutex)
{
  g_return_if_fail (mutex);

  if (!g_thread_supported ())
    return;

  if (mutex->depth > 1)
    {
      mutex->depth--;
      return;
    }
  g_system_thread_assign (mutex->owner, zero_thread);
  g_static_mutex_unlock (&mutex->mutex);  
}

void
g_static_rec_mutex_lock_full   (GStaticRecMutex *mutex,
				guint            depth)
{
  GSystemThread self;
  g_return_if_fail (mutex);

  if (!g_thread_supported ())
    return;

  G_THREAD_UF (thread_self, (&self));

  if (g_system_thread_equal (self, mutex->owner))
    {
      mutex->depth += depth;
      return;
    }
  g_static_mutex_lock (&mutex->mutex);
  g_system_thread_assign (mutex->owner, self);
  mutex->depth = depth;
}

guint    
g_static_rec_mutex_unlock_full (GStaticRecMutex *mutex)
{
  gint depth;

  g_return_val_if_fail (mutex, 0);

  if (!g_thread_supported ())
    return 1;

  depth = mutex->depth;

  g_system_thread_assign (mutex->owner, zero_thread);
  mutex->depth = 0;
  g_static_mutex_unlock (&mutex->mutex);

  return depth;
}

void
g_static_rec_mutex_free (GStaticRecMutex *mutex)
{
  g_return_if_fail (mutex);

  g_static_mutex_free (&mutex->mutex);
}

void     
g_static_private_init (GStaticPrivate *private_key)
{
  private_key->index = 0;
}

gpointer
g_static_private_get (GStaticPrivate *private_key)
{
  GRealThread *self = (GRealThread*) g_thread_self ();
  GArray *array;

  array = self->private_data;
  if (!array)
    return NULL;

  if (!private_key->index)
    return NULL;
  else if (private_key->index <= array->len)
    return g_array_index (array, GStaticPrivateNode, 
			  private_key->index - 1).data;
  else
    return NULL;
}

void
g_static_private_set (GStaticPrivate *private_key, 
		      gpointer        data,
		      GDestroyNotify  notify)
{
  GRealThread *self = (GRealThread*) g_thread_self ();
  GArray *array;
  static guint next_index = 0;
  GStaticPrivateNode *node;

  array = self->private_data;
  if (!array)
    {
      array = g_array_new (FALSE, TRUE, sizeof (GStaticPrivateNode));
      self->private_data = array;
    }

  if (!private_key->index)
    {
      G_LOCK (g_thread);

      if (!private_key->index)
	{
	  if (g_thread_free_indeces)
	    {
	      private_key->index = 
		GPOINTER_TO_UINT (g_thread_free_indeces->data);
	      g_thread_free_indeces = 
		g_slist_delete_link (g_thread_free_indeces,
				     g_thread_free_indeces);
	    }
	  else
	    private_key->index = ++next_index;
	}

      G_UNLOCK (g_thread);
    }

  if (private_key->index > array->len)
    g_array_set_size (array, private_key->index);

  node = &g_array_index (array, GStaticPrivateNode, private_key->index - 1);
  if (node->destroy)
    {
      gpointer ddata = node->data;
      GDestroyNotify ddestroy = node->destroy;

      node->data = data;
      node->destroy = notify;

      ddestroy (ddata);
    }
  else
    {
      node->data = data;
      node->destroy = notify;
    }
}

void     
g_static_private_free (GStaticPrivate *private_key)
{
  gulong index = private_key->index;
  GSList *list;

  if (!index)
    return;
  
  private_key->index = 0;

  G_LOCK (g_thread);
  list =  g_thread_all_threads;
  while (list)
    {
      GRealThread *thread = list->data;
      GArray *array = thread->private_data;
      list = list->next;

      if (array && index <= array->len)
	{
	  GStaticPrivateNode *node = &g_array_index (array, 
						     GStaticPrivateNode, 
						     index - 1);
	  gpointer ddata = node->data;
	  GDestroyNotify ddestroy = node->destroy;

	  node->data = NULL;
	  node->destroy = NULL;

	  if (ddestroy) 
	    {
	      G_UNLOCK (g_thread);
	      ddestroy (ddata);
	      G_LOCK (g_thread);
	      }
	}
    }
  g_thread_free_indeces = g_slist_prepend (g_thread_free_indeces, 
					   GUINT_TO_POINTER (index));
  G_UNLOCK (g_thread);
}

static void
g_thread_cleanup (gpointer data)
{
  if (data)
    {
      GRealThread* thread = data;
      if (thread->private_data)
	{
	  GArray* array = thread->private_data;
	  guint i;
	  
	  for (i = 0; i < array->len; i++ )
	    {
	      GStaticPrivateNode *node = 
		&g_array_index (array, GStaticPrivateNode, i);
	      if (node->destroy)
		node->destroy (node->data);
	    }
	  g_array_free (array, TRUE);
	}

      /* We only free the thread structure, if it isn't joinable. If
         it is, the structure is freed in g_thread_join */
      if (!thread->thread.joinable)
	{
	  G_LOCK (g_thread);
	  g_thread_all_threads = g_slist_remove (g_thread_all_threads, data);
	  G_UNLOCK (g_thread);
	  
	  /* Just to make sure, this isn't used any more */
	  g_system_thread_assign (thread->system_thread, zero_thread);
	  g_free (thread);
	}
    }
}

static void
g_thread_fail (void)
{
  g_error ("The thread system is not yet initialized.");
}

static gpointer
g_thread_create_proxy (gpointer data)
{
  GRealThread* thread = data;

  g_assert (data);

#ifdef G_THREAD_USE_PID_SURROGATE
  thread->pid = getpid ();
#endif /* G_THREAD_USE_PID_SURROGATE */

  /* This has to happen before G_LOCK, as that might call g_thread_self */
  g_private_set (g_thread_specific_private, data);

  /* the lock makes sure, that thread->system_thread is written,
     before thread->thread.func is called. See g_thread_create. */
  G_LOCK (g_thread);
  G_UNLOCK (g_thread);
 
#ifdef G_THREAD_USE_PID_SURROGATE
  if (g_thread_use_default_impl)
    SET_PRIO (thread->pid, thread->thread.priority);
#endif /* G_THREAD_USE_PID_SURROGATE */

  thread->retval = thread->thread.func (thread->thread.data);

  return NULL;
}

GThread* 
g_thread_create_full (GThreadFunc 		 func,
		      gpointer 		 data,
		      gulong 		 stack_size,
		      gboolean 		 joinable,
		      gboolean 		 bound,
		      GThreadPriority 	 priority,
		      GError                **error)
{
  GRealThread* result = g_new (GRealThread, 1);
  GError *local_error = NULL;
  g_return_val_if_fail (func, NULL);
  g_return_val_if_fail (priority >= G_THREAD_PRIORITY_LOW, NULL);
  g_return_val_if_fail (priority <= G_THREAD_PRIORITY_URGENT, NULL);
  
  result->thread.joinable = joinable;
  result->thread.priority = priority;
  result->thread.func = func;
  result->thread.data = data;
  result->private_data = NULL; 
  G_LOCK (g_thread);
  G_THREAD_UF (thread_create, (g_thread_create_proxy, result, 
			       stack_size, joinable, bound, priority,
			       &result->system_thread, &local_error));
  g_thread_all_threads = g_slist_prepend (g_thread_all_threads, result);
  G_UNLOCK (g_thread);

  if (local_error)
    {
      g_propagate_error (error, local_error);
      g_free (result);
      return NULL;
    }

  return (GThread*) result;
}

void
g_thread_exit (gpointer retval)
{
  GRealThread* real = (GRealThread*) g_thread_self ();
  real->retval = retval;
  G_THREAD_CF (thread_exit, (void)0, ());
}

gpointer
g_thread_join (GThread* thread)
{
  GRealThread* real = (GRealThread*) thread;
  gpointer retval;

  g_return_val_if_fail (thread, NULL);
  g_return_val_if_fail (thread->joinable, NULL);
  g_return_val_if_fail (!g_system_thread_equal (real->system_thread, 
						zero_thread), NULL);

  G_THREAD_UF (thread_join, (&real->system_thread));

  retval = real->retval;

  G_LOCK (g_thread);
  g_thread_all_threads = g_slist_remove (g_thread_all_threads, thread);
  G_UNLOCK (g_thread);

  /* Just to make sure, this isn't used any more */
  thread->joinable = 0;
  g_system_thread_assign (real->system_thread, zero_thread);

  /* the thread structure for non-joinable threads is freed upon
     thread end. We free the memory here. This will leave a loose end,
     if a joinable thread is not joined. */

  g_free (thread);

  return retval;
}

void
g_thread_set_priority (GThread* thread, 
		       GThreadPriority priority)
{
  GRealThread* real = (GRealThread*) thread;

  g_return_if_fail (thread);
  g_return_if_fail (!g_system_thread_equal (real->system_thread, zero_thread));
  g_return_if_fail (priority >= G_THREAD_PRIORITY_LOW);
  g_return_if_fail (priority <= G_THREAD_PRIORITY_URGENT);

  thread->priority = priority;

#ifdef G_THREAD_USE_PID_SURROGATE
  if (g_thread_use_default_impl)
    SET_PRIO (real->pid, priority);
  else
#endif /* G_THREAD_USE_PID_SURROGATE */
    G_THREAD_CF (thread_set_priority, (void)0, 
		 (&real->system_thread, priority));
}

GThread*
g_thread_self (void)
{
  GRealThread* thread = g_private_get (g_thread_specific_private);

  if (!thread)
    {  
      /* If no thread data is available, provide and set one.  This
         can happen for the main thread and for threads, that are not
         created by GLib. */
      thread = g_new (GRealThread, 1);
      thread->thread.joinable = FALSE; /* This is a save guess */
      thread->thread.priority = G_THREAD_PRIORITY_NORMAL; /* This is
							     just a guess */
      thread->thread.func = NULL;
      thread->thread.data = NULL;
      thread->private_data = NULL;

      if (g_thread_supported ())
	G_THREAD_UF (thread_self, (&thread->system_thread));

#ifdef G_THREAD_USE_PID_SURROGATE
      thread->pid = getpid ();
#endif /* G_THREAD_USE_PID_SURROGATE */
      
      g_private_set (g_thread_specific_private, thread); 
      
      G_LOCK (g_thread);
      g_thread_all_threads = g_slist_prepend (g_thread_all_threads, thread);
      G_UNLOCK (g_thread);
    }
  
  return (GThread*)thread;
}

void
g_static_rw_lock_init (GStaticRWLock* lock)
{
  static GStaticRWLock init_lock = G_STATIC_RW_LOCK_INIT;

  g_return_if_fail (lock);

  *lock = init_lock;
}

static void inline 
g_static_rw_lock_wait (GCond** cond, GStaticMutex* mutex)
{
  if (!*cond)
      *cond = g_cond_new ();
  g_cond_wait (*cond, g_static_mutex_get_mutex (mutex));
}

static void inline 
g_static_rw_lock_signal (GStaticRWLock* lock)
{
  if (lock->want_to_write && lock->write_cond)
    g_cond_signal (lock->write_cond);
  else if (lock->want_to_read && lock->read_cond)
    g_cond_broadcast (lock->read_cond);
}

void 
g_static_rw_lock_reader_lock (GStaticRWLock* lock)
{
  g_return_if_fail (lock);

  if (!g_threads_got_initialized)
    return;

  g_static_mutex_lock (&lock->mutex);
  lock->want_to_read++;
  while (lock->write || lock->want_to_write) 
    g_static_rw_lock_wait (&lock->read_cond, &lock->mutex);
  lock->want_to_read--;
  lock->read_counter++;
  g_static_mutex_unlock (&lock->mutex);
}

gboolean 
g_static_rw_lock_reader_trylock (GStaticRWLock* lock)
{
  gboolean ret_val = FALSE;

  g_return_val_if_fail (lock, FALSE);

  if (!g_threads_got_initialized)
    return TRUE;

  g_static_mutex_lock (&lock->mutex);
  if (!lock->write && !lock->want_to_write)
    {
      lock->read_counter++;
      ret_val = TRUE;
    }
  g_static_mutex_unlock (&lock->mutex);
  return ret_val;
}

void 
g_static_rw_lock_reader_unlock  (GStaticRWLock* lock)
{
  g_return_if_fail (lock);

  if (!g_threads_got_initialized)
    return;

  g_static_mutex_lock (&lock->mutex);
  lock->read_counter--;
  if (lock->read_counter == 0)
    g_static_rw_lock_signal (lock);
  g_static_mutex_unlock (&lock->mutex);
}

void 
g_static_rw_lock_writer_lock (GStaticRWLock* lock)
{
  g_return_if_fail (lock);

  if (!g_threads_got_initialized)
    return;

  g_static_mutex_lock (&lock->mutex);
  lock->want_to_write++;
  while (lock->write || lock->read_counter)
    g_static_rw_lock_wait (&lock->write_cond, &lock->mutex);
  lock->want_to_write--;
  lock->write = TRUE;
  g_static_mutex_unlock (&lock->mutex);
}

gboolean 
g_static_rw_lock_writer_trylock (GStaticRWLock* lock)
{
  gboolean ret_val = FALSE;

  g_return_val_if_fail (lock, FALSE);
  
  if (!g_threads_got_initialized)
    return TRUE;

  g_static_mutex_lock (&lock->mutex);
  if (!lock->write && !lock->read_counter)
    {
      lock->write = TRUE;
      ret_val = TRUE;
    }
  g_static_mutex_unlock (&lock->mutex);
  return ret_val;
}

void 
g_static_rw_lock_writer_unlock (GStaticRWLock* lock)
{
  g_return_if_fail (lock);
  
  if (!g_threads_got_initialized)
    return;

  g_static_mutex_lock (&lock->mutex);
  lock->write = FALSE; 
  g_static_rw_lock_signal (lock);
  g_static_mutex_unlock (&lock->mutex);
}

void 
g_static_rw_lock_free (GStaticRWLock* lock)
{
  g_return_if_fail (lock);
  
  if (lock->read_cond)
    {
      g_cond_free (lock->read_cond);
      lock->read_cond = NULL;
    }
  if (lock->write_cond)
    {
      g_cond_free (lock->write_cond);
      lock->write_cond = NULL;
    }
  g_static_mutex_free (&lock->mutex);
}
