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

#ifndef __G_QUEUE_H__
#define __G_QUEUE_H__

#include <glib/glist.h>

G_BEGIN_DECLS

typedef struct _GQueue		GQueue;

struct _GQueue
{
  GList *head;
  GList *tail;
  guint  length;
};

/* Queues
 */
GQueue*  g_queue_new            (void);
void     g_queue_free           (GQueue  *queue);
void     g_queue_push_head      (GQueue  *queue,
				 gpointer data);
void     g_queue_push_tail      (GQueue  *queue,
				 gpointer data);
gpointer g_queue_pop_head       (GQueue  *queue);
gpointer g_queue_pop_tail       (GQueue  *queue);
gboolean g_queue_is_empty       (GQueue  *queue);
gpointer g_queue_peek_head      (GQueue  *queue);
gpointer g_queue_peek_tail      (GQueue  *queue);
void     g_queue_push_head_link (GQueue  *queue,
				 GList   *link);
void     g_queue_push_tail_link (GQueue  *queue,
				 GList   *link);
GList*   g_queue_pop_head_link  (GQueue  *queue);
GList*   g_queue_pop_tail_link  (GQueue  *queue);

G_END_DECLS

#endif /* __G_QUEUE_H__ */
