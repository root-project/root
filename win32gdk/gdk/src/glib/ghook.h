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

#ifndef __G_HOOK_H__
#define __G_HOOK_H__

#include <gmem.h>

G_BEGIN_DECLS

typedef struct _GHook		GHook;
typedef struct _GHookList	GHookList;

typedef gint		(*GHookCompareFunc)	(GHook		*new_hook,
						 GHook		*sibling);
typedef gboolean	(*GHookFindFunc)	(GHook		*hook,
						 gpointer	 data);
typedef void		(*GHookMarshaller)	(GHook		*hook,
						 gpointer	 data);
typedef gboolean	(*GHookCheckMarshaller)	(GHook		*hook,
						 gpointer	 data);
typedef void		(*GHookFunc)		(gpointer	 data);
typedef gboolean	(*GHookCheckFunc)	(gpointer	 data);
typedef void		(*GHookFreeFunc)	(GHookList      *hook_list,
						 GHook          *hook);

/* Callback maintenance functions
 */
#define G_HOOK_FLAG_USER_SHIFT	(4)
typedef enum
{
  G_HOOK_FLAG_ACTIVE	= 1 << 0,
  G_HOOK_FLAG_IN_CALL	= 1 << 1,
  G_HOOK_FLAG_MASK	= 0x0f
} GHookFlagMask;

#define	G_HOOK_DEFERRED_DESTROY	((GHookFreeFunc) 0x01)

struct _GHookList
{
  guint		 seq_id;
  guint		 hook_size;
  guint		 is_setup : 1;
  GHook		*hooks;
  GMemChunk	*hook_memchunk;
  GHookFreeFunc	 hook_free; /* virtual function */
  GHookFreeFunc	 hook_destroy; /* virtual function */
};

struct _GHook
{
  gpointer	 data;
  GHook		*next;
  GHook		*prev;
  guint		 ref_count;
  guint		 hook_id;
  guint		 flags;
  gpointer	 func;
  GDestroyNotify destroy;
};

#define	G_HOOK_ACTIVE(hook)		((((GHook*) hook)->flags & \
					  G_HOOK_FLAG_ACTIVE) != 0)
#define	G_HOOK_IN_CALL(hook)		((((GHook*) hook)->flags & \
					  G_HOOK_FLAG_IN_CALL) != 0)
#define G_HOOK_IS_VALID(hook)		(((GHook*) hook)->hook_id != 0 && \
					 G_HOOK_ACTIVE (hook))
#define G_HOOK_IS_UNLINKED(hook)	(((GHook*) hook)->next == NULL && \
					 ((GHook*) hook)->prev == NULL && \
					 ((GHook*) hook)->hook_id == 0 && \
					 ((GHook*) hook)->ref_count == 0)

void	 g_hook_list_init		(GHookList		*hook_list,
					 guint			 hook_size);
void	 g_hook_list_clear		(GHookList		*hook_list);
GHook*	 g_hook_alloc			(GHookList		*hook_list);
void	 g_hook_free			(GHookList		*hook_list,
					 GHook			*hook);
void	 g_hook_ref			(GHookList		*hook_list,
					 GHook			*hook);
void	 g_hook_unref			(GHookList		*hook_list,
					 GHook			*hook);
gboolean g_hook_destroy			(GHookList		*hook_list,
					 guint			 hook_id);
void	 g_hook_destroy_link		(GHookList		*hook_list,
					 GHook			*hook);
void	 g_hook_prepend			(GHookList		*hook_list,
					 GHook			*hook);
void	 g_hook_insert_before		(GHookList		*hook_list,
					 GHook			*sibling,
					 GHook			*hook);
void	 g_hook_insert_sorted		(GHookList		*hook_list,
					 GHook			*hook,
					 GHookCompareFunc	 func);
GHook*	 g_hook_get			(GHookList		*hook_list,
					 guint			 hook_id);
GHook*	 g_hook_find			(GHookList		*hook_list,
					 gboolean		 need_valids,
					 GHookFindFunc		 func,
					 gpointer		 data);
GHook*	 g_hook_find_data		(GHookList		*hook_list,
					 gboolean		 need_valids,
					 gpointer		 data);
GHook*	 g_hook_find_func		(GHookList		*hook_list,
					 gboolean		 need_valids,
					 gpointer		 func);
GHook*	 g_hook_find_func_data		(GHookList		*hook_list,
					 gboolean		 need_valids,
					 gpointer		 func,
					 gpointer		 data);
/* return the first valid hook, and increment its reference count */
GHook*	 g_hook_first_valid		(GHookList		*hook_list,
					 gboolean		 may_be_in_call);
/* return the next valid hook with incremented reference count, and
 * decrement the reference count of the original hook
 */
GHook*	 g_hook_next_valid		(GHookList		*hook_list,
					 GHook			*hook,
					 gboolean		 may_be_in_call);

/* GHookCompareFunc implementation to insert hooks sorted by their id */
gint	 g_hook_compare_ids		(GHook			*new_hook,
					 GHook			*sibling);

/* convenience macros */
#define	 g_hook_append( hook_list, hook )  \
     g_hook_insert_before ((hook_list), NULL, (hook))

/* invoke all valid hooks with the (*GHookFunc) signature.
 */
void	 g_hook_list_invoke		(GHookList		*hook_list,
					 gboolean		 may_recurse);
/* invoke all valid hooks with the (*GHookCheckFunc) signature,
 * and destroy the hook if FALSE is returned.
 */
void	 g_hook_list_invoke_check	(GHookList		*hook_list,
					 gboolean		 may_recurse);
/* invoke a marshaller on all valid hooks.
 */
void	 g_hook_list_marshal		(GHookList		*hook_list,
					 gboolean		 may_recurse,
					 GHookMarshaller	 marshaller,
					 gpointer		 data);
void	 g_hook_list_marshal_check	(GHookList		*hook_list,
					 gboolean		 may_recurse,
					 GHookCheckMarshaller	 marshaller,
					 gpointer		 data);

G_END_DECLS

#endif /* __G_HOOK_H__ */

