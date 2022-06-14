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

/* 
 * MT safe
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "glib.h"


#ifndef DISABLE_MEM_POOLS
struct _GAllocator /* from gmem.c */
{
  gchar         *name;
  guint16        n_preallocs;
  guint          is_unused : 1;
  guint          type : 4;
  GAllocator    *last;
  GMemChunk     *mem_chunk;
  GList		*free_lists; /* implementation specific */
};

static GAllocator	*current_allocator = NULL;
G_LOCK_DEFINE_STATIC (current_allocator);

/* HOLDS: current_allocator_lock */
static void
g_list_validate_allocator (GAllocator *allocator)
{
  g_return_if_fail (allocator != NULL);
  g_return_if_fail (allocator->is_unused == TRUE);

  if (allocator->type != G_ALLOCATOR_LIST)
    {
      allocator->type = G_ALLOCATOR_LIST;
      if (allocator->mem_chunk)
	{
	  g_mem_chunk_destroy (allocator->mem_chunk);
	  allocator->mem_chunk = NULL;
	}
    }

  if (!allocator->mem_chunk)
    {
      allocator->mem_chunk = g_mem_chunk_new (allocator->name,
					      sizeof (GList),
					      sizeof (GList) * allocator->n_preallocs,
					      G_ALLOC_ONLY);
      allocator->free_lists = NULL;
    }

  allocator->is_unused = FALSE;
}

void
g_list_push_allocator(GAllocator *allocator)
{
  G_LOCK (current_allocator);
  g_list_validate_allocator (allocator);
  allocator->last = current_allocator;
  current_allocator = allocator;
  G_UNLOCK (current_allocator);
}

void
g_list_pop_allocator (void)
{
  G_LOCK (current_allocator);
  if (current_allocator)
    {
      GAllocator *allocator;

      allocator = current_allocator;
      current_allocator = allocator->last;
      allocator->last = NULL;
      allocator->is_unused = TRUE;
    }
  G_UNLOCK (current_allocator);
}

static inline GList*
_g_list_alloc (void)
{
  GList *list;

  G_LOCK (current_allocator);
  if (!current_allocator)
    {
      GAllocator *allocator = g_allocator_new ("GLib default GList allocator",
					       128);
      g_list_validate_allocator (allocator);
      allocator->last = NULL;
      current_allocator = allocator;
    }
  if (!current_allocator->free_lists)
    {
      list = g_chunk_new (GList, current_allocator->mem_chunk);
      list->data = NULL;
    }
  else
    {
      if (current_allocator->free_lists->data)
	{
	  list = current_allocator->free_lists->data;
	  current_allocator->free_lists->data = list->next;
	  list->data = NULL;
	}
      else
	{
	  list = current_allocator->free_lists;
	  current_allocator->free_lists = list->next;
	}
    }
  G_UNLOCK (current_allocator);
  list->next = NULL;
  list->prev = NULL;
  
  return list;
}

GList*
g_list_alloc (void)
{
  return _g_list_alloc ();
}

void
g_list_free (GList *list)
{
  if (list)
    {
      GList *last_node = list;

#ifdef ENABLE_GC_FRIENDLY
      while (last_node->next)
        {
          last_node->data = NULL;
          last_node->prev = NULL;
          last_node = last_node->next;
        }
      last_node->data = NULL;
      last_node->prev = NULL;
#else /* !ENABLE_GC_FRIENDLY */
      list->data = list->next;  
#endif /* ENABLE_GC_FRIENDLY */

      G_LOCK (current_allocator);
      last_node->next = current_allocator->free_lists;
      current_allocator->free_lists = list;
      G_UNLOCK (current_allocator);
    }
}

static inline void
_g_list_free_1 (GList *list)
{
  if (list)
    {
      list->data = NULL;  

#ifdef ENABLE_GC_FRIENDLY
      list->prev = NULL;  
#endif /* ENABLE_GC_FRIENDLY */

      G_LOCK (current_allocator);
      list->next = current_allocator->free_lists;
      current_allocator->free_lists = list;
      G_UNLOCK (current_allocator);
    }
}

void
g_list_free_1 (GList *list)
{
  _g_list_free_1 (list);
}

#else /* DISABLE_MEM_POOLS */

#define _g_list_alloc g_list_alloc
GList*
g_list_alloc (void)
{
  GList *list;
  
  list = g_new0 (GList, 1);
  
  return list;
}

void
g_list_free (GList *list)
{
  GList *last;
  
  while (list)
    {
      last = list;
      list = list->next;
      g_free (last);
    }
}

#define _g_list_free_1 g_list_free_1
void
g_list_free_1 (GList *list)
{
  g_free (list);
}

#endif

GList*
g_list_append (GList	*list,
	       gpointer	 data)
{
  GList *new_list;
  GList *last;
  
  new_list = _g_list_alloc ();
  new_list->data = data;
  
  if (list)
    {
      last = g_list_last (list);
      /* g_assert (last != NULL); */
      last->next = new_list;
      new_list->prev = last;

      return list;
    }
  else
    return new_list;
}

GList*
g_list_prepend (GList	 *list,
		gpointer  data)
{
  GList *new_list;
  
  new_list = _g_list_alloc ();
  new_list->data = data;
  
  if (list)
    {
      if (list->prev)
	{
	  list->prev->next = new_list;
	  new_list->prev = list->prev;
	}
      list->prev = new_list;
      new_list->next = list;
    }
  
  return new_list;
}

GList*
g_list_insert (GList	*list,
	       gpointer	 data,
	       gint	 position)
{
  GList *new_list;
  GList *tmp_list;
  
  if (position < 0)
    return g_list_append (list, data);
  else if (position == 0)
    return g_list_prepend (list, data);
  
  tmp_list = g_list_nth (list, position);
  if (!tmp_list)
    return g_list_append (list, data);
  
  new_list = _g_list_alloc ();
  new_list->data = data;
  
  if (tmp_list->prev)
    {
      tmp_list->prev->next = new_list;
      new_list->prev = tmp_list->prev;
    }
  new_list->next = tmp_list;
  tmp_list->prev = new_list;
  
  if (tmp_list == list)
    return new_list;
  else
    return list;
}

GList*
g_list_insert_before (GList   *list,
		      GList   *sibling,
		      gpointer data)
{
  if (!list)
    {
      list = g_list_alloc ();
      list->data = data;
      g_return_val_if_fail (sibling == NULL, list);
      return list;
    }
  else if (sibling)
    {
      GList *node;

      node = g_list_alloc ();
      node->data = data;
      if (sibling->prev)
	{
	  node->prev = sibling->prev;
	  node->prev->next = node;
	  node->next = sibling;
	  sibling->prev = node;
	  return list;
	}
      else
	{
	  node->next = sibling;
	  sibling->prev = node;
	  g_return_val_if_fail (sibling == list, node);
	  return node;
	}
    }
  else
    {
      GList *last;

      last = list;
      while (last->next)
	last = last->next;

      last->next = g_list_alloc ();
      last->next->data = data;
      last->next->prev = last;

      return list;
    }
}

GList *
g_list_concat (GList *list1, GList *list2)
{
  GList *tmp_list;
  
  if (list2)
    {
      tmp_list = g_list_last (list1);
      if (tmp_list)
	tmp_list->next = list2;
      else
	list1 = list2;
      list2->prev = tmp_list;
    }
  
  return list1;
}

GList*
g_list_remove (GList	     *list,
	       gconstpointer  data)
{
  GList *tmp;
  
  tmp = list;
  while (tmp)
    {
      if (tmp->data != data)
	tmp = tmp->next;
      else
	{
	  if (tmp->prev)
	    tmp->prev->next = tmp->next;
	  if (tmp->next)
	    tmp->next->prev = tmp->prev;
	  
	  if (list == tmp)
	    list = list->next;
	  
	  _g_list_free_1 (tmp);
	  
	  break;
	}
    }
  return list;
}

GList*
g_list_remove_all (GList	*list,
		   gconstpointer data)
{
  GList *tmp = list;

  while (tmp)
    {
      if (tmp->data != data)
	tmp = tmp->next;
      else
	{
	  GList *next = tmp->next;

	  if (tmp->prev)
	    tmp->prev->next = next;
	  else
	    list = next;
	  if (next)
	    next->prev = tmp->prev;

	  _g_list_free_1 (tmp);
	  tmp = next;
	}
    }
  return list;
}

static inline GList*
_g_list_remove_link (GList *list,
		     GList *link)
{
  if (link)
    {
      if (link->prev)
	link->prev->next = link->next;
      if (link->next)
	link->next->prev = link->prev;
      
      if (link == list)
	list = list->next;
      
      link->next = NULL;
      link->prev = NULL;
    }
  
  return list;
}

GList*
g_list_remove_link (GList *list,
		    GList *link)
{
  return _g_list_remove_link (list, link);
}

GList*
g_list_delete_link (GList *list,
		    GList *link)
{
  list = _g_list_remove_link (list, link);
  _g_list_free_1 (link);

  return list;
}

GList*
g_list_copy (GList *list)
{
  GList *new_list = NULL;

  if (list)
    {
      GList *last;

      new_list = _g_list_alloc ();
      new_list->data = list->data;
      last = new_list;
      list = list->next;
      while (list)
	{
	  last->next = _g_list_alloc ();
	  last->next->prev = last;
	  last = last->next;
	  last->data = list->data;
	  list = list->next;
	}
    }

  return new_list;
}

GList*
g_list_reverse (GList *list)
{
  GList *last;
  
  last = NULL;
  while (list)
    {
      last = list;
      list = last->next;
      last->next = last->prev;
      last->prev = list;
    }
  
  return last;
}

GList*
g_list_nth (GList *list,
	    guint  n)
{
  while ((n-- > 0) && list)
    list = list->next;
  
  return list;
}

GList*
g_list_nth_prev (GList *list,
		 guint  n)
{
  while ((n-- > 0) && list)
    list = list->prev;
  
  return list;
}

gpointer
g_list_nth_data (GList     *list,
		 guint      n)
{
  while ((n-- > 0) && list)
    list = list->next;
  
  return list ? list->data : NULL;
}

GList*
g_list_find (GList         *list,
	     gconstpointer  data)
{
  while (list)
    {
      if (list->data == data)
	break;
      list = list->next;
    }
  
  return list;
}

GList*
g_list_find_custom (GList         *list,
		    gconstpointer  data,
		    GCompareFunc   func)
{
  g_return_val_if_fail (func != NULL, list);

  while (list)
    {
      if (! func (list->data, data))
	return list;
      list = list->next;
    }

  return NULL;
}


gint
g_list_position (GList *list,
		 GList *link)
{
  gint i;

  i = 0;
  while (list)
    {
      if (list == link)
	return i;
      i++;
      list = list->next;
    }

  return -1;
}

gint
g_list_index (GList         *list,
	      gconstpointer  data)
{
  gint i;

  i = 0;
  while (list)
    {
      if (list->data == data)
	return i;
      i++;
      list = list->next;
    }

  return -1;
}

GList*
g_list_last (GList *list)
{
  if (list)
    {
      while (list->next)
	list = list->next;
    }
  
  return list;
}

GList*
g_list_first (GList *list)
{
  if (list)
    {
      while (list->prev)
	list = list->prev;
    }
  
  return list;
}

guint
g_list_length (GList *list)
{
  guint length;
  
  length = 0;
  while (list)
    {
      length++;
      list = list->next;
    }
  
  return length;
}

void
g_list_foreach (GList	 *list,
		GFunc	  func,
		gpointer  user_data)
{
  while (list)
    {
      GList *next = list->next;
      (*func) (list->data, user_data);
      list = next;
    }
}


GList*
g_list_insert_sorted (GList        *list,
                      gpointer      data,
                      GCompareFunc  func)
{
  GList *tmp_list = list;
  GList *new_list;
  gint cmp;

  g_return_val_if_fail (func != NULL, list);
  
  if (!list) 
    {
      new_list = _g_list_alloc ();
      new_list->data = data;
      return new_list;
    }
  
  cmp = (*func) (data, tmp_list->data);
  
  while ((tmp_list->next) && (cmp > 0))
    {
      tmp_list = tmp_list->next;
      cmp = (*func) (data, tmp_list->data);
    }

  new_list = _g_list_alloc ();
  new_list->data = data;

  if ((!tmp_list->next) && (cmp > 0))
    {
      tmp_list->next = new_list;
      new_list->prev = tmp_list;
      return list;
    }
   
  if (tmp_list->prev)
    {
      tmp_list->prev->next = new_list;
      new_list->prev = tmp_list->prev;
    }
  new_list->next = tmp_list;
  tmp_list->prev = new_list;
 
  if (tmp_list == list)
    return new_list;
  else
    return list;
}

static GList *
g_list_sort_merge (GList     *l1, 
		   GList     *l2,
		   GFunc     compare_func,
		   gboolean  use_data,
		   gpointer  user_data)
{
  GList list, *l, *lprev;
  gint cmp;

  l = &list; 
  lprev = NULL;

  while (l1 && l2)
    {
      if (use_data)
	cmp = ((GCompareDataFunc) compare_func) (l1->data, l2->data, user_data);
      else
	cmp = ((GCompareFunc) compare_func) (l1->data, l2->data);

      if (cmp <= 0)
        {
	  l->next = l1;
	  l = l->next;
	  l->prev = lprev; 
	  lprev = l;
	  l1 = l1->next;
        } 
      else 
	{
	  l->next = l2;
	  l = l->next;
	  l->prev = lprev; 
	  lprev = l;
	  l2 = l2->next;
        }
    }
  l->next = l1 ? l1 : l2;
  l->next->prev = l;

  return list.next;
}

static GList* 
g_list_sort_real (GList    *list,
		  GFunc     compare_func,
		  gboolean  use_data,
		  gpointer  user_data)
{
  GList *l1, *l2;
  
  if (!list) 
    return NULL;
  if (!list->next) 
    return list;
  
  l1 = list; 
  l2 = list->next;

  while ((l2 = l2->next) != NULL)
    {
      if ((l2 = l2->next) == NULL) 
	break;
      l1 = l1->next;
    }
  l2 = l1->next; 
  l1->next = NULL; 

  return g_list_sort_merge (g_list_sort_real (list, compare_func, use_data, user_data),
			    g_list_sort_real (l2, compare_func, use_data, user_data),
			    compare_func,
			    use_data,
			    user_data);
}

GList *
g_list_sort (GList        *list,
	     GCompareFunc  compare_func)
{
  return g_list_sort_real (list, (GFunc) compare_func, FALSE, NULL);
			    
}

GList *
g_list_sort_with_data (GList            *list,
		       GCompareDataFunc  compare_func,
		       gpointer          user_data)
{
  return g_list_sort_real (list, (GFunc) compare_func, TRUE, user_data);
}

static GList* 
g_list_sort2 (GList       *list,
	      GCompareFunc compare_func)
{
  GSList *runs = NULL;
  GList *tmp;

  /* Degenerate case.  */
  if (!list) return NULL;

  /* Assume: list = [12,2,4,11,2,4,6,1,1,12].  */
  for (tmp = list; tmp; )
    {
      GList *tmp2;
      for (tmp2 = tmp;
	   tmp2->next && compare_func (tmp2->data, tmp2->next->data) <= 0;
	   tmp2 = tmp2->next)
	/* Nothing */;
      runs = g_slist_append (runs, tmp);
      tmp = tmp2->next;
      tmp2->next = NULL;
    }
  /* Now: runs = [[12],[2,4,11],[2,4,6],[1,1,12]].  */
  
  while (runs->next)
    {
      /* We have more than one run.  Merge pairwise.  */
      GSList *dst, *src, *dstprev = NULL;
      dst = src = runs;
      while (src && src->next)
	{
	  dst->data = g_list_sort_merge (src->data,
					 src->next->data,
					 (GFunc) compare_func,
					 FALSE, NULL);
	  dstprev = dst;
	  dst = dst->next;
	  src = src->next->next;
	}

      /* If number of runs was odd, just keep the last.  */
      if (src)
	{
	  dst->data = src->data;
	  dstprev = dst;
	  dst = dst->next;
	}

      dstprev->next = NULL;
      g_slist_free (dst);
    }

  /* After 1st loop: runs = [[2,4,11,12],[1,1,2,4,6,12]].  */
  /* After 2nd loop: runs = [[1,1,2,2,4,4,6,11,12,12]].  */

  list = runs->data;
  g_slist_free (runs);
  return list;
}
