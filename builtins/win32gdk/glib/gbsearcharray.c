/* GObject - GLib Type, Object, Parameter and Signal Library
 * Copyright (C) 2000-2001 Tim Janik
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
 * You should have received a copy of the GNU Lesser General
 * Public License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place, Suite 330,
 * Boston, MA 02111-1307, USA.
 */
#define G_IMPLEMENT_INLINES 1
#define __G_BSEARCHARRAY_C__
#include "gbsearcharray.h"

#include	<string.h>


/* --- structures --- */
GBSearchArray*
g_bsearch_array_new (guint16             sizeof_node,
		     GBSearchCompareFunc node_cmp_func,
		     GBSearchArrayFlags  flags)
{
  GBSearchArray *barray;

  g_return_val_if_fail (sizeof_node > 0, NULL);
  g_return_val_if_fail (node_cmp_func != NULL, NULL);

  barray = g_new0 (GBSearchArray, 1);
  barray->sizeof_node = sizeof_node;
  barray->cmp_nodes = node_cmp_func;
  barray->flags = flags;

  return barray;
}

void
g_bsearch_array_destroy (GBSearchArray *barray)
{
  g_return_if_fail (barray != NULL);

#if 0
  if (barray->destroy_node)
    while (barray->n_nodes)
      {
	barray->destroy_node (((guint8*) barray->nodes) + (barray->n_nodes - 1) * barray->sizeof_node);
	barray->n_nodes--;
      }
#endif
  g_free (barray->nodes);
  g_free (barray);
}

static inline guint
upper_power2 (guint number)
{
#ifdef	DISABLE_MEM_POOLS
  return number;
#else	/* !DISABLE_MEM_POOLS */
  return number ? 1 << g_bit_storage (number - 1) : 0;
#endif	/* !DISABLE_MEM_POOLS */
}

static inline gpointer
bsearch_array_insert (GBSearchArray *barray,
		      gconstpointer  key_node,
		      gboolean       replace)
{
  gint sizeof_node;
  guint8 *check;
  
  sizeof_node = barray->sizeof_node;
  if (barray->n_nodes == 0)
    {
      guint new_size = barray->sizeof_node;
      
      if (barray->flags & G_BSEARCH_ARRAY_ALIGN_POWER2)
	new_size = upper_power2 (new_size);
      barray->nodes = g_realloc (barray->nodes, new_size);
      barray->n_nodes = 1;
      check = barray->nodes;
    }
  else
    {
      GBSearchCompareFunc cmp_nodes = barray->cmp_nodes;
      guint n_nodes = barray->n_nodes;
      guint8 *nodes = barray->nodes;
      gint cmp;
      guint i;
      
      nodes -= sizeof_node;
      do
	{
	  i = (n_nodes + 1) >> 1;
	  check = nodes + i * sizeof_node;
	  cmp = cmp_nodes (key_node, check);
	  if (cmp > 0)
	    {
	      n_nodes -= i;
	      nodes = check;
	    }
	  else if (cmp < 0)
	    n_nodes = i - 1;
	  else /* if (cmp == 0) */
	    {
	      if (replace)
		{
#if 0
		  if (barray->destroy_node)
		    barray->destroy_node (check);
#endif
		  memcpy (check, key_node, sizeof_node);
		}
	      return check;
	    }
	}
      while (n_nodes);
      /* grow */
      if (cmp > 0)
	check += sizeof_node;
      i = (check - ((guint8*) barray->nodes)) / sizeof_node;
      n_nodes = barray->n_nodes++;
      if (barray->flags & G_BSEARCH_ARRAY_ALIGN_POWER2)
	{
	  guint new_size = upper_power2 (barray->n_nodes * sizeof_node);
	  guint old_size = upper_power2 (n_nodes * sizeof_node);
	  
	  if (new_size != old_size)
	    barray->nodes = g_realloc (barray->nodes, new_size);
	}
      else
	barray->nodes = g_realloc (barray->nodes, barray->n_nodes * sizeof_node);
      check = ((guint8*) barray->nodes) + i * sizeof_node;
      g_memmove (check + sizeof_node, check, (n_nodes - i) * sizeof_node);
    }
  memcpy (check, key_node, sizeof_node);
  
  return check;
}

gpointer
g_bsearch_array_insert (GBSearchArray *barray,
			gconstpointer  key_node,
			gboolean       replace_existing)
{
  g_return_val_if_fail (barray != NULL, NULL);
  g_return_val_if_fail (key_node != NULL, NULL);
  
  return bsearch_array_insert (barray, key_node, replace_existing);
}

void
g_bsearch_array_remove_node (GBSearchArray *barray,
			     gpointer       _node_in_array)
{
  guint8 *nodes, *bound, *node_in_array = _node_in_array;
  guint old_size;
  
  g_return_if_fail (barray != NULL);
  
  nodes = barray->nodes;
  old_size = barray->sizeof_node;
  old_size *= barray->n_nodes;  /* beware of int widths */
  bound = nodes + old_size;
  
  g_return_if_fail (node_in_array >= nodes && node_in_array < bound);

#if 0
  if (barray->destroy_node)
    barray->destroy_node (node_in_array);
#endif
  bound -= barray->sizeof_node;
  barray->n_nodes -= 1;
  g_memmove (node_in_array, node_in_array + barray->sizeof_node, (bound - node_in_array) / barray->sizeof_node);
  
  if ((barray->flags & G_BSEARCH_ARRAY_DEFER_SHRINK) == 0)
    {
      guint new_size = bound - nodes;   /* old_size - barray->sizeof_node */
      
      if (barray->flags & G_BSEARCH_ARRAY_ALIGN_POWER2)
	{
	  new_size = upper_power2 (new_size);
	  old_size = upper_power2 (old_size);
	  if (old_size != new_size)
	    barray->nodes = g_realloc (barray->nodes, new_size);
	}
      else
	barray->nodes = g_realloc (barray->nodes, new_size);
    }
}

void
g_bsearch_array_remove (GBSearchArray *barray,
			gconstpointer  key_node)
{
  gpointer node_in_array;
  
  g_return_if_fail (barray != NULL);
  
  node_in_array = g_bsearch_array_lookup (barray, key_node);
  if (!node_in_array)
    g_warning (G_STRLOC ": unable to remove unexistant node");
  else
    g_bsearch_array_remove_node (barray, node_in_array);
}
