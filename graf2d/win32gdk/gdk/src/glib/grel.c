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
 * License along with this library; if not, write to the Free
 * Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
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

#include "glib.h"
#include <stdarg.h>
#include <string.h>

typedef struct _GRealTuples        GRealTuples;

struct _GRelation
{
  gint fields;
  gint current_field;
  
  GHashTable   *all_tuples;
  GHashTable  **hashed_tuple_tables;
  GMemChunk    *tuple_chunk;
  
  gint count;
};

struct _GRealTuples
{
  gint      len;
  gint      width;
  gpointer *data;
};

static gboolean
tuple_equal_2 (gconstpointer v_a,
	       gconstpointer v_b)
{
  gpointer* a = (gpointer*) v_a;
  gpointer* b = (gpointer*) v_b;
  
  return a[0] == b[0] && a[1] == b[1];
}

static guint
tuple_hash_2 (gconstpointer v_a)
{
  gpointer* a = (gpointer*) v_a;
  
  return (gulong)a[0] ^ (gulong)a[1];
}

static GHashFunc
tuple_hash (gint fields)
{
  switch (fields)
    {
    case 2:
      return tuple_hash_2;
    default:
      g_error ("no tuple hash for %d", fields);
    }
  
  return NULL;
}

static GEqualFunc
tuple_equal (gint fields)
{
  switch (fields)
    {
    case 2:
      return tuple_equal_2;
    default:
      g_error ("no tuple equal for %d", fields);
    }
  
  return NULL;
}

GRelation*
g_relation_new (gint fields)
{
  GRelation* rel = g_new0 (GRelation, 1);
  
  rel->fields = fields;
  rel->tuple_chunk = g_mem_chunk_new ("Relation Chunk",
				      fields * sizeof (gpointer),
				      fields * sizeof (gpointer) * 128,
				      G_ALLOC_AND_FREE);
  rel->all_tuples = g_hash_table_new (tuple_hash (fields), tuple_equal (fields));
  rel->hashed_tuple_tables = g_new0 (GHashTable*, fields);
  
  return rel;
}

static void
g_relation_free_array (gpointer key, gpointer value, gpointer user_data)
{
  g_hash_table_destroy ((GHashTable*) value);
}

void
g_relation_destroy (GRelation *relation)
{
  gint i;
  
  if (relation)
    {
      g_hash_table_destroy (relation->all_tuples);
      g_mem_chunk_destroy (relation->tuple_chunk);
      
      for (i = 0; i < relation->fields; i += 1)
	{
	  if (relation->hashed_tuple_tables[i])
	    {
	      g_hash_table_foreach (relation->hashed_tuple_tables[i], g_relation_free_array, NULL);
	      g_hash_table_destroy (relation->hashed_tuple_tables[i]);
	    }
	}
      
      g_free (relation->hashed_tuple_tables);
      g_free (relation);
    }
}

void
g_relation_index (GRelation   *relation,
		  gint         field,
		  GHashFunc    hash_func,
		  GEqualFunc   key_equal_func)
{
  g_return_if_fail (relation != NULL);
  
  g_return_if_fail (relation->count == 0 && relation->hashed_tuple_tables[field] == NULL);
  
  relation->hashed_tuple_tables[field] = g_hash_table_new (hash_func, key_equal_func);
}

void
g_relation_insert (GRelation   *relation,
		   ...)
{
  gpointer* tuple = g_chunk_new (gpointer, relation->tuple_chunk);
  va_list args;
  gint i;
  
  va_start(args, relation);
  
  for (i = 0; i < relation->fields; i += 1)
    tuple[i] = va_arg(args, gpointer);
  
  va_end(args);
  
  g_hash_table_insert (relation->all_tuples, tuple, tuple);
  
  relation->count += 1;
  
  for (i = 0; i < relation->fields; i += 1)
    {
      GHashTable *table;
      gpointer    key;
      GHashTable *per_key_table;
      
      table = relation->hashed_tuple_tables[i];
      
      if (table == NULL)
	continue;
      
      key = tuple[i];
      per_key_table = g_hash_table_lookup (table, key);
      
      if (per_key_table == NULL)
	{
	  per_key_table = g_hash_table_new (tuple_hash (relation->fields), tuple_equal (relation->fields));
	  g_hash_table_insert (table, key, per_key_table);
	}
      
      g_hash_table_insert (per_key_table, tuple, tuple);
    }
}

static void
g_relation_delete_tuple (gpointer tuple_key,
			 gpointer tuple_value,
			 gpointer user_data)
{
  gpointer      *tuple = (gpointer*) tuple_value;
  GRelation     *rel = (GRelation *) user_data;
  gint           j;
  
  g_assert (tuple_key == tuple_value);
  
  for (j = 0; j < rel->fields; j += 1)
    {
      GHashTable *one_table = rel->hashed_tuple_tables[j];
      gpointer    one_key;
      GHashTable *per_key_table;
      
      if (one_table == NULL)
	continue;
      
      if (j == rel->current_field)
	/* can't delete from the table we're foreaching in */
	continue;
      
      one_key = tuple[j];
      
      per_key_table = g_hash_table_lookup (one_table, one_key);
      
      g_hash_table_remove (per_key_table, tuple);
    }
  
  g_hash_table_remove (rel->all_tuples, tuple);
  
  rel->count -= 1;
}

gint
g_relation_delete  (GRelation     *relation,
		    gconstpointer  key,
		    gint           field)
{
  GHashTable *table = relation->hashed_tuple_tables[field];
  GHashTable *key_table;
  gint        count = relation->count;
  
  g_return_val_if_fail (relation != NULL, 0);
  g_return_val_if_fail (table != NULL, 0);
  
  key_table = g_hash_table_lookup (table, key);
  
  if (!key_table)
    return 0;
  
  relation->current_field = field;
  
  g_hash_table_foreach (key_table, g_relation_delete_tuple, relation);
  
  g_hash_table_remove (table, key);
  
  g_hash_table_destroy (key_table);
  
  /* @@@ FIXME: Remove empty hash tables. */
  
  return count - relation->count;
}

static void
g_relation_select_tuple (gpointer tuple_key,
			 gpointer tuple_value,
			 gpointer user_data)
{
  gpointer    *tuple = (gpointer*) tuple_value;
  GRealTuples *tuples = (GRealTuples*) user_data;
  gint stride = sizeof (gpointer) * tuples->width;
  
  g_assert (tuple_key == tuple_value);
  
  memcpy (tuples->data + (tuples->len * tuples->width),
	  tuple,
	  stride);
  
  tuples->len += 1;
}

GTuples*
g_relation_select (GRelation     *relation,
		   gconstpointer  key,
		   gint           field)
{
  GHashTable  *table = relation->hashed_tuple_tables[field];
  GHashTable  *key_table;
  GRealTuples *tuples = g_new0 (GRealTuples, 1);
  gint count;
  
  g_return_val_if_fail (relation != NULL, NULL);
  g_return_val_if_fail (table != NULL, NULL);
  
  key_table = g_hash_table_lookup (table, key);
  
  if (!key_table)
    return (GTuples*)tuples;
  
  count = g_relation_count (relation, key, field);
  
  tuples->data = g_malloc (sizeof (gpointer) * relation->fields * count);
  tuples->width = relation->fields;
  
  g_hash_table_foreach (key_table, g_relation_select_tuple, tuples);
  
  g_assert (count == tuples->len);
  
  return (GTuples*)tuples;
}

gint
g_relation_count (GRelation     *relation,
		  gconstpointer  key,
		  gint           field)
{
  GHashTable  *table = relation->hashed_tuple_tables[field];
  GHashTable  *key_table;
  
  g_return_val_if_fail (relation != NULL, 0);
  g_return_val_if_fail (table != NULL, 0);
  
  key_table = g_hash_table_lookup (table, key);
  
  if (!key_table)
    return 0;
  
  return g_hash_table_size (key_table);
}

gboolean
g_relation_exists (GRelation   *relation, ...)
{
  gpointer* tuple = g_chunk_new (gpointer, relation->tuple_chunk);
  va_list args;
  gint i;
  gboolean result;
  
  va_start(args, relation);
  
  for (i = 0; i < relation->fields; i += 1)
    tuple[i] = va_arg(args, gpointer);
  
  va_end(args);
  
  result = g_hash_table_lookup (relation->all_tuples, tuple) != NULL;
  
  g_mem_chunk_free (relation->tuple_chunk, tuple);
  
  return result;
}

void
g_tuples_destroy (GTuples *tuples0)
{
  GRealTuples *tuples = (GRealTuples*) tuples0;
  
  if (tuples)
    {
      g_free (tuples->data);
      g_free (tuples);
    }
}

gpointer
g_tuples_index (GTuples     *tuples0,
		gint         index,
		gint         field)
{
  GRealTuples *tuples = (GRealTuples*) tuples0;
  
  g_return_val_if_fail (tuples0 != NULL, NULL);
  g_return_val_if_fail (field < tuples->width, NULL);
  
  return tuples->data[index * tuples->width + field];
}

/* Print
 */

static void
g_relation_print_one (gpointer tuple_key,
		      gpointer tuple_value,
		      gpointer user_data)
{
  gint i;
  GString *gstring;
  GRelation* rel = (GRelation*) user_data;
  gpointer* tuples = (gpointer*) tuple_value;

  gstring = g_string_new ("[");
  
  for (i = 0; i < rel->fields; i += 1)
    {
      g_string_printfa (gstring, "%p", tuples[i]);
      
      if (i < (rel->fields - 1))
	g_string_append (gstring, ",");
    }
  
  g_string_append (gstring, "]");
  g_log (g_log_domain_glib, G_LOG_LEVEL_INFO, gstring->str);
  g_string_free (gstring, TRUE);
}

static void
g_relation_print_index (gpointer tuple_key,
			gpointer tuple_value,
			gpointer user_data)
{
  GRelation* rel = (GRelation*) user_data;
  GHashTable* table = (GHashTable*) tuple_value;
  
  g_log (g_log_domain_glib, G_LOG_LEVEL_INFO, "*** key %p", tuple_key);
  
  g_hash_table_foreach (table,
			g_relation_print_one,
			rel);
}

void
g_relation_print (GRelation *relation)
{
  gint i;
  
  g_log (g_log_domain_glib, G_LOG_LEVEL_INFO, "*** all tuples (%d)", relation->count);
  
  g_hash_table_foreach (relation->all_tuples,
			g_relation_print_one,
			relation);
  
  for (i = 0; i < relation->fields; i += 1)
    {
      if (relation->hashed_tuple_tables[i] == NULL)
	continue;
      
      g_log (g_log_domain_glib, G_LOG_LEVEL_INFO, "*** index %d", i);
      
      g_hash_table_foreach (relation->hashed_tuple_tables[i],
			    g_relation_print_index,
			    relation);
    }
  
}
