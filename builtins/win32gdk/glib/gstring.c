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

/* 
 * MT safe
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "glib.h"

struct _GStringChunk
{
  GHashTable *const_table;
  GSList     *storage_list;
  gsize       storage_next;    
  gsize       this_size;       
  gsize       default_size;    
};

G_LOCK_DEFINE_STATIC (string_mem_chunk);
static GMemChunk *string_mem_chunk = NULL;

/* Hash Functions.
 */

gboolean
g_str_equal (gconstpointer v1,
	     gconstpointer v2)
{
  const gchar *string1 = v1;
  const gchar *string2 = v2;
  
  return strcmp (string1, string2) == 0;
}

/* 31 bit hash function */
guint
g_str_hash (gconstpointer key)
{
  const char *p = key;
  guint h = *p;

  if (h)
    for (p += 1; *p != '\0'; p++)
      h = (h << 5) - h + *p;

  return h;
}

#define MY_MAXSIZE ((gsize)-1)

static inline gsize
nearest_power (gsize base, gsize num)    
{
  if (num > MY_MAXSIZE / 2)
    {
      return MY_MAXSIZE;
    }
  else
    {
      gsize n = base;

      while (n < num)
	n <<= 1;
      
      return n;
    }
}

/* String Chunks.
 */

GStringChunk*
g_string_chunk_new (gsize default_size)    
{
  GStringChunk *new_chunk = g_new (GStringChunk, 1);
  gsize size = 1;    

  size = nearest_power (1, default_size);

  new_chunk->const_table       = NULL;
  new_chunk->storage_list      = NULL;
  new_chunk->storage_next      = size;
  new_chunk->default_size      = size;
  new_chunk->this_size         = size;

  return new_chunk;
}

void
g_string_chunk_free (GStringChunk *chunk)
{
  GSList *tmp_list;

  g_return_if_fail (chunk != NULL);

  if (chunk->storage_list)
    {
      for (tmp_list = chunk->storage_list; tmp_list; tmp_list = tmp_list->next)
	g_free (tmp_list->data);

      g_slist_free (chunk->storage_list);
    }

  if (chunk->const_table)
    g_hash_table_destroy (chunk->const_table);

  g_free (chunk);
}

gchar*
g_string_chunk_insert (GStringChunk *chunk,
		       const gchar  *string)
{
  gsize len = strlen (string);
  char* pos;

  g_return_val_if_fail (chunk != NULL, NULL);

  if ((chunk->storage_next + len + 1) > chunk->this_size)
    {
      gsize new_size = nearest_power (chunk->default_size, len + 1);

      chunk->storage_list = g_slist_prepend (chunk->storage_list,
					     g_new (char, new_size));

      chunk->this_size = new_size;
      chunk->storage_next = 0;
    }

  pos = ((char *) chunk->storage_list->data) + chunk->storage_next;

  strcpy (pos, string);

  chunk->storage_next += len + 1;

  return pos;
}

gchar*
g_string_chunk_insert_const (GStringChunk *chunk,
			     const gchar  *string)
{
  char* lookup;

  g_return_val_if_fail (chunk != NULL, NULL);

  if (!chunk->const_table)
    chunk->const_table = g_hash_table_new (g_str_hash, g_str_equal);

  lookup = (char*) g_hash_table_lookup (chunk->const_table, (gchar *)string);

  if (!lookup)
    {
      lookup = g_string_chunk_insert (chunk, string);
      g_hash_table_insert (chunk->const_table, lookup, lookup);
    }

  return lookup;
}

/* Strings.
 */
static void
g_string_maybe_expand (GString* string,
		       gsize    len) 
{
  if (string->len + len >= string->allocated_len)
    {
      string->allocated_len = nearest_power (1, string->len + len + 1);
      string->str = g_realloc (string->str, string->allocated_len);
    }
}

GString*
g_string_sized_new (gsize dfl_size)    
{
  GString *string;

  G_LOCK (string_mem_chunk);
  if (!string_mem_chunk)
    string_mem_chunk = g_mem_chunk_new ("string mem chunk",
					sizeof (GString),
					1024, G_ALLOC_AND_FREE);

  string = g_chunk_new (GString, string_mem_chunk);
  G_UNLOCK (string_mem_chunk);

  string->allocated_len = 0;
  string->len   = 0;
  string->str   = NULL;

  g_string_maybe_expand (string, MAX (dfl_size, 2));
  string->str[0] = 0;

  return string;
}

GString*
g_string_new (const gchar *init)
{
  GString *string;

  string = g_string_sized_new (init ? strlen (init) + 2 : 2);

  if (init)
    g_string_append (string, init);

  return string;
}

GString*
g_string_new_len (const gchar *init,
                  gssize       len)    
{
  GString *string;

  if (len < 0)
    return g_string_new (init);
  else
    {
      string = g_string_sized_new (len);
      
      if (init)
        g_string_append_len (string, init, len);
      
      return string;
    }
}

gchar*
g_string_free (GString *string,
	       gboolean free_segment)
{
  gchar *segment;

  g_return_val_if_fail (string != NULL, NULL);

  if (free_segment)
    {
      g_free (string->str);
      segment = NULL;
    }
  else
    segment = string->str;

  G_LOCK (string_mem_chunk);
  g_mem_chunk_free (string_mem_chunk, string);
  G_UNLOCK (string_mem_chunk);

  return segment;
}

gboolean
g_string_equal (const GString *v,
                const GString *v2)
{
  gchar *p, *q;
  GString *string1 = (GString *) v;
  GString *string2 = (GString *) v2;
  gsize i = string1->len;    

  if (i != string2->len)
    return FALSE;

  p = string1->str;
  q = string2->str;
  while (i)
    {
      if (*p != *q)
	return FALSE;
      p++;
      q++;
      i--;
    }
  return TRUE;
}

/* 31 bit hash function */
guint
g_string_hash (const GString *str)
{
  const gchar *p = str->str;
  gsize n = str->len;    
  guint h = 0;

  while (n--)
    {
      h = (h << 5) - h + *p;
      p++;
    }

  return h;
}

GString*
g_string_assign (GString     *string,
		 const gchar *rval)
{
  g_return_val_if_fail (string != NULL, NULL);
  g_return_val_if_fail (rval != NULL, string);
  
  g_string_truncate (string, 0);
  g_string_append (string, rval);

  return string;
}

GString*
g_string_truncate (GString *string,
		   gsize    len)    
{
  g_return_val_if_fail (string != NULL, NULL);

  string->len = MIN (len, string->len);
  string->str[string->len] = 0;

  return string;
}

/**
 * g_string_set_size:
 * @string: a #GString
 * @len: the new length
 * 
 * Sets the length of a #GString. If the length is less than
 * the current length, the string will be truncated. If the
 * length is greater than the current length, the contents
 * of the newly added area are undefined. (However, as
 * always, string->str[string->len] will be a nul byte.) 
 * 
 * Return value: @string
 **/
GString*
g_string_set_size (GString *string,
		   gsize    len)    
{
  g_return_val_if_fail (string != NULL, NULL);

  if (len >= string->allocated_len)
    g_string_maybe_expand (string, len - string->len);
  
  string->len = len;
  string->str[len] = 0;

  return string;
}

GString*
g_string_insert_len (GString     *string,
		     gssize       pos,    
		     const gchar *val,
		     gssize       len)    
{
  g_return_val_if_fail (string != NULL, NULL);
  g_return_val_if_fail (val != NULL, string);

  if (len < 0)
    len = strlen (val);

  if (pos < 0)
    pos = string->len;
  else
    g_return_val_if_fail (pos <= string->len, string);
  
  g_string_maybe_expand (string, len);

  /* If we aren't appending at the end, move a hunk
   * of the old string to the end, opening up space
   */
  if (pos < string->len)
    g_memmove (string->str + pos + len, string->str + pos, string->len - pos);
  
  /* insert the new string */
  g_memmove (string->str + pos, val, len);

  string->len += len;

  string->str[string->len] = 0;

  return string;
}

GString*
g_string_append (GString     *string,
		 const gchar *val)
{  
  g_return_val_if_fail (string != NULL, NULL);
  g_return_val_if_fail (val != NULL, string);

  return g_string_insert_len (string, -1, val, -1);
}

GString*
g_string_append_len (GString	 *string,
                     const gchar *val,
                     gssize       len)    
{
  g_return_val_if_fail (string != NULL, NULL);
  g_return_val_if_fail (val != NULL, string);

  return g_string_insert_len (string, -1, val, len);
}

GString*
g_string_append_c (GString *string,
		   gchar    c)
{
  g_return_val_if_fail (string != NULL, NULL);

  return g_string_insert_c (string, -1, c);
}

/**
 * g_string_append_unichar:
 * @string: a #GString
 * @wc: a Unicode character
 * 
 * Converts a Unicode character into UTF-8, and appends it
 * to the string.
 * 
 * Return value: @string
 **/
GString*
g_string_append_unichar (GString  *string,
			 gunichar  wc)
{  
  g_return_val_if_fail (string != NULL, NULL);
  
  return g_string_insert_unichar (string, -1, wc);
}

GString*
g_string_prepend (GString     *string,
		  const gchar *val)
{
  g_return_val_if_fail (string != NULL, NULL);
  g_return_val_if_fail (val != NULL, string);
  
  return g_string_insert_len (string, 0, val, -1);
}

GString*
g_string_prepend_len (GString	  *string,
                      const gchar *val,
                      gssize       len)    
{
  g_return_val_if_fail (string != NULL, NULL);
  g_return_val_if_fail (val != NULL, string);

  return g_string_insert_len (string, 0, val, len);
}

GString*
g_string_prepend_c (GString *string,
		    gchar    c)
{  
  g_return_val_if_fail (string != NULL, NULL);
  
  return g_string_insert_c (string, 0, c);
}

/**
 * g_string_append_unichar:
 * @string: a #GString
 * @wc: a Unicode character
 * 
 * Converts a Unicode character into UTF-8, and prepends it
 * to the string.
 * 
 * Return value: @string
 **/
GString*
g_string_prepend_unichar (GString  *string,
			  gunichar  wc)
{  
  g_return_val_if_fail (string != NULL, NULL);
  
  return g_string_insert_unichar (string, 0, wc);
}

GString*
g_string_insert (GString     *string,
		 gssize       pos,    
		 const gchar *val)
{
  g_return_val_if_fail (string != NULL, NULL);
  g_return_val_if_fail (val != NULL, string);
  if (pos >= 0)
    g_return_val_if_fail (pos <= string->len, string);
  
  return g_string_insert_len (string, pos, val, -1);
}

GString*
g_string_insert_c (GString *string,
		   gssize   pos,    
		   gchar    c)
{
  g_return_val_if_fail (string != NULL, NULL);

  g_string_maybe_expand (string, 1);

  if (pos < 0)
    pos = string->len;
  else
    g_return_val_if_fail (pos <= string->len, string);
  
  /* If not just an append, move the old stuff */
  if (pos < string->len)
    g_memmove (string->str + pos + 1, string->str + pos, string->len - pos);

  string->str[pos] = c;

  string->len += 1;

  string->str[string->len] = 0;

  return string;
}

/**
 * g_string_insert_unichar:
 * @string: a #Gstring
 * @pos: the position at which to insert character, or -1 to
 *       append at the end of the string.
 * @wc: a Unicode character
 * 
 * Converts a Unicode character into UTF-8, and insert it
 * into the string at the given position.
 * 
 * Return value: @string
 **/
GString*
g_string_insert_unichar (GString *string,
			 gssize   pos,    
			 gunichar wc)
{  
  gchar buf[6];
  gint charlen;

  /* We could be somewhat more efficient here by computing
   * the length, adding the space, then converting into that
   * space, by cut-and-pasting the internals of g_unichar_to_utf8.
   */
  g_return_val_if_fail (string != NULL, NULL);

  charlen = g_unichar_to_utf8 (wc, buf);
  return g_string_insert_len (string, pos, buf, charlen);
}

GString*
g_string_erase (GString *string,
		gsize    pos,    
		gsize    len)    
{
  g_return_val_if_fail (string != NULL, NULL);
  g_return_val_if_fail (pos >= 0, string);
  g_return_val_if_fail (pos <= string->len, string);

  if (len < 0)
    len = string->len - pos;
  else
    {
      g_return_val_if_fail (pos + len <= string->len, string);

      if (pos + len < string->len)
	g_memmove (string->str + pos, string->str + pos + len, string->len - (pos + len));
    }

  string->len -= len;
  
  string->str[string->len] = 0;

  return string;
}

/**
 * g_string_ascii_down:
 * @string: a GString
 * 
 * Converts all upper case ASCII letters to lower case ASCII letters.
 * 
 * Return value: passed-in @string pointer, with all the upper case
 *               characters converted to lower case in place, with
 *               semantics that exactly match g_ascii_tolower.
 **/
GString*
g_string_ascii_down (GString *string)
{
  gchar *s;
  gint n = string->len;

  g_return_val_if_fail (string != NULL, NULL);

  s = string->str;

  while (n)
    {
      *s = g_ascii_tolower (*s);
      s++;
      n--;
    }

  return string;
}

/**
 * g_string_ascii_up:
 * @string: a GString
 * 
 * Converts all lower case ASCII letters to upper case ASCII letters.
 * 
 * Return value: passed-in @string pointer, with all the lower case
 *               characters converted to upper case in place, with
 *               semantics that exactly match g_ascii_toupper.
 **/
GString*
g_string_ascii_up (GString *string)
{
  gchar *s;
  gint n = string->len;

  g_return_val_if_fail (string != NULL, NULL);

  s = string->str;

  while (n)
    {
      *s = g_ascii_toupper (*s);
      s++;
      n--;
    }

  return string;
}

GString*
g_string_down (GString *string)
{
  guchar *s;
  glong n = string->len;    

  g_return_val_if_fail (string != NULL, NULL);

  s = (guchar *) string->str;

  while (n)
    {
      if (isupper (*s))
	*s = tolower (*s);
      s++;
      n--;
    }

  return string;
}

GString*
g_string_up (GString *string)
{
  guchar *s;
  glong n = string->len;

  g_return_val_if_fail (string != NULL, NULL);

  s = (guchar *) string->str;

  while (n)
    {
      if (islower (*s))
	*s = toupper (*s);
      s++;
      n--;
    }

  return string;
}

static void
g_string_printfa_internal (GString     *string,
			   const gchar *fmt,
			   va_list      args)
{
  gchar *buffer;

  buffer = g_strdup_vprintf (fmt, args);
  g_string_append (string, buffer);
  g_free (buffer);
}

void
g_string_printf (GString *string,
		 const gchar *fmt,
		 ...)
{
  va_list args;

  g_string_truncate (string, 0);

  va_start (args, fmt);
  g_string_printfa_internal (string, fmt, args);
  va_end (args);
}

void
g_string_printfa (GString *string,
		  const gchar *fmt,
		  ...)
{
  va_list args;

  va_start (args, fmt);
  g_string_printfa_internal (string, fmt, args);
  va_end (args);
}
