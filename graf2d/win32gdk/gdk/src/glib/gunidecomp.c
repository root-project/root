/* decomp.c - Character decomposition.
 *
 *  Copyright (C) 1999, 2000 Tom Tromey
 *  Copyright 2000 Red Hat, Inc.
 *
 * The Gnome Library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * The Gnome Library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with the Gnome Library; see the file COPYING.LIB.  If not,
 * write to the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 *   Boston, MA 02111-1307, USA.
 */

#include "glib.h"
#include "gunidecomp.h"
#include "gunicomp.h"

#include <config.h>

#include <stdlib.h>

/* We cheat a bit and cast type values to (char *).  We detect these
   using the &0xff trick.  */
#define CC(Page, Char) \
  ((((GPOINTER_TO_INT(combining_class_table[Page])) & 0xff) \
    == GPOINTER_TO_INT(combining_class_table[Page])) \
   ? GPOINTER_TO_INT(combining_class_table[Page]) \
   : (combining_class_table[Page][Char]))

#define COMBINING_CLASS(Char) \
     (((Char) > (G_UNICODE_LAST_CHAR)) ? 0 : CC((Char) >> 8, (Char) & 0xff))

/**
 * g_unicode_canonical_ordering:
 * @string: a UCS-4 encoded string.
 * @len: the maximum length of @string to use.
 *
 * Computes the canonical ordering of a string in-place.  
 * This rearranges decomposed characters in the string 
 * according to their combining classes.  See the Unicode 
 * manual for more information. 
 **/
void
g_unicode_canonical_ordering (gunichar *string,
			      gsize     len)
{
  gsize i;
  int swap = 1;

  while (swap)
    {
      int last;
      swap = 0;
      last = COMBINING_CLASS (string[0]);
      for (i = 0; i < len - 1; ++i)
	{
	  int next = COMBINING_CLASS (string[i + 1]);
	  if (next != 0 && last > next)
	    {
	      gsize j;
	      /* Percolate item leftward through string.  */
	      for (j = i; j > 0; --j)
		{
		  gunichar t;
		  if (COMBINING_CLASS (string[j]) <= next)
		    break;
		  t = string[j + 1];
		  string[j + 1] = string[j];
		  string[j] = t;
		  swap = 1;
		}
	      /* We're re-entering the loop looking at the old
		 character again.  */
	      next = last;
	    }
	  last = next;
	}
    }
}

static guchar *
find_decomposition (gunichar ch,
		    gboolean compat)
{
  int start = 0;
  int end = G_N_ELEMENTS (decomp_table);
  
  if (ch >= decomp_table[start].ch &&
      ch <= decomp_table[end - 1].ch)
    {
      while (TRUE)
	{
	  int half = (start + end) / 2;
	  if (ch == decomp_table[half].ch)
	    {
	      int offset;

	      if (compat)
		{
		  offset = decomp_table[half].compat_offset;
		  if (offset == 0xff)
		    offset = decomp_table[half].canon_offset;
		}
	      else
		{
		  offset = decomp_table[half].canon_offset;
		  if (offset == 0xff)
		    return NULL;
		}
	      
	      return decomp_table[half].expansion + offset;
	    }
	  else if (half == start)
	    break;
	  else if (ch > decomp_table[half].ch)
	    start = half;
	  else
	    end = half;
	}
    }

  return NULL;
}

/**
 * g_unicode_canonical_decomposition:
 * @ch: a Unicode character.
 * @result_len: location to store the length of the return value.
 *
 * Computes the canonical decomposition of a Unicode character.  
 * 
 * Return value: a newly allocated string of Unicode characters.
 *   @result_len is set to the resulting length of the string.
 **/
gunichar *
g_unicode_canonical_decomposition (gunichar ch,
				   gsize   *result_len)
{
  guchar *decomp = find_decomposition (ch, FALSE);
  gunichar *r;

  if (decomp)
    {
      /* Found it.  */
      int i, len;
      /* We store as a double-nul terminated string.  */
      for (len = 0; (decomp[len] || decomp[len + 1]);
	   len += 2)
	;
      
      /* We've counted twice as many bytes as there are
	 characters.  */
      *result_len = len / 2;
      r = malloc (len / 2 * sizeof (gunichar));
      
      for (i = 0; i < len; i += 2)
	{
	  r[i / 2] = (decomp[i] << 8 | decomp[i + 1]);
	}
    }
  else
    {
      /* Not in our table.  */
      r = malloc (sizeof (gunichar));
      *r = ch;
      *result_len = 1;
    }

  /* Supposedly following the Unicode 2.1.9 table means that the
     decompositions come out in canonical order.  I haven't tested
     this, but we rely on it here.  */
  return r;
}

#define CI(Page, Char) \
  ((((GPOINTER_TO_INT(compose_table[Page])) & 0xff) \
    == GPOINTER_TO_INT(compose_table[Page])) \
   ? GPOINTER_TO_INT(compose_table[Page]) \
   : (compose_table[Page][Char]))

#define COMPOSE_INDEX(Char) \
     (((Char) > (G_UNICODE_LAST_CHAR)) ? 0 : CI((Char) >> 8, (Char) & 0xff))

gboolean
combine (gunichar  a,
	 gunichar  b,
	 gunichar *result)
{
  gushort index_a, index_b;

  index_a = COMPOSE_INDEX(a);
  if (index_a >= COMPOSE_FIRST_SINGLE_START && index_a < COMPOSE_SECOND_START)
    {
      if (b == compose_first_single[index_a - COMPOSE_FIRST_SINGLE_START][0])
	{
	  *result = compose_first_single[index_a - COMPOSE_FIRST_SINGLE_START][1];
	  return TRUE;
	}
      else
	return FALSE;
    }
  
  index_b = COMPOSE_INDEX(b);
  if (index_b >= COMPOSE_SECOND_SINGLE_START)
    {
      if (a == compose_second_single[index_b - COMPOSE_SECOND_SINGLE_START][0])
	{
	  *result = compose_second_single[index_b - COMPOSE_SECOND_SINGLE_START][1];
	  return TRUE;
	}
      else
	return FALSE;
    }

  if (index_a >= COMPOSE_FIRST_START && index_a < COMPOSE_FIRST_SINGLE_START &&
      index_b >= COMPOSE_SECOND_START && index_a < COMPOSE_SECOND_SINGLE_START)
    {
      gunichar res = compose_array[index_a - COMPOSE_FIRST_START][index_b - COMPOSE_SECOND_START];

      if (res)
	{
	  *result = res;
	  return TRUE;
	}
    }

  return FALSE;
}

gunichar *
_g_utf8_normalize_wc (const gchar    *str,
		      gssize          max_len,
		      GNormalizeMode  mode)
{
  gsize n_wc;
  gunichar *wc_buffer;
  const char *p;
  gsize last_start;
  gboolean do_compat = (mode == G_NORMALIZE_NFKC ||
			mode == G_NORMALIZE_NFKD);
  gboolean do_compose = (mode == G_NORMALIZE_NFC ||
			 mode == G_NORMALIZE_NFKC);

  n_wc = 0;
  p = str;
  while ((max_len < 0 || p < str + max_len) && *p)
    {
      gunichar wc = g_utf8_get_char (p);

      guchar *decomp = find_decomposition (wc, do_compat);

      if (decomp)
	{
	  int len;
	  /* We store as a double-nul terminated string.  */
	  for (len = 0; (decomp[len] || decomp[len + 1]);
	       len += 2)
	    ;
	  n_wc += len / 2;
	}
      else
	n_wc++;

      p = g_utf8_next_char (p);
    }

  wc_buffer = g_new (gunichar, n_wc + 1);

  last_start = 0;
  n_wc = 0;
  p = str;
  while ((max_len < 0 || p < str + max_len) && *p)
    {
      gunichar wc = g_utf8_get_char (p);
      guchar *decomp;
      int cc;
      gsize old_n_wc = n_wc;
	  
      decomp = find_decomposition (wc, do_compat);
	  
      if (decomp)
	{
	  int len;
	  /* We store as a double-nul terminated string.  */
	  for (len = 0; (decomp[len] || decomp[len + 1]);
	       len += 2)
	    wc_buffer[n_wc++] = (decomp[len] << 8 | decomp[len + 1]);
	}
      else
	wc_buffer[n_wc++] = wc;

      if (n_wc > 0)
	{
	  cc = COMBINING_CLASS (wc_buffer[old_n_wc]);

	  if (cc == 0)
	    {
	      g_unicode_canonical_ordering (wc_buffer + last_start, n_wc - last_start);
	      last_start = old_n_wc;
	    }
	}
      
      p = g_utf8_next_char (p);
    }

  if (n_wc > 0)
    {
      g_unicode_canonical_ordering (wc_buffer + last_start, n_wc - last_start);
      last_start = n_wc;
    }
	  
  wc_buffer[n_wc] = 0;

  /* All decomposed and reordered */ 


  if (do_compose && n_wc > 0)
    {
      gsize i, j;
      int last_cc = 0;
      last_start = 0;
      
      for (i = 0; i < n_wc; i++)
	{
	  int cc = COMBINING_CLASS (wc_buffer[i]);

	  if (i > 0 &&
	      (last_cc == 0 || last_cc != cc) &&
	      combine (wc_buffer[last_start], wc_buffer[i],
		       &wc_buffer[last_start]))
	    {
	      for (j = i + 1; j < n_wc; j++)
		wc_buffer[j-1] = wc_buffer[j];
	      n_wc--;
	      i--;
	      
	      if (i == last_start)
		last_cc = 0;
	      else
		last_cc = COMBINING_CLASS (wc_buffer[i-1]);
	      
	      continue;
	    }

	  if (cc == 0)
	    last_start = i;

	  last_cc = cc;
	}
    }

  wc_buffer[n_wc] = 0;

  return wc_buffer;
}

/**
 * g_utf8_normalize:
 * @str: a UTF-8 encoded string.
 * @len: length of @str, in bytes, or -1 if @str is nul-terminated.
 * @mode: the type of normalization to perform.
 * 
 * Converts a string into canonical form, standardizing
 * such issues as whether a character with an accent
 * is represented as a base character and combining
 * accent or as a single precomposed character. You
 * should generally call g_utf8_normalize() before
 * comparing two Unicode strings.
 *
 * The normalization mode %G_NORMALIZE_DEFAULT only
 * standardizes differences that do not affect the
 * text content, such as the above-mentioned accent
 * representation. %G_NORMALIZE_ALL also standardizes
 * the "compatibility" characters in Unicode, such
 * as SUPERSCRIPT THREE to the standard forms
 * (in this case DIGIT THREE). Formatting information
 * may be lost but for most text operations such
 * characters should be considered the same.
 * For example, g_utf8_collate() normalizes
 * with %G_NORMALIZE_ALL as its first step.
 *
 * %G_NORMALIZE_DEFAULT_COMPOSE and %G_NORMALIZE_ALL_COMPOSE
 * are like %G_NORMALIZE_DEFAULT and %G_NORMALIZE_ALL,
 * but returned a result with composed forms rather
 * than a maximally decomposed form. This is often
 * useful if you intend to convert the string to
 * a legacy encoding or pass it to a system with
 * less capable Unicode handling.
 * 
 * Return value: a newly allocated string, that is the 
 *   normalized form of @str.
 **/
gchar *
g_utf8_normalize (const gchar    *str,
		  gssize          len,
		  GNormalizeMode  mode)
{
  gunichar *result_wc = _g_utf8_normalize_wc (str, len, mode);
  gchar *result;
  
  result = g_ucs4_to_utf8 (result_wc, -1, NULL, NULL, NULL);
  g_free (result_wc);

  return result;
}
