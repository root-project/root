/* GLIB - Library of useful routines for C programming
 * Copyright (C) 1995-1997, 1999  Peter Mattis, Red Hat, Inc.
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
#include "gpattern.h"

#include "gmacros.h"
#include "gmessages.h"
#include "gmem.h"
#include "gutils.h" /* inline hassle */
#include <string.h>


/* --- functions --- */
static inline gboolean
g_pattern_ph_match (const gchar *match_pattern,
		    const gchar *match_string)
{
  register const gchar *pattern, *string;
  register gchar ch;

  pattern = match_pattern;
  string = match_string;

  ch = *pattern;
  pattern++;
  while (ch)
    {
      switch (ch)
	{
	case '?':
	  if (!*string)
	    return FALSE;
	  string++;
	  break;

	case '*':
	  do
	    {
	      ch = *pattern;
	      pattern++;
	      if (ch == '?')
		{
		  if (!*string)
		    return FALSE;
		  string++;
		}
	    }
	  while (ch == '*' || ch == '?');
	  if (!ch)
	    return TRUE;
	  do
	    {
	      while (ch != *string)
		{
		  if (!*string)
		    return FALSE;
		  string++;
		}
	      string++;
	      if (g_pattern_ph_match (pattern, string))
		return TRUE;
	    }
	  while (*string);
	  break;

	default:
	  if (ch == *string)
	    string++;
	  else
	    return FALSE;
	  break;
	}

      ch = *pattern;
      pattern++;
    }

  return *string == 0;
}

gboolean
g_pattern_match (GPatternSpec *pspec,
		 guint         string_length,
		 const gchar  *string,
		 const gchar  *string_reversed)
{
  g_return_val_if_fail (pspec != NULL, FALSE);
  g_return_val_if_fail (string != NULL, FALSE);
  g_return_val_if_fail (string_reversed != NULL, FALSE);

  switch (pspec->match_type)
    {
    case G_MATCH_ALL:
      return g_pattern_ph_match (pspec->pattern, string);

    case G_MATCH_ALL_TAIL:
      return g_pattern_ph_match (pspec->pattern_reversed, string_reversed);

    case G_MATCH_HEAD:
      if (pspec->pattern_length > string_length)
	return FALSE;
      else if (pspec->pattern_length == string_length)
	return strcmp (pspec->pattern, string) == 0;
      else if (pspec->pattern_length)
	return strncmp (pspec->pattern, string, pspec->pattern_length) == 0;
      else
	return TRUE;

    case G_MATCH_TAIL:
      if (pspec->pattern_length > string_length)
	return FALSE;
      else if (pspec->pattern_length == string_length)
	return strcmp (pspec->pattern_reversed, string_reversed) == 0;
      else if (pspec->pattern_length)
	return strncmp (pspec->pattern_reversed,
			string_reversed,
			pspec->pattern_length) == 0;
      else
	return TRUE;

    case G_MATCH_EXACT:
      if (pspec->pattern_length != string_length)
	return FALSE;
      else
	return strcmp (pspec->pattern_reversed, string_reversed) == 0;

    default:
      g_return_val_if_fail (pspec->match_type < G_MATCH_LAST, FALSE);
      return FALSE;
    }
}

GPatternSpec*
g_pattern_spec_new (const gchar *pattern)
{
  GPatternSpec *pspec;
  gchar *p, *t;
  const gchar *h;
  guint hw = 0, tw = 0, hj = 0, tj = 0;

  g_return_val_if_fail (pattern != NULL, NULL);

  pspec = g_new (GPatternSpec, 1);
  pspec->pattern_length = strlen (pattern);
  pspec->pattern = strcpy (g_new (gchar, pspec->pattern_length + 1), pattern);
  pspec->pattern_reversed = g_new (gchar, pspec->pattern_length + 1);
  t = pspec->pattern_reversed + pspec->pattern_length;
  *(t--) = 0;
  h = pattern;
  while (t >= pspec->pattern_reversed)
    {
      register gchar c = *(h++);

      if (c == '*')
	{
	  if (t < h)
	    hw++;
	  else
	    tw++;
	}
      else if (c == '?')
	{
	  if (t < h)
	    hj++;
	  else
	    tj++;
	}

      *(t--) = c;
    }
  pspec->match_type = hw > tw || (hw == tw && hj > tj) ? G_MATCH_ALL_TAIL : G_MATCH_ALL;

  if (hj || tj)
    return pspec;

  if (hw == 0 && tw == 0)
    {
      pspec->match_type = G_MATCH_EXACT;
      return pspec;
    }

  if (hw)
    {
      p = pspec->pattern;
      while (*p == '*')
	p++;
      if (p > pspec->pattern && !strchr (p, '*'))
	{
	  gchar *tmp;

	  pspec->match_type = G_MATCH_TAIL;
	  pspec->pattern_length = strlen (p);
	  tmp = pspec->pattern;
	  pspec->pattern = strcpy (g_new (gchar, pspec->pattern_length + 1), p);
	  g_free (tmp);
	  g_free (pspec->pattern_reversed);
	  pspec->pattern_reversed = g_new (gchar, pspec->pattern_length + 1);
	  t = pspec->pattern_reversed + pspec->pattern_length;
	  *(t--) = 0;
	  h = pspec->pattern;
	  while (t >= pspec->pattern_reversed)
	    *(t--) = *(h++);
	  return pspec;
	}
    }

  if (tw)
    {
      p = pspec->pattern_reversed;
      while (*p == '*')
	p++;
      if (p > pspec->pattern_reversed && !strchr (p, '*'))
	{
	  gchar *tmp;

	  pspec->match_type = G_MATCH_HEAD;
	  pspec->pattern_length = strlen (p);
	  tmp = pspec->pattern_reversed;
	  pspec->pattern_reversed = strcpy (g_new (gchar, pspec->pattern_length + 1), p);
	  g_free (tmp);
	  g_free (pspec->pattern);
	  pspec->pattern = g_new (gchar, pspec->pattern_length + 1);
	  t = pspec->pattern + pspec->pattern_length;
	  *(t--) = 0;
	  h = pspec->pattern_reversed;
	  while (t >= pspec->pattern)
	    *(t--) = *(h++);
	}
    }

  return pspec;
}

gboolean
g_pattern_match_string (GPatternSpec *pspec,
			const gchar  *string)
{
  gchar *string_reversed, *t;
  const gchar *h;
  guint length;
  gboolean ergo;

  g_return_val_if_fail (pspec != NULL, FALSE);
  g_return_val_if_fail (string != NULL, FALSE);

  length = strlen (string);
  string_reversed = g_new (gchar, length + 1);
  t = string_reversed + length;
  *(t--) = 0;
  h = string;
  while (t >= string_reversed)
    *(t--) = *(h++);

  ergo = g_pattern_match (pspec, length, string, string_reversed);
  g_free (string_reversed);

  return ergo;
}

gboolean
g_pattern_match_simple (const gchar *pattern,
			const gchar *string)
{
  GPatternSpec *pspec;
  gboolean ergo;

  g_return_val_if_fail (pattern != NULL, FALSE);
  g_return_val_if_fail (string != NULL, FALSE);

  pspec = g_pattern_spec_new (pattern);
  ergo = g_pattern_match_string (pspec, string);
  g_pattern_spec_free (pspec);

  return ergo;
}

void
g_pattern_spec_free (GPatternSpec *pspec)
{
  g_return_if_fail (pspec != NULL);

  g_free (pspec->pattern);
  g_free (pspec->pattern_reversed);
  g_free (pspec);
}
