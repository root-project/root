/* gunicollate.c - Collation
 *
 *  Copyright 2001 Red Hat, Inc.
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

#include <locale.h>
#include <string.h>
#ifdef __STDC_ISO_10646__
#include <wchar.h>
#endif

#include "glib.h"

extern gunichar *_g_utf8_normalize_wc (const gchar    *str,
				       gssize          max_len,
				       GNormalizeMode  mode);

/**
 * g_utf8_collate:
 * @str1: a UTF-8 encoded string
 * @str2: a UTF-8 encoded string
 * 
 * Compares two strings for ordering using the linguistically
 * correct rules for the current locale. When sorting a large
 * number of strings, it will be significantly faster to
 * obtain collation keys with g_utf8_collate_key() and 
 * compare the keys with <function>strcmp()</function> when 
 * sorting instead of sorting the original strings.
 * 
 * Return value: -1 if @str1 compares before @str2, 0 if they
 *   compare equal, 1 if @str1 compares after @str2.
 **/
gint
g_utf8_collate (const gchar *str1,
		const gchar *str2)
{
  gint result;
  
#ifdef __STDC_ISO_10646__

  gunichar *str1_norm = _g_utf8_normalize_wc (str1, -1, G_NORMALIZE_ALL_COMPOSE);
  gunichar *str2_norm = _g_utf8_normalize_wc (str2, -1, G_NORMALIZE_ALL_COMPOSE);

  result = wcscoll ((wchar_t *)str1_norm, (wchar_t *)str2_norm);

  g_free (str1_norm);
  g_free (str2_norm);

#else /* !__STDC_ISO_10646__ */

  const gchar *charset;
  gchar *str1_norm = g_utf8_normalize (str1, -1, G_NORMALIZE_ALL_COMPOSE);
  gchar *str2_norm = g_utf8_normalize (str2, -1, G_NORMALIZE_ALL_COMPOSE);

  if (g_get_charset (&charset))
    {
      result = strcoll (str1_norm, str2_norm);
    }
  else
    {
      gchar *str1_locale = g_convert (str1_norm, -1, "UTF-8", charset, NULL, NULL, NULL);
      gchar *str2_locale = g_convert (str2_norm, -1, "UTF-8", charset, NULL, NULL, NULL);

      if (str1_locale && str2_locale)
	result =  strcoll (str1_locale, str2_locale);
      else if (str1_locale)
	result = -1;
      else if (str2_locale)
	result = 1;
      else
	result = strcmp (str1_norm, str2_norm);

      g_free (str1_locale);
      g_free (str2_locale);
    }

  g_free (str1_norm);
  g_free (str2_norm);

#endif /* __STDC_ISO_10646__ */

  return result;
}

#ifdef __STDC_ISO_10646__
/* We need UTF-8 encoding of numbers to encode the weights if
 * we are using wcsxfrm. However, we aren't encoding Unicode
 * characters, so we can't simply use g_unichar_to_utf8.
 *
 * The following routine is taken (with modification) from GNU
 * libc's strxfrm routine:
 *
 * Copyright (C) 1995-1999,2000,2001 Free Software Foundation, Inc.
 * Written by Ulrich Drepper <drepper@cygnus.com>, 1995.
 */
static inline int
utf8_encode (char *buf, wchar_t val)
{
  int retval;

  if (val < 0x80)
    {
      if (buf)
	*buf++ = (char) val;
      retval = 1;
    }
  else
    {
      int step;

      for (step = 2; step < 6; ++step)
        if ((val & (~(guint32)0 << (5 * step + 1))) == 0)
          break;
      retval = step;

      if (buf)
	{
	  *buf = (unsigned char) (~0xff >> step);
	  --step;
	  do
	    {
	      buf[step] = 0x80 | (val & 0x3f);
	      val >>= 6;
	    }
	  while (--step > 0);
	  *buf |= val;
	}
    }

  return retval;
}
#endif /* __STDC_ISO_10646__ */

/**
 * g_utf8_collate_key:
 * @str: a UTF-8 encoded string.
 * @len: length of @str, in bytes, or -1 if @str is nul-terminated.
 *
 * Converts a string into a collation key that can be compared
 * with other collation keys using <function>strcmp()</function>. 
 * The results of comparing the collation keys of two strings 
 * with <function>strcmp()</function> will always be the same as 
 * comparing the two original keys with g_utf8_collate().
 * 
 * Return value: a newly allocated string. This string should
 *   be freed with g_free() when you are done with it.
 **/
gchar *
g_utf8_collate_key (const gchar *str,
		    gssize       len)
{
  gchar *result;
  size_t xfrm_len;
  
#ifdef __STDC_ISO_10646__

  gunichar *str_norm = _g_utf8_normalize_wc (str, len, G_NORMALIZE_ALL_COMPOSE);
  wchar_t *result_wc;
  size_t i;
  size_t result_len = 0;

  setlocale (LC_COLLATE, "");

  xfrm_len = wcsxfrm (NULL, (wchar_t *)str_norm, 0);
  result_wc = g_new (wchar_t, xfrm_len + 1);
  wcsxfrm (result_wc, (wchar_t *)str_norm, xfrm_len + 1);

  for (i=0; i < xfrm_len; i++)
    result_len += utf8_encode (NULL, result_wc[i]);

  result = g_malloc (result_len + 1);
  result_len = 0;
  for (i=0; i < xfrm_len; i++)
    result_len += utf8_encode (result + result_len, result_wc[i]);

  result[result_len] = '\0';

  g_free (result_wc);
  g_free (str_norm);

  return result;
#else /* !__STDC_ISO_10646__ */

  const gchar *charset;
  gchar *str_norm = g_utf8_normalize (str, len, G_NORMALIZE_ALL_COMPOSE);

  if (g_get_charset (&charset))
    {
      xfrm_len = strxfrm (NULL, str_norm, 0);
      result = g_malloc (xfrm_len + 1);
      strxfrm (result, str_norm, xfrm_len + 1);
    }
  else
    {
      gchar *str_locale = g_convert (str_norm, -1, "UTF-8", charset, NULL, NULL, NULL);

      if (str_locale)
	{
	  xfrm_len = strxfrm (NULL, str_locale, 0);
	  result = g_malloc (xfrm_len + 2);
	  result[0] = 'A';
	  strxfrm (result + 1, str_locale, xfrm_len + 1);
	  
	  g_free (str_locale);
	}
      else
	{
	  xfrm_len = strlen (str_norm);
	  result = g_malloc (xfrm_len + 2);
	  result[0] = 'B';
	  memcpy (result + 1, str_norm, xfrm_len);
	  result[xfrm_len+1] = '\0';
	}
    }

  g_free (str_norm);
#endif /* __STDC_ISO_10646__ */

  return result;
}
