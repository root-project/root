/* GLIB - Library of useful routines for C programming
 *
 * gconvert.c: Convert between character sets using iconv
 * Copyright Red Hat Inc., 2000
 * Authors: Havoc Pennington <hp@redhat.com>, Owen Taylor <otaylor@redhat.com
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

#include <iconv.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>

#include "glib.h"
#include "config.h"

#ifdef G_PLATFORM_WIN32
#define STRICT
#include <windows.h>
#undef STRICT
#endif

#include "glibintl.h"

GQuark 
g_convert_error_quark()
{
  static GQuark quark;
  if (!quark)
    quark = g_quark_from_static_string ("g_convert_error");

  return quark;
}

#if defined(USE_LIBICONV) && !defined (_LIBICONV_H)
#error libiconv in use but included iconv.h not from libiconv
#endif
#if !defined(USE_LIBICONV) && defined (_LIBICONV_H)
#error libiconv not in use but included iconv.h is from libiconv
#endif

/**
 * g_iconv_open:
 * @to_codeset: destination codeset
 * @from_codeset: source codeset
 * 
 * Same as the standard UNIX routine iconv_open(), but
 * may be implemented via libiconv on UNIX flavors that lack
 * a native implementation.
 * 
 * GLib provides g_convert() and g_locale_to_utf8() which are likely
 * more convenient than the raw iconv wrappers.
 * 
 * Return value: a "conversion descriptor"
 **/
GIConv
g_iconv_open (const gchar  *to_codeset,
	      const gchar  *from_codeset)
{
  iconv_t cd = iconv_open (to_codeset, from_codeset);
  
  return (GIConv)cd;
}

/**
 * g_iconv:
 * @converter: conversion descriptor from g_iconv_open()
 * @inbuf: bytes to convert
 * @inbytes_left: inout parameter, bytes remaining to convert in @inbuf
 * @outbuf: converted output bytes
 * @outbytes_left: inout parameter, bytes available to fill in @outbuf
 * 
 * Same as the standard UNIX routine iconv(), but
 * may be implemented via libiconv on UNIX flavors that lack
 * a native implementation.
 *
 * GLib provides g_convert() and g_locale_to_utf8() which are likely
 * more convenient than the raw iconv wrappers.
 * 
 * Return value: count of non-reversible conversions, or -1 on error
 **/
size_t 
g_iconv (GIConv   converter,
	 gchar  **inbuf,
	 gsize   *inbytes_left,
	 gchar  **outbuf,
	 gsize   *outbytes_left)
{
  iconv_t cd = (iconv_t)converter;

  return iconv (cd, inbuf, inbytes_left, outbuf, outbytes_left);
}

/**
 * g_iconv_close:
 * @converter: a conversion descriptor from g_iconv_open()
 *
 * Same as the standard UNIX routine iconv_close(), but
 * may be implemented via libiconv on UNIX flavors that lack
 * a native implementation. Should be called to clean up
 * the conversion descriptor from iconv_open() when
 * you are done converting things.
 *
 * GLib provides g_convert() and g_locale_to_utf8() which are likely
 * more convenient than the raw iconv wrappers.
 * 
 * Return value: -1 on error, 0 on success
 **/
gint
g_iconv_close (GIConv converter)
{
  iconv_t cd = (iconv_t)converter;

  return iconv_close (cd);
}

static GIConv
open_converter (const gchar *to_codeset,
                const gchar *from_codeset,
		GError     **error)
{
  GIConv cd = g_iconv_open (to_codeset, from_codeset);

  if (cd == (iconv_t) -1)
    {
      /* Something went wrong.  */
      if (errno == EINVAL)
        g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_NO_CONVERSION,
                     _("Conversion from character set `%s' to `%s' is not supported"),
                     from_codeset, to_codeset);
      else
        g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_FAILED,
                     _("Could not open converter from `%s' to `%s': %s"),
                     from_codeset, to_codeset, strerror (errno));
    }

  return cd;

}

/**
 * g_convert:
 * @str:           the string to convert
 * @len:           the length of the string
 * @to_codeset:    name of character set into which to convert @str
 * @from_codeset:  character set of @str.
 * @bytes_read:    location to store the number of bytes in the
 *                 input string that were successfully converted, or %NULL.
 *                 Even if the conversion was succesful, this may be 
 *                 less than len if there were partial characters
 *                 at the end of the input. If the error
 *                 G_CONVERT_ERROR_ILLEGAL_SEQUENCE occurs, the value
 *                 stored will the byte fofset after the last valid
 *                 input sequence.
 * @bytes_written: the stored in the output buffer (not including the
 *                 terminating nul.
 * @error:         location to store the error occuring, or %NULL to ignore
 *                 errors. Any of the errors in #GConvertError may occur.
 *
 * Convert a string from one character set to another.
 *
 * Return value: If the conversion was successful, a newly allocated
 *               NUL-terminated string, which must be freed with
 *               g_free. Otherwise %NULL and @error will be set.
 **/
gchar*
g_convert (const gchar *str,
           gssize       len,  
           const gchar *to_codeset,
           const gchar *from_codeset,
           gsize       *bytes_read, 
	   gsize       *bytes_written, 
	   GError     **error)
{
  gchar *res;
  GIConv cd;
  
  g_return_val_if_fail (str != NULL, NULL);
  g_return_val_if_fail (to_codeset != NULL, NULL);
  g_return_val_if_fail (from_codeset != NULL, NULL);
     
  cd = open_converter (to_codeset, from_codeset, error);

  if (cd == (GIConv) -1)
    {
      if (bytes_read)
        *bytes_read = 0;
      
      if (bytes_written)
        *bytes_written = 0;
      
      return NULL;
    }

  res = g_convert_with_iconv (str, len, cd,
			      bytes_read, bytes_written,
			      error);
  
  g_iconv_close (cd);

  return res;
}

/**
 * g_convert_with_iconv:
 * @str:           the string to convert
 * @len:           the length of the string
 * @converter:     conversion descriptor from g_iconv_open()
 * @bytes_read:    location to store the number of bytes in the
 *                 input string that were successfully converted, or %NULL.
 *                 Even if the conversion was succesful, this may be 
 *                 less than len if there were partial characters
 *                 at the end of the input. If the error
 *                 G_CONVERT_ERROR_ILLEGAL_SEQUENCE occurs, the value
 *                 stored will the byte fofset after the last valid
 *                 input sequence.
 * @bytes_written: the stored in the output buffer (not including the
 *                 terminating nul.
 * @error:         location to store the error occuring, or %NULL to ignore
 *                 errors. Any of the errors in #GConvertError may occur.
 *
 * Convert a string from one character set to another.
 *
 * Return value: If the conversion was successful, a newly allocated
 *               NUL-terminated string, which must be freed with
 *               g_free. Otherwise %NULL and @error will be set.
 **/
gchar*
g_convert_with_iconv (const gchar *str,
		      gssize       len,
		      GIConv       converter,
		      gsize       *bytes_read, 
		      gsize       *bytes_written, 
		      GError     **error)
{
  gchar *dest;
  gchar *outp;
  const gchar *p;
  gsize inbytes_remaining;
  gsize outbytes_remaining;
  gsize err;
  gsize outbuf_size;
  gboolean have_error = FALSE;
  
  g_return_val_if_fail (str != NULL, NULL);
  g_return_val_if_fail (converter != (GIConv) -1, NULL);
     
  if (len < 0)
    len = strlen (str);

  p = str;
  inbytes_remaining = len;
  outbuf_size = len + 1; /* + 1 for nul in case len == 1 */
  
  outbytes_remaining = outbuf_size - 1; /* -1 for nul */
  outp = dest = g_malloc (outbuf_size);

 again:
  
  err = g_iconv (converter, (char **)&p, &inbytes_remaining, &outp, &outbytes_remaining);

  if (err == (size_t) -1)
    {
      switch (errno)
	{
	case EINVAL:
	  /* Incomplete text, do not report an error */
	  break;
	case E2BIG:
	  {
	    size_t used = outp - dest;

	    outbuf_size *= 2;
	    dest = g_realloc (dest, outbuf_size);
		
	    outp = dest + used;
	    outbytes_remaining = outbuf_size - used - 1; /* -1 for nul */

	    goto again;
	  }
	case EILSEQ:
	  g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_ILLEGAL_SEQUENCE,
		       _("Invalid byte sequence in conversion input"));
	  have_error = TRUE;
	  break;
	default:
	  g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_FAILED,
		       _("Error during conversion: %s"),
		       strerror (errno));
	  have_error = TRUE;
	  break;
	}
    }

  *outp = '\0';
  
  if (bytes_read)
    *bytes_read = p - str;
  else
    {
      if ((p - str) != len) 
	{
          if (!have_error)
            {
              g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_PARTIAL_INPUT,
                           _("Partial character sequence at end of input"));
              have_error = TRUE;
            }
	}
    }

  if (bytes_written)
    *bytes_written = outp - dest;	/* Doesn't include '\0' */

  if (have_error)
    {
      g_free (dest);
      return NULL;
    }
  else
    return dest;
}

/**
 * g_convert_with_fallback:
 * @str:          the string to convert
 * @len:          the length of the string
 * @to_codeset:   name of character set into which to convert @str
 * @from_codeset: character set of @str.
 * @fallback:     UTF-8 string to use in place of character not
 *                present in the target encoding. (This must be
 *                in the target encoding), if %NULL, characters
 *                not in the target encoding will be represented
 *                as Unicode escapes \x{XXXX} or \x{XXXXXX}.
 * @bytes_read:   location to store the number of bytes in the
 *                input string that were successfully converted, or %NULL.
 *                Even if the conversion was succesful, this may be 
 *                less than len if there were partial characters
 *                at the end of the input.
 * @bytes_written: the stored in the output buffer (not including the
 *                 terminating nul.
 * @error:        location to store the error occuring, or %NULL to ignore
 *                errors. Any of the errors in #GConvertError may occur.
 *
 * Convert a string from one character set to another, possibly
 * including fallback sequences for characters not representable
 * in the output. Note that it is not guaranteed that the specification
 * for the fallback sequences in @fallback will be honored. Some
 * systems may do a approximate conversion from @from_codeset
 * to @to_codeset in their iconv() functions, in which case GLib
 * will simply return that approximate conversion.
 *
 * Return value: If the conversion was successful, a newly allocated
 *               NUL-terminated string, which must be freed with
 *               g_free. Otherwise %NULL and @error will be set.
 **/
gchar*
g_convert_with_fallback (const gchar *str,
			 gssize       len,    
			 const gchar *to_codeset,
			 const gchar *from_codeset,
			 gchar       *fallback,
			 gsize       *bytes_read,
			 gsize       *bytes_written,
			 GError     **error)
{
  gchar *utf8;
  gchar *dest;
  gchar *outp;
  const gchar *insert_str = NULL;
  const gchar *p;
  gsize inbytes_remaining;   
  const gchar *save_p = NULL;
  gsize save_inbytes = 0;
  gsize outbytes_remaining; 
  gsize err;
  GIConv cd;
  gsize outbuf_size;
  gboolean have_error = FALSE;
  gboolean done = FALSE;

  GError *local_error = NULL;
  
  g_return_val_if_fail (str != NULL, NULL);
  g_return_val_if_fail (to_codeset != NULL, NULL);
  g_return_val_if_fail (from_codeset != NULL, NULL);
     
  if (len < 0)
    len = strlen (str);
  
  /* Try an exact conversion; we only proceed if this fails
   * due to an illegal sequence in the input string.
   */
  dest = g_convert (str, len, to_codeset, from_codeset, 
		    bytes_read, bytes_written, &local_error);
  if (!local_error)
    return dest;

  if (!g_error_matches (local_error, G_CONVERT_ERROR, G_CONVERT_ERROR_ILLEGAL_SEQUENCE))
    {
      g_propagate_error (error, local_error);
      return NULL;
    }
  else
    g_error_free (local_error);

  local_error = NULL;
  
  /* No go; to proceed, we need a converter from "UTF-8" to
   * to_codeset, and the string as UTF-8.
   */
  cd = open_converter (to_codeset, "UTF-8", error);
  if (cd == (GIConv) -1)
    {
      if (bytes_read)
        *bytes_read = 0;
      
      if (bytes_written)
        *bytes_written = 0;
      
      return NULL;
    }

  utf8 = g_convert (str, len, "UTF-8", from_codeset, 
		    bytes_read, &inbytes_remaining, error);
  if (!utf8)
    return NULL;

  /* Now the heart of the code. We loop through the UTF-8 string, and
   * whenever we hit an offending character, we form fallback, convert
   * the fallback to the target codeset, and then go back to
   * converting the original string after finishing with the fallback.
   *
   * The variables save_p and save_inbytes store the input state
   * for the original string while we are converting the fallback
   */
  p = utf8;

  outbuf_size = len + 1; /* + 1 for nul in case len == 1 */
  outbytes_remaining = outbuf_size - 1; /* -1 for nul */
  outp = dest = g_malloc (outbuf_size);

  while (!done && !have_error)
    {
      size_t inbytes_tmp = inbytes_remaining;
      err = g_iconv (cd, (char **)&p, &inbytes_tmp, &outp, &outbytes_remaining);
      inbytes_remaining = inbytes_tmp;

      if (err == (size_t) -1)
	{
	  switch (errno)
	    {
	    case EINVAL:
	      g_assert_not_reached();
	      break;
	    case E2BIG:
	      {
		size_t used = outp - dest;

		outbuf_size *= 2;
		dest = g_realloc (dest, outbuf_size);
		
		outp = dest + used;
		outbytes_remaining = outbuf_size - used - 1; /* -1 for nul */
		
		break;
	      }
	    case EILSEQ:
	      if (save_p)
		{
		  /* Error converting fallback string - fatal
		   */
		  g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_ILLEGAL_SEQUENCE,
			       _("Cannot convert fallback '%s' to codeset '%s'"),
			       insert_str, to_codeset);
		  have_error = TRUE;
		  break;
		}
	      else
		{
		  if (!fallback)
		    { 
		      gunichar ch = g_utf8_get_char (p);
		      insert_str = g_strdup_printf ("\\x{%0*X}",
						    (ch < 0x10000) ? 4 : 6,
						    ch);
		    }
		  else
		    insert_str = fallback;
		  
		  save_p = g_utf8_next_char (p);
		  save_inbytes = inbytes_remaining - (save_p - p);
		  p = insert_str;
		  inbytes_remaining = strlen (p);
		}
	      break;
	    default:
	      g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_FAILED,
			   _("Error during conversion: %s"),
			   strerror (errno));
	      have_error = TRUE;
	      break;
	    }
	}
      else
	{
	  if (save_p)
	    {
	      if (!fallback)
		g_free ((gchar *)insert_str);
	      p = save_p;
	      inbytes_remaining = save_inbytes;
	      save_p = NULL;
	    }
	  else
	    done = TRUE;
	}
    }

  /* Cleanup
   */
  *outp = '\0';
  
  g_iconv_close (cd);

  if (bytes_written)
    *bytes_written = outp - str;	/* Doesn't include '\0' */

  g_free (utf8);

  if (have_error)
    {
      if (save_p && !fallback)
	g_free ((gchar *)insert_str);
      g_free (dest);
      return NULL;
    }
  else
    return dest;
}

/*
 * g_locale_to_utf8
 *
 * 
 */

static gchar *
strdup_len (const gchar *string,
	    gssize       len,
	    gsize       *bytes_written,
	    gsize       *bytes_read,
	    GError      **error)
	 
{
  gsize real_len;

  if (!g_utf8_validate (string, -1, NULL))
    {
      if (bytes_read)
	*bytes_read = 0;
      if (bytes_written)
	*bytes_written = 0;

      g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_ILLEGAL_SEQUENCE,
		   _("Invalid byte sequence in conversion input"));
      return NULL;
    }
  
  if (len < 0)
    real_len = strlen (string);
  else
    {
      real_len = 0;
      
      while (real_len < len && string[real_len])
	real_len++;
    }
  
  if (bytes_read)
    *bytes_read = real_len;
  if (bytes_written)
    *bytes_written = real_len;

  return g_strndup (string, real_len);
}

/**
 * g_locale_to_utf8:
 * @opsysstring:   a string in the encoding of the current locale
 * @len:           the length of the string, or -1 if the string is
 *                 NULL-terminated.
 * @bytes_read:    location to store the number of bytes in the
 *                 input string that were successfully converted, or %NULL.
 *                 Even if the conversion was succesful, this may be 
 *                 less than len if there were partial characters
 *                 at the end of the input. If the error
 *                 G_CONVERT_ERROR_ILLEGAL_SEQUENCE occurs, the value
 *                 stored will the byte fofset after the last valid
 *                 input sequence.
 * @bytes_written: the stored in the output buffer (not including the
 *                 terminating nul.
 * @error: location to store the error occuring, or %NULL to ignore
 *                 errors. Any of the errors in #GConvertError may occur.
 * 
 * Converts a string which is in the encoding used for strings by
 * the C runtime (usually the same as that used by the operating
 * system) in the current locale into a UTF-8 string.
 * 
 * Return value: The converted string, or %NULL on an error.
 **/
gchar *
g_locale_to_utf8 (const gchar  *opsysstring,
		  gssize        len,            
		  gsize        *bytes_read,    
		  gsize        *bytes_written,
		  GError      **error)
{
#ifdef G_PLATFORM_WIN32

  gint i, clen, total_len, wclen, first;
  wchar_t *wcs, wc;
  gchar *result, *bp;
  const wchar_t *wcp;

  if (len == -1)
    len = strlen (opsysstring);
  
  wcs = g_new (wchar_t, len);
  wclen = MultiByteToWideChar (CP_ACP, 0, opsysstring, len, wcs, len);

  wcp = wcs;
  total_len = 0;
  for (i = 0; i < wclen; i++)
    {
      wc = *wcp++;

      if (wc < 0x80)
	total_len += 1;
      else if (wc < 0x800)
	total_len += 2;
      else if (wc < 0x10000)
	total_len += 3;
      else if (wc < 0x200000)
	total_len += 4;
      else if (wc < 0x4000000)
	total_len += 5;
      else
	total_len += 6;
    }

  result = g_malloc (total_len + 1);
  
  wcp = wcs;
  bp = result;
  for (i = 0; i < wclen; i++)
    {
      wc = *wcp++;

      if (wc < 0x80)
	{
	  first = 0;
	  clen = 1;
	}
      else if (wc < 0x800)
	{
	  first = 0xc0;
	  clen = 2;
	}
      else if (wc < 0x10000)
	{
	  first = 0xe0;
	  clen = 3;
	}
      else if (wc < 0x200000)
	{
	  first = 0xf0;
	  clen = 4;
	}
      else if (wc < 0x4000000)
	{
	  first = 0xf8;
	  clen = 5;
	}
      else
	{
	  first = 0xfc;
	  clen = 6;
	}
      
      /* Woo-hoo! */
      switch (clen)
	{
	case 6: bp[5] = (wc & 0x3f) | 0x80; wc >>= 6; /* Fall through */
	case 5: bp[4] = (wc & 0x3f) | 0x80; wc >>= 6; /* Fall through */
	case 4: bp[3] = (wc & 0x3f) | 0x80; wc >>= 6; /* Fall through */
	case 3: bp[2] = (wc & 0x3f) | 0x80; wc >>= 6; /* Fall through */
	case 2: bp[1] = (wc & 0x3f) | 0x80; wc >>= 6; /* Fall through */
	case 1: bp[0] = wc | first;
	}

      bp += clen;
    }
  *bp = 0;

  g_free (wcs);

  if (bytes_read)
    *bytes_read = len;
  if (bytes_written)
    *bytes_written = total_len;
  
  return result;

#else  /* !G_PLATFORM_WIN32 */

  const char *charset;

  if (g_get_charset (&charset))
    return strdup_len (opsysstring, len, bytes_read, bytes_written, error);
  else
    return g_convert (opsysstring, len, 
		      "UTF-8", charset, bytes_read, bytes_written, error);

#endif /* !G_PLATFORM_WIN32 */
}

/**
 * g_locale_from_utf8:
 * @utf8string:    a UTF-8 encoded string 
 * @len:           the length of the string, or -1 if the string is
 *                 NULL-terminated.
 * @bytes_read:    location to store the number of bytes in the
 *                 input string that were successfully converted, or %NULL.
 *                 Even if the conversion was succesful, this may be 
 *                 less than len if there were partial characters
 *                 at the end of the input. If the error
 *                 G_CONVERT_ERROR_ILLEGAL_SEQUENCE occurs, the value
 *                 stored will the byte fofset after the last valid
 *                 input sequence.
 * @bytes_written: the stored in the output buffer (not including the
 *                 terminating nul.
 * @error: location to store the error occuring, or %NULL to ignore
 *                 errors. Any of the errors in #GConvertError may occur.
 * 
 * Converts a string from UTF-8 to the encoding used for strings by
 * the C runtime (usually the same as that used by the operating
 * system) in the current locale.
 * 
 * Return value: The converted string, or %NULL on an error.
 **/
gchar *
g_locale_from_utf8 (const gchar *utf8string,
		    gssize       len,            
		    gsize       *bytes_read,    
		    gsize       *bytes_written,
		    GError     **error)
{
#ifdef G_PLATFORM_WIN32

  gint i, mask, clen, mblen;
  wchar_t *wcs, *wcp;
  gchar *result;
  guchar *cp, *end, c;
  gint n;
  
  if (len == -1)
    len = strlen (utf8string);
  
  /* First convert to wide chars */
  cp = (guchar *) utf8string;
  end = cp + len;
  n = 0;
  wcs = g_new (wchar_t, len + 1);
  wcp = wcs;
  while (cp != end)
    {
      mask = 0;
      c = *cp;

      if (c < 0x80)
	{
	  clen = 1;
	  mask = 0x7f;
	}
      else if ((c & 0xe0) == 0xc0)
	{
	  clen = 2;
	  mask = 0x1f;
	}
      else if ((c & 0xf0) == 0xe0)
	{
	  clen = 3;
	  mask = 0x0f;
	}
      else if ((c & 0xf8) == 0xf0)
	{
	  clen = 4;
	  mask = 0x07;
	}
      else if ((c & 0xfc) == 0xf8)
	{
	  clen = 5;
	  mask = 0x03;
	}
      else if ((c & 0xfc) == 0xfc)
	{
	  clen = 6;
	  mask = 0x01;
	}
      else
	{
	  g_free (wcs);
	  return NULL;
	}

      if (cp + clen > end)
	{
	  g_free (wcs);
	  return NULL;
	}

      *wcp = (cp[0] & mask);
      for (i = 1; i < clen; i++)
	{
	  if ((cp[i] & 0xc0) != 0x80)
	    {
	      g_free (wcs);
	      return NULL;
	    }
	  *wcp <<= 6;
	  *wcp |= (cp[i] & 0x3f);
	}

      cp += clen;
      wcp++;
      n++;
    }
  if (cp != end)
    {
      g_free (wcs);
      return NULL;
    }

  /* n is the number of wide chars constructed */

  /* Convert to a string in the current ANSI codepage */

  result = g_new (gchar, 3 * n + 1);
  mblen = WideCharToMultiByte (CP_ACP, 0, wcs, n, result, 3*n, NULL, NULL);
  result[mblen] = 0;
  g_free (wcs);

  if (bytes_read)
    *bytes_read = len;
  if (bytes_written)
    *bytes_written = mblen;
  
  return result;

#else  /* !G_PLATFORM_WIN32 */
  
  const gchar *charset;

  if (g_get_charset (&charset))
    return strdup_len (utf8string, len, bytes_read, bytes_written, error);
  else
    return g_convert (utf8string, len,
		      charset, "UTF-8", bytes_read, bytes_written, error);

#endif /* !G_PLATFORM_WIN32 */
}

/**
 * g_filename_to_utf8:
 * @opsysstring:   a string in the encoding for filenames
 * @len:           the length of the string, or -1 if the string is
 *                 NULL-terminated.
 * @bytes_read:    location to store the number of bytes in the
 *                 input string that were successfully converted, or %NULL.
 *                 Even if the conversion was succesful, this may be 
 *                 less than len if there were partial characters
 *                 at the end of the input. If the error
 *                 G_CONVERT_ERROR_ILLEGAL_SEQUENCE occurs, the value
 *                 stored will the byte fofset after the last valid
 *                 input sequence.
 * @bytes_written: the stored in the output buffer (not including the
 *                 terminating nul.
 * @error: location to store the error occuring, or %NULL to ignore
 *                 errors. Any of the errors in #GConvertError may occur.
 * 
 * Converts a string which is in the encoding used for filenames
 * into a UTF-8 string.
 * 
 * Return value: The converted string, or %NULL on an error.
 **/
gchar*
g_filename_to_utf8 (const gchar *opsysstring, 
		    gssize       len,           
		    gsize       *bytes_read,   
		    gsize       *bytes_written,
		    GError     **error)
{
#ifdef G_PLATFORM_WIN32
  return g_locale_to_utf8 (opsysstring, len,
			   bytes_read, bytes_written,
			   error);
#else  /* !G_PLATFORM_WIN32 */
      
  if (getenv ("G_BROKEN_FILENAMES"))
    return g_locale_to_utf8 (opsysstring, len,
			     bytes_read, bytes_written,
			     error);
  else
    return strdup_len (opsysstring, len, bytes_read, bytes_written, error);
#endif /* !G_PLATFORM_WIN32 */
}

/**
 * g_filename_from_utf8:
 * @utf8string:    a UTF-8 encoded string 
 * @len:           the length of the string, or -1 if the string is
 *                 NULL-terminated.
 * @bytes_read:    location to store the number of bytes in the
 *                 input string that were successfully converted, or %NULL.
 *                 Even if the conversion was succesful, this may be 
 *                 less than len if there were partial characters
 *                 at the end of the input. If the error
 *                 G_CONVERT_ERROR_ILLEGAL_SEQUENCE occurs, the value
 *                 stored will the byte fofset after the last valid
 *                 input sequence.
 * @bytes_written: the stored in the output buffer (not including the
 *                 terminating nul.
 * @error: location to store the error occuring, or %NULL to ignore
 *                 errors. Any of the errors in #GConvertError may occur.
 * 
 * Converts a string from UTF-8 to the encoding used for filenames.
 * 
 * Return value: The converted string, or %NULL on an error.
 **/
gchar*
g_filename_from_utf8 (const gchar *utf8string,
		      gssize       len,            
		      gsize       *bytes_read,    
		      gsize       *bytes_written,
		      GError     **error)
{
#ifdef G_PLATFORM_WIN32
  return g_locale_from_utf8 (utf8string, len,
			     bytes_read, bytes_written,
			     error);
#else  /* !G_PLATFORM_WIN32 */
  if (getenv ("G_BROKEN_FILENAMES"))
    return g_locale_from_utf8 (utf8string, len,
			       bytes_read, bytes_written,
			       error);
  else
    return strdup_len (utf8string, len, bytes_read, bytes_written, error);
#endif /* !G_PLATFORM_WIN32 */
}

/* Test of haystack has the needle prefix, comparing case
 * insensitive. haystack may be UTF-8, but needle must
 * contain only ascii. */
static gboolean
has_case_prefix (const gchar *haystack, const gchar *needle)
{
  const gchar *h, *n;
  
  /* Eat one character at a time. */
  h = haystack;
  n = needle;

  while (*n && *h &&
	 g_ascii_tolower (*n) == g_ascii_tolower (*h))
    {
      n++;
      h++;
    }
  
  return *n == '\0';
}

typedef enum {
  UNSAFE_ALL        = 0x1,  /* Escape all unsafe characters   */
  UNSAFE_ALLOW_PLUS = 0x2,  /* Allows '+'  */
  UNSAFE_PATH       = 0x4,  /* Allows '/' and '?' and '&' and '='  */
  UNSAFE_DOS_PATH   = 0x8,  /* Allows '/' and '?' and '&' and '=' and ':' */
  UNSAFE_HOST       = 0x10, /* Allows '/' and ':' and '@' */
  UNSAFE_SLASHES    = 0x20  /* Allows all characters except for '/' and '%' */
} UnsafeCharacterSet;

static const guchar acceptable[96] = {
 /* X0   X1   X2   X3   X4   X5   X6   X7   X8   X9   XA   XB   XC   XD   XE   XF */
  0x00,0x3F,0x20,0x20,0x20,0x00,0x2C,0x3F,0x3F,0x3F,0x3F,0x22,0x20,0x3F,0x3F,0x1C, /* 2X  !"#$%&'()*+,-./   */
  0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x38,0x20,0x20,0x2C,0x20,0x2C, /* 3X 0123456789:;<=>?   */
  0x30,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F, /* 4X @ABCDEFGHIJKLMNO   */
  0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x20,0x20,0x20,0x20,0x3F, /* 5X PQRSTUVWXYZ[\]^_   */
  0x20,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F, /* 6X `abcdefghijklmno   */
  0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x20,0x20,0x20,0x3F,0x20  /* 7X pqrstuvwxyz{|}~DEL */
};

static const gchar hex[16] = "0123456789ABCDEF";

/* Note: This escape function works on file: URIs, but if you want to
 * escape something else, please read RFC-2396 */
static gchar *
g_escape_uri_string (const gchar *string, 
		     UnsafeCharacterSet mask)
{
#define ACCEPTABLE(a) ((a)>=32 && (a)<128 && (acceptable[(a)-32] & use_mask))

  const gchar *p;
  gchar *q;
  gchar *result;
  int c;
  gint unacceptable;
  UnsafeCharacterSet use_mask;
  
  g_return_val_if_fail (mask == UNSAFE_ALL
			|| mask == UNSAFE_ALLOW_PLUS
			|| mask == UNSAFE_PATH
			|| mask == UNSAFE_DOS_PATH
			|| mask == UNSAFE_HOST
			|| mask == UNSAFE_SLASHES, NULL);
  
  unacceptable = 0;
  use_mask = mask;
  for (p = string; *p != '\0'; p++)
    {
      c = *p;
      if (!ACCEPTABLE (c)) 
	unacceptable++;
    }
  
  result = g_malloc (p - string + unacceptable * 2 + 1);
  
  use_mask = mask;
  for (q = result, p = string; *p != '\0'; p++)
    {
      c = (unsigned char)*p;
      
      if (!ACCEPTABLE (c))
	{
	  *q++ = '%'; /* means hex coming */
	  *q++ = hex[c >> 4];
	  *q++ = hex[c & 15];
	}
      else
	*q++ = *p;
    }
  
  *q = '\0';
  
  return result;
}


static gchar *
g_escape_file_uri (const gchar *hostname,
		   const gchar *pathname)
{
  char *escaped_hostname = NULL;
  char *escaped_path;
  char *res;

  if (hostname && *hostname != '\0')
    {
      escaped_hostname = g_escape_uri_string (hostname, UNSAFE_HOST);
    }

  escaped_path = g_escape_uri_string (pathname, UNSAFE_DOS_PATH);

  res = g_strconcat ("file://",
		     (escaped_hostname) ? escaped_hostname : "",
		     (*escaped_path != '/') ? "/" : "",
		     escaped_path,
		     NULL);

  g_free (escaped_hostname);
  g_free (escaped_path);
  
  return res;
}

static int
unescape_character (const char *scanner)
{
  int first_digit;
  int second_digit;

  first_digit = g_ascii_xdigit_value (*scanner++);
  
  if (first_digit < 0) 
    return -1;
  
  second_digit = g_ascii_xdigit_value (*scanner++);
  if (second_digit < 0) 
    return -1;
  
  return (first_digit << 4) | second_digit;
}

static gchar *
g_unescape_uri_string (const gchar *escaped,
		       const gchar *illegal_characters,
		       int          len)
{
  const gchar *in, *in_end;
  gchar *out, *result;
  int character;
  
  if (escaped == NULL)
    return NULL;

  if (len < 0)
    len = strlen (escaped);

    result = g_malloc (len + 1);
  
  out = result;
  for (in = escaped, in_end = escaped + len; in < in_end && *in != '\0'; in++)
    {
      character = *in;
      if (character == '%')
	{
	  character = unescape_character (in + 1);
      
	  /* Check for an illegal character. We consider '\0' illegal here. */
	  if (character == 0
	      || (illegal_characters != NULL
		  && strchr (illegal_characters, (char)character) != NULL))
	    {
	      g_free (result);
	      return NULL;
	    }
	  in += 2;
	}
      *out++ = character;
    }
  
  *out = '\0';
  
  g_assert (out - result <= strlen (escaped));

  if (!g_utf8_validate (result, -1, NULL))
    {
      g_free (result);
      return NULL;
    }
  
  return result;
}

/**
 * g_filename_from_uri:
 * @uri: a uri describing a filename (escaped, encoded in UTF-8)
 * @hostname: Location to store hostname for the URI, or %NULL.
 *            If there is no hostname in the URI, %NULL will be
 *            stored in this location.
 * @error: location to store the error occuring, or %NULL to ignore
 *         errors. Any of the errors in #GConvertError may occur.
 * 
 * Converts an escaped UTF-8 encoded URI to a local filename in the
 * encoding used for filenames. 
 * 
 * Return value: a newly allocated string holding the resulting
 *               filename, or %NULL on an error.
 **/
gchar *
g_filename_from_uri (const char *uri,
		     char      **hostname,
		     GError    **error)
{
  const char *path_part;
  const char *host_part;
  char *unescaped_hostname;
  char *result;
  char *filename;
  int offs;

  if (hostname)
    *hostname = NULL;

  if (!has_case_prefix (uri, "file:/"))
    {
      g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_NOT_ABSOLUTE_FILE_URI,
		   _("The URI `%s' is not an absolute URI using the file scheme"),
		   uri);
      return NULL;
    }
  
  path_part = uri + strlen ("file:");
  
  if (strchr (path_part, '#') != NULL)
    {
      g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_INVALID_URI,
		   _("The local file URI `%s' may not include a `#'"),
		   uri);
      return NULL;
    }
	
  if (has_case_prefix (path_part, "///")) 
    path_part += 2;
  else if (has_case_prefix (path_part, "//"))
    {
      path_part += 2;
      host_part = path_part;

      path_part = strchr (path_part, '/');

      if (path_part == NULL)
	{
	  g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_INVALID_URI,
		       _("The URI `%s' is invalid"),
		       uri);
	  return NULL;
	}

      unescaped_hostname = g_unescape_uri_string (host_part, "", path_part - host_part);
      if (unescaped_hostname == NULL)
	{
	  g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_INVALID_URI,
		       _("The hostname of the URI `%s' contains invalidly escaped characters"),
		       uri);
	  return NULL;
	}
      
      if (hostname)
	*hostname = unescaped_hostname;
      else
	g_free (unescaped_hostname);
    }

  filename = g_unescape_uri_string (path_part, "/", -1);

  if (filename == NULL)
    {
      g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_INVALID_URI,
		   _("The URI `%s' contains invalidly escaped characters"),
		   uri);
      return NULL;
    }

  /* DOS uri's are like "file://host/c:\foo", so we need to check if we need to
   * drop the initial slash */
  offs = 0;
  if (g_path_is_absolute (filename+1))
    offs = 1;
  
  result = g_filename_from_utf8 (filename + offs, -1, NULL, NULL, error);
  g_free (filename);
  
  return result;
}

/**
 * g_filename_to_uri:
 * @filename: an absolute filename specified in the encoding
 *            used for filenames by the operating system.
 * @hostname: A UTF-8 encoded hostname, or %NULL for none.
 * @error: location to store the error occuring, or %NULL to ignore
 *         errors. Any of the errors in #GConvertError may occur.
 * 
 * Converts an absolute filename to an escaped UTF-8 encoded URI.
 * 
 * Return value: a newly allocated string holding the resulting
 *               URI, or %NULL on an error.
 **/
gchar *
g_filename_to_uri   (const char *filename,
		     char       *hostname,
		     GError    **error)
{
  char *escaped_uri;
  char *utf8_filename;

  g_return_val_if_fail (filename != NULL, NULL);

  if (!g_path_is_absolute (filename))
    {
      g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_NOT_ABSOLUTE_PATH,
		   _("The pathname '%s' is not an absolute path"),
		   filename);
      return NULL;
    }

  utf8_filename = g_filename_to_utf8 (filename, -1, NULL, NULL, error);
  if (utf8_filename == NULL)
    return NULL;
  
  if (hostname &&
      !g_utf8_validate (hostname, -1, NULL))
    {
      g_free (utf8_filename);
      g_set_error (error, G_CONVERT_ERROR, G_CONVERT_ERROR_ILLEGAL_SEQUENCE,
		   _("Invalid byte sequence in hostname"));
      return NULL;
    }
  
  escaped_uri = g_escape_file_uri (hostname,
				   utf8_filename);
  g_free (utf8_filename);
  
  return escaped_uri;
}

