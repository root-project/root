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

#include "glib.h"

static GError* 
g_error_new_valist(GQuark         domain,
                   gint           code,
                   const gchar   *format,
                   va_list        args)
{
  GError *error;
  
  error = g_new (GError, 1);
  
  error->domain = domain;
  error->code = code;
  error->message = g_strdup_vprintf (format, args);
  
  return error;
}

/**
 * g_error_new:
 * @domain: error domain 
 * @code: error code
 * @format: printf()-style format for error message
 * @Varargs: parameters for message format
 * 
 * Creates a new #GError with the given @domain and @code,
 * and a message formatted with @format.
 * 
 * Return value: a new #GError
 **/
GError*
g_error_new (GQuark       domain,
             gint         code,
             const gchar *format,
             ...)
{
  GError* error;
  va_list args;

  g_return_val_if_fail (format != NULL, NULL);
  g_return_val_if_fail (domain != 0, NULL);

  va_start (args, format);
  error = g_error_new_valist (domain, code, format, args);
  va_end (args);

  return error;
}

/**
 * g_error_new_literal:
 * @domain: error domain
 * @code: error code
 * @message: error message
 * 
 * Creates a new #GError; unlike g_error_new(), @message is not
 * a printf()-style format string. Use this function if @message
 * contains text you don't have control over, that could include
 * printf() escape sequences.
 * 
 * Return value: a new #GError
 **/
GError*
g_error_new_literal (GQuark         domain,
                     gint           code,
                     const gchar   *message)
{
  GError* err;

  g_return_val_if_fail (message != NULL, NULL);
  g_return_val_if_fail (domain != 0, NULL);

  err = g_new (GError, 1);

  err->domain = domain;
  err->code = code;
  err->message = g_strdup (message);
  
  return err;
}

/**
 * g_error_free:
 * @error: a #GError
 *
 * Frees a #GError and associated resources.
 * 
 **/
void
g_error_free (GError *error)
{
  g_return_if_fail (error != NULL);  

  g_free (error->message);

  g_free (error);
}

/**
 * g_error_copy:
 * @error: a #GError
 * 
 * Makes a copy of @error.
 * 
 * Return value: a new #GError
 **/
GError*
g_error_copy (const GError *error)
{
  GError *copy;
  
  g_return_val_if_fail (error != NULL, NULL);

  copy = g_new (GError, 1);

  *copy = *error;

  copy->message = g_strdup (error->message);

  return copy;
}

/**
 * g_error_matches:
 * @error: a #GError
 * @domain: an error domain
 * @code: an error code
 * 
 * Returns TRUE if @error matches @domain and @code, FALSE
 * otherwise.
 * 
 * Return value: whether @error has @domain and @code
 **/
gboolean
g_error_matches (const GError *error,
                 GQuark        domain,
                 gint          code)
{
  return error &&
    error->domain == domain &&
    error->code == code;
}

#define ERROR_OVERWRITTEN_WARNING "GError set over the top of a previous GError or uninitialized memory.\n" \
               "This indicates a bug in someone's code. You must ensure an error is NULL before it's set.\n" \
               "The overwriting error message was: %s"

/**
 * g_set_error:
 * @err: a return location for a #GError, or NULL
 * @domain: error domain
 * @code: error code 
 * @format: printf()-style format
 * @Varargs: args for @format 
 * 
 * Does nothing if @err is NULL; if @err is non-NULL, then *@err must
 * be NULL. A new #GError is created and assigned to *@err.
 **/
void
g_set_error (GError      **err,
             GQuark        domain,
             gint          code,
             const gchar  *format,
             ...)
{
  GError *new;
  
  va_list args;

  if (err == NULL)
    return;
  
  va_start (args, format);
  new = g_error_new_valist (domain, code, format, args);
  va_end (args);

  if (*err == NULL)
    *err = new;
  else
    g_warning (ERROR_OVERWRITTEN_WARNING, new->message);    
}

/**
 * g_propagate_error:
 * @dest: error return location
 * @src: error to move into the return location
 * 
 * If @dest is NULL, free @src; otherwise,
 * moves @src into *@dest. *@dest must be NULL.
 **/
void    
g_propagate_error (GError       **dest,
		   GError        *src)
{
  g_return_if_fail (src != NULL);
  
  if (dest == NULL)
    {
      if (src)
        g_error_free (src);
      return;
    }
  else
    {
      if (*dest != NULL)
        g_warning (ERROR_OVERWRITTEN_WARNING, src->message);
      else
        *dest = src;
    }
}

/**
 * g_clear_error:
 * @err: a #GError return location
 * 
 * If @err is NULL, does nothing. If @err is non-NULL,
 * calls g_error_free() on *@err and sets *@err to NULL.
 **/
void
g_clear_error (GError **err)
{
  if (err && *err)
    {
      g_error_free (*err);
      *err = NULL;
    }
}
