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

#ifndef __G_STRING_H__
#define __G_STRING_H__

#include <g_types.h>

G_BEGIN_DECLS

typedef struct _GString		GString;
typedef struct _GStringChunk	GStringChunk;

struct _GString
{
  gchar *str;
  gint len;
};

/* String Chunks
 */
GStringChunk* g_string_chunk_new	   (gint size);
void	      g_string_chunk_free	   (GStringChunk *chunk);
gchar*	      g_string_chunk_insert	   (GStringChunk *chunk,
					    const gchar	 *string);
gchar*	      g_string_chunk_insert_const  (GStringChunk *chunk,
					    const gchar	 *string);


/* Strings
 */
GString*     g_string_new	        (const gchar	 *init);
GString*     g_string_new_len           (const gchar     *init,
                                         gint             len);
GString*     g_string_sized_new         (guint		  dfl_size);
gchar*	     g_string_free	        (GString	 *string,
					 gboolean	  free_segment);
gboolean     g_string_equal             (const GString	 *v,
					 const GString 	 *v2);
guint        g_string_hash              (const GString   *str);
GString*     g_string_assign            (GString	 *string,
					 const gchar	 *rval);
GString*     g_string_truncate          (GString	 *string,
					 guint		  len);
GString*     g_string_insert_len        (GString         *string,
                                         gint             pos,
                                         const gchar     *val,
                                         gint             len);
GString*     g_string_append            (GString	 *string,
			                 const gchar	 *val);
GString*     g_string_append_len        (GString	 *string,
			                 const gchar	 *val,
                                         gint             len);
GString*     g_string_append_c          (GString	 *string,
					 gchar		  c);
GString*     g_string_prepend           (GString	 *string,
					 const gchar	 *val);
GString*     g_string_prepend_c         (GString	 *string,
					 gchar		  c);
GString*     g_string_prepend_len       (GString	 *string,
			                 const gchar	 *val,
                                         gint             len);
GString*     g_string_insert            (GString	 *string,
					 gint		  pos,
					 const gchar	 *val);
GString*     g_string_insert_c          (GString	 *string,
					 gint		  pos,
					 gchar		  c);
GString*     g_string_erase	        (GString	 *string,
					 gint		  pos,
					 gint		  len);
GString*     g_string_down              (GString	 *string);
GString*     g_string_up                (GString	 *string);
void         g_string_sprintf           (GString	 *string,
					 const gchar	 *format,
					 ...) G_GNUC_PRINTF (2, 3);
void         g_string_sprintfa          (GString	 *string,
					 const gchar	 *format,
					 ...) G_GNUC_PRINTF (2, 3);

G_END_DECLS

#endif /* __G_STRING_H__ */

