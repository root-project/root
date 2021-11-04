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

#ifndef __G_UTILS_H__
#define __G_UTILS_H__

#include <glib/gtypes.h>
#include <stdarg.h>

G_BEGIN_DECLS

#ifdef G_OS_WIN32

/* On native Win32, directory separator is the backslash, and search path
 * separator is the semicolon.
 */
#define G_DIR_SEPARATOR '\\'
#define G_DIR_SEPARATOR_S "\\"
#define G_SEARCHPATH_SEPARATOR ';'
#define G_SEARCHPATH_SEPARATOR_S ";"

#else  /* !G_OS_WIN32 */

/* Unix */

#define G_DIR_SEPARATOR '/'
#define G_DIR_SEPARATOR_S "/"
#define G_SEARCHPATH_SEPARATOR ':'
#define G_SEARCHPATH_SEPARATOR_S ":"

#endif /* !G_OS_WIN32 */

/* Define G_VA_COPY() to do the right thing for copying va_list variables.
 * glibconfig.h may have already defined G_VA_COPY as va_copy or __va_copy.
 */
#if !defined (G_VA_COPY)
#  if defined (__GNUC__) && defined (__PPC__) && (defined (_CALL_SYSV) || defined (_WIN32))
#    define G_VA_COPY(ap1, ap2)	  (*(ap1) = *(ap2))
#  elif defined (G_VA_COPY_AS_ARRAY)
#    define G_VA_COPY(ap1, ap2)	  g_memmove ((ap1), (ap2), sizeof (va_list))
#  else /* va_list is a pointer */
#    define G_VA_COPY(ap1, ap2)	  ((ap1) = (ap2))
#  endif /* va_list is a pointer */
#endif /* !G_VA_COPY */

/* inlining hassle. for compilers that don't allow the `inline' keyword,
 * mostly because of strict ANSI C compliance or dumbness, we try to fall
 * back to either `__inline__' or `__inline'.
 * we define G_CAN_INLINE, if the compiler seems to be actually
 * *capable* to do function inlining, in which case inline function bodys
 * do make sense. we also define G_INLINE_FUNC to properly export the
 * function prototypes if no inlining can be performed.
 * inline function bodies have to be special cased with G_CAN_INLINE and a
 * .c file specific macro to allow one compiled instance with extern linkage
 * of the functions by defining G_IMPLEMENT_INLINES and the .c file macro.
 */
#ifdef G_IMPLEMENT_INLINES
#  define G_INLINE_FUNC extern
#  undef  G_CAN_INLINE
#endif
#ifndef G_INLINE_FUNC
#  define G_CAN_INLINE 1
#endif
#if defined (G_HAVE_INLINE) && defined (__GNUC__) && defined (__STRICT_ANSI__)
#  undef inline
#  define inline __inline__
#elif !defined (G_HAVE_INLINE)
#  undef inline
#  if defined (G_HAVE___INLINE__)
#    define inline __inline__
#  elif defined (G_HAVE___INLINE)
#    define inline __inline
#  else /* !inline && !__inline__ && !__inline */
#    define inline  /* don't inline, then */
#    ifndef G_INLINE_FUNC
#      undef G_CAN_INLINE
#    endif
#  endif
#endif
#ifndef G_INLINE_FUNC
#  if defined (__GNUC__) && (__OPTIMIZE__)
#    define G_INLINE_FUNC extern inline
#  elif defined (G_CAN_INLINE) && !defined (__GNUC__)
#    define G_INLINE_FUNC static inline
#  else /* can't inline */
#    define G_INLINE_FUNC extern
#    undef G_CAN_INLINE
#  endif
#endif /* !G_INLINE_FUNC */

/* Retrive static string info
 */
G_CONST_RETURN gchar* g_get_user_name      (void);
G_CONST_RETURN gchar* g_get_real_name      (void);
G_CONST_RETURN gchar* g_get_home_dir       (void);
G_CONST_RETURN gchar* g_get_tmp_dir        (void);
gchar*                g_get_prgname        (void);
void                  g_set_prgname        (const gchar *prgname);


typedef struct _GDebugKey	GDebugKey;
struct _GDebugKey
{
  gchar *key;
  guint	 value;
};

/* Miscellaneous utility functions
 */
guint                 g_parse_debug_string (const gchar     *string,
					    const GDebugKey *keys,
					    guint            nkeys);

gint                  g_snprintf           (gchar       *string,
					    gulong       n,
					    gchar const *format,
					    ...) G_GNUC_PRINTF (3, 4);
gint                  g_vsnprintf          (gchar       *string,
					    gulong       n,
					    gchar const *format,
					    va_list      args);

/* Check if a file name is an absolute path */
gboolean              g_path_is_absolute   (const gchar *file_name);

/* In case of absolute paths, skip the root part */
G_CONST_RETURN gchar* g_path_skip_root     (const gchar *file_name);

#ifndef G_DISABLE_DEPRECATED

/* These two functions are deprecated and will be removed in the next
 * major release of GLib. Use g_path_get_dirname/g_path_get_basename
 * instead. Whatch out! The string returned by g_path_get_basename
 * must be g_freed, while the string returned by g_basename must not.*/
G_CONST_RETURN gchar* g_basename           (const gchar *file_name);
#define g_dirname g_path_get_dirname

#endif /* G_DISABLE_DEPRECATED */

/* The returned strings are newly allocated with g_malloc() */
gchar*                g_get_current_dir    (void);
gchar*                g_path_get_basename  (const gchar *file_name);
gchar*                g_path_get_dirname   (const gchar *file_name);


/* Set the pointer at the specified location to NULL */
void                  g_nullify_pointer    (gpointer    *nullify_location);

/* Get the codeset for the current locale */
/* gchar * g_get_codeset    (void); */

/* return the environment string for the variable. The returned memory
 * must not be freed. */
G_CONST_RETURN gchar* g_getenv             (const gchar *variable);


/* we try to provide a usefull equivalent for ATEXIT if it is
 * not defined, but use is actually abandoned. people should
 * use g_atexit() instead.
 */
typedef	void		(*GVoidFunc)		(void);
#ifndef ATEXIT
# define ATEXIT(proc)	g_ATEXIT(proc)
#else
# define G_NATIVE_ATEXIT
#endif /* ATEXIT */
/* we use a GLib function as a replacement for ATEXIT, so
 * the programmer is not required to check the return value
 * (if there is any in the implementation) and doesn't encounter
 * missing include files.
 */
void	g_atexit		(GVoidFunc    func);

/* Look for an executable in PATH, following execvp() rules */
gchar*  g_find_program_in_path  (const gchar *program);

/* Bit tests
 */
G_INLINE_FUNC gint	g_bit_nth_lsf (gulong  mask,
				       gint    nth_bit);
G_INLINE_FUNC gint	g_bit_nth_msf (gulong  mask,
				       gint    nth_bit);
G_INLINE_FUNC guint	g_bit_storage (gulong  number);

/* Trash Stacks
 * elements need to be >= sizeof (gpointer)
 */
typedef struct _GTrashStack     GTrashStack;
struct _GTrashStack
{
  GTrashStack *next;
};

G_INLINE_FUNC void	g_trash_stack_push	(GTrashStack **stack_p,
						 gpointer      data_p);
G_INLINE_FUNC gpointer	g_trash_stack_pop	(GTrashStack **stack_p);
G_INLINE_FUNC gpointer	g_trash_stack_peek	(GTrashStack **stack_p);
G_INLINE_FUNC guint	g_trash_stack_height	(GTrashStack **stack_p);

/* inline function implementations
 */
#if defined (G_CAN_INLINE) || defined (__G_UTILS_C__)
G_INLINE_FUNC gint
g_bit_nth_lsf (gulong mask,
	       gint   nth_bit)
{
  do
    {
      nth_bit++;
      if (mask & ((gulong)1 << (gulong) nth_bit))
	return nth_bit;
    }
  while (nth_bit < 32);
  return -1;
}
G_INLINE_FUNC gint
g_bit_nth_msf (gulong mask,
	       gint   nth_bit)
{
  if (nth_bit < 0)
    nth_bit = GLIB_SIZEOF_LONG * 8;
  do
    {
      nth_bit--;
      if (mask & ((gulong)1 << (gulong) nth_bit))
	return nth_bit;
    }
  while (nth_bit > 0);
  return -1;
}
G_INLINE_FUNC guint
g_bit_storage (gulong number)
{
  /*register*/ guint n_bits = 0;
  
  do
    {
      n_bits++;
      number >>= 1;
    }
  while (number);
  return n_bits;
}
G_INLINE_FUNC void
g_trash_stack_push (GTrashStack **stack_p,
		    gpointer      data_p)
{
  GTrashStack *data = (GTrashStack *) data_p;

  data->next = *stack_p;
  *stack_p = data;
}
G_INLINE_FUNC gpointer
g_trash_stack_pop (GTrashStack **stack_p)
{
  GTrashStack *data;

  data = *stack_p;
  if (data)
    {
      *stack_p = data->next;
      /* NULLify private pointer here, most platforms store NULL as
       * subsequent 0 bytes
       */
      data->next = NULL;
    }

  return data;
}
G_INLINE_FUNC gpointer
g_trash_stack_peek (GTrashStack **stack_p)
{
  GTrashStack *data;

  data = *stack_p;

  return data;
}
G_INLINE_FUNC guint
g_trash_stack_height (GTrashStack **stack_p)
{
  GTrashStack *data;
  guint i = 0;

  for (data = *stack_p; data; data = data->next)
    i++;

  return i;
}
#endif  /* G_CAN_INLINE || __G_UTILS_C__ */

/* Glib version.
 * we prefix variable declarations so they can
 * properly get exported in windows dlls.
 */
GLIB_VAR const guint glib_major_version;
GLIB_VAR const guint glib_minor_version;
GLIB_VAR const guint glib_micro_version;
GLIB_VAR const guint glib_interface_age;
GLIB_VAR const guint glib_binary_age;

#define GLIB_CHECK_VERSION(major,minor,micro)    \
    (GLIB_MAJOR_VERSION > (major) || \
     (GLIB_MAJOR_VERSION == (major) && GLIB_MINOR_VERSION > (minor)) || \
     (GLIB_MAJOR_VERSION == (major) && GLIB_MINOR_VERSION == (minor) && \
      GLIB_MICRO_VERSION >= (micro)))

G_END_DECLS

#endif /* __G_UTILS_H__ */



