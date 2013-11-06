/* GLIB - Library of useful routines for C programming
* Copyright (C) 1995-1998  Peter Mattis, Spencer Kimball and Josh MacDonald
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

/* 
* MT safe for the unix part, FIXME: make the win32 part MT safe as well.
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
#include <errno.h>
#ifdef HAVE_PWD_H
#include <pwd.h>
#endif
#include <sys/types.h>
#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif

/* implement gutils's inline functions
*/
#define	G_IMPLEMENT_INLINES 1
#define	__G_UTILS_C__
#include "glib.h"

#ifdef	MAXPATHLEN
#define	G_PATH_LENGTH	MAXPATHLEN
#elif	defined (PATH_MAX)
#define	G_PATH_LENGTH	PATH_MAX
#elif   defined (_PC_PATH_MAX)
#define	G_PATH_LENGTH	sysconf(_PC_PATH_MAX)
#else	
#define G_PATH_LENGTH   2048
#endif

#ifdef G_PLATFORM_WIN32
#  define STRICT			/* Strict typing, please */
#  include <windows.h>
#  undef STRICT
#  include <ctype.h>
#endif /* G_PLATFORM_WIN32 */

#ifdef G_OS_WIN32
#  include <direct.h>
#endif

#ifdef HAVE_CODESET
#include <langinfo.h>
#endif

const guint glib_major_version = GLIB_MAJOR_VERSION;
const guint glib_minor_version = GLIB_MINOR_VERSION;
const guint glib_micro_version = GLIB_MICRO_VERSION;
const guint glib_interface_age = 2; // GTW - no idea, didn't seem to be set anywhere! Not set or used by ROOT.
const guint glib_binary_age = 2; // GTW - no idea, doesn't seem to be set or used by ROOT.

// Have it, but defines wrong. Scary! GTW
#if 0
#if !defined (HAVE_MEMMOVE) && !defined (HAVE_WORKING_BCOPY)
void 
	g_memmove (gpointer dest, gconstpointer src, gulong len)
{
	gchar* destptr = dest;
	const gchar* srcptr = src;
	if (src + len < dest || dest + len < src)
	{
		bcopy (src, dest, len);
		return;
	}
	else if (dest <= src)
	{
		while (len--)
			*(destptr++) = *(srcptr++);
	}
	else
	{
		destptr += len;
		srcptr += len;
		while (len--)
			*(--destptr) = *(--srcptr);
	}
}
#endif /* !HAVE_MEMMOVE && !HAVE_WORKING_BCOPY */
#endif

void
	g_atexit (GVoidFunc func)
{
	gint result;
	const gchar *error = NULL;

	/* keep this in sync with glib.h */

#ifdef	G_NATIVE_ATEXIT
	result = ATEXIT (func);
	if (result)
		error = g_strerror (errno);
#elif defined (HAVE_ATEXIT)
#  ifdef NeXT /* @#%@! NeXTStep */
	result = !atexit ((void (*)(void)) func);
	if (result)
		error = g_strerror (errno);
#  else
	result = atexit ((void (*)(void)) func);
	if (result)
		error = g_strerror (errno);
#  endif /* NeXT */
#elif defined (HAVE_ON_EXIT)
	result = on_exit ((void (*)(int, void *)) func, NULL);
	if (result)
		error = g_strerror (errno);
#else
	result = 0;
	error = "no implementation";
#endif /* G_NATIVE_ATEXIT */

	if (error)
		g_error ("Could not register atexit() function: %s", error);
}

/* Based on execvp() from GNU Libc.
* Some of this code is cut-and-pasted into gspawn.c
*/

static gchar*
	my_strchrnul (const gchar *str, gchar c)
{
	gchar *p = (gchar*)str;
	while (*p && (*p != c))
		++p;

	return p;
}

#ifdef G_OS_WIN32

gchar *inner_find_program_in_path (const gchar *program);

gchar*
	g_find_program_in_path (const gchar *program)
{
	const gchar *last_dot = strrchr (program, '.');

	if (last_dot == NULL || strchr (last_dot, '\\') != NULL)
	{
		const gint program_length = strlen (program);
		const gchar *pathext = getenv ("PATHEXT");
		const gchar *p;
		gchar *decorated_program;
		gchar *retval;

		if (pathext == NULL)
			pathext = ".com;.exe;.bat";

		p = pathext;
		do
		{
			pathext = p;
			p = my_strchrnul (pathext, ';');

			decorated_program = g_malloc (program_length + (p-pathext) + 1);
			memcpy (decorated_program, program, program_length);
			memcpy (decorated_program+program_length, pathext, p-pathext);
			decorated_program [program_length + (p-pathext)] = '\0';

			retval = inner_find_program_in_path (decorated_program);
			g_free (decorated_program);

			if (retval != NULL)
				return retval;
		} while (*p++ != '\0');
		return NULL;
	}
	else
		return inner_find_program_in_path (program);
}

#define g_find_program_in_path inner_find_program_in_path
#endif

/**
* g_find_program_in_path:
* @program: a program name
* 
* Locates the first executable named @program in the user's path, in the
* same way that execvp() would locate it. Returns an allocated string
* with the absolute path name, or NULL if the program is not found in
* the path. If @program is already an absolute path, returns a copy of
* @program if @program exists and is executable, and NULL otherwise.
* 
* On Windows, if @program does not have a file type suffix, tries to
* append the suffixes in the PATHEXT environment variable (if that
* doesn't exists, the suffixes .com, .exe, and .bat) in turn, and
* then look for the resulting file name in the same way as
* CreateProcess() would. This means first in the directory where the
* program was loaded from, then in the current directory, then in the
* Windows 32-bit system directory, then in the Windows directory, and
* finally in the directories in the PATH environment variable. If
* the program is found, the return value contains the full name
* including the type suffix.
*
* Return value: absolute path, or NULL
**/
gchar*
	g_find_program_in_path (const gchar *program)
{
	const gchar *path, *p;
	gchar *name, *freeme;
#ifdef G_OS_WIN32
	gchar *path_tmp;
#endif
	size_t len;
	size_t pathlen;

	g_return_val_if_fail (program != NULL, NULL);

	/* If it is an absolute path, or a relative path including subdirectories,
	* don't look in PATH.
	*/
	if (g_path_is_absolute (program)
		|| strchr (program, G_DIR_SEPARATOR) != NULL)
	{
		if (g_file_test (program, G_FILE_TEST_IS_EXECUTABLE))
			return g_strdup (program);
		else
			return NULL;
	}

	path = g_getenv ("PATH");
#ifdef G_OS_UNIX
	if (path == NULL)
	{
		/* There is no `PATH' in the environment.  The default
		* search path in GNU libc is the current directory followed by
		* the path `confstr' returns for `_CS_PATH'.
		*/

		/* In GLib we put . last, for security, and don't use the
		* unportable confstr(); UNIX98 does not actually specify
		* what to search if PATH is unset. POSIX may, dunno.
		*/

		path = "/bin:/usr/bin:.";
	}
#else
	{
		gchar *tmp;
		gchar moddir[MAXPATHLEN], sysdir[MAXPATHLEN], windir[MAXPATHLEN];

		GetModuleFileName (NULL, moddir, sizeof (moddir));
		tmp = g_path_get_dirname (moddir);
		GetSystemDirectory (sysdir, sizeof (sysdir));
		GetWindowsDirectory (windir, sizeof (windir));
		path_tmp = g_strconcat (tmp, ";.;", sysdir, ";", windir,
			(path != NULL ? ";" : NULL),
			(path != NULL ? path : NULL),
			NULL);
		g_free (tmp);
		path = path_tmp;
	}
#endif

	len = strlen (program) + 1;
	pathlen = strlen (path);
	freeme = name = g_malloc (pathlen + len + 1);

	/* Copy the file name at the top, including '\0'  */
	memcpy (name + pathlen + 1, program, len);
	name = name + pathlen;
	/* And add the slash before the filename  */
	*name = G_DIR_SEPARATOR;

	p = path;
	do
	{
		char *startp;

		path = p;
		p = my_strchrnul (path, G_SEARCHPATH_SEPARATOR);

		if (p == path)
			/* Two adjacent colons, or a colon at the beginning or the end
			* of `PATH' means to search the current directory.
			*/
			startp = name + 1;
		else
			startp = memcpy (name - (p - path), path, p - path);

		if (g_file_test (startp, G_FILE_TEST_IS_EXECUTABLE))
		{
			gchar *ret;
			ret = g_strdup (startp);
			g_free (freeme);
#ifdef G_OS_WIN32
			g_free (path_tmp);
#endif
			return ret;
		}
	}
	while (*p++ != '\0');

	g_free (freeme);
#ifdef G_OS_WIN32
	g_free (path_tmp);
#endif

	return NULL;
}

gint
	g_snprintf (gchar	*str,
	gulong	 n,
	gchar const *fmt,
	...)
{
#ifdef	HAVE_VSNPRINTF
	va_list args;
	gint retval;

	g_return_val_if_fail (str != NULL, 0);
	g_return_val_if_fail (n > 0, 0);
	g_return_val_if_fail (fmt != NULL, 0);

	va_start (args, fmt);
	retval = vsnprintf (str, n, fmt, args);
	va_end (args);

	if (retval < 0)
	{
		str[n-1] = '\0';
		retval = strlen (str);
	}

	return retval;
#else	/* !HAVE_VSNPRINTF */
	gchar *printed;
	va_list args;

	g_return_val_if_fail (str != NULL, 0);
	g_return_val_if_fail (n > 0, 0);
	g_return_val_if_fail (fmt != NULL, 0);

	va_start (args, fmt);
	printed = g_strdup_vprintf (fmt, args);
	va_end (args);

	strncpy (str, printed, n);
	str[n-1] = '\0';

	g_free (printed);

	return strlen (str);
#endif	/* !HAVE_VSNPRINTF */
}

gint
	g_vsnprintf (gchar	 *str,
	gulong	  n,
	gchar const *fmt,
	va_list      args)
{
#ifdef	HAVE_VSNPRINTF
	gint retval;

	g_return_val_if_fail (str != NULL, 0);
	g_return_val_if_fail (n > 0, 0);
	g_return_val_if_fail (fmt != NULL, 0);

	retval = vsnprintf (str, n, fmt, args);

	if (retval < 0)
	{
		str[n-1] = '\0';
		retval = strlen (str);
	}

	return retval;
#else	/* !HAVE_VSNPRINTF */
	gchar *printed;

	g_return_val_if_fail (str != NULL, 0);
	g_return_val_if_fail (n > 0, 0);
	g_return_val_if_fail (fmt != NULL, 0);

	printed = g_strdup_vprintf (fmt, args);
	strncpy (str, printed, n);
	str[n-1] = '\0';

	g_free (printed);

	return strlen (str);
#endif /* !HAVE_VSNPRINTF */
}

guint	     
	g_parse_debug_string  (const gchar     *string, 
	const GDebugKey *keys, 
	guint	        nkeys)
{
	guint i;
	guint result = 0;

	g_return_val_if_fail (string != NULL, 0);

	if (!g_ascii_strcasecmp (string, "all"))
	{
		for (i=0; i<nkeys; i++)
			result |= keys[i].value;
	}
	else
	{
		const gchar *p = string;
		const gchar *q;
		gboolean done = FALSE;

		while (*p && !done)
		{
			q = strchr (p, ':');
			if (!q)
			{
				q = p + strlen(p);
				done = TRUE;
			}

			for (i=0; i<nkeys; i++)
				if (g_ascii_strncasecmp(keys[i].key, p, q - p) == 0 &&
					keys[i].key[q - p] == '\0')
					result |= keys[i].value;

			p = q + 1;
		}
	}

	return result;
}

G_CONST_RETURN gchar*
	g_basename (const gchar	   *file_name)
{
	register gchar *base;

	g_return_val_if_fail (file_name != NULL, NULL);

	base = strrchr (file_name, G_DIR_SEPARATOR);
	if (base)
		return base + 1;

#ifdef G_OS_WIN32
	if (g_ascii_isalpha (file_name[0]) && file_name[1] == ':')
		return (gchar*) file_name + 2;
#endif /* G_OS_WIN32 */

	return (gchar*) file_name;
}

gchar*
	g_path_get_basename (const gchar   *file_name)
{
	register gssize base;             
	register gssize last_nonslash;    
	gsize len;    
	gchar *retval;

	g_return_val_if_fail (file_name != NULL, NULL);

	if (file_name[0] == '\0')
		/* empty string */
			return g_strdup (".");

	last_nonslash = strlen (file_name) - 1;

	while (last_nonslash >= 0 && file_name [last_nonslash] == G_DIR_SEPARATOR)
		last_nonslash--;

	if (last_nonslash == -1)
		/* string only containing slashes */
			return g_strdup (G_DIR_SEPARATOR_S);

#ifdef G_OS_WIN32
	if (last_nonslash == 1 && g_ascii_isalpha (file_name[0]) && file_name[1] == ':')
		/* string only containing slashes and a drive */
			return g_strdup (G_DIR_SEPARATOR_S);
#endif /* G_OS_WIN32 */

	base = last_nonslash;

	while (base >=0 && file_name [base] != G_DIR_SEPARATOR)
		base--;

#ifdef G_OS_WIN32
	if (base == -1 && g_ascii_isalpha (file_name[0]) && file_name[1] == ':')
		base = 1;
#endif /* G_OS_WIN32 */

	len = last_nonslash - base;
	retval = g_malloc (len + 1);
	memcpy (retval, file_name + base + 1, len);
	retval [len] = '\0';
	return retval;
}

gboolean
	g_path_is_absolute (const gchar *file_name)
{
	g_return_val_if_fail (file_name != NULL, FALSE);

	if (file_name[0] == G_DIR_SEPARATOR
#ifdef G_OS_WIN32
		|| file_name[0] == '/'
#endif
		)
		return TRUE;

#ifdef G_OS_WIN32
	/* Recognize drive letter on native Windows */
	if (g_ascii_isalpha (file_name[0]) && file_name[1] == ':' && (file_name[2] == G_DIR_SEPARATOR || file_name[2] == '/'))
		return TRUE;
#endif /* G_OS_WIN32 */

	return FALSE;
}

G_CONST_RETURN gchar*
	g_path_skip_root (const gchar *file_name)
{
	g_return_val_if_fail (file_name != NULL, NULL);

#ifdef G_PLATFORM_WIN32
	/* Skip \\server\share (Win32) or //server/share (Cygwin) */
	if (file_name[0] == G_DIR_SEPARATOR &&
		file_name[1] == G_DIR_SEPARATOR &&
		file_name[2])
	{
		gchar *p;

		if ((p = strchr (file_name + 2, G_DIR_SEPARATOR)) > file_name + 2 &&
			p[1])
		{
			file_name = p + 1;

			while (file_name[0] && file_name[0] != G_DIR_SEPARATOR)
				file_name++;

			/* Possibly skip a backslash after the share name */
			if (file_name[0] == G_DIR_SEPARATOR)
				file_name++;

			return (gchar *)file_name;
		}
	}
#endif

	/* Skip initial slashes */
	if (file_name[0] == G_DIR_SEPARATOR)
	{
		while (file_name[0] == G_DIR_SEPARATOR)
			file_name++;
		return (gchar *)file_name;
	}

#ifdef G_OS_WIN32
	/* Skip X:\ */
	if (g_ascii_isalpha (file_name[0]) && file_name[1] == ':' && file_name[2] == G_DIR_SEPARATOR)
		return (gchar *)file_name + 3;
#endif

	return NULL;
}

gchar*
	g_path_get_dirname (const gchar	   *file_name)
{
	register gchar *base;
	register gsize len;    

	g_return_val_if_fail (file_name != NULL, NULL);

	base = strrchr (file_name, G_DIR_SEPARATOR);
	if (!base)
		return g_strdup (".");
	while (base > file_name && *base == G_DIR_SEPARATOR)
		base--;
	len = (guint) 1 + base - file_name;

	base = g_new (gchar, len + 1);
	g_memmove (base, file_name, len);
	base[len] = 0;

	return base;
}

gchar*
	g_get_current_dir (void)
{
	gchar *buffer = NULL;
	gchar *dir = NULL;
	static gulong max_len = 0;

	if (max_len == 0) 
		max_len = (G_PATH_LENGTH == -1) ? 2048 : G_PATH_LENGTH;

	/* We don't use getcwd(3) on SUNOS, because, it does a popen("pwd")
	* and, if that wasn't bad enough, hangs in doing so.
	*/
#if	(defined (sun) && !defined (__SVR4)) || !defined(HAVE_GETCWD)
	buffer = g_new (gchar, max_len + 1);
	*buffer = 0;
	dir = getwd (buffer);
#else	/* !sun || !HAVE_GETCWD */
	while (max_len < G_MAXULONG / 2)
	{
		buffer = g_new (gchar, max_len + 1);
		*buffer = 0;
		dir = getcwd (buffer, max_len);

		if (dir || errno != ERANGE)
			break;

		g_free (buffer);
		max_len *= 2;
	}
#endif	/* !sun || !HAVE_GETCWD */

	if (!dir || !*buffer)
	{
		/* hm, should we g_error() out here?
		* this can happen if e.g. "./" has mode \0000
		*/
		buffer[0] = G_DIR_SEPARATOR;
		buffer[1] = 0;
	}

	dir = g_strdup (buffer);
	g_free (buffer);

	return dir;
}

G_CONST_RETURN gchar*
	g_getenv (const gchar *variable)
{
#ifndef G_OS_WIN32
	g_return_val_if_fail (variable != NULL, NULL);

	return getenv (variable);
#else
	G_LOCK_DEFINE_STATIC (getenv);
	struct env_struct
	{
		gchar *key;
		gchar *value;
	} *env;
	static GArray *environs = NULL;
	gchar *system_env;
	guint length, i;
	gchar dummy[2];

	g_return_val_if_fail (variable != NULL, NULL);

	G_LOCK (getenv);

	if (!environs)
		environs = g_array_new (FALSE, FALSE, sizeof (struct env_struct));

	/* First we try to find the envinronment variable inside the already
	* found ones.
	*/

	for (i = 0; i < environs->len; i++)
	{
		env = &g_array_index (environs, struct env_struct, i);
		if (strcmp (env->key, variable) == 0)
		{
			g_assert (env->value);
			G_UNLOCK (getenv);
			return env->value;
		}
	}

	/* If not found, we ask the system */

	system_env = getenv (variable);
	if (!system_env)
	{
		G_UNLOCK (getenv);
		return NULL;
	}

	/* On Windows NT, it is relatively typical that environment variables
	* contain references to other environment variables. Handle that by
	* calling ExpandEnvironmentStrings.
	*/

	g_array_set_size (environs, environs->len + 1);

	env = &g_array_index (environs, struct env_struct, environs->len - 1);

	/* First check how much space we need */
	length = ExpandEnvironmentStrings (system_env, dummy, 2);

	/* Then allocate that much, and actualy do the expansion and insert
	* the new found pair into our buffer 
	*/

	env->value = g_malloc (length);
	env->key = g_strdup (variable);

	ExpandEnvironmentStrings (system_env, env->value, length);

	G_UNLOCK (getenv);
	return env->value;
#endif
}


G_LOCK_DEFINE_STATIC (g_utils_global);

static	gchar	*g_tmp_dir = NULL;
static	gchar	*g_user_name = NULL;
static	gchar	*g_real_name = NULL;
static	gchar	*g_home_dir = NULL;

/* HOLDS: g_utils_global_lock */
static void
	g_get_any_init (void)
{
	if (!g_tmp_dir)
	{
		g_tmp_dir = g_strdup (g_getenv ("TMPDIR"));
		if (!g_tmp_dir)
			g_tmp_dir = g_strdup (g_getenv ("TMP"));
		if (!g_tmp_dir)
			g_tmp_dir = g_strdup (g_getenv ("TEMP"));

#ifdef P_tmpdir
		if (!g_tmp_dir)
		{
			gsize k;    
			g_tmp_dir = g_strdup (P_tmpdir);
			k = strlen (g_tmp_dir);
			if (k > 1 && g_tmp_dir[k - 1] == G_DIR_SEPARATOR)
				g_tmp_dir[k - 1] = '\0';
		}
#endif

		if (!g_tmp_dir)
		{
#ifndef G_OS_WIN32
			g_tmp_dir = g_strdup ("/tmp");
#else /* G_OS_WIN32 */
			g_tmp_dir = g_strdup ("C:\\");
#endif /* G_OS_WIN32 */
		}

		if (!g_home_dir)
			g_home_dir = g_strdup (g_getenv ("HOME"));

#ifdef G_OS_WIN32
		/* In case HOME is Unix-style (it happens), convert it to
		* Windows style.
		*/
		if (g_home_dir)
		{
			gchar *p;
			while ((p = strchr (g_home_dir, '/')) != NULL)
				*p = '\\';
		}

		if (!g_home_dir)
		{
			/* USERPROFILE is probably the closest equivalent to $HOME? */
			if (getenv ("USERPROFILE") != NULL)
				g_home_dir = g_strdup (g_getenv ("USERPROFILE"));
		}

		if (!g_home_dir)
		{
			/* At least at some time, HOMEDRIVE and HOMEPATH were used
			* to point to the home directory, I think. But on Windows
			* 2000 HOMEDRIVE seems to be equal to SYSTEMDRIVE, and
			* HOMEPATH is its root "\"?
			*/
			if (getenv ("HOMEDRIVE") != NULL && getenv ("HOMEPATH") != NULL)
			{
				gchar *homedrive, *homepath;

				homedrive = g_strdup (g_getenv ("HOMEDRIVE"));
				homepath = g_strdup (g_getenv ("HOMEPATH"));

				g_home_dir = g_strconcat (homedrive, homepath, NULL);
				g_free (homedrive);
				g_free (homepath);
			}
		}
#endif /* G_OS_WIN32 */

#ifdef HAVE_PWD_H
		{
			struct passwd *pw = NULL;
			gpointer buffer = NULL;
			gint error;

#  if defined (HAVE_POSIX_GETPWUID_R) || defined (HAVE_NONPOSIX_GETPWUID_R)
			struct passwd pwd;
#    ifdef _SC_GETPW_R_SIZE_MAX  
			/* This reurns the maximum length */
			glong bufsize = sysconf (_SC_GETPW_R_SIZE_MAX);

			if (bufsize < 0)
				bufsize = 64;
#    else /* _SC_GETPW_R_SIZE_MAX */
			glong bufsize = 64;
#    endif /* _SC_GETPW_R_SIZE_MAX */

			do
			{
				g_free (buffer);
				buffer = g_malloc (bufsize);
				errno = 0;

#    ifdef HAVE_POSIX_GETPWUID_R
				error = getpwuid_r (getuid (), &pwd, buffer, bufsize, &pw);
				error = error < 0 ? errno : error;
#    else /* HAVE_NONPOSIX_GETPWUID_R */
#      ifdef _AIX
				error = getpwuid_r (getuid (), &pwd, buffer, bufsize);
				pw = error == 0 ? &pwd : NULL;
#      else /* !_AIX */
				pw = getpwuid_r (getuid (), &pwd, buffer, bufsize);
				error = pw ? 0 : errno;
#      endif /* !_AIX */            
#    endif /* HAVE_NONPOSIX_GETPWUID_R */

				if (!pw)
				{
					/* we bail out prematurely if the user id can't be found
					* (should be pretty rare case actually), or if the buffer
					* should be sufficiently big and lookups are still not
					* successfull.
					*/
					if (error == 0 || error == ENOENT)
					{
						g_warning ("getpwuid_r(): failed due to unknown user id (%lu)",
							(gulong) getuid ());
						break;
					}
					if (bufsize > 32 * 1024)
					{
						g_warning ("getpwuid_r(): failed due to: %s.",
							g_strerror (error));
						break;
					}

					bufsize *= 2;
				}
			}
			while (!pw);
#  endif /* HAVE_POSIX_GETPWUID_R || HAVE_NONPOSIX_GETPWUID_R */

			if (!pw)
			{
				setpwent ();
				pw = getpwuid (getuid ());
				endpwent ();
			}
			if (pw)
			{
				g_user_name = g_strdup (pw->pw_name);
				g_real_name = g_strdup (pw->pw_gecos);
				if (!g_home_dir)
					g_home_dir = g_strdup (pw->pw_dir);
			}
			g_free (buffer);
		}

#else /* !HAVE_PWD_H */

#  ifdef G_OS_WIN32
		{
			guint len = 17;
			gchar buffer[17];

			if (GetUserName ((LPTSTR) buffer, (LPDWORD) &len))
			{
				g_user_name = g_strdup (buffer);
				g_real_name = g_strdup (buffer);
			}
		}
#  endif /* G_OS_WIN32 */

#endif /* !HAVE_PWD_H */

#ifdef __EMX__
		/* change '\\' in %HOME% to '/' */
		g_strdelimit (g_home_dir, "\\",'/');
#endif
		if (!g_user_name)
			g_user_name = g_strdup ("somebody");
		if (!g_real_name)
			g_real_name = g_strdup ("Unknown");
		else
		{
			gchar *p;

			for (p = g_real_name; *p; p++)
				if (*p == ',')
				{
					*p = 0;
					p = g_strdup (g_real_name);
					g_free (g_real_name);
					g_real_name = p;
					break;
				}
		}
	}
}

G_CONST_RETURN gchar*
	g_get_user_name (void)
{
	G_LOCK (g_utils_global);
	if (!g_tmp_dir)
		g_get_any_init ();
	G_UNLOCK (g_utils_global);

	return g_user_name;
}

G_CONST_RETURN gchar*
	g_get_real_name (void)
{
	G_LOCK (g_utils_global);
	if (!g_tmp_dir)
		g_get_any_init ();
	G_UNLOCK (g_utils_global);

	return g_real_name;
}

/* Return the home directory of the user. If there is a HOME
* environment variable, its value is returned, otherwise use some
* system-dependent way of finding it out. If no home directory can be
* deduced, return NULL.
*/

G_CONST_RETURN gchar*
	g_get_home_dir (void)
{
	G_LOCK (g_utils_global);
	if (!g_tmp_dir)
		g_get_any_init ();
	G_UNLOCK (g_utils_global);

	return g_home_dir;
}

/* Return a directory to be used to store temporary files. This is the
* value of the TMPDIR, TMP or TEMP environment variables (they are
* checked in that order). If none of those exist, use P_tmpdir from
* stdio.h.  If that isn't defined, return "/tmp" on POSIXly systems,
* and C:\ on Windows.
*/

G_CONST_RETURN gchar*
	g_get_tmp_dir (void)
{
	G_LOCK (g_utils_global);
	if (!g_tmp_dir)
		g_get_any_init ();
	G_UNLOCK (g_utils_global);

	return g_tmp_dir;
}

static gchar *g_prgname = NULL;

gchar*
	g_get_prgname (void)
{
	gchar* retval;

	G_LOCK (g_utils_global);
	retval = g_prgname;
	G_UNLOCK (g_utils_global);

	return retval;
}

void
	g_set_prgname (const gchar *prgname)
{
	gchar *c;

	G_LOCK (g_utils_global);
	c = g_prgname;
	g_prgname = g_strdup (prgname);
	g_free (c);
	G_UNLOCK (g_utils_global);
}

guint
	g_direct_hash (gconstpointer v)
{
	return GPOINTER_TO_UINT (v);
}

gboolean
	g_direct_equal (gconstpointer v1,
	gconstpointer v2)
{
	return v1 == v2;
}

gboolean
	g_int_equal (gconstpointer v1,
	gconstpointer v2)
{
	return *((const gint*) v1) == *((const gint*) v2);
}

guint
	g_int_hash (gconstpointer v)
{
	return *(const gint*) v;
}

/**
* g_nullify_pointer:
* @nullify_location: the memory address of the pointer.
* 
* Set the pointer at the specified location to %NULL.
**/
void
	g_nullify_pointer (gpointer *nullify_location)
{
	g_return_if_fail (nullify_location != NULL);

	*nullify_location = NULL;
}

/**
* g_get_codeset:
* 
* Get the codeset for the current locale.
* 
* Return value: a newly allocated string containing the name
* of the codeset. This string must be freed with g_free().
**/
gchar *
	g_get_codeset (void)
{
#ifdef HAVE_CODESET  
	char *result = nl_langinfo (CODESET);
	return g_strdup (result);
#else
#ifdef G_PLATFORM_WIN32
	return g_strdup_printf ("CP%d", GetACP ());
#else
	/* FIXME: Do something more intelligent based on setlocale (LC_CTYPE, NULL)
	*/
	return g_strdup ("ISO-8859-1");
#endif
#endif
}

#ifdef ENABLE_NLS

#include <libintl.h>


#ifdef G_OS_WIN32

/* On Windows we don't want any hard-coded path names */

#undef GLIB_LOCALE_DIR
/* It's OK to leak the g_win32_get_...() and g_strdup_printf() results
* below, as this macro is called only once. */
#define GLIB_LOCALE_DIR					      	\
	g_win32_get_package_installation_subdirectory			\
	(GETTEXT_PACKAGE,						\
	g_strdup_printf ("libglib-%d.%d-%d.dll",			\
	GLIB_MAJOR_VERSION,				\
	GLIB_MINOR_VERSION,				\
	GLIB_MICRO_VERSION - GLIB_BINARY_AGE),	\
	"share\\locale")

#endif /* !G_OS_WIN32 */

G_CONST_RETURN gchar *
	_glib_gettext (const gchar *str)
{
	static gboolean _glib_gettext_initialized = FALSE;

	if (!_glib_gettext_initialized)
	{
		bindtextdomain(GETTEXT_PACKAGE, GLIB_LOCALE_DIR);
		_glib_gettext_initialized = TRUE;
	}

	return dgettext (GETTEXT_PACKAGE, str);
}

#endif /* ENABLE_NLS */


