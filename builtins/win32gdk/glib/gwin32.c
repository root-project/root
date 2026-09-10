/* GLIB - Library of useful routines for C programming
 * Copyright (C) 1995-1998  Peter Mattis, Spencer Kimball and Josh MacDonald
 * Copyright (C) 1998-1999  Tor Lillqvist
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

#include "glibconfig.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#define STRICT			/* Strict typing, please */
#include <windows.h>
#undef STRICT
#ifndef G_WITH_CYGWIN
#include <direct.h>
#endif
#include <errno.h>
#include <ctype.h>
#ifdef _MSC_VER
#  include <io.h>
#endif /* _MSC_VER */

#include "glib.h"

#ifdef G_WITH_CYGWIN
#include <sys/cygwin.h>
#endif

#ifndef G_WITH_CYGWIN

gint
g_win32_ftruncate (gint  fd,
		   guint size)
{
  HANDLE hfile;
  guint curpos;

  g_return_val_if_fail (fd >= 0, -1);
  
  hfile = (HANDLE) _get_osfhandle (fd);
  curpos = SetFilePointer (hfile, 0, NULL, FILE_CURRENT);
  if (curpos == 0xFFFFFFFF
      || SetFilePointer (hfile, size, NULL, FILE_BEGIN) == 0xFFFFFFFF
      || !SetEndOfFile (hfile))
    {
      gint error = GetLastError ();

      switch (error)
	{
	case ERROR_INVALID_HANDLE:
	  errno = EBADF;
	  break;
	default:
	  errno = EIO;
	  break;
	}

      return -1;
    }

  return 0;
}

DIR*
g_win32_opendir (const char *dirname)
{
  DIR *result;
  gchar *mask;
  guint k;

  g_return_val_if_fail (dirname != NULL, NULL);

  result = g_new0 (DIR, 1);
  result->find_file_data = g_new0 (WIN32_FIND_DATA, 1);
  result->dir_name = g_strdup (dirname);
  
  k = strlen (result->dir_name);
  if (k && result->dir_name[k - 1] == '\\')
    {
      result->dir_name[k - 1] = '\0';
    }
  mask = g_strdup_printf ("%s\\*", result->dir_name);

  result->find_file_handle = (gulong) FindFirstFile (mask,
					     (LPWIN32_FIND_DATA) result->find_file_data);
  g_free (mask);

  if (result->find_file_handle == (gulong) INVALID_HANDLE_VALUE)
    {
      int error = GetLastError ();

      g_free (result->dir_name);
      g_free (result->find_file_data);
      g_free (result);
      switch (error)
	{
	default:
	  errno = EIO;
	  return NULL;
	}
    }
  result->just_opened = TRUE;

  return result;
}

struct dirent*
g_win32_readdir (DIR *dir)
{
  gchar *basename;

  g_return_val_if_fail (dir != NULL, NULL);

  if (dir->just_opened)
    dir->just_opened = FALSE;
  else
    {
      if (!FindNextFile ((HANDLE) dir->find_file_handle,
			 (LPWIN32_FIND_DATA) dir->find_file_data))
	{
	  int error = GetLastError ();

	  switch (error)
	    {
	    case ERROR_NO_MORE_FILES:
	      return NULL;
	    default:
	      errno = EIO;
	      return NULL;
	    }
	}
    }
  
  basename = g_path_get_basename (((LPWIN32_FIND_DATA) dir->find_file_data)->cFileName);

  strcpy (dir->readdir_result.d_name, basename);

  g_free (basename);
      
  return &dir->readdir_result;
}

void
g_win32_rewinddir (DIR *dir)
{
  gchar *mask;

  g_return_if_fail (dir != NULL);

  if (!FindClose ((HANDLE) dir->find_file_handle))
    g_warning ("gwin_rewinddir(): FindClose() failed\n");

  mask = g_strdup_printf ("%s\\*", dir->dir_name);
  dir->find_file_handle = (gulong) FindFirstFile (mask,
					  (LPWIN32_FIND_DATA) dir->find_file_data);
  g_free (mask);

  if (dir->find_file_handle == (gulong) INVALID_HANDLE_VALUE)
    {
      int error = GetLastError ();

      switch (error)
	{
	default:
	  errno = EIO;
	  return;
	}
    }
  dir->just_opened = TRUE;
}  

gint
g_win32_closedir (DIR *dir)
{
  g_return_val_if_fail (dir != NULL, -1);

  if (!FindClose ((HANDLE) dir->find_file_handle))
    {
      int error = GetLastError ();

      switch (error)
	{
	default:
	  errno = EIO; return -1;
	}
    }

  g_free (dir->dir_name);
  g_free (dir->find_file_data);
  g_free (dir);

  return 0;
}
#endif

/* msvc 5.0 headers don't have latest language and sublanguage codes */
#ifndef LANG_ARMENIAN
#define LANG_ARMENIAN 0x2b
#endif
#ifndef LANG_ASSAMESE
#define LANG_ASSAMESE 0x4d
#endif
#ifndef LANG_AZERI
#define LANG_AZERI 0x2c
#endif
#ifndef LANG_BENGALI
#define LANG_BENGALI 0x45
#endif
#ifndef LANG_GEORGIAN
#define LANG_GEORGIAN 0x37
#endif
#ifndef LANG_GUJARATI
#define LANG_GUJARATI 0x47
#endif
#ifndef LANG_HINDI
#define LANG_HINDI 0x39
#endif
#ifndef LANG_KANNADA
#define LANG_KANNADA 0x4b
#endif
#ifndef LANG_KASHMIRI
#define LANG_KASHMIRI 0x60
#endif
#ifndef LANG_KAZAK
#define LANG_KAZAK 0x3f
#endif
#ifndef LANG_KONKANI
#define LANG_KONKANI 0x57
#endif
#ifndef LANG_MACEDONIAN
#define LANG_MACEDONIAN 0x2f
#endif
#ifndef LANG_MALAY
#define LANG_MALAY 0x3e
#endif
#ifndef LANG_MALAYALAM
#define LANG_MALAYALAM 0x4c
#endif
#ifndef LANG_MANIPURI
#define LANG_MANIPURI 0x58
#endif
#ifndef LANG_MARATHI
#define LANG_MARATHI 0x4e
#endif
#ifndef LANG_NEPALI
#define LANG_NEPALI 0x61
#endif
#ifndef LANG_ORIYA
#define LANG_ORIYA 0x48
#endif
#ifndef LANG_PUNJABI
#define LANG_PUNJABI 0x46
#endif
#ifndef LANG_SANSKRIT
#define LANG_SANSKRIT 0x4f
#endif
#ifndef LANG_SINDHI
#define LANG_SINDHI 0x59
#endif
#ifndef LANG_SWAHILI
#define LANG_SWAHILI 0x41
#endif
#ifndef LANG_TAMIL
#define LANG_TAMIL 0x49
#endif
#ifndef LANG_TATAR
#define LANG_TATAR 0x44
#endif
#ifndef LANG_TELUGU
#define LANG_TELUGU 0x4a
#endif
#ifndef LANG_URDU
#define LANG_URDU 0x20
#endif
#ifndef LANG_UZBEK
#define LANG_UZBEK 0x43
#endif

#ifndef SUBLANG_CHINESE_MACAU
#define SUBLANG_CHINESE_MACAU 0x05
#endif
#ifndef SUBLANG_ENGLISH_ZIMBABWE
#define SUBLANG_ENGLISH_ZIMBABWE 0x0c
#endif
#ifndef SUBLANG_ENGLISH_PHILIPPINES
#define SUBLANG_ENGLISH_PHILIPPINES 0x0d
#endif
#ifndef SUBLANG_FRENCH_MONACO
#define SUBLANG_FRENCH_MONACO 0x06
#endif
#ifndef SUBLANG_KASHMIRI_INDIA
#define SUBLANG_KASHMIRI_INDIA 0x02
#endif
#ifndef SUBLANG_MALAY_BRUNEI_DARUSSALAM
#define SUBLANG_MALAY_BRUNEI_DARUSSALAM 0x02
#endif
#ifndef SUBLANG_NEPALI_INDIA
#define SUBLANG_NEPALI_INDIA 0x02
#endif
#ifndef SUBLANG_URDU_PAKISTAN
#define SUBLANG_URDU_PAKISTAN 0x01
#endif
#ifndef SUBLANG_URDU_INDIA
#define SUBLANG_URDU_INDIA 0x02
#endif
#ifndef SUBLANG_UZBEK_CYRILLIC
#define SUBLANG_UZBEK_CYRILLIC 0x02
#endif

/**
 * g_win32_getlocale:
 *
 * The setlocale in the Microsoft C library uses locale names of the
 * form "English_United States.1252" etc. We want the Unixish standard
 * form "en", "zh_TW" etc. This function gets the current thread
 * locale from Windows and returns it as a string of the above form
 * for use in forming file names etc. The returned string should be
 * deallocated with g_free().
 *
 * Returns: allocated locale name
 */

gchar *
g_win32_getlocale (void)
{
  LCID lcid;
  gchar *ev;
  gint primary, sub;
  gchar *l = NULL, *sl = NULL;
  gchar bfr[20];

  if ((ev = getenv ("LC_ALL")) != NULL
      || (ev = getenv ("LC_CTYPE")) != NULL
      || (ev = getenv ("LANG")) != NULL)
    return g_strdup (ev);

  lcid = GetThreadLocale ();
  primary = PRIMARYLANGID (LANGIDFROMLCID (lcid));
  sub = SUBLANGID (LANGIDFROMLCID (lcid));
  switch (primary)
    {
    case LANG_AFRIKAANS: l = "af"; break;
    case LANG_ALBANIAN: l = "sq"; break;
    case LANG_ARABIC:
      l = "ar";
      switch (sub)
	{
	case SUBLANG_ARABIC_SAUDI_ARABIA: sl = "SA"; break;
	case SUBLANG_ARABIC_IRAQ: sl = "IQ"; break;
	case SUBLANG_ARABIC_EGYPT: sl = "EG"; break;
	case SUBLANG_ARABIC_LIBYA: sl = "LY"; break;
	case SUBLANG_ARABIC_ALGERIA: sl = "DZ"; break;
	case SUBLANG_ARABIC_MOROCCO: sl = "MA"; break;
	case SUBLANG_ARABIC_TUNISIA: sl = "TN"; break;
	case SUBLANG_ARABIC_OMAN: sl = "OM"; break;
	case SUBLANG_ARABIC_YEMEN: sl = "YE"; break;
	case SUBLANG_ARABIC_SYRIA: sl = "SY"; break;
	case SUBLANG_ARABIC_JORDAN: sl = "JO"; break;
	case SUBLANG_ARABIC_LEBANON: sl = "LB"; break;
	case SUBLANG_ARABIC_KUWAIT: sl = "KW"; break;
	case SUBLANG_ARABIC_UAE: sl = "AE"; break;
	case SUBLANG_ARABIC_BAHRAIN: sl = "BH"; break;
	case SUBLANG_ARABIC_QATAR: sl = "QA"; break;
	}
      break;
    case LANG_ARMENIAN: l = "hy"; break;
    case LANG_ASSAMESE: l = "as"; break;
    case LANG_AZERI: l = "az"; break;
    case LANG_BASQUE: l = "eu"; break;
    case LANG_BELARUSIAN: l = "be"; break;
    case LANG_BENGALI: l = "bn"; break;
    case LANG_BULGARIAN: l = "bg"; break;
    case LANG_CATALAN: l = "ca"; break;
    case LANG_CHINESE:
      l = "zh";
      switch (sub)
	{
	case SUBLANG_CHINESE_TRADITIONAL: sl = "TW"; break;
	case SUBLANG_CHINESE_SIMPLIFIED: sl = "CH"; break;
	case SUBLANG_CHINESE_HONGKONG: sl = "HK"; break;
	case SUBLANG_CHINESE_SINGAPORE: sl = "SG"; break;
	case SUBLANG_CHINESE_MACAU: sl = "MO"; break;
	}
      break;
    case LANG_CROATIAN:		/* LANG_CROATIAN == LANG_SERBIAN
				 * What used to be called Serbo-Croatian
				 * should really now be two separate
				 * languages because of political reasons.
				 * (Says tml, who knows nothing about Serbian
				 * or Croatian.)
				 * (I can feel those flames coming already.)
				 */
      switch (sub)
	{
	case SUBLANG_SERBIAN_LATIN: l = "sp"; break;
	case SUBLANG_SERBIAN_CYRILLIC: l = "sr"; break;
	default: l = "hr";	/* ??? */
	}
      break;
    case LANG_CZECH: l = "cs"; break;
    case LANG_DANISH: l = "da"; break;
    case LANG_DUTCH:
      l = "nl";
      switch (sub)
	{
	case SUBLANG_DUTCH_BELGIAN: sl = "BE"; break;
	}
      break;
    case LANG_ENGLISH:
      l = "en";
      switch (sub)
	{
	/* SUBLANG_ENGLISH_US == SUBLANG_DEFAULT. Heh. I thought
	 * English was the language spoken in England.
	 * Oh well.
	 */
	case SUBLANG_ENGLISH_UK: sl = "GB"; break;
	case SUBLANG_ENGLISH_AUS: sl = "AU"; break;
	case SUBLANG_ENGLISH_CAN: sl = "CA"; break;
	case SUBLANG_ENGLISH_NZ: sl = "NZ"; break;
	case SUBLANG_ENGLISH_EIRE: sl = "IE"; break;
	case SUBLANG_ENGLISH_SOUTH_AFRICA: sl = "ZA"; break;
	case SUBLANG_ENGLISH_JAMAICA: sl = "JM"; break;
	case SUBLANG_ENGLISH_CARIBBEAN: sl = "@caribbean"; break; /* ??? */
	case SUBLANG_ENGLISH_BELIZE: sl = "BZ"; break;
	case SUBLANG_ENGLISH_TRINIDAD: sl = "TT"; break;
	case SUBLANG_ENGLISH_ZIMBABWE: sl = "ZW"; break;
	case SUBLANG_ENGLISH_PHILIPPINES: sl = "PH"; break;
	}
      break;
    case LANG_ESTONIAN: l = "et"; break;
    case LANG_FAEROESE: l = "fo"; break;
    case LANG_FARSI: l = "fa"; break;
    case LANG_FINNISH: l = "fi"; break;
    case LANG_FRENCH:
      l = "fr";
      switch (sub)
	{
	case SUBLANG_FRENCH_BELGIAN: sl = "BE"; break;
	case SUBLANG_FRENCH_CANADIAN: sl = "CA"; break;
	case SUBLANG_FRENCH_SWISS: sl = "CH"; break;
	case SUBLANG_FRENCH_LUXEMBOURG: sl = "LU"; break;
	case SUBLANG_FRENCH_MONACO: sl = "MC"; break;
	}
      break;
    case LANG_GEORGIAN: l = "ka"; break;
    case LANG_GERMAN:
      l = "de";
      switch (sub)
	{
	case SUBLANG_GERMAN_SWISS: sl = "CH"; break;
	case SUBLANG_GERMAN_AUSTRIAN: sl = "AT"; break;
	case SUBLANG_GERMAN_LUXEMBOURG: sl = "LU"; break;
	case SUBLANG_GERMAN_LIECHTENSTEIN: sl = "LI"; break;
	}
      break;
    case LANG_GREEK: l = "el"; break;
    case LANG_GUJARATI: l = "gu"; break;
    case LANG_HEBREW: l = "he"; break;
    case LANG_HINDI: l = "hi"; break;
    case LANG_HUNGARIAN: l = "hu"; break;
    case LANG_ICELANDIC: l = "is"; break;
    case LANG_INDONESIAN: l = "id"; break;
    case LANG_ITALIAN:
      l = "it";
      switch (sub)
	{
	case SUBLANG_ITALIAN_SWISS: sl = "CH"; break;
	}
      break;
    case LANG_JAPANESE: l = "ja"; break;
    case LANG_KANNADA: l = "kn"; break;
    case LANG_KASHMIRI:
      l = "ks";
      switch (sub)
	{
	case SUBLANG_KASHMIRI_INDIA: sl = "IN"; break;
	}
      break;
    case LANG_KAZAK: l = "kk"; break;
    case LANG_KONKANI: l = "kok"; break; /* ??? */
    case LANG_KOREAN: l = "ko"; break;
    case LANG_LATVIAN: l = "lv"; break;
    case LANG_LITHUANIAN: l = "lt"; break;
    case LANG_MACEDONIAN: l = "mk"; break;
    case LANG_MALAY:
      l = "ms";
      switch (sub)
	{
	case SUBLANG_MALAY_BRUNEI_DARUSSALAM: sl = "BN"; break;
	}
      break;
    case LANG_MALAYALAM: l = "ml"; break;
    case LANG_MANIPURI: l = "mni"; break;
    case LANG_MARATHI: l = "mr"; break;
    case LANG_NEPALI:
      l = "ne";
      switch (sub)
	{
	case SUBLANG_NEPALI_INDIA: sl = "IN"; break;
	}
      break;
    case LANG_NORWEGIAN:
      l = "no";
      switch (sub)
	{
	/* SUBLANG_NORWEGIAN_BOKMAL == SUBLANG_DEFAULT */
	case SUBLANG_NORWEGIAN_NYNORSK: l = "nn"; break;
	}
      break;
    case LANG_ORIYA: l = "or"; break;
    case LANG_POLISH: l = "pl"; break;
    case LANG_PORTUGUESE:
      l = "pt";
      switch (sub)
	{
	/* Hmm. SUBLANG_PORTUGUESE_BRAZILIAN == SUBLANG_DEFAULT */
	case SUBLANG_PORTUGUESE_BRAZILIAN: sl = "BR"; break;
	}
      break;
    case LANG_PUNJABI: l = "pa"; break;
    case LANG_ROMANIAN: l = "ro"; break;
    case LANG_RUSSIAN: l = "ru"; break;
    case LANG_SANSKRIT: l = "sa"; break;
    case LANG_SINDHI: l = "sd"; break;
    case LANG_SLOVAK: l = "sk"; break;
    case LANG_SLOVENIAN: l = "sl"; break;
    case LANG_SPANISH:
      l = "es";
      switch (sub)
	{
	case SUBLANG_SPANISH_MEXICAN: sl = "MX"; break;
	case SUBLANG_SPANISH_MODERN: sl = "@modern"; break;	/* ??? */
	case SUBLANG_SPANISH_GUATEMALA: sl = "GT"; break;
	case SUBLANG_SPANISH_COSTA_RICA: sl = "CR"; break;
	case SUBLANG_SPANISH_PANAMA: sl = "PA"; break;
	case SUBLANG_SPANISH_DOMINICAN_REPUBLIC: sl = "DO"; break;
	case SUBLANG_SPANISH_VENEZUELA: sl = "VE"; break;
	case SUBLANG_SPANISH_COLOMBIA: sl = "CO"; break;
	case SUBLANG_SPANISH_PERU: sl = "PE"; break;
	case SUBLANG_SPANISH_ARGENTINA: sl = "AR"; break;
	case SUBLANG_SPANISH_ECUADOR: sl = "EC"; break;
	case SUBLANG_SPANISH_CHILE: sl = "CL"; break;
	case SUBLANG_SPANISH_URUGUAY: sl = "UY"; break;
	case SUBLANG_SPANISH_PARAGUAY: sl = "PY"; break;
	case SUBLANG_SPANISH_BOLIVIA: sl = "BO"; break;
	case SUBLANG_SPANISH_EL_SALVADOR: sl = "SV"; break;
	case SUBLANG_SPANISH_HONDURAS: sl = "HN"; break;
	case SUBLANG_SPANISH_NICARAGUA: sl = "NI"; break;
	case SUBLANG_SPANISH_PUERTO_RICO: sl = "PR"; break;
	}
      break;
    case LANG_SWAHILI: l = "sw"; break;
    case LANG_SWEDISH:
      l = "sv";
      switch (sub)
	{
	case SUBLANG_SWEDISH_FINLAND: sl = "FI"; break;
	}
      break;
    case LANG_TAMIL: l = "ta"; break;
    case LANG_TATAR: l = "tt"; break;
    case LANG_TELUGU: l = "te"; break;
    case LANG_THAI: l = "th"; break;
    case LANG_TURKISH: l = "tr"; break;
    case LANG_UKRAINIAN: l = "uk"; break;
    case LANG_URDU:
      l = "ur";
      switch (sub)
	{
	case SUBLANG_URDU_PAKISTAN: sl = "PK"; break;
	case SUBLANG_URDU_INDIA: sl = "IN"; break;
	}
      break;
    case LANG_UZBEK:
      l = "uz";
      switch (sub)
	{
	case SUBLANG_UZBEK_CYRILLIC: sl = "@cyrillic"; break;
	}
      break;
    case LANG_VIETNAMESE: l = "vi"; break;
    default: l = "xx"; break;
    }
  strcpy (bfr, l);
  if (sl != NULL)
    {
      if (sl[0] != '@')
	strcat (bfr, "_");
      strcat (bfr, sl);
    }

  return g_strdup (bfr);
}

/**
 * g_win32_error_message:
 * @error: error code
 *
 * Translate a Win32 error code (as returned by GetLastError()) into
 * the corresponding message. The message is either language neutral,
 * or in the thread's language, or the user's language, the system's
 * langauge, or US English (see docs for FormatMessage). The returned
 * string should be deallocated with g_free().
 *
 * Returns: allocated error message
 */
gchar *
g_win32_error_message (gint error)
{
  gchar *msg;
  gchar *retval;
  int nbytes;

  FormatMessage (FORMAT_MESSAGE_ALLOCATE_BUFFER
		 |FORMAT_MESSAGE_IGNORE_INSERTS
		 |FORMAT_MESSAGE_FROM_SYSTEM,
		 NULL, error, 0,
		 (LPTSTR) &msg, 0, NULL);
  nbytes = strlen (msg);

  if (nbytes > 2 && msg[nbytes-1] == '\n' && msg[nbytes-2] == '\r')
    msg[nbytes-2] = '\0';
  
  retval = g_strdup (msg);

  if (msg != NULL)
    LocalFree (msg);

  return retval;
}

static gchar *
get_package_directory_from_module (gchar *module_name)
{
  static GHashTable *module_dirs = NULL;
  HMODULE hmodule = NULL;
  gchar *fn;
  gchar *p;
  gchar *result;

  if (module_dirs == NULL)
    module_dirs = g_hash_table_new (g_str_hash, g_str_equal);
  
  result = g_hash_table_lookup (module_dirs, module_name ? module_name : "");
      
  if (result)
    return g_strdup (result);

  if (module_name)
    {
      hmodule = GetModuleHandle (module_name);
      if (!hmodule)
	return NULL;
    }

  fn = g_malloc (MAX_PATH);
  if (!GetModuleFileName (hmodule, fn, MAX_PATH))
    return NULL;

#ifdef G_WITH_CYGWIN
  /* In Cygwin we need to have POSIX paths */
  {
    gchar tmp[MAX_PATH];

    cygwin_conv_to_posix_path(fn, tmp);
    g_free(fn);
    fn = g_strdup(tmp);
  }
#endif

  if ((p = strrchr (fn, G_DIR_SEPARATOR)) != NULL)
    *p = '\0';

  p = strrchr (fn, G_DIR_SEPARATOR);
  if (p && (g_ascii_strcasecmp (p + 1, "bin") == 0 ||
	    g_ascii_strcasecmp (p + 1, "lib") == 0))
    *p = '\0';

  g_hash_table_insert (module_dirs, module_name ? module_name : "", fn);

  return g_strdup (fn);
}

/**
 * g_win32_get_package_installation_directory:
 * @package: An identifier for a software package, or NULL
 * @dll_name: The name of a DLL that a package provides, or NULL
 *
 * Try to determine the installation directory for a software package.
 * Typically used by GNU software packages.
 *
 * @package should be a short identifier for the package. Typically it
 * is the same identifier as used for GETTEXT_PACKAGE in software
 * configured accoring to GNU standards. The function first looks in
 * the Windows Registry for the value #InstallationDirectory in the
 * key #HKLM\Software\@package, and if that value exists and is a
 * string, returns that.
 *
 * If @package is NULL, or the above value isn't found in the
 * Registry, but @dll_name is non-NULL, it should name a DLL loaded
 * into the current process. Typically that would be the name of the
 * DLL calling this function, looking for its installation
 * directory. The function then asks Windows what directory that DLL
 * was loaded from. If that directory's last component is "bin" or
 * "lib", the parent directory is returned, otherwise the directory
 * itself. If that DLL isn't loaded, the function proceeds as if
 * @dll_name was NULL.
 *
 * If both @package and @dll_name are NULL, the directory from where
 * the main executable of the process was loaded is uses instead in
 * the same way as above.
 *
 * The return value should be freed with g_free() when not needed any longer.  */

gchar *
g_win32_get_package_installation_directory (gchar *package,
					    gchar *dll_name)
{
  static GHashTable *package_dirs = NULL;
  gchar *result = NULL;
  gchar *key;
  HKEY reg_key = NULL;
  DWORD type;
  DWORD nbytes;

  if (package != NULL)
    {
      if (package_dirs == NULL)
	package_dirs = g_hash_table_new (g_str_hash, g_str_equal);
      
      result = g_hash_table_lookup (package_dirs, package);
      
      if (result && result[0])
	return g_strdup (result);
      
      key = g_strconcat ("Software\\", package, NULL);
      
      nbytes = 0;
      if ((RegOpenKeyEx (HKEY_CURRENT_USER, key, 0,
			 KEY_QUERY_VALUE, &reg_key) == ERROR_SUCCESS
	   && RegQueryValueEx (reg_key, "InstallationDirectory", 0,
			       &type, NULL, &nbytes) == ERROR_SUCCESS)
	  ||
	  ((RegOpenKeyEx (HKEY_LOCAL_MACHINE, key, 0,
			 KEY_QUERY_VALUE, &reg_key) == ERROR_SUCCESS
	   && RegQueryValueEx (reg_key, "InstallationDirectory", 0,
			       &type, NULL, &nbytes) == ERROR_SUCCESS)
	   && type == REG_SZ))
	{
	  result = g_malloc (nbytes + 1);
	  RegQueryValueEx (reg_key, "InstallationDirectory", 0,
			   &type, result, &nbytes);
	  result[nbytes] = '\0';
	}

      if (reg_key != NULL)
	RegCloseKey (reg_key);
      
      g_free (key);
      
    }
  if (result)
    {
      g_hash_table_insert (package_dirs, package, result);
      return g_strdup (result);
    }

  if (dll_name != NULL)
    result = get_package_directory_from_module (dll_name);

  if (result == NULL)
    result = get_package_directory_from_module (NULL);

  return result;
}

/**
 * g_win32_get_package_installation_subdirectory:
 * @package: An identifier for a software package, or NULL
 * @dll_name: The name of a DLL that a package provides, or NULL
 * @subdir: A subdirectory of the package installation directory.
 *
 * Returns a string containg the path of the subdirectory @subdir in
 * the return value from calling
 * g_win32_get_package_installation_directory() with the @package and
 * @dll_name parameters. The return value should be freed with
 * g_free() when no longer needed.
 */

gchar *
g_win32_get_package_installation_subdirectory (gchar *package,
					       gchar *dll_name,
					       gchar *subdir)
{
  gchar *prefix;
  gchar *sep;

  prefix = g_win32_get_package_installation_directory (package, dll_name);

  sep = (prefix[strlen (prefix) - 1] == G_DIR_SEPARATOR ?
	 "" : G_DIR_SEPARATOR_S);

  return g_strconcat (prefix, sep, subdir, NULL);
}
