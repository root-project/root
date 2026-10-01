/* gspawn-win32-helper.c - Helper program for process launching on Win32.
 *
 *  Copyright 2000 Red Hat, Inc.
 *  Copyright 2000 Tor Lillqvist
 *
 * GLib is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * GLib is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GLib; see the file COPYING.LIB.  If not, write
 * to the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#undef G_LOG_DOMAIN
#include "glib.h"
#define GSPAWN_HELPER
#include "gspawn-win32.c"	/* For shared definitions */

static GString *debugstring;

static void
write_err_and_exit (gint fd,
		    gint msg)
{
  gint en = errno;
  
  if (debug)
    {
      debugstring = g_string_new ("");
      g_string_append (debugstring,
		       g_strdup_printf ("writing error code %d and errno %d",
					msg, en));
      MessageBox (NULL, debugstring->str, "gspawn-win32-helper", 0);
    }

  write (fd, &msg, sizeof(msg));
  write (fd, &en, sizeof(en));
  
  _exit (1);
}

#ifdef __GNUC__
#  ifndef _stdcall
#    define _stdcall  __attribute__((stdcall))
#  endif
#endif

/* We build gspawn-win32-helper.exe as a Windows GUI application
 * to avoid any temporarily flashing console windows in case
 * the gspawn function is invoked by a GUI program. Thus, no main()
 * but a WinMain(). We do, however, still use argc and argv tucked
 * away in the global __argc and __argv by the C runtime startup code.
 */

int _stdcall
WinMain (struct HINSTANCE__ *hInstance,
	 struct HINSTANCE__ *hPrevInstance,
	 char               *lpszCmdLine,
	 int                 nCmdShow)
{
  int child_err_report_fd;
  int i;
  int fd;
  int mode;
  gint zero = 0;

  SETUP_DEBUG();

  if (debug)
    {
      debugstring = g_string_new ("");

      g_string_append (debugstring,
		       g_strdup_printf ("g-spawn-win32-helper: "
					"argc = %d, argv: ",
					__argc));
      for (i = 0; i < __argc; i++)
	{
	  if (i > 0)
	    g_string_append (debugstring, " ");
	  g_string_append (debugstring, __argv[i]);
	}
      
      MessageBox (NULL, debugstring->str, "gspawn-win32-helper", 0);
    }

  g_assert (__argc >= ARG_COUNT);

  /* argv[ARG_CHILD_ERR_REPORT] is the file descriptor onto which
   * write error messages.
   */
  child_err_report_fd = atoi (__argv[ARG_CHILD_ERR_REPORT]);

  /* argv[ARG_STDIN..ARG_STDERR] are the file descriptors that should
   * be dup2'd to stdin, stdout and stderr, '-' if the corresponding
   * std* should be let alone, and 'z' if it should be connected to
   * the bit bucket NUL:.
   */
  if (__argv[ARG_STDIN][0] == '-')
    ; /* Nothing */
  else if (__argv[ARG_STDIN][0] == 'z')
    {
      fd = open ("NUL:", O_RDONLY);
      if (fd != 0)
	{
	  dup2 (fd, 0);
	  close (fd);
	}
    }
  else
    {
      fd = atoi (__argv[ARG_STDIN]);
      if (fd != 0)
	{
	  dup2 (fd, 0);
	  close (fd);
	}
    }

  if (__argv[ARG_STDOUT][0] == '-')
    ; /* Nothing */
  else if (__argv[ARG_STDOUT][0] == 'z')
    {
      fd = open ("NUL:", O_WRONLY);
      if (fd != 1)
	{
	  dup2 (fd, 1);
	  close (fd);
	}
    }
  else
    {
      fd = atoi (__argv[ARG_STDOUT]);
      if (fd != 1)
	{
	  dup2 (fd, 1);
	  close (fd);
	}
    }

  if (__argv[ARG_STDERR][0] == '-')
    ; /* Nothing */
  else if (__argv[ARG_STDERR][0] == 'z')
    {
      fd = open ("NUL:", O_WRONLY);
      if (fd != 2)
	{
	  dup2 (fd, 2);
	  close (fd);
	}
    }
  else
    {
      fd = atoi (__argv[ARG_STDERR]);
      if (fd != 2)
	{
	  dup2 (fd, 2);
	  close (fd);
	}
    }

  /* __argv[ARG_WORKING_DIRECTORY] is the directory in which to run the
   * process.  If "-", don't change directory.
   */
  if (__argv[ARG_WORKING_DIRECTORY][0] == '-' &&
      __argv[ARG_WORKING_DIRECTORY][1] == 0)
    ; /* Nothing */
  else if (chdir (__argv[ARG_WORKING_DIRECTORY]) < 0)
    write_err_and_exit (child_err_report_fd,
			CHILD_CHDIR_FAILED);

  /* __argv[ARG_CLOSE_DESCRIPTORS] is "y" if file descriptors from 3
   *  upwards should be closed
   */

  if (__argv[ARG_CLOSE_DESCRIPTORS][0] == 'y')
    for (i = 3; i < 1000; i++)	/* FIXME real limit? */
      if (i != child_err_report_fd)
	close (i);

  /* __argv[ARG_WAIT] is "w" to wait for the program to exit */

  if (__argv[ARG_WAIT][0] == 'w')
    mode = P_WAIT;
  else
    mode = P_NOWAIT;

  /* __argv[ARG_USE_PATH] is "y" to use PATH, otherwise not */

  /* __argv[ARG_PROGRAM] is program file to run,
   * __argv[ARG_PROGRAM+1]... is its __argv.
   */

  if (debug)
    {
      debugstring = g_string_new ("");
      g_string_append (debugstring,
		       g_strdup_printf ("calling %s on program %s, __argv: ",
					(__argv[ARG_USE_PATH][0] == 'y' ?
					 "spawnvp" : "spawnv"),
					__argv[ARG_PROGRAM]));
      i = ARG_PROGRAM+1;
      while (__argv[i])
	g_string_append (debugstring, __argv[i++]);
      MessageBox (NULL, debugstring->str, "gspawn-win32-helper", 0);
    }

  if (__argv[ARG_USE_PATH][0] == 'y')
    {
      if (spawnvp (mode, __argv[ARG_PROGRAM], __argv+ARG_PROGRAM) < 0)
	write_err_and_exit (child_err_report_fd, CHILD_SPAWN_FAILED);
    }
  else
    {
      if (spawnv (mode, __argv[ARG_PROGRAM], __argv+ARG_PROGRAM) < 0)
	write_err_and_exit (child_err_report_fd, CHILD_SPAWN_FAILED);
    }
  write (child_err_report_fd, &zero, sizeof (zero));
  write (child_err_report_fd, &zero, sizeof (zero));
  Sleep (10000);
  return 0;
}

