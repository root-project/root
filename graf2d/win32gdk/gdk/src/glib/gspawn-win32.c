/* gspawn-win32.c - Process launching on Win32
 *
 *  Copyright 2000 Red Hat, Inc.
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

/*
 * Implementation details on Win32.
 *
 * - There is no way to set the no-inherit flag for
 *   a "file descriptor" in the MS C runtime. The flag is there,
 *   and the dospawn() function uses it, but unfortunately
 *   this flag can only be set when opening the file.
 * - As there is no fork(), we cannot reliably change directory
 *   before starting the child process. (There might be several threads
 *   running, and the current directory is common for all threads.)
 *
 * Thus, we must in most cases use a helper program to handle closing
 * of (inherited) file descriptors and changing of directory. In fact,
 * we do it all the time.
 */

/* Define this to get some logging all the time */
/* #define G_SPAWN_WIN32_DEBUG */

#include "glib.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <windows.h>
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <direct.h>

#include "glibintl.h"

#ifdef G_SPAWN_WIN32_DEBUG
  static int debug = 1;
  #define SETUP_DEBUG() /* empty */

#else
  static int debug = -1;
  #define SETUP_DEBUG()					\
    G_STMT_START					\
      {							\
	if (debug == -1)				\
	  {						\
	    if (getenv ("G_SPAWN_WIN32_DEBUG") != NULL)	\
	      debug = 1;				\
	    else					\
	      debug = 0;				\
	  }						\
      }							\
    G_STMT_END
#endif

enum
{
  CHILD_NO_ERROR,
  CHILD_CHDIR_FAILED,
  CHILD_SPAWN_FAILED,
};

enum {
  ARG_CHILD_ERR_REPORT = 1,
  ARG_STDIN,
  ARG_STDOUT,
  ARG_STDERR,
  ARG_WORKING_DIRECTORY,
  ARG_CLOSE_DESCRIPTORS,
  ARG_USE_PATH,
  ARG_WAIT,
  ARG_PROGRAM,
  ARG_COUNT = ARG_PROGRAM
};

#ifndef GSPAWN_HELPER

static gboolean make_pipe            (gint                  p[2],
                                      GError              **error);
static gboolean fork_exec_with_pipes (gboolean              dont_wait,
				      const gchar          *working_directory,
                                      gchar               **argv,
                                      gchar               **envp,
                                      gboolean              close_descriptors,
                                      gboolean              search_path,
                                      gboolean              stdout_to_null,
                                      gboolean              stderr_to_null,
                                      gboolean              child_inherits_stdin,
                                      GSpawnChildSetupFunc  child_setup,
                                      gpointer              user_data,
                                      gint                 *standard_input,
                                      gint                 *standard_output,
                                      gint                 *standard_error,
				      gint                 *exit_status,
                                      GError              **error);

GQuark
g_spawn_error_quark (void)
{
  static GQuark quark = 0;
  if (quark == 0)
    quark = g_quark_from_static_string ("g-exec-error-quark");
  return quark;
}

/**
 * g_spawn_async:
 * @working_directory: child's current working directory, or NULL to inherit parent's
 * @argv: child's argument vector
 * @envp: child's environment, or NULL to inherit parent's
 * @flags: flags from #GSpawnFlags
 * @child_setup: function to run in the child just before exec()
 * @user_data: user data for @child_setup
 * @child_pid: return location for child process ID, or NULL
 * @error: return location for error
 * 
 * See g_spawn_async_with_pipes() for a full description; this function
 * simply calls the g_spawn_async_with_pipes() without any pipes.
 * 
 * Return value: TRUE on success, FALSE if error is set
 **/
gboolean
g_spawn_async (const gchar          *working_directory,
               gchar               **argv,
               gchar               **envp,
               GSpawnFlags           flags,
               GSpawnChildSetupFunc  child_setup,
               gpointer              user_data,
               gint                 *child_pid,
               GError              **error)
{
  g_return_val_if_fail (argv != NULL, FALSE);
  
  return g_spawn_async_with_pipes (working_directory,
                                   argv, envp,
                                   flags,
                                   child_setup,
                                   user_data,
                                   child_pid,
                                   NULL, NULL, NULL,
                                   error);
}

/* Avoids a danger in threaded situations (calling close()
 * on a file descriptor twice, and another thread has
 * re-opened it since the first close)
 */
static gint
close_and_invalidate (gint *fd)
{
  gint ret;

  ret = close (*fd);
  *fd = -1;

  return ret;
}

typedef enum
{
  READ_FAILED = 0, /* FALSE */
  READ_OK,
  READ_EOF
} ReadResult;

static ReadResult
read_data (GString     *str,
           GIOChannel  *iochannel,
           GError     **error)
{
  GIOError gioerror;
  gsize bytes;
  gchar buf[4096];

 again:
  
  gioerror = g_io_channel_read (iochannel, buf, sizeof (buf), &bytes);

  if (bytes == 0)
    return READ_EOF;
  else if (bytes > 0)
    {
      g_string_append_len (str, buf, bytes);
      return READ_OK;
    }
  else if (gioerror == G_IO_ERROR_AGAIN)
    goto again;
  else if (gioerror != G_IO_ERROR_NONE)
    {
      g_set_error (error,
                   G_SPAWN_ERROR,
                   G_SPAWN_ERROR_READ,
                   _("Failed to read data from child process"));
      
      return READ_FAILED;
    }
  else
    return READ_OK;
}

/**
 * g_spawn_sync:
 * @working_directory: child's current working directory, or NULL to inherit parent's
 * @argv: child's argument vector
 * @envp: child's environment, or NULL to inherit parent's
 * @flags: flags from #GSpawnFlags
 * @child_setup: function to run in the child just before exec()
 * @user_data: user data for @child_setup
 * @standard_output: return location for child output 
 * @standard_error: return location for child error messages
 * @exit_status: child exit status, as returned by waitpid()
 * @error: return location for error
 *
 * Executes a child synchronously (waits for the child to exit before returning).
 * All output from the child is stored in @standard_output and @standard_error,
 * if those parameters are non-NULL. If @exit_status is non-NULL, the exit status
 * of the child is stored there as it would be by waitpid(); standard UNIX
 * macros such as WIFEXITED() and WEXITSTATUS() must be used to evaluate the
 * exit status. If an error occurs, no data is returned in @standard_output,
 * @standard_error, or @exit_status.
 * 
 * This function calls g_spawn_async_with_pipes() internally; see that function
 * for full details on the other parameters.
 * 
 * Return value: TRUE on success, FALSE if an error was set.
 **/
gboolean
g_spawn_sync (const gchar          *working_directory,
              gchar               **argv,
              gchar               **envp,
              GSpawnFlags           flags,
              GSpawnChildSetupFunc  child_setup,
              gpointer              user_data,
              gchar               **standard_output,
              gchar               **standard_error,
              gint                 *exit_status,
              GError              **error)     
{
  gint outpipe = -1;
  gint errpipe = -1;
  GIOChannel *outchannel = NULL;
  GIOChannel *errchannel = NULL;
  GPollFD outfd, errfd;
  GPollFD fds[2];
  gint nfds;
  gint outindex = -1;
  gint errindex = -1;
  gint ret;
  GString *outstr = NULL;
  GString *errstr = NULL;
  gboolean failed;
  gint status;
  
  g_return_val_if_fail (argv != NULL, FALSE);
  g_return_val_if_fail (!(flags & G_SPAWN_DO_NOT_REAP_CHILD), FALSE);
  g_return_val_if_fail (standard_output == NULL ||
                        !(flags & G_SPAWN_STDOUT_TO_DEV_NULL), FALSE);
  g_return_val_if_fail (standard_error == NULL ||
                        !(flags & G_SPAWN_STDERR_TO_DEV_NULL), FALSE);
  
  /* Just to ensure segfaults if callers try to use
   * these when an error is reported.
   */
  if (standard_output)
    *standard_output = NULL;

  if (standard_error)
    *standard_error = NULL;
  
  if (!fork_exec_with_pipes (FALSE,
                             working_directory,
                             argv,
                             envp,
                             !(flags & G_SPAWN_LEAVE_DESCRIPTORS_OPEN),
                             (flags & G_SPAWN_SEARCH_PATH) != 0,
                             (flags & G_SPAWN_STDOUT_TO_DEV_NULL) != 0,
                             (flags & G_SPAWN_STDERR_TO_DEV_NULL) != 0,
                             (flags & G_SPAWN_CHILD_INHERITS_STDIN) != 0,
                             child_setup,
                             user_data,
                             NULL,
                             standard_output ? &outpipe : NULL,
                             standard_error ? &errpipe : NULL,
			     &status,
                             error))
    return FALSE;

  /* Read data from child. */
  
  failed = FALSE;

  if (outpipe >= 0)
    {
      outstr = g_string_new ("");
      outchannel = g_io_channel_win32_new_fd (outpipe);
      g_io_channel_win32_make_pollfd (outchannel,
				      G_IO_IN | G_IO_ERR | G_IO_HUP,
				      &outfd);
    }
      
  if (errpipe >= 0)
    {
      errstr = g_string_new ("");
      errchannel = g_io_channel_win32_new_fd (errpipe);
      g_io_channel_win32_make_pollfd (errchannel,
				      G_IO_IN | G_IO_ERR | G_IO_HUP,
				      &errfd);
    }

  /* Read data until we get EOF on both pipes. */
  while (!failed &&
         (outpipe >= 0 ||
          errpipe >= 0))
    {
      nfds = 0;
      if (outpipe >= 0)
	{
	  fds[nfds] = outfd;
	  outindex = nfds;
	  nfds++;
	}
      if (errpipe >= 0)
	{
	  fds[nfds] = errfd;
	  errindex = nfds;
	  nfds++;
	}

      if (debug)
	g_print ("%s:g_spawn_sync: calling g_io_channel_win32_poll, nfds=%d\n",
		 __FILE__, nfds);

      ret = g_io_channel_win32_poll (fds, nfds, -1);

      if (ret < 0)
        {
          failed = TRUE;

          g_set_error (error,
                       G_SPAWN_ERROR,
                       G_SPAWN_ERROR_READ,
                       _("Unexpected error in g_io_channel_win32_poll() reading data from a child process"));
              
          break;
        }

      if (outpipe >= 0 && (fds[outindex].revents & G_IO_IN))
        {
          switch (read_data (outstr, outchannel, error))
            {
            case READ_FAILED:
	      if (debug)
		g_print ("g_spawn_sync: outchannel: READ_FAILED\n");
              failed = TRUE;
              break;
            case READ_EOF:
	      if (debug)
		g_print ("g_spawn_sync: outchannel: READ_EOF\n");
              g_io_channel_unref (outchannel);
	      outchannel = NULL;
              close_and_invalidate (&outpipe);
              break;
            default:
	      if (debug)
		g_print ("g_spawn_sync: outchannel: OK\n");
              break;
            }

          if (failed)
            break;
        }

      if (errpipe >= 0 && (fds[errindex].revents & G_IO_IN))
        {
          switch (read_data (errstr, errchannel, error))
            {
            case READ_FAILED:
	      if (debug)
		g_print ("g_spawn_sync: errchannel: READ_FAILED\n");
              failed = TRUE;
              break;
            case READ_EOF:
	      if (debug)
		g_print ("g_spawn_sync: errchannel: READ_EOF\n");
	      g_io_channel_unref (errchannel);
	      errchannel = NULL;
              close_and_invalidate (&errpipe);
              break;
            default:
	      if (debug)
		g_print ("g_spawn_sync: errchannel: OK\n");
              break;
            }

          if (failed)
            break;
        }
    }

  /* These should only be open still if we had an error.  */
  
  if (outchannel != NULL)
    g_io_channel_unref (outchannel);
  if (errchannel != NULL)
    g_io_channel_unref (errchannel);
  if (outpipe >= 0)
    close_and_invalidate (&outpipe);
  if (errpipe >= 0)
    close_and_invalidate (&errpipe);
  
  if (failed)
    {
      if (outstr)
        g_string_free (outstr, TRUE);
      if (errstr)
        g_string_free (errstr, TRUE);

      return FALSE;
    }
  else
    {
      if (exit_status)
        *exit_status = status;
      
      if (standard_output)        
        *standard_output = g_string_free (outstr, FALSE);

      if (standard_error)
        *standard_error = g_string_free (errstr, FALSE);

      return TRUE;
    }
}

/**
 * g_spawn_async_with_pipes:
 * @working_directory: child's current working directory, or NULL to inherit parent's
 * @argv: child's argument vector
 * @envp: child's environment, or NULL to inherit parent's
 * @flags: flags from #GSpawnFlags
 * @child_setup: function to run in the child just before exec()
 * @user_data: user data for @child_setup
 * @child_pid: return location for child process ID, or NULL
 * @standard_input: return location for file descriptor to write to child's stdin, or NULL
 * @standard_output: return location for file descriptor to read child's stdout, or NULL
 * @standard_error: return location for file descriptor to read child's stderr, or NULL
 * @error: return location for error
 *
 * Executes a child program asynchronously (your program will not
 * block waiting for the child to exit). The child program is
 * specified by the only argument that must be provided, @argv. @argv
 * should be a NULL-terminated array of strings, to be passed as the
 * argument vector for the child. The first string in @argv is of
 * course the name of the program to execute. By default, the name of
 * the program must be a full path; the PATH shell variable will only
 * be searched if you pass the %G_SPAWN_SEARCH_PATH flag.
 *
 * @envp is a NULL-terminated array of strings, where each string
 * has the form <literal>KEY=VALUE</literal>. This will become
 * the child's environment. If @envp is NULL, the child inherits its
 * parent's environment.
 *
 * @flags should be the bitwise OR of any flags you want to affect the
 * function's behavior. The %G_SPAWN_DO_NOT_REAP_CHILD means that the
 * child will not be automatically reaped; you must call waitpid() or
 * handle SIGCHLD yourself, or the child will become a zombie.
 * %G_SPAWN_LEAVE_DESCRIPTORS_OPEN means that the parent's open file
 * descriptors will be inherited by the child; otherwise all
 * descriptors except stdin/stdout/stderr will be closed before
 * calling exec() in the child. %G_SPAWN_SEARCH_PATH means that
 * <literal>argv[0]</literal> need not be an absolute path, it
 * will be looked for in the user's PATH. %G_SPAWN_STDOUT_TO_DEV_NULL
 * means that the child's standad output will be discarded, instead
 * of going to the same location as the parent's standard output.
 * %G_SPAWN_STDERR_TO_DEV_NULL means that the child's standard error
 * will be discarded. %G_SPAWN_CHILD_INHERITS_STDIN means that
 * the child will inherit the parent's standard input (by default,
 * the child's standard input is attached to /dev/null).
 *
 * @child_setup and @user_data are a function and user data to be
 * called in the child after GLib has performed all the setup it plans
 * to perform (including creating pipes, closing file descriptors,
 * etc.) but before calling exec(). That is, @child_setup is called
 * just before calling exec() in the child. Obviously actions taken in
 * this function will only affect the child, not the parent. 
 *
 * If non-NULL, @child_pid will be filled with the child's process
 * ID. You can use the process ID to send signals to the child, or
 * to waitpid() if you specified the %G_SPAWN_DO_NOT_REAP_CHILD flag.
 *
 * If non-NULL, the @standard_input, @standard_output, @standard_error
 * locations will be filled with file descriptors for writing to the child's
 * standard input or reading from its standard output or standard error.
 * The caller of g_spawn_async_with_pipes() must close these file descriptors
 * when they are no longer in use. If these parameters are NULL, the
 * corresponding pipe won't be created.
 *
 * @error can be NULL to ignore errors, or non-NULL to report errors.
 * If an error is set, the function returns FALSE. Errors
 * are reported even if they occur in the child (for example if the
 * executable in <literal>argv[0]</literal> is not found). Typically
 * the <literal>message</literal> field of returned errors should be displayed
 * to users. Possible errors are those from the #G_SPAWN_ERROR domain.
 *
 * If an error occurs, @child_pid, @standard_input, @standard_output,
 * and @standard_error will not be filled with valid values.
 * 
 * Return value: TRUE on success, FALSE if an error was set
 **/
gboolean
g_spawn_async_with_pipes (const gchar          *working_directory,
                          gchar               **argv,
                          gchar               **envp,
                          GSpawnFlags           flags,
                          GSpawnChildSetupFunc  child_setup,
                          gpointer              user_data,
                          gint                 *child_pid,
                          gint                 *standard_input,
                          gint                 *standard_output,
                          gint                 *standard_error,
                          GError              **error)
{
  g_return_val_if_fail (argv != NULL, FALSE);
  g_return_val_if_fail (standard_output == NULL ||
                        !(flags & G_SPAWN_STDOUT_TO_DEV_NULL), FALSE);
  g_return_val_if_fail (standard_error == NULL ||
                        !(flags & G_SPAWN_STDERR_TO_DEV_NULL), FALSE);
  /* can't inherit stdin if we have an input pipe. */
  g_return_val_if_fail (standard_input == NULL ||
                        !(flags & G_SPAWN_CHILD_INHERITS_STDIN), FALSE);
  
  return fork_exec_with_pipes (!(flags & G_SPAWN_DO_NOT_REAP_CHILD),
                               working_directory,
                               argv,
                               envp,
                               !(flags & G_SPAWN_LEAVE_DESCRIPTORS_OPEN),
                               (flags & G_SPAWN_SEARCH_PATH) != 0,
                               (flags & G_SPAWN_STDOUT_TO_DEV_NULL) != 0,
                               (flags & G_SPAWN_STDERR_TO_DEV_NULL) != 0,
                               (flags & G_SPAWN_CHILD_INHERITS_STDIN) != 0,
                               child_setup,
                               user_data,
                               standard_input,
                               standard_output,
                               standard_error,
			       NULL,
                               error);
}

/**
 * g_spawn_command_line_sync:
 * @command_line: a command line 
 * @standard_output: return location for child output
 * @standard_error: return location for child errors
 * @exit_status: return location for child exit status
 * @error: return location for errors
 *
 * A simple version of g_spawn_sync() with little-used parameters
 * removed, taking a command line instead of an argument vector.  See
 * g_spawn_sync() for full details. @command_line will be parsed by
 * g_shell_parse_argv(). Unlike g_spawn_sync(), the %G_SPAWN_SEARCH_PATH flag
 * is enabled. Note that %G_SPAWN_SEARCH_PATH can have security
 * implications, so consider using g_spawn_sync() directly if
 * appropriate. Possible errors are those from g_spawn_sync() and those
 * from g_shell_parse_argv().
 * 
 * Return value: TRUE on success, FALSE if an error was set
 **/
gboolean
g_spawn_command_line_sync (const gchar  *command_line,
                           gchar       **standard_output,
                           gchar       **standard_error,
                           gint         *exit_status,
                           GError      **error)
{
  gboolean retval;
  gchar **argv = 0;

  g_return_val_if_fail (command_line != NULL, FALSE);
  
  if (!g_shell_parse_argv (command_line,
                           NULL, &argv,
                           error))
    return FALSE;
  
  retval = g_spawn_sync (NULL,
                         argv,
                         NULL,
                         G_SPAWN_SEARCH_PATH,
                         NULL,
                         NULL,
                         standard_output,
                         standard_error,
                         exit_status,
                         error);
  g_strfreev (argv);

  return retval;
}

/**
 * g_spawn_command_line_async:
 * @command_line: a command line
 * @error: return location for errors
 * 
 * A simple version of g_spawn_async() that parses a command line with
 * g_shell_parse_argv() and passes it to g_spawn_async(). Runs a
 * command line in the background. Unlike g_spawn_async(), the
 * %G_SPAWN_SEARCH_PATH flag is enabled, other flags are not. Note
 * that %G_SPAWN_SEARCH_PATH can have security implications, so
 * consider using g_spawn_async() directly if appropriate. Possible
 * errors are those from g_shell_parse_argv() and g_spawn_async().
 * 
 * Return value: TRUE on success, FALSE if error is set.
 **/
gboolean
g_spawn_command_line_async (const gchar *command_line,
                            GError     **error)
{
  gboolean retval;
  gchar **argv = 0;

  g_return_val_if_fail (command_line != NULL, FALSE);

  if (!g_shell_parse_argv (command_line,
                           NULL, &argv,
                           error))
    return FALSE;
  
  retval = g_spawn_async (NULL,
                          argv,
                          NULL,
                          G_SPAWN_SEARCH_PATH,
                          NULL,
                          NULL,
                          NULL,
                          error);
  g_strfreev (argv);

  return retval;
}

static gint
do_exec (gboolean              dont_wait,
	 gint                  child_err_report_fd,
         gint                  stdin_fd,
         gint                  stdout_fd,
         gint                  stderr_fd,
         const gchar          *working_directory,
         gchar               **argv,
         gchar               **envp,
         gboolean              close_descriptors,
         gboolean              search_path,
         gboolean              stdout_to_null,
         gboolean              stderr_to_null,
         gboolean              child_inherits_stdin,
         GSpawnChildSetupFunc  child_setup,
         gpointer              user_data)
{
  const gchar **new_argv;
  gchar args[ARG_COUNT][10];
  gint i;
  int argc = 0;

  SETUP_DEBUG();

  while (argv[argc])
    ++argc;

  new_argv = g_new (gchar *, argc + 1 + ARG_COUNT);

  new_argv[0] = "gspawn-win32-helper";
  sprintf (args[ARG_CHILD_ERR_REPORT], "%d", child_err_report_fd);
  new_argv[ARG_CHILD_ERR_REPORT] = args[ARG_CHILD_ERR_REPORT];

  if (stdin_fd >= 0)
    {
      sprintf (args[ARG_STDIN], "%d", stdin_fd);
      new_argv[ARG_STDIN] = args[ARG_STDIN];
    }
  else if (child_inherits_stdin)
    {
      /* Let stdin be alone */
      new_argv[ARG_STDIN] = "-";
    }
  else
    {
      /* Keep process from blocking on a read of stdin */
      new_argv[ARG_STDIN] = "z";
    }

  if (stdout_fd >= 0)
    {
      sprintf (args[ARG_STDOUT], "%d", stdout_fd);
      new_argv[ARG_STDOUT] = args[ARG_STDOUT];
    }
  else if (stdout_to_null)
    {
      new_argv[ARG_STDOUT] = "z";
    }
  else
    {
      new_argv[ARG_STDOUT] = "-";
    }

  if (stderr_fd >= 0)
    {
      sprintf (args[ARG_STDERR], "%d", stderr_fd);
      new_argv[ARG_STDERR] = args[ARG_STDERR];
    }
  else if (stderr_to_null)
    {
      new_argv[ARG_STDERR] = "z";
    }
  else
    {
      new_argv[ARG_STDERR] = "-";
    }

  if (working_directory && *working_directory)
    new_argv[ARG_WORKING_DIRECTORY] = working_directory;
  else
    new_argv[ARG_WORKING_DIRECTORY] = "-";

  if (close_descriptors)
    new_argv[ARG_CLOSE_DESCRIPTORS] = "y";
  else
    new_argv[ARG_CLOSE_DESCRIPTORS] = "-";

  if (search_path)
    new_argv[ARG_USE_PATH] = "y";
  else
    new_argv[ARG_USE_PATH] = "-";

  if (dont_wait)
    new_argv[ARG_WAIT] = "-";
  else
    new_argv[ARG_WAIT] = "w";

  for (i = 0; i <= argc; i++)
    new_argv[ARG_PROGRAM + i] = argv[i];

  /* Call user function just before we execute the helper program,
   * which executes the program. Dunno what's the usefulness of this.
   * A child setup function used on Unix probably isn't of much use
   * as such on Win32, anyhow.
   */
  if (child_setup)
    {
      (* child_setup) (user_data);
    }

  if (debug)
    {
      g_print ("calling gspawn-win32-helper with argv:\n");
      for (i = 0; i < argc + 1 + ARG_COUNT; i++)
	g_print ("argv[%d]: %s\n", i, (new_argv[i] ? new_argv[i] : "NULL"));
    }
  
  if (envp != NULL)
    /* Let's hope envp hasn't mucked with PATH so that
     * gspawn-win32-helper.exe isn't found.
     */
    spawnvpe (P_NOWAIT, "gspawn-win32-helper", new_argv, envp);
  else
    spawnvp (P_NOWAIT, "gspawn-win32-helper", new_argv);

  /* FIXME: What if gspawn-win32-helper.exe isn't found? */

  /* Close the child_err_report_fd and the other process's ends of the
   * pipes in this process, otherwise the reader will never get
   * EOF.
   */
  close (child_err_report_fd);
  if (stdin_fd >= 0)
    close (stdin_fd);
  if (stdout_fd >= 0)
    close (stdout_fd);
  if (stderr_fd >= 0)
    close (stderr_fd);

  g_free ((void*)new_argv);

  return 0;
}

static gboolean
read_ints (int      fd,
           gint*    buf,
           gint     n_ints_in_buf,
           gint    *n_ints_read,
           GError **error)
{
  gint bytes = 0;
  
  while (bytes < sizeof(gint)*n_ints_in_buf)
    {
      gint chunk;

      if (debug)
	g_print ("%s:read_ints: trying to read %d bytes from pipe...\n",
		 __FILE__,
		 sizeof(gint)*n_ints_in_buf - bytes);

      chunk = read (fd, ((gchar*)buf) + bytes,
		    sizeof(gint)*n_ints_in_buf - bytes);

      if (debug)
	g_print ("... got %d bytes\n", chunk);
          
      if (chunk < 0)
        {
          /* Some weird shit happened, bail out */
              
          g_set_error (error,
                       G_SPAWN_ERROR,
                       G_SPAWN_ERROR_FAILED,
                       _("Failed to read from child pipe (%s)"),
                       g_strerror (errno));

          return FALSE;
        }
      else if (chunk == 0)
        break; /* EOF */
      else
	bytes += chunk;
    }

  *n_ints_read = bytes/sizeof(gint);

  return TRUE;
}

static gboolean
fork_exec_with_pipes (gboolean              dont_wait,
                      const gchar          *working_directory,
                      gchar               **argv,
                      gchar               **envp,
                      gboolean              close_descriptors,
                      gboolean              search_path,
                      gboolean              stdout_to_null,
                      gboolean              stderr_to_null,
		      gboolean              child_inherits_stdin,
                      GSpawnChildSetupFunc  child_setup,
                      gpointer              user_data,
                      gint                 *standard_input,
                      gint                 *standard_output,
                      gint                 *standard_error,
		      gint                 *exit_status,
                      GError              **error)     
{
  gint stdin_pipe[2] = { -1, -1 };
  gint stdout_pipe[2] = { -1, -1 };
  gint stderr_pipe[2] = { -1, -1 };
  gint child_err_report_pipe[2] = { -1, -1 };
  gint status;
  //gint bytes;
  gint buf[2];
  gint n_ints = 0;
  
  if (!make_pipe (child_err_report_pipe, error))
    return FALSE;

  if (standard_input && !make_pipe (stdin_pipe, error))
    goto cleanup_and_fail;
  
  if (standard_output && !make_pipe (stdout_pipe, error))
    goto cleanup_and_fail;

  if (standard_error && !make_pipe (stderr_pipe, error))
    goto cleanup_and_fail;

  status = do_exec (dont_wait,
		    child_err_report_pipe[1],
		    stdin_pipe[0],
		    stdout_pipe[1],
		    stderr_pipe[1],
		    working_directory,
		    argv,
		    envp,
		    close_descriptors,
		    search_path,
		    stdout_to_null,
		    stderr_to_null,
		    child_inherits_stdin,
		    child_setup,
		    user_data);
      
  if (!read_ints (child_err_report_pipe[0],
		  buf, 2, &n_ints,
		  error))
    goto cleanup_and_fail;
        
  if (n_ints == 2)
    {
      /* Error from the child. */
      
      switch (buf[0])
	{
	case CHILD_NO_ERROR:
	  break;
	  
	case CHILD_CHDIR_FAILED:
	  g_set_error (error,
		       G_SPAWN_ERROR,
		       G_SPAWN_ERROR_CHDIR,
		       _("Failed to change to directory '%s' (%s)"),
		       working_directory,
		       g_strerror (buf[1]));
	  goto cleanup_and_fail;
	  
	case CHILD_SPAWN_FAILED:
	  g_set_error (error,
		       G_SPAWN_ERROR,
		       G_SPAWN_ERROR_FAILED,
		       _("Failed to execute child process (%s)"),
		       g_strerror (buf[1]));
	  goto cleanup_and_fail;
	}
    }

  /* Success against all odds! return the information */
      
  if (standard_input)
    *standard_input = stdin_pipe[1];
  if (standard_output)
    *standard_output = stdout_pipe[0];
  if (standard_error)
    *standard_error = stderr_pipe[0];
  if (exit_status)
    *exit_status = status;
  
  return TRUE;

 cleanup_and_fail:
  close_and_invalidate (&child_err_report_pipe[0]);
  close_and_invalidate (&child_err_report_pipe[1]);
  close_and_invalidate (&stdin_pipe[0]);
  close_and_invalidate (&stdin_pipe[1]);
  close_and_invalidate (&stdout_pipe[0]);
  close_and_invalidate (&stdout_pipe[1]);
  close_and_invalidate (&stderr_pipe[0]);
  close_and_invalidate (&stderr_pipe[1]);

  return FALSE;
}

static gboolean
make_pipe (gint     p[2],
           GError **error)
{
  if (pipe (p) < 0)
    {
      g_set_error (error,
                   G_SPAWN_ERROR,
                   G_SPAWN_ERROR_FAILED,
                   _("Failed to create pipe for communicating with child process (%s)"),
                   g_strerror (errno));
      return FALSE;
    }
  else
    return TRUE;
}

#endif /* !GSPAWN_HELPER */
