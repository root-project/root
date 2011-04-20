/* @(#)root/clib:$Id$ */
/* Author: */

/*
 * Copyright (C) 1991, 1992 by Chris Thewalt (thewalt@ce.berkeley.edu)
 *
 * Permission to use, copy, modify, and distribute this software
 * for any purpose and without fee is hereby granted, provided
 * that the above copyright notices appear in all copies and that both the
 * copyright notice and this permission notice appear in supporting
 * documentation.  This software is provided "as is" without express or
 * implied warranty.
 */

/*
 *************************** Motivation **********************************

   Many interactive programs read input line by line, but would like to
   provide line editing and history functionality to the end-user that
   runs the program.

   The input-edit package provides that functionality.  As far as the
   programmer is concerned, the program only asks for the next line
   of input. However, until the user presses the RETURN key they can use
   emacs-style line editing commands and can traverse the history of lines
   previously typed.

   Other packages, such as GNU's readline, have greater capability but are
   also substantially larger.  Input-edit is small, since it uses neither
   stdio nor any termcap features, and is also quite portable.  It only uses
   \b to backspace and \007 to ring the bell on errors.  Since it cannot
   edit multiple lines it scrolls long lines left and right on the same line.

   Input edit uses classic (not ANSI) C, and should run on any Unix
   system (BSD or SYSV), PC's with the MSC compiler, or Vax/VMS (untested by me).
   Porting the package to new systems basicaly requires code to read a
   character when it is typed without echoing it, everything else should be OK.

   I have run the package on:

        DECstation 5000, Ultrix 4.2 with cc and gcc
        Sun Sparc 2, SunOS 4.1.1, with cc
        SGI Iris, IRIX System V.3, with cc
        PC, DRDOS 5.0, with MSC 6.0

   The description below is broken into two parts, the end-user (editing)
   interface and the programmer interface.  Send bug reports, fixes and
   enhancements to:

   Chris Thewalt (thewalt@ce.berkeley.edu)
   2/4/92

   PS: I don't have, and don't want to add, a vi mode, sorry.

 ************************** End-User Interface ***************************

   Entering printable keys generally inserts new text into the buffer (unless
   in overwrite mode, see below).  Other special keys can be used to modify
   the text in the buffer.  In the description of the keys below, ^n means
   Control-n, or holding the CONTROL key down while pressing "n". M-B means
   Meta-B (or Alt-B). Errors will ring the terminal bell.

   ^A/^E   : Move cursor to beginning/end of the line.
   ^F/^B   : Move cursor forward/backward one character.
   ^D      : Delete the character under the cursor.
   ^H, DEL : Delete the character to the left of the cursor.
   ^K      : Kill from the cursor to the end of line.
   ^L      : Redraw current line.
   ^O      : Toggle overwrite/insert mode. Initially in insert mode. Text
          added in overwrite mode (including yanks) overwrite
          existing text, while insert mode does not overwrite.
   ^P/^N   : Move to previous/next item on history list.
   ^R/^S   : Perform incremental reverse/forward search for string on
          the history list.  Typing normal characters adds to the current
          search string and searches for a match. Typing ^R/^S marks
          the start of a new search, and moves on to the next match.
          Typing ^H or DEL deletes the last character from the search
          string, and searches from the starting location of the last search.
          Therefore, repeated DEL's appear to unwind to the match nearest
          the point at which the last ^R or ^S was typed.  If DEL is
          repeated until the search string is empty the search location
          begins from the start of the history list.  Typing ESC or
          any other editing character accepts the current match and
          loads it into the buffer, terminating the search.
   ^T      : Toggle the characters under and to the left of the cursor.
   ^U      : Kill from beginning to the end of the line.
   ^Y      : Yank previously killed text back at current location.  Note that
          this will overwrite or insert, depending on the current mode.
   M-F/M-B : Move cursor forward/backward one word.
   M-D     : Delete the word under the cursor.
   ^SPC    : Set mark.
   ^W      : Kill from mark to point.
   ^X      : Exchange mark and point.
   TAB     : By default adds spaces to buffer to get to next TAB stop
          (just after every 8th column), although this may be rebound by the
          programmer, as described below.
   NL, CR  : returns current buffer to the program.

   DOS and ANSI terminal arrow key sequences are recognized, and act like:

   up    : same as ^P
   down  : same as ^N
   left  : same as ^B
   right : same as ^F

 ************************** Programmer Interface ***************************

   The programmer accesses input-edit through five functions, and optionally
   through three additional function pointer hooks.  The five functions are:

   char *Getline(const char *prompt)

        Prints the prompt and allows the user to edit the current line. A
        pointer to the line is returned when the user finishes by
        typing a newline or a return.  Unlike GNU readline, the returned
        pointer points to a static buffer, so it should not be free'd, and
        the buffer contains the newline character.  The user enters an
        end-of-file by typing ^D on an empty line, in which case the
        first character of the returned buffer is '\0'.  Getline never
        returns a NULL pointer.  The getline function sets terminal modes
        needed to make it work, and resets them before returning to the
        caller.  The getline function also looks for characters that would
        generate a signal, and resets the terminal modes before raising the
        signal condition.  If the signal handler returns to getline,
        the screen is automatically redrawn and editing can continue.
        Getline now requires both the input and output stream be connected
        to the terminal (not redirected) so the main program should check
        to make sure this is true.  If input or output have been redirected
        the main program should use buffered IO (stdio) rather than
        the slow 1 character read()s that getline uses (note: this limitation
        has been removed).

   char *Getlinem(int mode, const char *prompt)

        mode: -1 = init, 0 = line mode, 1 = one char at a time mode, 2 = cleanup

        More specialized version of the previous function. Depending on
        the mode, it behaves differently. Its main use is to allow
        character by character input from the input stream (useful when
        in an X eventloop). It will return NULL as long as no newline
        has been received. Its use is typically as follows:
        1) In the program initialization part one calls: Getlinem(-1,"prompt>")
        2) In the X inputhandler: if ((line = Getlinem(1,NULL))) {
        3) In the termination routine: Getlinem(2,NULL)
        With mode=0 the function behaves exactly like the previous function.

   void Gl_config(const char *which, int value)

        Set some config options. Which can be:
          "noecho":  do not echo characters (used for passwd input)
          "erase":   do erase line after return (used for text scrollers)

   void Gl_setwidth(int width)

        Set the width of the terminal to the specified width. The default
        width is 80 characters, so this function need only be called if the
        width of the terminal is not 80.  Since horizontal scrolling is
        controlled by this parameter it is important to get it right.

   void Gl_histinit(char *file)

        This function reads a history list from file. So lines from a
        previous session can be used again.

   void Gl_histadd(char *buf)

        The Gl_histadd function checks to see if the buf is not empty or
        whitespace, and also checks to make sure it is different than
        the last saved buffer to avoid repeats on the history list.
        If the buf is a new non-blank string a copy is made and saved on
        the history list, so the caller can re-use the specified buf.

   The main loop in testgl.c, included in this directory, shows how the
   input-edit package can be used:

   extern char *Getline();
   extern void  Gl_histadd();
   main()
   {
    char *p;
    Gl_histinit(".hist");
    do {
        p = Getline("PROMPT>>>> ");
        Gl_histadd(p);
        fputs(p, stdout);
    } while (*p != 0);
   }

   In order to allow the main program to have additional access to the buffer,
   to implement things such as completion or auto-indent modes, three
   function pointers can be bound to user functions to modify the buffer as
   described below.  By default Gl_in_hook and Gl_out_hook are set to NULL,
   and Gl_tab_hook is bound to a function that inserts spaces until the next
   logical tab stop is reached.  The user can reassign any of these pointers
   to other functions.  Each of the functions bound to these hooks receives
   the current buffer as the first argument, and must return the location of
   the leftmost change made in the buffer.  If the buffer isn't modified the
   functions should return -1.  When the hook function returns the screen is
   updated to reflect any changes made by the user function.

   int (*Gl_tab_hook)(char *buf, int prompt_width, int *cursor_loc)

        If Gl_tab_hook is non-NULL, it is called whenever a tab is typed.
        In addition to receiving the buffer, the current prompt width is
        given (needed to do tabbing right) and a pointer to the cursor
        offset is given, where a 0 offset means the first character in the
        line.  Not only does the cursor_loc tell the programmer where the
        TAB was received, but it can be reset so that the cursor will end
        up at the specified location after the screen is redrawn.

   int (*Gl_beep_hook)()
        Called if \007 (beep) is about to be printed. Return !=0 if handled.
 */

extern "C" {
/********************* exported interface ********************************/


char* Getline(const char* prompt);   /* read a line of input */
char* Getlinem(int mode, const char* prompt);   /* allows reading char by char */
void Gl_config(const char* which, int value);    /* set some options */
void Gl_setwidth(int w);             /* specify width of screen */
void Gl_windowchanged();             /* call after SIGWINCH signal */
int Gl_eof();
void Gl_histinit(char* file);    /* read entries from old histfile */
void Gl_histadd(char* buf);          /* adds entries to hist */
void Gl_setColors(const char* colorTab, const char* colorTabComp, const char* colorBracket,
                  const char* colorBadBracket, const char* colorPrompt);    /* set the colours (replace default colours) for enhanced output */

int (* Gl_tab_hook)(char* buf, int prompt_width, int* loc) = 0;
int (* Gl_beep_hook)() = 0;
int (* Gl_in_key)(int ch) = 0;
}

/******************** imported interface *********************************/

#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <list>
#include <string>
#include <fstream>

/** newer imported interfaces **/
#include "editline.h"

const char* hist_file = 0;  // file name for the command history (read and write)

/******************** internal interface *********************************/

#define BUF_SIZE 1024

// # History file size, once HistSize is reached remove all but HistSave entries,
// # set to 0 to turn off command recording.
// # Can be overridden by environment variable ROOT_HIST=size[:save],
// # the ":save" part is optional.
// # Rint.HistSize:         500
// # Set to -1 for sensible default (80% of HistSize), set to 0 to disable history.
// # Rint.HistSave:         400
int size_lines = 500;
int save_lines = -1;

/************************ nonportable part *********************************/

extern "C" {
void
Gl_config(const char* which, int value) {
   if (strcmp(which, "noecho") == 0) {
      setEcho(!value);
   } else {
      // unsupported directive
      printf("gl_config: %s ?\n", which);
   }
}


/******************** fairly portable part *********************************/

void
Gl_setwidth(int /*w*/) {
   termResize();  // no need to pass in width as new func detects term size itself
}


void
Gl_windowchanged() {
#ifdef TIOCGWINSZ

   if (isatty(0)) {
      static char lenv[32], cenv[32];
      struct winsize wins;
      ioctl(0, TIOCGWINSZ, &wins);

      if (wins.ws_col == 0) {
         wins.ws_col = 80;
      }

      if (wins.ws_row == 0) {
         wins.ws_row = 24;
      }

      Gl_setwidth(wins.ws_col);

      sprintf(lenv, "LINES=%d", wins.ws_row);
      putenv(lenv);
      sprintf(cenv, "COLUMNS=%d", wins.ws_col);
      putenv(cenv);
   }
#endif
} // Gl_windowchanged


/* The new and hopefully improved Getlinem method!
 * Uses readline() from libeditline.
 * History_t and editing are also handled by libeditline.
 * Modes: -1 = init, 0 = line mode, 1 = one char at a time mode, 2 = cleanup
 */
char*
Getlinem(int mode, const char* prompt) {
   static char sprompt[80] = { 0 };
   char* input_buffer;
   rl_tab_hook = Gl_tab_hook;
   rl_in_key_hook = Gl_in_key;

   static int getline_initialized = 0;

   if (hist_file && getline_initialized == 0) {
      //rl_initialize();		// rl_initialize already being called by history_stifle()
      read_history(hist_file);
      getline_initialized = 1;
   }

   // mode 2 = cleanup
   if (mode == 2) {
      rl_reset_terminal();
   }

   // mode -1 = init
   if (mode == -1) {
      if (prompt) {
         strncpy(sprompt, prompt, sizeof(sprompt) - 1);
         sprompt[sizeof(sprompt) - 1] = 0; // force 0 termination
      }
      input_buffer = readline(sprompt, true /*newline*/);

      return input_buffer;
   }

   // mode 1 = one char at a time
   if (mode == 1) {
      if (prompt) {
         strncpy(sprompt, prompt, sizeof(sprompt) - 1);
         sprompt[sizeof(sprompt) - 1] = 0; // force 0 termination
      }

      // note: input_buffer will be null unless complete line entered
      input_buffer = readline(sprompt, false /*no newline*/);

      // if complete line is entered, add to history and return buffer, otherwise return null
      char* ch = input_buffer;

      if (input_buffer) {
         if (!*input_buffer) {
            // signal EOF
            return input_buffer;
         }
         while (*ch && *ch != '\a') {
            if (*ch == '\n') {
               // line complete!
               return input_buffer;
            }
            ++ch;
         }
      }
   }
   return NULL;
} // Getlinem


void
Gl_setColors(const char* colorTab, const char* colorTabComp, const char* colorBracket,
             const char* colorBadBracket, const char* colorPrompt) {
   // call to enhance.cxx to set colours
   setColors(colorTab, colorTabComp, colorBracket, colorBadBracket, colorPrompt);
}


char*
Getline(const char* prompt) {
   // Get a line of user input, showing prompt.
   // Does not return after every character entered, but
   // only returns once the user has hit return.
   // For ROOT Getline.c backward compatibility reasons,
   // the returned value is volatile and will be overwritten
   // by the subsequent call to Getline() or Getlinem(),
   // so copy the string if it needs to stay around.
   // The returned value must not be deleted.
   // The returned string contains a trailing newline '\n'.

   Getlinem(-1, prompt); // init
   char* answer = 0;
   do {
      answer = Getlinem(1, prompt);
   } while (!answer);
   return answer;
}


/******************* History_t stuff **************************************/

void
Gl_histsize(int size, int save) {
   stifle_history(save);
   size_lines = size;
   save_lines = save;
}


void
Gl_histinit(char* file) {
   if (size_lines == 0 || save_lines == 0) {
      // history recording disabled
      return;
   }

   hist_file = file;
   if (size_lines > 0) {
      int linecount = 0;
      std::list<std::string> lines;
      {
         std::ifstream in(file);
         if (!in) {
            return;
         }

         lines.push_back(std::string());
         while(in && std::getline(in, lines.back())) {
            lines.push_back(std::string());
            ++linecount;
         }
         lines.pop_back();
      }

      if (linecount > size_lines) {
         // we need to reduce it to 
         if (save_lines == -1) {
            // set default
            save_lines = size_lines * 80 / 100;
         }
         std::ofstream out(file);
         if (!out) {
            return;
         }

         int skipLines = linecount - save_lines;
         for (std::list<std::string>::const_iterator iS = lines.begin(),
                 eS = lines.end(); iS != eS; ++iS) {
            if (skipLines) {
               --skipLines;
            } else {
               out << *iS << std::endl;
            }
         }

      }
   }
}


void
Gl_histadd(char* buf) {
   // Add to history; write the file out in case
   // the process is abort()ed by executing the line.
   add_history(buf);
   if (hist_file) {
      write_history(hist_file);
   }
}


int
Gl_eof() {
   return rl_eof();
}


} // extern "C"
