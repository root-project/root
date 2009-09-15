// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

#ifndef _el_builtins_h_included
#define _el_builtins_h_included 1
////////////////////////////////////////////////////////////////////////
// This file is part of the liblineedit code. See el.h for the
// full license (BSD).
// File added by stephan@s11n.net, 28 Nov 2004
////////////////////////////////////////////////////////////////////////

#include "el.h"


/**
   el_builtin_t holds information about builtin functions.
 */
el_public struct el_builtin_t {
   typedef int (* handler_func)(EditLine*, int, const char**);
   const char* name;
   handler_func func;
   const char* help;
   el_builtin_t(): name(0),
      func(0),
      help(0) {}

   el_builtin_t(const char* n, handler_func f, const char* h): name(n),
      func(f),
      help(h) {}

};

/**
   Registers f as a built-in function handler.

   (added by stephan)
 */
void el_register_function(const char* name, el_builtin_t::handler_func f, const char* help = NULL);

/**
   Built-in function to show the list of available functions to the
   user.

   (added by stephan)
 */
int el_func_show_function_list(EditLine* el, int, const char**);     // impl: parse.c

/*
   Built-in to re-initialize the readline interface. Only useful
   when using the readline compatibility mode.

   (added by stephan)
 */
int el_func_readline_reinit(EditLine*, int, const char**);     // impl: readline.c

/**
   Returns an array of el_builtin_t objects. The caller does NOT own it - it is a shared
   list. The list may become invalided upon later registration of functions, so don't
   hold on to it.

   The variable count is set to the number of items in the returned list.

   (added by stephan)
 */
el_builtin_t** el_builtins_list(int* count);

/**
   Returns the builtin function for the given name, or NULL if none is
   found. The client may modify the function, but should not delete it
   - it is owned by the library.

   (added by stephan)
 */
el_builtin_t* el_builtin_by_name(const char*);

#endif // _el_builtins_h_included
