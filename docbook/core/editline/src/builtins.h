// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

#ifndef _el_builtins_h_included
#define _el_builtins_h_included 1
////////////////////////////////////////////////////////////////////////
// This file is part of the liblineedit code. See el.fH for the
// full license (BSD).
// File added by stephan@s11n.net, 28 Nov 2004
////////////////////////////////////////////////////////////////////////

#include "el.h"


/**
   ElBuiltin_t holds information about builtin functions.
 */
el_public struct ElBuiltin_t {
   typedef int (* handler_func)(EditLine_t*, int, const char**);
   const char* fName;
   handler_func fFunc;
   const char* fHelp;
   ElBuiltin_t(): fName(0),
      fFunc(0),
      fHelp(0) {}

   ElBuiltin_t(const char* n, handler_func f, const char* h): fName(n),
      fFunc(f),
      fHelp(h) {}

};

/**
   Registers f as a built-in function handler.

   (added by stephan)
 */
void el_register_function(const char* name, ElBuiltin_t::handler_func f, const char* help = NULL);

/**
   Built-in function to show the list of available functions to the
   user.

   (added by stephan)
 */
int el_func_show_function_list(EditLine_t* el, int, const char**);     // impl: parse.c

/*
   Built-in to re-initialize the readline interface. Only useful
   when using the readline compatibility mode.

   (added by stephan)
 */
int el_func_readline_reinit(EditLine_t*, int, const char**);     // impl: readline.c

/**
   Returns an array of ElBuiltin_t objects. The caller does NOT own it - it is a shared
   list. The list may become invalided upon later registration of functions, so don't
   hold on to it.

   The variable count is set to the number of items in the returned list.

   (added by stephan)
 */
ElBuiltin_t** el_builtins_list(int* count);

/**
   Returns the builtin function for the given name, or NULL if none is
   found. The client may modify the function, but should not delete it
   - it is owned by the library.

   (added by stephan)
 */
ElBuiltin_t* el_builtin_by_name(const char*);

#endif // _el_builtins_h_included
