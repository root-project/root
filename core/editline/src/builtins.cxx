// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

////////////////////////////////////////////////////////////////////////
// This file is part of the liblineedit code. See el.h for the
// full license (BSD).
// File added by stephan@s11n.net, 28 Nov 2004
////////////////////////////////////////////////////////////////////////

#include "builtins.h"
#include "editline.h"
#include <string>
#include <vector>
#include <map>


typedef std::vector<el_builtin_t*> BuiltinVec;
typedef std::map<std::string, el_builtin_t> BuiltinMap;

/**
   Returns the internal map of builtin functions.
 */
BuiltinMap&
el_internal_builtins() {
   static BuiltinMap el_builtins;
   return el_builtins;
}


void
el_builtins_init() {    // internal func

   if (0 == el_internal_builtins().size()) {
      el_register_function("el-bind", map_bind, "Bind keyboard commands.");             //   // map.h/c
      el_register_function("el-echotc", term_echotc, "???");             // term.h/c
      el_register_function("el-edit", el_editmode, "Toggle line-editing mode on and off.");             // el.h/c
      el_register_function("el-history", hist_list, "Show command history.");             // hist.h/c
      el_register_function("el-telltc", term_telltc, "???");             // term.h/c
      el_register_function("el-settc", term_settc, "???");             // term.h/c
      el_register_function("el-setty", tty_stty, "???");             // tty.h/c
      el_register_function("el-help", el_func_show_function_list, "Shows list of built-in functions.");
      el_register_function("el-rl-reset", el_func_readline_reinit, "Re-initializes readline compatibility layer.");
   }
}


int el_builtins_bogo_register = (el_builtins_init(), 0);


void
el_register_function(const char* name, el_builtin_t::handler_func f, const char* help) {
   el_internal_builtins()[name] = el_builtin_t(name, f, help);
}


BuiltinVec el_builtins_vec;
el_builtin_t**
el_builtins_list(int* count) {
   BuiltinMap::iterator it = el_internal_builtins().begin();
   BuiltinMap::iterator et = el_internal_builtins().end();
   el_builtins_vec.clear();
   *count = 0;

   for ( ; et != it; ++it) {
      el_builtins_vec.push_back(&(*it).second);
      ++*count;
   }
   // fprintf( stderr, "el_builtins_list() size=%d\n", *count );
   return &el_builtins_vec[0];
}


/**
   Calls rl_initialize(), to reset the internal EditLine to a sane
   state.
 */
int
el_func_readline_reinit(EditLine* el, int, const char**) {
   fprintf(el->el_outfile,
           "Reinitializing readline compatibility interface.\n");
   int ret = rl_initialize();
   return ret;
}


/**
   Shows list of built-in functions via el->el_outfile.
 */
int
el_func_show_function_list(EditLine* el, int, const char**) {
   fprintf(el->el_outfile, "List of libeditline builtin functions:\n");
   BuiltinMap::iterator it = el_internal_builtins().begin();
   BuiltinMap::iterator et = el_internal_builtins().end();
   el_builtin_t* bi = 0;

   for ( ; et != it; ++it) {
      bi = &((*it).second);
      fprintf(el->el_outfile,
              "%-32s\t\t%s\n",
              (bi->name),
              ((NULL != bi->help) ? bi->help : "")
      );
   }
   return 0;
}


el_builtin_t*
el_builtin_by_name(const char* name) {
   BuiltinMap::iterator it = el_internal_builtins().find(name);
   BuiltinMap::iterator et = el_internal_builtins().end();

   if (et == it) {
      return NULL;
   }
   return &(*it).second;
}
