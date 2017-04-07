// @(#)root/r:$Id$
// Author: Omar Zapata   29/08/2013
// The tab-completion interface was based in R's readline code.
#ifndef ROOT_R_TRCompletion
#define ROOT_R_TRCompletion

#include <RExports.h>

#if !defined(_READLINE_H_)

#if !defined (PARAMS)
#  if defined (__STDC__) || defined (__GNUC__) || defined (__cplusplus)
#    define PARAMS(protos) protos
#  else
#    define PARAMS(protos) ()
#  endif
#endif
extern "C"
{
   typedef char **rl_completion_func_t PARAMS((const char *, int, int));
   typedef char *rl_compentry_func_t PARAMS((const char *, int));
   extern char **rl_completion_matches PARAMS((const char *, rl_compentry_func_t *));
   extern char *readline PARAMS((const char *));
   extern void add_history PARAMS((const char *));
   extern rl_completion_func_t *rl_attempted_completion_function;
   extern char *rl_line_buffer;
   extern int rl_completion_append_character;
   extern int rl_attempted_completion_over;
}
#endif //_READLINE_H_

namespace ROOT {
   namespace R {
      char *R_completion_generator(const char *text, int state);
      char **R_custom_completion(const char *text, int start, int end);
      //Readline variables.
      extern SEXP
      RComp_assignBufferSym,
      RComp_assignStartSym,
      RComp_assignEndSym,
      RComp_assignTokenSym,
      RComp_completeTokenSym,
      RComp_getFileCompSym,
      RComp_retrieveCompsSym;
      extern SEXP rcompgen_rho;
   }
}

#endif
