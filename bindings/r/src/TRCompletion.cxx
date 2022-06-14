// Author: Omar Zapata  Omar.Zapata@cern.ch   2014

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include"TRCompletion.h"
namespace ROOT {
   namespace R {

      SEXP RComp_assignBufferSym,
           RComp_assignStartSym,
           RComp_assignEndSym,
           RComp_assignTokenSym,
           RComp_completeTokenSym,
           RComp_getFileCompSym,
           RComp_retrieveCompsSym;

      SEXP rcompgen_rho;
   }
}

char *ROOT::R::R_completion_generator(const char *text, int state)
{
   // If this is a new word to complete, initialize now.  This
   // involves saving 'text' to somewhere R can get it, calling
   // completeToken(), and retrieving the completions.
   //NOTE: R based code and ajusted to Rcpp
   static int list_index, ncomp;
   static char **compstrings;


   if (!state) {
      int i;
      SEXP completions,
           assignCall = PROTECT(Rf_lang2(ROOT::R::RComp_assignTokenSym, Rf_mkString(text))),
           completionCall = PROTECT(Rf_lang1(ROOT::R::RComp_completeTokenSym)),
           retrieveCall = PROTECT(Rf_lang1(ROOT::R::RComp_retrieveCompsSym));
      const void *vmax = vmaxget();

      Rf_eval(assignCall, ROOT::R::rcompgen_rho);
      Rf_eval(completionCall, ROOT::R::rcompgen_rho);
      PROTECT(completions = Rf_eval(retrieveCall, ROOT::R::rcompgen_rho));
      list_index = 0;
      ncomp = Rf_length(completions);
      if (ncomp > 0) {
         compstrings = (char **) malloc(ncomp * sizeof(char *));
         if (!compstrings)  return (char *)NULL;
         for (i = 0; i < ncomp; i++)
            compstrings[i] = strdup(Rf_translateChar(STRING_ELT(completions, i)));
      }
      UNPROTECT(4);
      vmaxset(vmax);
   }

   if (list_index < ncomp)
      return compstrings[list_index++];
   else {
      /* nothing matched or remaining, returns NULL. */
      if (ncomp > 0) free(compstrings);
   }
   return (char *)NULL;
}


char **ROOT::R::R_custom_completion(const char *text, int start, int end)
{
   //NOTE: R based code and ajusted to Rcpp
   char **matches = (char **)NULL;
   SEXP infile,
        linebufferCall = PROTECT(Rf_lang2(ROOT::R::RComp_assignBufferSym,
                                          Rf_mkString(rl_line_buffer))),
                         startCall = PROTECT(Rf_lang2(ROOT::R::RComp_assignStartSym, Rf_ScalarInteger(start))),
                         endCall = PROTECT(Rf_lang2(ROOT::R::RComp_assignEndSym, Rf_ScalarInteger(end)));
   SEXP filecompCall;

   // We don't want spaces appended at the end. It's nedded everytime
   // since readline>=6 resets it to ' '
   rl_completion_append_character = '\0';

   Rf_eval(linebufferCall, ROOT::R::rcompgen_rho);
   Rf_eval(startCall, ROOT::R::rcompgen_rho);
   Rf_eval(endCall, ROOT::R::rcompgen_rho);
   UNPROTECT(3);
   matches = rl_completion_matches(text, ROOT::R::R_completion_generator);
   filecompCall = PROTECT(Rf_lang1(ROOT::R::RComp_getFileCompSym));
   infile = PROTECT(Rf_eval(filecompCall, ROOT::R::rcompgen_rho));
   if (!Rf_asLogical(infile)) rl_attempted_completion_over = 1;
   UNPROTECT(2);
   return matches;
}
