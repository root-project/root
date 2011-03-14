/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file intrpt.c
 ************************************************************************
 * Description:
 *  Signal handling function
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#ifdef WIN32
#include <io.h>
#endif

extern "C" {

extern int G__browsing; /* used in disp.c and intrpt.c */

#ifdef G__EH_SIGNAL
static int G__error_handle_flag = 0;
#endif

#ifdef G__EH_SIGNAL
/************************************************************************
* New interrupt routine
************************************************************************/
void G__error_handle(signame)
int signame;
{
   G__error_handle_flag = 1;
   switch (signame) {
#ifdef SIGFPE
      case SIGFPE:
         G__genericerror("Floating exception");
         signal(SIGFPE, G__error_handle);
         break;
#endif
#ifdef SIGSEGV
      case SIGSEGV:
         G__genericerror("Segmentation violation");
         signal(SIGSEGV, G__error_handle);
         break;
#endif
#ifdef SIGILL
      case SIGILL:
         G__genericerror("Illegal instruction");
         signal(SIGILL, G__error_handle);
         break;
#endif
#ifdef SIGEMT
      case SIGEMT:
         G__genericerror("Out of memory");
         signal(SIGEMT, G__error_handle);
         break;
#endif
#ifdef SIGBUS
      case SIGBUS:
         G__genericerror("Bus error");
         signal(SIGBUS, G__error_handle);
         break;
#endif
      default:
         G__genericerror("Error: Unknown");
         break;
   }
}
#endif

/************************************************************************
* Interrupt routines
*
*  keyboard interrupt
*  floating exception
*  out of memory
*  kill process
*  segmentation violation
*  bus error
*  time out
************************************************************************/

/******************************************************************
* G__breakkey()
******************************************************************/
void G__breakkey(int signame)
{
   /*********************************************************
    * stop browsing 
    *********************************************************/
   G__browsing = 0;

   /*********************************************************
    * G__step is set so that the program stops at next 
    * interpreted statement.
    *********************************************************/
   G__step++;
   G__setdebugcond();
   G__disp_mask = 0;

   /*********************************************************
    * immediate pause in prerun
    *********************************************************/
   if (G__prerun) {
      G__fprinterr(G__serr, "\n!!! Pause at prerun. signal(%d)\n", signame);
      G__step--;
      G__setdebugcond();
      G__pause();
   }
   /*********************************************************
    * immediate pause if called twice
    *********************************************************/
   else if (G__step > 1) {
      G__fprinterr(G__serr, "\n!!! Break in the middle of compiled statement. signal(%d)\n", signame);
      G__pause();
      if (G__return > G__RETURN_NORMAL) {
         G__fprinterr(G__serr, "!!! Sorry, continue until compiled code finishes\n");
         G__fprinterr(G__serr, "!!! Use qqq for immediate termination\n");
      }
   }
   else if (G__asm_exec) {
      G__fprinterr(G__serr, "\n!!! Middle of loop compilation run. signal(%d)\n", signame);
   }
   signal(SIGINT, G__breakkey);
}


/******************************************************************
* G__killproc()
******************************************************************/
void G__killproc(int signame)
{
   fprintf(G__sout, "\n!!! Process killed by interrupt. signal(%d)\n", signame);
   G__exit(EXIT_FAILURE);
}

/******************************************************************
* G__errorprompt()
******************************************************************/
int G__errorprompt(const char *nameoferror)
{
   // --
#ifdef G__EH_SIGNAL
   if (G__error_handle_flag) {
      G__error_handle_flag = 0;
      return 0;
   }
#endif
   G__step = 1;
   G__setdebugcond();
   G__genericerror(nameoferror);
#ifdef G__MEMTEST
   fprintf(G__memhist, "%s\n", nameoferror);
#endif
   G__no_exec = 0;
   fflush(G__sout);
   fflush(G__serr);
#ifdef WIN32
   if (! isatty(0) ) {
#else
   if (! isatty(0) || (getpgrp() != tcgetpgrp(STDOUT_FILENO)) ) {
#endif
      // If the input is not a tty or we are in the background, no need to ask the user!
      G__close_inputfiles();
      exit(EXIT_FAILURE);
   }
#ifdef SIGALRM
   G__fprinterr(G__serr, "\n\nPress return or process will be terminated in %d sec by timeout.\n", G__TIMEOUT);
   fflush(G__serr);
   signal(SIGALRM, G__timeout);
   alarm(G__TIMEOUT);
#endif
   G__pause();
#ifdef SIGALRM
   alarm(0);
   G__fprinterr(G__serr, "\n\nTimeout cancelled.\n");
   fflush(G__serr);
#endif
   while (G__return < G__RETURN_EXIT1) {
      int stat = G__pause();
      if (stat) {
         break;
      }
   }
   if (G__return >= G__RETURN_EXIT1) {
      G__close_inputfiles();
      exit(EXIT_FAILURE);
   }
   return 0;
}

/******************************************************************
* G__timeout()
******************************************************************/
void G__timeout(int signame)
{
   fflush(G__serr);
   G__fprinterr(G__serr, "\n\nSignal(%d) Error time out. Exit program.\n", signame);
   fflush(G__serr);
   G__close_inputfiles();
   exit(EXIT_FAILURE);
}

/******************************************************************
* G__floatexception()
******************************************************************/
void G__floatexception(int signame)
{
   G__fprinterr(G__serr, "signal(%d) ", signame);
   signal(SIGFPE, G__floatexception);
   G__errorprompt("Error: Floating point exception");
}

/******************************************************************
* G__segmentviolation()
******************************************************************/
void G__segmentviolation(int signame)
{
   G__fprinterr(G__serr, "signal(%d) ", signame);
   signal(SIGSEGV, G__segmentviolation);
   G__errorprompt("Error: Segmentation violation");
}

/******************************************************************
* G__outofmemory()
******************************************************************/
void G__outofmemory(int signame)
{
   G__fprinterr(G__serr, "signal(%d) ", signame);
#ifdef SIGEMT
   signal(SIGEMT, G__outofmemory);
#endif
   G__errorprompt("Error: Out of memory");
}

/******************************************************************
* G__buserror()
******************************************************************/
void G__buserror(int signame)
{
   G__fprinterr(G__serr, "signal(%d) ", signame);
#ifdef SIGBUS
   signal(SIGBUS, G__buserror);
#endif
   G__errorprompt("Error: Bus error");
}

/******************************************************************
* G__errorexit()
******************************************************************/
void G__errorexit(int signame)
{
   G__fprinterr(G__serr, "Error: caught signal(%d)\n", signame);
   signal(signame, SIG_DFL);
   exit(EXIT_FAILURE);
}


/************************************************************************
* End of Interrupt routines
************************************************************************/


#ifdef G__SIGNAL
/******************************************************************
* G__call_interrupt()
******************************************************************/
int G__call_interruptfunc(char *func)
{
#ifdef G__ASM
   G__ALLOC_ASMENV;
#endif
   char store_var_type;

#ifdef G__ASM
   G__STORE_ASMENV;
#endif
   store_var_type = G__var_type;
   G__var_type = 'p';
   G__getexpr(func);
#ifdef G__ASM
   G__RECOVER_ASMENV;
#endif
   G__var_type = store_var_type;
   return(0);
}

#define G__OLDIMPLEMENTATION1945

#ifdef _MVC_VER
# define G__PASTE0(A) A
# define G__PASTE(A,B) G__PASTE0(A)B
#else
# define G__PASTE(A,B) A##B
#endif

#define G__DEF_FSIG(FUNC, SIG, CINTSIG, ARG)      \
   void FUNC(int) { \
   G__FastAllocString temp(G__ONELINE); \
   signal(SIG, ARG); \
   if (CINTSIG) { \
      temp.Format("%s()", CINTSIG); \
      CINTSIG = NULL; \
      G__call_interruptfunc(temp); \
   } \
}
      

/******************************************************************
* G__fsigabrt()
******************************************************************/
G__DEF_FSIG(G__fsigabrt, SIGABRT, G__SIGABRT, SIG_DFL)

/******************************************************************
* G__fsigfpe()
******************************************************************/
G__DEF_FSIG(G__fsigfpe, SIGFPE, G__SIGFPE, G__floatexception)

/******************************************************************
* G__fsigill()
******************************************************************/
G__DEF_FSIG(G__fsigill, SIGILL, G__SIGILL, SIG_DFL)

/******************************************************************
* G__fsigint()
******************************************************************/
G__DEF_FSIG(G__fsigint, SIGINT, G__SIGINT, G__breakkey)

/******************************************************************
* G__fsigsegv()
******************************************************************/
G__DEF_FSIG(G__fsigsegv, SIGSEGV, G__SIGSEGV, G__segmentviolation)

/******************************************************************
* G__fsigterm()
******************************************************************/
G__DEF_FSIG(G__fsigterm, SIGTERM, G__SIGTERM, SIG_DFL)

#ifdef SIGHUP
/******************************************************************
* G__fsighup()
******************************************************************/
G__DEF_FSIG(G__fsighup, SIGHUP, G__SIGHUP, SIG_DFL)
#endif

#ifdef SIGQUIT
/******************************************************************
* G__fsigquit()
******************************************************************/
G__DEF_FSIG(G__fsigquit, SIGQUIT, G__SIGQUIT, SIG_DFL)
#endif

#ifdef SIGTSTP
/******************************************************************
* G__fsigtstp()
******************************************************************/
G__DEF_FSIG(G__fsigtstp, SIGTSTP, G__SIGTSTP, SIG_DFL)
#endif

#ifdef SIGTTIN
/******************************************************************
* G__fsigttin()
******************************************************************/
G__DEF_FSIG(G__fsigttin, SIGTTIN, G__SIGTTIN, SIG_DFL)
#endif

#ifdef SIGTTOU
/******************************************************************
* G__fsigttou()
******************************************************************/
G__DEF_FSIG(G__fsigttou, SIGTTOU, G__SIGTTOU, SIG_DFL)
#endif

#ifdef SIGALRM
/******************************************************************
* G__fsigalrm()
******************************************************************/
G__DEF_FSIG(G__fsigalrm, SIGALRM, G__SIGALRM, SIG_DFL)
#endif

#ifdef SIGUSR1
/******************************************************************
* G__fsigusr1()
******************************************************************/
G__DEF_FSIG(G__fsigusr1, SIGUSR1, G__SIGUSR1, SIG_DFL);
#endif

#ifdef SIGUSR2
/******************************************************************
* G__fsigusr2()
******************************************************************/
G__DEF_FSIG(G__fsigusr2, SIGUSR2, G__SIGUSR2, SIG_DFL);
#endif

#endif /* G__SIGNAL */

} /* extern "C" */

/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:3
 * c-continued-statement-offset:3
 * c-brace-offset:-3
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-3
 * compile-command:"make -k"
 * End:
 */
