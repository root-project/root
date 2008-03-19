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

using namespace Cint::Internal;

#ifdef G__EH_SIGNAL
static int Cint::Internal::G__error_handle_flag=0;
#endif

#ifdef G__EH_SIGNAL
/************************************************************************
* New interrupt routine
************************************************************************/
void Cint::Internal::G__error_handle(int signame)
{
  G__error_handle_flag=1;
  switch(signame) {
#ifdef SIGFPE
  case SIGFPE:
    G__genericerror("Floating exception");
    signal(SIGFPE,G__error_handle);
    break;
#endif
#ifdef SIGSEGV
  case SIGSEGV:
    G__genericerror("Segmentation violation");
    signal(SIGSEGV,G__error_handle);
    break;
#endif
#ifdef SIGILL
  case SIGILL:
    G__genericerror("Illegal instruction");
    signal(SIGILL,G__error_handle);
    break;
#endif
#ifdef SIGEMT
  case SIGEMT:
    G__genericerror("Out of memory");
    signal(SIGEMT,G__error_handle);
    break;
#endif
#ifdef SIGBUS
  case SIGBUS:
    G__genericerror("Bus error");
    signal(SIGBUS,G__error_handle);
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
extern "C" void G__breakkey(int signame)
{
  /*********************************************************
   * stop browsing 
   *********************************************************/
  G__browsing=0;

  /*********************************************************
   * G__step is set so that the program stops at next 
   * interpreted statement.
   *********************************************************/
  G__step++;
  G__setdebugcond();
  G__disp_mask=0;

  /*********************************************************
   * immediate pause in prerun
   *********************************************************/
  if(G__prerun) {
    G__fprinterr(G__serr,"\n!!! Pause at prerun. signal(%d)\n",signame);
    G__step--;
    G__setdebugcond();
    G__pause();
  }
  /*********************************************************
   * immediate pause if called twice
   *********************************************************/
  else if(G__step>1) {
    G__fprinterr(G__serr,"\n!!! Break in the middle of compiled statement. signal(%d)\n",signame);
    G__pause();
    if(G__return>G__RETURN_NORMAL) {
      G__fprinterr(G__serr, "!!! Sorry, continue until compiled code finishes\n");
      G__fprinterr(G__serr, "!!! Use qqq for immediate termination\n");
    }
  }
  else if(G__asm_exec) {
    G__fprinterr(G__serr, "\n!!! Middle of loop compilation run. signal(%d)\n",signame);
  }
  signal(SIGINT,G__breakkey);
}


/******************************************************************
* G__killproc()
******************************************************************/
void Cint::Internal::G__killproc(int signame)
{
  fprintf(G__sout,"\n!!! Process killed by interrupt. signal(%d)\n",signame);
  G__exit(EXIT_FAILURE);
}

/******************************************************************
* G__errorprompt()
******************************************************************/
int Cint::Internal::G__errorprompt(char *nameoferror)
{

#ifdef G__EH_SIGNAL
  if(G__error_handle_flag) {
    G__error_handle_flag=0;
    return(0);
  }
#endif

  G__step=1;
  G__setdebugcond();

  G__genericerror(nameoferror);
#ifdef G__MEMTEST
  fprintf(G__memhist,"%s\n",nameoferror);
#endif

  G__no_exec=0;
  fflush(G__sout);
  fflush(G__serr);


#ifdef SIGALRM
  G__fprinterr(G__serr,
          "Press return or process will be terminated in %dsec by timeout\n"
          ,G__TIMEOUT);
  signal(SIGALRM,G__timeout);
  alarm(G__TIMEOUT);
#endif

  G__pause();

#ifdef SIGALRM
  alarm(0);
  G__fprinterr(G__serr,"Time out cancelled\n");
#endif

  while(G__return<G__RETURN_EXIT1) {
    if(G__pause()) break;
  }

  if(G__return>=G__RETURN_EXIT1) {
    G__close_inputfiles();
    exit(EXIT_FAILURE);
  }
  return(0);
}

/******************************************************************
* G__timeout()
******************************************************************/
void Cint::Internal::G__timeout(int signame)
{
  G__fprinterr(G__serr,"\nsignal(%d) Error time out. Exit program.\n",signame);

  G__close_inputfiles();
  exit(EXIT_FAILURE);
}

/******************************************************************
* G__floatexception()
******************************************************************/
void Cint::Internal::G__floatexception(int signame)
{
  G__fprinterr(G__serr,"signal(%d) ",signame); 
  signal(SIGFPE,G__floatexception);
  G__errorprompt("Error: Floating point exception");
}

/******************************************************************
* G__segmentviolation()
******************************************************************/
void Cint::Internal::G__segmentviolation(int signame)
{
  G__fprinterr(G__serr,"signal(%d) ",signame); 
  signal(SIGSEGV,G__segmentviolation);
  G__errorprompt("Error: Segmentation violation");
}

/******************************************************************
* G__outofmemory()
******************************************************************/
void Cint::Internal::G__outofmemory(int signame)
{
  G__fprinterr(G__serr,"signal(%d) ",signame); 
#ifdef SIGEMT
  signal(SIGEMT,G__outofmemory);
#endif
  G__errorprompt("Error: Out of memory");
}

/******************************************************************
* G__buserror()
******************************************************************/
void Cint::Internal::G__buserror(int signame)
{
  G__fprinterr(G__serr,"signal(%d) ",signame); 
#ifdef SIGBUS
  signal(SIGBUS,G__buserror);
#endif
  G__errorprompt("Error: Bus error");
}

/******************************************************************
* G__errorexit()
******************************************************************/
void Cint::Internal::G__errorexit(int signame)
{
  G__fprinterr(G__serr,"Error: caught signal(%d)\n",signame); 
  signal(signame,SIG_DFL);
  exit(EXIT_FAILURE);
}


/************************************************************************
* End of Interrupt routines
************************************************************************/


#ifdef G__SIGNAL
/******************************************************************
* G__call_interrupt()
******************************************************************/
int Cint::Internal::G__call_interruptfunc(char *func)
{
#ifdef G__ASM
  G__ALLOC_ASMENV;
#endif
  int store_var_type;

#ifdef G__ASM
  G__STORE_ASMENV;
#endif
  store_var_type = G__var_type;
  G__var_type='p';
  G__getexpr(func);
#ifdef G__ASM
  G__RECOVER_ASMENV;
#endif
  G__var_type=store_var_type;
  return(0);
}

/******************************************************************
* G__fsigabrt()
******************************************************************/
void Cint::Internal::G__fsigabrt(int)
{
  char temp[G__ONELINE];
  signal(SIGABRT,SIG_DFL);
  if(G__SIGABRT) {
#define G__OLDIMPLEMENTATION1945
    sprintf(temp,"%s()",G__SIGABRT);
    G__SIGABRT = NULL;
    G__call_interruptfunc(temp);
  }
}

/******************************************************************
* G__fsigfpe()
******************************************************************/
void Cint::Internal::G__fsigfpe(int)
{
  char temp[G__ONELINE];
  signal(SIGFPE,G__floatexception);
  if(G__SIGFPE) {
    sprintf(temp,"%s()",G__SIGFPE);
    G__SIGFPE = NULL;
    G__call_interruptfunc(temp);
  }
}

/******************************************************************
* G__fsigill()
******************************************************************/
void Cint::Internal::G__fsigill(int)
{
  char temp[G__ONELINE];
  signal(SIGILL,SIG_DFL);
  if(G__SIGILL) {
    sprintf(temp,"%s()",G__SIGILL);
    G__SIGILL = NULL;
    G__call_interruptfunc(temp);
  }
}

/******************************************************************
* G__fsigint()
******************************************************************/
void Cint::Internal::G__fsigint(int)
{
  char temp[G__ONELINE];
  signal(SIGINT,G__breakkey);
  if(G__SIGINT) {
    sprintf(temp,"%s()",G__SIGINT);
    G__SIGINT = NULL;
    G__call_interruptfunc(temp);
  }
}

/******************************************************************
* G__fsigsegv()
******************************************************************/
void Cint::Internal::G__fsigsegv(int)
{
  char temp[G__ONELINE];
  signal(SIGSEGV,G__segmentviolation);
  if(G__SIGSEGV) {
    sprintf(temp,"%s()",G__SIGSEGV);
    G__SIGSEGV = NULL;
    G__call_interruptfunc(temp);
  }
}

/******************************************************************
* G__fsigterm()
******************************************************************/
void Cint::Internal::G__fsigterm(int)
{
  char temp[G__ONELINE];
  signal(SIGTERM,SIG_DFL);
  if(G__SIGTERM) {
    sprintf(temp,"%s()",G__SIGTERM);
    G__SIGTERM = NULL;
    G__call_interruptfunc(temp);
  }
}

#ifdef SIGHUP
/******************************************************************
* G__fsighup()
******************************************************************/
void Cint::Internal::G__fsighup(int)
{
  char temp[G__ONELINE];
  signal(SIGHUP,SIG_DFL);
  if(G__SIGHUP) {
    sprintf(temp,"%s()",G__SIGHUP);
    G__SIGHUP = NULL;
    G__call_interruptfunc(temp);
  }
}
#endif

#ifdef SIGQUIT
/******************************************************************
* G__fsigquit()
******************************************************************/
void Cint::Internal::G__fsigquit(int)
{
  char temp[G__ONELINE];
  signal(SIGQUIT,SIG_DFL);
  if(G__SIGQUIT) {
    sprintf(temp,"%s()",G__SIGQUIT);
    G__SIGQUIT = NULL;
    G__call_interruptfunc(temp);
  }
}
#endif

#ifdef SIGTSTP
/******************************************************************
* G__fsigtstp()
******************************************************************/
void Cint::Internal::G__fsigtstp(int)
{
  char temp[G__ONELINE];
  signal(SIGTSTP,SIG_DFL);
  if(G__SIGTSTP) {
    sprintf(temp,"%s()",G__SIGTSTP);
    G__SIGTSTP = NULL;
    G__call_interruptfunc(temp);
  }
}
#endif

#ifdef SIGTTIN
/******************************************************************
* G__fsigttin()
******************************************************************/
void Cint::Internal::G__fsigttin(int)
{
  char temp[G__ONELINE];
  signal(SIGTTIN,SIG_DFL);
  if(G__SIGTTIN) {
    sprintf(temp,"%s()",G__SIGTTIN);
    G__SIGTTIN = NULL;
    G__call_interruptfunc(temp);
  }
}
#endif

#ifdef SIGTTOU
/******************************************************************
* G__fsigttou()
******************************************************************/
void Cint::Internal::G__fsigttou(int)
{
  char temp[G__ONELINE];
  signal(SIGTTOU,SIG_DFL);
  if(G__SIGTTOU) {
    sprintf(temp,"%s()",G__SIGTTOU);
    G__SIGTTOU = NULL;
    G__call_interruptfunc(temp);
  }
}
#endif

#ifdef SIGALRM
/******************************************************************
* G__fsigalrm()
******************************************************************/
void Cint::Internal::G__fsigalrm(int)
{
  char temp[G__ONELINE];
  signal(SIGALRM,SIG_DFL);
  if(G__SIGALRM) {
    sprintf(temp,"%s()",G__SIGALRM);
    G__SIGALRM = NULL;
    G__call_interruptfunc(temp);
  }
}
#endif

#ifdef SIGUSR1
/******************************************************************
* G__fsigusr1()
******************************************************************/
void Cint::Internal::G__fsigusr1(int)
{
  char temp[G__ONELINE];
  signal(SIGUSR1,SIG_DFL);
  if(G__SIGUSR1) {
    sprintf(temp,"%s()",G__SIGUSR1);
    G__SIGUSR1 = NULL;
    G__call_interruptfunc(temp);
  }
}
#endif

#ifdef SIGUSR2
/******************************************************************
* G__fsigusr2()
******************************************************************/
void Cint::Internal::G__fsigusr2(int)
{
  char temp[G__ONELINE];
  signal(SIGUSR2,SIG_DFL);
  if(G__SIGUSR2) {
    sprintf(temp,"%s()",G__SIGUSR2);
    G__SIGUSR2 = NULL;
    G__call_interruptfunc(temp);
  }
}
#endif

#endif /* G__SIGNAL */

/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:2
 * c-continued-statement-offset:2
 * c-brace-offset:-2
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-2
 * compile-command:"make -k"
 * End:
 */
