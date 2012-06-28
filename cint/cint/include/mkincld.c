/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file include/mkincld.c
 ************************************************************************
 * Description:
 *  Create standard include files at cint installation
 ************************************************************************
 * Author                  Masaharu Goto
 * Copyright(c) 1995~1999  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
#include <limits.h>
#include <time.h>
#ifndef G__NONANSI
#ifndef G__SUNOS4
#include <float.h>
#endif
#endif
#include <math.h>
#include <errno.h>
#include <stdlib.h>
#include <locale.h>
#include <stddef.h>
#include <signal.h>

#ifdef G__NEWSOS4
#include <ctype.h>
#ifndef _IOFBF
#define _IOFBF (0)
#endif
#ifndef SIG_ERR
#define SIG_ERR (-1)
#endif
typedef int clock_t;
#endif /* G__NEWSOS4 */

/* char *dir="include/"; */
char *dir="";


/*******************************************************************
* check_pointersize()
*******************************************************************/
int check_pointersize()
{
  fprintf(stdout,"sizeof(long)=%ld , sizeof(void*)=%ld\n"
	  ,(long)sizeof(long),(long)sizeof(void*));
  if(sizeof(long)<sizeof(void*)) {
    fprintf(stderr,"\n");
    fprintf(stderr,"CINT INSTALLATION FATAL ERROR :\n");
    fprintf(stderr,"        sizeof(long)=%ld < sizeof(void*)=%ld\n"
	    ,(long)sizeof(long),(long)sizeof(void*));
    fprintf(stderr,"  SORRY, CAN NOT INSTALL CINT.\n");
    fprintf(stderr,"Size of long must be greater or equal to size of void*\n\n");
    exit(EXIT_FAILURE);
  }
  return(0);
}

/*******************************************************************
* testdup()
*******************************************************************/
int testdup(fp,iden)
FILE *fp;
char *iden;
{
  fprintf(fp,"#ifndef %s\n",iden);
  fprintf(fp,"#define %s\n",iden);
  return(0);
}


/*******************************************************************
* 
*******************************************************************/
#define UINT_TYPEDEF_PREFER_INT(fp,type,ctype)                          \
  if(sizeof(type)==sizeof(unsigned char))                               \
    fprintf(fp,"typedef unsigned char %s;\n",ctype);                    \
  else if(sizeof(type)==sizeof(unsigned short))                         \
    fprintf(fp,"typedef unsigned short %s;\n",ctype);                   \
  else if(sizeof(type)==sizeof(unsigned int))                           \
    fprintf(fp,"typedef unsigned int %s;\n",ctype);                     \
  else if(sizeof(type)==sizeof(unsigned long))                          \
    fprintf(fp,"typedef unsigned long %s;\n",ctype);                    \
  else if(sizeof(type)==sizeof(unsigned long)*2) {                      \
    fprintf(fp,"typedef struct %s {\n",ctype);                          \
    fprintf(fp,"  unsigned long l,u;\n");                               \
    fprintf(fp,"  %s(unsigned long i=0){l=i;u=0;}\n",ctype);            \
    fprintf(fp,"  void operator=(unsigned long i){l=i;u=0;}\n");        \
    fprintf(fp,"} %s;\n",ctype);                                        \
    fprintf(fp,"#pragma link off class %s;\n",ctype);                   \
    fprintf(fp,"#pragma link off typedef %s;\n",ctype);                 \
  }                                                                     \
  else {                                                                \
    fprintf(fp,"typedef struct %s {\n",ctype);                          \
    fprintf(fp,"  char dmy[%ld];\n",(long)sizeof(type));                \
    fprintf(fp,"} %s;\n",ctype);                                        \
    fprintf(fp,"#pragma link off class %s;\n",ctype);                   \
    fprintf(fp,"#pragma link off typedef %s;\n",ctype);                 \
  }

#define INT_TYPEDEF_PREFER_INT(fp,type,ctype)                           \
  if(sizeof(type)==sizeof(char))                                        \
    fprintf(fp,"typedef char %s;\n",ctype);                             \
  else if(sizeof(type)==sizeof(short))                                  \
    fprintf(fp,"typedef short %s;\n",ctype);                            \
  else if(sizeof(type)==sizeof(int))                                    \
    fprintf(fp,"typedef int %s;\n",ctype);                              \
  else if(sizeof(type)==sizeof(long))                                   \
    fprintf(fp,"typedef long %s;\n",ctype);                             \
  else if(sizeof(type)==sizeof(long)*2) {                               \
    fprintf(fp,"typedef struct %s {\n",ctype);                          \
    fprintf(fp,"  long l,u;\n");                                        \
    fprintf(fp,"  %s(long i=0){l=i;u=0;}\n",ctype);                     \
    fprintf(fp,"  void operator=(long i){l=i;u=0;}\n");                 \
    fprintf(fp,"} %s;\n",ctype);                                        \
    fprintf(fp,"#pragma link off class %s;\n",ctype);                   \
    fprintf(fp,"#pragma link off typedef %s;\n",ctype);                 \
  }                                                                     \
  else {                                                                \
    fprintf(fp,"typedef struct %s {\n",ctype);                          \
    fprintf(fp,"  char dmy[%ld];\n",(long)sizeof(type));                \
    fprintf(fp,"} %s;\n",ctype);                                        \
    fprintf(fp,"#pragma link off class %s;\n",ctype);                   \
    fprintf(fp,"#pragma link off typedef %s;\n",ctype);                 \
  }

#define UINT_TYPEDEF_PREFER_LONG(fp,type,ctype)                         \
  if(sizeof(type)==sizeof(unsigned char))                               \
    fprintf(fp,"typedef unsigned char %s;\n",ctype);                    \
  else if(sizeof(type)==sizeof(unsigned short))                         \
    fprintf(fp,"typedef unsigned short %s;\n",ctype);                   \
  else if(sizeof(type)==sizeof(unsigned long))                          \
    fprintf(fp,"typedef unsigned long %s;\n",ctype);                    \
  else if(sizeof(type)==sizeof(unsigned int))                           \
    fprintf(fp,"typedef unsigned int %s;\n",ctype);                     \
  else if(sizeof(type)==sizeof(unsigned long)*2) {                      \
    fprintf(fp,"typedef struct %s {\n",ctype);                          \
    fprintf(fp,"  unsigned long l,u;\n");                               \
    fprintf(fp,"  %s(unsigned long i=0){l=i;u=0;}\n",ctype);            \
    fprintf(fp,"  void operator=(unsigned long i){l=i;u=0;}\n");        \
    fprintf(fp,"} %s;\n",ctype);                                        \
    fprintf(fp,"#pragma link off class %s;\n",ctype);                   \
    fprintf(fp,"#pragma link off typedef %s;\n",ctype);                 \
  }                                                                     \
  else {                                                                \
    fprintf(fp,"typedef struct %s {\n",ctype);                          \
    fprintf(fp,"  char dmy[%ld];\n",(long)sizeof(type));                \
    fprintf(fp,"} %s;\n",ctype);                                        \
    fprintf(fp,"#pragma link off class %s;\n",ctype);                   \
    fprintf(fp,"#pragma link off typedef %s;\n",ctype);                 \
  }

#define INT_TYPEDEF_PREFER_LONG(fp,type,ctype)                          \
  if(sizeof(type)==sizeof(char))                                        \
    fprintf(fp,"typedef char %s;\n",ctype);                             \
  else if(sizeof(type)==sizeof(short))                                  \
    fprintf(fp,"typedef short %s;\n",ctype);                            \
  else if(sizeof(type)==sizeof(long))                                   \
    fprintf(fp,"typedef long %s;\n",ctype);                             \
  else if(sizeof(type)==sizeof(int))                                    \
    fprintf(fp,"typedef int %s;\n",ctype);                              \
  else if(sizeof(type)==sizeof(long)*2) {                               \
    fprintf(fp,"typedef struct %s {\n",ctype);                          \
    fprintf(fp,"  long l,u;\n");                                        \
    fprintf(fp,"  %s(long i=0){l=i;u=0;}\n",ctype);                     \
    fprintf(fp,"  void operator=(long i){l=i;u=0;}\n");                 \
    fprintf(fp,"} %s;\n",ctype);                                        \
    fprintf(fp,"#pragma link off class %s;\n",ctype);                   \
    fprintf(fp,"#pragma link off typedef %s;\n",ctype);                 \
  }                                                                     \
  else {                                                                \
    fprintf(fp,"typedef struct %s {\n",ctype);                          \
    fprintf(fp,"  char dmy[%ld];\n",(long)sizeof(type));                \
    fprintf(fp,"} %s;\n",ctype);                                        \
    fprintf(fp,"#pragma link off class %s;\n",ctype);                   \
    fprintf(fp,"#pragma link off typedef %s;\n",ctype);                 \
  }

/*******************************************************************
*******************************************************************/
int gen_stdio()
{
  FILE *fp;
  char filename[200];
  char *header="stdio.h";

  sprintf(filename,"%s%s",dir,header);
  fp=fopen(filename,"w");
  testdup(fp,"G__STDIO_H");

#ifndef G__OLDIMPLEMENTATION280
  fprintf(fp,"#ifndef NULL\n");
  fprintf(fp,"#pragma setstdio\n");
  fprintf(fp,"#endif\n");
#endif

  INT_TYPEDEF_PREFER_LONG(fp,fpos_t,"fpos_t");
/* see v6_init.cxx, G__platformMacro
  UINT_TYPEDEF_PREFER_INT(fp,size_t,"size_t"); */
  fprintf(fp,"#define \t_IOFBF (%d)\n",_IOFBF);
  fprintf(fp,"#define \t_IOLBF (%d)\n",_IOLBF);
  fprintf(fp,"#define \t_IONBF (%d)\n",_IONBF);
  fprintf(fp,"#define \tBUFSIZ (%d)\n",BUFSIZ);
  /* EOF */
#ifndef G__NONANSI
#ifndef G__SUNOS4
  fprintf(fp,"#define \tFILENAME_MAX (%d)\n",FILENAME_MAX);
#endif
#endif
  /* fprintf(fp,"#define \tFOPEN_MAX (%d)\n",OPEN_MAX - G__MAXFILE); */
  fprintf(fp,"#define \tL_tmpnam (%d)\n",L_tmpnam);
#ifndef G__NONANSI
#ifndef G__SUNOS4
  fprintf(fp,"#define \tTMP_MAX (%d)\n",TMP_MAX);
#endif
#endif
  fprintf(fp,"#ifndef SEEK_CUR\n");
  fprintf(fp,"#define \tSEEK_CUR (%d)\n",SEEK_CUR);
  fprintf(fp,"#endif\n");
  fprintf(fp,"#ifndef SEEK_END\n");
  fprintf(fp,"#define \tSEEK_END (%d)\n",SEEK_END);
  fprintf(fp,"#endif\n");
  fprintf(fp,"#ifndef SEEK_SET\n");
  fprintf(fp,"#define \tSEEK_SET (%d)\n",SEEK_SET);
  fprintf(fp,"#endif\n");
  /* stderr */
  /* stdin */
  /* stdout */

  fprintf(fp,"#ifdef __cplusplus\n");
  fprintf(fp,"#include <bool.h>\n");
  fprintf(fp,"#endif\n");

  fprintf(fp,"#pragma include_noerr <stdfunc.dll>\n");

  fprintf(fp,"#endif\n");
  fclose(fp);
  return(0);
}

/*******************************************************************
*******************************************************************/
int gen_limits()
{
  FILE *fp;
  char filename[200];
  char *header="limits.h";

  sprintf(filename,"%s%s",dir,header);
  fp=fopen(filename,"w");
  testdup(fp,"G__LIMITS_H");

  fprintf(fp,"#define \tCHAR_BIT (%d)\n",CHAR_BIT);
  fprintf(fp,"#define \tCHAR_MAX (%d)\n",CHAR_MAX);
  fprintf(fp,"#define \tCHAR_MIN (%d)\n",CHAR_MIN);
  fprintf(fp,"#define \tINT_MAX (%d)\n",INT_MAX);
  fprintf(fp,"#define \tINT_MIN (%d)\n",INT_MIN);
  fprintf(fp,"#define \tLONG_MAX (%ld)\n",LONG_MAX);
  fprintf(fp,"#define \tLONG_MIN (%ld)\n",LONG_MIN);
  fprintf(fp,"#define \tSCHAR_MAX (%d)\n",SCHAR_MAX);
  fprintf(fp,"#define \tSCHAR_MIN (%d)\n",SCHAR_MIN);
  fprintf(fp,"#define \tSHRT_MAX (%d)\n",SHRT_MAX);
  fprintf(fp,"#define \tSHRT_MIN (%d)\n",SHRT_MIN);
  fprintf(fp,"#define \tUCHAR_MAX (%dU)\n",UCHAR_MAX);
  fprintf(fp,"const unsigned int  \tUINT_MAX =(%uU);\n",UINT_MAX);
  fprintf(fp,"const unsigned long \tULONG_MAX =(%luU);\n",ULONG_MAX);
  fprintf(fp,"#define \tUSHRT_MAX (%uU)\n",USHRT_MAX);

  fprintf(fp,"#endif\n");
  fclose(fp);
  return(0);
}

/*******************************************************************
*******************************************************************/
int gen_time()
{
  FILE *fp;
  char filename[200];
  char *header="time.h";

  sprintf(filename,"%s%s",dir,header);
  fp=fopen(filename,"w");
  testdup(fp,"G__TIME_H");

  UINT_TYPEDEF_PREFER_LONG(fp,clock_t,"clock_t");
  INT_TYPEDEF_PREFER_LONG(fp,time_t,"time_t");

#ifndef G__OLDIMPLEMENTATION467
  fprintf(fp,"#ifndef G__STDSTRUCT\n");
  fprintf(fp,"#pragma setstdstruct\n");
  fprintf(fp,"#endif\n");
#else
  fprintf(fp,"struct tm {\n");
  fprintf(fp,"   int	tm_sec;\n");
  fprintf(fp,"   int	tm_min;\n");
  fprintf(fp,"   int	tm_hour;\n");
  fprintf(fp,"   int	tm_mday;\n");
  fprintf(fp,"   int	tm_mon;\n");
  fprintf(fp,"   int	tm_year;\n");
  fprintf(fp,"   int	tm_wday;\n");
  fprintf(fp,"   int	tm_yday;\n");
  fprintf(fp,"   int	tm_isdst;\n");
  fprintf(fp,"};\n");
#endif

#ifdef CLK_TCK
  fprintf(fp,"#define \tCLK_TCK (%ld)\n",(long)CLK_TCK);
#endif

  fprintf(fp,"#endif\n");
  fclose(fp);
  return(0);
}

/*******************************************************************
*******************************************************************/
#ifndef G__NONANSI
#ifndef G__SUNOS4
int gen_float()
{
  FILE *fp;
  char filename[200];
  char *header="float.h";

  sprintf(filename,"%s%s",dir,header);
  fp=fopen(filename,"w");
  testdup(fp,"G__FLOAT_H");

  fprintf(fp,"#define \tDBL_DIG (%d)\n",DBL_DIG);
  fprintf(fp,"#define \tDBL_EPSILON (%g)\n",DBL_EPSILON);
  fprintf(fp,"#define \tDBL_MANT_DIG (%d)\n",DBL_MANT_DIG);
  fprintf(fp,"#define \tDBL_MAX (%g)\n",DBL_MAX);
  fprintf(fp,"#define \tDBL_MAX_10_EXP (%d)\n",DBL_MAX_10_EXP);
  fprintf(fp,"#define \tDBL_MAX_EXP (%d)\n",DBL_MAX_EXP);
  fprintf(fp,"#define \tDBL_MIN (%g)\n",DBL_MIN);
  fprintf(fp,"#define \tDBL_MIN_10_EXP (%d)\n",DBL_MIN_10_EXP);
  fprintf(fp,"#define \tDBL_MIN_EXP (%d)\n",DBL_MIN_EXP);

  fprintf(fp,"#define \tFLT_DIG (%d)\n",FLT_DIG);
  fprintf(fp,"#define \tFLT_EPSILON (%g)\n",FLT_EPSILON);
  fprintf(fp,"#define \tFLT_MANT_DIG (%d)\n",FLT_MANT_DIG);
  fprintf(fp,"#define \tFLT_MAX (%g)\n",FLT_MAX);
  fprintf(fp,"#define \tFLT_MAX_10_EXP (%d)\n",FLT_MAX_10_EXP);
  fprintf(fp,"#define \tFLT_MAX_EXP (%d)\n",FLT_MAX_EXP);
  fprintf(fp,"#define \tFLT_MIN (%g)\n",FLT_MIN);
  fprintf(fp,"#define \tFLT_MIN_10_EXP (%d)\n",FLT_MIN_10_EXP);
  fprintf(fp,"#define \tFLT_MIN_EXP (%d)\n",FLT_MIN_EXP);

  fprintf(fp,"#define \tFLT_RADIX (%d)\n",FLT_RADIX);
  fprintf(fp,"#define \tFLT_ROUNDS (%d)\n",FLT_ROUNDS);

  fprintf(fp,"#define \tLDBL_DIG (%d)\n",DBL_DIG);
  fprintf(fp,"#define \tLDBL_EPSILON (%g)\n",DBL_EPSILON);
  fprintf(fp,"#define \tLDBL_MANT_DIG (%d)\n",DBL_MANT_DIG);
  fprintf(fp,"#define \tLDBL_MAX (%g)\n",DBL_MAX);
  fprintf(fp,"#define \tLDBL_MAX_10_EXP (%d)\n",DBL_MAX_10_EXP);
  fprintf(fp,"#define \tLDBL_MAX_EXP (%d)\n",DBL_MAX_EXP);
  fprintf(fp,"#define \tLDBL_MIN (%g)\n",DBL_MIN);
  fprintf(fp,"#define \tLDBL_MIN_10_EXP (%d)\n",DBL_MIN_10_EXP);
  fprintf(fp,"#define \tLDBL_MIN_EXP (%d)\n",DBL_MIN_EXP);

  fprintf(fp,"#endif\n");
  fclose(fp);
  return(0);
}
#endif
#endif
  
/*******************************************************************
*******************************************************************/
int gen_math()
{
  FILE *fp;
  char filename[200];
  char *header="math.h";

  sprintf(filename,"%s%s",dir,header);
  fp=fopen(filename,"w");
  testdup(fp,"G__MATH_H");

  fprintf(fp,"#define \tEDOM (%d)\n",EDOM);
  fprintf(fp,"#define \tERANGE (%d)\n",ERANGE);
#if defined(G__NONANSI) || defined(G__SUNOS4)
  fprintf(fp,"#define \tHUGE_VAL (%g)\n",1.79e+308); /* HUGE_VAL */
#else
  fprintf(fp,"#define \tHUGE_VAL (%g)\n",DBL_MAX); /* HUGE_VAL */
#endif

  fprintf(fp,"#pragma include_noerr <stdfunc.dll>\n");

  fprintf(fp,"#endif\n");
  fclose(fp);
  return(0);
}

/*******************************************************************
*******************************************************************/
int gen_errno()
{
  FILE *fp;
  char filename[200];
  char *header="errno.h";

  sprintf(filename,"%s%s",dir,header);
  fp=fopen(filename,"w");
  testdup(fp,"G__ERRNO_H");

  fprintf(fp,"/* extern int errno; */\n");

  fprintf(fp,"#endif\n");
  fclose(fp);
  return(0);
}
  
/*******************************************************************
*******************************************************************/
int gen_stdlib()
{
  FILE *fp;
  char filename[200];
  char *header="stdlib.h";

  sprintf(filename,"%s%s",dir,header);
  fp=fopen(filename,"w");
  testdup(fp,"G__STDLIB_H");

#ifndef G__OLDIMPLEMENTATION467
  fprintf(fp,"#ifndef G__STDSTRUCT\n");
  fprintf(fp,"#pragma setstdstruct\n");
  fprintf(fp,"#endif\n");
#else
  fprintf(fp,"typedef struct {\n");
  fprintf(fp,"   int quot;\n");
  fprintf(fp,"   int rem;\n");
  fprintf(fp,"} div_t;\n");
  fprintf(fp,"typedef struct {\n");
  fprintf(fp,"   long int quot;\n");
  fprintf(fp,"   long int rem;\n");
  fprintf(fp,"} ldiv_t;\n");
#endif
/* see v6_init.cxx, G__platformMacro
  UINT_TYPEDEF_PREFER_INT(fp,size_t,"size_t"); */
  fprintf(fp,"#define \tEXIT_FAILURE (%d)\n",EXIT_FAILURE);
  fprintf(fp,"#define \tEXIT_SUCCESS (%d)\n",EXIT_SUCCESS);
  fprintf(fp,"#define \tMB_CUR_MAX (%ld)\n",(long)MB_CUR_MAX);
  fprintf(fp,"#define \tMB_LEN_MAX (%d)\n",MB_LEN_MAX);
#ifndef G__NONANSI
#ifndef G__SUNOS4
  fprintf(fp,"#define \tRAND_MAX (%d)\n",RAND_MAX);
#endif
#endif

#ifndef G__NONANSI
  UINT_TYPEDEF_PREFER_INT(fp,wchar_t,"wchar_t");
#else
  fprintf(fp,"typedef unsigned short wchar_t;\n");
#endif

  fprintf(fp,"#pragma include_noerr <stdfunc.dll>\n");

  fprintf(fp,"#endif\n");
  fclose(fp);
  return(0);
}

/*******************************************************************
*******************************************************************/
int gen_locale()
{
  FILE *fp;
  char filename[200];
  char *header="locale.h";

  sprintf(filename,"%s%s",dir,header);
  fp=fopen(filename,"w");
  testdup(fp,"G__LOCALE_H");

#ifndef G__OLDIMPLEMENTATION467
  fprintf(fp,"#ifndef G__STDSTRUCT\n");
  fprintf(fp,"#pragma setstdstruct\n");
  fprintf(fp,"#endif\n");
#else
  fprintf(fp,"struct lconv {\n");
  fprintf(fp,"   char *decimal_point;\n");
  fprintf(fp,"   char *thousands_sep;\n");
  fprintf(fp,"   char *grouping;\n");
  fprintf(fp,"   char *int_curr_symbol;\n");
  fprintf(fp,"   char *currency_symbol;\n");
  fprintf(fp,"   char *mon_decimal_point;\n");
  fprintf(fp,"   char *mon_thousands_sep;\n");
  fprintf(fp,"   char *mon_grouping;\n");
  fprintf(fp,"   char *positive_sign;\n");
  fprintf(fp,"   char *negative_sign;\n");
  fprintf(fp,"   char int_frac_digits;\n");
  fprintf(fp,"   char frac_digits;\n");
  fprintf(fp,"   char p_cs_precedes;\n");
  fprintf(fp,"   char p_sep_by_space;\n");
  fprintf(fp,"   char n_cs_precedes;\n");
  fprintf(fp,"   char n_sep_by_space;\n");
  fprintf(fp,"   char p_sign_posn;\n");
  fprintf(fp,"   char n_sign_posn;\n");
  fprintf(fp,"};\n");
#endif

  fprintf(fp,"#define \tLC_ALL (%d)\n",LC_ALL);
  fprintf(fp,"#define \tLC_COLLATE (%d)\n",LC_COLLATE);
  fprintf(fp,"#define \tLC_CTYPE (%d)\n",LC_CTYPE);
  fprintf(fp,"#define \tLC_MONETARY (%d)\n",LC_MONETARY);
  fprintf(fp,"#define \tLC_NUMERIC (%d)\n",LC_NUMERIC);
  fprintf(fp,"#define \tLC_TIME (%d)\n",LC_TIME);

  fprintf(fp,"#endif\n");
  fclose(fp);
  return(0);
}

/*******************************************************************
*******************************************************************/
int gen_stddef()
{
  FILE *fp;
  char filename[200];
  char *header="stddef.h";

  sprintf(filename,"%s%s",dir,header);
  fp=fopen(filename,"w");
  testdup(fp,"G__STDDEF_H");

  /* NULL */
  /* offsetof(); */
#ifdef __GNUC__
  fprintf(fp,"#if (G__GNUC==2)\n");
  INT_TYPEDEF_PREFER_INT(fp,ptrdiff_t,"ptrdiff_t");
  fprintf(fp,"#else\n");
  INT_TYPEDEF_PREFER_LONG(fp,ptrdiff_t,"ptrdiff_t");
  fprintf(fp,"#endif\n");
#else
  INT_TYPEDEF_PREFER_LONG(fp,ptrdiff_t,"ptrdiff_t");
#endif
/* see v6_init.cxx, G__platformMacro
  UINT_TYPEDEF_PREFER_INT(fp,size_t,"size_t"); */

#ifndef G__NONANSI
  UINT_TYPEDEF_PREFER_INT(fp,wchar_t,"wchar_t");
#else
  fprintf(fp,"typedef unsigned short wchar_t;\n");
#endif

  fprintf(fp,"#endif\n");
  fclose(fp);
  return(0);
}


/*******************************************************************
*******************************************************************/
int gen_signal()
{
  FILE *fp;
  char filename[200];
  char *header="signal.h";

  sprintf(filename,"%s%s",dir,header);
  fp=fopen(filename,"w");
  testdup(fp,"G__SIGNAL_H");

  fprintf(fp,"#define \tSIG_DFL (%ld)\n",(long)SIG_DFL);
  fprintf(fp,"#define \tSIG_ERR (%ld)\n",(long)SIG_ERR);
  fprintf(fp,"#define \tSIG_IGN (%ld)\n",(long)SIG_IGN);
  fprintf(fp,"#define \tSIGABRT (%d)\n",SIGABRT);
  fprintf(fp,"#define \tSIGFPE (%d)\n",SIGFPE);
  fprintf(fp,"#define \tSIGILL (%d)\n",SIGILL);
  fprintf(fp,"#define \tSIGINT (%d)\n",SIGINT);
  fprintf(fp,"#define \tSIGSEGV (%d)\n",SIGSEGV);
  fprintf(fp,"#define \tSIGTERM (%d)\n",SIGTERM);
  fprintf(fp,"/* non ANSI signals */\n");
#ifdef SIGHUP
  fprintf(fp,"#define \tSIGHUP (%d)\n",SIGHUP);
#endif
#ifdef SIGQUIT
  fprintf(fp,"#define \tSIGQUIT (%d)\n",SIGQUIT);
#endif
#ifdef SIGTSTP
  fprintf(fp,"#define \tSIGTSTP (%d)\n",SIGTSTP);
#endif
#ifdef SIGTTIN
  fprintf(fp,"#define \tSIGTTIN (%d)\n",SIGTTIN);
#endif
#ifdef SIGTTOU
  fprintf(fp,"#define \tSIGTTOU (%d)\n",SIGTTOU);
#endif
#ifdef SIGALRM
  fprintf(fp,"#define \tSIGALRM (%d)\n",SIGALRM);
#endif
#ifdef SIGUSR1
  fprintf(fp,"#define \tSIGUSR1 (%d)\n",SIGUSR1);
#endif
#ifdef SIGUSR2
  fprintf(fp,"#define \tSIGUSR2 (%d)\n",SIGUSR2);
#endif
#ifdef HELLO
  fprintf(fp,"#define \tHELLO (%d)\n",HELLO);
#endif

  fprintf(fp,"#endif\n");
  fclose(fp);
  return(0);
}

/*******************************************************************
* main()
*******************************************************************/
int main()
{
  check_pointersize();
  gen_stdio();
  gen_limits();
  gen_time();
#ifndef G__NONANSI
#ifndef G__SUNOS4
  gen_float();
#endif
#endif
  gen_math();
  gen_errno();
  gen_stdlib();
  gen_locale();
  gen_stddef();
  gen_signal();
  exit(EXIT_SUCCESS);
}
