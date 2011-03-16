/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file g__cfunc.c
 ************************************************************************
 * Description:
 *  ANSI C library function interface routine
 ************************************************************************
 * Copyright(c) 1995~2003  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


/* ANSI C */
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <signal.h>
#include <assert.h>
#include <time.h>
#include <locale.h>

/* POSIX.1 */
#include <fcntl.h>

/* POSIX.2 */

/* OTHER */
int getopt();

/* CINT */
#include "common.h"
/* #include "G__ci.h" */
/* #include "G__header.h" */
/* extern int G__no_exec_compile; */

#ifndef G__WIN32
#include <unistd.h>
#endif


#if defined(G__ANSI) || defined(G__ANSIHEADER)
/* int memcmp(const void *region1,const void *region2,size_t count);
   void* memcpy(void *region1,const void *region2,size_t n); */
#elif defined(_AIX)
/* IBM AIX */
int memcmp(const void *region1,const void *region2,size_t count);
void* memcpy(void *region1,const void *region2,size_t n);
#elif defined(G__NEWSOS4) || defined(G__NEWSOS6)
/* Sony NewsOS */
int memcmp();
#elif !defined(__MWERKS__) && !defined(__alpha) 
/* if not MetroWerks compiler on Mac or Alpha OSF or NewsOS */
int memcmp();
void* memcpy();
#endif
#if !defined(G__NOMATHLIB) && !defined(floor) \
   && defined(G__FUNCPOINTER) && (_MSC_VER >= 1400)
   static double G__floor_MSVC8(double f) {return floor(f);}
#endif

/*************************************************
* function prototype for standard C
*************************************************/


G__COMPLETIONLIST G__completionlist[] = {
#if defined(abort) || !defined(G__FUNCPOINTER)
	{"abort",NULL},
#else
	{"abort",(void (*)())abort},
#endif
#if defined(abs) || !defined(G__FUNCPOINTER)
	{"abs",NULL},
#else
	{"abs",(void (*)())abs},
#endif
#ifndef G__NOMATHLIB
#if defined(acos) || !defined(G__FUNCPOINTER)
	{"acos",NULL},
#else
	{"acos",(void (*)())acos},
#endif
#endif
#if defined(asctime) || !defined(G__FUNCPOINTER)
	{"asctime",NULL},
#else
	{"asctime",(void (*)())asctime},
#endif
#ifndef G__NOMATHLIB
#if defined(asin) || !defined(G__FUNCPOINTER)
	{"asin",NULL},
#else
	{"asin",(void (*)())asin},
#endif
#if defined(atan) || !defined(G__FUNCPOINTER)
	{"atan",NULL},
#else
	{"atan",(void (*)())atan},
#endif
#if defined(atan2) || !defined(G__FUNCPOINTER)
	{"atan2",NULL},
#else
	{"atan2",(void (*)())atan2},
#endif
#endif/* G__NOMATHLIB */
#if defined(atof) || !defined(G__FUNCPOINTER)
	{"atof",NULL},
#else
	{"atof",(void (*)())atof},
#endif
#if defined(atoi) || !defined(G__FUNCPOINTER)
	{"atoi",NULL},
#else
	{"atoi",(void (*)())atoi},
#endif
#if defined(atol) || !defined(G__FUNCPOINTER)
	{"atol",NULL},
#else
	{"atol",(void (*)())atol},
#endif
#if defined(calloc) || !defined(G__FUNCPOINTER)
	{"calloc",NULL},
#else
	{"calloc",(void (*)())calloc},
#endif
#if defined(ceil) || !defined(G__FUNCPOINTER)
	{"ceil",NULL},
#else
	{"ceil",(void (*)())ceil},
#endif
#if defined(clearerr) || !defined(G__FUNCPOINTER)
	{"clearerr",NULL},
#else
	{"clearerr",(void (*)())clearerr},
#endif
#if defined(clock) || !defined(G__FUNCPOINTER)
	{"clock",NULL},
#else
	{"clock",(void (*)())clock},
#endif
#ifndef G__NOMATHLIB
#if defined(cos) || !defined(G__FUNCPOINTER)
	{"cos",NULL},
#else
	{"cos",(void (*)())cos},
#endif
#if defined(cosh) || !defined(G__FUNCPOINTER)
	{"cosh",NULL},
#else
	{"cosh",(void (*)())cosh},
#endif
#endif /* G__NOMATHLIB */
#if defined(ctime) || !defined(G__FUNCPOINTER)
	{"ctime",NULL},
#else
	{"ctime",(void (*)())ctime},
#endif
#if defined(difftime) || !defined(G__FUNCPOINTER)
	{"difftime",NULL},
#else
	{"difftime",(void (*)())difftime},
#endif
#if !defined(G__NONANSI) && !defined(G__SUNOS4)
#if defined(div) || !defined(G__FUNCPOINTER)
	{"div",NULL},
#else
	{"div",(void (*)())div},
#endif
#endif
#ifndef G__NOMATHLIB
#if defined(exp) || !defined(G__FUNCPOINTER)
	{"exp",NULL},
#else
	{"exp",(void (*)())exp},
#endif
#if defined(fabs) || !defined(G__FUNCPOINTER)
	{"fabs",NULL},
#else
	{"fabs",(void (*)())fabs},
#endif
#endif /* G__NOMATHLIB */
#if defined(fclose) || !defined(G__FUNCPOINTER)
	{"fclose",NULL},
#else
	{"fclose",(void (*)())fclose},
#endif
#if defined(feof) || !defined(G__FUNCPOINTER)
	{"feof",NULL},
#else
	{"feof",(void (*)())feof},
#endif
#if defined(ferror) || !defined(G__FUNCPOINTER)
	{"ferror",NULL},
#else
	{"ferror",(void (*)())ferror},
#endif
#if defined(fflush) || !defined(G__FUNCPOINTER)
	{"fflush",NULL},
#else
	{"fflush",(void (*)())fflush},
#endif
#if defined(fgetc) || !defined(G__FUNCPOINTER)
	{"fgetc",NULL},
#else
	{"fgetc",(void (*)())fgetc},
#endif
#if defined(fgetpos) || !defined(G__FUNCPOINTER)
	{"fgetpos",NULL},
#else
	{"fgetpos",(void (*)())fgetpos},
#endif
#if defined(fgets) || !defined(G__FUNCPOINTER)
	{"fgets",NULL},
#else
	{"fgets",(void (*)())fgets},
#endif
#ifndef G__NOMATHLIB
#if defined(floor) || !defined(G__FUNCPOINTER)
	{"floor",NULL},
#else
#if _MSC_VER >= 1400
	{"floor",(void (*)())G__floor_MSVC8},
#else
	{"floor",(void (*)())floor},
#endif /* MSVC 1400 */
#endif
#if defined(fmod) || !defined(G__FUNCPOINTER)
	{"fmod",NULL},
#else
	{"fmod",(void (*)())fmod},
#endif
#endif /* G__NOMATHLIB */
#if defined(fopen) || !defined(G__FUNCPOINTER)
	{"fopen",NULL},
#else
	{"fopen",(void (*)())fopen},
#endif
#if defined(fputc) || !defined(G__FUNCPOINTER)
	{"fputc",NULL},
#else
	{"fputc",(void (*)())fputc},
#endif
#if defined(fputs) || !defined(G__FUNCPOINTER)
	{"fputs",NULL},
#else
	{"fputs",(void (*)())fputs},
#endif
#if defined(fread) || !defined(G__FUNCPOINTER)
	{"fread",NULL},
#else
	{"fread",(void (*)())fread},
#endif
#if defined(free) || !defined(G__FUNCPOINTER)
	{"free",NULL},
#else
	{"free",(void (*)())free},
#endif
#if defined(freopen) || !defined(G__FUNCPOINTER)
	{"freopen",NULL},
#else
	{"freopen",(void (*)())freopen},
#endif
#if defined(frexp) || !defined(G__FUNCPOINTER)
	{"frexp",NULL},
#else
	{"frexp",(void (*)())frexp},
#endif
#if defined(fseek) || !defined(G__FUNCPOINTER)
	{"fseek",NULL},
#else
	{"fseek",(void (*)())fseek},
#endif
#if defined(fsetpos) || !defined(G__FUNCPOINTER)
	{"fsetpos",NULL},
#else
	{"fsetpos",(void (*)())fsetpos},
#endif
#if defined(ftell) || !defined(G__FUNCPOINTER)
	{"ftell",NULL},
#else
	{"ftell",(void (*)())ftell},
#endif
#if defined(fwrite) || !defined(G__FUNCPOINTER)
	{"fwrite",NULL},
#else
	{"fwrite",(void (*)())fwrite},
#endif
#if defined(getc) || !defined(G__FUNCPOINTER)
	{"getc",NULL},
#else
	{"getc",(void (*)())getc},
#endif
#if defined(getchar) || !defined(G__FUNCPOINTER)
	{"getchar",NULL},
#else
	{"getchar",(void (*)())getchar},
#endif
#if defined(getenv) || !defined(G__FUNCPOINTER)
	{"getenv",NULL},
#else
	{"getenv",(void (*)())getenv},
#endif
	{"gets",NULL},
#if defined(gmtime) || !defined(G__FUNCPOINTER)
	{"gmtime",NULL},
#else
	{"gmtime",(void (*)())gmtime},
#endif
#if defined(isalnum) || !defined(G__FUNCPOINTER)
	{"isalnum",NULL},
#else
	{"isalnum",(void (*)())isalnum},
#endif
#if defined(isalpha) || !defined(G__FUNCPOINTER)
	{"isalpha",NULL},
#else
	{"isalpha",(void (*)())isalpha},
#endif
#if defined(iscntrl) || !defined(G__FUNCPOINTER)
	{"iscntrl",NULL},
#else
	{"iscntrl",(void (*)())iscntrl},
#endif
#if defined(isdigit) || !defined(G__FUNCPOINTER)
	{"isdigit",NULL},
#else
	{"isdigit",(void (*)())isdigit},
#endif
#if defined(isgraph) || !defined(G__FUNCPOINTER)
	{"isgraph",NULL},
#else
	{"isgraph",(void (*)())isgraph},
#endif
#if defined(islower) || !defined(G__FUNCPOINTER)
	{"islower",NULL},
#else
	{"islower",(void (*)())islower},
#endif
#if defined(isprint) || !defined(G__FUNCPOINTER)
	{"isprint",NULL},
#else
	{"isprint",(void (*)())isprint},
#endif
#if defined(ispunct) || !defined(G__FUNCPOINTER)
	{"ispunct",NULL},
#else
	{"ispunct",(void (*)())ispunct},
#endif
#if defined(isspace) || !defined(G__FUNCPOINTER)
	{"isspace",NULL},
#else
	{"isspace",(void (*)())isspace},
#endif
#if defined(isupper) || !defined(G__FUNCPOINTER)
	{"isupper",NULL},
#else
	{"isupper",(void (*)())isupper},
#endif
#if defined(isxdigit) || !defined(G__FUNCPOINTER)
	{"isxdigit",NULL},
#else
	{"isxdigit",(void (*)())isxdigit},
#endif
#if defined(labs) || !defined(G__FUNCPOINTER)
	{"labs",NULL},
#else
	{"labs",(void (*)())labs},
#endif
#ifndef G__NOMATHLIB
#if defined(ldexp) || !defined(G__FUNCPOINTER)
	{"ldexp",NULL},
#else
	{"ldexp",(void (*)())ldexp},
#endif
#endif /* G__NOMATHLIB */
#if !defined(G__NONANSI) && !defined(G__SUNOS4)
#if defined(ldiv) || !defined(G__FUNCPOINTER)
	{"ldiv",NULL},
#else
	{"ldiv",(void (*)())ldiv},
#endif
#endif
#if defined(localeconv) || !defined(G__FUNCPOINTER)
	{"localeconv",NULL},
#else
	{"localeconv",(void (*)())localeconv},
#endif
#if defined(localtime) || !defined(G__FUNCPOINTER)
	{"localtime",NULL},
#else
	{"localtime",(void (*)())localtime},
#endif
#ifndef G__NOMATHLIB
#if defined(log) || !defined(G__FUNCPOINTER)
	{"log",NULL},
#else
	{"log",(void (*)())log},
#endif
#if defined(log10) || !defined(G__FUNCPOINTER)
	{"log10",NULL},
#else
	{"log10",(void (*)())log10},
#endif
#endif /* G__NOMATHLIB */
#if defined(malloc) || !defined(G__FUNCPOINTER)
	{"malloc",NULL},
#else
	{"malloc",(void (*)())malloc},
#endif
#if defined(mblen) || !defined(G__FUNCPOINTER)
	{"mblen",NULL},
#else
	{"mblen",(void (*)())mblen},
#endif
#if defined(mbstowcs) || !defined(G__FUNCPOINTER)
	{"mbstowcs",NULL},
#else
	{"mbstowcs",(void (*)())mbstowcs},
#endif
#if defined(mbtowc) || !defined(G__FUNCPOINTER)
	{"mbtowc",NULL},
#else
	{"mbtowc",(void (*)())mbtowc},
#endif
#if defined(memchr) || !defined(G__FUNCPOINTER)
	{"memchr",NULL},
#else
	{"memchr",(void (*)())memchr},
#endif
#if defined(memcmp) || !defined(G__FUNCPOINTER)
	{"memcmp",NULL},
#else
	{"memcmp",(void (*)())memcmp},
#endif
#if defined(memcpy) || !defined(G__FUNCPOINTER)
	{"memcpy",NULL},
#else
	{"memcpy",(void (*)())memcpy},
#endif
#if defined(memmove) || !defined(G__FUNCPOINTER)
	{"memmove",NULL},
#else
	{"memmove",(void (*)())memmove},
#endif
#if defined(memset) || !defined(G__FUNCPOINTER)
	{"memset",NULL},
#else
	{"memset",(void (*)())memset},
#endif
#if defined(mktime) || !defined(G__FUNCPOINTER)
	{"mktime",NULL},
#else
	{"mktime",(void (*)())mktime},
#endif
#if defined(modf) || !defined(G__FUNCPOINTER)
	{"modf",NULL},
#else
	{"modf",(void (*)())modf},
#endif
#if defined(perror) || !defined(G__FUNCPOINTER)
	{"perror",NULL},
#else
	{"perror",(void (*)())perror},
#endif
#ifndef G__NOMATHLIB
#if defined(pow) || !defined(G__FUNCPOINTER)
	{"pow",NULL},
#else
	{"pow",(void (*)())pow},
#endif
#endif/* G__NOMATHLIB */
#if defined(putc) || !defined(G__FUNCPOINTER)
	{"putc",NULL},
#else
	{"putc",(void (*)())putc},
#endif
#if defined(putchar) || !defined(G__FUNCPOINTER)
	{"putchar",NULL},
#else
	{"putchar",(void (*)())putchar},
#endif
#if defined(puts) || !defined(G__FUNCPOINTER)
	{"puts",NULL},
#else
	{"puts",(void (*)())puts},
#endif
#if defined(raise) || !defined(G__FUNCPOINTER)
	{"raise",NULL},
#else
	{"raise",(void (*)())raise},
#endif
#if defined(rand) || !defined(G__FUNCPOINTER)
	{"rand",NULL},
#else
	{"rand",(void (*)())rand},
#endif
#if defined(realloc) || !defined(G__FUNCPOINTER)
	{"realloc",NULL},
#else
	{"realloc",(void (*)())realloc},
#endif
#if defined(remove) || !defined(G__FUNCPOINTER)
	{"remove",NULL},
#else
	{"remove",(void (*)())remove},
#endif
#if defined(rename) || !defined(G__FUNCPOINTER)
	{"rename",NULL},
#else
	{"rename",(void (*)())rename},
#endif
#if defined(rewind) || !defined(G__FUNCPOINTER)
	{"rewind",NULL},
#else
	{"rewind",(void (*)())rewind},
#endif
#if defined(setbuf) || !defined(G__FUNCPOINTER)
	{"setbuf",NULL},
#else
	{"setbuf",(void (*)())setbuf},
#endif
#if defined(setlocale) || !defined(G__FUNCPOINTER)
	{"setlocale",NULL},
#else
	{"setlocale",(void (*)())setlocale},
#endif
#if defined(setvbuf) || !defined(G__FUNCPOINTER)
	{"setvbuf",NULL},
#else
	{"setvbuf",(void (*)())setvbuf},
#endif
#ifndef G__NOMATHLIB
#if defined(sin) || !defined(G__FUNCPOINTER)
	{"sin",NULL},
#else
	{"sin",(void (*)())sin},
#endif
#if defined(sinh) || !defined(G__FUNCPOINTER)
	{"sinh",NULL},
#else
	{"sinh",(void (*)())sinh},
#endif
#if defined(sqrt) || !defined(G__FUNCPOINTER)
	{"sqrt",NULL},
#else
	{"sqrt",(void (*)())sqrt},
#endif
#endif /* G__NOMATHLIB */
#if defined(srand) || !defined(G__FUNCPOINTER)
	{"srand",NULL},
#else
	{"srand",(void (*)())srand},
#endif
#if defined(strcat) || !defined(G__FUNCPOINTER)
	{"strcat",NULL},
#else
	{"strcat",(void (*)())strcat},
#endif
#if defined(strchr) || !defined(G__FUNCPOINTER)
	{"strchr",NULL},
#else
	{"strchr",(void (*)())strchr},
#endif
#if defined(strcmp) || !defined(G__FUNCPOINTER)
	{"strcmp",NULL},
#else
	{"strcmp",(void (*)())strcmp},
#endif
#if defined(strcoll) || !defined(G__FUNCPOINTER)
	{"strcoll",NULL},
#else
	{"strcoll",(void (*)())strcoll},
#endif
#if defined(strcpy) || !defined(G__FUNCPOINTER)
	{"strcpy",NULL},
#else
	{"strcpy",(void (*)())strcpy},
#endif
#if defined(strcspn) || !defined(G__FUNCPOINTER)
	{"strcspn",NULL},
#else
	{"strcspn",(void (*)())strcspn},
#endif
#if defined(strerror) || !defined(G__FUNCPOINTER)
	{"strerror",NULL},
#else
	{"strerror",(void (*)())strerror},
#endif
#if defined(strftime) || !defined(G__FUNCPOINTER)
	{"strftime",NULL},
#else
	{"strftime",(void (*)())strftime},
#endif
#if defined(strlen) || !defined(G__FUNCPOINTER)
	{"strlen",NULL},
#else
	{"strlen",(void (*)())strlen},
#endif
#if defined(strncat) || !defined(G__FUNCPOINTER)
	{"strncat",NULL},
#else
	{"strncat",(void (*)())strncat},
#endif
#if defined(strncmp) || !defined(G__FUNCPOINTER)
	{"strncmp",NULL},
#else
	{"strncmp",(void (*)())strncmp},
#endif
#if defined(strncpy) || !defined(G__FUNCPOINTER)
	{"strncpy",NULL},
#else
	{"strncpy",(void (*)())strncpy},
#endif
#if defined(strpbrk) || !defined(G__FUNCPOINTER)
	{"strpbrk",NULL},
#else
	{"strpbrk",(void (*)())strpbrk},
#endif
#if defined(strrchr) || !defined(G__FUNCPOINTER)
	{"strrchr",NULL},
#else
	{"strrchr",(void (*)())strrchr},
#endif
#if defined(strspn) || !defined(G__FUNCPOINTER)
	{"strspn",NULL},
#else
	{"strspn",(void (*)())strspn},
#endif
#if defined(strstr) || !defined(G__FUNCPOINTER)
	{"strstr",NULL},
#else
	{"strstr",(void (*)())strstr},
#endif
#if defined(strtod) || !defined(G__FUNCPOINTER)
	{"strtod",NULL},
#else
	{"strtod",(void (*)())strtod},
#endif
#if defined(strtok) || !defined(G__FUNCPOINTER)
	{"strtok",NULL},
#else
	{"strtok",(void (*)())strtok},
#endif
#if defined(strtol) || !defined(G__FUNCPOINTER)
	{"strtol",NULL},
#else
	{"strtol",(void (*)())strtol},
#endif
#if defined(strtoul) || !defined(G__FUNCPOINTER)
	{"strtoul",NULL},
#else
	{"strtoul",(void (*)())strtoul},
#endif
#if defined(strxfrm) || !defined(G__FUNCPOINTER)
	{"strxfrm",NULL},
#else
	{"strxfrm",(void (*)())strxfrm},
#endif
#if defined(system) || !defined(G__FUNCPOINTER)
	{"system",NULL},
#else
	{"system",(void (*)())system},
#endif
#ifndef G__NOMATHLIB
#if defined(tan) || !defined(G__FUNCPOINTER)
	{"tan",NULL},
#else
	{"tan",(void (*)())tan},
#endif
#if defined(tanh) || !defined(G__FUNCPOINTER)
	{"tanh",NULL},
#else
	{"tanh",(void (*)())tanh},
#endif
#endif/* G__NOMATHLIB */
#if defined(time) || !defined(G__FUNCPOINTER)
	{"time",NULL},
#else
	{"time",(void (*)())time},
#endif
#if defined(tmpfile) || !defined(G__FUNCPOINTER)
	{"tmpfile",NULL},
#else
	{"tmpfile",(void (*)())tmpfile},
#endif
#if defined(tmpnam) || !defined(G__FUNCPOINTER)
	{"tmpnam",NULL},
#else
#if ((__GNUC__>=3)||(__GNUC__>=2)&&(__GNUC_MINOR__>=96))&&(defined(__linux)||defined(__linux__))
	{"tmpnam",NULL},
#else
	{"tmpnam",(void (*)())tmpnam},
#endif
#endif
#if defined(tolower) || !defined(G__FUNCPOINTER)
	{"tolower",NULL},
#else
	{"tolower",(void (*)())tolower},
#endif
#if defined(toupper) || !defined(G__FUNCPOINTER)
	{"toupper",NULL},
#else
	{"toupper",(void (*)())toupper},
#endif
#if defined(ungetc) || !defined(G__FUNCPOINTER)
	{"ungetc",NULL},
#else
	{"ungetc",(void (*)())ungetc},
#endif
#if defined(wcstombs) || !defined(G__FUNCPOINTER)
	{"wcstombs",NULL},
#else
	{"wcstombs",(void (*)())wcstombs},
#endif
#if defined(wctomb) || !defined(G__FUNCPOINTER)
	{"wctomb",NULL},
#else
	{"wctomb",(void (*)())wctomb},
#endif
#if defined(fprintf) || !defined(G__FUNCPOINTER)
	{"fprintf",NULL},
#else
	{"fprintf",(void (*)())fprintf},
#endif
#if defined(printf) || !defined(G__FUNCPOINTER)
	{"printf",NULL},
#else
	{"printf",(void (*)())printf},
#endif
#if defined(sprintf) || !defined(G__FUNCPOINTER)
	{"sprintf",NULL},
#else
	{"sprintf",(void (*)())sprintf},
#endif
#if defined(fscanf) || !defined(G__FUNCPOINTER)
	{"fscan",NULL},
#else
	{"fscanf",(void (*)())fscanf},
#endif
#if defined(scanf) || !defined(G__FUNCPOINTER)
	{"scan",NULL},
#else
	{"scanf",(void (*)())scanf},
#endif
#if defined(sscanf) || !defined(G__FUNCPOINTER)
	{"sscan",NULL},
#else
	{"sscanf",(void (*)())sscanf},
#endif
#if defined(exit) || !defined(G__FUNCPOINTER)
	{"exit",NULL},
#else
	{"exit",(void (*)())exit},
#endif
#if defined(atexit) || !defined(G__FUNCPOINTER)
	{"atexit",NULL},
#else
	{"atexit",(void (*)())atexit},
#endif
#if defined(qsort) || !defined(G__FUNCPOINTER)
	{"qsort",NULL},
#else
	{"qsort",(void (*)())qsort},
#endif
#if defined(bsearch) || !defined(G__FUNCPOINTER)
	{"bsearch",NULL},
#else
	{"bsearch",(void (*)())bsearch},
#endif
#if defined(getopt) || !defined(G__FUNCPOINTER)
	{"getopt",NULL},
#else
	{"getopt",(void (*)())getopt},
#endif
	{"G__pause",NULL},
	{"G__loadfile",NULL},
	{"G__unloadfile",NULL},
	{"G__reloadfile",NULL},
	{"G__tracemode",NULL},
	{"G__gettracemode",NULL},
	{"G__debugmode",NULL},
	{"G__setbreakpoint",NULL},
	{"G__stepmode",NULL},
	{"G__getstepmode",NULL},
	{"G__optimizemode",NULL},
	{"G__getoptimizemode",NULL},
	/* {"G__breakline",NULL}, */
	{"G__split",NULL},
	{"G__readline",NULL},
	{"G__cmparray",NULL},
	{"G__setarray",NULL},
	{"G__graph",NULL},
	{"G__input",NULL},
	{"G__search_next_member",NULL},
	{"G__what_type",NULL},
	{"G__storeobject",NULL},
	{"G__scanobject",NULL},
	{"G__dumpobject",NULL},
	{"G__loadobject",NULL},
	{"G__lock_variable",NULL},
	{"G__unlock_variable",NULL},

	{"G__IsInMacro",NULL},
	{"G__getmakeinfo",NULL},
	{"G__set_atpause",NULL},
	{"G__set_aterror",NULL},
	{"G__set_smartunload",NULL},
	{"G__AddConsole",NULL},
	{"G__FreeConsole",NULL},
	{"G__typeid",NULL},
	{"G__lasterror_filename",NULL},
	{"G__lasterror_linenum",NULL},
	{"G__loadsystemfile",NULL},
	{"G__p2f2funcname",NULL},
	{"G__isinterpretedp2f",NULL},
	{"G__deleteglobal",NULL},
	{"G__deletevariable",NULL},
	{"G__optimizemode",NULL},
	{"G__getoptimizemode",NULL},
	{"G__clearerror",NULL},
	{"G__calc",NULL},
	{"G__exec_text",NULL},
	{"G__exec_tempfile",NULL},
#ifndef G__OLDIMPLEMENTATION1546
	{"G__load_text",NULL},
#endif
	{(char *)NULL,NULL }
};


/******************************************************
* G__compiled_func(result7,funcname,libp,hash)
*
*  Compiled user function execution filter
* Automatically generated by G__genfunc
*
*    G__genfunc [userfunc]
*
* Author : Masaharu Goto YHD R/D  22 Mar 1991
*
******************************************************/

extern long G__int();
extern double G__double();
extern char  *G__string();
extern struct G__input_file G__ifile;


int G__compiled_func(result7,funcname,libp,hash)
G__value *result7;
const char *funcname;
struct G__param *libp;
int hash;
{
#ifdef G__NO_STDLIBS
  return(0);
#endif
  G__CHECK(G__SECURE_STANDARDLIB,1,return(0));

  /******************************************************************
  * high priority routines 1
  ******************************************************************/

  if((hash==325)&&(strcmp(funcname,"cos")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("cos",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__letdouble(result7,'d',(double)cos(G__double(libp->para[0])));
    return(1);
  }

  if((hash==330)&&(strcmp(funcname,"sin")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("sin",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__letdouble(result7,'d',(double)sin(G__double(libp->para[0])));
    return(1);
  }

  if((hash==323)&&(strcmp(funcname,"tan")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("tan",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__letdouble(result7,'d',(double)tan(G__double(libp->para[0])));
    return(1);
  }

  if((hash==333)&&(strcmp(funcname,"exp")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("exp",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__CHECKDRANGE(0,-HUGE_VAL,709.0);
    G__letdouble(result7,'d',(double)exp(G__double(libp->para[0])));
    return(1);
  }

  if((hash==322)&&(strcmp(funcname,"log")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("log",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__CHECKDRANGE(0,0.0,HUGE_VAL);
    G__letdouble(result7,'d',(double)log(G__double(libp->para[0])));
    return(1);
  }

  if((hash==419)&&(strcmp(funcname,"log10")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("log10",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__CHECKDRANGE(0,0.0,HUGE_VAL);
    G__letdouble(result7,'d',(double)log10(G__double(libp->para[0])));
    return(1);
  }

  if((hash==412)&&(strcmp(funcname,"fabs")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("fabs",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__letdouble(result7,'d',(double)fabs(G__double(libp->para[0])));
    return(1);
  }

  if((hash==458)&&(strcmp(funcname,"sqrt")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("sqrt",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__letdouble(result7,'d',(double)sqrt(G__double(libp->para[0])));
    return(1);
  }

  if((hash==537)&&(strcmp(funcname,"fgets")==0)) {
#ifndef G__MASKERROR
    if(3!=libp->paran) {
      G__printerror("fgets",3,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'C',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKLRANGE(1,0,SHRT_MAX);
    G__CHECKNONULL(2,'E');
    G__letint(result7,'C',(long)fgets((char *)G__int(libp->para[0]),(int)G__int(libp->para[1]),(FILE *)G__int(libp->para[2])));
    return(1);
  }


  /******************************************************************
  * high priority routines 2
  ******************************************************************/

#ifndef G__NOMATHLIB
  if((hash==342)&&(strcmp(funcname,"pow")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("pow",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__letdouble(result7,'d',(double)pow(G__double(libp->para[0]),G__double(libp->para[1])));
    return(1);
  }

  if((hash==546)&&(strcmp(funcname,"floor")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("floor",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__letdouble(result7,'d',(double)floor(G__double(libp->para[0])));
    return(1);
  }

  if((hash==422)&&(strcmp(funcname,"fmod")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("fmod",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    if(0.0==G__double(libp->para[1])) {
      G__genericerror("Error: fmod() divided by zero");
      *result7=G__null;
      return(1);
    }
    G__letdouble(result7,'d',(double)fmod(G__double(libp->para[0]),G__double(libp->para[1])));
    return(1);
  }

  if((hash==422)&&(strcmp(funcname,"acos")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) { 
      G__printerror("acos",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__CHECKDRANGE(0,-1.0,1.0);
    G__letdouble(result7,'d',(double)acos(G__double(libp->para[0])));
    return(1);
  }

  if((hash==427)&&(strcmp(funcname,"asin")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) { 
      G__printerror("asin",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__CHECKDRANGE(0,-1.0,1.0);
    G__letdouble(result7,'d',(double)asin(G__double(libp->para[0])));
    return(1);
  }

  if((hash==420)&&(strcmp(funcname,"atan")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) { 
      G__printerror("atan",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__letdouble(result7,'d',(double)atan(G__double(libp->para[0])));
    return(1);
  }

  if((hash==470)&&(strcmp(funcname,"atan2")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) { 
      G__printerror("atan2",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__letdouble(result7,'d',(double)atan2(G__double(libp->para[0]),G__double(libp->para[1])));
    return(1);
  }

  if((hash==429)&&(strcmp(funcname,"cosh")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("cosh",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__CHECKDRANGE(0,-710.0,710.0);
    G__letdouble(result7,'d',(double)cosh(G__double(libp->para[0])));
    return(1);
  }

  if((hash==434)&&(strcmp(funcname,"sinh")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("sinh",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__letdouble(result7,'d',(double)sinh(G__double(libp->para[0])));
    return(1);
  }

  if((hash==427)&&(strcmp(funcname,"tanh")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("tanh",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__letdouble(result7,'d',(double)tanh(G__double(libp->para[0])));
    return(1);
  }

  if((hash==310)&&(strcmp(funcname,"abs")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) { 
      G__printerror("abs",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__letint(result7,'i',(long)abs((int)G__int(libp->para[0])));
    return(1);
  }
#endif /* G__NOMATHLIB */

  if((hash==426)&&(strcmp(funcname,"atof")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("atof",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__letdouble(result7,'d',(double)atof((char *)G__int(libp->para[0])));
    return(1);
  }

  if((hash==429)&&(strcmp(funcname,"atoi")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("atoi",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__letint(result7,'i',(long)atoi((char *)G__int(libp->para[0])));
    return(1);
  }

  if((hash==432)&&(strcmp(funcname,"atol")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("atol",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'l',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__letint(result7,'l',(long)atol((char *)G__int(libp->para[0])));
    return(1);
  }

  /******************************************************************
  * high priority routines 3
  ******************************************************************/


  if((hash==648)&&(strcmp(funcname,"fflush")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("fflush",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'E');
    G__letint(result7,'i',(long)fflush((FILE *)G__int(libp->para[0])));
    return(1);
  }

  if((hash==521)&&(strcmp(funcname,"fgetc")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("fgetc",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'E');
    G__letint(result7,'i',(long)fgetc((FILE *)G__int(libp->para[0])));
    return(1);
  }

  if((hash==418)&&(strcmp(funcname,"labs")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("labs",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'l',(long)0);
      return(1);
    }
    G__letint(result7,'l',(long)labs((long)G__int(libp->para[0])));
    return(1);
  }

  /******************************************************************
  * un-classified routines
  ******************************************************************/


  if((hash==742)&&(strcmp(funcname,"asctime")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) { 
      G__printerror("asctime",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'C',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'U'); /* not checking struct tag */
    G__letint(result7,'C',(long)asctime((struct tm*)G__int(libp->para[0])));
    return(1);
  }

  if((hash==622)&&(strcmp(funcname,"calloc")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("calloc",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'Y',(long)0);
      return(1);
    }
    G__CHECKLRANGE(0,0,INT_MAX);
    G__CHECKLRANGE(1,0,INT_MAX);
    G__letint(result7,'Y',(long)calloc((size_t)G__int(libp->para[0]),(size_t)G__int(libp->para[1])));
    return(1);
  }

  if((hash==413)&&(strcmp(funcname,"ceil")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("ceil",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__letdouble(result7,'d',(double)ceil(G__double(libp->para[0])));
    return(1);
  }


  if((hash==524)&&(strcmp(funcname,"clock")==0)) {
#ifndef G__MASKERROR
    if(0!=libp->paran) {
      G__printerror("clock",0,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'k',(long)0);
      return(1);
    }
    G__letint(result7,'k',(long)clock());
    return(1);
  }



  if((hash==530)&&(strcmp(funcname,"ctime")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("ctime",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'C',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'Y'); /* not checking type */
    G__letint(result7,'C',(long)ctime((time_t *)G__int(libp->para[0])));
    return(1);
  }

  if((hash==840)&&(strcmp(funcname,"difftime")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("difftime",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__letdouble(result7,'d',(double)difftime((time_t)G__int(libp->para[0]),(time_t)G__int(libp->para[1])));
    return(1);
  }

#if !defined(G__NONANSI) && !defined(G__SUNOS4)
  if((hash==323)&&(strcmp(funcname,"div")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("div",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'u',(long)0);
      result7->typenum = G__defined_typename("div_t");
      if (result7->typenum >= 0) {
         result7->tagnum = G__newtype.tagnum[result7->typenum];
      } else {
         result7->tagnum = -1;
      }
      return(1);
    }
    result7->typenum = G__defined_typename("div_t");
    result7->tagnum = G__newtype.tagnum[result7->typenum];
    result7->type = 'u';
    G__alloc_tempobject(result7->tagnum,result7->typenum);
    *(div_t*)G__p_tempbuf->obj.obj.i = div((int)G__int(libp->para[0])
					   ,(int)G__int(libp->para[1]));
    result7->obj.i = G__p_tempbuf->obj.obj.i;
    result7->ref = G__p_tempbuf->obj.obj.i;
    return(1);
  }
#endif


  if((hash==416)&&(strcmp(funcname,"feof")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("feof",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'E');
    G__letint(result7,'i',(long)feof((FILE *)G__int(libp->para[0])));
    return(1);
  }

  if((hash==656)&&(strcmp(funcname,"ferror")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("ferror",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'E');
    G__letint(result7,'i',(long)ferror((FILE *)G__int(libp->para[0])));
    return(1);
  }

  if((hash==760)&&(strcmp(funcname,"fgetpos")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("fgetpos",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'E');
    G__CHECKNONULL(1,'Y');
    G__letint(result7,'i',(long)fgetpos((FILE *)G__int(libp->para[0]),(fpos_t *)G__int(libp->para[1])));
    return(1);
  }

  if((hash==546)&&(strcmp(funcname,"fputc")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("fputc",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(1,'E');
    G__letint(result7,'i',(long)fputc((int)G__int(libp->para[0]),(FILE *)G__int(libp->para[1])));
    return(1);
  }

  if((hash==562)&&(strcmp(funcname,"fputs")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("fputs",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'E');
    G__letint(result7,'i',(long)fputs((char *)G__int(libp->para[0]),(FILE *)G__int(libp->para[1])));
    return(1);
  }

  if((hash==514)&&(strcmp(funcname,"fread")==0)) {
#ifndef G__MASKERROR
    if(4!=libp->paran) {
      G__printerror("fread",4,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'h',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'Y');
    G__CHECKNONULL(3,'E');
    G__letint(result7,'h',(long)fread((void *)G__int(libp->para[0]),(size_t)G__int(libp->para[1]),(size_t)G__int(libp->para[2]),(FILE *)G__int(libp->para[3])));
    return(1);
  }

  if((hash==418)&&(strcmp(funcname,"free")==0)) {
    G__CHECK(G__SECURE_MALLOC,1,return(1));
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("free",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      *result7=G__null;
      return(1);
    }
    /* G__CHECKNONULL(0,'Y'); */ /* no check, ANSI defined this as no effect */
#ifdef G__SECURITY
    if(G__security&G__SECURE_GARBAGECOLLECTION) {
      G__del_alloctable((void*)G__int(libp->para[0]));
    }
#endif
    free((void *)G__int(libp->para[0]));
    *result7=G__null;
    return(1);
  }

  if((hash==751)&&(strcmp(funcname,"freopen")==0)) {
#ifndef G__MASKERROR
    if(3!=libp->paran) {
      G__printerror("freopen",3,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'E',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'C');
    G__CHECKNONULL(2,'E');
    G__letint(result7,'E',(long)freopen((char *)G__int(libp->para[0]),(char *)G__int(libp->para[1]),(FILE *)G__int(libp->para[2])));
    return(1);
  }

  if((hash==549)&&(strcmp(funcname,"frexp")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("frexp",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__CHECKNONULL(1,'I');
    G__letdouble(result7,'d',(double)frexp(G__double(libp->para[0]),(int *)G__int(libp->para[1])));
    return(1);
  }

  if((hash==526)&&(strcmp(funcname,"fseek")==0)) {
#ifndef G__MASKERROR
    if(3!=libp->paran) {
      G__printerror("fseek",3,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'E');
    G__letint(result7,'i',(long)fseek((FILE *)G__int(libp->para[0]),(long)G__int(libp->para[1]),(int)G__int(libp->para[2])));
    return(1);
  }

  if((hash==772)&&(strcmp(funcname,"fsetpos")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("fsetpos",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'E');
    G__CHECKNONULL(1,'Y'); /* not checking type */
    G__letint(result7,'i',(long)fsetpos((FILE *)G__int(libp->para[0]),(fpos_t *)G__int(libp->para[1])));
    return(1);
  }

  if((hash==535)&&(strcmp(funcname,"ftell")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("ftell",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'l',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'E');
    G__letint(result7,'l',(long)ftell((FILE *)G__int(libp->para[0])));
    return(1);
  }

  if((hash==657)&&(strcmp(funcname,"fwrite")==0)) {
#ifndef G__MASKERROR
    if(4!=libp->paran) {
      G__printerror("fwrite",4,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'h',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'Y');
    G__CHECKNONULL(3,'E');
    G__letint(result7,'h',(long)fwrite((void *)G__int(libp->para[0]),(size_t)G__int(libp->para[1]),(size_t)G__int(libp->para[2]),(FILE *)G__int(libp->para[3])));
    return(1);
  }

  if((hash==419)&&(strcmp(funcname,"getc")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("getc",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'E');
    G__letint(result7,'i',(long)getc((FILE *)G__int(libp->para[0])));
    return(1);
  }

  if((hash==734)&&(strcmp(funcname,"getchar")==0)) {
#ifndef G__MASKERROR
    if(0!=libp->paran) {
      G__printerror("getchar",0,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__letint(result7,'i',(long)fgetc(G__intp_sin));
    return(1);
  }

  if((hash==649)&&(strcmp(funcname,"getenv")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("getenv",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'C',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__letint(result7,'C',(long)getenv((char *)G__int(libp->para[0])));
    return(1);
  }

  if((hash==435)&&(strcmp(funcname,"gets")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("gets",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'C',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__letint(result7,'C',(long)fgets((char *)G__int(libp->para[0]),G__ONELINE,G__intp_sin));
    return(1);
  }

  if((hash==643)&&(strcmp(funcname,"gmtime")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("gmtime",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'U',(long)0);
      result7->tagnum = G__defined_tagname("tm",0);
      return(1);
    }
    G__CHECKNONULL(0,'Y'); /* not checking type */
    G__letint(result7,'U',(long)gmtime((time_t *)G__int(libp->para[0])));
    result7->tagnum = G__defined_tagname("tm",0);
    return(1);
  }

  if((hash==761)&&(strcmp(funcname,"isalnum")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("isalnum",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__letint(result7,'i',(long)isalnum((int)G__int(libp->para[0])));
    return(1);
  }

  if((hash==738)&&(strcmp(funcname,"isalpha")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("isalpha",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__letint(result7,'i',(long)isalpha((int)G__int(libp->para[0])));
    return(1);
  }

  if((hash==767)&&(strcmp(funcname,"iscntrl")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("iscntrl",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__letint(result7,'i',(long)iscntrl((int)G__int(libp->para[0])));
    return(1);
  }

  if((hash==749)&&(strcmp(funcname,"isdigit")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("isdigit",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__letint(result7,'i',(long)isdigit((int)G__int(libp->para[0])));
    return(1);
  }

  if((hash==750)&&(strcmp(funcname,"isgraph")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("isgraph",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__letint(result7,'i',(long)isgraph((int)G__int(libp->para[0])));
    return(1);
  }

  if((hash==773)&&(strcmp(funcname,"islower")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("islower",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__letint(result7,'i',(long)islower((int)G__int(libp->para[0])));
    return(1);
  }

  if((hash==777)&&(strcmp(funcname,"isprint")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("isprint",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__letint(result7,'i',(long)isprint((int)G__int(libp->para[0])));
    return(1);
  }

  if((hash==774)&&(strcmp(funcname,"ispunct")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("ispunct",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__letint(result7,'i',(long)ispunct((int)G__int(libp->para[0])));
    return(1);
  }

  if((hash==744)&&(strcmp(funcname,"isspace")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("isspace",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__letint(result7,'i',(long)isspace((int)G__int(libp->para[0])));
    return(1);
  }

  if((hash==776)&&(strcmp(funcname,"isupper")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("isupper",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__letint(result7,'i',(long)isupper((int)G__int(libp->para[0])));
    return(1);
  }

  if((hash==869)&&(strcmp(funcname,"isxdigit")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("isxdigit",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__letint(result7,'i',(long)isxdigit((int)G__int(libp->para[0])));
    return(1);
  }


  if((hash==541)&&(strcmp(funcname,"ldexp")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("ldexp",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__letdouble(result7,'d',(double)ldexp(G__double(libp->para[0]),(int)G__int(libp->para[1])));
    return(1);
  }

#if !defined(G__NONANSI) && !defined(G__SUNOS4)
  if((hash==431)&&(strcmp(funcname,"ldiv")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("ldiv",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'u',(long)0);
      result7->typenum = G__defined_typename("ldiv_t");
      result7->tagnum = G__newtype.tagnum[result7->typenum];
      return(1);
    }
    result7->typenum = G__defined_typename("ldiv_t");
    result7->tagnum = G__newtype.tagnum[result7->typenum];
    result7->type = 'u';
    G__alloc_tempobject(result7->tagnum,result7->typenum);
    *(ldiv_t*)G__p_tempbuf->obj.obj.i = ldiv((long)G__int(libp->para[0])
					   ,(long)G__int(libp->para[1]));
    result7->obj.i = G__p_tempbuf->obj.obj.i;
    result7->ref = G__p_tempbuf->obj.obj.i;
    return(1);
  }
#endif

  if((hash==1062)&&(strcmp(funcname,"localeconv")==0)) {
#ifndef G__MASKERROR
    if(0!=libp->paran) {
      G__printerror("localeconv",0,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'U',(long)0);
      result7->tagnum = G__defined_tagname("lconv",0);
      return(1);
    }
    G__letint(result7,'U',(long)localeconv());
    result7->tagnum = G__defined_tagname("lconv",0);
    return(1);
  }

  if((hash==954)&&(strcmp(funcname,"localtime")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("localtime",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'U',(long)0);
      result7->tagnum = G__defined_tagname("tm",0);
      return(1);
    }
    G__CHECKNONULL(0,'Y'); /* not checking type */
    G__letint(result7,'U',(long)localtime((time_t*)G__int(libp->para[0])));
    result7->tagnum = G__defined_tagname("tm",0);
    return(1);
  }


  if((hash==632)&&(strcmp(funcname,"malloc")==0)) {
    *result7=G__null;
    G__CHECK(G__SECURE_CAST2P,1,return(1));
    G__CHECK(G__SECURE_MALLOC,1,return(1));
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("malloc",1,libp->paran);
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'Y',(long)0);
      return(1);
    }
    G__CHECKLRANGE(0,0,INT_MAX);
    G__letint(result7,'Y',(long)malloc((size_t)G__int(libp->para[0])));
#ifdef G__SECURITY
    if(G__security&G__SECURE_GARBAGECOLLECTION) {
      G__add_alloctable((void*)result7->obj.i,'y',-1);
    }
#endif
    return(1);
  }

  if((hash==526)&&(strcmp(funcname,"mblen")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("mblen",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__letint(result7,'i',(long)mblen((char *)G__int(libp->para[0]),(size_t)G__int(libp->para[1])));
    return(1);
  }

  if((hash==882)&&(strcmp(funcname,"mbstowcs")==0)) {
#ifndef G__MASKERROR
    if(3!=libp->paran) {
      G__printerror("mbstowcs",3,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'h',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'Y'); /* not checking type */
    G__CHECKNONULL(1,'C');
    G__letint(result7,'h',(long)mbstowcs((wchar_t *)G__int(libp->para[0]),(char *)G__int(libp->para[1]),(size_t)G__int(libp->para[2])));
    return(1);
  }

  if((hash==652)&&(strcmp(funcname,"mbtowc")==0)) {
#ifndef G__MASKERROR
    if(3!=libp->paran) {
      G__printerror("mbtowc",3,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'Y');
    G__CHECKNONULL(1,'C');
    G__letint(result7,'i',(long)mbtowc((wchar_t *)G__int(libp->para[0]),(char *)G__int(libp->para[1]),(size_t)G__int(libp->para[2])));
    return(1);
  }

  if((hash==636)&&(strcmp(funcname,"memchr")==0)) {
#ifndef G__MASKERROR
    if(3!=libp->paran) {
      G__printerror("memchr",3,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'Y',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'Y');
    G__letint(result7,'Y',(long)memchr((void *)G__int(libp->para[0]),(int)G__int(libp->para[1]),(size_t)G__int(libp->para[2])));
    return(1);
  }

  if((hash==639)&&(strcmp(funcname,"memcmp")==0)) {
#ifndef G__MASKERROR
    if(3!=libp->paran) {
      G__printerror("memcmp",3,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'Y');
    G__CHECKNONULL(1,'Y');
    G__letint(result7,'i',(long)memcmp((void *)G__int(libp->para[0]),(void *)G__int(libp->para[1]),(size_t)G__int(libp->para[2])));
    return(1);
  }

  if((hash==651)&&(strcmp(funcname,"memcpy")==0)) {
#ifndef G__MASKERROR
    if(3!=libp->paran) {
      G__printerror("memcpy",3,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'Y',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'Y');
    G__CHECKNONULL(1,'Y');
    G__letint(result7,'Y',(long)memcpy((void *)G__int(libp->para[0]),(void *)G__int(libp->para[1]),(size_t)G__int(libp->para[2])));
    return(1);
  }

  if((hash==758)&&(strcmp(funcname,"memmove")==0)) {
#ifndef G__MASKERROR
    if(3!=libp->paran) {
      G__printerror("memmove",3,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'Y',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'Y');
    G__CHECKNONULL(1,'Y');
    G__letint(result7,'Y',(long)memmove((void *)G__int(libp->para[0]),(void *)G__int(libp->para[1]),(size_t)G__int(libp->para[2])));
    return(1);
  }

  if((hash==651)&&(strcmp(funcname,"memset")==0)) {
#ifndef G__MASKERROR
    if(3!=libp->paran) {
      G__printerror("memset",3,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'Y',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'Y');
    G__letint(result7,'Y',(long)memset((void *)G__int(libp->para[0]),(int)G__int(libp->para[1]),(size_t)G__int(libp->para[2])));
    return(1);
  }

  if((hash==647)&&(strcmp(funcname,"mktime")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("mktime",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      result7->typenum = G__defined_typename("time_t");
      result7->tagnum = G__newtype.tagnum[result7->typenum];
      result7->type = G__newtype.type[result7->typenum];
      result7->obj.i = 0;
      result7->ref = 0;
      return(1);
    }
    G__CHECKNONULL(0,'U'); /* not checking struct tag */
    G__letint(result7,'l',(long)mktime((struct tm *)G__int(libp->para[0])));

    result7->typenum = G__defined_typename("time_t");
    result7->tagnum = G__newtype.tagnum[result7->typenum];
    result7->type = G__newtype.type[result7->typenum];
    result7->ref = 0;
    return(1);
  }

  if((hash==422)&&(strcmp(funcname,"modf")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("modf",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__letdouble(result7,'d',(double)modf(G__double(libp->para[0]),(double *)G__int(libp->para[1])));
    return(1);
  }

  if((hash==666)&&(strcmp(funcname,"perror")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("perror",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      *result7=G__null;
      return(1);
    }
    perror((char *)G__int(libp->para[0]));
    *result7=G__null;
    return(1);
  }


  if((hash==444)&&(strcmp(funcname,"putc")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("putc",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(1,'E');
    G__letint(result7,'i',(long)putc((int)G__int(libp->para[0]),(FILE *)G__int(libp->para[1])));
    return(1);
  }

  if((hash==759)&&(strcmp(funcname,"putchar")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("putchar",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__letint(result7,'i'
	      ,(long)fputc((int)G__int(libp->para[0]),G__intp_sout));
    return(1);
  }

  if((hash==460)&&(strcmp(funcname,"puts")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("puts",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__letint(result7,'i',(long)fputs((char *)G__int(libp->para[0])
				      ,(FILE *)G__intp_sout));
    fputc('\n',(FILE*)G__intp_sout);
    return(1);
  }

  if((hash==532)&&(strcmp(funcname,"raise")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("raise",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__letint(result7,'i',(long)raise((int)G__int(libp->para[0])));
    return(1);
  }

  if((hash==421)&&(strcmp(funcname,"rand")==0)) {
#ifndef G__MASKERROR
    if(0!=libp->paran) {
      G__printerror("rand",0,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__letint(result7,'i',(long)rand()); // Explicit user request, this can not be avoided.
    return(1);
  }

  if((hash==738)&&(strcmp(funcname,"realloc")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("realloc",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'Y',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'Y');
    G__letint(result7,'Y',(long)realloc((void *)G__int(libp->para[0]),(size_t)G__int(libp->para[1])));
    return(1);
  }

  if((hash==654)&&(strcmp(funcname,"remove")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("remove",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__letint(result7,'i',(long)remove((char *)G__int(libp->para[0])));
    return(1);
  }

  if((hash==632)&&(strcmp(funcname,"rename")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("rename",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'C');
    G__letint(result7,'i',(long)rename((char *)G__int(libp->para[0]),(char *)G__int(libp->para[1])));
    return(1);
  }

  if((hash==649)&&(strcmp(funcname,"rewind")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("rewind",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      *result7=G__null;
      return(1);
    }
    G__CHECKNONULL(0,'E');
    rewind((FILE *)G__int(libp->para[0]));
    *result7=G__null;
    return(1);
  }

  if((hash==649)&&(strcmp(funcname,"setbuf")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("setbuf",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      *result7=G__null;
      return(1);
    }
    G__CHECKNONULL(0,'E');
    G__CHECKNONULL(0,'C');
    setbuf((FILE *)G__int(libp->para[0]),(char *)G__int(libp->para[1]));
    *result7=G__null;
    return(1);
  }

  if((hash==956)&&(strcmp(funcname,"setlocale")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("setlocale",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      *result7=G__null;
      return(1);
    }
    G__CHECKNONULL(1,'C');
    G__letint(result7,'C',(long)setlocale((int)G__int(libp->para[0]),(char*)G__int(libp->para[1])));
    return(1);
  }

  if((hash==767)&&(strcmp(funcname,"setvbuf")==0)) {
#ifndef G__MASKERROR
    if(4!=libp->paran) {
      G__printerror("setvbuf",4,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'E');
    G__CHECKNONULL(1,'C');
    G__letint(result7,'i',(long)setvbuf((FILE *)G__int(libp->para[0]),(char *)G__int(libp->para[1]),(int)G__int(libp->para[2]),(size_t)G__int(libp->para[3])));
    return(1);
  }

  if((hash==536)&&(strcmp(funcname,"srand")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("srand",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      *result7=G__null;
      return(1);
    }
    srand((unsigned int)G__int(libp->para[0]));
    *result7=G__null;
    return(1);
  }

  if((hash==657)&&(strcmp(funcname,"strcat")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("strcat",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'C',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'C');
    {
       char *dst = (char*)G__int(libp->para[0]);
       char *src = (char*)G__int(libp->para[1]);
       char* res = strcat(dst,src);
       G__letint(result7,'C',(long)res);
    }
    return(1);
  }

  if((hash==662)&&(strcmp(funcname,"strchr")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("strchr",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'C',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__letint(result7,'C',(long)strchr((char *)G__int(libp->para[0]),(int)G__int(libp->para[1])));
    return(1);
  }

  if((hash==665)&&(strcmp(funcname,"strcmp")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("strcmp",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'C');
    G__letint(result7,'i',(long)strcmp((char *)G__int(libp->para[0]),(char *)G__int(libp->para[1])));
    return(1);
  }

  if((hash==771)&&(strcmp(funcname,"strcoll")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("strcoll",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'C');
    G__letint(result7,'i',(long)strcoll((char *)G__int(libp->para[0]),(char *)G__int(libp->para[1])));
    return(1);
  }

  if((hash==677)&&(strcmp(funcname,"strcpy")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("strcpy",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'C',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'C');
    G__letint(result7,'C',(long)strcpy((char *)G__int(libp->para[0]),(char *)G__int(libp->para[1])));
    return(1);
  }

  if((hash==781)&&(strcmp(funcname,"strcspn")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("strcspn",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'h',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'C');
    G__letint(result7,'h',(long)strcspn((char *)G__int(libp->para[0]),(char *)G__int(libp->para[1])));
    return(1);
  }

  if((hash==899)&&(strcmp(funcname,"strerror")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("strerror",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'C',(long)0);
      return(1);
    }
    G__letint(result7,'C',(long)strerror((int)G__int(libp->para[0])));
    return(1);
  }

  if((hash==878)&&(strcmp(funcname,"strftime")==0)) {
#ifndef G__MASKERROR
    if(4!=libp->paran) {
      G__printerror("strftime",4,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'l',(long)0);
      result7->typenum = G__defined_typename("size_t");
      result7->tagnum = G__newtype.tagnum[result7->typenum];
      result7->type = G__newtype.type[result7->typenum];
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(2,'C');
    G__CHECKNONULL(3,'U'); /* not checking struct tag */
    G__letint(result7,'l',(long)strftime((char*)G__int(libp->para[0])
					 ,(size_t)G__int(libp->para[1])
					 ,(char*)G__int(libp->para[2])
					 ,(struct tm*)G__int(libp->para[3])));
    result7->typenum = G__defined_typename("size_t");
    result7->tagnum = G__newtype.tagnum[result7->typenum];
    result7->type = G__newtype.type[result7->typenum];
    return(1);
  }

  if((hash==664)&&(strcmp(funcname,"strlen")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("strlen",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'h',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__letint(result7,'h',(long)strlen((char *)G__int(libp->para[0])));
    return(1);
  }

  if((hash==767)&&(strcmp(funcname,"strncat")==0)) {
#ifndef G__MASKERROR
    if(3!=libp->paran) {
      G__printerror("strncat",3,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'C',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'C');
    G__letint(result7,'C',(long)strncat((char *)G__int(libp->para[0]),(char *)G__int(libp->para[1]),(size_t)G__int(libp->para[2])));
    return(1);
  }

  if((hash==775)&&(strcmp(funcname,"strncmp")==0)) {
#ifndef G__MASKERROR
    if(3!=libp->paran) {
      G__printerror("strncmp",3,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'C');
    G__letint(result7,'i',(long)strncmp((char *)G__int(libp->para[0]),(char *)G__int(libp->para[1]),(size_t)G__int(libp->para[2])));
    return(1);
  }

  if((hash==787)&&(strcmp(funcname,"strncpy")==0)) {
#ifndef G__MASKERROR
    if(3!=libp->paran) {
      G__printerror("strncpy",3,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'C',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'C');
    G__letint(result7,'C',(long)strncpy((char *)G__int(libp->para[0]),(char *)G__int(libp->para[1]),(size_t)G__int(libp->para[2])));
    return(1);
  }

  if((hash==776)&&(strcmp(funcname,"strpbrk")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("strpbrk",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'C',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'C');
    G__letint(result7,'C',(long)strpbrk((char *)G__int(libp->para[0]),(char *)G__int(libp->para[1])));
    return(1);
  }

  if((hash==776)&&(strcmp(funcname,"strrchr")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("strrchr",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'C',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__letint(result7,'C',(long)strrchr((char *)G__int(libp->para[0]),(int)G__int(libp->para[1])));
    return(1);
  }

  if((hash==682)&&(strcmp(funcname,"strspn")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("strspn",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'h',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'C');
    G__letint(result7,'h',(long)strspn((char *)G__int(libp->para[0]),(char *)G__int(libp->para[1])));
    return(1);
  }

  if((hash==690)&&(strcmp(funcname,"strstr")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("strstr",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'C',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'C');
    G__letint(result7,'C',(long)strstr((char *)G__int(libp->para[0]),(char *)G__int(libp->para[1])));
    return(1);
  }

  if((hash==672)&&(strcmp(funcname,"strtod")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("strtod",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letdouble(result7,'d',(double)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKTYPE(1,'C','i');
    G__letdouble(result7,'d',(double)strtod((char *)G__int(libp->para[0]),(char **)G__int(libp->para[1])));
    return(1);
  }

  if((hash==679)&&(strcmp(funcname,"strtok")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("strtok",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'C',(long)0);
      return(1);
    }
    G__CHECKTYPE(0,'C','i');
    G__CHECKNONULL(1,'C');
    G__letint(result7,'C',(long)strtok((char *)G__int(libp->para[0]),(char *)G__int(libp->para[1])));
    return(1);
  }

  if((hash==680)&&(strcmp(funcname,"strtol")==0)) {
#ifndef G__MASKERROR
    if(3!=libp->paran) {
      G__printerror("strtol",3,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'l',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKTYPE(1,'C','i');
    G__CHECKLRANGE(2,0,36);
    G__letint(result7,'l',(long)strtol((char *)G__int(libp->para[0]),(char **)G__int(libp->para[1]),(int)G__int(libp->para[2])));
    return(1);
  }

  if((hash==797)&&(strcmp(funcname,"strtoul")==0)) {
#ifndef G__MASKERROR
    if(3!=libp->paran) {
      G__printerror("strtoul",3,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'k',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKTYPE(1,'C','i');
    G__CHECKLRANGE(2,0,36);
    G__letint(result7,'k',(long)strtoul((char *)G__int(libp->para[0]),(char **)G__int(libp->para[1]),(int)G__int(libp->para[2])));
    return(1);
  }

  if((hash==790)&&(strcmp(funcname,"strxfrm")==0)) {
#ifndef G__MASKERROR
    if(3!=libp->paran) {
      G__printerror("strxfrm",3,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'h',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'C');
    G__letint(result7,'h',(long)strxfrm((char *)G__int(libp->para[0]),(char *)G__int(libp->para[1]),(size_t)G__int(libp->para[2])));
    return(1);
  }

  if((hash==677)&&(strcmp(funcname,"system")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("system",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKTYPE(0,'C','i');
    G__letint(result7,'i',(long)system((char *)G__int(libp->para[0])));
    return(1);
  }

  if((hash==431)&&(strcmp(funcname,"time")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("time",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'l',(long)0);
      return(1);
    }
    /* G__CHECKNONULL(0,'Y'); */ /* tp can be null due to ANSI */
    G__letint(result7,'l',(long)time((time_t *)G__int(libp->para[0])));
    return(1);
  }

  if((hash==753)&&(strcmp(funcname,"tmpfile")==0)) {
#ifndef G__MASKERROR
    if(0!=libp->paran) {
      G__printerror("tmpfile",0,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'E',(long)0);
      return(1);
    }
    G__letint(result7,'E',(long)tmpfile()); // This is an explicit user request/call, this can not be avoided.
    return(1);
  }

  if((hash==653)&&(strcmp(funcname,"tmpnam")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("tmpnam",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'C',(long)0);
      return(1);
    }
    G__CHECKTYPE(0,'C','i');
    G__CHECKNONULL(0,'C');
#if ((__GNUC__>=3)||(__GNUC__>=2)&&(__GNUC_MINOR__>=96))&&(defined(__linux)||defined(__linux__))
    {
      char *p = (char*)G__int(libp->para[0]);
#ifdef P_tmpdir
      sprintf(p,"%s/XXXXXX",P_tmpdir);
#else
      sprintf(p,"/tmp/XXXXXX");
#endif
      G__letint(result7,'C',(long)mkstemp(p));
    }
#else
    G__letint(result7,'C',(long)tmpnam((char *)G__int(libp->para[0])));
#endif
    return(1);
  }

  if((hash==780)&&(strcmp(funcname,"tolower")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("tolower",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__letint(result7,'i',(long)tolower((int)G__int(libp->para[0])));
    return(1);
  }

  if((hash==783)&&(strcmp(funcname,"toupper")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("toupper",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__letint(result7,'i',(long)toupper((int)G__int(libp->para[0])));
    return(1);
  }

  if((hash==646)&&(strcmp(funcname,"ungetc")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("ungetc",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(1,'E');
    G__letint(result7,'i',(long)ungetc((int)G__int(libp->para[0]),(FILE *)G__int(libp->para[1])));
    return(1);
  }

  if((hash==882)&&(strcmp(funcname,"wcstombs")==0)) {
#ifndef G__MASKERROR
    if(3!=libp->paran) {
      G__printerror("wcstombs",3,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'h',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'Y'); /* not checking type */
    G__letint(result7,'h',(long)wcstombs((char *)G__int(libp->para[0]),(wchar_t *)G__int(libp->para[1]),(size_t)G__int(libp->para[2])));
    return(1);
  }

  if((hash==652)&&(strcmp(funcname,"wctomb")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("wctomb",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__letint(result7,'i',(long)wctomb((char *)G__int(libp->para[0]),(wchar_t)G__int(libp->para[1])));
    return(1);
  }

  /******************************************************************
  * mid priority routines
  ******************************************************************/


  /******************************************************************
  * low priority routines
  ******************************************************************/

  if((hash==536)&&(strcmp(funcname,"fopen")==0)) {
#ifndef G__MASKERROR
    if(2!=libp->paran) {
      G__printerror("fopen",2,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'E',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'C');
    G__letint(result7,'E',(long)fopen((char *)G__int(libp->para[0]),(char *)G__int(libp->para[1])));
#ifdef G__SECURITY
    if(G__security&G__SECURE_GARBAGECOLLECTION) {
      G__add_alloctable((void*)result7->obj.i,'E',-1);
    }
#endif
    return(1);
  }

  if((hash==636)&&(strcmp(funcname,"fclose")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("fclose",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      G__letint(result7,'i',(long)0);
      return(1);
    }
    G__CHECKNONULL(0,'E');
#ifdef G__SECURITY
    if(G__security&G__SECURE_GARBAGECOLLECTION) {
      G__del_alloctable((void*)G__int(libp->para[0]));
    }
#endif
    G__letint(result7,'i',(long)fclose((FILE *)G__int(libp->para[0])));
    return(1);
  }

  if((hash==848)&&(strcmp(funcname,"clearerr")==0)) {
#ifndef G__MASKERROR
    if(1!=libp->paran) {
      G__printerror("clearerr",1,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      *result7=G__null;
      return(1);
    }
    G__CHECKNONULL(0,'E');
    clearerr((FILE *)G__int(libp->para[0]));
    *result7=G__null;
    return(1);
  }

  if((hash==536)&&(strcmp(funcname,"abort")==0)) {
#ifndef G__MASKERROR
    if(0!=libp->paran) {
      G__printerror("abort",0,libp->paran);
      *result7=G__null;
      return(1);
    }
#endif
    if(G__no_exec_compile) {
      *result7=G__null;
      return(1);
    }
    abort();
    *result7=G__null;
    return(1);
  }


  return(0);
}

void G__list_sut(fp) FILE *fp; {
   fprintf(fp,"/* List of sut----------------------------------------------------------------*/\n");
   fprintf(fp,"   void abort();\n");
   fprintf(fp,"   int abs(int n);\n");
#ifndef G__NOMATHLIB
   fprintf(fp,"   double acos(double arg);\n");
   fprintf(fp,"   double asin(double arg);\n");
#endif /* G__NOMATHLIB */
   fprintf(fp,"   char* asctime(struct tm* timestruct);\n");
#ifndef G__NOMATHLIB
   fprintf(fp,"   double atan(double arg);\n");
   fprintf(fp,"   double atan2(double num,double den);\n");
#endif /* G__NOMATHLIB */
   fprintf(fp,"   double atof(char *string);\n");
   fprintf(fp,"   int atoi(char *string);\n");
   fprintf(fp,"   long atol(char *string);\n");
   fprintf(fp,"   void *calloc(size_t count,size_t size);\n");
   fprintf(fp,"   double ceil(double z);\n");
   fprintf(fp,"   void clearerr(FILE *fp);\n");
   fprintf(fp,"   clock_t clock();\n");
#ifndef G__NOMATHLIB
   fprintf(fp,"   double cos(double radian);\n");
   fprintf(fp,"   double cosh(double value);\n");
#endif /* G__NOMATHLIB */
   fprintf(fp,"   char *ctime(time_t *timeptr);\n");
   fprintf(fp,"   double difftime(time_t newtime,time_t oldtime);\n");
#if !defined(G__NONANSI) && !defined(G__SUNOS4)
   fprintf(fp,"   div_t div(int numerator,int denominator);\n");
#endif
#ifndef G__NOMATHLIB
   fprintf(fp,"   double exp(double z);\n");
   fprintf(fp,"   double fabs(double z);\n");
#endif /* G__NOMATHLIB */
   fprintf(fp,"   int fclose(FILE *fp);\n");
   fprintf(fp,"   int feof(FILE *fp);\n");
   fprintf(fp,"   int ferror(FILE *fp);\n");
   fprintf(fp,"   int fflush(FILE *fp);\n");
   fprintf(fp,"   int fgetc(FILE *fp);\n");
   fprintf(fp,"   int fgetpos(FILE *fp,fpos_t *position);\n");
   fprintf(fp,"   char *fgets(char *string,int n,FILE *fp);\n");
#ifndef G__NOMATHLIB
   fprintf(fp,"   double floor(double z);\n");
   fprintf(fp,"   double fmod(double number,double divisor);\n");
#endif /* G__NOMATHLIB */
   fprintf(fp,"   FILE *fopen(char *file,char *mode);\n");
   fprintf(fp,"   int fputc(int character,FILE *fp);\n");
   fprintf(fp,"   int fputs(char *string,FILE *fp);\n");
   fprintf(fp,"   size_t fread(void *buffer,size_t size,size_t n,FILE *fp);\n");
   fprintf(fp,"   void free(void *ptr);\n");
   fprintf(fp,"   FILE *freopen(char *file,char *mode,FILE *fp);\n");
   fprintf(fp,"   double frexp(double real,int *exp1);\n");
   fprintf(fp,"   int fseek(FILE *fp,long offset,int whence);\n");
   fprintf(fp,"   int fsetpos(FILE *fp,fpos_t *position);\n");
   fprintf(fp,"   long ftell(FILE *fp);\n");
   fprintf(fp,"   size_t fwrite(void *buffer,size_t size,size_t n,FILE *fp);\n");
   fprintf(fp,"   int getc(FILE *fp);\n");
   fprintf(fp,"   int getchar();\n");
   fprintf(fp,"   char *getenv(char *variable);\n");
   fprintf(fp,"   char *gets(char *buffer);\n");
   fprintf(fp,"   struct tm* gmtime(time_t *caltime);\n");
   fprintf(fp,"   int isalnum(int c);\n");
   fprintf(fp,"   int isalpha(int c);\n");
   fprintf(fp,"   int iscntrl(int c);\n");
   fprintf(fp,"   int isdigit(int c);\n");
   fprintf(fp,"   int isgraph(int c);\n");
   fprintf(fp,"   int islower(int c);\n");
   fprintf(fp,"   int isprint(int c);\n");
   fprintf(fp,"   int ispunct(int c);\n");
   fprintf(fp,"   int isspace(int c);\n");
   fprintf(fp,"   int isupper(int c);\n");
   fprintf(fp,"   int isxdigit(int c);\n");
   fprintf(fp,"   long labs(long n);\n");
#ifndef G__NOMATHLIB
   fprintf(fp,"   double ldexp(double number,int n);\n");
#endif /* G__NOMATHLIB */
#if !defined(G__NONANSI) && !defined(G__SUNOS4)
   fprintf(fp,"   ldiv_t ldiv(long numerator,long denominator);\n");
#endif
   fprintf(fp,"   struct lconv* localeconv();\n");
   fprintf(fp,"   struct tm* localtime(time_t* timeptr);\n");
#ifndef G__NOMATHLIB
   fprintf(fp,"   double log(double z);\n");
   fprintf(fp,"   double log10(double z);\n");
#endif /* G__NOMATHLIB */
   fprintf(fp,"   void *malloc(size_t size);\n");
   fprintf(fp,"   int mblen(char *address,size_t number);\n");
   fprintf(fp,"   size_t mbstowcs(wchar_t *widechar,char *multibyte,size_t number);\n");
   fprintf(fp,"   int mbtowc(wchar_t *charptr,char *address,size_t number);\n");
   fprintf(fp,"   void *memchr(void *region,int character,size_t n);\n");
   fprintf(fp,"   int memcmp(void *region1,void *region2,size_t count);\n");
   fprintf(fp,"   void *memcpy(void *region1,void *region2,size_t n);\n");
   fprintf(fp,"   void *memmove(void *region1,void *region2,size_t count);\n");
   fprintf(fp,"   void *memset(void *buffer,int character,size_t n);\n");
   fprintf(fp,"   time_t mktime(struct tm *timeptr);\n");
#ifndef G__NOMATHLIB
   fprintf(fp,"   double modf(double real,double *ip);\n");
#endif /* G__NOMATHLIB */
   fprintf(fp,"   void perror(char *string);\n");
   fprintf(fp,"   double pow(double z,double x);\n");
   fprintf(fp,"   int putc(int character,FILE *fp);\n");
   fprintf(fp,"   int putchar(int character);\n");
   fprintf(fp,"   int puts(char *string);\n");
   fprintf(fp,"   int raise(int signal);\n");
   fprintf(fp,"   int rand();\n");
   fprintf(fp,"   void *realloc(void *ptr,size_t size);\n");
   fprintf(fp,"   int remove(char *filename);\n");
   fprintf(fp,"   int rename(char *old,char *new);\n");
   fprintf(fp,"   void rewind(FILE *fp);\n");
   fprintf(fp,"   void setbuf(FILE *fp,char *buffer);\n");
   fprintf(fp,"   char* setlocale(int position,char *locale);\n");
   fprintf(fp,"   int setvbuf(FILE *fp,char *buffer,int mode,size_t size);\n");
#ifndef G__NOMATHLIB
   fprintf(fp,"   double sin(double radian);\n");
   fprintf(fp,"   double sinh(double value);\n");
   fprintf(fp,"   double sqrt(double z);\n");
#endif /* G__NOMATHLIB */
   fprintf(fp,"   void srand(unsigned int seed);\n");
   fprintf(fp,"   char *strcat(char *string1,char *string2);\n");
   fprintf(fp,"   char *strchr(char *string,int character);\n");
   fprintf(fp,"   int strcmp(char *string1,char *string2);\n");
   fprintf(fp,"   int strcoll(char *string1,char *string2);\n");
   fprintf(fp,"   char *strcpy(char *string1,char *string2);\n");
   fprintf(fp,"   size_t strcspn(char *string1,char *string2);\n");
   fprintf(fp,"   char *strerror(int error);\n");
   fprintf(fp,"   size_t strftime(char *string,size_t maximum,char *format,struct tm*brokentime);\n");
   fprintf(fp,"   size_t strlen(char *string);\n");
   fprintf(fp,"   char *strncat(char *string1,char *string2,size_t n);\n");
   fprintf(fp,"   int strncmp(char *string1,char *string2,size_t n);\n");
   fprintf(fp,"   char *strncpy(char *string1,char *string2,size_t n);\n");
   fprintf(fp,"   char *strpbrk(char *string1,char *string2);\n");
   fprintf(fp,"   char *strrchr(char *string,int character);\n");
   fprintf(fp,"   size_t strspn(char *string1,char *string2);\n");
   fprintf(fp,"   char *strstr(char *string1,char *string2);\n");
   fprintf(fp,"   double strtod(char *string,char **tailptr);\n");
   fprintf(fp,"   char *strtok(char *string1,char *string2);\n");
   fprintf(fp,"   long strtol(char *sprt,char **tailptr,int base);\n");
   fprintf(fp,"   unsigned long strtoul(char *sprt,char **tailptr,int base);\n");
   fprintf(fp,"   size_t strxfrm(char *string1,char *string2,size_t n);\n");
   fprintf(fp,"   int system(char *program);\n");
#ifndef G__NOMATHLIB
   fprintf(fp,"   double tan(double radian);\n");
   fprintf(fp,"   double tanh(double value);\n");
#endif /* G__NOMATHLIB */
   fprintf(fp,"   time_t time(time_t *tp);\n");
   fprintf(fp,"   FILE *tmpfile();\n");
   fprintf(fp,"   char *tmpnam(char *name);\n");
   fprintf(fp,"   int tolower(int c);\n");
   fprintf(fp,"   int toupper(int c);\n");
   fprintf(fp,"   int ungetc(int character,FILE *fp);\n");
   fprintf(fp,"   size_t wcstombs(char *multibyte,wchar_t *widechar,size_t number);\n");
   fprintf(fp,"   int wctomb(char *string,wchar_t widecharacter);\n");
   fprintf(fp,"   int fprintf(FILE *fp,char *format,arglist,...);\n");
   fprintf(fp,"   int printf(char *format,arglist,...);\n");
   fprintf(fp,"   int sprintf(char *string,char *format,arglist,...);\n");
   fprintf(fp,"   int fscanf(FILE *fp,char *format,arglist,...);\n");
   fprintf(fp,"   int scanf(char *format,arglist,...);\n");
   fprintf(fp,"   int sscanf(char *string,char *format,arglist,...);\n");
   fprintf(fp,"   exit(int status);\n");
   fprintf(fp,"   atexit(void(*function)(void));\n");
   fprintf(fp,"   void qsort(void *array,size_t number,size_t size,int (*comparison)(void *arg1,void *arg2));\n");
   fprintf(fp,"   void bsearch(void *item,void *array,size_t number,size_t size,int (*comparison)(void *arg1,void *arg2));\n");
   fprintf(fp,"   char getopt(int argc,char **argv,char *options);\n");
   fprintf(fp,"   char *G__input(char prompt[]);\n");
   fprintf(fp,"   int G__pause();\n");
   fprintf(fp,"   int G__tracemode(int on_off);\n");
   /* fprintf(fp,"   int G__breakline(int line);\n"); */
   fprintf(fp,"   int G__setbreakpoint(char* line,char* file);\n");
   fprintf(fp,"   int G__stepmode(int on_off);\n");
   fprintf(fp,"   [anytype] G__calc(const char *expression);\n");
   fprintf(fp,"   [anytype] G__exec_text(const char *unnamedmacro);\n");
   fprintf(fp,"   [anytype] G__exec_tempfile(const char *file);\n");
#ifndef G__OLDIMPLEMENTATION1546
   fprintf(fp,"   char* G__load_text(const char *namedmacro);\n");
#endif
   fprintf(fp,"   int G__loadfile(const char *file);\n");
   fprintf(fp,"   int G__unloadfile(const char *file);\n");
   fprintf(fp,"   int G__reloadfile(const char *file);\n");
   fprintf(fp,"   int G__loadsystemfile(const char* sysdllname);\n");
   fprintf(fp,"   void G__add_ipath(const char *pathname);\n");
   fprintf(fp,"   int G__split(char *line,char *string,int argc,char *argv[]);\n");
   fprintf(fp,"   int G__readline(FILE *fp,char *line,char *argbuf,int *argn,char *arg[]);\n");
   fprintf(fp,"   int G__cmparray(short array1[],short array2[],int num,short mask);\n");
   fprintf(fp,"   G__setarray(short array[],int num,short mask,char *mode);\n");
   fprintf(fp,"   G__graph(double xdata[],double ydata[],int ndata,char *title,int mode);\n");
#ifndef G__NSEARCHMEMBER
   fprintf(fp,"   char *G__search_next_member(char *name,int state);\n");
   fprintf(fp,"   void *G__what_type(char *name,char *type,char *tagname,char *typename);\n");
#endif
#ifndef G__NSTOREOBJECT
   fprintf(fp,"   int *G__storeobject(void *buf1,void *buf2);\n");
   fprintf(fp,"   int *G__scanobject(void *buf);\n");
   fprintf(fp,"   int *G__dumpobject(char *file,void *buf,int size);\n");
   fprintf(fp,"   int *G__loadobject(char *file,void *buf,int size);\n");
#endif
   fprintf(fp,"   int G__lock_variable(char *varname);\n");
   fprintf(fp,"   int G__unlock_variable(char *varname);\n");

   /* added 1999 Oct 23 */
   fprintf(fp,"   int G__IsInMacro();\n");
   fprintf(fp,"   char* G__getmakeinfo(char* paramname);\n");
   fprintf(fp,"   void G__set_atpause(void* p2f);\n");
   fprintf(fp,"   void G__set_aterror(void (*p2f)());\n");
   fprintf(fp,"   void G__set_ignoreinclude(int (*p2f)(char* fname,char* expandedfname));\n");
   fprintf(fp,"   void G__set_smartload(int smartunload);\n");
   fprintf(fp,"   int G__AddConsole();\n");
   fprintf(fp,"   int G__FreeConsole();\n");
   /* fprintf(fp,"   char* G__type2string();\n"); */
   fprintf(fp,"   type_info G__typeid(char* typename);\n");
   fprintf(fp,"   char* G__lasterror_filename();\n");
   fprintf(fp,"   int G__lasterror_linenum();\n");
   fprintf(fp,"   char* G__p2f2funcname(void* p2f);\n");
   fprintf(fp,"   int G__isinterpretedp2f(void* p2f);\n");
   fprintf(fp,"   int G__deleteglobal(void* ptr);\n");
   fprintf(fp,"   int G__deletevariable(char* varname);\n");
   fprintf(fp,"   int G__optimizemode(int optlevel);\n");
   fprintf(fp,"   int G__getoptimizemode();\n");
   fprintf(fp,"   void G__clearerror();\n");
   fprintf(fp,"   int G__defined(const char* type_name);\n");
}


