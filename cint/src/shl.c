/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file shl.c
 ************************************************************************
 * Description:
 *  Define macro
 ************************************************************************
 * Copyright(c) 1995~2002  Masaharu Goto 
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#include "common.h"
#include "dllrev.h"


/*********************************************************************
* dynamic link library(shared library) enhancement
*********************************************************************/
#ifdef G__SHAREDLIB
/***************************************************
* OSF compliant or SunOS
****************************************************/
#if defined(G__OSFDLL)
typedef void* G__SHLHANDLE;
#if defined(G__ROOT) && defined(_AIX) && defined(G__AIXDLFCN)
#include <aixdlfcn.h>
#else
#include <dlfcn.h>
#endif
#define TYPE_PROCEDURE 1
#define TYPE_DATA 2
#ifndef RTLD_LAZY
#define RTLD_LAZY 1
#endif
/***************************************************
* HP-UX 8.0 or later
****************************************************/
#elif defined(__hpux) || defined(_HIUX_SOURCE)
#include <dl.h>
#if defined(G__HPUXCPPDLL) && !defined(__STDCPP__)
#include <CC/cxxdl.h>
#endif
typedef shl_t G__SHLHANDLE;
#if defined(__hp9000s300)||defined(__hp9000s400)
# ifndef G__DLL_SYM_UNDERSCORE
#define G__DLL_SYM_UNDERSCORE
# endif
#endif
/***************************************************
* Win32
****************************************************/
#elif defined(G__WIN32)
#include <windows.h>
typedef HINSTANCE G__SHLHANDLE;
#define TYPE_PROCEDURE 1
#define TYPE_DATA 2
/***************************************************
* VMS
****************************************************/
#elif defined(G__VMS)
#include <lib$routines.h>
#include <descrip.h>
typedef char* G__SHLHANDLE;
#define TYPE_PROCEDURE 1
#define TYPE_DATA 2
/***************************************************
* Non of above
****************************************************/
#else /* non of above */
/* #error "Error: G__OSFDLL must be give with G__SHAREDLIB" */
typedef void* G__SHLHANDLE;
#endif /* !__hpux && !G__OSFDLL */

/***************************************************
* Common settings
****************************************************/
#define G__GLOBALSETUP   "G__globalsetup"
#define G__COMPILED_FUNC "G__compiled_func"
#define G__REVISION      "G__revision"
#define G__LIST_SUT      "G__list_sut"
#define G__LIST_STUB     "G__list_stub"
#define G__LIST_STRUCT   "G__list_struct"
#define G__LIST_GLOBAL   "G__list_global"
#define G__COMPLETION    "G__completionlist"
#define G__DLL_REVISION  "G__dll_revision"

#define G__CALL_SETUP(setupfunc)  \
    sharedlib_func=(int (*)())G__shl_findsym(&G__sl_handle[allsl],setupfunc,TYPE_PROCEDURE);   \
    if(sharedlib_func!=NULL) (*sharedlib_func)()

#else /* G__SHAREDLIB */

typedef void* G__SHLHANDLE;

#endif /* G__SHAREDLIB */

G__SHLHANDLE G__sl_handle[G__MAX_SL];
short G__allsl=0;

#ifdef G__DLL_SYM_UNDERSCORE
static int G__sym_underscore=1;
#else
static int G__sym_underscore=0;
#endif

void G__set_sym_underscore(x) int x; { G__sym_underscore=x; }
int G__get_sym_underscore() { return(G__sym_underscore); }


#ifndef __CINT__
G__SHLHANDLE G__dlopen G__P((char *path));
void *G__shl_findsym G__P((G__SHLHANDLE *phandle,char *sym,short type));
int G__dlclose G__P((G__SHLHANDLE handle));
#endif

#ifndef G__PHILIPPE1
/****************************************************
* OSF or SunOS
****************************************************/
#if defined(G__OSFDLL)
#if defined(__FreeBSD__) || defined(__OpenBSD__) || (defined(__alpha) && !defined(__linux) && !defined(__linux__) && !defined(linux)) || (defined(G__SUNOS4) && defined(G__NONANSI))
# define G__RTLD_NOW RTLD_NOW
# define G__RTLD_LAZY RTLD_LAZY
#else
#ifndef RTLD_GLOBAL
#define RTLD_GLOBAL 0
#endif
# define G__RTLD_NOW  (RTLD_GLOBAL | RTLD_NOW)
# define G__RTLD_LAZY (RTLD_GLOBAL | RTLD_LAZY)
#endif
/****************************************************
* HP-UX
****************************************************/
#elif defined(__hpux) || defined(_HIUX_SOURCE)
# define G__RTLD_NOW BIND_IMMEDIATE
# define G__RTLD_LAZY BIND_DEFERRED
/****************************************************
* Win32
****************************************************/
#elif defined(G__WIN32)
  /* Do not know how to force binding at load time */
# define G__RTLD_NOW 0
# define G__RTLD_LAZY 0
/****************************************************
* Other
****************************************************/
#else
# define G__RTLD_NOW 0
# define G__RTLD_LAZY 0
#endif /* G__WIN32 */

#ifdef G__SHAREDLIB
static int G__RTLD_flag = G__RTLD_LAZY;
#endif


#ifndef G__OLDIMPLEMENTATION1207
int G__ispermanentsl = 0;
G__DLLINIT G__initpermanentsl ;

/**************************************************************************
* G__loadsystemfile
**************************************************************************/
int G__loadsystemfile(filename) 
char *filename;
{
  int result;
  int len = strlen(filename);
#if defined(R__FBSD) || defined(R__OBSD)
  char soext[]=SOEXT;
#endif
  if((len>3&& (strcmp(filename+len-3,".sl")==0 ||
	       strcmp(filename+len-3,".dl")==0 ||
	       strcmp(filename+len-3,".so")==0)) ||
     (len>4&& (strcmp(filename+len-4,".dll")==0 ||
	       strcmp(filename+len-4,".DLL")==0)) ||
#if defined(R__FBSD) || defined(R__OBSD)
     (len>strlen(soext) && strcmp(filename+len-strlen(soext), soext)==0) ||
#endif
     (len>2&& (strcmp(filename+len-2,".a")==0 ||
	       strcmp(filename+len-2,".A")==0))
     ) {
  }
  else {
    G__fprinterr(G__serr,"Error: G__loadsystemfile can only load DLL");
    G__printlinenum();
    return(G__LOADFILE_FAILURE);
  }
  G__ispermanentsl=1;
  result = G__loadfile(filename);
  G__ispermanentsl=0;
  return(result);
}
#endif


/***********************************************************************
* G__Set_RTLD_NOW() && G__Set_RTLD_LAZY()
*
***********************************************************************/
void G__Set_RTLD_NOW() {
#ifdef G__SHAREDLIB
  G__RTLD_flag = G__RTLD_NOW;
#endif
}
void G__Set_RTLD_LAZY() {
#ifdef G__SHAREDLIB
  G__RTLD_flag = G__RTLD_LAZY;
#endif
}
#endif /* G__PHILIPPE1 */

/***********************************************************************
* G__dlopen()
*
***********************************************************************/
G__SHLHANDLE G__dlopen(path)
char *path;
{
  G__SHLHANDLE handle;
#ifdef G__SHAREDLIB

#ifndef G__PHILIPPE1

/****************************************************
* OSF or SunOS
****************************************************/
#if defined(G__OSFDLL)

  handle = dlopen(path,G__RTLD_flag);
  if(!handle) G__fprinterr(G__serr,"dlopen error: %s\n",dlerror());

/****************************************************
* HP-UX
****************************************************/
# elif defined(__hpux) || defined(_HIUX_SOURCE)
#  if defined(G__HPUXCPPDLL) && !defined(__STDCPP__)
  handle = cxxshl_load(path,G__RTLD_flag,0L);
#  else
  handle = shl_load(path,G__RTLD_flag,0L);
#  endif
#ifndef G__OLDIMPLEMENTATION984
#  if defined(__GNUC__)
  {
     /* find all _GLOBAL__FI_* functions to initialize global static objects */
     struct shl_symbol *symbols;
     int nsym;
     nsym = shl_getsymbols(handle, TYPE_PROCEDURE, EXPORT_SYMBOLS|NO_VALUES,
                           (void *(*)())malloc, &symbols);
     if (nsym != -1) {
        void (*ctor)();
        int i;
        for (i = 0; i < nsym; i++) {
           if (symbols[i].type == TYPE_PROCEDURE) {
              if (!strncmp(symbols[i].name, "_GLOBAL__FI_", 12)) {
                 ctor = (void (*)())G__shl_findsym(&handle, symbols[i].name,
TYPE_PROCEDURE);
                 if (ctor) (*ctor)();
              }
           }
        }
        free(symbols);
     }
  }
#  endif
#endif
/****************************************************
* Win32
****************************************************/
# elif defined(G__WIN32)
  handle = LoadLibrary(path);
/****************************************************
* VMS
****************************************************/
# elif defined(G__VMS)
  handle = path;
/****************************************************
* Non of above
****************************************************/
# else /* non of above */
  handle = (G__SHLHANDLE)NULL;
# endif

#else /* G__PHILIPPE1 */

/****************************************************
* OSF or SunOS
****************************************************/
#if defined(G__OSFDLL)
#if defined(__FreeBSD__) || (defined(__alpha) && !defined(__linux) && !defined(__linux__) && !defined(linux)) || (defined(G__SUNOS4) && defined(G__NONANSI))
  handle = dlopen(path,RTLD_LAZY);
#else
#ifndef RTLD_GLOBAL
#define RTLD_GLOBAL 0
#endif
  handle = dlopen(path,RTLD_GLOBAL | RTLD_LAZY);
#endif
#ifndef G__OLDIMPLEMENTATION861
  if(!handle) G__fprinterr(G__serr,"dlopen error: %s\n",dlerror());
#endif
/****************************************************
* HP-UX
****************************************************/
# elif defined(__hpux) || defined(_HIUX_SOURCE)
#  if defined(G__HPUXCPPDLL) && !defined(__STDCPP__)
  handle = cxxshl_load(path,BIND_DEFERRED,0L);
#  else
  handle = shl_load(path,BIND_DEFERRED,0L);
#  endif
#ifndef G__OLDIMPLEMENTATION984
#  if defined(__GNUC__)
  {
     /* find all _GLOBAL__FI_* functions to initialize global static objects */
     struct shl_symbol *symbols;
     int nsym;
     nsym = shl_getsymbols(handle, TYPE_PROCEDURE, EXPORT_SYMBOLS|NO_VALUES,
                           (void *(*)())malloc, &symbols);
     if (nsym != -1) {
        void (*ctor)();
        int i;
        for (i = 0; i < nsym; i++) {
           if (symbols[i].type == TYPE_PROCEDURE) {
              if (!strncmp(symbols[i].name, "_GLOBAL__FI_", 12)) {
                 ctor = (void (*)())G__shl_findsym(&handle, symbols[i].name,
TYPE_PROCEDURE);
                 if (ctor) (*ctor)();
              }
           }
        }
        free(symbols);
     }
  }
#  endif
#endif
/****************************************************
* Win32
****************************************************/
# elif defined(G__WIN32)
  handle = LoadLibrary(path);
/****************************************************
* VMS
****************************************************/
#elif defined(G__VMS)
  handle = path;
/****************************************************
* Non of above
****************************************************/
# else /* non of above */
  handle = (G__SHLHANDLE)NULL;
# endif

#endif /* G__PHILIPPE1 */

#else /* G__SHAREDLIB */
  handle = (G__SHLHANDLE)NULL;
#endif /* G__SHAREDLIB */

  return(handle);
}

/***********************************************************************
* G__shl_findsym()
*
***********************************************************************/
void *G__shl_findsym(phandle,sym,type)
G__SHLHANDLE *phandle;
char *sym;
short type;
{
  void *func = (void*)NULL;

#ifdef G__VMS
  char *file_s, *sym_s, *phandle_s;
  char pfile[G__ONELINE],pfile1[G__ONELINE],*p,*post;
  int lib_status;
  int lib_func;
  struct dsc$descriptor_s file_d;
  struct dsc$descriptor_s sym_d;
  struct dsc$descriptor_s phandle_d;
#endif
  char sym_underscore[G__ONELINE];

#ifndef G__OLDIMPLEMENTATION1911
  if(0 && type) return((void*)NULL);
#endif

  if(G__sym_underscore) {
    sym_underscore[0]='_';
    strcpy(sym_underscore+1,sym);
  }
  else {
    strcpy(sym_underscore,sym);
  }

  if(!(*phandle)) return(func);

#ifdef G__SHAREDLIB
/****************************************************
* OSF or SunOS
****************************************************/
#if defined(G__OSFDLL)
  func = dlsym(*phandle,sym_underscore);
/****************************************************
* HP-UX
****************************************************/
# elif defined(__hpux) || defined(_HIUX_SOURCE)
  shl_findsym(phandle,sym_underscore,type,(void*)(&func));
/****************************************************
* Win32
****************************************************/
# elif defined(G__WIN32)
  func = (void*)GetProcAddress(*phandle,sym_underscore);
/****************************************************
* VMS
****************************************************/
# elif defined(G__VMS)

/*
Set up character string descriptors for the  call to lib$find_image_symbol.
The first argument needs to be the filename alone without the directory info.
The last argument needs the complete file name including device and extension.
*/
/*We need to see if the symbol contains the name of the file without
directories or extensions because that is what is in what rootcint generates.
lib$find_image_symbol crashes if we look for a symbol that is not in there,
so we try to catch symbols not in there before we call it*/
  strcpy(pfile,G__ifile.name);

  p = strrchr(pfile,']');

  if (p) {
     p++;
  }
  else {
     p = pfile;
  }

  post = strchr(p,'.');
  if( post ) *post = 0;

/*printf("\nG__ifile.name is %s sym is %s\n, p is %s",G__ifile.name,sym,p);*/
  if(!strstr(sym,p)) return 0;

/*We also have no G__c_... stuff in cint files generated with rootcint.  Don't
  call those either!!!*/
  if(strstr(sym,"G__c_")) return 0;

  strcpy(pfile1,*phandle);
  file_s = strrchr(pfile1,']')+1;
  post = strrchr(file_s,'.');
  if( post ) *post = 0;
  file_d.dsc$a_pointer = file_s;
  file_d.dsc$w_length = strlen(file_s);
  file_d.dsc$b_dtype = DSC$K_DTYPE_T;
  file_d.dsc$b_class = DSC$K_CLASS_S;

  sym_s = sym;
  if(strstr(sym_s,"G__cpp_dllrev")) {
/*   This one is not defined as extern "C" in rootcint, so name is mangled */
/*   We need to mangle this one to match up */
     strcpy(&sym_s[strlen(sym_s)],"__xv");
     }
  sym_d.dsc$a_pointer = sym_s;
  sym_d.dsc$w_length = strlen(sym_s);
  sym_d.dsc$b_dtype = DSC$K_DTYPE_T;
  sym_d.dsc$b_class = DSC$K_CLASS_S;

  phandle_s = *phandle;
  phandle_d.dsc$a_pointer = phandle_s;
  phandle_d.dsc$w_length = strlen(phandle_s);
  phandle_d.dsc$b_dtype = DSC$K_DTYPE_T;
  phandle_d.dsc$b_class = DSC$K_CLASS_S;

printf("\nfile_s %s sym_s %s phandle_s %s",file_s,sym_s,phandle_s);
/*printf("\n lengths are %d %d %d",file_d.dsc$w_length,sym_d.dsc$w_length,phandle_d.dsc$w_length);*/

  lib_status = lib$find_image_symbol(&file_d,&sym_d,&lib_func,&phandle_d);

  func = (void*)lib_func;

/****************************************************
* Non of above
****************************************************/
# else /* non of above */
# endif
#endif /* G__SHAREDLIB */

  return(func);
}


/***********************************************************************
* G__dlclose()
*
***********************************************************************/
int G__dlclose(handle)
G__SHLHANDLE handle;
{
  if(!handle) return(0);

#ifdef G__SHAREDLIB
/****************************************************
* OSF or SunOS
****************************************************/
#if defined(G__OSFDLL)
  return(dlclose(handle));
/****************************************************
* HP-UX
****************************************************/
# elif defined(__hpux) || defined(_HIUX_SOURCE)
#ifdef G__OLDIMPLEMENTATION1136
#  if defined(G__HPUXCPPDLL) && !defined(__STDCPP__)
  return(cxxshl_unload(handle));
#  else
  return(shl_unload(handle));
#  endif
#endif
#ifndef G__OLDIMPLEMENTATION984
#  if defined(__GNUC__)
  {
     /* find all _GLOBAL__FD_* functions to destruct global static objects */
     struct shl_symbol *symbols;
     int nsym;
     nsym = shl_getsymbols(handle, TYPE_PROCEDURE, EXPORT_SYMBOLS|NO_VALUES,
                           (void *(*)())malloc, &symbols);
     if (nsym != -1) {
        void (*dtor)();
        int i;
        for (i = 0; i < nsym; i++) {
           if (symbols[i].type == TYPE_PROCEDURE) {
              if (!strncmp(symbols[i].name, "_GLOBAL__FD_", 12)) {
                 dtor = (void (*)())G__shl_findsym(&handle, symbols[i].name,
TYPE_PROCEDURE);
                 if (dtor) (*dtor)();
              }
           }
        }
        free(symbols);
     }
  }
#  endif
#endif
#ifndef G__OLDIMPLEMENTATION1136
#  if defined(G__HPUXCPPDLL) && !defined(__STDCPP__)
  return(cxxshl_unload(handle));
#  else
  return(shl_unload(handle));
#  endif
#endif
/****************************************************
* Win32
****************************************************/
# elif defined(G__WIN32)
  FreeLibrary(handle);
  return(0);
/****************************************************
* Non of above
****************************************************/
# else /* non of above */
  return(0);
# endif
#else /* G__SHAREDLIB */
  return(0);
#endif /* G__SHAREDLIB */
}

#ifndef G__OLDIMPLEMENTATION1273
/***********************************************************************
* G__smart_shl_unload()
***********************************************************************/
void G__smart_shl_unload(allsl)
int allsl;
{
  if(G__sl_handle[allsl]) {
    if(G__dlclose(G__sl_handle[allsl]) == -1) {
      G__fprinterr(G__serr,"Error: Dynamic link library unloading error\n");
    }
    G__sl_handle[allsl]=0;
  }
}
#endif

#ifdef G__SHAREDLIB
/***********************************************************************
* int G__free_shl_upto()
*
*  Can replace G__free_shl()
*
***********************************************************************/
int G__free_shl_upto(allsl)
int allsl;
{
  /*************************************************************
   * Unload shared library
   *************************************************************/
  while((--G__allsl)>=allsl) {
    if(G__dlclose(G__sl_handle[G__allsl]) == -1) {
      G__fprinterr(G__serr,"Error: Dynamic link library unloading error\n");
    }
    else {
      G__sl_handle[G__allsl]=0;
    }
  }
  G__allsl=allsl;
  return(0);
}

#endif

#ifndef G__OLDIMPLEMENTATION863
/**************************************************************************
* G__findsym()
**************************************************************************/
void* G__findsym(fname)
char* fname;
{
#ifdef G__SHAREDLIB
  int i;
  void *p;
  for(i=0;i<G__allsl;i++) {
    p = (void*)G__shl_findsym(&G__sl_handle[i],fname,TYPE_PROCEDURE);
    if(p) return(p);
  }
#endif
  return((void*)NULL);
}
#endif


/**************************************************************************
* G__revprint()
**************************************************************************/
int G__revprint(fp)
FILE *fp;
{
  G__cintrevision(fp);
  G__list_sut(fp);
  return(0);
}


#ifdef G__NEVER
/**************************************************************************
* G__dump_header()
**************************************************************************/
int G__dump_header(outfile)
char *outfile;
{
  return( outfile? 1 : 0 );
}
#endif


#ifdef G__ROOT
extern int G__call_setup_funcs();
#endif

#ifdef G__SHAREDLIB
/**************************************************************************
 * G__show_dllrev
 **************************************************************************/
void G__show_dllrev(shlfile,sharedlib_func)
char *shlfile;
int (*sharedlib_func)();
{
  G__fprinterr(G__serr,"%s:DLLREV=%d\n",shlfile,(*sharedlib_func)());
  G__fprinterr(G__serr,"  This cint accepts DLLREV=%d~%d and creates %d\n"
	  ,G__ACCEPTDLLREV_FROM,G__ACCEPTDLLREV_UPTO
	  ,G__CREATEDLLREV);
}

#ifndef G__OLDIMPLEMENTATION1525
/**************************************************************************
 * G__show_dllrev
 **************************************************************************/
#if !defined(G__OLDIMPLEMENTATION1485)
typedef void (*G__SetCintApiPointers_t) G__P((void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*));
#elif !defined(G__OLDIMPLEMENTATION1546)
typedef void (*G__SetCintApiPointers_t) G__P((void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*));
#else
typedef void (*G__SetCintApiPointers_t) G__P((void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
	void*,void*,void*,void*,void*,void*,void*));
#endif

/**************************************************************************
* G__SetCIntApiPointers
*
**************************************************************************/
void G__SetCintApiPointers(pslhandle,fname)
G__SHLHANDLE *pslhandle;
char *fname;
{
  G__SetCintApiPointers_t SetCintApi;
  SetCintApi = (G__SetCintApiPointers_t)
    G__shl_findsym(pslhandle,fname,TYPE_PROCEDURE);
  if(SetCintApi)
    (*SetCintApi)(
	G__main,
	G__setothermain,
	G__getnumbaseclass,
	G__setnewtype,
	G__setnewtypeindex,
	G__resetplocal,
	G__getgvp,
	G__resetglobalenv,
	G__lastifuncposition,
	G__resetifuncposition,
	G__setnull,
	G__getstructoffset,
	G__getaryconstruct,
	G__gettempbufpointer,
	G__setsizep2memfunc,
	G__getsizep2memfunc,
	G__get_linked_tagnum,
	G__tagtable_setup,
	G__search_tagname,
	G__search_typename,
	G__defined_typename,
	G__tag_memvar_setup,
	G__memvar_setup,
	G__tag_memvar_reset,
	G__tag_memfunc_setup,
	G__memfunc_setup,
	G__memfunc_next,
	G__memfunc_para_setup,
	G__tag_memfunc_reset,
	G__letint,
	G__letdouble,
	G__store_tempobject,
	G__inheritance_setup,
	G__add_compiledheader,
	G__add_ipath,
	G__add_macro,
	G__check_setup_version,
	G__int,
	G__double,
	G__calc,
	G__loadfile,
	G__unloadfile,
	G__init_cint,
	G__scratch_all,
	G__setdouble,
	G__setint,
	G__stubstoreenv,
	G__stubrestoreenv,
	G__getstream,
	G__type2string,
	G__alloc_tempobject,
	G__set_p2fsetup,
	G__free_p2fsetup,
	G__genericerror,
	G__tmpnam,
	G__setTMPDIR,
	G__setPrerun,
	G__readline,
	G__getFuncNow,
	G__getIfileFp,
	G__incIfileLineNumber,
	G__setReturn,
	G__getPrerun,
	G__getDispsource,
	G__getSerr,
	G__getIsMain,
	G__setIsMain,
	G__setStep,
	G__getStepTrace,
	G__setDebug,
	G__getDebugTrace,
	G__set_asm_noverflow,
	G__get_no_exec,
	G__get_no_exec_compile,
	G__setdebugcond,
	G__init_process_cmd,
	G__process_cmd,
	G__pause,
	G__input,
	G__split,
	G__getIfileLineNumber,
	G__addpragma,
	G__add_setup_func,
	G__remove_setup_func,
	G__setgvp,
	G__set_stdio_handle,
	G__setautoconsole,
	G__AllocConsole,
	G__FreeConsole,
	G__getcintready,
	G__security_recover,
	G__breakkey,
	G__stepmode,
	G__tracemode,
	G__getstepmode,
	G__gettracemode,
	G__printlinenum,
	G__search_typename2,
	G__set_atpause,
	G__set_aterror,
	G__p2f_void_void,
	G__setglobalcomp,
	G__getmakeinfo,
	G__get_security_error,
	G__map_cpp_name,
	G__Charref,
	G__Shortref,
	G__Intref,
	G__Longref,
	G__UCharref,
	G__UShortref,
	G__UIntref,
	G__ULongref,
	G__Floatref,
	G__Doubleref,
	G__loadsystemfile,
	G__set_ignoreinclude,
	G__exec_tempfile,
	G__exec_text,
	G__lasterror_filename,
	G__lasterror_linenum,
	G__va_arg_put
#ifndef G__OLDIMPLEMENTATION1546
	,G__load_text
	,G__set_emergencycallback
#endif
#ifndef G__OLDIMPLEMENTATION1485
	,G__set_errmsgcallback
#endif
	);
}

#endif

/**************************************************************************
* G__shl_load()
*
* Comment:
*  This function can handle both old and new style DLL.
**************************************************************************/
int G__shl_load(shlfile)
char *shlfile;
{
  /* int fail = 0; */
  int store_globalcomp;
  char *p;
  char *post;
  char dllid[G__ONELINE];
  int (*sharedlib_func)();
  int error=0,cintdll=0;
  char dllidheader[G__ONELINE];
#ifndef G__OLDIMPLEMENTATION885
  int allsl=G__allsl;
#endif

  if(G__allsl==G__MAX_SL) {
    G__shl_load_error(shlfile ,"Too many DLL");
#ifndef G__OLDIMPLEMENTATION1908
    return(-1);
#else
    return(EXIT_FAILURE);
#endif
  }
  else ++G__allsl;

#ifndef G__OLDIMPLEMENTATION1566
#ifdef G__ROOT
  /* this pointer must be set before calling dlopen! */
  G__initpermanentsl = (void (*)())NULL;
#endif
#endif

  G__sl_handle[allsl] = G__dlopen(shlfile);

#ifndef G__OLDIMPLEMENTATION1525
  if(G__sym_underscore) {
    G__SetCintApiPointers(&G__sl_handle[allsl],"_G__SetCCintApiPointers");
    G__SetCintApiPointers(&G__sl_handle[allsl],"_G__SetCppCintApiPointers");
  }
  else {
    G__SetCintApiPointers(&G__sl_handle[allsl],"G__SetCCintApiPointers");
    G__SetCintApiPointers(&G__sl_handle[allsl],"G__SetCppCintApiPointers");
  }
#endif

  if(NULL==G__sl_handle[allsl]) {
    if(G__ispragmainclude) {
      if(G__dispmsg>=G__DISPWARN) {
	G__fprinterr(G__serr,"Warning: Can not load Dynamic Link Library %s",shlfile);
	G__printlinenum();
      }
#ifndef G__OLDIMPLEMENTATION936
      --G__allsl;
#endif
#ifndef G__OLDIMPLEMENTATION1908
      return(-1);
#else
      return(EXIT_FAILURE);
#endif
    }
    else {
      G__shl_load_error(shlfile,"Load Error");
#ifndef G__OLDIMPLEMENTATION936
      --G__allsl;
#endif
#ifndef G__OLDIMPLEMENTATION1908
      return(-1);
#else
      return(EXIT_FAILURE);
#endif
    }
  }

  /* set file name */
#ifndef G__OLDIMPLEMENTATION1896
  if(G__ifile.name!=shlfile) strcpy(G__ifile.name,shlfile);
#else
  strcpy(G__ifile.name,shlfile);
#endif

#ifndef G__OLDIMPLEMENTATION670
#ifdef G__WIN32
  p = shlfile;
  while(p) {
    p = strchr(p,'/');
    if(p) *p = '\\';
  }
#endif
#endif

  /* Split filename and get DLLID string */
  p = strrchr(shlfile,'/');
  if(p) {
    p++;
  }
  else {
    p = strrchr(shlfile,'\\');
    if(p) {
      p++;
    }
    else {
      p = shlfile;
    }
  }

#ifdef G__VMS
/*Have to do things differently for VMS files with directories attached*/
  p = strrchr(shlfile,']');
  if(p) {
    p++;
  }
  else {
    p = shlfile;
  }
#endif

  strcpy(dllidheader,p);
  post = strchr(dllidheader,'.');
  if(post)  *post = '\0';

  sprintf(dllid,"G__cpp_dllrev");
  sharedlib_func=
    (int (*)())G__shl_findsym(&G__sl_handle[allsl],dllid,TYPE_PROCEDURE);
#ifndef G__OLDIMPLEMENTATION1169
  if(sharedlib_func && ((*sharedlib_func)()>G__ACCEPTDLLREV_UPTO
     || (*sharedlib_func)()<G__ACCEPTDLLREV_FROM)) {
#else
  if(sharedlib_func && (*sharedlib_func)()!=G__DLLREV) {
#endif
    G__check_setup_version((*sharedlib_func)(),shlfile);
    error++;
  }
  if(sharedlib_func) {
    cintdll++;
    if(G__asm_dbg) G__show_dllrev(shlfile,sharedlib_func);
  }

  sprintf(dllid,"G__cpp_dllrev%s",dllidheader);
  sharedlib_func=
    (int (*)())G__shl_findsym(&G__sl_handle[allsl],dllid,TYPE_PROCEDURE);
#ifndef G__OLDIMPLEMENTATION1169
  if(sharedlib_func && ((*sharedlib_func)()>G__ACCEPTDLLREV_UPTO 
     || (*sharedlib_func)()<G__ACCEPTDLLREV_FROM)) {
#else
  if(sharedlib_func && (*sharedlib_func)()!=G__DLLREV) {
#endif
    G__check_setup_version((*sharedlib_func)(),shlfile);
    error++;
  }
  if(sharedlib_func) {
    cintdll++;
    if(G__asm_dbg) G__show_dllrev(shlfile,sharedlib_func);
  }

  sprintf(dllid,"G__c_dllrev");
  sharedlib_func=
    (int (*)())G__shl_findsym(&G__sl_handle[allsl],dllid,TYPE_PROCEDURE);
#ifndef G__OLDIMPLEMENTATION1169
  if(sharedlib_func && ((*sharedlib_func)()>G__ACCEPTDLLREV_UPTO
     || (*sharedlib_func)()<G__ACCEPTDLLREV_FROM)) {
#else
  if(sharedlib_func && (*sharedlib_func)()!=G__DLLREV) {
#endif
    G__check_setup_version((*sharedlib_func)(),shlfile);
    error++;
  }
  if(sharedlib_func) {
    cintdll++;
    if(G__asm_dbg) G__show_dllrev(shlfile,sharedlib_func);
  }

  sprintf(dllid,"G__c_dllrev%s",dllidheader);
  sharedlib_func=
    (int (*)())G__shl_findsym(&G__sl_handle[allsl],dllid,TYPE_PROCEDURE);
#ifndef G__OLDIMPLEMENTATION1169
  if(sharedlib_func && ((*sharedlib_func)()>G__ACCEPTDLLREV_UPTO
     || (*sharedlib_func)()<G__ACCEPTDLLREV_FROM)) {
#else
  if(sharedlib_func && (*sharedlib_func)()!=G__DLLREV) {
#endif
    G__check_setup_version((*sharedlib_func)(),shlfile);
    error++;
  }
  if(sharedlib_func) {
    cintdll++;
    if(G__asm_dbg) G__show_dllrev(shlfile,sharedlib_func);
  }

  if(error) {
    G__shl_load_error(shlfile ,"Revision mismatch");
#ifdef G__OLDIMPLEMENTATION885
    ++G__allsl;
#endif
#ifndef G__OLDIMPLEMENTATION936
    --G__allsl;
#endif
#ifndef G__OLDIMPLEMENTATION1908
    return(-1);
#else
    return(EXIT_FAILURE);
#endif
  }
  if(G__asm_dbg&&0==cintdll) {
    if(G__dispmsg>=G__DISPWARN) {
      G__fprinterr(G__serr,"Warning: No CINT symbol table in %s\n",shlfile);
    }
  }


  /*
   * initialize global variables in
   * shared library. */

  G__prerun=1;
  G__setdebugcond();
  store_globalcomp=G__globalcomp;
  G__globalcomp=G__NOLINK;


  sprintf(dllid,"G__cpp_setup%s",dllidheader);
  G__CALL_SETUP("G__set_cpp_environment");
#ifndef G__OLDIMPLEMENTATION1053
  G__CALL_SETUP("G__cpp_setup_tagtable");
#endif
  G__CALL_SETUP("G__cpp_setup_inheritance");
  G__CALL_SETUP("G__cpp_setup_typetable");
  /* G__CALL_SETUP("G__cpp_setup_memvar");
   * G__CALL_SETUP("G__cpp_setup_memfunc"); */
  G__CALL_SETUP("G__cpp_setup_global");
  G__CALL_SETUP("G__cpp_setup_func");
#ifdef G__OLDIMPLEMENTATION1053
  G__CALL_SETUP("G__cpp_setup_tagtable");
#endif
  if(sharedlib_func==NULL) {
    G__CALL_SETUP(dllid);
  }
#ifdef G__ROOT
#ifdef G__OLDIMPLEMENTATION1566
#ifndef G__OLDIMPLEMENTATION1207
  G__initpermanentsl = (void (*)())NULL;
#endif
#endif
  if (sharedlib_func==NULL) G__call_setup_funcs();
#endif

  sprintf(dllid,"G__c_setup%s",dllidheader);
  G__CALL_SETUP("G__set_c_environment");
  G__CALL_SETUP("G__c_setup_typetable");
  /* G__CALL_SETUP("G__c_setup_memvar"); */
  G__CALL_SETUP("G__c_setup_global");
  G__CALL_SETUP("G__c_setup_func");
  G__CALL_SETUP("G__c_setup_tagtable");
  if(sharedlib_func==NULL) {
    G__CALL_SETUP(dllid);
  }

  if(0==G__sizep2memfunc) {
    sprintf(dllid,"G__get_sizep2memfunc%s",dllidheader);
    p = strchr(dllid,'.');
    if(p)  *p = '\0';
    G__CALL_SETUP(dllid);
  }

  sharedlib_func=
    (int (*)())G__shl_findsym(&G__sl_handle[allsl],G__GLOBALSETUP,TYPE_PROCEDURE);
  if(sharedlib_func!=NULL) {
    (*sharedlib_func)();
  }

  G__prerun=0;
  G__setdebugcond();
  G__globalcomp=store_globalcomp;

#ifndef G__OLDIMPLEMENTATION1207
  if(G__ispermanentsl) {
    if(!G__initpermanentsl)
      G__initpermanentsl = 
	(void (*)())G__shl_findsym(&G__sl_handle[allsl],"G__cpp_setup"
				   ,TYPE_PROCEDURE); 
    if(!G__initpermanentsl) {
      sprintf(dllid,"G__cpp_setup%s",dllidheader);
      G__initpermanentsl =
	(void (*)())G__shl_findsym(&G__sl_handle[allsl],dllid,TYPE_PROCEDURE); 
    }
    --G__allsl;
  }
  else {
    G__initpermanentsl = (void (*)())NULL;
  }
#endif

#ifdef G__OLDIMPLEMENTATION885
  ++G__allsl; /* anyway increment this */
#endif

  strcpy(G__ifile.name,"");
#ifndef G__OLDIMPLEMENTATION1908
  return(EXIT_SUCCESS);
#else
  return(allsl);
#endif
}
#endif


/*******************************************************************
* G__listshlfunc()
*
*******************************************************************/
void G__listshlfunc(fout)
FILE *fout;
{
#ifndef G__OLDIMPLEMENTATION1911
  if(0 && fout) return;
#endif
}
/*******************************************************************
* G__listshl()
*
*******************************************************************/
void G__listshl(G__temp)
FILE *G__temp;
{
#ifndef G__OLDIMPLEMENTATION1911
  if(0 && G__temp) return;
#endif
}

#ifdef G__TRUEP2F
/******************************************************************
* G__p2f2funchandle()
******************************************************************/
struct G__ifunc_table* G__p2f2funchandle(p2f,p_ifunc,pindex)
void *p2f;
struct G__ifunc_table* p_ifunc;
int* pindex;
{
  struct G__ifunc_table *ifunc;
  int ig15;
  ifunc=p_ifunc;
  do {
    for(ig15=0;ig15<ifunc->allifunc;ig15++) {
      if(
#ifndef G__OLDIMPLEMENTATION1478
	 ifunc->pentry[ig15] && 
#endif
	 ifunc->pentry[ig15]->tp2f==p2f) {
	*pindex = ig15;
	return(ifunc);
      }
#ifndef G__OLDIMPLEMENTATION1846
      if(ifunc->pentry[ig15] && 
	 ifunc->pentry[ig15]->bytecode==p2f) {
	*pindex = ig15;
	return(ifunc);
      }
#endif
    }
  } while((ifunc=ifunc->next)) ;
  *pindex = -1;
  return(ifunc);
}

/******************************************************************
* G__p2f2funcname()
******************************************************************/
char* G__p2f2funcname(p2f)
void *p2f;
{
  struct G__ifunc_table *ifunc;
  int ig15;
  ifunc=G__p2f2funchandle(p2f,G__p_ifunc,&ig15);
  if(ifunc) return(ifunc->funcname[ig15]);
  return((char*)NULL);
}

/******************************************************************
* G__isinterpretedfunc()
******************************************************************/
int G__isinterpretedp2f(p2f)
void *p2f;
{
  struct G__ifunc_table *ifunc;
  int ig15;
  ifunc=G__p2f2funchandle(p2f,G__p_ifunc,&ig15);
  if(ifunc) {
    if(
#ifndef G__OLDIMPLEMENTATION2012
       -1 != ifunc->pentry[ig15]->size
#else
       -1 != ifunc->pentry[ig15]->filenum
#endif
       ) {
      if(ifunc->pentry[ig15]->bytecode) {
	return(G__BYTECODEFUNC);
      }
      else {
	return(G__INTERPRETEDFUNC);
      }
    }
    else {
      if(ifunc->pentry[ig15]->p==ifunc->pentry[ig15]->tp2f) {
	return(G__COMPILEDINTERFACEMETHOD);
      }
      else {
	return(G__COMPILEDTRUEFUNC);
      }
    }
  }
  return(G__UNKNOWNFUNC);
}
#endif


/******************************************************************
* G__pointer2func()
*
* Called by
*   G__getvariable()
*
* Calling fucntion by pointer to function
******************************************************************/
G__value G__pointer2func(obj_p2f,parameter0,parameter1,known3)
G__value *obj_p2f;
char *parameter0 ;
char *parameter1;
int *known3;
{
  G__value result3;
  char result7[G__ONELINE];
  int ig15,ig35;
  struct G__ifunc_table *ifunc;
#ifdef G__SHAREDLIB
  /* G__COMPLETIONLIST *completionlist; */
#endif

  /* get value of pointer to function */
  if(obj_p2f) result3 = *obj_p2f;
  else        result3 = G__getitem(parameter0+1);

#ifndef G__OLDIMPLEMENTATION679
  /* operator overloading */
  if('U'==result3.type && G__PARANORMAL==result3.obj.reftype.reftype) {
    /* int store_tagnum = G__tagnum; */
    /* long store_struct_offset = G__store_struct_offset; */
#ifdef G__ASM
    if(G__asm_noverflow) {
      G__asm_inst[G__asm_cp] = G__PUSHSTROS;
      G__asm_inst[G__asm_cp+1] = G__SETSTROS;
      G__inc_cp_asm(2,0);
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp-2);
	G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp-1);
      }
#endif
    }
#endif
    G__tagnum = result3.tagnum;
    G__store_struct_offset = result3.obj.i;
    parameter1[strlen(parameter1)-1]='\0';
    switch(parameter1[0]) {
    case '[':
      sprintf(result7,"operator[](%s)",parameter1+1);
      break;
    case '(':
      sprintf(result7,"operator()(%s)",parameter1+1);
      break;
    }
    result3 = G__getfunction(result7,known3,G__CALLMEMFUNC);
#ifdef G__ASM
    if(G__asm_noverflow) {
      G__asm_inst[G__asm_cp] = G__POPSTROS;
      G__inc_cp_asm(1,0);
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp-1);
#endif
    }
#endif
    return(result3);
  }
#endif /* ON679 */

#ifdef G__ASM
  G__abortbytecode();
  if(G__no_exec_compile) {
    *known3=1;
    return(G__null);
  }
#endif

  if(result3.obj.i==0) {
    G__fprinterr(G__serr,
	    "Error: Can not access pointer to function 0x%lx from interpreter(1) FILE:%s LINE:%d\n"
	    ,G__int(result3)
	    ,G__ifile.name,G__ifile.line_number);
    return(G__null);
  }

  /* search for interpreted and use precompiled function */
  result7[0]='\0';
#ifdef G__TRUEP2F
  ifunc=G__p2f2funchandle((void*)result3.obj.i,G__p_ifunc,&ig15);
  if(ifunc) sprintf(result7,"%s%s",ifunc->funcname[ig15],parameter1);
#ifdef G__PTR2MEMFUNC
#ifndef G__OLDIMPLEMENTATION1654
  else {
    int itag;
    for(itag=0;itag<G__struct.alltag;itag++) {
      ifunc=G__p2f2funchandle((void*)result3.obj.i,G__struct.memfunc[itag],&ig15);
      if(ifunc && ifunc->staticalloc[ig15]) {
	sprintf(result7,"%s::%s%s",G__fulltagname(itag,1),ifunc->funcname[ig15],parameter1);
	break;
      }
    }
  }
#endif
#endif
#else
  ifunc=G__p_ifunc;
  do {
    for(ig15=0;ig15<ifunc->allifunc;ig15++) {
      if(strcmp(ifunc->funcname[ig15],(char *)result3.obj.i)==0){
	sprintf(result7,"%s%s",(char *)result3.obj.i,parameter1);
      }
    }
  } while(ifunc=ifunc->next) ;
#endif

  if(result7[0]=='\0') {
    /* if interpreted or user precompiled function not found,
     * search for compiled ANSI lib function */
    ig15=0;
    ig35=0;
    while( (((long)(G__completionlist[ig15].name))!=0) &&
	  (ig35==0)) {
      if((long)G__completionlist[ig15].pfunc==result3.obj.i){
	sprintf(result7,"%s%s" ,G__completionlist[ig15].name ,parameter1);
	ig35=1;
      }
      ++ig15;
    }
  }

  if(result7[0]=='\0') {
    /* if interpreted function not found,
     * search for compiled ANSI lib macro(function) by name
     */
    ig15=0;
    ig35=0;
    while( (((long)(G__completionlist[ig15].name))!=0) &&
	  (ig35==0)) {
      if(strcmp(G__completionlist[ig15].name,(char *)result3.obj.i)==0) {
	sprintf(result7,"%s%s",G__completionlist[ig15].name,parameter1);
	ig35=1;
      }
      ++ig15;
    }
  }

  /* appropreate function not found */
  if(result7[0]=='\0') {
    G__fprinterr(G__serr,
	    "Error: Can not access pointer to function 0x%lx from interpreter(2) FILE:%s LINE:%d\n"
	    ,G__int(result3)
	    ,G__ifile.name,G__ifile.line_number);
    return(G__null);
  }

  return(G__getfunction(result7,known3,G__TRYNORMAL));
}

#ifndef G__OLDIMPLEMENTATION648
/******************************************************************
* G__removetagid()
******************************************************************/
void G__removetagid(buf)
char *buf;
{
  int i;
  if(strncmp("class ",buf,6)==0 || strncmp("union ",buf,6)==0) {
    i=6;
    while(buf[i]) { buf[i-6] = buf[i]; ++i; }
    buf[i-6] = '\0';
  }
  else if(strncmp("struct ",buf,7)==0) {
    i=7;
    while(buf[i]) { buf[i-7] = buf[i]; ++i; }
    buf[i-7] = '\0';
  }
  else if(strncmp("enum ",buf,5)==0) {
    i=5;
    while(buf[i]) { buf[i-5] = buf[i]; ++i; }
    buf[i-5] = '\0';
  }
}

/******************************************************************
* G__getp2ftype()
******************************************************************/
int G__getp2ftype(ifunc,ifn)
struct G__ifunc_table *ifunc;
int ifn;
{
  char temp[G__MAXNAME*2],temp1[G__MAXNAME];
  char *p;
  int typenum;
  int i;

  strcpy(temp1, G__type2string(ifunc->type[ifn],ifunc->p_tagtable[ifn]
				 ,ifunc->p_typetable[ifn],ifunc->reftype[ifn]
				 ,ifunc->isconst[ifn]));
  G__removetagid(temp1);

  if(isupper(ifunc->type[ifn])) sprintf(temp,"%s *(*)(",temp1);
  else                          sprintf(temp,"%s (*)(",temp1);
  p = temp + strlen(temp);
  for(i=0;i<ifunc->para_nu[ifn];i++) {
    if(i) *p++ = ',';
    strcpy(temp1,G__type2string(ifunc->para_type[ifn][i]
				,ifunc->para_p_tagtable[ifn][i]
				,ifunc->para_p_typetable[ifn][i]
				,ifunc->para_reftype[ifn][i]
				,ifunc->para_isconst[ifn][i]));
    G__removetagid(temp1);
    strcpy(p,temp1);
    p = temp + strlen(temp);
  }
  strcpy(p,")");

  typenum = G__defined_typename(temp);

  return(typenum);
}
#endif /* ON648 */

/******************************************************************
* G__search_func()
*
* Called by
*   G__getvariable()
*
*  Used to return pointer to function. Cint handles pointer to
* function as pointer to char which contains function name.
******************************************************************/
char *G__search_func(funcname,buf)
char *funcname;
G__value *buf;
{
  int i=0;
  struct G__ifunc_table *ifunc;
#ifdef G__SHAREDLIB
  /* G__COMPLETIONLIST *completionlist; */
  /* int isl; */
#endif

  buf->tagnum = -1;
  buf->typenum = -1;


  /* search for interpreted and user precompied function */
  ifunc=G__p_ifunc;
  do {
    for(i=0;i<ifunc->allifunc;i++) {
      if(
#ifndef G__OLDIMPLEMENTATION1561
	 ifunc->funcname[i] && funcname &&
#endif
	 strcmp(ifunc->funcname[i],funcname)==0) {
#ifdef G__TRUEP2F
	if(
#ifndef G__OLDIMPLEMENTATION2012
	   -1 == ifunc->pentry[i]->size
#else
	   -1 == ifunc->pentry[i]->filenum
#endif
	   ) { /* precompiled function */
#ifndef G__OLDIMPLEMENTATION2191
	  G__letint(buf,'1',(long)ifunc->pentry[i]->tp2f);
#else
	  G__letint(buf,'Q',(long)ifunc->pentry[i]->tp2f);
#endif
	  buf->typenum = G__getp2ftype(ifunc,i);
	}
#ifdef G__ASM_WHOLEFUNC
	else if(ifunc->pentry[i]->bytecode) { /* bytecode function */
#ifndef G__OLDIMPLEMENTATION821
	  G__letint(buf,'Y',(long)ifunc->pentry[i]->tp2f);
#else
	  G__letint(buf,'Q',(long)ifunc->pentry[i]->tp2f);
#endif
	  buf->typenum = G__getp2ftype(ifunc,i);
	}
#endif
	else { /* interpreted function */
	  G__letint(buf,'C',(long)ifunc->pentry[i]->tp2f);
	}
#else
	G__letint(buf,'C',(long)ifunc->funcname[i]);
#endif
	return(ifunc->funcname[i]);
      }
    }
  } while((ifunc=ifunc->next)) ;

#ifdef __CINT__
  if(NULL==G__completionlist) {
    *buf = G__null;
    return(NULL);
  }
#endif

  /* search for compiled ANSI library function */
  i=0;
  while(G__completionlist[i].name!=NULL) {
    if(
#ifndef G__OLDIMPLEMENTATION1561
       funcname &&
#endif
       strcmp(G__completionlist[i].name,funcname)==0) {
      if((long)G__completionlist[i].pfunc!=0) {
#ifndef G__OLDIMPLEMENTATION2191
	G__letint(buf,'1',(long)G__completionlist[i].pfunc);
#else
	G__letint(buf,'Q',(long)G__completionlist[i].pfunc);
#endif
      }
      else {
	G__letint(buf,'C',(long)G__completionlist[i].name);
      }
      return(G__completionlist[i].name);
    }
    i++;
  }

  *buf = G__null;
  return(NULL);

}


/*******************************************************************
* G__search_next_member()
*
* Search variable and function name within the scope
*******************************************************************/

char *G__search_next_member(text,state)
char *text;
int state;
{
  static int list_index,len,index_item  /* ,cbp */;
  static char completionbuf[G__ONELINE];
  char *name,*result;
  int flag=1;
  static struct G__var_array *var;
  static char memtext[G__MAXNAME];
  static int isstruct;
  static G__value buf;
  char *dot,*point,*scope=NULL;
  static struct G__ifunc_table *ifunc;

#ifdef G__NEVER /* G__SHAREDLIB */
  static G__COMPLETIONLIST *completionlist;
  static int isl;
#endif

  /* If this is a new word to complete, initialize now. This
     includes saving the length of TEXT for efficiency, and intializing
     the index variable to 0 */

  /*********************************************************************
   * if G__search_next_member is called first time in the environment
   *********************************************************************/
  if(!state) {

    /*************************************************************
     * check if struct member or not
     *************************************************************/
    strcpy(completionbuf,text);
    dot=strrchr(completionbuf,'.');
    point=G__strrstr(completionbuf,"->");
    scope=G__strrstr(completionbuf,"::");

    /*************************************************************
     * struct member
     *************************************************************/
    if( dot || point || scope) {
      isstruct = 1;

      if(scope>dot && scope>point) {
	strcpy(memtext,scope+2);
	*scope='\0';
	if(dot<point) dot = point+2;
	else if(dot)  ++dot;
	else          dot = completionbuf;
	buf.tagnum = G__defined_tagname(dot,0);
	*scope=':';
	*(scope+2)='\0';
      }	
      else if(dot>point) {
	scope = (char*)NULL;
	strcpy(memtext,dot+1);
	*dot='\0';
	buf = G__calc_internal(completionbuf);
	*dot='.';
	*(dot+1)='\0';
      }
      else {
	scope = (char*)NULL;
	strcpy(memtext,point+2);
	*point='\0';
	buf = G__calc_internal(completionbuf);
	*point='-';
	*(point+2)='\0';
      }
      /**********************************************************
       * if tag can not be identified, no completion candidates.
       **********************************************************/
      if(buf.tagnum<0) {
	return((char *)NULL);
      }
      /**********************************************************
       * if tag can be identified, set tag member table to var.
       **********************************************************/
      else {
	G__incsetup_memvar(buf.tagnum);
	var = G__struct.memvar[buf.tagnum] ;
	len = strlen(memtext);
	G__incsetup_memfunc(buf.tagnum);
	ifunc = G__struct.memfunc[buf.tagnum];
      }
    }

    /************************************************************
     * global or local function name or variable
     ************************************************************/
    else {
      isstruct = 0;
      len = strlen(text);
      if(len==0) return((char *)NULL);  /* bug fix */
    }
    /************************************************************
     * initialization
     ************************************************************/
    /* cbp=0; */
    list_index = 0;
    index_item = 0;
#ifdef G__NEVER /* G__SHAREDLIB */
    completionlist = G__completionlist;
    isl=0;
#endif
  }

  /* Return the next name which partially matchs from the list */
  if(isstruct) {
    while(flag) {
      switch(index_item) {
      case 0: /* struct member */
	G__ASSERT(var);
	if(list_index<var->allvar) {
	  name = var->varnamebuf[list_index] ;
	  break;
	}
	else {
	  if(var->next) {
	    var = var->next;
	    G__ASSERT(var->allvar>0);
	    list_index=0;
	    name = var->varnamebuf[list_index] ;
	    break;
	  }
	  else {
	    index_item++;
	    list_index=0;
	    var = (struct G__var_array *)NULL;
	    /* don't break */
	  }
	}
      case 1: /* member function */
	G__ASSERT(ifunc);
	if(list_index<ifunc->allifunc) {
	  name = ifunc->funcname[list_index];
	  break;
	}
	else {
	  if(ifunc->next) {
	    ifunc=ifunc->next;
	    G__ASSERT(ifunc->allifunc>0);
	    list_index = 0;
	    name = ifunc->funcname[list_index];
	    break;
	  }
	  else {
	    index_item++;
	    list_index=0;
	    ifunc=(struct G__ifunc_table *)NULL;
	    /* do not break */
	  }
	}
      case 2: /* class name */
	if(list_index<G__struct.alltag) {
	  if(scope) {
	    name =(char*)NULL;
	    do {
	      if(G__struct.parent_tagnum[list_index]==buf.tagnum) {
		name = G__struct.name[list_index];
		break;
	      }
	    } while(list_index++<G__struct.alltag) ;
	  }
	  else {
	    name = G__struct.name[list_index];
	  }
	  break;
	}
	else {
	  index_item++;
	  list_index=0;
	}
      case 3: /* typedef name */
	if(list_index<G__newtype.alltype) {
	  if(scope) {
	    name =(char*)NULL;
	    do {
	      if(G__newtype.parent_tagnum[list_index]==buf.tagnum) {
		name = G__newtype.name[list_index];
		break;
	      }
	    } while(list_index++<G__newtype.alltype) ;
	  }
	  else {
	    name = G__newtype.name[list_index];
	  }
	  break;
	}
	else {
	  index_item++;
	  list_index=0;
	}
      default:
	flag=0;
	name=(char *)NULL;
      }
      list_index++;

      if(name!=NULL) {
	if(strncmp(name,memtext,(size_t)len) ==0) {
	  /***************************************************
	   * BUG FIX, return value has to be malloced pointer
	   ***************************************************/
	  switch(index_item) {
	  case 0:
	  case 2:
	  case 3:
	    if(1 || G__PUBLIC==var->access[list_index-1]) {
	      result = (char *)malloc((strlen(completionbuf)+strlen(name)+1));
	      sprintf(result,"%s%s",completionbuf,name);
	      return(result);
	    }
	    break;
	  case 1:
	    if(1 || G__PUBLIC==ifunc->access[list_index-1]) {
	      result = (char *)malloc((strlen(completionbuf)+strlen(name)+2));
	      sprintf(result,"%s%s(",completionbuf,name);
	      return(result);
	    }
	    break;
	  default:
	    return((char *)NULL);
	  }
	}
      }
    }
  }
  else {
    while(flag) {
      switch(index_item) {
      case 0: /* compiled function name */
#ifdef G__NEVER /* G__SHAREDLIB */
	name = completionlist[list_index].name;
	if(name==NULL) {
	  if(isl<G__allsl) {
	    completionlist=(G__COMPLETIONLIST*)G__shl_findsym(&G__sl_handle[isl]
							      ,G__COMPLETION
							      ,TYPE_DATA);
	    list_index = -1;
	    isl++;
	    break;
	  }
	  else {
	    index_item++;
	    list_index=0;
	    ifunc=G__p_ifunc;
	  }
	}
	else break;
#else
	name = G__completionlist[list_index].name;
	if(name==NULL) {
	  index_item++;
	  list_index=0;
	  ifunc=G__p_ifunc;
	}
	else break;
#endif
      case 1: /* interpreted function name */
	G__ASSERT(ifunc);
	if(list_index<ifunc->allifunc) {
	  name = ifunc->funcname[list_index];
	  break;
	}
	else {
	  if(ifunc->next) {
	    ifunc=ifunc->next;
	    if(0==ifunc->allifunc) {
	      index_item++;
	      list_index=0;
	      ifunc=(struct G__ifunc_table *)NULL;
	      /* do not break */
	    }
	    else {
	      list_index = 0;
	      name = ifunc->funcname[list_index];
	      break;
	    }
	  }
	  else {
	    index_item++;
	    list_index=0;
	    ifunc=(struct G__ifunc_table *)NULL;
	    /* do not break */
	  }
	}
	var = &G__global;
      case 2: /* global variables */
	G__ASSERT(var);
	if(list_index<var->allvar) {
	  name = var->varnamebuf[list_index] ;
	  break;
	}
	else {
	  if(var->next) {
	    var = var->next;
	    G__ASSERT(var->allvar>0);
	    list_index=0;
	    name = var->varnamebuf[list_index] ;
	    break;
	  }
	  else {
	    index_item++;
	    list_index=0;
	    var = (struct G__var_array *)NULL;
	    /* don't break */
	  }
	}
	var = G__p_local;
      case 3: /* local variables */
	if(!var) {
	  index_item++;
	  list_index=0;
	}
	else if(list_index<var->allvar) {
	  name = var->varnamebuf[list_index] ;
	  break;
	}
	else {
	  if(var->next) {
	    var = var->next;
	    G__ASSERT(var->allvar>0);
	    list_index=0;
	    name = var->varnamebuf[list_index] ;
	    break;
	  }
	  else {
	    index_item++;
	    list_index=0;
	    var = (struct G__var_array *)NULL;
	    /* don't break */
	  }
	}
      case 4: /* class name */
	if(list_index<G__struct.alltag) {
	  name = G__struct.name[list_index];
	  break;
	}
	else {
	  index_item++;
	  list_index=0;
	}
      case 5: /* template name */
	if(list_index<G__newtype.alltype) {
	  name = G__newtype.name[list_index];
	  break;
	}
	else {
	  index_item++;
	  list_index=0;
	}
      default:
	flag=0;
	name=(char *)NULL;
			break;
      }
      list_index++;

      if(name!=NULL) {
	if(strncmp(name,text,(size_t)len) ==0) {
	  /***************************************************
	   * BUG FIX, return value has to be malloced pointer
	   ***************************************************/
	  switch(index_item) {
	  case 0:
	  case 1:
	    result = (char *)malloc((strlen(name)+2));
	    sprintf(result,"%s(",name);
	    return(result);
	  case 2:
	  case 3:
	  case 4:
	  case 5:
	    result = (char *)malloc((strlen(name)+1));
	    strcpy(result,name);
	    return(result);
	  default:
	    return((char *)NULL);
	  }
	  /***************************************************
	   * BUG FIX, return value has to be malloced pointer
	   ***************************************************/
	}
      }
    }
  }

  /* if no names matched , then return NULL */
  return((char *)NULL);
}

#ifndef G__OLDIMPLEMENTATION1908
#ifdef G__SHAREDLIB
G__SHLHANDLE G__ShlHandle=(G__SHLHANDLE)0;
#ifndef G__OLDIMPLEMENTATION2012
int G__Shlfilenum = -1;
#endif
#endif

/**************************************************************************
 * G__REsetShlHandle()
 **************************************************************************/
void G__ResetShlHandle()
{
#ifdef G__SHAREDLIB
  G__ShlHandle = (G__SHLHANDLE)0;
#ifndef G__OLDIMPLEMENTATION2012
  G__Shlfilenum = -1;
#endif
#endif
}

/**************************************************************************
 * G__GetShlHandle()
 **************************************************************************/
void* G__GetShlHandle()
{
#ifdef G__SHAREDLIB
  return((void*)G__ShlHandle);
#else
  return((void*)0);
#endif
}

#ifndef G__OLDIMPLEMENTATION2012
/**************************************************************************
 * G__GetShlFilenum()
 **************************************************************************/
int G__GetShlFilenum()
{
#ifdef G__SHAREDLIB
#ifndef G__OLDIMPLEMENTATION2012
  return(G__Shlfilenum);
#else
  return(0);
#endif
#else
  return(0);
#endif
}
#endif

/**************************************************************************
 * G__SetShlHandle
 **************************************************************************/
void* G__SetShlHandle(filename)
char *filename;
{
#ifdef G__SHAREDLIB
  int i,isl;
  for(i=0;i<G__nfile;i++) {
    if(0==strcmp(G__srcfile[i].filename,filename)) {
      isl = G__srcfile[i].slindex;
      if(-1!=isl) {
#ifndef G__OLDIMPLEMENTATION2012
	G__Shlfilenum = i;
#endif
	G__ShlHandle = G__sl_handle[isl];
	return((void*)G__ShlHandle);
      }
      else {
	return 0;
      }
    }
  }
#endif
  return 0;
}

/**************************************************************************
 * G__GccNameMangle
 **************************************************************************/
char* G__GccNameMangle(buf,ifunc,ifn)
char* buf;
struct G__ifunc_table *ifunc;
int ifn;
{
  char *funcname = ifunc->funcname[ifn];
  char tmp[4];
  int i;
  tmp[1]=0;
  sprintf(buf,"_Z%d%s",strlen(funcname),funcname);

  for(i=0;i<ifunc->para_nu[ifn];i++) {
    if(isupper(ifunc->para_type[ifn][i])) strcat(buf,"P");
    if(G__PARAREFERENCE==ifunc->para_reftype[ifn][i]) strcat(buf,"R");
    if(G__CONSTVAR&ifunc->para_isconst[ifn][i]) strcat(buf,"K");
    switch(tolower(ifunc->para_type[ifn][i])) {
    case 'c':
    case 's':
    case 'i':
    case 'l':
    case 'f':
    case 'd':
      tmp[0] = ifunc->para_type[ifn][i];
      break;
    case 'b': tmp[0]='h'; break;
    case 'r': tmp[0]='t'; break;
    case 'h': tmp[0]='j'; break;
    case 'k': tmp[0]='m'; break;
    case 'y': tmp[0]='v'; break;
    default:
      break;
    }
    strcat(buf,tmp);
  }
  if(0==ifunc->para_nu[ifn]) strcat(buf,"v");
  return(buf);
}

/**************************************************************************
 * G__Vc6TypeMangle
 * void X, char D, short F, int H, long J, float M, double N, 
 * unsigned char E, unsigned short G, unsigned int I, unsigned long K
 * class name  V[name]@@
 * type * PA, const type* PB, type *const QA, const type* const QB
 * type& AA, const type& AB, 
 **************************************************************************/
char* G__Vc6TypeMangle(type,tagnum,reftype,isconst)
int type;
int tagnum;
int reftype;
int isconst;
{
  static char buf[G__MAXNAME];
  buf[0] = 0;
  if(isupper(type)) {
    if((G__CONSTVAR&isconst) && 
       (G__PCONSTVAR&isconst) &&
       (G__PARAREFERENCE!=reftype)) strcat(buf,"QB");
    else if(0==(G__CONSTVAR&isconst) && 
	    (G__PCONSTVAR&isconst) &&
	    (G__PARAREFERENCE!=reftype)) strcat(buf,"QA");
    else if((G__CONSTVAR&isconst) && 
	    (0==(G__PCONSTVAR&isconst)) &&
	    (G__PARAREFERENCE!=reftype)) strcat(buf,"PB");
    else if((0==(G__CONSTVAR&isconst)) && 
	    (0==(G__PCONSTVAR&isconst)) &&
	    (G__PARAREFERENCE!=reftype)) strcat(buf,"PA");
    else if((G__CONSTVAR&isconst) && 
	    (0==(G__PCONSTVAR&isconst)) &&
	    (G__PARAREFERENCE==reftype)) strcat(buf,"AB");
    else if((0==(G__CONSTVAR&isconst)) && 
	    (0==(G__PCONSTVAR&isconst)) &&
	    (G__PARAREFERENCE==reftype)) strcat(buf,"AA");
    else strcat(buf,"PA");
  }
  switch(tolower(type)) {
  case 'y': strcat(buf,"X"); break;
  case 'c': strcat(buf,"D"); break;
  case 's': strcat(buf,"F"); break;
  case 'i': strcat(buf,"H"); break;
  case 'l': strcat(buf,"J"); break;
  case 'f': strcat(buf,"M"); break;
  case 'd': strcat(buf,"N"); break;
  case 'b': strcat(buf,"E"); break;
  case 'r': strcat(buf,"G"); break;
  case 'h': strcat(buf,"I"); break;
  case 'k': strcat(buf,"K"); break;
  case 'u': strcat(buf,"V"); strcat(buf,G__struct.name[tagnum]); 
    strcat(buf,"@@"); break;
  case 'e': strcpy(buf,"PAU_iobuf@@"); break;
  default:
    break;
  }
  return(buf);
}

/**************************************************************************
 * G__Vc6NameMangle
 * ?[fname]@[tagname]@YA[ret][a1][a2]...@Z
 * ?[fname]@[tagname]@YA[ret]XZ
 **************************************************************************/
char* G__Vc6NameMangle(buf,ifunc,ifn)
char* buf;
struct G__ifunc_table *ifunc;
int ifn;
{
  char *funcname = ifunc->funcname[ifn];
  int i;

  /* funcname */
  sprintf(buf,"?%s@",funcname);

  /* scope */
  if(-1!=ifunc->tagnum) strcat(buf,G__struct.name[ifunc->tagnum]);
  strcat(buf,"@YA");

  /* return type */
  strcat(buf,G__Vc6TypeMangle(ifunc->type[ifn]
			      ,ifunc->p_tagtable[ifn]
			      ,ifunc->reftype[ifn]
			      ,ifunc->isconst[ifn]));

  /* arguments */
  for(i=0;i<ifunc->para_nu[ifn];i++) {
    strcat(buf,G__Vc6TypeMangle(ifunc->para_type[ifn][i]
				,ifunc->para_p_tagtable[ifn][i]
				,ifunc->para_reftype[ifn][i]
				,ifunc->para_isconst[ifn][i]));
  }
  if(0==ifunc->para_nu[ifn]) strcat(buf,"X");
  else strcat(buf,"@");

  /* end */
  strcat(buf,"Z");

  return(buf);
}

/**************************************************************************
 * G__FindSymbol
 **************************************************************************/
void* G__FindSymbol(ifunc,ifn)
struct G__ifunc_table *ifunc;
int ifn;
{
#ifdef G__SHAREDLIB
  char *funcname=ifunc->funcname[ifn];
  void *p2f=0;
  if(G__ShlHandle) {
    char buf[G__ONELINE];

    /* funcname, VC++, GCC, C function */
    p2f = (void*)G__shl_findsym(&G__ShlHandle,funcname,TYPE_PROCEDURE);

    /* _funcname,  BC++, C function */
    if(!p2f) {
      buf[0]='_';
      strcpy(buf+1,funcname);
      p2f = (void*)G__shl_findsym(&G__ShlHandle,buf,TYPE_PROCEDURE);
    }

    /* GCC , C++ function */
    if(!p2f) {
      p2f = (void*)G__shl_findsym(&G__ShlHandle
				  ,G__GccNameMangle(buf,ifunc,ifn)
				  ,TYPE_PROCEDURE);
    }

    /* VC++ , C++ function */
    if(!p2f) {
      p2f = (void*)G__shl_findsym(&G__ShlHandle
				  ,G__Vc6NameMangle(buf,ifunc,ifn)
				  ,TYPE_PROCEDURE);
    }
  }
  return(p2f);
#else
  return((void*)0);
#endif
}

/**************************************************************************
 * G__FindSym
 **************************************************************************/
void* G__FindSym(filename,funcname)
char *filename;
char *funcname;
{
#ifdef G__SHAREDLIB
  void *p2f=0;
  G__SHLHANDLE store_ShlHandle = G__ShlHandle;
  if(!G__SetShlHandle(filename)) return 0;

  p2f = (void*)G__shl_findsym(&G__ShlHandle,funcname,TYPE_PROCEDURE);

  G__ShlHandle = store_ShlHandle;
  return(p2f);
#else
  return((void*)0);
#endif
}
#endif

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
