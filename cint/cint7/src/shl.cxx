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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#if defined(TRU64)
#define __EXTENSIONS__
#endif

#include "common.h"
#include "dllrev.h"
#include "Dict.h"
#include <vector>
#include "Reflex/Builder/TypeBuilder.h"

using namespace Cint::Internal;

/*********************************************************************
* dynamic link library(shared library) enhancement
*********************************************************************/
#ifdef G__SHAREDLIB
/***************************************************
* OSF compliant or SunOS
****************************************************/
#if defined(G__OSFDLL)
typedef void* G__SHLHANDLE;
#include <dlfcn.h>
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
    sharedlib_func=(int (*)())G__shl_findsym(&G__sl_handle[allsl].handle,setupfunc,TYPE_PROCEDURE);   \
    if(sharedlib_func!=NULL) (*sharedlib_func)()

int Cint::Internal::G__ispermanentsl = 0;
std::list<G__DLLINIT>* Cint::Internal::G__initpermanentsl = 0;

#else /* G__SHAREDLIB */

typedef void* G__SHLHANDLE;

#endif /* G__SHAREDLIB */

#ifdef G__SHAREDLIB
short Cint::Internal::G__allsl=0;
#endif

struct G__CintSlHandle {
    G__CintSlHandle(G__SHLHANDLE h = 0, bool p = false) : handle(h),ispermanent(p) {}
    G__SHLHANDLE handle;
 
    bool ispermanent;
 };
 
static std::vector<G__CintSlHandle> G__sl_handle;

#ifdef G__DLL_SYM_UNDERSCORE
static int G__sym_underscore=1;
#else
static int G__sym_underscore=0;
#endif

extern "C" void G__set_sym_underscore(int x) { G__sym_underscore=x; }
extern "C" int G__get_sym_underscore() { return(G__sym_underscore); }

#ifndef __CINT__
extern "C" G__SHLHANDLE G__dlopen(const char *path);
extern "C" void *G__shl_findsym(G__SHLHANDLE *phandle,const char *sym,short type);
extern "C" int G__dlclose(G__SHLHANDLE handle);
#endif

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


/**************************************************************************
* G__loadsystemfile
**************************************************************************/
extern "C" int G__loadsystemfile(const char *filename) 
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


/***********************************************************************
* G__Set_RTLD_NOW() && G__Set_RTLD_LAZY()
*
***********************************************************************/
extern "C" void G__Set_RTLD_NOW() {
#ifdef G__SHAREDLIB
  G__RTLD_flag = G__RTLD_NOW;
#endif
}
extern "C" void G__Set_RTLD_LAZY() {
#ifdef G__SHAREDLIB
  G__RTLD_flag = G__RTLD_LAZY;
#endif
}

/***********************************************************************
* G__dlopen()
*
***********************************************************************/
extern "C" G__SHLHANDLE G__dlopen(const char *path)
{
  G__SHLHANDLE handle;
#ifdef G__SHAREDLIB


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
/****************************************************
* Win32
****************************************************/
# elif defined(G__WIN32)
  handle = LoadLibrary(path);
  if (!handle) {
     void* msg;
     DWORD lasterr = ::GetLastError();
     ::FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | 
        FORMAT_MESSAGE_FROM_SYSTEM,
        NULL,
        lasterr,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR) &msg,
        0, NULL );
     G__fprinterr(G__serr,"%s: %s", path, (char*)msg);
     ::LocalFree(msg);
  }
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


#else /* G__SHAREDLIB */
  handle = (G__SHLHANDLE)NULL;
#endif /* G__SHAREDLIB */

  return(handle);
}

/***********************************************************************
* G__shl_findsym()
*
***********************************************************************/
#if defined(__hpux) || defined(_HIUX_SOURCE)
extern "C" void *G__shl_findsym(G__SHLHANDLE *phandle,const char *sym,short type)
#else
extern "C" void *G__shl_findsym(G__SHLHANDLE *phandle,const char *sym,short /* type */)
#endif
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
  G__StrBuf sym_underscore_sb(G__ONELINE);
  char *sym_underscore = sym_underscore_sb;

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
  dlerror(); dlerror(); /* avoid potential memory leak. */
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
extern "C" int G__dlclose(G__SHLHANDLE handle)
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
#  if defined(G__HPUXCPPDLL) && !defined(__STDCPP__)
  return(cxxshl_unload(handle));
#  else
  return(shl_unload(handle));
#  endif
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

/***********************************************************************
* G__smart_shl_unload()
***********************************************************************/
void Cint::Internal::G__smart_shl_unload(int allsl)
{
  if(G__sl_handle[allsl].handle) {
    if(G__dlclose(G__sl_handle[allsl].handle) == -1) {
      G__fprinterr(G__serr,"Error: Dynamic link library unloading error\n");
    }
    G__sl_handle[allsl].handle=0;
  }
}

#ifdef G__SHAREDLIB
/***********************************************************************
* int G__free_shl_upto()
*
*  Can replace G__free_shl()
*
***********************************************************************/
int Cint::Internal::G__free_shl_upto(int allsl)
{
  /*************************************************************
   * Unload shared library
   *************************************************************/
   int index = G__allsl;

   while((--index)>=allsl) {
      if (!G__sl_handle[index].ispermanent) {
         if(G__dlclose(G__sl_handle[index].handle) == -1) {
            G__fprinterr(G__serr,"Error: Dynamic link library unloading error\n");
         }
         else {
            G__sl_handle[index].handle=0;
         }
      }
   }
   // Now remove the holes
   int offset = 0;
   for(index=allsl;index<G__allsl;++index) {
      if (G__sl_handle[index].handle==0) {
         ++offset;
      } else if (offset) {
         G__sl_handle[index-offset]=G__sl_handle[index];
         G__sl_handle[index].handle = 0;
         G__sl_handle[index].ispermanent = false;
         for(int f=0;f<G__nfile;++f) {
            if (G__srcfile[f].slindex == index) {
               G__srcfile[f].slindex = index-offset;
            }
         }  
      }
   }
   int removed = offset;
   for(index=0;index<removed;++index) {
      G__sl_handle.pop_back();
   }
   G__allsl -= removed;
   return(0);
}

#endif

/**************************************************************************
* G__findsym()
**************************************************************************/
extern "C" void* G__findsym(const char *fname)
{
#ifdef G__SHAREDLIB
  int i;
  void *p;
  for(i=0;i<G__allsl;++i) {
    p = (void*)G__shl_findsym(&G__sl_handle[i].handle,(char*)fname,TYPE_PROCEDURE);
    if(p) return(p);
  }
#endif
  return((void*)NULL);
}


/**************************************************************************
* G__revprint()
**************************************************************************/
int Cint::Internal::G__revprint(FILE *fp)
{
  G__cintrevision(fp);
  G__list_sut(fp);
  return(0);
}


#ifdef G__NEVER
/**************************************************************************
* G__dump_header()
**************************************************************************/
int Cint::Internal::G__dump_header(outfile)
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
extern "C" void G__show_dllrev(const char *shlfile,int (*sharedlib_func)())
{
  G__fprinterr(G__serr,"%s:DLLREV=%d\n",shlfile,(*sharedlib_func)());
  G__fprinterr(G__serr,"  This cint accepts DLLREV=%d~%d and creates %d\n"
          ,G__ACCEPTDLLREV_FROM,G__ACCEPTDLLREV_UPTO
          ,G__CREATEDLLREV);
}

/**************************************************************************
* G__SetCIntApiPointers
*
**************************************************************************/
extern "C" void G__SetCintApiPointers(G__SHLHANDLE *pslhandle,const char *fname)
{
  typedef void (*G__SetCintApiPointers_t)(void* a[G__NUMBER_OF_API_FUNCTIONS]);
  G__SetCintApiPointers_t SetCintApi = \
     (G__SetCintApiPointers_t) G__shl_findsym(pslhandle,fname,TYPE_PROCEDURE);
  if(SetCintApi) {
    void* a[G__NUMBER_OF_API_FUNCTIONS];
#undef G__DECL_API
#undef G__DUMMYTOCHECKFORDUPLICATES
#define G__DECL_API(IDX, RET, NAME, ARGS) \
   a[IDX] = (void*) NAME;
#define G__DUMMYTOCHECKFORDUPLICATES(X)
#include "G__ci_fproto.h"

    (*SetCintApi)(a);
  }
}


/**************************************************************************
* G__shl_load()
*
* Comment:
*  This function can handle both old and new style DLL.
*  This function will modify the input string.
**************************************************************************/
int Cint::Internal::G__shl_load(char *shlfile)
{
  /* int fail = 0; */
  int store_globalcomp;
  char *p;
  char *post;
  G__StrBuf dllid_sb(G__ONELINE);
  char *dllid = dllid_sb;
  int (*sharedlib_func)();
  int error=0,cintdll=0;
  G__StrBuf dllidheader_sb(G__ONELINE);
  char *dllidheader = dllidheader_sb;

#ifdef G__ROOT
  /* this pointer must be set before calling dlopen! */
  if (!G__initpermanentsl) G__initpermanentsl = new std::list<G__DLLINIT>;
  else G__initpermanentsl->clear();
#endif

  // The dlopen might induce (via the autoloader) calls to G__load[system]file
  int store_ispermanentsl = G__ispermanentsl;
  G__ispermanentsl = 0;
  G__sl_handle.push_back( G__CintSlHandle(G__dlopen(shlfile)) );
  G__ispermanentsl = store_ispermanentsl;
  int allsl = G__allsl;
  ++G__allsl;

  if(G__sym_underscore) {
    G__SetCintApiPointers(&G__sl_handle[allsl].handle,"_G__SetCCintApiPointers");
    G__SetCintApiPointers(&G__sl_handle[allsl].handle,"_G__SetCppCintApiPointers");
  }
  else {
    G__SetCintApiPointers(&G__sl_handle[allsl].handle,"G__SetCCintApiPointers");
    G__SetCintApiPointers(&G__sl_handle[allsl].handle,"G__SetCppCintApiPointers");
  }

  if(NULL==G__sl_handle[allsl].handle) {
    if(G__ispragmainclude) {
      if(G__dispmsg>=G__DISPWARN) {
        G__fprinterr(G__serr,"Warning: Can not load Dynamic Link Library %s",shlfile);
        G__printlinenum();
      }
      --G__allsl;
      return(-1);
    }
    else {
      G__shl_load_error(shlfile,"Load Error");
      --G__allsl;
      return(-1);
    }
  }

  /* set file name */
  if(G__ifile.name!=shlfile) strcpy(G__ifile.name,shlfile);

#ifdef G__WIN32
  p = shlfile;
  while(p) {
    p = strchr(p,'/');
    if(p) *p = '\\';
  }
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
    (int (*)())G__shl_findsym(&G__sl_handle[allsl].handle,dllid,TYPE_PROCEDURE);
  if(sharedlib_func && ((*sharedlib_func)()>G__ACCEPTDLLREV_UPTO
     || (*sharedlib_func)()<G__ACCEPTDLLREV_FROM)) {
    G__check_setup_version((*sharedlib_func)(),shlfile);
    error++;
  }
  if(sharedlib_func) {
    cintdll++;
    if(G__asm_dbg) G__show_dllrev(shlfile,sharedlib_func);
  }

  sprintf(dllid,"G__cpp_dllrev%s",dllidheader);
  sharedlib_func=
    (int (*)())G__shl_findsym(&G__sl_handle[allsl].handle,dllid,TYPE_PROCEDURE);
  if(sharedlib_func && ((*sharedlib_func)()>G__ACCEPTDLLREV_UPTO 
     || (*sharedlib_func)()<G__ACCEPTDLLREV_FROM)) {
    G__check_setup_version((*sharedlib_func)(),shlfile);
    error++;
  }
  if(sharedlib_func) {
    cintdll++;
    if(G__asm_dbg) G__show_dllrev(shlfile,sharedlib_func);
  }

  sprintf(dllid,"G__c_dllrev");
  sharedlib_func=
    (int (*)())G__shl_findsym(&G__sl_handle[allsl].handle,dllid,TYPE_PROCEDURE);
  if(sharedlib_func && ((*sharedlib_func)()>G__ACCEPTDLLREV_UPTO
     || (*sharedlib_func)()<G__ACCEPTDLLREV_FROM)) {
    G__check_setup_version((*sharedlib_func)(),shlfile);
    error++;
  }
  if(sharedlib_func) {
    cintdll++;
    if(G__asm_dbg) G__show_dllrev(shlfile,sharedlib_func);
  }

  sprintf(dllid,"G__c_dllrev%s",dllidheader);
  sharedlib_func=
    (int (*)())G__shl_findsym(&G__sl_handle[allsl].handle,dllid,TYPE_PROCEDURE);
  if(sharedlib_func && ((*sharedlib_func)()>G__ACCEPTDLLREV_UPTO
     || (*sharedlib_func)()<G__ACCEPTDLLREV_FROM)) {
    G__check_setup_version((*sharedlib_func)(),shlfile);
    error++;
  }
  if(sharedlib_func) {
    cintdll++;
    if(G__asm_dbg) G__show_dllrev(shlfile,sharedlib_func);
  }

  if(error) {
    G__shl_load_error(shlfile ,"Revision mismatch");
    --G__allsl;
    return(-1);
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
  G__CALL_SETUP("G__cpp_setup_tagtable");
  G__CALL_SETUP("G__cpp_setup_inheritance");
  G__CALL_SETUP("G__cpp_setup_typetable");
  /* G__CALL_SETUP("G__cpp_setup_memvar");
   * G__CALL_SETUP("G__cpp_setup_memfunc"); */
  G__CALL_SETUP("G__cpp_setup_global");
  G__CALL_SETUP("G__cpp_setup_func");
  if(sharedlib_func==NULL) {
    G__CALL_SETUP(dllid);
  }
#ifdef G__ROOT
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
    (int (*)())G__shl_findsym(&G__sl_handle[allsl].handle,G__GLOBALSETUP,TYPE_PROCEDURE);
  if(sharedlib_func!=NULL) {
    (*sharedlib_func)();
  }

  G__prerun=0;
  G__setdebugcond();
  G__globalcomp=store_globalcomp;

  if(G__ispermanentsl) {
    G__DLLINIT initsl = 0;
    //if(!G__initpermanentsl)
      initsl =
        (void (*)())G__shl_findsym(&G__sl_handle[allsl].handle,"G__cpp_setup"
                                   ,TYPE_PROCEDURE); 
    if(!initsl) {
      sprintf(dllid,"G__cpp_setup%s",dllidheader);
      initsl =
        (void (*)())G__shl_findsym(&G__sl_handle[allsl].handle,dllid,TYPE_PROCEDURE); 
    }
    if (initsl) G__initpermanentsl->push_back(initsl);
    G__sl_handle[allsl].ispermanent = true;
  }
  else {
    G__initpermanentsl->clear();
  }


  strcpy(G__ifile.name,"");
  return(allsl);
}
#endif


/*******************************************************************
* G__listshlfunc()
*
*******************************************************************/
void Cint::Internal::G__listshlfunc(FILE * /* fout */)
{
}
/*******************************************************************
* G__listshl()
*
*******************************************************************/
void Cint::Internal::G__listshl(FILE * /* G__temp */)
{
}

#ifdef G__TRUEP2F
/******************************************************************
* G__p2f2funchandle()
******************************************************************/
extern "C" struct G__ifunc_table* G__p2f2funchandle(void *p2f,struct G__ifunc_table* p_ifunc,int* pindex)
{
   ::Reflex::Scope ifunc = G__Dict::GetDict().GetScope(p_ifunc);
  
   if (ifunc) {
      for(size_t ig15 = 0; ig15 < ifunc.FunctionMemberSize(); ++ig15) {
         const ::Reflex::Member func( ifunc.FunctionMemberAt(ig15) );
         if (func) {
            G__RflxFuncProperties *prop = G__get_funcproperties(func);
            if (prop->entry.tp2f==p2f || prop->entry.bytecode==p2f) {
               *pindex = -2; // since we return the id of the function itself and not its scope's id [we had: ig15];
               return (G__ifunc_table*)func.Id();
            }
         }
      }
   }
   *pindex = -1;
   return(0);
}

/******************************************************************
* G__p2f2funcname()
******************************************************************/
extern "C" char* G__p2f2funcname(void *p2f)
{

   int ig15;
   struct G__ifunc_table *ifunc = G__p2f2funchandle(p2f,(G__ifunc_table*)G__p_ifunc.Id(),&ig15);

   if(ifunc) {
      ::Reflex::Member m( G__Dict::GetDict().GetFunction(ifunc, ig15));
      static char buf[G__LONGLINE];
      strcpy(buf,m.Name(::Reflex::SCOPED).c_str());
      return buf;
   }

   for(::Reflex::Type_Iterator iter = ::Reflex::Type::Type_Begin();
      iter != ::Reflex::Type::Type_End(); ++iter) 
   {
      ifunc = G__p2f2funchandle(p2f,(G__ifunc_table*)iter->Id(),&ig15);
      if(ifunc) {
         ::Reflex::Member m( G__Dict::GetDict().GetFunction(ifunc, ig15));
         static char buf[G__LONGLINE];
         sprintf(buf,"%s::%s",iter->Name(::Reflex::SCOPED).c_str(),m.Name().c_str());
         return(buf);
      }
   }
   return((char*)NULL);
}

/******************************************************************
* G__isinterpretedfunc()
******************************************************************/
extern "C" int G__isinterpretedp2f(void *p2f)
{
  struct G__ifunc_table *ifunc;
  int ig15;
  ifunc=G__p2f2funchandle(p2f,(G__ifunc_table*)G__p_ifunc.Id(),&ig15);
  if(ifunc) {
     ::Reflex::Member func( G__Dict::GetDict().GetFunction(ifunc, ig15));
     G__RflxFuncProperties *prop = G__get_funcproperties(func);
    if(
       -1 != prop->entry.size
       ) {
      if(prop->entry.bytecode) {
        return(G__BYTECODEFUNC);
      }
      else {
        return(G__INTERPRETEDFUNC);
      }
    }
    else {
      if(prop->entry.p==prop->entry.tp2f) {
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
* Calling function by pointer to function
******************************************************************/
G__value Cint::Internal::G__pointer2func(G__value *obj_p2f,char *parameter0 ,char *parameter1,int *known3)
{
  G__value result3;
  G__StrBuf result7_sb(G__ONELINE);
  char *result7 = result7_sb;
  int ig15,ig35;
  struct G__ifunc_table *ifunc;
#ifdef G__SHAREDLIB
  /* G__COMPLETIONLIST *completionlist; */
#endif

  /* get value of pointer to function */
  if(obj_p2f) result3 = *obj_p2f;
  else        result3 = G__getitem(parameter0+1);

  /* operator overloading */
  if('U'==G__get_type(G__value_typenum(result3)) && G__PARANORMAL==G__get_reftype(G__value_typenum(result3))) {
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
    G__set_G__tagnum(result3);
    G__store_struct_offset = (char*)result3.obj.i;
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
  ifunc=G__p2f2funchandle((void*)result3.obj.i,(G__ifunc_table*)G__p_ifunc.Id(),&ig15);
  
  if(ifunc) {
     ::Reflex::Member m( G__Dict::GetDict().GetFunction(ifunc, ig15));
     sprintf(result7,"%s%s",m.Name().c_str(),parameter1);
  }
#ifdef G__PTR2MEMFUNC
  else {
     for(::Reflex::Scope_Iterator iter = ::Reflex::Scope::Scope_Begin();
         iter != ::Reflex::Scope::Scope_End(); ++iter) {
        
        ifunc = G__p2f2funchandle((void*)result3.obj.i,(G__ifunc_table*)iter->Id(),&ig15);
        if(ifunc) {
           ::Reflex::Member func( G__Dict::GetDict().GetFunction(ifunc, ig15));
           if (func.IsStatic()) {
                 sprintf(result7,"%s%s",func.Name(::Reflex::SCOPED).c_str(),parameter1);
              break;
           }
        }
     }
  }
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

/******************************************************************
* G__getp2ftype()
******************************************************************/
static ::Reflex::Type G__getp2ftype(const ::Reflex::Member &func)
{
#if 0
   char *p;
   int i;
   
   std::string temp1( func.TypeOf().ReturnType().Name(::Reflex::SCOPED) );
   G__removetagid(temp1);

   std::string temp (temp1);
   if (func.TypeOf().ReturnType().IsPointer()) {
      temp += " *(*)(";
   } else {
      temp += " (*)(";
   }

   p = temp + strlen(temp);
   for(i=0;i<ifunc->para_nu[ifn];i++) {
      if(i) *p++ = ',';
      strcpy(temp1,G__type2string(ifunc->para_type[ifn][i]
                                  ,ifunc->para_p_tagtable[ifn][i]
                                  ,G__get_typenum(ifunc->para_p_typetable[ifn][i])
                                  ,ifunc->para_reftype[ifn][i]
                                  ,ifunc->para_isconst[ifn][i]));
      G__removetagid(temp1);
      strcpy(p,temp1);
      p = temp + strlen(temp);
   }
   strcpy(p,")");
#endif
   Reflex::Type functype( func.TypeOf() );
   if (!functype.IsPointer()) {
      functype = Reflex::PointerBuilder( functype );
   }
   return functype;
}

/******************************************************************
* G__search_func()
*
* Called by
*   G__getvariable()
*
*  Used to return pointer to function. Cint handles pointer to
* function as pointer to char which contains function name.
******************************************************************/
bool Cint::Internal::G__search_func(char *funcname,G__value *buf)
{
  int i=0;
#ifdef G__SHAREDLIB
  /* G__COMPLETIONLIST *completionlist; */
  /* int isl; */
#endif

  G__value_typenum(*buf) = ::Reflex::Type(); 

  /* search for interpreted and user precompiled function */
  ::Reflex::Scope scope(::Reflex::Scope::GlobalScope());
  for(::Reflex::Member_Iterator iter = scope.FunctionMember_Begin();
      iter != scope.FunctionMember_End(); ++iter) {

     if (*iter && iter->Name() == funcname) {
        G__RflxFuncProperties *prop = G__get_funcproperties(*iter);
#ifdef G__TRUEP2F
        if(
           -1 == prop->entry.size
           ) { /* precompiled function */
#ifndef G__OLDIMPLEMENTATION2191
          G__letint(buf,'1',(long)prop->entry.tp2f);
#else
          G__letint(buf,'Q',(long)prop->entry.tp2f);
#endif
          G__value_typenum(*buf) = G__getp2ftype(*iter);
        }
#ifdef G__ASM_WHOLEFUNC
        else if(prop->entry.bytecode) { /* bytecode function */
          G__letint(buf,'Y',(long)prop->entry.tp2f);
          G__value_typenum(*buf) = G__getp2ftype(*iter);
        }
#endif
        else { /* interpreted function */
          G__letint(buf,'C',(long)prop->entry.tp2f);
           G__value_typenum(*buf) = G__getp2ftype(*iter);
        }
#else
        // This is wrong! ... However there is (hopefully) no reason
        // to use the old implementation (2191)
        G__letint(buf,'C',(long)iter->Name().c_str());
        G__value_typenum(*buf) = G__getp2ftype(*iter);
#endif
        return true;
     }
  }

#ifdef __CINT__
  if(NULL==G__completionlist) {
    *buf = G__null;
    return false;
  }
#endif

  /* search for compiled ANSI library function */
  i=0;
  while(G__completionlist[i].name!=NULL) {
    if(
       funcname &&
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
  return false;

}


/*******************************************************************
* G__search_next_member()
*
* Search variable and function name within the scope
*******************************************************************/

char *Cint::Internal::G__search_next_member(char *text,int state)
{
   static unsigned int list_index,len,index_item  /* ,cbp */;
   static char completionbuf[G__ONELINE];
   std::string name;
   char *result;
   int flag=1;
   static ::Reflex::Scope varscope;
   static char memtext[G__MAXNAME];
   static int isstruct;
   static G__value buf;
   char *dot,*point,*scope=NULL;
   static ::Reflex::Scope funcscope;

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
            G__value_typenum(buf) = G__Dict::GetDict().GetType( G__defined_tagname(dot,0) );
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
         if(!G__value_typenum(buf)) {
            return((char *)NULL);
         }
         /**********************************************************
          * if tag can be identified, set tag member table to var.
          **********************************************************/
         else {
            G__incsetup_memvar((G__value_typenum(buf)));
            varscope = G__value_typenum(buf);
            len = strlen(memtext);
            G__incsetup_memfunc((G__value_typenum(buf)));
            funcscope = G__value_typenum(buf); 
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
            G__ASSERT(varscope);
            if(list_index< varscope.DataMemberSize()) {
               name = varscope.DataMemberAt(list_index).Name();
               break;
            }
            else {
               index_item++;
               list_index=0;
               varscope = ::Reflex::Scope(); 
            }
         case 1: /* member function */
            G__ASSERT(funcscope);
            if(list_index<funcscope.FunctionMemberSize()) {
               name = funcscope.FunctionMemberAt(list_index).Name();
               break;
            }
            else {
               index_item++;
               list_index=0;
               funcscope = ::Reflex::Scope();
            }
         case 2: /* class name */
            if(list_index<((unsigned int)G__struct.alltag)) {
               if(scope) {
                  name =(char*)NULL;
                  do {
                     if(G__struct.parent_tagnum[list_index]==G__get_tagnum(G__value_typenum(buf))) {
                        name = G__struct.name[list_index];
                        break;
                     }
                  } while(list_index++<((unsigned int)G__struct.alltag)) ;
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
            {
               static ::Reflex::Scope scopetype;
               static ::Reflex::Type_Iterator iter;

               ::Reflex::Type typedf;
               if (!scopetype) {
                  if(scope) {
                     scopetype = G__value_typenum(buf);
                     iter = scopetype.SubType_Begin();
                  } else {
                     scopetype = ::Reflex::Scope::GlobalScope();
                  }
               }

               name =(char*)NULL;
               do {
                  if ( (*iter).IsTypedef() ) {
                     name = (*iter).Name();
                     break;
                  }
               } while( (++iter) != scopetype.SubType_End() );

               if ( iter == scopetype.SubType_End() ){
                  scopetype = ::Reflex::Type();
                  index_item++;
                  list_index=0;
               }
            }
         default:
            flag=0;
            name=(char *)NULL;
         }
         list_index++;

         if(name.size()) {
            if(strncmp(name.c_str(),memtext,(size_t)len) ==0) {
               /***************************************************
                * BUG FIX, return value has to be malloced pointer
                ***************************************************/
               switch(index_item) {
               case 0:
               case 2:
               case 3:
                  if(1 || varscope.DataMemberAt(list_index-1).IsPublic()) {
                     result = (char *)malloc((strlen(completionbuf)+name.size()+1));
                     sprintf(result,"%s%s",completionbuf,name.c_str());
                     return(result);
                  }
                  break;
               case 1:
                  if(1 || funcscope.FunctionMemberAt(list_index-1).IsPublic()) {
                     result = (char *)malloc((strlen(completionbuf)+name.size()+2));
                     sprintf(result,"%s%s(",completionbuf,name.c_str());
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
                  completionlist=(G__COMPLETIONLIST*)G__shl_findsym(&G__sl_handle[isl].handle
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
            if(name.size()==0) {
               index_item++;
               list_index=0;
               funcscope = G__p_ifunc;
            }
            else break;
#endif
         case 1: /* interpreted function name */
            G__ASSERT(funcscope);
            if(list_index<funcscope.FunctionMemberSize()) {
               name = funcscope.FunctionMemberAt(list_index).Name();
               break;
            }
            else {
               index_item++;
               list_index=0;
               funcscope = ::Reflex::Scope(); 
               /* do not break */
            }
            varscope = ::Reflex::Scope::GlobalScope();
         case 2: /* global variables */
            G__ASSERT(varscope);
            if(list_index<varscope.DataMemberSize()) {
               name = varscope.DataMemberAt(list_index).Name();
               break;
            }
            else {
               index_item++;
               list_index=0;
               varscope = ::Reflex::Scope(); 
            }
            varscope = G__p_local;
         case 3: /* local variables */
            if(!varscope) {
               index_item++;
               list_index=0;
            }
            else if(list_index<varscope.DataMemberSize()) {
               name =  varscope.DataMemberAt(list_index).Name();
               break;
            }
            else {
               index_item++;
               list_index=0;
               varscope = ::Reflex::Scope();
               /* don't break */
            }
         case 4: /* class name */
            if(list_index<((unsigned int)G__struct.alltag)) {
               name = G__struct.name[list_index];
               break;
            }
            else {
               index_item++;
               list_index=0;
            }
         case 5: /* template name */
            {
               ::Reflex::Type typedf =
                  G__Dict::GetDict().GetTypedef(list_index);
               if (typedf) {
                  name = typedf.Name(::Reflex::SCOPED);
                  break;
               }
               else {
                  index_item++;
                  list_index=0;
               }
            }
         default:
            flag=0;
            name=(char *)NULL;
            break;
         }
         list_index++;

         if(name.size()) {
            if(strncmp(name.c_str(),text,(size_t)len) ==0) {
               /***************************************************
                * BUG FIX, return value has to be malloced pointer
                ***************************************************/
               switch(index_item) {
               case 0:
               case 1:
                  result = (char *)malloc((name.size()+2));
                  sprintf(result,"%s(",name.c_str());
                  return(result);
               case 2:
               case 3:
               case 4:
               case 5:
                  result = (char *)malloc((name.size()+1));
                  strcpy(result,name.c_str());
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

#ifdef G__SHAREDLIB
G__SHLHANDLE G__ShlHandle=(G__SHLHANDLE)0;
int G__Shlfilenum = -1;
#endif

/**************************************************************************
 * G__REsetShlHandle()
 **************************************************************************/
void Cint::Internal::G__ResetShlHandle()
{
#ifdef G__SHAREDLIB
  G__ShlHandle = (G__SHLHANDLE)0;
  G__Shlfilenum = -1;
#endif
}

/**************************************************************************
 * G__GetShlHandle()
 **************************************************************************/
void* Cint::Internal::G__GetShlHandle()
{
#ifdef G__SHAREDLIB
  return((void*)G__ShlHandle);
#else
  return((void*)0);
#endif
}

/**************************************************************************
 * G__GetShlFilenum()
 **************************************************************************/
int Cint::Internal::G__GetShlFilenum()
{
#ifdef G__SHAREDLIB
  return(G__Shlfilenum);
#else
  return(0);
#endif
}

/**************************************************************************
 * G__SetShlHandle
 **************************************************************************/
void* Cint::Internal::G__SetShlHandle(char *filename)
{
#ifdef G__SHAREDLIB
  int i,isl;
  for(i=0;i<G__nfile;i++) {
    if(0==strcmp(G__srcfile[i].filename,filename)) {
      isl = G__srcfile[i].slindex;
      if(-1!=isl) {
        G__Shlfilenum = i;
        G__ShlHandle = G__sl_handle[isl].handle;
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
static char* G__GccNameMangle(char* buf,const ::Reflex::Member &ifunc)
{
   std::string funcname( ifunc.Name() );
  char tmp[4];
  unsigned int i;
  tmp[1]=0;
  sprintf(buf,"_Z%lu%s",(unsigned long)funcname.length(),funcname.c_str());

  ::Reflex::Type type ( ifunc.TypeOf() );
  for(i=0;i<ifunc.FunctionParameterSize();++i) {
     ::Reflex::Type ptype( type.FunctionParameterAt(i) );
     if(isupper(G__get_type(ptype))) strcat(buf,"P");
     if(G__PARAREFERENCE==G__get_reftype(ptype)) strcat(buf,"R");
     if(G__CONSTVAR&G__get_isconst(ptype)) strcat(buf,"K");
     switch(tolower(G__get_type(ptype))) { 
     case 'c':
     case 's':
     case 'i':
     case 'l':
     case 'f':
     case 'd':
        tmp[0] = G__get_type(ptype);
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
  if(0==type.FunctionParameterSize()) strcat(buf,"v");
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
static char* G__Vc6TypeMangle(int type,int tagnum,int reftype,int isconst)
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
static char* G__Vc6NameMangle(char* buf,const ::Reflex::Member &ifunc) 
{
  const char *funcname =  ifunc.Name().c_str();
  unsigned int i;

  /* funcname */
  sprintf(buf,"?%s@",funcname);

  /* scope */
  if(-1!=G__get_tagnum(ifunc.DeclaringScope())) strcat(buf,G__struct.name[G__get_tagnum(ifunc.DeclaringScope())]);
  strcat(buf,"@YA");

  /* return type */
  strcat(buf,G__Vc6TypeMangle(    G__get_type(ifunc.TypeOf().ReturnType())
                              , G__get_tagnum(ifunc.TypeOf().ReturnType())
                              ,G__get_reftype(ifunc.TypeOf().ReturnType())
                              ,G__get_isconst(ifunc.TypeOf().ReturnType())));

  /* arguments */
  for(i=0;i<ifunc.TypeOf().FunctionParameterSize();++i) {
     ::Reflex::Type ptype( ifunc.TypeOf().FunctionParameterAt(i) );
     strcat(buf,G__Vc6TypeMangle(G__get_type(ptype)
                                 ,G__get_tagnum(ptype)
                                 ,G__get_reftype(ptype)
                                 ,G__get_isconst(ptype)));
  }
  if(0==ifunc.TypeOf().FunctionParameterSize()) strcat(buf,"X");
  else strcat(buf,"@");

  /* end */
  strcat(buf,"Z");

  return(buf);
}

/**************************************************************************
 * G__FindSymbol
 **************************************************************************/
void* Cint::Internal::G__FindSymbol(const ::Reflex::Member &ifunc)
{
#ifdef G__SHAREDLIB
   const char *funcname=ifunc.Name().c_str();
   void *p2f=0;
   if(G__ShlHandle) {
      G__StrBuf buf_sb(G__ONELINE);
      char *buf = buf_sb;

      /* funcname, VC++, GCC, C function */
      p2f = (void*)G__shl_findsym(&G__ShlHandle,(char*)funcname,TYPE_PROCEDURE);

      /* _funcname,  BC++, C function */
      if(!p2f) {
         buf[0]='_';
         strcpy(buf+1,funcname);
         p2f = (void*)G__shl_findsym(&G__ShlHandle,buf,TYPE_PROCEDURE);
      }

      /* GCC , C++ function */
      if(!p2f) {
         p2f = (void*)G__shl_findsym(&G__ShlHandle
                                     ,G__GccNameMangle(buf,ifunc)
                                     ,TYPE_PROCEDURE);
      }

      /* VC++ , C++ function */
      if(!p2f) {
         p2f = (void*)G__shl_findsym(&G__ShlHandle
                                     ,G__Vc6NameMangle(buf,ifunc)
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
void* Cint::Internal::G__FindSym(const char *filename,const char *funcname)
{
#ifdef G__SHAREDLIB
  void *p2f=0;
  G__SHLHANDLE store_ShlHandle = G__ShlHandle;
  if(!G__SetShlHandle((char*)filename)) return 0;

  p2f = (void*)G__shl_findsym(&G__ShlHandle,(char*)funcname,TYPE_PROCEDURE);

  G__ShlHandle = store_ShlHandle;
  return(p2f);
#else
  return((void*)0);
#endif
}

static const char *G__dladdr(void (*func)())
{
#ifdef G__SHAREDLIB
	// Wrapper around dladdr (and friends)
#if defined(__CYGWIN__) && defined(__GNUC__)
   return 0;
#elif defined(G__WIN32)
   MEMORY_BASIC_INFORMATION mbi;
   if (!VirtualQuery (func, &mbi, sizeof (mbi)))
   {
      return 0;
   }
   
   HMODULE hMod = (HMODULE) mbi.AllocationBase;
   static char moduleName[MAX_PATH];
   
   if (!GetModuleFileNameA (hMod, moduleName, sizeof (moduleName)))
   {
      return 0;
   }
   return moduleName;
#else
   Dl_info info;
   if (dladdr((void*)func,&info)==0) {
      // Not in a known share library, let's give up
      return 0;
   } else {
      //fprintf(stdout,"Found address in %s\n",info.dli_fname);
      return info.dli_fname;
   }
#endif 

#else // G__SHAREDLIB
   return 0;
#endif //G__SHAREDLIB
}

// G__RegisterLibrary
int Cint::Internal::G__RegisterLibrary(void (*func)()) 
{
   // This function makes sure that the library that contains 'func' is
   // known to have been loaded by the CINT system and return 
   // the filenum (i.e. index in G__srcfile).
   
   const char *libname = G__dladdr( func );
   if (libname && libname[0]) {
      size_t lenLibName = strlen(libname);
      G__StrBuf sbLibName(lenLibName);
      strcpy(sbLibName, libname);
      // remove soversion at the end: .12.34
      size_t cutat = lenLibName - 1;
      while (cutat > 2) {
         if (!isdigit(sbLibName[cutat])) { break; }
         // Skip first digit
         --cutat;
         // Skip 2nd digit if any
         if (isdigit(sbLibName[cutat])) { --cutat; }
         if (sbLibName[cutat] != '.') { break;  }
         // Skip period
         --cutat;
         sbLibName[cutat + 1] = 0;
      }
      return G__register_sharedlib( sbLibName );
   }
   return -1;
}   

// G__UnregisterLibrary
int Cint::Internal::G__UnregisterLibrary(void (*func)()) {
   // This function makes sure that the library that contains 'func' is
   // known to have been laoded by the CINT system.
   
   const char *libname = G__dladdr( func );
   if (libname) {
      return G__unregister_sharedlib( libname );
   }
   return 0;
}   

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
