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
#include <vector>
#include <string>

extern "C" {
  void G__set_alloclockfunc(void(*foo)());
  void G__set_allocunlockfunc(void(*foo)());
}

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

std::list<G__DLLINIT>* G__initpermanentsl = 0;

#else /* G__SHAREDLIB */

typedef void* G__SHLHANDLE;

#endif /* G__SHAREDLIB */

struct G__CintSlHandle {
   G__CintSlHandle(G__SHLHANDLE h = 0, bool p = false) : handle(h),ispermanent(p) {}
   G__SHLHANDLE handle;

   bool ispermanent;
};

std::vector<G__CintSlHandle> G__sl_handle; // [G__MAX_SL];

extern "C" {

short G__allsl=0;

#ifdef G__DLL_SYM_UNDERSCORE
static int G__sym_underscore=1;
#else
static int G__sym_underscore=0;
#endif

void G__set_sym_underscore(int x) { G__sym_underscore=x; }
int G__get_sym_underscore() { return(G__sym_underscore); }


#ifndef __CINT__
G__SHLHANDLE G__dlopen G__P((const char *path));
void *G__shl_findsym G__P((G__SHLHANDLE *phandle,const char *sym,short type));
int G__dlclose G__P((G__SHLHANDLE handle));
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


int G__ispermanentsl = 0;

/**************************************************************************
* G__loadsystemfile
**************************************************************************/
int G__loadsystemfile(const char *filename) 
{
  int result;
  size_t len = strlen(filename);
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

extern int (*G__p_class_autoloading)(char*, char*);
int (*G__store_p_class_autoloading)(char*, char*);

typedef std::vector<std::pair<std::string,std::string> > G__autoload_requests_type;
static G__autoload_requests_type *G__autoload_requests;

int G__dlopen_class_autoloading_intercept(char* classname, char *libname) 
{
   G__autoload_requests->push_back( make_pair( std::string(classname), std::string(libname) ) );
   return 0;
}

/***********************************************************************
* G__dlopen()
*
***********************************************************************/
G__SHLHANDLE G__dlopen(const char *path)
{
   G__SHLHANDLE handle;
#ifdef G__SHAREDLIB
   
   // Intercept and delay any potential autoloading.
   G__autoload_requests_type requests;
   if (G__store_p_class_autoloading == 0) {
      G__store_p_class_autoloading = G__p_class_autoloading;
      G__set_class_autoloading_callback( G__dlopen_class_autoloading_intercept );
      G__autoload_requests = &requests;
   }
   
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
# else /* non of above */
   handle = (G__SHLHANDLE)NULL;
# endif
   
   // Replay the delayed autoloading.
   if ( &requests == G__autoload_requests) {
      G__set_class_autoloading_callback( G__store_p_class_autoloading );
      G__store_p_class_autoloading = 0;
      G__autoload_requests = 0;
      G__autoload_requests_type::const_iterator end( requests.end() );
      for(G__autoload_requests_type::const_iterator iter = requests.begin(); iter != end; ++iter )
      {
         G__p_class_autoloading( (char*)iter->first.c_str(), (char*)iter->second.c_str() );
      }
   }
      
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
void *G__shl_findsym(G__SHLHANDLE *phandle,const char *sym,short type)
#else
void *G__shl_findsym(G__SHLHANDLE *phandle,const char *sym,short /* type */)
#endif
{
  void *func = (void*)NULL;

  G__FastAllocString sym_underscore(strlen(sym) + 2 /* underscore */ + 5 /* __xv */);

  if(G__sym_underscore) {
    sym_underscore[0]='_';
    strcpy(sym_underscore+1,sym); // Okay we allocated enough space
  }
  else {
    sym_underscore = sym;
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
int G__dlclose(G__SHLHANDLE handle)
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
void G__smart_shl_unload(int allsl)
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
int G__free_shl_upto(short allsl)
{
  /*************************************************************
   * Unload shared library
   *************************************************************/
   short index = G__allsl;

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
   short offset = 0;
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
   short removed = offset;
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
void* G__findsym(const char *fname)
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
int G__revprint(FILE *fp)
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
void G__show_dllrev(const char *shlfile,int (*sharedlib_func)())
{
  G__fprinterr(G__serr,"%s:DLLREV=%d\n",shlfile,(*sharedlib_func)());
  G__fprinterr(G__serr,"  This cint accepts DLLREV=%d~%d and creates %d\n"
          ,G__ACCEPTDLLREV_FROM,G__ACCEPTDLLREV_UPTO
          ,G__CREATEDLLREV);
}

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
        void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
        void*, void*));
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
        void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,
        void*));
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
        void*,void*,void*,void*,void*,void*,void*,void*,void*,void*));
#endif

/**************************************************************************
* G__SetCIntApiPointers
*
**************************************************************************/
void G__SetCintApiPointers(G__SHLHANDLE *pslhandle,const char *fname)
{
  G__SetCintApiPointers_t SetCintApi;
  SetCintApi = (G__SetCintApiPointers_t)
    G__shl_findsym(pslhandle,fname,TYPE_PROCEDURE);
  if(SetCintApi)
    (*SetCintApi)(
        (void*)G__main,
        (void*)G__setothermain,
        (void*)G__getnumbaseclass,
        (void*)G__setnewtype,
        (void*)G__setnewtypeindex,
        (void*)G__resetplocal,
        (void*)G__getgvp,
        (void*)G__resetglobalenv,
        (void*)G__lastifuncposition,
        (void*)G__resetifuncposition,
        (void*)G__setnull,
        (void*)G__getstructoffset,
        (void*)G__getaryconstruct,
        (void*)G__gettempbufpointer,
        (void*)G__setsizep2memfunc,
        (void*)G__getsizep2memfunc,
        (void*)G__get_linked_tagnum,
        (void*)G__tagtable_setup,
        (void*)G__search_tagname,
        (void*)G__search_typename,
        (void*)G__defined_typename,
        (void*)G__tag_memvar_setup,
        (void*)G__memvar_setup,
        (void*)G__tag_memvar_reset,
        (void*)G__tag_memfunc_setup,
        (void*)G__memfunc_setup,
        (void*)G__memfunc_next,
        (void*)G__memfunc_para_setup,
        (void*)G__tag_memfunc_reset,
        (void*)G__letint,
        (void*)G__letdouble,
        (void*)G__store_tempobject,
        (void*)G__inheritance_setup,
        (void*)G__add_compiledheader,
        (void*)G__add_ipath,
        (void*)G__add_macro,
        (void*)G__check_setup_version,
        (void*)G__int,
        (void*)G__double,
        (void*)G__calc,
        (void*)G__loadfile,
        (void*)G__unloadfile,
        (void*)G__init_cint,
        (void*)G__scratch_all,
        (void*)G__setdouble,
        (void*)G__setint,
        (void*)G__stubstoreenv,
        (void*)G__stubrestoreenv,
        (void*)G__getstream,
        (void*)G__type2string,
        (void*)G__alloc_tempobject,
        (void*)G__set_p2fsetup,
        (void*)G__free_p2fsetup,
        (void*)G__genericerror,
        (void*)G__tmpnam,
        (void*)G__setTMPDIR,
        (void*)G__setPrerun,
        (void*)G__readline,
        (void*)G__getFuncNow,
        (void*)G__getIfileFp,
        (void*)G__incIfileLineNumber,
        (void*)G__setReturn,
        (void*)G__getPrerun,
        (void*)G__getDispsource,
        (void*)G__getSerr,
        (void*)G__getIsMain,
        (void*)G__setIsMain,
        (void*)G__setStep,
        (void*)G__getStepTrace,
        (void*)G__setDebug,
        (void*)G__getDebugTrace,
        (void*)G__set_asm_noverflow,
        (void*)G__get_no_exec,
        (void*)G__get_no_exec_compile,
        (void*)G__setdebugcond,
        (void*)G__init_process_cmd,
        (void*)G__process_cmd,
        (void*)G__pause,
        (void*)G__input,
        (void*)G__split,
        (void*)G__getIfileLineNumber,
        (void*)G__addpragma,
        (void*)G__add_setup_func,
        (void*)G__remove_setup_func,
        (void*)G__setgvp,
        (void*)G__set_stdio_handle,
        (void*)G__setautoconsole,
        (void*)G__AllocConsole,
        (void*)G__FreeConsole,
        (void*)G__getcintready,
        (void*)G__security_recover,
        (void*)G__breakkey,
        (void*)G__stepmode,
        (void*)G__tracemode,
        (void*)G__getstepmode,
        (void*)G__gettracemode,
        (void*)G__printlinenum,
        (void*)G__search_typename2,
        (void*)G__set_atpause,
        (void*)G__set_aterror,
        (void*)G__p2f_void_void,
        (void*)G__setglobalcomp,
        (void*)G__getmakeinfo,
        (void*)G__get_security_error,
        (void*)G__map_cpp_name,
        (void*)G__Charref,
        (void*)G__Shortref,
        (void*)G__Intref,
        (void*)G__Longref,
        (void*)G__UCharref,
        (void*)G__UShortref,
        (void*)G__UIntref,
        (void*)G__ULongref,
        (void*)G__Floatref,
        (void*)G__Doubleref,
        (void*)G__loadsystemfile,
        (void*)G__set_ignoreinclude,
        (void*)G__exec_tempfile,
        (void*)G__exec_text,
        (void*)G__lasterror_filename,
        (void*)G__lasterror_linenum,
        (void*)G__va_arg_put
#ifndef G__OLDIMPLEMENTATION1546
        ,(void*)G__load_text
        ,(void*)G__set_emergencycallback
#endif
#ifndef G__OLDIMPLEMENTATION1485
        ,(void*)G__set_errmsgcallback
#endif
   ,(void*)G__get_ifile
        ,(void*)G__set_alloclockfunc
        ,(void*)G__set_allocunlockfunc
        );
}


/**************************************************************************
* G__shl_load()
*
* Comment:
*  This function can handle both old and new style DLL.
**************************************************************************/
int G__shl_load(char *shlfile)
{
  /* int fail = 0; */
  int store_globalcomp;
  char *p;
  char *post;
  int (*sharedlib_func)();
  int error=0,cintdll=0;

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
      G__sl_handle.pop_back();
      --G__allsl;
      return(-1);
    }
  }

  /* set file name */
  if(G__ifile.name!=shlfile) G__strlcpy(G__ifile.name,shlfile,G__MAXFILENAME);

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

  size_t lendllidheader = strlen(p) + 1;
  G__FastAllocString dllidheader(lendllidheader);
  dllidheader = p;
  post = strchr(dllidheader,'.');
  if(post)  *post = '\0';

  G__FastAllocString dllid(lendllidheader); {
     dllid = "G__cpp_dllrev";
  }
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

  dllid.Format("G__cpp_dllrev%s",dllidheader());
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

  dllid = "G__c_dllrev";
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

  dllid.Format("G__c_dllrev%s",dllidheader());
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


  dllid.Format("G__cpp_setup%s",dllidheader());
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

  dllid.Format("G__c_setup%s",dllidheader());
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
     dllid.Format("G__get_sizep2memfunc%s",dllidheader());
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
      initsl = (void (*)())G__shl_findsym(&G__sl_handle[allsl].handle,"G__cpp_setup"
                                          ,TYPE_PROCEDURE); 
      if(!initsl) {
         dllid.Format("G__cpp_setup%s",dllidheader());
         initsl = (void (*)())G__shl_findsym(&G__sl_handle[allsl].handle,dllid,TYPE_PROCEDURE); 
      }
      if (initsl) G__initpermanentsl->push_back(initsl);
      G__sl_handle[allsl].ispermanent = true;
   }
   
  G__ifile.name[0] = '\0';
  return(allsl);
}
#endif


/*******************************************************************
* G__listshlfunc()
*
*******************************************************************/
void G__listshlfunc(FILE * /* fout */)
{
}
/*******************************************************************
* G__listshl()
*
*******************************************************************/
void G__listshl(FILE * /* G__temp */)
{
}

#ifdef G__TRUEP2F
/******************************************************************
* G__p2f2funchandle_internal()
******************************************************************/
struct G__ifunc_table_internal* G__p2f2funchandle_internal(void *p2f,struct G__ifunc_table_internal* ifunc,int* pindex)
{
  int ig15;
  do {
    for(ig15=0;ig15<ifunc->allifunc;ig15++) {
      if(
         ifunc->pentry[ig15] && 
         ifunc->pentry[ig15]->tp2f==p2f) {
        *pindex = ig15;
        return ifunc;
      }
      if(ifunc->pentry[ig15] && 
         ifunc->pentry[ig15]->bytecode==p2f) {
        *pindex = ig15;
        return ifunc;
      }
    }
  } while((ifunc=ifunc->next)) ;
  *pindex = -1;
  return ifunc;
}

/******************************************************************
* G__p2f2funchandle()
******************************************************************/
struct G__ifunc_table* G__p2f2funchandle(void *p2f,struct G__ifunc_table* p_ifunc,int* pindex)
{
   return G__get_ifunc_ref(G__p2f2funchandle_internal(p2f, G__get_ifunc_internal(p_ifunc), pindex));
}

/******************************************************************
* G__p2f2funcname()
******************************************************************/
char* G__p2f2funcname(void *p2f)
{
  int tagnum;
  struct G__ifunc_table_internal *ifunc;
  int ig15;
  ifunc=G__p2f2funchandle_internal(p2f,G__p_ifunc,&ig15);
  if(ifunc) return(ifunc->funcname[ig15]);

  for(tagnum=0;tagnum<G__struct.alltag;tagnum++) {
    ifunc=G__p2f2funchandle_internal(p2f,G__struct.memfunc[tagnum],&ig15);
    if(ifunc) {
       static G__FastAllocString buf(G__LONGLINE);
       buf.Format("%s::%s",G__fulltagname(tagnum,1),ifunc->funcname[ig15]);
      return(buf);
    }
  }
  return((char*)NULL);
}

/******************************************************************
* G__isinterpretedfunc()
******************************************************************/
int G__isinterpretedp2f(void *p2f)
{
  struct G__ifunc_table_internal *ifunc;
  int ig15;
  ifunc=G__p2f2funchandle_internal(p2f,G__p_ifunc,&ig15);
  if(ifunc) {
    if(
       -1 != ifunc->pentry[ig15]->size
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
* Calling function by pointer to function
******************************************************************/
G__value G__pointer2func(G__value *obj_p2f,char *parameter0 ,char *parameter1,int *known3)
{
  G__value result3;
  G__FastAllocString result7(G__ONELINE);
  int ig15,ig35;
  struct G__ifunc_table_internal *ifunc;
#ifdef G__SHAREDLIB
  /* G__COMPLETIONLIST *completionlist; */
#endif

  /* get value of pointer to function */
  if(obj_p2f) result3 = *obj_p2f;
  else        result3 = G__getitem(parameter0+1);

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
       result7.Format("operator[](%s)",parameter1+1);
      break;
    case '(':
       result7.Format("operator()(%s)",parameter1+1);
      break;
    }
    result3 = G__getfunction(result7(),known3,G__CALLMEMFUNC);
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
  ifunc=G__p2f2funchandle_internal((void*)result3.obj.i,G__p_ifunc,&ig15);
  if(ifunc) result7.Format("%s%s",ifunc->funcname[ig15],parameter1);
#ifdef G__PTR2MEMFUNC
  else {
    int itag;
    for(itag=0;itag<G__struct.alltag;itag++) {
      ifunc=G__p2f2funchandle_internal((void*)result3.obj.i,G__struct.memfunc[itag],&ig15);
      if(ifunc && ifunc->staticalloc[ig15]) {
         result7.Format("%s::%s%s",G__fulltagname(itag,1),ifunc->funcname[ig15],parameter1);
        break;
      }
    }
  }
#endif
#else
  ifunc=G__p_ifunc;
  do {
    for(ig15=0;ig15<ifunc->allifunc;ig15++) {
      if(strcmp(ifunc->funcname[ig15],(char *)result3.obj.i)==0){
         result7.Format("%s%s",(char *)result3.obj.i,parameter1);
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
         result7.Format("%s%s" ,G__completionlist[ig15].name ,parameter1);
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
         result7.Format("%s%s",G__completionlist[ig15].name,parameter1);
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
* G__removetagid()
******************************************************************/
static void G__removetagid(G__FastAllocString &buf)
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
int G__getp2ftype(struct G__ifunc_table_internal *ifunc,int ifn)
{
   G__FastAllocString temp(G__MAXNAME*2);
   G__FastAllocString temp1(G__MAXNAME);
   size_t p;
   int typenum;
   int i;

   temp1 = G__type2string(ifunc->type[ifn],ifunc->p_tagtable[ifn]
                                 ,ifunc->p_typetable[ifn],ifunc->reftype[ifn]
                                 ,ifunc->isconst[ifn]);
   G__removetagid(temp1);

   if(isupper(ifunc->type[ifn])) temp.Format("%s *(*)(",temp1());
   else                          temp.Format("%s (*)(",temp1());
   p = strlen(temp);
   for(i=0;i<ifunc->para_nu[ifn];i++) {
      if(i) temp[p++] = ',';
      temp1 = G__type2string(ifunc->param[ifn][i]->type
                             ,ifunc->param[ifn][i]->p_tagtable
                             ,ifunc->param[ifn][i]->p_typetable
                             ,ifunc->param[ifn][i]->reftype
                             ,ifunc->param[ifn][i]->isconst);
      G__removetagid(temp1);
      temp.Replace(p,temp1);
      p = strlen(temp);
   }
   temp.Replace(p,")");

   typenum = G__defined_typename(temp);

   return(typenum);
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
char *G__search_func(const char *funcname,G__value *buf)
{
  int i=0;
  struct G__ifunc_table_internal *ifunc;
#ifdef G__SHAREDLIB
  /* G__COMPLETIONLIST *completionlist; */
  /* int isl; */
#endif

  buf->tagnum = -1;
  buf->typenum = -1;


  /* search for interpreted and user precompied function */
  ifunc= &G__ifunc;
  do {
    for(i=0;i<ifunc->allifunc;i++) {
      if(
         ifunc->funcname[i] && funcname &&
         strcmp(ifunc->funcname[i],funcname)==0) {
#ifdef G__TRUEP2F
        if(
           -1 == ifunc->pentry[i]->size
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
          G__letint(buf,'Y',(long)ifunc->pentry[i]->tp2f);
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
  return(NULL);

}


/*******************************************************************
* G__search_next_member()
*
* Search variable and function name within the scope
*******************************************************************/

char *G__search_next_member(const char *text,int state)
{
  static int list_index,index_item  /* ,cbp */;
  static size_t len;
  static char completionbuf[G__ONELINE];
  char *name,*result;
  int flag=1;
  static struct G__var_array *var;
  static char memtext[G__MAXNAME];
  static int isstruct;
  static G__value buf;
  char *dot,*point,*scope=NULL;
  static struct G__ifunc_table_internal *ifunc;

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
    G__strlcpy(completionbuf,text,G__ONELINE);
    dot=strrchr(completionbuf,'.');
    point=(char*)G__strrstr(completionbuf,"->");
    scope=(char*)G__strrstr(completionbuf,"::");

    /*************************************************************
     * struct member
     *************************************************************/
    if( dot || point || scope) {
      isstruct = 1;

      if(scope>dot && scope>point) {
        G__strlcpy(memtext,scope+2,G__MAXNAME);
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
        G__strlcpy(memtext,dot+1,G__MAXNAME);
        *dot='\0';
        buf = G__calc_internal(completionbuf);
        *dot='.';
        *(dot+1)='\0';
      }
      else {
        scope = (char*)NULL;
        G__strlcpy(memtext,point+2,G__MAXNAME);
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
            ifunc=(struct G__ifunc_table_internal *)NULL;
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
             result = (char *)malloc((strlen(completionbuf)+strlen(name)+1));
             sprintf(result,"%s%s",completionbuf,name); // Okay we allocated enough space
             return(result);
             break;
          case 1:
             result = (char *)malloc((strlen(completionbuf)+strlen(name)+2));
             sprintf(result,"%s%s(",completionbuf,name); // Okay, we allocated enought space
             return(result);
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
              ifunc=(struct G__ifunc_table_internal *)NULL;
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
            ifunc=(struct G__ifunc_table_internal *)NULL;
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
            sprintf(result,"%s(",name); // Okay we allocated enough space.
            return(result);
          case 2:
          case 3:
          case 4:
          case 5:
            result = (char *)malloc((strlen(name)+1));
            strcpy(result,name); // Okay, we allocated enough space
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
void G__ResetShlHandle()
{
#ifdef G__SHAREDLIB
  G__ShlHandle = (G__SHLHANDLE)0;
  G__Shlfilenum = -1;
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

/**************************************************************************
 * G__GetShlFilenum()
 **************************************************************************/
int G__GetShlFilenum()
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
void* G__SetShlHandle(char *filename)
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
char* G__GccNameMangle(G__FastAllocString &buf,struct G__ifunc_table_internal *ifunc,int ifn)
{
  char *funcname = ifunc->funcname[ifn];
  char tmp[4];
  int i;
  tmp[1]=0;
  buf.Format("_Z%lu%s",(unsigned long)strlen(funcname),funcname);

  for(i=0;i<ifunc->para_nu[ifn];i++) {
    if(isupper(ifunc->param[ifn][i]->type)) buf += "P";
    if(G__PARAREFERENCE==ifunc->param[ifn][i]->reftype) buf += "R";
    if(G__CONSTVAR&ifunc->param[ifn][i]->isconst) buf += "K";
    switch(tolower(ifunc->param[ifn][i]->type)) {
    case 'c':
    case 's':
    case 'i':
    case 'l':
    case 'f':
    case 'd':
      tmp[0] = ifunc->param[ifn][i]->type;
      break;
    case 'b': tmp[0]='h'; break;
    case 'r': tmp[0]='t'; break;
    case 'h': tmp[0]='j'; break;
    case 'k': tmp[0]='m'; break;
    case 'y': tmp[0]='v'; break;
    default:
      break;
    }
    buf += tmp;
  }
  if(0==ifunc->para_nu[ifn]) buf += "v";
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
char* G__Vc6TypeMangle(int type,int tagnum,int reftype,int isconst)
{
   static G__FastAllocString buf(G__MAXNAME);
   buf[0] = 0;
  if(isupper(type)) {
    if((G__CONSTVAR&isconst) && 
       (G__PCONSTVAR&isconst) &&
       (G__PARAREFERENCE!=reftype)) buf += "QB";
    else if(0==(G__CONSTVAR&isconst) && 
            (G__PCONSTVAR&isconst) &&
            (G__PARAREFERENCE!=reftype)) buf += "QA";
    else if((G__CONSTVAR&isconst) && 
            (0==(G__PCONSTVAR&isconst)) &&
            (G__PARAREFERENCE!=reftype)) buf += "PB";
    else if((0==(G__CONSTVAR&isconst)) && 
            (0==(G__PCONSTVAR&isconst)) &&
            (G__PARAREFERENCE!=reftype)) buf += "PA";
    else if((G__CONSTVAR&isconst) && 
            (0==(G__PCONSTVAR&isconst)) &&
            (G__PARAREFERENCE==reftype)) buf += "AB";
    else if((0==(G__CONSTVAR&isconst)) && 
            (0==(G__PCONSTVAR&isconst)) &&
            (G__PARAREFERENCE==reftype)) buf += "AA";
    else buf += "PA";
  }
  switch(tolower(type)) {
  case 'y': buf += "X"; break;
  case 'c': buf += "D"; break;
  case 's': buf += "F"; break;
  case 'i': buf += "H"; break;
  case 'l': buf += "J"; break;
  case 'f': buf += "M"; break;
  case 'd': buf += "N"; break;
  case 'b': buf += "E"; break;
  case 'r': buf += "G"; break;
  case 'h': buf += "I"; break;
  case 'k': buf += "K"; break;
  case 'u': buf += "V"; buf += G__struct.name[tagnum]; 
    buf += "@@"; break;
  case 'e': buf = "PAU_iobuf@@"; break;
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
char* G__Vc6NameMangle(G__FastAllocString &buf,struct G__ifunc_table_internal *ifunc,int ifn)
{
  char *funcname = ifunc->funcname[ifn];
  int i;

  /* funcname */
  buf.Format("?%s@",funcname);

  /* scope */
  if(-1!=ifunc->tagnum) buf += G__struct.name[ifunc->tagnum];
  buf += "@YA";

  /* return type */
  buf += G__Vc6TypeMangle(ifunc->type[ifn]
                          ,ifunc->p_tagtable[ifn]
                          ,ifunc->reftype[ifn]
                          ,ifunc->isconst[ifn]);

  /* arguments */
  for(i=0;i<ifunc->para_nu[ifn];i++) {
    buf += G__Vc6TypeMangle(ifunc->param[ifn][i]->type
                            ,ifunc->param[ifn][i]->p_tagtable
                            ,ifunc->param[ifn][i]->reftype
                            ,ifunc->param[ifn][i]->isconst);
  }
  if(0==ifunc->para_nu[ifn]) buf += "X";
  else buf += "@";

  /* end */
  buf += "Z";

  return(buf);
}

/**************************************************************************
 * G__FindSymbol
 **************************************************************************/
void* G__FindSymbol(struct G__ifunc_table_internal *ifunc,int ifn)
{
#ifdef G__SHAREDLIB
  char *funcname=ifunc->funcname[ifn];
  void *p2f=0;
  if(G__ShlHandle) {
    G__FastAllocString buf(G__ONELINE);

    /* funcname, VC++, GCC, C function */
    p2f = (void*)G__shl_findsym(&G__ShlHandle,funcname,TYPE_PROCEDURE);

    /* _funcname,  BC++, C function */
    if(!p2f) {
       buf = "_";
       buf += funcname;
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
void* G__FindSym(char *filename,char *funcname)
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
   
const char *G__dladdr(void (*func)())
{
#ifndef G__SHAREDLIB
   return 0;
#else //G__SHAREDLIB
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

#endif //G__SHAREDLIB
}

// G__RegisterLibrary
void* G__RegisterLibrary(void (*func)()) {
   // This function makes sure that the library that contains 'func' is
   // known to have been loaded by the CINT system.

   const char *libname = G__dladdr( func );
   if (libname && libname[0]) {
      size_t lenLibName = strlen(libname);
      G__FastAllocString sbLibName(lenLibName);
      sbLibName = libname;
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
      G__register_sharedlib( sbLibName );
   }
   return 0;
}   

// G__RegisterLibrary
void* G__UnregisterLibrary(void (*func)()) {
   // This function makes sure that the library that contains 'func' is
   // known to have been laoded by the CINT system.
      
   const char *libname = G__dladdr( func );
   if (libname) {
      G__unregister_sharedlib( libname );
   }
   return 0;
}   
   
} /* extern "C" */

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
