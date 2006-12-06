/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/************************************************************************
 * demo/multilibcint/mt.c 
 * 
 * Description:
 *  Cint's multi-thread workaround library. 
 *  Refer to README.txt in this directory.
 ************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(G__WIN32) || defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#include <dlfcn.h>
#include <unistd.h>
#endif

/************************************************************************
 * platform dependent declaration, needs to be fixed
 ************************************************************************/
#if defined(G__WIN32) || defined(_WIN32)
typedef HINSTANCE G__SHLHANDLE;
typedef HANDLE pthread_t;
#else
typedef void* G__SHLHANDLE;
# define G__RTLD_NOW  (RTLD_GLOBAL | RTLD_NOW)
# define G__RTLD_LAZY (RTLD_GLOBAL | RTLD_LAZY)
//static int G__RTLD_flag = RTLD_LAZY;
static int G__RTLD_flag = RTLD_NOW;
#endif


/************************************************************************
 * static declaration
 ************************************************************************/
#define G__MAXLIBCINT  32
static char cintsysdir[512];
typedef int (*G__f_init_cint) (char*);
typedef int (*G__f_scratch_all) ();

struct G__LIBCINT {
  char name[512];
  char args[512];
  G__SHLHANDLE handle;
  int active;
  G__f_init_cint f_init_cint;
  G__f_scratch_all f_scratch_all;
};
static struct G__LIBCINT libcint[G__MAXLIBCINT];

/************************************************************************
 * STATIC(PRIVATE) FUNCTIONS
 ************************************************************************/

/************************************************************************
 * G__getlibcint()
 ************************************************************************/
static int G__getlibcint(char *args) {
  char cintlib[512];
  int i;

  /* get cintsysdir if not set */
  if(0==cintsysdir[0]) {
    char *p;
    p = getenv("CINTSYSDIR");
    if(p) strcpy(cintsysdir,p);
    else {
      fprintf(stderr,"Error: Environment variable CINTSYSDIR not set\n");
      return(0);
    }
  }

  /* If there is an already loaded libcintX.so and it is idle, reuse it */
  for(i=0;i<G__MAXLIBCINT;i++) {
    if(0==libcint[i].active && libcint[i].handle) {
      libcint[i].active = 1;
      /* return libcintid */
      return(i);
    }
  }

  /* If there is no reusable libcintX.so already, create another copy */
  for(i=0;i<G__MAXLIBCINT;i++) {
    if(0==libcint[i].handle) {
      /* create another copy of libcint.so */
      libcint[i].active = 1;
      strcpy(libcint[i].args,args);
#if defined(G__WIN32) || defined(_WIN32)
      sprintf(libcint[i].name,"\\temp\\libcint%d.dll",i);
      sprintf(cintlib,"copy %s\\libcint.dll %s",cintsysdir,libcint[i].name);
#else
      sprintf(libcint[i].name,"/tmp/libcint%d.so",i);
      sprintf(cintlib,"cp %s/libcint.so %s",cintsysdir,libcint[i].name);
#endif
      system(cintlib);
      
      /* explicitly load shared library */
      strcpy(cintlib,libcint[i].name);
#if defined(G__WIN32) || defined(_WIN32)
      libcint[i].handle = LoadLibrary(cintlib);
#else
      libcint[i].handle = dlopen(cintlib,G__RTLD_flag);
#endif
      if(!libcint[i].handle) break;

      /* find symbols */
#if defined(G__WIN32) || defined(_WIN32)
      libcint[i].f_init_cint 
	= (G__f_init_cint)GetProcAddress(libcint[i].handle,"G__init_cint");
      libcint[i].f_scratch_all 
	= (G__f_scratch_all)GetProcAddress(libcint[i].handle,"G__scratch_all");
#else
      libcint[i].f_init_cint 
	= (G__f_init_cint)dlsym(libcint[i].handle,"G__init_cint");
      libcint[i].f_scratch_all 
	= (G__f_scratch_all)dlsym(libcint[i].handle,"G__scratch_all");
#endif

      /* return libcintid */
      return(i);
    }
  }

  fprintf(stderr,"Error: Can not open new libcint\n");
  return(-1);
}

#if 0
/************************************************************************
 * G__getlibcinthandle()
 ************************************************************************/
static G__SHLHANDLE G__getlibcinthandle(int i)
{
  return(libcint[i].handle);
}
#endif

/************************************************************************
 * G__getinitcint()
 ************************************************************************/
static G__f_init_cint G__getinitcint(int i)
{
  return(libcint[i].f_init_cint);
}

/************************************************************************
 * G__resetactive()
 ***********************************************************************/
static void G__resetactive(int i)
{
  (*libcint[i].f_scratch_all)();
  libcint[i].active = 0;
}

/************************************************************************
 * G__getargs()
 ************************************************************************/
static char* G__getargs(int i)
{
  return(libcint[i].args);
}

/************************************************************************
 * G__cintthreadfunction()
 ************************************************************************/
static void *G__cintthreadfunction(void* arg) {
  G__f_init_cint f_init_cint;
  char buf[100];
  int libcintid = (int)arg;

  if(-1==libcintid) {
    fprintf(stderr,"Error: Failed to open new libcint\n");
#if defined(G__WIN32) || defined(_WIN32)
#else
    pthread_exit((void*)0);
#endif
    return(0);
  }

  /* Run G__init_cint("cint [arguments]") */
  f_init_cint = G__getinitcint(libcintid);
  if(!f_init_cint) {
    fprintf(stderr,"Error: Failed to get G__init_cint function\n");
    goto end;
  }
  sprintf(buf,"cint %s",G__getargs(libcintid));
  (*f_init_cint)(buf);

 end:
  G__resetactive(libcintid);
#if defined(G__WIN32) || defined(_WIN32)
#else
  pthread_exit((void*)0);
#endif
  return(0);
}

/************************************************************************
 *
 * EXPORTED FUNCTIONS
 *
 ************************************************************************/

/************************************************************************
 * G__createcintthread()
 *
 * Description: 
 *  Create and run cint thread in background.
 *
 * Arguments:
 *  Return value    : Handle to created thread 
 *  char* args      : Command line argument that is given to cint
 ************************************************************************/
pthread_t G__createcintthread(char* args) {
#if defined(G__WIN32) || defined(_WIN32)
  DWORD ThreadID;
#endif
  void *arg;
  int libcintid;
  pthread_t thread;
  libcintid = G__getlibcint(args);
  arg = (void*)(libcintid);
#if defined(G__WIN32) || defined(_WIN32)
  thread = CreateThread(NULL,0
                         ,(LPTHREAD_START_ROUTINE)G__cintthreadfunction
                         ,arg,0
                         ,&ThreadID);
#else
  pthread_create(&thread,NULL,G__cintthreadfunction,arg);
#endif
  return(thread);
}

/************************************************************************
 * G__joincintthread()
 *
 * Description:
 *  Wait for Cint thread to finish and join the main thread.
 *
 * Arguments:
 *  pthread_t t1    : Handle to an waiting thread
 *  int timeout     : Valid only for windows, timeout by second, default=0
 ************************************************************************/
void G__joincintthread(pthread_t t1,int timeout) {
#ifdef G__WIN32
  WaitForSingleObject(t1,timeout); // Wait for background job to finish
  CloseHandle(t1);
#else
  void *p;
  pthread_join(t1,&p);
#endif
}


/************************************************************************
 * G__clearlibcint()
 *
 * Description:
 *  Clean up multi-thread Cint garbage in \temp or /tmp directory.
 *  Call this function only at very end of the process.
 *
 ************************************************************************/
void G__clearlibcint() {
  int active=0;
  int i;
  for(i=0;i<G__MAXLIBCINT;i++) active += libcint[i].active;

  if(0==active) {
    for(i=0;i<G__MAXLIBCINT;i++) {
      /* Reset interpreter core */
      if(libcint[i].f_scratch_all) (*libcint[i].f_scratch_all)();

      /* Close shared library handle and remove /tmp/libcintX.so */
#if defined(G__WIN32) || defined(_WIN32)
      if(libcint[i].handle) FreeLibrary(libcint[i].handle);
#else
      if(libcint[i].handle) dlclose(libcint[i].handle);
#endif
      if(libcint[i].name[0]) remove(libcint[i].name);

      /* Clear table entry for libcintX.so */
      libcint[i].name[0] = 0;
      libcint[i].handle = (G__SHLHANDLE)0;
    }
  }
  else {
    /* Something needs to be implemented here */
  }
}
