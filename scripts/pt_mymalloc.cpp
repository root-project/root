#include <malloc.h>
#include <sys/types.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>
#include <string.h>
#include <string>
#include <errno.h>
#include <pthread.h>
#include <dlfcn.h>

using namespace std;

struct PerfTrackMallocInterposition {
   PerfTrackMallocInterposition(): fFifoFD(-1), fPerfData() {
      SetFunc((void**)&fPMalloc, "malloc");
      SetFunc((void**)&fPRealloc, "realloc");
      SetFunc((void**)&fPFree, "free");

      pthread_mutex_init(&fgPTMutex, 0);

      static char envPreload[] = "LD_PRELOAD=";
      putenv(envPreload);

      const char* fifoname = getenv("PT_FIFONAME");
      if (fifoname) {
         fFifoFD = open(fifoname, O_WRONLY);
         if (fFifoFD < 0) printf( "Error opening file: %s\n", strerror( errno ) ); 
      } else {
         printf("Error: PT_FIFONAME not set!\n");
         int * i = (int*)0x123;
         *i = 12;
      }
      fPerfData[3]=699692586; // for collector to know that stored values are valid

   }

   ~PerfTrackMallocInterposition() {
      pthread_mutex_destroy(&fgPTMutex);

      if (fFifoFD >= 0 && write(fFifoFD, &fPerfData, sizeof(fPerfData))==-1)
         printf("Error in writing %s\n", strerror(errno));
   }

   void SetFunc(void** ppFunc, const char* name) const {
      *ppFunc = dlsym(RTLD_NEXT, name);
      const char* error = dlerror();
      if (error != NULL) {
         fputs(error, stderr);
         exit(1);
      }
   }

   void* (*fPMalloc)(size_t);
   void* (*fPRealloc)(void*, size_t);
   void  (*fPFree)(void*);

   int fFifoFD; // file decriptor of FIFO
   long fPerfData[4]; // 0: heap, 1: maxheap, 2: sum allocs, 3: 699692586
   static pthread_mutex_t fgPTMutex;
};

pthread_mutex_t PerfTrackMallocInterposition::fgPTMutex = PTHREAD_MUTEX_INITIALIZER;

PerfTrackMallocInterposition& gPT() {
   static PerfTrackMallocInterposition pt;
   return pt;
}

void *malloc(size_t size)
{
  pthread_mutex_lock(&gPT().fgPTMutex);
  char* result = (char *)(*gPT().fPMalloc)(size+sizeof(int)+sizeof(size_t));

  gPT().fPerfData[0]+=size; // heap
  gPT().fPerfData[2]+=size; // only allocs
  if (gPT().fPerfData[0]>gPT().fPerfData[1]) gPT().fPerfData[1]=gPT().fPerfData[0]; // maximum (peak) heap
  pthread_mutex_unlock(&gPT().fgPTMutex);
   
  *(int *)result=699692586;
  *(size_t *)(result+sizeof(int))=size;

  return (void*) (result+sizeof(int)+sizeof(size_t));
}

void free(void *ptr)
{
  if (ptr==0) return;

  int v1=*(int *) ((char*)ptr-sizeof(int)-sizeof(size_t));      
  if (v1!=699692586){
     (*gPT().fPFree)(ptr);
     return;
  }
  size_t v2=* (size_t *) ((char*)ptr-sizeof(size_t));

  pthread_mutex_lock(&gPT().fgPTMutex);

  gPT().fPerfData[0]-=v2;
 
  (*gPT().fPFree)((char*)ptr-sizeof(int)-sizeof(size_t)); 

  pthread_mutex_unlock(&gPT().fgPTMutex);
}

void *realloc(void *ptr, size_t size)
{
  char *result;
  int v1=699692586;
  if (ptr!=0) v1=*(int *) ((char*)ptr-sizeof(int)-sizeof(size_t));

  pthread_mutex_lock(&gPT().fgPTMutex);
  if (v1!=699692586 || ptr==0){ 
    if (ptr==0 && size!=0){ // behaves as malloc   
      gPT().fPerfData[0]+=size; // heap mem
      gPT().fPerfData[2]+=size; // only allocs
      if (gPT().fPerfData[0]>gPT().fPerfData[1]) gPT().fPerfData[1]=gPT().fPerfData[0]; // maximum heap
	       
      result=(char *)(*gPT().fPRealloc)(0,size+sizeof(int)+sizeof(size_t));       
      *(int *)result=699692586;
      *(size_t *)(result+sizeof(int))=size;
    }
    else{  
      result=(char *)(*gPT().fPRealloc)(ptr,size); // old realloc call 
    }
  }
  else
    {
      size_t v2=*(size_t *) ((char*)ptr-sizeof(size_t));
      if (size==0){ // behaves as free
	gPT().fPerfData[0]-=v2;
	result=(char *)(*gPT().fPRealloc)((char*)ptr-sizeof(int)-sizeof(size_t),0);
      }
      else{
	gPT().fPerfData[0]=gPT().fPerfData[0]+size-v2; // heap
	gPT().fPerfData[2]=gPT().fPerfData[2]+size-v2; // only allocs
	if (gPT().fPerfData[0]>gPT().fPerfData[1]) gPT().fPerfData[1]=gPT().fPerfData[0]; // maximum heap
		 
	result = (char *)(*gPT().fPRealloc)((char*)ptr-sizeof(int)-sizeof(size_t),size+sizeof(int)+sizeof(size_t));
	*(int *)result=699692586;
	*(size_t *)(result+sizeof(int))=size;
      }
    }
		 
 pthread_mutex_unlock(&gPT().fgPTMutex); 

 if (v1!=699692586 || size==0) return (void*) (result); 
  else return (void*) (result+sizeof(int)+sizeof(size_t)); 
}

