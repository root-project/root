#include <sys/types.h>
#include <cerrno>
#include <dlfcn.h>
#include <fcntl.h>
#if defined(__APPLE__)
#include <cstdlib>
#else
#include <malloc.h>
#endif
#include <pthread.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

// Intercepts calls to malloc, realloc, free, by planting replacement symbols.
// This library is meant to be LD_PRELOAD'ed to do its job.
//
// Collects statistics and pipes them back through a FIFO specified in the
// env var PT_FIFONAME.
//
// Manipulates allocations by prepending a TAG value of size int and the size of
// the allocation (size_t).
//
// The initialization of the counting structures and forwarding function pointers
// happens upon the first call to malloc / realloc / free; this assumes that no
// threads have been created before the first call to malloc / realloc / free.
//

class PerfTrackMallocInterposition {
public:
   static PerfTrackMallocInterposition& Instance() {
      static PerfTrackMallocInterposition pt;
      return pt;
   }

   void* Malloc(size_t size);
   void* Realloc(void* ptr, size_t size);
   void  Free(void* ptr);

private:
   // Statistics elements
   enum EPerfDataType {
      kPDCurrentHeap,
      kPDMaxHeap,
      kPDSumAllocs,
      kPDTag,
      kNumPerfDataTypes
   };

   PerfTrackMallocInterposition(): fFifoFD(-1), fPerfData() {
      // Initialize data structures, mutex, fifo.
      SetFunc((void**)&fPMalloc, "malloc");
      SetFunc((void**)&fPRealloc, "realloc");
      SetFunc((void**)&fPFree, "free");

      pthread_mutex_init(&fgPTMutex, 0);

      static char envPreload[] = "LD_PRELOAD=";
      putenv(envPreload);

      // Open the FIFO:
      static const char* fifoenv = "PT_FIFONAME";
      const char* fifoname = std::getenv(fifoenv);
      if (fifoname) {
         fFifoFD = open(fifoname, O_WRONLY);
         if (fFifoFD < 0) {
            printf("%s:%d: Error opening FIFO %s: %s\n", __FILE__, __LINE__, fifoname, strerror(errno)); 
         }
      } else {
         printf("%s:%d: %s not set: %s\n", __FILE__, __LINE__, fifoenv, strerror(errno)); 
      }
      fPerfData[kPDTag]=699692586; // for collector to know that stored values are valid
   }

   ~PerfTrackMallocInterposition() {
      // Tear down.
      pthread_mutex_destroy(&fgPTMutex);

      if (fFifoFD >= 0 && write(fFifoFD, &fPerfData, sizeof(fPerfData))==-1)
         printf("%s:%d: Error writing statistics to fifo: %s\n", __FILE__, __LINE__, strerror(errno)); 
   }

   void SetFunc(void** ppFunc, const char* name) const {
      // Find the next (i.e. not ours :-) symbol called name.
      *ppFunc = dlsym(RTLD_NEXT, name);
      const char* error = dlerror();
      if (error != NULL) {
         printf("%s:%d: Error looking for original symbol %s: %s\n", __FILE__, __LINE__, name, error);
         exit(1);
      }
   }

   void IncHeap(size_t size) {
      // Increase our heap statistics counter.
      fPerfData[kPDCurrentHeap] += size; // heap
      fPerfData[kPDSumAllocs] += size; // only allocs
      if (fPerfData[kPDCurrentHeap] > fPerfData[kPDMaxHeap])
         fPerfData[kPDMaxHeap] = fPerfData[kPDCurrentHeap];
   }

   void* (*fPMalloc)(size_t);
   void* (*fPRealloc)(void*, size_t);
   void  (*fPFree)(void*);

   int fFifoFD; // file decriptor of FIFO
   long fPerfData[kNumPerfDataTypes]; // statistics data
   static pthread_mutex_t fgPTMutex; // protects statistics in multithreaded.
};

pthread_mutex_t PerfTrackMallocInterposition::fgPTMutex = PTHREAD_MUTEX_INITIALIZER;

void* PerfTrackMallocInterposition::Malloc(size_t size) {
   // Malloc with statistics
   pthread_mutex_lock(&fgPTMutex);
   char* result = (char *)(*fPMalloc)(size+sizeof(int)+sizeof(size_t));
   IncHeap(size);
   pthread_mutex_unlock(&fgPTMutex);
   
   *(int *)result=699692586;
   *(size_t *)(result+sizeof(int))=size;

   return (void*) (result+sizeof(int)+sizeof(size_t));
}

void* PerfTrackMallocInterposition::Realloc(void* ptr, size_t size) {
   // Realloc with statistics
  char *result;
  int v1=699692586;
  if (ptr!=0) v1=*(int *) ((char*)ptr-sizeof(int)-sizeof(size_t));

  pthread_mutex_lock(&fgPTMutex);
  if (v1!=699692586 || ptr==0){ 
    if (ptr==0 && size!=0){ // behaves as malloc   
       IncHeap(size);
      result=(char *)(*fPRealloc)(0,size+sizeof(int)+sizeof(size_t));       
      *(int *)result=699692586;
      *(size_t *)(result+sizeof(int))=size;
    }
    else {
      result=(char *)(*fPRealloc)(ptr,size); // old realloc call 
    }
  }
  else
    {
      size_t v2=*(size_t *) ((char*)ptr-sizeof(size_t));
      if (size==0){ // behaves as free
	fPerfData[kPDCurrentHeap]-=v2;
	result=(char *)(*fPRealloc)((char*)ptr-sizeof(int)-sizeof(size_t),0);
      }
      else{
         IncHeap(size-v2);
		 
	result = (char *)(*fPRealloc)((char*)ptr-sizeof(int)-sizeof(size_t),size+sizeof(int)+sizeof(size_t));
	*(int *)result=699692586;
	*(size_t *)(result+sizeof(int))=size;
      }
    }
		 
 pthread_mutex_unlock(&fgPTMutex); 

 if (v1!=699692586 || size==0) return (void*) (result); 
  else return (void*) (result+sizeof(int)+sizeof(size_t)); 

}


void PerfTrackMallocInterposition::Free(void* ptr) {
   // Free with statistics
   if (ptr==0) return;

  int v1=*(int *) ((char*)ptr-sizeof(int)-sizeof(size_t));      
  if (v1!=699692586){
     (*fPFree)(ptr);
     return;
  }
  size_t v2=* (size_t *) ((char*)ptr-sizeof(size_t));

  pthread_mutex_lock(&fgPTMutex);

  fPerfData[kPDCurrentHeap]-=v2;
 
  (*fPFree)((char*)ptr-sizeof(int)-sizeof(size_t)); 

  pthread_mutex_unlock(&fgPTMutex);

}

// Replacement symbols:

void *malloc(size_t size) {
   return PerfTrackMallocInterposition::Instance().Malloc(size);
}

void free(void *ptr) {
   PerfTrackMallocInterposition::Instance().Free(ptr);
}

void *realloc(void *ptr, size_t size) {
   return PerfTrackMallocInterposition::Instance().Realloc(ptr, size);
}

