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

using namespace std;
static void *(*old_malloc_hook)(size_t, const void *);
static void (*old_free_hook)(void*, const void *);
static void *(*old_realloc_hook)(void *, size_t, const void *);
static void init_my_hooks(void);
static void *my_malloc_hook(size_t, const void *);
static void my_free_hook (void*, const void *);
static void *my_realloc_hook(void *, size_t, const void *);
static void writeMem(void);

int fd;
long arr [4];
char* fifoname;
pthread_mutex_t pt_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Override initializing hook from the C library. */
void (*__malloc_initialize_hook)(void) = init_my_hooks;

static void init_my_hooks(void) {
   static char envPreload[] = "LD_PRELOAD=";
   putenv(envPreload);

   pthread_mutex_init(&pt_mutex, 0);
  {
    fifoname = getenv("PT_FIFONAME");
    /*
    char *path=NULL;
    path = getcwd(NULL, 0);
    stringstream sstm;
    sstm << path << "/pt_myfifo_" << getpid();
    fifoname = new char[sstm.str().length() + 1];
    strcpy(fifoname, sstm.str().c_str());
    */
  }

  fd= open(fifoname, O_WRONLY);
  if( fd < 0 ) printf( "Error opening file: %s\n", strerror( errno ) ); 
  arr[3]=699692586; // for collector to know that stored values are valid
  
  atexit(writeMem);
 
  old_malloc_hook = __malloc_hook;
  __malloc_hook = my_malloc_hook;
  old_free_hook = __free_hook;
  __free_hook = my_free_hook;
  old_realloc_hook = __realloc_hook;
  __realloc_hook = my_realloc_hook;
}

static void *my_malloc_hook(size_t size, const void *caller)
{
  pthread_mutex_lock(&pt_mutex);

  char *result;

  __malloc_hook = old_malloc_hook;
  __free_hook = old_free_hook;
  __realloc_hook = old_realloc_hook;
	   
  result = (char *)malloc(size+sizeof(int)+sizeof(size_t));
  old_malloc_hook = __malloc_hook;
  old_free_hook = __free_hook;
  old_realloc_hook = __realloc_hook;
	   
  arr[0]+=size; // heap
  arr[2]+=size; // only allocs
  if (arr[0]>arr[1]) arr[1]=arr[0]; // maximum (peak) heap
	   
  *(int *)result=699692586;
  *(size_t *)(result+sizeof(int))=size;

  __malloc_hook = my_malloc_hook;
  __free_hook = my_free_hook;
  __realloc_hook = my_realloc_hook;

  pthread_mutex_unlock(&pt_mutex);

  return (void*) (result+sizeof(int)+sizeof(size_t));
}

static void my_free_hook (void *ptr, const void *caller)
{
  if (ptr==0) return;

  pthread_mutex_lock(&pt_mutex);

  __free_hook = old_free_hook;
  __malloc_hook = old_malloc_hook;
  __realloc_hook = old_realloc_hook;
       
  int v1=*(int *) ((char*)ptr-sizeof(int)-sizeof(size_t));      
  if (v1!=699692586){
    free(ptr);
    old_free_hook = __free_hook;
    old_malloc_hook = __malloc_hook;
    old_realloc_hook = __realloc_hook;
    __free_hook = my_free_hook;
    __malloc_hook = my_malloc_hook;
    __realloc_hook = my_realloc_hook;
    pthread_mutex_unlock(&pt_mutex);
    return;
  }
  size_t v2=* (size_t *) ((char*)ptr-sizeof(size_t));
  arr[0]-=v2;
 
  free ((char*)ptr-sizeof(int)-sizeof(size_t)); 
  old_free_hook = __free_hook;
  old_malloc_hook = __malloc_hook;
  old_realloc_hook = __realloc_hook;
  __free_hook = my_free_hook;
  __malloc_hook = my_malloc_hook;
  __realloc_hook = my_realloc_hook;

  pthread_mutex_unlock(&pt_mutex);
}

static void *my_realloc_hook(void *ptr, size_t size, const void *caller)
{
  char *result;

  pthread_mutex_lock(&pt_mutex);

  __malloc_hook = old_malloc_hook;
  __free_hook = old_free_hook;
  __realloc_hook = old_realloc_hook;

  int v1=699692586;
  if (ptr!=0) v1=*(int *) ((char*)ptr-sizeof(int)-sizeof(size_t));
  if (v1!=699692586 || ptr==0){ 
    if (ptr==0 && size!=0){ // behaves as malloc   
      arr[0]+=size; // heap mem
      arr[2]+=size; // only allocs
      if (arr[0]>arr[1]) arr[1]=arr[0]; // maximum heap
	       
      result=(char *)realloc(0,size+sizeof(int)+sizeof(size_t));       
      *(int *)result=699692586;
      *(size_t *)(result+sizeof(int))=size;
    }
    else{  
      result=(char *)realloc(ptr,size); // old realloc call 
    }
  }
  else
    {
      size_t v2=*(size_t *) ((char*)ptr-sizeof(size_t));
      if (size==0){ // behaves as free
	arr[0]-=v2;
	result=(char *)realloc((char*)ptr-sizeof(int)-sizeof(size_t),0);
      }
      else{
	arr[0]=arr[0]+size-v2; // heap
	arr[2]=arr[2]+size-v2; // only allocs
	if (arr[0]>arr[1]) arr[1]=arr[0]; // maximum heap
		 
	result = (char *)realloc((char*)ptr-sizeof(int)-sizeof(size_t),size+sizeof(int)+sizeof(size_t));
	*(int *)result=699692586;
	*(size_t *)(result+sizeof(int))=size;
      }
    }
  old_malloc_hook = __malloc_hook;
  old_free_hook = __free_hook;
  old_realloc_hook = __realloc_hook;
  __malloc_hook = my_malloc_hook;
  __free_hook = my_free_hook;
  __realloc_hook = my_realloc_hook;
		 
 pthread_mutex_unlock(&pt_mutex); 

 if (v1!=699692586 || size==0) return (void*) (result); 
  else return (void*) (result+sizeof(int)+sizeof(size_t)); 
}

static void writeMem(void)
{
  
  pthread_mutex_destroy(&pt_mutex);

  if (write(fd,&arr,4*sizeof(long))==-1)
      printf("Error in writing %s\n", strerror( errno ));
}
