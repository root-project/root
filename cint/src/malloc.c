/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file malloc.c
 ************************************************************************
 * Description:
 *  Allocate automatic variable arena 
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

#ifdef G__SHMGLOBAL
/******************************************************************
* 
******************************************************************/
#include <unistd.h>
#include <sys/types.h>
#include <sys/shm.h>

int G__myshmid;
void* G__shmbuffer;
int* G__pthreadnum;
int G__shmsize = 0x10000;

/******************************************************************
* G__createthread()
******************************************************************/
int G__CreateThread() {
  return((int)fork());
}

/******************************************************************
* G__shmcalloc()
******************************************************************/
void* G__shmmalloc(size)
int size;
{
  int* poffset = (int*)G__shmbuffer;
  void* result;
  int alignsize;

  if(size<=sizeof(double)) alignsize = size;
  else                     alignsize = sizeof(double);
  if((*poffset)%alignsize) *poffset += sizeof(alignsize)-(*poffset)%alignsize;

  result = (void*)((long)G__shmbuffer + (*poffset));

  (*poffset) += size;
  if((*poffset) >= G__shmsize) return((void*)0);

  return(result);
}

/******************************************************************
* G__shmcalloc()
******************************************************************/
void* G__shmcalloc(atomsize,num)
int atomsize;
int num;
{
  int i;
  int size = atomsize * num;
  void* result = G__shmmalloc(size);
  if(!result) return(result);
  for(i=0;i<size;i++) *((char*)result+i) = (char)0;
  return(result);
}

/******************************************************************/
/* #define calloc G__shmcalloc */

/******************************************************************
* G__shmfinish()
******************************************************************/
void G__shmfinish() {
  /* free shared memory */
  switch(*G__pthreadnum) {
  case 0:
    break;
  case 1:
    (*G__pthreadnum)--;
    shmdt(G__shmbuffer);
    shmctl(G__myshmid,IPC_RMID,0);
    break;
  default:
    (*G__pthreadnum)--;
    shmdt(G__shmbuffer);
    break;
  }
}

/******************************************************************
* G__shminit()
******************************************************************/
void G__shminit() {
  /* Prepare keys */
  key_t mykey;
  const char projid = 'Q';
  char keyfile[256];
  int* poffset;

  G__getcintsysdir();
  sprintf(keyfile,"%s/cint",G__cintsysdir);
  mykey=ftok(keyfile,projid);
  /* printf("mykey=%x\n",mykey); */

  /* Shared Memory */
  G__myshmid = shmget(mykey,G__shmsize,SHM_R|SHM_W|IPC_CREAT);
  /* printf("myshmid=%x\n",G__myshmid); */
  if(-1 == G__myshmid) {
    fprintf(stderr,"shmget failed\n");
    exit(1);
  }

  G__shmbuffer = shmat(G__myshmid,0,0);
  /* printf("shmbuffer=%p\n",G__shmbuffer); */
  if((void*)(~0)==G__shmbuffer) {
    fprintf(stderr,"shmat failed\n");
    exit(1);
  }

  /* set offset address */
  poffset = (int*)G__shmcalloc(sizeof(int),1);
  *poffset = 0;

  /* set thread num count */
  G__pthreadnum = (int*)G__shmcalloc(sizeof(int),1);
  *G__pthreadnum = 1;

  /* set finish function */
  atexit(G__shmfinish);
}

/******************************************************************/
#endif /* G__SHMGLOBAL */

#ifndef G__PHILIPPE21
extern int G__const_noerror;
#endif

/******************************************************************
* static long G__getstaticobject()
*
******************************************************************/
static long G__getstaticobject()
{
  char temp[G__ONELINE];
  int hash,i;
  struct G__var_array *var;

  if(-1!=G__memberfunc_tagnum) /* questionable */
    sprintf(temp,"%s\\%x\\%x\\%x",G__varname_now,G__func_page,G__func_now
	    ,G__memberfunc_tagnum);
  else
    sprintf(temp,"%s\\%x\\%x" ,G__varname_now,G__func_page,G__func_now);

  G__hash(temp,hash,i)
  var = &G__global;
  do {
    i=0;
    while(i<var->allvar) {
      if((var->hash[i]==hash)&&(strcmp(var->varnamebuf[i],temp)==0)) {
	return(var->p[i]);
      }
      i++;
    }
    var = var->next;
  } while(var);
#ifndef G__OLDIMPLEMENTATION1519
  if(0==G__const_noerror) {
    G__fprinterr(G__serr,"Error: No memory for static %s ",temp);
    G__genericerror((char*)NULL);
  }
#else
#ifndef G__PHILIPPE21
  if(0==G__const_noerror) 
    G__fprinterr(G__serr,"Error: No memory for static %s ",temp);
#else
  G__fprinterr(G__serr,"Error: No memory for static %s ",temp);
#endif
  G__genericerror((char*)NULL);
#endif
  return(0);
}

/******************************************************************
* int G__malloc(n,size,item)
*
*  Allocate memory
******************************************************************/
long G__malloc(n,bsize,item) /* used to be int */
int n;
int bsize;
char *item;
{
  long allocmem; /* used to be int */
  int size;

#ifdef G__MEMTEST
  fprintf(G__memhist,"G__malloc(%d,%d,%s)\n",n,bsize,item);
#endif

  /****************************************************
   * Calculate total malloc size
   ****************************************************/
  size = n*bsize;

#ifndef G__OLDIMPLEMENTATION523
  /* experimental reference type in bytecode */
  if(G__globalvarpointer!=G__PVOID &&
     G__ASM_FUNC_COMPILE==G__asm_wholefunction) {
    G__globalvarpointer=G__PVOID;
    size=G__LONGALLOC;
  }
#endif
  
  if(G__globalvarpointer==G__PVOID) {
    /********************************************
     * Interpretively allocate memory area 
     ********************************************/
#ifdef G__ASM_WHOLEFUNC
    if(G__def_struct_member==0 && G__ASM_FUNC_NOP==G__asm_wholefunction) {
#else
    if(G__def_struct_member==0) {
#endif
      /*************************************
       * Static variable in function scope
       * which is already allocated at
       * pre-RUN.
       *************************************/
      if((G__static_alloc==1)&&(G__prerun==0)
#ifndef G__OLDIMPLEMENTATION858
	 && 0<=G__func_now
#endif
	 ) {
	return(G__getstaticobject());
      }
      /*************************************
       * Allocate memory area. Normal case
       *************************************/
      else {
	if(G__prerun) allocmem=(long)calloc((size_t)n ,(size_t)bsize);
	else          allocmem=(long)malloc((size_t)size);
	if(allocmem==(long)NULL) G__malloc_error(item);
      }
      return(allocmem);
    }
    
    /********************************************
     * Interpretively calculate struct offset
     ********************************************/
    else { /* G__def_struct_member==1 || G__ASM_FUNC_COMPILE */
      /***********************************
       * In case of struct
       ***********************************/
      /********************************************
       * Conservative padding strategy
       ********************************************/
      if(G__struct.type[G__tagdefining]=='s' ||
	 G__struct.type[G__tagdefining]=='c') {
	/***********************************
	 * allocate static data member
	 ***********************************/
	if(G__static_alloc) {
	  if(G__ASM_FUNC_COMPILE==G__asm_wholefunction) {
	    return(G__getstaticobject());
	  }
	  else {
	    allocmem=(long)calloc((size_t)n ,(size_t)bsize);
	    if(allocmem==(long)NULL) G__malloc_error(item);
	  }
	  return(allocmem);
	}
	/***********************************
	 * Get padding size
	 ***********************************/
	if(bsize>(int)G__DOUBLEALLOC) allocmem=G__DOUBLEALLOC;
	else	    allocmem=bsize;
	/***********************************
	 * Get padding size
	 ***********************************/
	G__struct.size[G__tagdefining] += size;
	/***********************************
	 * padding 
	 ***********************************/
	if(allocmem&&G__struct.size[G__tagdefining]%allocmem!=0){
	  G__struct.size[G__tagdefining]
	    += allocmem - G__struct.size[G__tagdefining]%allocmem;
	}
	return(G__struct.size[G__tagdefining]-size);
      }
      /***********************************
       * In case of union
       ***********************************/
      else if(G__struct.type[G__tagdefining]=='u') {
	if(G__struct.size[G__tagdefining]<size) {
	  G__struct.size[G__tagdefining] = size;
	  if((size%2)==1) 
	    G__struct.size[G__tagdefining]++;
	}
	return(0);
      }
#ifndef G__OLDIMPLEMENTATION612
      /***********************************
       * In case of namespace
       ***********************************/
      else if(G__struct.type[G__tagdefining]=='n') {
	allocmem=(long)calloc((size_t)n ,(size_t)bsize);
	if(allocmem==(long)NULL) G__malloc_error(item);
	return(allocmem);
      }
#endif
    }
  }
  else {
    /********************************************
     * Get address from compiled object
     * No need to calculate offset.
     ********************************************/
    return(G__globalvarpointer);
  }
  return(-1); /* this should never happen, avoiding lint error */
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
