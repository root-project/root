/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file memtest.c
 ************************************************************************
 * Description:
 *  Memory leak test utility.
 * Comment:
 *  This source is usually unnecessary. 
 ************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#define G__MEMTEST_C
#define G__MEMTEST_H
#include "common.h"

extern "C" {

#if !defined(G__SMALLOBJECT) && defined(G__DEBUG)

#define G__MEMTEST
#define G__MALLOCSIZE 10000

#define G__SAVEMEMORY
#define G__DUMPMEMHISTORY

#define G__TYPE_FOPEN   "fopen()"
#define G__TYPE_MALLOC  "malloc()"

extern struct G__input_file G__ifile;

typedef struct {
  void *p;
  int alive;
  int use;
  int size;
  char *type;
} G__memtest;

G__memtest G__mem[G__MALLOCSIZE];
int G__imem=0;
FILE *G__memhist=NULL;


#ifndef G__OLDIMPLEMENTATION2226
/*************************************************************************
* G__break_memtest
*
*************************************************************************/
int G__nth_memory = -1;
int G__mth_event = -1;
void G__break_memtest(i,comment) 
int i;
char* comment;
{
  static int m=1;
  if(G__nth_memory==i) {
    if(G__mth_event==m || G__mth_event==0) {
      printf("0x%lx=%s()\talive=%d\tuse=%d i=%d %dth FILE:%s LINE:%d\n"
           ,(long)G__mem[i].p,comment,G__mem[i].alive,G__mem[i].use,i,m
           ,G__ifile.name,G__ifile.line_number);
      /*G__pause();*/
    }
    ++m;
  }
  else if(i<-1) {
    printf("");
  }
}
#endif

/*************************************************************************
* G__TEST_Malloc()
*
*************************************************************************/
void *G__TEST_Malloc(size)
size_t size;
{
  int i=0;
  int j;
  void *result;
  char *pc;
  
  result = malloc(size);
  
  /*******************************************************
   * Set strange data 
   *******************************************************/
  pc=(char *)result;
  for(j=0;j<(int)size;j++) {
    *(pc+j) = (char)0xa3;
  }
  
#ifdef G__SAVEMEMORY
  while(G__mem[i].alive!=0 && i<G__MALLOCSIZE) ++i;
  if(i>=G__MALLOCSIZE) {
    G__genericerror("!!! Sorry memory parity checker capacity overflow");
  }
  if(i>=G__imem) {
    G__imem=i+1;
  }
  G__mem[i].p = result;
  G__mem[i].alive = 1;
  G__mem[i].size = size;
  G__mem[i].type= G__TYPE_MALLOC;
#else
  while(i<G__imem && G__mem[i].p!=result) ++i;
  
  if(i<G__imem) {
    G__mem[i].alive++;
    G__mem[i].use = G__mem[i].use+1;
    G__mem[i].size = size;
    G__mem[i].type= G__TYPE_MALLOC;
  }
  else {
    G__mem[i].p = result;
    G__mem[i].alive = 1;
    G__mem[i].use = 1 ;
    G__mem[i].size = size;
    G__mem[i].type= G__TYPE_MALLOC;
    ++G__imem;
  }
#endif
#ifdef G__DUMPMEMHISTORY
  fprintf(G__memhist
          ,"0x%lx=malloc(%d)\talive=%d\tuse=%d i=%d FILE:%s LINE:%d\n"
          ,(long)G__mem[i].p,size,G__mem[i].alive,G__mem[i].use,i
          ,G__ifile.name,G__ifile.line_number);
  fflush(G__memhist);
#ifndef G__OLDIMPLEMENTATION2226
  G__break_memtest(i,"malloc");
#endif
#endif
  return(result);
}

/*************************************************************************
 * G__TEST_Calloc()
 *
 *************************************************************************/
void *G__TEST_Calloc(n,bsize)
size_t n,bsize;
{
  int i=0;
  /* int j; */
  void *result;
  /* char *pc; */
  int size;
  
  size=n*bsize;
  result = calloc(n,bsize);
  
  /*******************************************************
   * Set strange data 
   *******************************************************/
  /*
    pc=(char *)result;
    for(j=0;j<size;j++) {
    *(pc+j) = (char)0xa3;
    }
    */
  
#ifdef G__SAVEMEMORY
  while(G__mem[i].alive!=0 && i<G__MALLOCSIZE) ++i;
  if(i>=G__MALLOCSIZE) {
    G__genericerror("!!! Sorry memory parity checker capacity overflow");
  }
  if(i>=G__imem) {
    G__imem=i+1;
  }
  G__mem[i].p = result;
  G__mem[i].alive = 1;
  G__mem[i].size = size;
  G__mem[i].type= G__TYPE_MALLOC;
#else
  while(i<G__imem && G__mem[i].p!=result) ++i;
  
  if(i<G__imem) {
    G__mem[i].alive++;
    G__mem[i].use = G__mem[i].use+1;
    G__mem[i].size = size;
    G__mem[i].type= G__TYPE_MALLOC;
  }
  else {
    G__mem[i].p = result;
    G__mem[i].alive = 1;
    G__mem[i].use = 1 ;
    G__mem[i].size = size;
    G__mem[i].type= G__TYPE_MALLOC;
    ++G__imem;
  }
#endif
#ifdef G__DUMPMEMHISTORY
  fprintf(G__memhist
          ,"0x%lx=calloc(%d,%d)\talive=%d\tuse=%d i=%d FILE:%s LINE:%d\n"
          ,(long)G__mem[i].p,n,bsize,G__mem[i].alive,G__mem[i].use,i
          ,G__ifile.name,G__ifile.line_number);
  fflush(G__memhist);
#ifndef G__OLDIMPLEMENTATION2226
  G__break_memtest(i,"calloc");
#endif
#endif
  return(result);
}

/*************************************************************************
 * G__TEST_Free()
 *
 *************************************************************************/
void G__TEST_Free(p)
void *p;
{
  int i=0;
  int j;
  char *pc;
  
  while(i<G__imem && G__mem[i].p!=p) ++i;
  if(i<G__imem) {
    G__mem[i].alive--;
    
    /***************************************************
     * Set strange data
     ***************************************************/
    pc = (char *)p;
    for(j=0;j<G__mem[i].size;j++) {
      *(pc+j) = (char)0xa5;
    }
    
#ifdef G__DUMPMEMHISTORY
    if(G__mem[i].alive<0)
      fprintf(G__memhist
              ,"0x%lx=free()\talive=%d\tuse=%d i=%d FILE:%s LINE:%d\n"
              ,(long)G__mem[i].p,G__mem[i].alive,G__mem[i].use,i
              ,G__ifile.name,G__ifile.line_number);
    else
      fprintf(G__memhist
              ,"0x%lx=free()\talive=%d\tuse=%d i=%d FILE:%s LINE:%d\n"
              ,(long)G__mem[i].p,G__mem[i].alive,G__mem[i].use,i
              ,G__ifile.name,G__ifile.line_number);
#ifndef G__OLDIMPLEMENTATION2226
    G__break_memtest(i,"free");
#endif
#endif
  }
  else {
#ifdef G__DUMPMEMHISTORY
    fprintf(G__memhist ,"free(0x%lx) not allocated FILE:%s LINE:%d\n"
            ,(long)p,G__ifile.name,G__ifile.line_number);
#endif
    G__fprinterr(G__serr,"free(0x%lx) not allocated",(long)p);
    G__printlinenum();
#ifndef G__OLDIMPLEMENTATION2226
    G__break_memtest(-2,"free");
#endif
  }
#ifdef G__DUMPMEMHISTORY
  fflush(G__memhist);
#endif

  free(p);
}

/*************************************************************************
 * G__TEST_Realloc()
 *
 *************************************************************************/
void *G__TEST_Realloc(p,size)
void *p;
size_t size;
{
  int i=0;
  int j;
  char *pc;
  void *tmp;
  
  if((void*)NULL==p) {
    return(G__TEST_Malloc(size));
  }
  if(0==size) {
    G__TEST_Free(p);
    return((void*)NULL);
  }
  
  tmp=realloc(p,size);
  
  while(i<G__imem && G__mem[i].p!=p) ++i;
  if(i<G__imem) {
    
#ifdef G__DUMPMEMHISTORY
    fprintf(G__memhist
            ,"0x%lx=realloc(0x%lx,%d)\talive=%d\tuse=%d i=%d FILE:%s LINE:%d\n"
            ,(long)tmp,(long)G__mem[i].p,size ,G__mem[i].alive,G__mem[i].use,i
            ,G__ifile.name,G__ifile.line_number);
#ifndef G__OLDIMPLEMENTATION2226
    G__break_memtest(i,"realloc");
#endif
#endif
    
    if(tmp) {
      G__mem[i].p=tmp;
      /***************************************************
       * Set strange data
       ***************************************************/
      if(G__mem[i].size<(int)size) {
        pc = (char *)G__mem[i].p;
        for(j=G__mem[i].size;j<(int)size;j++) {
          *(pc+j) = (char)0xa5;
        }
      }
      G__mem[i].size = size;
    }
    
  }
  else {
#ifdef G__DUMPMEMHISTORY
    fprintf(G__memhist ,"realloc(0x%lx,%d) not allocated FILE:%s LINE:%d\n"
            ,(long)p,size,G__ifile.name,G__ifile.line_number);
#endif
    G__fprinterr(G__serr,"realloc(0x%lx,%ld) not allocated",(long)p,size);
    G__printlinenum();
#ifndef G__OLDIMPLEMENTATION2226
    G__break_memtest(-2,"realloc");
#endif
  }
#ifdef G__DUMPMEMHISTORY
  fflush(G__memhist);
#endif
  
  return(tmp);
}

/*************************************************************************
 * G__memanalysis()
 *
 *************************************************************************/
int G__memanalysis()
{
  int i;
  if(G__memhist==NULL) G__memhist=fopen("G__memhist","w");
#ifdef G__SAVEMEMORY
  for(i=0;i<G__MALLOCSIZE;i++) {
    G__mem[i].p=0;
    G__mem[i].alive=0;
  }
#endif
  return(0);
}

/*************************************************************************
 * G__memresult()
 *
 *************************************************************************/
int G__memresult()
{
  int i;
  fprintf(G__memhist,"\n================MALLOC STATUS===============\n");
  for(i=0;i<G__imem;i++) {
    if(G__mem[i].alive) {
#ifdef G__DUMPMEMHISTORY
      fprintf(G__memhist ,"0x%lx\talive=%d\tuse=%d i=%d\tERROR %s\n"
              ,(long)G__mem[i].p,G__mem[i].alive,G__mem[i].use,i ,G__mem[i].type);
#endif
      G__fprinterr(G__serr,"0x%lx\talive=%d\tuse=%d i=%d\tERROR %s\n"
              ,(long)G__mem[i].p,G__mem[i].alive,G__mem[i].use,i ,G__mem[i].type);
    }
    else {
#ifdef G__DUMPMEMHISTORY
      fprintf(G__memhist ,"0x%lx\talive=%d\tuse=%d i=%d %s\n"
              ,(long)G__mem[i].p,G__mem[i].alive,G__mem[i].use,i ,G__mem[i].type);
#endif
    }
  }
#ifdef G__DUMPMEMHISTORY
  fprintf(G__memhist,"\n====================END=====================\n");
  fflush(G__memhist);
#endif
  return(0);
}


/*************************************************************************
 * G__DUMMY_Free()
 *
 *************************************************************************/
void G__DUMMY_Free(p)
void *p;
{
  free(p);
}




/*************************************************************************
 * G__TEST_fopen()
 *
 *************************************************************************/
void *G__TEST_fopen(fname,mode)
char *fname,*mode;
{
  int i=0;
  /* int j; */
  FILE *result;
  /* char *pc; */
  
  result = fopen(fname,mode);
  if(result==NULL) return(result);
  
#ifdef G__SAVEMEMORY
  while(G__mem[i].alive!=0 && i<G__MALLOCSIZE) ++i;
  if(i>=G__MALLOCSIZE) {
    G__genericerror("!!! Sorry memory parity checker capacity overflow");
  }
  if(i>=G__imem) {
    G__imem=i+1;
  }
  G__mem[i].p = (void *)result;
  G__mem[i].alive = 1;
  G__mem[i].size = 0;
  G__mem[i].type= G__TYPE_FOPEN;
#else
  while(i<G__imem && G__mem[i].p!=result) ++i;
  
  if(i<G__imem) {
    G__mem[i].alive++;
    G__mem[i].use = G__mem[i].use+1;
    G__mem[i].size = 0;
    G__mem[i].type= G__TYPE_FOPEN;
  }
  else {
    G__mem[i].p = (void *)result;
    G__mem[i].alive = 1;
    G__mem[i].use = 1 ;
    G__mem[i].size = 0;
    G__mem[i].type= G__TYPE_FOPEN;
    ++G__imem;
  }
#endif
#ifdef G__DUMPMEMHISTORY
  fprintf(G__memhist
          ,"0x%lx=fopen(%s,%s)\talive=%d\tuse=%d i=%d FILE:%s LINE:%d\n"
          ,(long)G__mem[i].p,fname,mode,G__mem[i].alive,G__mem[i].use,i
          ,G__ifile.name,G__ifile.line_number);
  fflush(G__memhist);
#endif
  return(result);
}

/*************************************************************************
 * G__TEST_fclose()
 *
 *************************************************************************/
int G__TEST_fclose(p)
FILE *p;
{
  int i=0;
  /* int j; */
  
  while(i<G__imem && G__mem[i].p!=(void *)p) ++i;
  if(i<G__imem) {
    G__mem[i].alive--;
    
#ifdef G__DUMPMEMHISTORY
    fprintf(G__memhist
            ,"0x%lx=fclose()\talive=%d\tuse=%d i=%d FILE:%s LINE:%d\n"
            ,(long)G__mem[i].p,G__mem[i].alive,G__mem[i].use,i
            ,G__ifile.name,G__ifile.line_number);
#endif
  }
  else {
#ifdef G__DUMPMEMHISTORY
    fprintf(G__memhist ,"fclose(0x%lx) not opened FILE:%s LINE:%d\n"
            ,(long)p,G__ifile.name,G__ifile.line_number);
#endif
    G__fprinterr(G__serr,"fclose(0x%lx) not opened",(long)p);
    G__printlinenum();
  }
#ifdef G__DUMPMEMHISTORY
  fflush(G__memhist);
#endif
  
  return(fclose(p));
}


/*************************************************************************
 * G__TEST_tmpfile()
 *
 *************************************************************************/
void *G__TEST_tmpfile()
{
  int i=0;
  /* int j; */
  FILE *result;
  /* char *pc; */
  
  result = tmpfile();
  if(result==NULL) return(result);
  
#ifdef G__SAVEMEMORY
  while(G__mem[i].alive!=0 && i<G__MALLOCSIZE) ++i;
  if(i>=G__MALLOCSIZE) {
    G__genericerror("!!! Sorry memory parity checker capacity overflow");
  }
  if(i>=G__imem) {
    G__imem=i+1;
  }
  G__mem[i].p = (void *)result;
  G__mem[i].alive = 1;
  G__mem[i].size = 0;
  G__mem[i].type= G__TYPE_FOPEN;
#else
  while(i<G__imem && G__mem[i].p!=result) ++i;
  
  if(i<G__imem) {
    G__mem[i].alive++;
    G__mem[i].use = G__mem[i].use+1;
    G__mem[i].size = 0;
    G__mem[i].type= G__TYPE_FOPEN;
  }
  else {
    G__mem[i].p = (void *)result;
    G__mem[i].alive = 1;
    G__mem[i].use = 1 ;
    G__mem[i].size = 0;
    G__mem[i].type= G__TYPE_FOPEN;
    ++G__imem;
  }
#endif
#ifdef G__DUMPMEMHISTORY
  fprintf(G__memhist
          ,"0x%lx=tmpfile()\talive=%d\tuse=%d i=%d FILE:%s LINE:%d\n"
          ,(long)G__mem[i].p,G__mem[i].alive,G__mem[i].use,i
          ,G__ifile.name,G__ifile.line_number);
  fflush(G__memhist);
#endif
  return(result);
}

#endif /* G__SMALLOBJECT */

#ifndef G__OLDIMPLEMENTATION2226
#ifdef G__DEBUG
void G__setmemtestbreak(int n,int m)
#else
void G__setmemtestbreak(int,int)
#endif
{
#ifdef G__DEBUG
  G__nth_memory = n;
  G__mth_event = m;
#endif
}
#endif

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
