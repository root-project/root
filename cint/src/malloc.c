/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file malloc.c
 ************************************************************************
 * Description:
 *  Allocate automatic variable arena 
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto (MXJ02154@niftyserve.or.jp)
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

#ifndef G__OLDIMPLEMENTATION514
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
  fprintf(G__serr,"Error: No memory for static %s ",temp);
  G__genericerror((char*)NULL);
  return(0);
}
#endif

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
#ifdef G__OLDIMPLEMENTATION514
  char temp[G__ONELINE];
  int hash,i;
  struct G__var_array *var;
#endif

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
#ifndef G__OLDIMPLEMENTATION514
	return(G__getstaticobject());
#else
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
	fprintf(G__serr,"Error: No memory for static %s ",temp);
	G__genericerror((char*)NULL);
#endif
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
#ifndef G__OLDIMPLEMENTATION514
	  if(G__ASM_FUNC_COMPILE==G__asm_wholefunction) {
	    return(G__getstaticobject());
	  }
	  else {
	    allocmem=(long)calloc((size_t)n ,(size_t)bsize);
	    if(allocmem==(long)NULL) G__malloc_error(item);
	  }
#else
	  allocmem=(long)calloc((size_t)n ,(size_t)bsize);
	  if(allocmem==(long)NULL) G__malloc_error(item);
#endif
	  return(allocmem);
	}
	/***********************************
	 * Get padding size
	 ***********************************/
	if(bsize>G__DOUBLEALLOC) allocmem=G__DOUBLEALLOC;
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
