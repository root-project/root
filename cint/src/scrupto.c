/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file scrupto.c
 ************************************************************************
 * Description:
 *  Partial cleanup function
 ************************************************************************
 * Copyright(c) 1995~2004  Masaharu Goto 
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


/***********************************************************************
* How to use environment rewinding feature
*
*    G__dictposition pos;
*    G__store_dictposition(&pos);
*           .
*    // Do anything, 
*           .
*    if(!G__isfuncbusy(pos.nfile)) G__scratch_upto(&pos);
*
*
***********************************************************************/


/***********************************************************************
* void G__store_dictposition()
*
*
***********************************************************************/
void G__store_dictposition(dictpos)
struct G__dictposition *dictpos;
{
#ifndef G__OLDIMPLEMENTATION1035
  G__LockCriticalSection();
#endif
  /* global variable position */
  dictpos->var = &G__global;
  while(dictpos->var->next) dictpos->var=dictpos->var->next;
  dictpos->ig15 = dictpos->var->allvar;

  dictpos->tagnum = G__struct.alltag;
  dictpos->conststringpos = G__plastconststring;
  dictpos->typenum = G__newtype.alltype;

  /* global function position */
  dictpos->ifunc = &G__ifunc;
  while(dictpos->ifunc->next) dictpos->ifunc=dictpos->ifunc->next;
  dictpos->ifn = dictpos->ifunc->allifunc;
  
  /* include path */
  dictpos->ipath = &G__ipathentry;
  while(dictpos->ipath->next) dictpos->ipath=dictpos->ipath->next;

  /* preprocessfilekey */
  dictpos->preprocessfilekey = &G__preprocessfilekey;
  while(dictpos->preprocessfilekey->next) 
    dictpos->preprocessfilekey=dictpos->preprocessfilekey->next;

#ifdef G__SHAREDLIB
  dictpos->allsl = G__allsl;
#endif

  dictpos->nfile = G__nfile;

  /* function macro */
  dictpos->deffuncmacro = &G__deffuncmacro;
  while(dictpos->deffuncmacro->next)
    dictpos->deffuncmacro=dictpos->deffuncmacro->next;

  /* template class */
  dictpos->definedtemplateclass = &G__definedtemplateclass;
  while(dictpos->definedtemplateclass->next)
    dictpos->definedtemplateclass=dictpos->definedtemplateclass->next;

  /* function template */
  dictpos->definedtemplatefunc = &G__definedtemplatefunc;
  while(dictpos->definedtemplatefunc->next)
    dictpos->definedtemplatefunc=dictpos->definedtemplatefunc->next;

#ifndef G__OLDIMPLEMENTATION2014
#ifndef G__OLDIMPLEMENTATION2024
  if(0!=dictpos->ptype && (char*)G__PVOID!=dictpos->ptype) {
    free((void*)dictpos->ptype);
    dictpos->ptype = (char*)NULL;
  }
#endif
  if(0==dictpos->ptype) {
    int i;
    dictpos->ptype = (char*)malloc(G__struct.alltag+1);
    for(i=0;i<G__struct.alltag;i++) dictpos->ptype[i] = G__struct.type[i];
  }
#endif

#ifndef G__OLDIMPLEMENTATION1035
  G__UnlockCriticalSection();
#endif
}

/***********************************************************************
* void G__scratch_upto()
*
*
***********************************************************************/
void G__scratch_upto(dictpos)
struct G__dictposition *dictpos;
{
  /* int i; */

  /* struct G__var_array *local; */

#ifndef G__OLDIMPLEMENTATION1207
  if(!dictpos) return;
#endif

#ifndef G__OLDIMPLEMENTATION1035
  G__LockCriticalSection();
#endif

#ifdef G__MEMTEST
  fprintf(G__memhist,"G__scratch_upto() start\n");
#endif


  /*******************************************
   * clear interpriveve global variables
   *******************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing global variables\n");
#endif
#ifndef G__FONS22
  G__destroy_upto(dictpos->var,1,dictpos->var,dictpos->ig15);
#else
  G__destroy_upto(&G__global,1,dictpos->var,dictpos->ig15);
#endif

#ifdef G__SECURITY
  /*************************************************************
   * Garbage collection
   *************************************************************/
  if(G__security&G__SECURE_GARBAGECOLLECTION) {
    G__garbagecollection();
  }
#endif

  /*************************************************************
   * Free struct tag name and member table
   *************************************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing struct tag table\n");
#endif
  G__free_struct_upto(dictpos->tagnum);

#ifdef G__OLDIMPLEMENTATION1418
  /*************************************************************
   * Free string constant
   *************************************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing string constants\n");
#endif
  G__free_string_upto(dictpos->conststringpos);
#endif

  /*************************************************************
   * Free typedef table
   *************************************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing typedef table\n");
#endif
  G__free_typedef_upto(dictpos->typenum);

  /*************************************************************
   * Initialize interpreted function table list
   *************************************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing interpret function table\n");
#endif
  G__free_ifunc_table_upto(&G__ifunc,dictpos->ifunc,dictpos->ifn);


  /********************************************
   * free include path list
   *********************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing include path list\n");
#endif
  G__free_ipath(dictpos->ipath);

#ifdef G__SHAREDLIB
  /********************************************
   * free dynamic link library
   *********************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing shared library\n");
#endif
  G__free_shl_upto(dictpos->allsl);
#endif

  /********************************************
   * free preprocessfilekey list
   *********************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing preprocess file key list\n");
#endif
  G__free_preprocessfilekey(dictpos->preprocessfilekey);

#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing function macro list\n");
#endif
  G__freedeffuncmacro(dictpos->deffuncmacro);

#ifdef G__TEMPLATECLASS
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing template class list\n");
#endif
  G__freedeftemplateclass(dictpos->definedtemplateclass);
#endif

#ifdef G__TEMPLATEFUNC
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing template function list\n");
#endif
  G__freetemplatefunc(dictpos->definedtemplatefunc);
#endif

#ifndef G__OLDIMPLEMENTATION2190
 {
   int nfile = G__nfile;
   while(nfile > dictpos->nfile) {
     struct G__dictposition *dictposx = G__srcfile[nfile].dictpos;
     if(dictposx && dictposx->ptype && (char*)G__PVOID!=dictposx->ptype ) {
       free((void*)dictposx->ptype);
       dictposx->ptype = (char*)NULL;
     }
     --nfile;
   }
 }
#endif
#ifndef G__OLDIMPLEMENTATION2026
  if(dictpos->ptype && (char*)G__PVOID!=dictpos->ptype ) {
    int i;
    for(i=0; i<G__struct.alltag; i++) G__struct.type[i] = dictpos->ptype[i];
    free((void*)dictpos->ptype);
    dictpos->ptype = (char*)NULL;
  }
#endif

  /* close source files */
#ifdef G__MEMTEST
  fprintf(G__memhist,"Closing input files\n");
#endif
  G__close_inputfiles_upto(dictpos->nfile);

#ifndef G__OLDIMPLEMENTATION1045
  G__tagdefining = -1;
#endif

#ifndef G__OLDIMPLEMENTATION1035
  G__UnlockCriticalSection();
#endif
}


/***********************************************************************
* int G__free_ifunc_table_upto()
*
***********************************************************************/
int G__free_ifunc_table_upto(ifunc,dictpos,ifn)
struct G__ifunc_table *ifunc;
struct G__ifunc_table *dictpos;
int ifn;
{
  int i,j;
  if(ifunc->next) {
#ifndef G__OLDIMPLEMENTATION489
    if(G__free_ifunc_table_upto(ifunc->next,dictpos,ifn)) return(1);
#else
    if(G__free_ifunc_table(ifunc->next)) return(1);
#endif
    free((void*)ifunc->next);
    ifunc->next=(struct G__ifunc_table *)NULL;
  }
  /* Freeing default parameter storage */
#ifndef G__OLDIMPLEMENTATION1039
  if(ifunc==dictpos && ifn==ifunc->allifunc) {return(1);}
#endif
  for(i=ifunc->allifunc-1;i>=0;i--) {
#ifdef G__MEMTEST
    fprintf(G__memhist,"func %s\n",ifunc->funcname[i]);
#endif
#ifndef G__OLDIMPLEMENTATION1706
    if(ifunc->override_ifunc[i] 
       && ifunc->override_ifunc[i]->funcname[ifunc->override_ifn[i]][0]) {
      struct G__ifunc_table *overridden = ifunc->override_ifunc[i];
      int ior=ifunc->override_ifn[i];
      overridden->hash[ior] = ifunc->hash[i];
      overridden->masking_ifunc[ior]=(struct G__ifunc_table*)NULL;
      overridden->masking_ifn[ior]=0;
      for(j=ifunc->para_nu[i]-1;j>=0;j--) {
	if((G__value*)(-1)==overridden->para_default[ior][j] &&
	   (char*)NULL==overridden->para_def[ior][j]) {
	  overridden->para_default[ior][j]=ifunc->para_default[i][j];
	  overridden->para_def[ior][j]=ifunc->para_def[i][j];
	  ifunc->para_default[i][j]=(G__value*)NULL;
	  ifunc->para_def[i][j]=(char*)NULL;
	}
      }
    }
    if(ifunc->masking_ifunc[i]) {
      struct G__ifunc_table *masking= ifunc->masking_ifunc[i];
      int ims=ifunc->masking_ifn[i];
      masking->override_ifunc[ims]=(struct G__ifunc_table*)NULL;
      masking->override_ifn[ims]=0;
      ifunc->masking_ifunc[i]=(struct G__ifunc_table*)NULL;
      ifunc->masking_ifn[i]=0;
    }
#endif
#ifndef G__OLDIMPLEMENTATION1543
    if(ifunc->funcname[i]) {
      free((void*)ifunc->funcname[i]);
      ifunc->funcname[i] = (char*)NULL;
    }
#endif
#ifdef G__ASM_WHOLEFUNC
    if(
#ifndef G__OLDIMPLEMENTATION1567
       ifunc->pentry[i] && 
#endif
       ifunc->pentry[i]->bytecode) {
      G__free_bytecode(ifunc->pentry[i]->bytecode);
      ifunc->pentry[i]->bytecode = (struct G__bytecodefunc*)NULL;
    }
#endif
#ifdef G__FRIEND
    G__free_friendtag(ifunc->friendtag[i]);
#endif
    for(j=ifunc->para_nu[i]-1;j>=0;j--) {
      if(ifunc->para_name[i][j]) {
	free((void*)ifunc->para_name[i][j]);
	ifunc->para_name[i][j]=(char*)NULL;
      }
      if(ifunc->para_def[i][j]) {
	free((void*)ifunc->para_def[i][j]);
	ifunc->para_def[i][j]=(char*)NULL;
      }
      if(ifunc->para_default[i][j] &&
	 (&G__default_parameter)!=ifunc->para_default[i][j] &&
	 (G__value*)(-1)!=ifunc->para_default[i][j]) {
	free((void*)ifunc->para_default[i][j]);
	ifunc->para_default[i][j]=(G__value*)NULL;
      }
    }

    if(ifunc==dictpos && ifn==i) {
      ifunc->allifunc=ifn;
      return(1);
    }
  }
  ifunc->page=0;
  return(0);
  /* Do not free 'ifunc' because it can be a global/static object */
}

/***********************************************************************
* int G__free_string_upto()
*
* Can replace G__free_string();
*
***********************************************************************/
int G__free_string_upto(conststringpos)
struct G__ConstStringList *conststringpos;
{
  struct G__ConstStringList *pconststring;
  pconststring = G__plastconststring;
  while(pconststring && pconststring != conststringpos) {
    G__plastconststring = pconststring;
    pconststring = pconststring->prev;
    free((void*)G__plastconststring->string);
    free((void*)G__plastconststring);
  }
  G__plastconststring = conststringpos;
  return(0);
}

/***********************************************************************
* int G__free_typedef_upto()
*
* Can replace G__free_typedef();
*
***********************************************************************/
int G__free_typedef_upto(typenum)
int typenum;
{
  while((--G__newtype.alltype)>=typenum) {
    free((void*)G__newtype.name[G__newtype.alltype]);
    G__newtype.name[G__newtype.alltype]=(char *)NULL;
    if(G__newtype.nindex[G__newtype.alltype]) {
      free((void*)G__newtype.index[G__newtype.alltype]);
      G__newtype.nindex[G__newtype.alltype]=0;
    }
  }
  G__newtype.alltype=typenum;
  return(0);
}

/***********************************************************************
* int G__free_struct_upto()
*
* Can replace G__free_struct();
*
***********************************************************************/
int G__free_struct_upto(tagnum)
int tagnum;
{
  struct G__var_array *var;
  int i,j,done;
  char com[G__ONELINE];
  long store_struct_offset;
  int store_tagnum;
#ifndef G__OLDIMPLEMENTATION459
  int ialltag;
#endif

#ifndef G__OLDIMPLEMENTATION459
  /*****************************************************************
  * clearing static members
  *****************************************************************/
  for(ialltag=G__struct.alltag-1;ialltag>=tagnum;ialltag--) {
#ifndef G__OLDIMPLEMENTATION2014
    if(G__struct.libname[ialltag]) {
      free((void*)G__struct.libname[ialltag]);
      G__struct.libname[ialltag] = (char*)NULL;
    }
#endif
    if(G__NOLINK==G__struct.iscpplink[ialltag]) {
      /* freeing static member variable if not precompiled */
      var=G__struct.memvar[ialltag];
      while(var) {
	for(i=0;i<var->allvar;i++) {
#ifndef G__OLDIMPLEMENTATION684
	  if(G__LOCALSTATIC==var->statictype[i] && 
	     G__COMPILEDGLOBAL!=var->globalcomp[i] &&
	     G__PARANORMAL==var->reftype[i]) {
#else
	  if(G__LOCALSTATIC==var->statictype[i] && 
	     G__COMPILEDGLOBAL!=var->globalcomp[i]) {
#endif
	    if('u'==var->type[i]) {
	      /* Static class object member try destructor */
	      sprintf(com,"~%s()",G__struct.name[var->p_tagtable[i]]);
	      store_struct_offset=G__store_struct_offset;
	      store_tagnum=G__tagnum;
	      G__store_struct_offset = var->p[i];
	      G__tagnum=var->p_tagtable[i];
	      if(G__dispsource)
#ifndef G__FONS31
	      G__fprinterr(G__serr,
		      "!!!Destroy static member object 0x%lx %s::~%s()\n"
		      ,var->p[i] ,G__struct.name[ialltag]
		      ,G__struct.name[i]);
#else
	      G__fprinterr(G__serr,
		      "!!!Destroy static member object 0x%x %s::~%s()\n"
		      ,var->p[i] ,G__struct.name[ialltag]
		      ,G__struct.name[i]);
#endif
	      done=0;
	      for(j=var->varlabel[i][1];j>=0;j--) {
		G__getfunction(com,&done,G__TRYDESTRUCTOR);
		if(0==done) break;
		G__store_struct_offset+=G__struct.size[i];
	      }
	      G__store_struct_offset = store_struct_offset;
	      G__tagnum=store_tagnum;
	    }
#ifndef G__OLDIMPLEMENTATION2052
	    if(G__CPPLINK!=G__struct.iscpplink[var->p_tagtable[i]])
	      free((void*)var->p[i]);
#else
	    free((void*)var->p[i]);
#endif
	  }
#ifndef G__OLDIMPLEMENTATION1543
	  if(var->varnamebuf[i]) {
	    free((void*)var->varnamebuf[i]);
	    var->varnamebuf[i] = (char*)NULL;
	  }
#endif
	}
	var=var->next;
      }
    }
    else {
      var=G__struct.memvar[ialltag];
      while(var) {
	for(i=0;i<var->allvar;i++) {
	  /* need to free compiled enum value */
	  if(G__LOCALSTATIC==var->statictype[i] && 
#ifndef G__OLDIMPLEMENTATION614
	     G__COMPILEDGLOBAL!=var->globalcomp[i] &&
#endif
	     -1!=var->p_tagtable[i]&&'e'==G__struct.type[var->p_tagtable[i]]) {
	    free((void*)var->p[i]);
	  }
#ifndef G__OLDIMPLEMENTATION1543
	  if(var->varnamebuf[i]) {
	    free((void*)var->varnamebuf[i]);
	    var->varnamebuf[i] = (char*)NULL;
	  }
#endif
	}
	var=var->next;
      }
    }
  }
#endif

  /*****************************************************************
  * clearing class definition
  *****************************************************************/
  while((--G__struct.alltag)>=tagnum) {
#ifdef G__MEMTEST
    fprintf(G__memhist,"struct %s\n",G__struct.name[G__struct.alltag]);
#endif
#ifndef G__OLDIMPLEMENTATION2084
    G__bc_delete_vtbl(G__struct.alltag);    
#endif
#ifdef G__ROOTSPECIAL
    if(G__struct.rootspecial[G__struct.alltag]) {
      free((void*)G__struct.rootspecial[G__struct.alltag]);
    }
#endif
#ifdef G__FRIEND
    G__free_friendtag(G__struct.friendtag[G__struct.alltag]);
#endif
    /* freeing class inheritance table */
    free((void*)G__struct.baseclass[G__struct.alltag]);
    G__struct.baseclass[G__struct.alltag] = (struct G__inheritance *)NULL;

    /* freeing member function table */
    G__free_ifunc_table(G__struct.memfunc[G__struct.alltag]);
    free((void*)G__struct.memfunc[G__struct.alltag]);
    G__struct.memfunc[G__struct.alltag]=(struct G__ifunc_table *)NULL;

#ifdef G__OLDIMPLEMENTATION459
    if(G__NOLINK==G__struct.iscpplink[G__struct.alltag]) {
      /* freeing static member variable if not precompiled */
      var=G__struct.memvar[G__struct.alltag];
      while(var) {
	for(i=0;i<var->allvar;i++) {
#ifdef G__NEWINHERIT
	  if(G__LOCALSTATIC==var->statictype[i]) {
#else
	  if(G__LOCALSTATIC==var->statictype[i] && 0==var->isinherit[i]) {
#endif
	    if('u'==var->type[i]) {
	      /* Static class object member try destructor */
	      sprintf(com,"~%s()",G__struct.name[var->p_tagtable[i]]);
	      store_struct_offset=G__store_struct_offset;
	      store_tagnum=G__tagnum;
	      G__store_struct_offset = var->p[i];
	      G__tagnum=var->p_tagtable[i];
	      if(G__dispsource)
#ifndef G__FONS31
	      G__fprinterr(G__serr,
		      "!!!Destroy static member object 0x%lx %s::~%s()\n"
		      ,var->p[i] ,G__struct.name[G__struct.alltag]
		      ,G__struct.name[i]);
#else
	      G__fprinterr(G__serr,
		      "!!!Destroy static member object 0x%x %s::~%s()\n"
		      ,var->p[i] ,G__struct.name[G__struct.alltag]
		      ,G__struct.name[i]);
#endif
	      done=0;
	      for(j=var->varlabel[i][1];j>=0;j--) {
		G__getfunction(com,&done,G__TRYDESTRUCTOR);
		if(0==done) break;
		G__store_struct_offset+=G__struct.size[i];
	      }
	      G__store_struct_offset = store_struct_offset;
	      G__tagnum=store_tagnum;
	    }
	    free((void*)var->p[i]);
	  }
	}
	var=var->next;
      }
    }
    else {
      var=G__struct.memvar[G__struct.alltag];
      while(var) {
	for(i=0;i<var->allvar;i++) {
	  /* need to free compiled enum value */
#ifdef G__NEWINHERIT
	  if(G__LOCALSTATIC==var->statictype[i] && 
	     -1!=var->p_tagtable[i]&&'e'==G__struct.type[var->p_tagtable[i]]) {
	    free((void*)var->p[i]);
	  }
#else
	  if(G__LOCALSTATIC==var->statictype[i] && 0==var->isinherit[i]
	     -1!=var->p_tagtable[i]&&'e'==G__struct.type[var->p_tagtable[i]]) {
	    free((void*)var->p[i]);
	  }
#endif
	}
	var=var->next;
      }
    }
#endif /* ON459 */

    /* freeing member variable table */
    G__free_member_table(G__struct.memvar[G__struct.alltag]);
    free((void*)G__struct.memvar[G__struct.alltag]);
    G__struct.memvar[G__struct.alltag]=(struct G__var_array *)NULL;

    /* freeing tagname */
    free((void*)G__struct.name[G__struct.alltag]);
    G__struct.name[G__struct.alltag]=(char *)NULL;
  }
  G__struct.alltag=tagnum;
  return(0);
}

#ifndef G__FONS21
/***********************************************************************
* void G__scratch_globals_upto(dictpos)
*
***********************************************************************/
void G__scratch_globals_upto(dictpos)
struct G__dictposition *dictpos;
{
  /*******************************************
   * clear interpreted global variables
   *******************************************/
#ifndef G__OLDIMPLEMENTATION1035
  G__LockCriticalSection();
#endif
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing global variables\n");
#endif
  G__destroy_upto(dictpos->var,1,dictpos->var,dictpos->ig15);
#ifndef G__OLDIMPLEMENTATION1035
  G__UnlockCriticalSection();
#endif
}
#endif


/***********************************************************************
* void G__destroy_upto()
*
***********************************************************************/
/* destroy local variable and free memory*/
int G__destroy_upto(var,global,dictpos,ig15)
struct G__var_array *var;
int global;
struct G__var_array *dictpos;
int ig15;
{
  int itemp=0,itemp1=0;
  
#ifdef G__OLDIMPLEMENTATION1802
  char temp[G__ONELINE];
#endif
  int store_tagnum;
  long store_struct_offset; /* used to be int */
  int store_return;
  int store_prerun;
#ifndef G__FONS22
  int remain=ig15;
#else
  int remain=0;
#endif
  int cpplink;
  int i,size;
  long address;
  
  /*******************************************
   * If there are any sub var array list,
   * destroy it too.
   *******************************************/
  if(var->next) {
#ifndef G__OLDIMPLEMENTATION1081
    if(var->allvar==G__MEMDEPTH) {
#endif
#ifndef G__FONS22
      /* Never true */
      if(G__destroy_upto(var->next,global,dictpos,0)) return(1);
#else
      if(G__destroy_upto(var->next,global,dictpos,ig15)) return(1);
#endif
      free((void*)var->next);
      var->next=NULL;
#ifndef G__OLDIMPLEMENTATION1081
    }
    else {
      fprintf(stderr,"!!!Fatal Error: Interpreter memory overwritten by illegal access.!!!\n");
      fprintf(stderr,"!!!Terminate session!!!\n");
    }
#endif
  }
  
  for(itemp=var->allvar-1;itemp>=remain;itemp--) {

#ifdef G__FONS22
    if(var==dictpos && itemp==ig15-1) {
      var->allvar = ig15;
      return(1);
    }
#endif
    
    /*****************************************
     * If the variable is a function scope
     * static variable, it will be kept.
     * else (auto and static body) free
     * allocated memory area.
     *****************************************/
    if((var->statictype[itemp]!=G__LOCALSTATIC||global) &&
       var->statictype[itemp]!=G__COMPILEDGLOBAL) {
      
      cpplink=0;
      /****************************************************
       * If C++ class, destructor has to be called
       ****************************************************/
      if(var->type[itemp]=='u' && 0==G__ansiheader && 0==G__prerun) {
#ifndef G__OLDIMPLEMENTATION1802
	char vv[G__BUFLEN];
	char *temp=vv;
#endif
	
	store_struct_offset = G__store_struct_offset;
	G__store_struct_offset = var->p[itemp]; /* duplication */
	
	store_tagnum = G__tagnum;
	G__tagnum = var->p_tagtable[itemp];
	
	store_return=G__return;
	G__return=G__RETURN_NON;

#ifndef G__OLDIMPLEMENTATION1802
	if(strlen(G__struct.name[G__tagnum])>G__BUFLEN-5)
	  temp = (char*)malloc(strlen(G__struct.name[G__tagnum])+5);
#endif
	
	sprintf(temp,"~%s()",G__struct.name[G__tagnum]);
	if(G__dispsource) {
#ifndef G__FONS31
	  G__fprinterr(G__serr,"\n!!!Calling destructor 0x%lx.%s for %s ary%d:link%d"
		  ,G__store_struct_offset ,temp ,var->varnamebuf[itemp]
		  ,var->varlabel[itemp][1],G__struct.iscpplink[G__tagnum]);
#else
	  G__fprinterr(G__serr,"\n!!!Calling destructor 0x%x.%s for %s ary%d:link%d"
		  ,G__store_struct_offset ,temp ,var->varnamebuf[itemp]
		  ,var->varlabel[itemp][1],G__struct.iscpplink[G__tagnum]);
#endif
	}
	
	
	store_prerun = G__prerun;
	G__prerun=0;
	/********************************************************
	 * destruction of array 
	 ********************************************************/
	if(G__CPPLINK==G__struct.iscpplink[G__tagnum]) {
#ifndef G__OLDIMPLEMENTATION1552
	  if(G__AUTOARYDISCRETEOBJ==var->statictype[itemp]) {
	    long store_globalvarpointer = G__globalvarpointer;
	    size=G__struct.size[G__tagnum];
	    for(i=var->varlabel[itemp][1];i>=0;--i) {
	      G__store_struct_offset = var->p[itemp]+size*i;
	      G__globalvarpointer = G__store_struct_offset;
	      G__getfunction(temp,&itemp1,G__TRYDESTRUCTOR); 
	      if(0==itemp1) break;
	    }
	    G__globalvarpointer = store_globalvarpointer;
	    free((void*)var->p[itemp]);
	  }
	  else {
	    G__store_struct_offset = var->p[itemp];
	    if((i=var->varlabel[itemp][1])>0
#ifndef G__OLDIMPLEMENTATION2025
	       || var->paran[itemp]
#endif
	       ) 
	      G__cpp_aryconstruct=i+1;
	    G__getfunction(temp,&itemp1,G__TRYDESTRUCTOR); 
	    G__cpp_aryconstruct=0;
	  }
#else
	  G__store_struct_offset = var->p[itemp];
	  if((i=var->varlabel[itemp][1])>0) G__cpp_aryconstruct=i+1;
	  G__getfunction(temp,&itemp1,G__TRYDESTRUCTOR); 
	  G__cpp_aryconstruct=0;
#endif
	  cpplink=1;
	}
	else {
	  size=G__struct.size[G__tagnum];
	  for(i=var->varlabel[itemp][1];i>=0;--i) {
	    G__store_struct_offset = var->p[itemp]+size*i;
	    if(G__dispsource) {
#ifndef G__FONS31
	      G__fprinterr(G__serr,"\n0x%lx.%s",G__store_struct_offset,temp);
#else
	      G__fprinterr(G__serr,"\n0x%x.%s",G__store_struct_offset,temp);
#endif
	    }
	    G__getfunction(temp,&itemp1,G__TRYDESTRUCTOR); 
	    if(0==itemp1) break;
	  }
	}
#ifndef G__OLDIMPLEMENTATION1802
	if(vv!=temp) free((void*)temp);
#endif
	G__prerun = store_prerun;
	
	G__store_struct_offset = store_struct_offset;
	G__tagnum = store_tagnum;
	G__return=store_return;
      } /*  end of type=='u' */

      else if(G__security&G__SECURE_GARBAGECOLLECTION && 
#ifndef G__OLDIMPLEMENTATION545
	      (!G__no_exec_compile) &&
#endif
	      isupper(var->type[itemp]) && var->p[itemp]) {
	i=var->varlabel[itemp][1]+1;
	do {
	  --i;
	  address = var->p[itemp] + G__LONGALLOC*i;
	  if(*((long*)address)) {
	    G__del_refcount((void*)(*((long*)address))
			    ,(void**)address);
	  }
	} while(i);
      }

#ifdef G__MEMTEST
      fprintf(G__memhist,"Free(%s)\n",var->varnamebuf[itemp]);
#endif
      if(G__NOLINK==cpplink && var->p[itemp]) free((void*)var->p[itemp]);
      
    } /* end of statictype==LOCALSTATIC or COMPILEDGLOBAL */
    
#ifdef G__DEBUG
    else if(G__memhist) {
#ifndef G__FONS31
      fprintf(G__memhist
	      ,"0x%lx (%s) not freed localstatic or compiledglobal FILE:%s LINE:%d\n"
	      ,var->p[itemp],var->varnamebuf[itemp]
	      ,G__ifile.name,G__ifile.line_number);
#else
      fprintf(G__memhist
	      ,"0x%x (%s) not freed localstatic or compiledglobal FILE:%s LINE:%d\n"
	      ,var->p[itemp],var->varnamebuf[itemp]
	      ,G__ifile.name,G__ifile.line_number);
#endif
    } 
#endif
    
    /* initialize varpointer and varlabel */
    /* var->varpointer[itemp]=0;*/
    for(itemp1=0;itemp1<G__MAXVARDIM;itemp1++) {
      var->varlabel[itemp][itemp1]=0;
    }
#ifndef G__OLDIMPLEMENTATION1543
    if(var->varnamebuf[itemp]) {
      free((void*)var->varnamebuf[itemp]);
      var->varnamebuf[itemp] = (char*)NULL;
    }
#endif

  }
  
  var->allvar = remain;
  return(0);
  
}


/***********************************************************************
* G__close_inputfiles_upto()
*
* Can not replace G__close_inputfiles()
*
***********************************************************************/
void G__close_inputfiles_upto(nfile)
int nfile;
{
#ifndef G__OLDIMPLEMENTATION1207
#ifdef G__SHAREDLIB
  struct G__filetable permanentsl[G__MAX_SL];
  int nperm=0;
#endif
#endif

#ifdef G__DUMPFILE
  if(G__dumpfile) G__dump_tracecoverage(G__dumpfile);
#endif
  while(G__nfile>nfile) {
    --G__nfile;
    if(G__srcfile[G__nfile].dictpos) {
      free((void*)G__srcfile[G__nfile].dictpos);
      G__srcfile[G__nfile].dictpos=(struct G__dictposition*)NULL;
    }
#ifndef G__OLDIMPLEMENTATION1273
    if(G__srcfile[G__nfile].hasonlyfunc) {
      free((void*)G__srcfile[G__nfile].hasonlyfunc);
      G__srcfile[G__nfile].hasonlyfunc=(struct G__dictposition*)NULL;
    }
#endif
#ifndef G__OLDIMPLEMENTATION1207
#ifdef G__SHAREDLIB
    if(G__srcfile[G__nfile].ispermanentsl) {
      permanentsl[nperm++] = G__srcfile[G__nfile];
      continue;
    }
#endif
#endif
    if(G__srcfile[G__nfile].fp) { 
      fclose(G__srcfile[G__nfile].fp);
#ifndef G__PHILIPPE0
      if(G__srcfile[G__nfile].prepname) {
	int j;
	for(j=G__nfile-1;j>=0;j--) {
	  if(G__srcfile[j].fp==G__srcfile[G__nfile].fp) 
	    G__srcfile[j].fp=(FILE*)NULL;
	}
      }
#endif
      G__srcfile[G__nfile].fp=(FILE*)NULL;
    }
    if(G__srcfile[G__nfile].breakpoint) {
      free((void*)G__srcfile[G__nfile].breakpoint);
      G__srcfile[G__nfile].breakpoint=(char*)NULL;
      G__srcfile[G__nfile].maxline=0;
    }
    if(G__srcfile[G__nfile].prepname) {
#ifndef G__OLDIMPLEMENTATION1920
      if('('!=G__srcfile[G__nfile].prepname[0])
	remove(G__srcfile[G__nfile].prepname);
#else
      remove(G__srcfile[G__nfile].prepname);
#endif
      free((void*)G__srcfile[G__nfile].prepname);
      G__srcfile[G__nfile].prepname=(char*)NULL;
    }
    if(G__srcfile[G__nfile].filename) {
#ifndef G__OLDIMPLEMENTATION1546
      int len = strlen(G__srcfile[G__nfile].filename);
      if(len>strlen(G__NAMEDMACROEXT2) && 
	 strcmp(G__srcfile[G__nfile].filename+len-strlen(G__NAMEDMACROEXT2),
		G__NAMEDMACROEXT2)==0) {
	remove(G__srcfile[G__nfile].filename);
      }
#endif
      free((void*)G__srcfile[G__nfile].filename);
      G__srcfile[G__nfile].filename=(char*)NULL;
    }
    G__srcfile[G__nfile].hash=0;
  }
  G__nfile=nfile;

#ifndef G__OLDIMPLEMENTATION1207
#ifdef G__SHAREDLIB
  while(nperm) {
    --nperm;
    G__srcfile[G__nfile++] = permanentsl[nperm];
    if(permanentsl[nperm].initsl) (*permanentsl[nperm].initsl)();
  }
#endif
#endif

  if(G__tempc[0]) {
    remove(G__tempc);
    G__tempc[0]='\0';
  }

  /*****************************************************************
  * Closing modified STDIOs.  May need to modify here.
  *  init.c, end.c, scrupto.c
  *****************************************************************/
  if(G__serr!=G__stderr && G__serr) {
    fclose(G__serr);
    G__serr=G__stderr;
  }
#ifndef G__OLDIMPLEMENTATION463
  if(G__sout!=G__stdout && G__sout) {
    fclose(G__sout);
    G__sout=G__stdout;
  }
  if(G__sin!=G__stdin && G__sin) {
    fclose(G__sin);
    G__sin=G__stdin;
  }
#endif
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
