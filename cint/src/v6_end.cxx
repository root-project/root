/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file end.c
 ************************************************************************
 * Description:
 *  Cleanup function
 ************************************************************************
 * Copyright(c) 1995~2001  Masaharu Goto 
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
* G__free_preprocessfilekey()
*
***********************************************************************/
void G__free_preprocessfilekey(pkey)
struct G__Preprocessfilekey *pkey;
{
  if(pkey->next) {
    G__free_preprocessfilekey(pkey->next);
    free((void*)pkey->next);
    pkey->next = (struct G__Preprocessfilekey*)NULL;
  }
  if(pkey->keystring) {
    free((void*)pkey->keystring);
    pkey->keystring=(char*)NULL;
  }
}

/***********************************************************************
* void G__scratch_all()
*
* Called by
*   G__main()
*   G__interpretexit();
*   G__exit()
*
***********************************************************************/
void G__scratch_all()
{
  struct G__var_array *local;

#ifndef G__OLDIMPLEMENTATION1035
  G__LockCriticalSection();
#endif

  G__lasterrorpos.line_number = 0;
  G__lasterrorpos.filenum = -1;
  

#ifndef G__OLDIMPLEMENTATION563
  G__cintready=0; /* reset ready flag for embedded use */
#endif

#ifdef G__MEMTEST
  fprintf(G__memhist,"G__scratch_all() start\n");
#endif

  /*******************************************
   * clear interpriveve global variables
   *******************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing local variables\n");
#endif
  local = G__p_local;
  while(local) {
    G__destroy(local,G__LOCAL_VAR);
    local=local->prev_local;
  }

  /*******************************************
   * clear interpriveve global variables
   *******************************************/
#ifndef G__OLDIMPLEMENTATION1559
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing temp object %d\n",G__templevel);
#endif
  if(G__p_tempbuf) {
    if(G__templevel>0) G__templevel = 0;
    G__free_tempobject();
  }
#endif

  /*******************************************
   * clear interpriveve global variables
   *******************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing global variables\n");
#endif
  G__destroy(&G__global,G__GLOBAL_VAR);

#ifndef G__OLDIMPLEMENTATION754
  /*******************************************
   * free exception handling buffer
   *******************************************/
  G__free_exceptionbuffer();
#endif

#ifdef G__SECURITY
  /*******************************************
   * garbage collection
   *******************************************/
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
  G__free_struct_upto(0);

  /*************************************************************
   * Free string constant
   *************************************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing string constants\n");
#endif
  G__free_string_upto(&G__conststringlist);

  /*************************************************************
   * Free typedef table
   *************************************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing typedef table\n");
#endif
  G__free_typedef_upto(0);

  /*************************************************************
   * Initialize interpreted function table list
   *************************************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing interpret function table\n");
#endif
  G__free_ifunc_table(&G__ifunc);
  G__ifunc.allifunc = 0;

  /********************************************
   * local variable is NULL at scratch 
   *********************************************/
  G__p_local = NULL;

  /********************************************
   * free include path list
   *********************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing include path list\n");
#endif
  G__free_ipath(&G__ipathentry);

#ifdef G__SHAREDLIB
  /********************************************
   * free dynamic link library
   *********************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing shared library\n");
#endif
  G__free_shl_upto(0);
#endif

  /********************************************
   * free preprocessfilekey list
   *********************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing preprocess file key list\n");
#endif
  G__free_preprocessfilekey(&G__preprocessfilekey);


#ifdef G__MEMTEST
  fprintf(G__memhist,"Closing macro file\n");
#endif
  if(G__mfp) {
#ifdef G__MEMTEST
    fprintf(G__memhist,"Ignore this error:");
    G__fprinterr(G__serr,"Ignore this error:");
#endif
    G__closemfp();
    G__mfp=NULL;
  }


  /* close source files */
#ifdef G__MEMTEST
  fprintf(G__memhist,"Closing input files\n");
#endif
  G__close_inputfiles();

#ifdef G__DUMPFILE
#ifdef G__MEMTEST
  fprintf(G__memhist,"Closing function call dump file\n");
  if(G__dumpfile && G__memhist!=G__dumpfile) fclose(G__dumpfile);
#else
  if(G__dumpfile) fclose(G__dumpfile);
#endif
  G__dumpfile=(FILE *)NULL;
#endif

  /* set function key */
  if(G__key!=0) system("key .cint_key -l execute");

#ifdef G__MEMTEST
  fprintf(G__memhist,"Closing readline dump files\n");
#endif
  while(G__dumpreadline[0]) {
    fclose(G__dumpreadline[0]);
    G__popdumpinput();
  }

#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing function macro list\n");
#endif
  G__freedeffuncmacro(&G__deffuncmacro);

#ifdef G__TEMPLATECLASS
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing template class list\n");
#endif
  G__freedeftemplateclass(&G__definedtemplateclass);
#endif

#ifdef G__TEMPLATEFUNC
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing template function list\n");
#endif
  G__freetemplatefunc(&G__definedtemplatefunc);
#endif

#ifndef G__OLDIMPLEMENTATION451
  /* delete user defined pragma statements */
  G__freepragma(G__paddpragma);
  G__paddpragma=(struct G__AppPragma*)NULL;
#endif

#ifndef G__OLDIMPLEMENTATION928
  if(G__allincludepath) {
    free(G__allincludepath);
    G__allincludepath=(char*)NULL;
  }
#endif

#ifndef G__OLDIMPLEMENTATION1451
  G__DeleteConstStringList(G__SystemIncludeDir);
  G__SystemIncludeDir = (struct G__ConstStringList*)NULL;
#endif

#ifndef G__OLDIMPLEMENTATION2034
  /* This implementation is premature in a sense that macro can not be 
   * rewound to file position */
  G__init_replacesymbol();
#endif

  /*************************************************************
   * Initialize cint body global variables
   *************************************************************/
#ifndef G__OLDIMPLEMENTATION1599
  G__init = 0;
#endif
  G__init_globals();

  G__reset_setup_funcs();

#ifndef G__OLDIMPLEMENTATION2227
  G__clear_errordictpos();
#endif

#ifdef G__MEMTEST
  G__memresult();
#endif

#ifndef G__OLDIMPLEMENTATION1035
  G__UnlockCriticalSection();
#endif
}



/***********************************************************************
* G__free_friendtag()
*
***********************************************************************/
void G__free_friendtag(friendtag)
struct G__friendtag *friendtag;
{
  if(friendtag 
#ifndef G__OLDIMPLEMENTATION1295
     && friendtag!=friendtag->next
#endif
     ) {
    G__free_friendtag(friendtag->next);
    free((void*)friendtag);
  }
}

/***********************************************************************
* int G__free_ifunc_table(ifunc)
*
* Called by
*   G__free_ifunc_table()  recursive call
*   G__free_struct()
*   G__scratch_all()
*
***********************************************************************/
int G__free_ifunc_table(ifunc)
struct G__ifunc_table *ifunc;
{
  int i,j;
#ifndef G__OLDIMPLEMENTATION1588
  int flag;
#endif
  if(ifunc->next) {
    G__free_ifunc_table(ifunc->next);
    free((void*)ifunc->next);
    ifunc->next=(struct G__ifunc_table *)NULL;
  }
  /* Freeing default parameter storage */
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
#ifndef G__OLDIMPLEMENTATION1588
      flag = 0;
#endif /* 1588 */
#ifndef G__OLDIMPLEMENTATION1543
    if(ifunc->funcname[i]) {
      free((void*)ifunc->funcname[i]);
      ifunc->funcname[i] = (char*)NULL;
#ifndef G__OLDIMPLEMENTATION1588
      flag = 1;
#endif /* 1588 */
    }
#endif
#ifdef G__ASM_WHOLEFUNC
    if(
#ifndef G__OLDIMPLEMENTATION1588
       flag &&
#endif
#ifndef G__OLDIMPLEMENTATION1501
       ifunc->pentry[i] && 
#endif
       ifunc->pentry[i]->bytecode) {
      G__free_bytecode(ifunc->pentry[i]->bytecode);
      ifunc->pentry[i]->bytecode = (struct G__bytecodefunc*)NULL;
    }
#endif
#ifdef G__FRIEND
#ifndef G__OLDIMPLEMENTATION1588
    if(flag) G__free_friendtag(ifunc->friendtag[i]);
#else
    G__free_friendtag(ifunc->friendtag[i]);
#endif
#endif
#ifndef G__OLDIMPLEMENTATION1588
    if(flag) {
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
#ifndef G__OLDIMPLEMENTATION1588
    }
#endif
  }
  ifunc->page=0;

  /* Do not free 'ifunc' because it can be a global/static object */
  return(0);
}

/***********************************************************************
* int G__free_member_table(ifunc)
*
*
***********************************************************************/
int G__free_member_table(mem)
struct G__var_array *mem;
{
  if(mem->next) {
    G__free_member_table(mem->next);
    free((void*)mem->next);
    mem->next=(struct G__var_array *)NULL;
  }
  return(0);
}


/***********************************************************************
* int G__free_ipath()
*
* Called by
*   G__free_ipath()   recursive call
*   G__scratch_all()
*
***********************************************************************/
int G__free_ipath(ipath)
struct G__includepath *ipath;
{
  if(ipath->next) {
    G__free_ipath(ipath->next);
    free((void*)ipath->next);
    ipath->next=(struct G__includepath *)NULL;
    free((void*)ipath->pathname);
    ipath->pathname=(char *)NULL;
  }
  return(0);
}






/***********************************************************************
* void G__destroy(var,isglobal)
***********************************************************************/
/* destroy local variable and free memory*/
void G__destroy(var,isglobal)
struct G__var_array *var;
int isglobal;
{
  int itemp=0,itemp1=0;
  
#ifdef G__OLDIMPLEMENTATION1802
  char temp[G__ONELINE];
#endif
  int store_tagnum;
  long store_struct_offset; /* used to be int */
  int store_return;
  int store_prerun;
  int remain=0;
  int cpplink;
  int i,size;
  long address;

#ifndef G__OLDIMPLEMENTATION2038
  /* This part is not needed in G__destroy_upto.  enclosing_scope and
   * inner_scope members are assigned only as local variable table for
   * bytecode function  in which case  G__destroy() is always used to
   * deallocate the table */
  var->enclosing_scope = (struct G__var_array*)NULL;
  if(var->inner_scope) {
    i=0;
    while(var->inner_scope[i]) {
      G__destroy(var->inner_scope[i],isglobal);
      free((void*)var->inner_scope[i]);
      ++i;
    }
  }
#endif
  
  /*******************************************
   * If there are any sub var array list,
   * destroy it too.
   *******************************************/
  if(var->next) {
#ifndef G__OLDIMPLEMENTATION1081
    if(var->allvar==G__MEMDEPTH) {
#endif
      G__destroy(var->next,isglobal);
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
    
    /*****************************************
     * If the variable is a function scope
     * static variable, it will be kept.
     * else (auto and static body) free
     * allocated memory area.
     *****************************************/
#ifdef G__ASM_WHOLEFUNC
    if(((var->statictype[itemp]!=G__LOCALSTATIC||G__GLOBAL_VAR==isglobal) && 
	var->statictype[itemp]!=G__COMPILEDGLOBAL &&
	G__BYTECODELOCAL_VAR!=isglobal)) {
       /* (G__BYTECODELOCAL_VAR==isglobal &&
	var->statictype[itemp]==G__LOCALSTATIC)) { */
#else
    if((var->statictype[itemp]!=G__LOCALSTATIC||isglobal) && 
       var->statictype[itemp]!=G__COMPILEDGLOBAL) {
#endif
      
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
	    if((i=var->varlabel[itemp][1])>0) G__cpp_aryconstruct=i+1;
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

#ifdef G__SECURITY
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
#endif

#ifdef G__MEMTEST
      fprintf(G__memhist,"Free(%s)\n",var->varnamebuf[itemp]);
#endif
      /* ??? Scott Snyder fixed as var->p[itemp]>0x10000 ??? */
      if(G__NOLINK==cpplink && var->p[itemp] 
#ifndef G__OLDIMPLEMENTATION1576
	 && -1!=var->p[itemp]
#endif
	 ) free((void*)var->p[itemp]);
      
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
  
}


/******************************************************************
* G__exit()
******************************************************************/
void G__exit(rtn)
int rtn;
{
	G__scratch_all();

#ifndef G__OLDIMPLEMENTATION463
	fflush(G__sout);
	fflush(G__serr);
#else
	fflush(stdout);
	fflush(stderr);
#endif
	exit(rtn);
}


/***********************************************************************
* G__call_atexit()
*
* Execute atexit function. atexit is reset before calling
* the function to avoid recursive atexit call.
***********************************************************************/
int G__call_atexit()
{
  char temp[G__ONELINE];
  if(G__breaksignal) G__fprinterr(G__serr,"!!! atexit() call\n");
  G__ASSERT(G__atexit);
  sprintf(temp,"%s()",G__atexit);
  G__atexit=NULL;
  G__getexpr(temp);
  return(0);
}


/***********************************************************************
* G__close_inputfiles()
*
* Called by
*    G__main()        When input file is not found.
*                     when '-c' option, this is not used.
*    G__interpretexit()
*    G__exit()
***********************************************************************/
int G__close_inputfiles()
{
  int iarg;
#ifdef G__DUMPFILE
  if(G__dumpfile) G__dump_tracecoverage(G__dumpfile);
#endif
  for(iarg=0;iarg<G__nfile;iarg++) {
    if(G__srcfile[iarg].dictpos) {
#ifndef G__OLDIMPLEMENTATION2227
      if(G__srcfile[iarg].dictpos->ptype &&
	 G__srcfile[iarg].dictpos->ptype!=(char*)G__PVOID) {
	free((void*)G__srcfile[iarg].dictpos->ptype);
      }
#endif
      free((void*)G__srcfile[iarg].dictpos);
      G__srcfile[iarg].dictpos=(struct G__dictposition*)NULL;
    }
#ifndef G__OLDIMPLEMENTATION1273
    if(G__srcfile[iarg].hasonlyfunc) {
      free((void*)G__srcfile[iarg].hasonlyfunc);
      G__srcfile[iarg].hasonlyfunc=(struct G__dictposition*)NULL;
    }
#endif
    if(G__srcfile[iarg].fp) { 
      fclose(G__srcfile[iarg].fp);
#ifndef G__PHILIPPE0
      if(G__srcfile[iarg].prepname) {
	int j;
	for(j=iarg+1;j<G__nfile;j++) {
	  if(G__srcfile[j].fp==G__srcfile[iarg].fp) 
	    G__srcfile[j].fp=(FILE*)NULL;
	}
      }
#endif
      G__srcfile[iarg].fp=(FILE*)NULL;
    }
    if(G__srcfile[iarg].breakpoint) {
      free((void*)G__srcfile[iarg].breakpoint);
      G__srcfile[iarg].breakpoint=(char*)NULL;
      G__srcfile[iarg].maxline=0;
    }
    if(G__srcfile[iarg].prepname) {
#ifndef G__OLDIMPLEMENTATION1920
      if('('!=G__srcfile[iarg].prepname[0]) remove(G__srcfile[iarg].prepname);
#else
      remove(G__srcfile[iarg].prepname);
#endif
      free((void*)G__srcfile[iarg].prepname);
      G__srcfile[iarg].prepname=(char*)NULL;
    }
    if(G__srcfile[iarg].filename) {
#ifndef G__OLDIMPLEMENTATION1546
      int len = strlen(G__srcfile[iarg].filename);
      if(len>(int)strlen(G__NAMEDMACROEXT2) && 
	 strcmp(G__srcfile[iarg].filename+len-strlen(G__NAMEDMACROEXT2),
		G__NAMEDMACROEXT2)==0) {
	remove(G__srcfile[iarg].filename);
      }
#endif
      free((void*)G__srcfile[iarg].filename);
      G__srcfile[iarg].filename=(char*)NULL;
    }
    G__srcfile[iarg].hash=0;
  }
  G__nfile=0;

  if(G__xfile[0]) {
    remove(G__xfile);
    G__xfile[0]='\0';
  }
  if(G__tempc[0]) {
    remove(G__tempc);
    G__tempc[0]='\0';
  }

  /*****************************************************************
  * Closing modified STDIOs.  May need to modify here.
  *  init.c, end.c, scrupto.c, pause.c
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
  return(0);
}


/***********************************************************************/
/* G__interpretexit()                                                  */
/*                                                                     */
/* Called by                                                           */
/*     G__main()                                                       */
/***********************************************************************/
int G__interpretexit()
{
  if(G__atexit) G__call_atexit();
  G__scratch_all();
  if(G__breaksignal) G__fprinterr(G__serr,"\nEND OF EXECUTION\n");
  return(0);
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
