/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file new.c
 ************************************************************************
 * Description:
 *  new delete
 ************************************************************************
 * Copyright(c) 1995~2003  Masaharu Goto 
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

#ifndef G__OLDIMPLEMENTATION1002
#ifdef G__ROOT
extern void* G__new_interpreted_object G__P((int size));
extern void G__delete_interpreted_object G__P((void* p));
#endif
#endif

/****************************************************************
* G__value G__new_operator()
* 
* Called by:
*   G__getpower()
*
*      V
*  new type
*  new type[10]
*  new type(53)
*  new (arena)type
*
****************************************************************/
G__value G__new_operator(expression)
char *expression;
{
  char arena[G__ONELINE];
#ifndef G__OLDIMPLEMENTATION585
  long memarena=0;
  int arenaflag=0;
#endif
  char construct[G__LONGLINE];
  char *type;
#ifndef G__OLDIMPLEMENTATION1246
  char *basictype;
#endif
  char *initializer;
  char *arrayindex;
  int p=0;
  int pinc;
  int size;
  int known;
  int i;
  long pointer=0;
  long store_struct_offset; /* used to be int */
  int store_tagnum;
  int store_typenum;
  int var_type=0;
  G__value result;
  int reftype=G__PARANORMAL;
  int typelen;
#ifndef G__OLDIMPLEMENTATION683
  int ispointer=0;
#endif
#ifndef G__OLDIMPLEMENTATION1052
  int typenum,tagnum;
#endif
#ifndef G__OLDIMPLEMENTATION1157
  int ld_flag = 0 ;
#endif


  G__CHECK(G__SECURE_MALLOC,1,return(G__null));

#ifndef G__OLDIMPLEMENTATION2087
  if(G__cintv6) {
    return(G__bc_new_operator(expression));
  }
#endif

  /******************************************************
   * get arena which is ignored due to limitation, however
   ******************************************************/
  if(expression[0]=='(') {
    G__getstream(expression+1,&p,arena,")");
    ++p;
    memarena = G__int(G__getexpr(arena));
    arenaflag=1;
#ifndef G__OLDIMPLEMENTATION705
#ifdef G__ASM
    if(G__asm_noverflow) {
      G__asm_inst[G__asm_cp] = G__SETGVP;
      G__asm_inst[G__asm_cp+1] = 0;
      G__inc_cp_asm(2,0);
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SETGVP 0\n",G__asm_cp);
#endif
    }
#endif
#endif /* ON705 */
  }
  else {
    arena[0]='\0';
  }
  
  /******************************************************
   * get initializer,arrayindex,type,pinc and size
   ******************************************************/
  type=expression+p;
  initializer=strchr(type,'(');
  arrayindex=strchr(type,'[');
  /* initializer and arrayindex are exclusive */
  if(initializer!=NULL && arrayindex!=NULL) {
    if(initializer<arrayindex) {
      arrayindex=NULL;
    }
    else {
      initializer=NULL;
    }
  }
#ifndef G__OLDIMPLEMENTATION1246
  if(initializer) *initializer = 0;
  basictype = G__strrstr(type,"::");
  if(initializer) *initializer = '(';
  if(!basictype) basictype = type;
  else basictype += 2;
#endif
#ifndef G__OLDIMPLEMENTATION1052
  typenum = G__defined_typename(type);
  if(-1!=typenum) tagnum = G__newtype.tagnum[typenum];
  else tagnum = -1;
#endif
  if(arrayindex) {
    pinc=G__getarrayindex(arrayindex);
    *arrayindex='\0';
#ifndef G__OLDIMPLEMENTATION1125
    if(-1==tagnum)  tagnum = G__defined_tagname(basictype,1);
#endif
#ifndef G__OLDIMPLEMENTATION1052
    if(-1!=tagnum) sprintf(construct,"%s()",G__struct.name[tagnum]);
#ifndef G__OLDIMPLEMENTATION1246
    else sprintf(construct,"%s()",basictype);
#else
    else sprintf(construct,"%s()",type);
#endif
#else /* 1052 */
    sprintf(construct,"%s()",type);
#endif /* 1052 */
#ifndef G__OLDIMPLEMENTATION911
    if(G__asm_wholefunction) G__abortbytecode(); 
#endif
  }
  else {
#ifdef G__ASM_IFUNC
    if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: LD %d from %x\n"
			     ,G__asm_cp,1 ,G__asm_dt);
#endif
      G__asm_inst[G__asm_cp]=G__LD;
      G__asm_inst[G__asm_cp+1]=G__asm_dt;
      G__asm_stack[G__asm_dt].obj.i=1;
      G__asm_stack[G__asm_dt].type='i';
      G__asm_stack[G__asm_dt].tagnum= -1;
      G__asm_stack[G__asm_dt].typenum= -1;
#ifndef G__OLDIMPLEMENTATION1157
      ld_flag = 1 ;
#else
      G__inc_cp_asm(2,1);
#endif
    }
#endif
    if(initializer) {
      pinc=1;
#ifndef G__OLDIMPLEMENTATION1658
      if(-1==tagnum) {
	*initializer = 0;
	tagnum = G__defined_tagname(basictype,1);
	*initializer = '(';
      }
#endif
#ifndef G__OLDIMPLEMENTATION1052
      if(-1!=tagnum) sprintf(construct,"%s%s"
			     ,G__struct.name[tagnum],initializer);
#ifndef G__OLDIMPLEMENTATION1246
      else strcpy(construct,basictype);
#else
      else strcpy(construct,type);
#endif
#else
      strcpy(construct,type);
#endif
      *initializer='\0';
    }
    else {
      pinc=1;
#ifndef G__OLDIMPLEMENTATION1052
      if(-1!=tagnum) sprintf(construct,"%s()",G__struct.name[tagnum]);
#ifndef G__OLDIMPLEMENTATION1246
      else sprintf(construct,"%s()",basictype);
#else
      else sprintf(construct,"%s()",type);
#endif
#else
      sprintf(construct,"%s()",type);
#endif
    }
  }
  
  size = G__Lsizeof(type);
  if(size == -1) {
    G__fprinterr(G__serr,"Error: type %s not defined FILE:%s LINE:%d\n"
	    ,type,G__ifile.name,G__ifile.line_number);
    return(G__null);
  }
  
  
  /******************************************************
   * Store member function executing environment
   ******************************************************/
  store_struct_offset = G__store_struct_offset;
  store_tagnum = G__tagnum;
  store_typenum = G__typenum;
#ifndef G__OLDIMPLEMENTATION1040
  result.ref=0;
#endif

#ifdef G__OLDIMPLEMENTATION1157
#ifdef G__ASM
  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SETMEMFUNCENV\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp]=G__SETMEMFUNCENV;
    G__inc_cp_asm(1,0);
  }
#endif
#endif /* 1157 */

  /******************************************************
   * pointer type idendification
   ******************************************************/
  typelen = strlen(type);
  while('*'==type[typelen-1]) {
#ifndef G__OLDIMPLEMENTATION683
    if(0==ispointer) ispointer=1;
    else {
      switch(reftype) {
      case G__PARANORMAL: reftype = G__PARAP2P; break;
#ifndef G__OLDIMPLEMENTATION707
      case G__PARAREFERENCE: break;
      default: ++reftype; break;
#else
      case G__PARAP2P: reftype = G__PARAP2P2P; break;
#endif
      }
    }
#else
    switch(reftype) {
    case G__PARANORMAL: reftype = G__PARAP2P; break;
    case G__PARAP2P: reftype = G__PARAP2P2P; break;
    }
#endif
    type[--typelen]='\0';
  }
  
  /******************************************************
   * identify type
   ******************************************************/
  G__typenum = G__defined_typename(type);
  if(G__typenum != -1) {
    G__tagnum=G__newtype.tagnum[G__typenum];
    var_type=G__newtype.type[G__typenum];
  }
  else {
    G__tagnum=G__defined_tagname(type,1); /* no error message */
    if(G__tagnum!= -1) {
      var_type='u';
    }
    else {
      if(strcmp(type,"int")==0) var_type='i';
      else if(strcmp(type,"char")==0) var_type='c';
      else if(strcmp(type,"short")==0) var_type='s';
      else if(strcmp(type,"long")==0) var_type='l';
      else if(strcmp(type,"float")==0) var_type='f';
      else if(strcmp(type,"double")==0) var_type='d';
      else if(strcmp(type,"void")==0) var_type='y';
      else if(strcmp(type,"FILE")==0) var_type='e';
      else if(strcmp(type,"unsignedint")==0) var_type='h';
      else if(strcmp(type,"unsignedchar")==0) var_type='b';
      else if(strcmp(type,"unsignedshort")==0) var_type='r';
      else if(strcmp(type,"unsignedlong")==0) var_type='l';
      else if(strcmp(type,"size_t")==0) var_type='l';
      else if(strcmp(type,"time_t")==0) var_type='l';
#ifndef G__OLDIMPLEMENTATION1604
      else if(strcmp(type,"bool")==0) var_type='g';
#endif
#ifndef G__OLDIMPLEMENTATION2189
      else if(strcmp(type,"longlong")==0) var_type='n';
      else if(strcmp(type,"unsignedlonglong")==0) var_type='m';
      else if(strcmp(type,"longdouble")==0) var_type='q';
#endif
    }
  }
#ifndef G__OLDIMPLEMENTATION683
  if(ispointer) var_type = toupper(var_type);
#endif

#ifndef G__OLDIMPLEMENTATION1157
#ifdef G__ASM
  if(G__asm_noverflow) {
    if(ld_flag) {
      if(-1==G__tagnum || G__CPPLINK!=G__struct.iscpplink[G__tagnum]
	 || ispointer || isupper(var_type)) {
	/* increment for LD 1, otherwise, cancel LD 1 */
	G__inc_cp_asm(2,1);
      }
      else {
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"Cancel LD 1\n");
#endif
      }
    }
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SETMEMFUNCENV\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp]=G__SETMEMFUNCENV;
    G__inc_cp_asm(1,0);
  }
#endif
#endif /* 1157 */
  
  /******************************************************
   * allocate memory if this is a class object and
   * not pre-compiled.
   ******************************************************/
  if(-1==G__tagnum || G__CPPLINK!=G__struct.iscpplink[G__tagnum]
#ifndef G__OLDIMPLEMENTATION881
     || ispointer
#endif
#ifndef G__OLDIMPLEMENTATION887
     || isupper(var_type)
#endif
     ) {
    if(G__no_exec_compile) pointer = pinc;
    else {
      if(arenaflag) pointer = memarena;
      else {
#ifndef G__OLDIMPLEMENTATION1002
#ifdef G__ROOT
	pointer = (long)G__new_interpreted_object(size*pinc);
#else
	pointer = (long)malloc( (size_t)(size*pinc) );
#endif
#else
	pointer = (long)malloc( (size_t)(size*pinc) );
#endif
      }
    }
    if(pointer==(long)NULL && 0==G__no_exec_compile) {
      G__fprinterr(G__serr,"Error: memory allocation for %s %s size=%d pinc=%d FILE:%s LINE:%d\n"
	      ,type,expression,size,pinc,G__ifile.name,G__ifile.line_number);
      G__tagnum=store_tagnum;
      G__typenum=store_typenum;
      return(G__null);
    }
    G__store_struct_offset = pointer;
#ifdef G__ASM_IFUNC
#ifdef G__ASM
    if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: NEWALLOC %d %d\n"
			     ,G__asm_cp,size,pinc);
#endif
      G__asm_inst[G__asm_cp] = G__NEWALLOC;
#ifndef G__OLDIMPLEMENTATION585
      if(memarena) G__asm_inst[G__asm_cp+1] = 0;
      else         G__asm_inst[G__asm_cp+1] = size; /* pinc is in stack */
#else
      G__asm_inst[G__asm_cp+1] = size; /* pinc is in stack, not bug */
#endif
#ifndef G__OLDIMPLEMENTATION595
      G__asm_inst[G__asm_cp+2] = (('u'==var_type)&&arrayindex)? 1 : 0;
#else
      G__asm_inst[G__asm_cp+2] = ('u'==var_type)? 1 : 0;
#endif
      G__inc_cp_asm(3,0);
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SET_NEWALLOC\n",G__asm_cp);
#endif
      G__asm_inst[G__asm_cp] = G__SET_NEWALLOC;
      G__asm_inst[G__asm_cp+1] = G__tagnum;
      G__asm_inst[G__asm_cp+2] = toupper(var_type);
      G__inc_cp_asm(3,0);
    }
#endif
#endif
  }
  
  /******************************************************
   * call constructor if struct, class
   ******************************************************/
  if(var_type=='u') {
#ifndef G__OLDIMPLEMENTATION2219
    if(G__struct.isabstract[G__tagnum]) {
      G__fprinterr(G__serr,"Error: abstract class object '%s' is created",G__struct.name[G__tagnum]);
      G__genericerror((char*)NULL);
#ifndef G__OLDIMPLEMENTATION2221
      G__display_purevirtualfunc(G__tagnum);
#endif
    }
#endif
    if(G__dispsource) {
      G__fprinterr(G__serr,"\n!!!Calling constructor 0x%lx.%s for new %s"
	      ,G__store_struct_offset,type,type);
    }
    
    if(G__CPPLINK==G__struct.iscpplink[G__tagnum]) {
      /* This is a pre-compiled class */
      long store_globalvarpointer = G__globalvarpointer;
      if(memarena) G__globalvarpointer = memarena;
#ifndef G__OLDIMPLEMENTATION1508
      else G__globalvarpointer = G__PVOID;
#endif
      if(arrayindex) {
	G__cpp_aryconstruct=pinc;
#ifndef G__OLDIMPLEMENTATION1437
#ifdef G__ASM
	if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	  if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SETARYINDEX\n" ,G__asm_cp);
#endif
	  G__asm_inst[G__asm_cp]=G__SETARYINDEX;
	  G__asm_inst[G__asm_cp+1]= 1;
	  G__inc_cp_asm(2,0);
	}
#endif
#endif
      }
      result=G__getfunction(construct,&known,G__CALLCONSTRUCTOR);
#ifndef G__OLDIMPLEMENTATION1437
#ifdef G__ASM
      if(arrayindex && G__asm_noverflow) {
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: RESETARYINDEX\n" ,G__asm_cp);
#endif
	G__asm_inst[G__asm_cp]=G__RESETARYINDEX;
	G__asm_inst[G__asm_cp+1] = 1;
	G__inc_cp_asm(2,0);
      }
#endif
#endif
      result.type=toupper(result.type);
      result.ref=0;
#ifndef G__OLDIMPLEMENTATION2068
      result.isconst = G__VARIABLE;
#endif
      G__cpp_aryconstruct=0;
      G__store_struct_offset = store_struct_offset;
      G__tagnum = store_tagnum;
      G__typenum = store_typenum;
      G__globalvarpointer = store_globalvarpointer;
#ifndef G__OLDIMPLEMENTATION705
#ifdef G__ASM
      if(memarena && G__asm_noverflow) {
	G__asm_inst[G__asm_cp] = G__SETGVP;
#ifndef G__OLDIMPLEMENTATION1659
	G__asm_inst[G__asm_cp+1] = -1;
#else
	G__asm_inst[G__asm_cp+1] = store_globalvarpointer;
#endif
	G__inc_cp_asm(2,0);
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SETGVP -1\n",G__asm_cp);
#endif
      }
#endif
#endif /* ON705 */

#ifdef G__ASM
      if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: RECMEMFUNCENV\n",G__asm_cp);
#endif
	G__asm_inst[G__asm_cp]=G__RECMEMFUNCENV;
	G__inc_cp_asm(1,0);
      }
#endif

#ifdef G__SECURITY
      if(G__security&G__SECURE_GARBAGECOLLECTION) {
	if(!G__no_exec_compile && 0==memarena) {
	  G__add_alloctable((void*)result.obj.i,result.type,result.tagnum);
#ifdef G__ASM
	  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: ADDALLOCTABLE\n",G__asm_cp);
#endif
	    G__asm_inst[G__asm_cp]=G__ADDALLOCTABLE;
	    G__inc_cp_asm(1,0);
	  }
#endif
	}
      }
#endif
      return(result);
    }
    else {
      /* This is an interpreted class */
      if(arrayindex && !G__no_exec_compile
#ifndef G__OLDIMPLEMENTATION1916
	 && 'e'!=G__struct.type[G__tagnum]
#endif
	 ) G__alloc_newarraylist(pointer,pinc);
      G__var_type='p';
      for(i=0;i<pinc;i++) {
#ifndef G__OLDIMPLEMENTATION980
	G__abortbytecode(); /* Disable bytecode */
        if(G__no_exec_compile) break;
#endif
	G__getfunction(construct,&known,G__TRYCONSTRUCTOR);
#ifndef G__OLDIMPLEMENTATION1040
	result.ref=0;
#endif
	if(known==0) {
#ifndef G__OLDIMPLEMENTATION736
	  if(initializer) {
	    G__value buf;
	    char *bp;
	    char *ep;
	    bp = strchr(construct,'(');
	    ep = strrchr(construct,')');
	    G__ASSERT(bp && ep) ;
	    *ep=0;
	    *bp=0;
	    ++bp;
#ifndef G__OLDIMPLEMENTATION1154
	    {
	      int cx,nx=0;
	      char tmpx[G__ONELINE];
	      cx = G__getstream(bp,&nx,tmpx,"),");
	      if(','==cx) {
		*ep=')';
		*(bp-1)='(';
		/* only to display error message */
		G__getfunction(construct,&known,G__CALLCONSTRUCTOR);
		break;
	      }
	    }
#endif
	    /* construct = "TYPE" , bp = "ARG" */
	    buf = G__getexpr(bp);
	    /* G__ASSERT(-1!=buf.tagnum); */
	    G__abortbytecode(); /* Disable bytecode */
	    if(-1!=buf.tagnum && 0==G__no_exec_compile) {
#ifndef G__OLDIMPLEMENTATION1614
	      if(buf.tagnum != G__tagnum) {
		G__fprinterr(G__serr
			     ,"Error: Illegal initialization of %s("
			     ,G__fulltagname(G__tagnum,1));
		G__fprinterr(G__serr,"%s)",G__fulltagname(buf.tagnum,1));
		G__genericerror((char*)NULL);
		return(G__null);
	      }
#endif
	      memcpy((void*)G__store_struct_offset,(void*)buf.obj.i
		     ,G__struct.size[buf.tagnum]);
	    }
	  }
#endif
	  break;
	}
	G__store_struct_offset += size;
	/* WARNING: FOLLOWING PART MUST BE REDESIGNED TO SUPPORT WHOLE 
	 * FUNCTION COMPILATION */
#ifdef G__ASM_IFUNC
#ifdef G__ASM
#ifndef G__OLDIMPLEMENTATION597
	G__abortbytecode(); /* Disable bytecode */
#endif /* ON597 */
	if(G__asm_noverflow) {
	  if(pinc>1) {
#ifdef G__ASM_DBG
	    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: ADDSTROS %d\n",G__asm_cp,size);
#endif
	    G__asm_inst[G__asm_cp] = G__ADDSTROS;
	    G__asm_inst[G__asm_cp+1] = size;
	    G__inc_cp_asm(2,0);
	  }
	}
#endif /* G__ASM */
#endif /* G__ASM_IFUNC */
      }
#ifdef G__OLDIMPLEMENTATION597
#ifdef G__ASM_IFUNC
#ifdef G__ASM
      if(G__asm_noverflow) {
	if(pinc>1) {
#ifdef G__ASM_DBG
	  if(G__asm_dbg) G__fprinterr(G__serr,"%3x: ADDSTROS %d\n",G__asm_cp
				 ,-size*pinc);
#endif
	  G__asm_inst[G__asm_cp] = G__ADDSTROS;
	  G__asm_inst[G__asm_cp+1] = - size*pinc;
	  G__inc_cp_asm(2,0);
	}
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SET_NEWALLOC\n",G__asm_cp);
#endif
	G__asm_inst[G__asm_cp] = G__SET_NEWALLOC;
	G__asm_inst[G__asm_cp+1] = G__tagnum;
	G__asm_inst[G__asm_cp+2] = 'U';
	G__inc_cp_asm(3,0);
      }
#endif /* G__ASM */
#endif /* G__ASM_IFUNC */
#endif /* ON597 */
#ifndef G__OLDIMPLEMENTATION1659
#ifdef G__ASM
      if(memarena && G__asm_noverflow) {
	G__asm_inst[G__asm_cp] = G__SETGVP;
	G__asm_inst[G__asm_cp+1] = -1;
	G__inc_cp_asm(2,0);
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SETGVP -1'\n",G__asm_cp);
#endif
      }
#endif
#endif /* ON1659 */
    }
  } /* end of if(var_type=='u') */
  else if(initializer) {
    /* construct = "TYPE(ARG)" */
    struct G__param para;
    int typenum;
    int hash;
    char *bp;
    char *ep;
    int store_var_type;
    bp = strchr(construct,'(');
    ep = strrchr(construct,')');
    *ep=0;
    *bp=0;
    ++bp;
    /* construct = "TYPE" , bp = "ARG" */
    typenum=G__defined_typename(construct);
    if(-1!=typenum) {
      strcpy(construct,G__type2string(G__newtype.type[typenum]
				      ,G__newtype.tagnum[typenum] ,-1
				      ,G__newtype.reftype[typenum] ,0));
    }
    hash = strlen(construct);
    store_var_type = G__var_type;
    G__var_type='p';
    para.para[0]=G__getexpr(bp); /* generates LD or LD_VAR etc... */
    G__var_type = store_var_type;
    if(!G__no_exec_compile) result.ref = pointer; 
    else                    result.ref = 0;
    /* following call generates CAST instruction */
#ifndef G__OLDIMPLEMENTATION924
    if(var_type=='U' && pointer) {
#ifndef G__OLDIMPLEMENTATION940
      if(0==G__no_exec_compile) *(long*)pointer = para.para[0].obj.i;
#else
      *(long*)pointer = para.para[0].obj.i;
#endif
    }
    else
      G__explicit_fundamental_typeconv(construct,hash,&para,&result);
#else
    G__explicit_fundamental_typeconv(construct,hash,&para,&result);
#endif
#ifdef G__ASM
    if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: LETNEWVAL\n",G__asm_cp);
#endif
      G__asm_inst[G__asm_cp]=G__LETNEWVAL;
      G__inc_cp_asm(1,0);
    }
#endif /* ASM */
#ifndef G__OLDIMPLEMENTATION1659
#ifdef G__ASM
    if(memarena && G__asm_noverflow) {
      G__asm_inst[G__asm_cp] = G__SETGVP;
      G__asm_inst[G__asm_cp+1] = -1;
      G__inc_cp_asm(2,0);
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SETGVP -1''\n",G__asm_cp);
#endif
    }
#endif
#endif /* ON1659 */
  }
  
#ifndef G__OLDIMPLEMENTATION683
  if(isupper(var_type)) {
    G__letint(&result,var_type,pointer);
    switch(reftype) {
    case G__PARANORMAL: result.obj.reftype.reftype = G__PARAP2P; break;
    case G__PARAP2P:    result.obj.reftype.reftype = G__PARAP2P2P; break;
#ifndef G__OLDIMPLEMENTATION707
    default:            result.obj.reftype.reftype = reftype+1; break;
#else
    default:            result.obj.reftype.reftype = G__PARAP2P2P; break;
#endif
    }
  }
  else {
    G__letint(&result,toupper(var_type),pointer);
    result.obj.reftype.reftype = reftype;
  }
#else
  G__letint(&result,toupper(var_type),pointer);
  result.obj.reftype.reftype = reftype;
#endif
  result.tagnum=G__tagnum;
  result.typenum=G__typenum;
  
  G__store_struct_offset = store_struct_offset;
  G__tagnum = store_tagnum;
  G__typenum = store_typenum;
#ifdef G__ASM
  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: RECMEMFUNCENV\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp]=G__RECMEMFUNCENV;
    G__inc_cp_asm(1,0);
  }
#endif

#ifdef G__SECURITY
  if(G__security&G__SECURE_GARBAGECOLLECTION) {
#ifndef G__OLDIMPLEMENTATION586
    if(!G__no_exec_compile && 0==memarena)
#else
    if(!G__no_exec_compile)
#endif
      G__add_alloctable((void*)result.obj.i,result.type,result.tagnum);
#ifdef G__ASM
    if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: ADDALLOCTABLE\n",G__asm_cp);
#endif
      G__asm_inst[G__asm_cp]=G__ADDALLOCTABLE;
      G__inc_cp_asm(1,0);
    }
#endif
  }
#endif
  
  return(result);
}

/****************************************************************
* G__getarrayindex()
* 
* Called by:
*   G__new_operator()
*
*  [x][y][z]     get x*y*z
*
****************************************************************/
int G__getarrayindex(indexlist)
char *indexlist;
{
  int p_inc=1;
  int p=1;
  char index[G__ONELINE];
  int c;
#ifndef G__OLDIMPLEMENTATION1147
  int store_var_type=G__var_type;
  G__var_type='p';
#endif
  
  c = G__getstream(indexlist,&p,index,"]");
  p_inc *= G__int(G__getexpr(index));
  while(*(indexlist+p)=='[') {
    ++p;
    c = G__getstream(indexlist,&p,index,"]");
    p_inc *= G__int(G__getexpr(index));
#ifdef G__ASM_IFUNC
#ifdef G__ASM
    if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: OP2 *\n" ,G__asm_cp);
#endif
      G__asm_inst[G__asm_cp]=G__OP2;
      G__asm_inst[G__asm_cp+1]=(long)'*';
      G__inc_cp_asm(2,0);
    }
#endif /* G__ASM */
#endif /* G__ASM_IFUNC */
  }
  G__ASSERT(']'==c);

#ifndef G__OLDIMPLEMENTATION1147
  G__var_type=store_var_type;
#endif
  
  return(p_inc);
}

/****************************************************************
* void G__delete_operator()
* 
* Called by:
*   G__exec_statement()
*   G__exec_statement()
*
*        V
* delete 
*
****************************************************************/
void G__delete_operator(expression,isarray)
char *expression;
int isarray;
{
  long store_struct_offset; /* used to be int */
  int store_tagnum,store_typenum;
  int done;
  char destruct[G__ONELINE];
  G__value buf;
  int pinc,i,size;
  int cpplink=0;
#ifndef G__OLDIMPLEMENTATION864
  int zeroflag=0;
#endif

#ifndef G__OLDIMPLEMENTATION2102
  if(G__cintv6) {
    /* THIS CASE IS NEVER USED */
    G__bc_delete_operator(expression,isarray);
    return;
  }
#endif
  

  buf=G__getitem(expression);
  if(islower(buf.type)) {
    G__fprinterr(G__serr,"Error: %s cannot delete",expression);
    G__genericerror((char*)NULL);
    return;
  }
#ifndef G__OLDIMPLEMENTATION864
  else if(0==buf.obj.i && 0==G__no_exec_compile && 
	  G__ASM_FUNC_NOP==G__asm_wholefunction) {
    zeroflag=1;
    G__no_exec_compile = 1;
    buf.obj.d = 0;
    buf.obj.i = 1;
  }
#else
  else if(0==buf.obj.i && 0==G__no_exec_compile && 
	  G__ASM_FUNC_NOP==G__asm_wholefunction) {
    G__fprinterr(G__serr,"Error: %s==NULL cannot delete",expression);
    G__genericerror((char*)NULL);
    return;
  }
#endif

  G__CHECK(G__SECURE_MALLOC,1,return);

#ifdef G__SECURITY
  if(G__security&G__SECURE_GARBAGECOLLECTION) {
    if(!G__no_exec_compile) G__del_alloctable((void*)buf.obj.i);
#ifdef G__ASM
    if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: DELALLOCTABLE\n",G__asm_cp);
#endif
      G__asm_inst[G__asm_cp]=G__DELALLOCTABLE;
      G__inc_cp_asm(1,0);
    }
#endif
  }
#endif
  
  /*********************************************************
   * Call destructor if struct of class
   *********************************************************/
  if(buf.type=='U' && G__PARANORMAL==buf.obj.reftype.reftype) {
    store_struct_offset = G__store_struct_offset;
    store_typenum = G__typenum;
    store_tagnum = G__tagnum;
    
    G__store_struct_offset = buf.obj.i; 
    G__typenum = buf.typenum;
    G__tagnum = buf.tagnum;
    
    sprintf(destruct,"~%s()",G__struct.name[G__tagnum]);
    if(G__dispsource) {
      G__fprinterr(G__serr,"\n!!!Calling destructor 0x%lx.%s for %s"
	      ,G__store_struct_offset ,destruct ,expression);
    }
    done=0;

#ifndef G__OLDIMPLEMENTATION659
    if(0==G__no_exec_compile && -1!=G__struct.virtual_offset[G__tagnum]&&
       G__tagnum!= 
       *(long*)(G__store_struct_offset+G__struct.virtual_offset[G__tagnum])) {
      int virtualtag=
       *(long*)(G__store_struct_offset+G__struct.virtual_offset[G__tagnum]);
      buf.obj.i -= G__find_virtualoffset(virtualtag);
    } 
#endif
    
    /*****************************************************
     * Push and set G__store_struct_offset
     *****************************************************/
#ifdef G__ASM_IFUNC
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
#ifndef G__OLDIMPLEMENTATION1437
      if(isarray) {
	G__asm_inst[G__asm_cp] = G__GETARYINDEX;
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: GETARYINDEX\n",G__asm_cp-2);
#endif
	G__inc_cp_asm(1,0);
#endif
      }
    }
#endif /* G__ASM */
#endif /* G__ASM_IFUNC */

    /*****************************************************
     * Call destructor
     *****************************************************/
    if(G__CPPLINK==G__struct.iscpplink[G__tagnum]) {
      /* pre-compiled class */
      if(isarray) G__cpp_aryconstruct=1;
      
#ifndef G__ASM_IFUNC
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
#endif /* G__ASM */
#endif /* !G__ASM_IFUNC */
      
      G__getfunction(destruct,&done,G__TRYDESTRUCTOR);
      /* Precompiled destructor must always exist here */
      
#ifndef G__ASM_IFUNC
#ifdef G__ASM
      if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp);
#endif
	G__asm_inst[G__asm_cp] = G__POPSTROS;
	G__inc_cp_asm(1,0);
      }
#endif /* G__ASM */
#endif /* !G__ASM_IFUNC */
      
      G__cpp_aryconstruct=0;
      cpplink=1;
    }
    else {
      /* interpreted class */
      /* WARNING: FOLLOWING PART MUST BE REDESIGNED TO SUPPORT WHOLE
       * FUNCTION COMPILATION */
      if(isarray) {
	if(!G__no_exec_compile)
	  pinc=G__free_newarraylist(G__store_struct_offset);
#ifndef G__OLDIMPLEMENTATION597
	else pinc = 1;
#endif
	size = G__struct.size[G__tagnum];
	for(i=pinc-1;i>=0;--i) {
	  G__store_struct_offset = buf.obj.i+size*i;
	  G__getfunction(destruct,&done ,G__TRYDESTRUCTOR);
#ifdef G__ASM_IFUNC
#ifdef G__ASM
#ifndef G__OLDIMPLEMENTATION597
	  if(0==done) break;
	  G__abortbytecode(); /* Disable bytecode */
#endif /* ON597 */
	  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: ADDSTROS %d\n",G__asm_cp,size);
#endif
	    G__asm_inst[G__asm_cp] = G__ADDSTROS;
	    G__asm_inst[G__asm_cp+1] = (long)size;
	    G__inc_cp_asm(2,0);
	  }
#endif /* G__ASM */
#endif /* !G__ASM_IFUNC */
	}
      }
      else {
	G__getfunction(destruct,&done,G__TRYDESTRUCTOR);
      }
    }

#ifdef G__ASM 
#ifdef G__SECURITY
    if(G__security&G__SECURE_GARBAGECOLLECTION &&G__asm_noverflow&&0==done) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: BASEDESTRUCT\n",G__asm_cp);
#endif
      G__asm_inst[G__asm_cp] = G__BASEDESTRUCT;
      G__asm_inst[G__asm_cp+1] = G__tagnum;
      G__asm_inst[G__asm_cp+2] = isarray;
      G__inc_cp_asm(3,0);
    }
#endif
#endif
    
    /*****************************************************
     * Push and set G__store_struct_offset
     *****************************************************/
#ifdef G__ASM_IFUNC
#ifdef G__ASM
    if(G__asm_noverflow) {
#ifndef G__OLDIMPLEMENTATION1437
      if(isarray) {
	G__asm_inst[G__asm_cp] = G__RESETARYINDEX;
	G__asm_inst[G__asm_cp+1] = 0;
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: RESETARYINDEX\n",G__asm_cp-2);
#endif
	G__inc_cp_asm(2,0);
      }
#endif
      if(G__CPPLINK!=G__struct.iscpplink[G__tagnum]) {
	/* if interpreted class, free memory */
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: DELETEFREE\n",G__asm_cp);
#endif
	G__asm_inst[G__asm_cp] = G__DELETEFREE;
	G__asm_inst[G__asm_cp+1] = isarray? 1 : 0;
	G__inc_cp_asm(2,0);
      }
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp+1);
#endif
      G__asm_inst[G__asm_cp] = G__POPSTROS;
      G__inc_cp_asm(1,0);
    }
#endif /* G__ASM */
#endif /* G__ASM_IFUNC */

    /*****************************************************
     * Push and set G__store_struct_offset
     *****************************************************/
    G__store_struct_offset = store_struct_offset;
    G__typenum = store_typenum;
    G__tagnum = store_tagnum;

  } /* end of if 'U' */

  else if(G__asm_noverflow) {
    G__asm_inst[G__asm_cp] = G__PUSHSTROS;
    G__asm_inst[G__asm_cp+1] = G__SETSTROS;
    G__inc_cp_asm(2,0);
#ifdef G__ASM_DBG
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp-2);
      G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp-1);
    }
#endif
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: DELETEFREE\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp] = G__DELETEFREE;
    G__asm_inst[G__asm_cp+1] = 0;
    G__inc_cp_asm(2,0);
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp] = G__POPSTROS;
    G__inc_cp_asm(1,0);
  }
  
  /*****************************************************
   * free memory if interpreted object
   *****************************************************/
  if(G__NOLINK==cpplink && !G__no_exec_compile) {
#ifndef G__OLDIMPLEMENTATION1002
#ifdef G__ROOT
    G__delete_interpreted_object((void*)buf.obj.i);
#else
    free((void *)buf.obj.i);
#endif
#else
    free((void *)buf.obj.i);
#endif
  }

/* #ifdef G__ROOT */
  /*****************************************************
   * assign NULL for deleted pointer variable
   *****************************************************/
  if(buf.ref && 0==G__no_exec && 0==G__no_exec_compile) *(long*)buf.ref = 0;
/* #endif G__ROOT */

#ifndef G__OLDIMPLEMENTATION864
  if(zeroflag) {
    G__no_exec_compile=0;
    buf.obj.i=0;
  }
#endif
}

/****************************************************************
* G__alloc_newarraylist()
****************************************************************/
int G__alloc_newarraylist(point,pinc)
long point;
int pinc;
{
  struct G__newarylist *newary;

#ifdef G__MEMTEST
  fprintf(G__memhist,"G__alloc_newarraylist(%lx,%d)\n",point,pinc);
#endif
	
  /****************************************************
   * Find out end of list
   ****************************************************/
  newary = &G__newarray;
  while(newary->next) newary = newary->next;
  
  
  /****************************************************
   * create next list
   ****************************************************/
  newary->next=(struct G__newarylist *)malloc(sizeof(struct G__newarylist));
  /****************************************************
   * store information
   ****************************************************/
  newary=newary->next;
  newary->point = point;
  newary->pinc = pinc;
  newary->next = (struct G__newarylist *)NULL;
  return(0);
}

/****************************************************************
* G__free_newarraylist()
****************************************************************/
int G__free_newarraylist(point)
long point;
{
  struct G__newarylist *newary,*prev;
  int pinc,flag=0;

#ifdef G__MEMTEST
  fprintf(G__memhist,"G__free_newarraylist(%lx)\n",point);
#endif
  
  /****************************************************
   * Search point
   ****************************************************/
  prev = &G__newarray;
  newary = G__newarray.next;
  while(newary) {
    if(newary->point == point) {
      flag=1;
      break;
    }
    prev = newary;
    newary = newary->next;
  }
  
  if(flag==0) {
    G__fprinterr(G__serr,"Error: delete[] on wrong object 0x%lx FILE:%s LINE:%d\n"
	    ,point,G__ifile.name,G__ifile.line_number);
    return(0);
  }
  
  /******************************************************
   * get malloc size information
   ******************************************************/
  pinc = newary->pinc;
  
  /******************************************************
   * delete newarraylist
   ******************************************************/
  prev->next = newary->next;
  free((void*)newary);
  
  /* return result */
  return(pinc);
}



/**************************************************************************
* G__handle_delete
*
* Parsing of 'delete obj' 'delete obj[]'
**************************************************************************/
int G__handle_delete(piout ,statement)
int *piout;
char *statement;
{
  int c;
  c=G__fgetstream(statement ,"[;");
  *piout=0;
  if('['==c) {
    if('\0'==statement[0]) {
      c=G__fgetstream(statement ,"]");
      c=G__fgetstream(statement ,";");
      *piout=1;
    }
    else {
      strcpy(statement+strlen(statement),"[");
      c=G__fgetstream(statement+strlen(statement),"]");
      strcpy(statement+strlen(statement),"]");
      c=G__fgetstream(statement+strlen(statement),";");
    }
  }
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
