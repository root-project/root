/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file stub.c
 ************************************************************************
 * Description:
 *  New style stub function interface
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto 
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



/**************************************************************************
* G__stubstoreenv()
*
*  Called from interface method source
*
**************************************************************************/
void G__stubstoreenv(env,p,tagnum)
struct G__StoreEnv *env;
void *p;
int tagnum;
{
  env->store_struct_offset = G__store_struct_offset;
  env->store_tagnum = G__tagnum;
  env->store_memberfunc_tagnum = G__memberfunc_tagnum;
  env->store_exec_memberfunc = G__exec_memberfunc;
  if(p) {
    G__store_struct_offset = (long)p;
#ifndef G__OLDIMPLEMENTATION1995
    G__tagnum = tagnum; 
#else
    /* G__tagnum = G__tagnum; */ /* I think this is intentional */
#endif
    G__memberfunc_tagnum = tagnum;
    G__exec_memberfunc = 1;
  }
  else {
    G__store_struct_offset = 0;
    G__tagnum = -1;
    G__memberfunc_tagnum = -1;
    G__exec_memberfunc = 0;
  }
#ifndef G__OLDIMPLEMENTATION1986
  G__stubcall=1;
#endif
}

/**************************************************************************
* G__stubrestoreenv()
*
*  Called from interface method source
*
**************************************************************************/
void G__stubrestoreenv(env)
struct G__StoreEnv *env;
{
  G__store_struct_offset = env->store_struct_offset;
  G__tagnum = env->store_tagnum ;
  G__memberfunc_tagnum = env->store_memberfunc_tagnum;
  G__exec_memberfunc = env->store_exec_memberfunc;
}

/**************************************************************************
* G__setdouble()
*
**************************************************************************/
void G__setdouble(pbuf,d,pd,type,tagnum,typenum,reftype)
G__value *pbuf;
double d;
void *pd;
int type;
int tagnum;
int typenum;
int reftype;
{
  pbuf->type = type;
  pbuf->tagnum= tagnum;
  pbuf->typenum= typenum;
  if(reftype) pbuf->ref = (long)pd;
  else        pbuf->ref = 0;
  pbuf->obj.d = d;
}

/**************************************************************************
* G__setint()
*
**************************************************************************/
void G__setint(pbuf,l,pl,type,tagnum,typenum,reftype)
G__value *pbuf;
long l;
void *pl;
int type;
int tagnum;
int typenum;
int reftype;
{
  pbuf->type = type;
  pbuf->tagnum= tagnum;
  pbuf->typenum= typenum;
  if(reftype) pbuf->ref = (long)pl;
  else        pbuf->ref = 0;
  pbuf->obj.i = l;
}


/**************************************************************************
* G__cppstub_setparam()
*
**************************************************************************/
static void G__cppstub_setparam(pformat,pbody,tagnum,ifn,ifunc,k)
char* pformat;
char* pbody;
int tagnum,ifn;
struct G__ifunc_table *ifunc;
int k;
{
  char paraname[G__MAXNAME];
  char temp[G__ONELINE];

#ifndef G__OLDIMPLEMENTATION1911
  if(0 && tagnum) return;
#endif

  if(ifunc->para_name[ifn][k]) strcpy(paraname,ifunc->para_name[ifn][k]);
  else sprintf(paraname,"a%d",k);

  if(k) strcat(pformat,",");
  strcat(pbody,",");

  if(ifunc->para_reftype[ifn][k]) {
#ifndef G__OLDIMPLEMENTATION401
    sprintf(temp,"*(%s*)(%%ld)"
	    ,G__type2string(ifunc->para_type[ifn][k]
			    ,ifunc->para_p_tagtable[ifn][k]
			    ,ifunc->para_p_typetable[ifn][k] ,0
			    ,ifunc->para_isconst[ifn][k]));
#else
    sprintf(temp,"*(%s*)%%ld"
	    ,G__type2string(ifunc->para_type[ifn][k]
			    ,ifunc->para_p_tagtable[ifn][k]
			    ,ifunc->para_p_typetable[ifn][k] ,0));
#endif
    strcat(pformat,temp);
    sprintf(temp,"(long)(&%s)",paraname);
    strcat(pbody,temp);
  }
  else {
    switch(ifunc->para_type[ifn][k]) {
    case 'u':
#ifndef G__OLDIMPLEMENTATION401
      sprintf(temp,"(%s)(%%ld)"
	      ,G__type2string(ifunc->para_type[ifn][k]
			      ,ifunc->para_p_tagtable[ifn][k]
			      ,ifunc->para_p_typetable[ifn][k] ,0
			      ,ifunc->para_isconst[ifn][k]));
#else
      sprintf(temp,"(%s)%%ld"
	      ,G__type2string(ifunc->para_type[ifn][k]
			      ,ifunc->para_p_tagtable[ifn][k]
			      ,ifunc->para_p_typetable[ifn][k] ,0));
#endif
      strcat(pformat,temp);
      sprintf(temp,"&%s",paraname);
      strcat(pbody,temp);
      break;
    case 'd':
    case 'f':
#ifndef G__OLDIMPLEMENTATION401
      sprintf(temp,"(%s)%%g"
	      ,G__type2string(ifunc->para_type[ifn][k]
			      ,ifunc->para_p_tagtable[ifn][k]
			      ,ifunc->para_p_typetable[ifn][k] ,0
			      ,ifunc->para_isconst[ifn][k]));
#else
      sprintf(temp,"(%s)%%g"
	      ,G__type2string(ifunc->para_type[ifn][k]
			      ,ifunc->para_p_tagtable[ifn][k]
			      ,ifunc->para_p_typetable[ifn][k] ,0));
#endif
      strcat(pformat,temp);
      sprintf(temp,"%s",paraname);
      strcat(pbody,temp);
      break;
    default:
#ifndef G__OLDIMPLEMENTATION401
      sprintf(temp,"(%s)(%%ld)"
	      ,G__type2string(ifunc->para_type[ifn][k]
			      ,ifunc->para_p_tagtable[ifn][k]
			      ,ifunc->para_p_typetable[ifn][k] ,0
			      ,ifunc->para_isconst[ifn][k]));
#else
      sprintf(temp,"(%s)%%ld"
	      ,G__type2string(ifunc->para_type[ifn][k]
			      ,ifunc->para_p_tagtable[ifn][k]
			      ,ifunc->para_p_typetable[ifn][k] ,0));
#endif
      strcat(pformat,temp);
      sprintf(temp,"(long)%s",paraname);
      strcat(pbody,temp);
      break;
    }
  }
}

/**************************************************************************
* G__cppstub_constructor()
*
**************************************************************************/
static void G__cppstub_genconstructor(fp,tagnum,ifn,ifunc)
FILE *fp;
int tagnum,ifn;
struct G__ifunc_table *ifunc;
{
#ifndef G__OLDIMPLEMENTATION1911
  if(0 && fp && ifn && ifunc) return;
#endif
  G__fprinterr(G__serr,"Limitation: Can not make STUB constructor, class %s\n"
	  ,G__fulltagname(tagnum,1));
}

/**************************************************************************
* G__cppstub_destructor()
*
**************************************************************************/
static void G__cppstub_gendestructor(fp,tagnum,ifn,ifunc)
FILE *fp;
int tagnum,ifn;
struct G__ifunc_table *ifunc;
{
#ifndef G__OLDIMPLEMENTATION1911
  if(0 && fp && ifn && ifunc) return;
#endif
  G__fprinterr(G__serr,"Limitation: Can not make STUB destructor, class %s\n"
	  ,G__fulltagname(tagnum,1));
}

/**************************************************************************
* G__cppstub_genfunc()
*
**************************************************************************/
#ifndef G__OLDIMPLEMENTATION1336
void G__cppstub_genfunc(fp,tagnum,ifn,ifunc,flag)
FILE *fp;
int tagnum,ifn;
struct G__ifunc_table *ifunc;
int flag;
#else
static void G__cppstub_genfunc(fp,tagnum,ifn,ifunc)
FILE *fp;
int tagnum,ifn;
struct G__ifunc_table *ifunc;
#endif
{
  int k;
  char pformat[G__ONELINE];
  char pbody[G__LONGLINE];

  /*******************************************************************
  * Function header
  *******************************************************************/
  if(-1==tagnum
#ifndef G__OLDIMPLEMENTATION1336
     || flag
#endif
     ) {
    fprintf(fp,"%s %s(\n"
	    ,G__type2string(ifunc->type[ifn],ifunc->p_tagtable[ifn]
			    ,ifunc->p_typetable[ifn],ifunc->reftype[ifn]
			    ,ifunc->isconst[ifn])
	    ,ifunc->funcname[ifn]);
  }
  else {
#ifndef G__OLDIMPLEMENTATION1892
    fprintf(fp,"%s "
	    ,G__type2string(ifunc->type[ifn],ifunc->p_tagtable[ifn]
			    ,ifunc->p_typetable[ifn],ifunc->reftype[ifn]
			    ,ifunc->isconst[ifn])
	    );
    fprintf(fp,"%s::%s(\n",G__fulltagname(tagnum,1),ifunc->funcname[ifn]);
#else
    fprintf(fp,"%s %s::%s(\n"
	    ,G__type2string(ifunc->type[ifn],ifunc->p_tagtable[ifn]
			    ,ifunc->p_typetable[ifn],ifunc->reftype[ifn]
			    ,ifunc->isconst[ifn])
	    ,G__fulltagname(tagnum,1)
	    ,ifunc->funcname[ifn]);
#endif
  }

  if(G__clock) {
    for(k=0;k<ifunc->para_nu[ifn];k++) {
      if(k) fprintf(fp,",");
      if(ifunc->para_name[ifn][k]) {
	fprintf(fp,"%s",ifunc->para_name[ifn][k]);
      }
      else {
	fprintf(fp,"a%d",k);
      }
    }
    fprintf(fp,")\n");
    for(k=0;k<ifunc->para_nu[ifn];k++) {
      fprintf(fp,"%s" ,G__type2string(ifunc->para_type[ifn][k]
				      ,ifunc->para_p_tagtable[ifn][k]
				      ,ifunc->para_p_typetable[ifn][k]
				      ,ifunc->para_reftype[ifn][k]
				      ,ifunc->para_isconst[ifn][k]));
      if(ifunc->para_name[ifn][k]) {
	fprintf(fp," %s;\n",ifunc->para_name[ifn][k]);
      }
      else {
	fprintf(fp," a%d;\n",k);
      }
    }
    fprintf(fp,"{\n");
  }
  else {
    for(k=0;k<ifunc->para_nu[ifn];k++) {
      if(k) fprintf(fp,",\n");
      fprintf(fp,"%s" ,G__type2string(ifunc->para_type[ifn][k]
				      ,ifunc->para_p_tagtable[ifn][k]
				      ,ifunc->para_p_typetable[ifn][k]
				      ,ifunc->para_reftype[ifn][k]
				      ,ifunc->para_isconst[ifn][k]));
      if(ifunc->para_name[ifn][k]) {
	fprintf(fp," %s",ifunc->para_name[ifn][k]);
      }
      else {
	fprintf(fp," a%d",k);
      }
    }
#ifndef G__OLDIMPLEMENTATION1809
    if(ifunc->isconst[ifn]&G__CONSTFUNC) fprintf(fp,") const {\n");
    else fprintf(fp,") {\n");
#else
    fprintf(fp,") {\n");
#endif
  }

  /*******************************************************************
  * local variable declaration and initialization 
  *******************************************************************/
  fprintf(fp,"  G__value buf;\n");
  fprintf(fp,"  struct G__StoreEnv storeenv;\n");
  fprintf(fp,"  char funccall[G__LONGLINE];\n");

  if(-1!=tagnum) {
    fprintf(fp
   ,"  G__stubstoreenv(&storeenv,(void*)this,G__get_linked_tagnum(&%s));\n"
            ,G__get_link_tagname(tagnum));
  }
  else {
    fprintf(fp,"  G__stubstoreenv(&storeenv,(void*)NULL,-1);\n");
  }

  /*******************************************************************
  * stub function call
  *******************************************************************/
  pformat[0] = '\0';
  pbody[0] = '\0';
  for(k=0;k<ifunc->para_nu[ifn];k++) {
    G__cppstub_setparam(pformat,pbody,tagnum,ifn,ifunc,k);
  }
  fprintf(fp,"  sprintf(funccall,\"%s(%s)\"%s);\n"
	  ,ifunc->funcname[ifn],pformat,pbody);
  fprintf(fp,"  buf=G__calc(funccall);\n");

  /*******************************************************************
  * clean up
  *******************************************************************/
  fprintf(fp,"  G__stubrestoreenv(&storeenv);\n");

  /*******************************************************************
  * return value
  *******************************************************************/
  if(ifunc->reftype[ifn]) {
    fprintf(fp,"  return(*(%s*)buf.obj.i);\n"
	    ,G__type2string(ifunc->type[ifn] ,ifunc->p_tagtable[ifn]
			    ,ifunc->p_typetable[ifn] ,0,0));
  }
  else {
    switch(ifunc->type[ifn]) {
    case 'u':
      fprintf(fp,"  return(*(%s*)buf.obj.i);\n"
	      ,G__type2string(ifunc->type[ifn] ,ifunc->p_tagtable[ifn]
			      ,ifunc->p_typetable[ifn] ,0,0));
      break;
    case 'd':
    case 'f':
      fprintf(fp,"  return((%s)buf.obj.d);\n"
	      ,G__type2string(ifunc->type[ifn] ,ifunc->p_tagtable[ifn]
			      ,ifunc->p_typetable[ifn] ,0,0));
      break;
    case 'y':
      break;
    default:
      fprintf(fp,"  return((%s)buf.obj.i);\n"
	      ,G__type2string(ifunc->type[ifn] ,ifunc->p_tagtable[ifn]
			      ,ifunc->p_typetable[ifn] ,0,0));
      break;
    }
  }
  
  fprintf(fp,"}\n\n");
}


/**************************************************************************
* G__cppstub_memfunc() 
*
*  Generate stub member function. Not used
* 
**************************************************************************/
void G__cppstub_memfunc(fp)
FILE *fp;
{
  int i,j;
  struct G__ifunc_table *ifunc;
  int isconstructor,iscopyconstructor,isdestructor,isassignmentoperator;

  fprintf(fp,"\n/*********************************************************\n");
  fprintf(fp,"* Member function Stub\n");
  fprintf(fp,"*********************************************************/\n");

  for(i=0;i<G__struct.alltag;i++) {
    if((G__CPPLINK==G__struct.globalcomp[i]||
	G__CLINK==G__struct.globalcomp[i])&&
       /* -1==(int)G__struct.parent_tagnum[i]&& */
       -1!=G__struct.line_number[i]&&G__struct.hash[i]&&
       '$'!=G__struct.name[i][0] && 'e'!=G__struct.type[i]) {
      ifunc = G__struct.memfunc[i];
      isconstructor=0;
      iscopyconstructor=0;
      isdestructor=0;
      isassignmentoperator=0;

      /* member function interface */
      fprintf(fp,"\n/* %s */\n",G__fulltagname(i,0));

      while(ifunc) {
	for(j=0;j<ifunc->allifunc;j++) {
	  
	  if(
#ifndef G__OLDIMPLEMENTATION2039
	     ifunc->hash[j]!=0 &&
#endif
	     -1==ifunc->pentry[j]->line_number
	     &&0==ifunc->ispurevirtual[j] && ifunc->hash[j] &&
	     (G__CPPSTUB==ifunc->globalcomp[j]||
	      G__CSTUB==ifunc->globalcomp[j])) {

	    if(strcmp(ifunc->funcname[j],G__struct.name[i])==0) {
	      /* constructor need special handling */
	      G__cppstub_genconstructor(fp,i,j,ifunc);
	    }
	    else if('~'==ifunc->funcname[j][0]) {
	      G__cppstub_gendestructor(fp,i,j,ifunc);
	    }
	    else {
#ifndef G__OLDIMPLEMENTATION1336
	      G__cppstub_genfunc(fp,i,j,ifunc,0);
#else
	      G__cppstub_genfunc(fp,i,j,ifunc);
#endif
	    }
	  } /* if(access) */
	} /* for(j) */
	ifunc=ifunc->next;
      } /* while(ifunc) */
    } /* if(globalcomp) */
  } /* for(i) */
}

/**************************************************************************
* G__cppstub_func() 
*
*  Generate stub global function
* 
**************************************************************************/
void G__cppstub_func(fp)
FILE *fp;
{
  int j;
  struct G__ifunc_table *ifunc;

  fprintf(fp,"\n/*********************************************************\n");
  fprintf(fp,"* Global function Stub\n");
  fprintf(fp,"*********************************************************/\n");
  ifunc = &G__ifunc;

  /* member function interface */
  while(ifunc) {
    for(j=0;j<ifunc->allifunc;j++) {
      if((G__CPPSTUB==ifunc->globalcomp[j]||G__CSTUB==ifunc->globalcomp[j])&& 
	 ifunc->hash[j]) {
	
#ifndef G__OLDIMPLEMENTATION1336
	G__cppstub_genfunc(fp,-1,j,ifunc,0);
#else
	G__cppstub_genfunc(fp,-1,j,ifunc);
#endif

      } /* if(access) */
    } /* for(j) */
    ifunc=ifunc->next;
  } /* while(ifunc) */
}


/**************************************************************************
* G__set_stubflags()
*
*  Set stub flags beginng at cirtain dictionary position
**************************************************************************/
void G__set_stubflags(dictpos)
struct G__dictposition *dictpos;
{
  int ig15;
  int tagnum;
  int ifn;

  /* global variable */
  while(dictpos->var) {
    for(ig15=dictpos->ig15;ig15<dictpos->var->allvar;ig15++) {
      if('p'!=dictpos->var->type[ig15]) {
	if(G__dispmsg>=G__DISPWARN) {
	  G__fprinterr(G__serr,
	       "Warning: global variable %s specified in stub file. Ignored\n"
		       ,dictpos->var->varnamebuf[ig15]);
	}

      }
    }
    dictpos->var=dictpos->var->next;
  }

  for(tagnum=dictpos->tagnum;tagnum<G__struct.alltag;tagnum++) {
#ifndef G__OLDIMPLEMENTATION706
    struct G__ifunc_table *ifunc;
    ifunc=G__struct.memfunc[tagnum];
    while(ifunc) {
      for(ifn=0;ifn<ifunc->allifunc;ifn++) {
	if(-1==ifunc->pentry[ifn]->line_number
	   &&0==ifunc->ispurevirtual[ifn] && ifunc->hash[ifn]) {
	  switch(G__globalcomp) {
	  case G__CPPLINK: ifunc->globalcomp[ifn]=G__CPPSTUB; break;
	  case G__CLINK:   ifunc->globalcomp[ifn]=G__CSTUB; break;
	  default: break;
	  }
	}
      }
      ifunc=ifunc->next;
    }
#else
    if(G__dispmsg>=G__DISPWARN) {
      G__fprinterr(G__serr,"Warning: class/struct/union %s specified in stub file\n"
		   ,G__struct.name[tagnum]);
    }
#endif
  }

#ifndef G__OLDIMPLEMENTATION2220
  if(dictpos->ifunc) {
    struct G__ifunc_table *ifunc = dictpos->ifunc;
    while(ifunc) {
      if(ifunc==dictpos->ifunc) ifn = dictpos->ifn;
      else ifn = 0;
      while(ifn<ifunc->allifunc) {
	switch(ifunc->globalcomp[ifn]) {
	case G__CPPLINK: ifunc->globalcomp[ifn]=G__CPPSTUB; break;
	case G__CLINK:   ifunc->globalcomp[ifn]=G__CSTUB; break;
	default: break;
	}
	++ifn;
      }
      ifunc = ifunc->next;
    }
  }
#else
  while(dictpos->ifunc) {
    for(ifn=dictpos->ifn;ifn<dictpos->ifunc->allifunc;ifn++) {
      switch(dictpos->ifunc->globalcomp[ifn]) {
      case G__CPPLINK: dictpos->ifunc->globalcomp[ifn]=G__CPPSTUB; break;
      case G__CLINK:   dictpos->ifunc->globalcomp[ifn]=G__CSTUB; break;
      default: break;
      }
    }
    dictpos->ifunc=dictpos->ifunc->next;
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
