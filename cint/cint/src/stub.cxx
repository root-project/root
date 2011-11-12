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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "Api.h"

extern "C" {

/**************************************************************************
* G__stubstoreenv()
*
*  Called from interface method source
*
**************************************************************************/
void G__stubstoreenv(G__StoreEnv *env,void *p,int tagnum)
{
  env->store_struct_offset = G__store_struct_offset;
  env->store_tagnum = G__tagnum;
  env->store_memberfunc_tagnum = G__memberfunc_tagnum;
  env->store_exec_memberfunc = G__exec_memberfunc;
  if(p) {
    G__store_struct_offset = (long)p;
    G__tagnum = tagnum; 
    G__memberfunc_tagnum = tagnum;
    G__exec_memberfunc = 1;
  }
  else {
    G__store_struct_offset = 0;
    G__tagnum = -1;
    G__memberfunc_tagnum = -1;
    G__exec_memberfunc = 0;
  }
}

/**************************************************************************
* G__stubrestoreenv()
*
*  Called from interface method source
*
**************************************************************************/
void G__stubrestoreenv(G__StoreEnv *env)
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
void G__setdouble(G__value *pbuf
                  ,double d,void *pd
                  ,int type,int tagnum,int typenum,int reftype)
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
void G__setint(G__value *pbuf,long l,void *pl
               ,int type,int tagnum,int typenum,int reftype)
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
static void G__cppstub_setparam(G__FastAllocString& pformat,G__FastAllocString& pbody
                                ,int /* tagnum */,int ifn,G__ifunc_table_internal *ifunc,int k)
{
  G__FastAllocString paraname(G__MAXNAME);
  G__FastAllocString temp(G__ONELINE);

  if(ifunc->param[ifn][k]->name) paraname = ifunc->param[ifn][k]->name;
  else paraname.Format("a%d",k);

  if(k) pformat += ",";
  pbody += ",";

  if(ifunc->param[ifn][k]->reftype) {
     temp.Format("*(%s*)(%%ld)"
            ,G__type2string(ifunc->param[ifn][k]->type
                            ,ifunc->param[ifn][k]->p_tagtable
                            ,ifunc->param[ifn][k]->p_typetable ,0
                            ,ifunc->param[ifn][k]->isconst));
    pformat += temp;
    temp.Format("(long)(&%s)",paraname());
    pbody += temp;
  }
  else {
    switch(ifunc->param[ifn][k]->type) {
    case 'u':
       temp.Format("(%s)(%%ld)"
              ,G__type2string(ifunc->param[ifn][k]->type
                              ,ifunc->param[ifn][k]->p_tagtable
                              ,ifunc->param[ifn][k]->p_typetable ,0
                              ,ifunc->param[ifn][k]->isconst));
      pformat += temp;
      temp.Format("&%s",paraname());
      pbody += temp;
      break;
    case 'd':
    case 'f':
       temp.Format("(%s)%%g"
              ,G__type2string(ifunc->param[ifn][k]->type
                              ,ifunc->param[ifn][k]->p_tagtable
                              ,ifunc->param[ifn][k]->p_typetable ,0
                              ,ifunc->param[ifn][k]->isconst));
      pformat += temp;
      temp = paraname;
      pbody += temp;
      break;
    default:
       temp.Format("(%s)(%%ld)"
              ,G__type2string(ifunc->param[ifn][k]->type
                              ,ifunc->param[ifn][k]->p_tagtable
                              ,ifunc->param[ifn][k]->p_typetable ,0
                              ,ifunc->param[ifn][k]->isconst));
      pformat += temp;
      temp.Format("(long)%s",paraname());
      pbody += temp;
      break;
    }
  }
}

/**************************************************************************
* G__cppstub_constructor()
*
**************************************************************************/
static void G__cppstub_genconstructor(FILE * /* fp */,int tagnum
                                      ,int /* ifn */,G__ifunc_table_internal * /* ifunc */)
{
  G__fprinterr(G__serr,"Limitation: Can not make STUB constructor, class %s\n"
          ,G__fulltagname(tagnum,1));
}

/**************************************************************************
* G__cppstub_destructor()
*
**************************************************************************/
static void G__cppstub_gendestructor(FILE * /* fp */,int tagnum
                                     ,int /* ifn */,G__ifunc_table_internal * /* ifunc */)
{
  G__fprinterr(G__serr,"Limitation: Can not make STUB destructor, class %s\n"
          ,G__fulltagname(tagnum,1));
}

/**************************************************************************
* G__cppstub_genfunc()
*
**************************************************************************/
static void G__cppstub_genfunc(FILE *fp,int tagnum,int ifn,G__ifunc_table_internal *ifunc)
{
  int k;
  G__FastAllocString pformat(G__ONELINE);
  G__FastAllocString pbody(G__LONGLINE);

  /*******************************************************************
  * Function header
  *******************************************************************/
  if(-1==tagnum
     ) {
    fprintf(fp,"%s %s(\n"
            ,G__type2string(ifunc->type[ifn],ifunc->p_tagtable[ifn]
                            ,ifunc->p_typetable[ifn],ifunc->reftype[ifn]
                            ,ifunc->isconst[ifn])
            ,ifunc->funcname[ifn]);
  }
  else {
    fprintf(fp,"%s "
            ,G__type2string(ifunc->type[ifn],ifunc->p_tagtable[ifn]
                            ,ifunc->p_typetable[ifn],ifunc->reftype[ifn]
                            ,ifunc->isconst[ifn])
            );
    fprintf(fp,"%s::%s(\n",G__fulltagname(tagnum,1),ifunc->funcname[ifn]);
  }

  if(G__clock) {
    for(k=0;k<ifunc->para_nu[ifn];k++) {
      if(k) fprintf(fp,",");
      if(ifunc->param[ifn][k]->name) {
        fprintf(fp,"%s",ifunc->param[ifn][k]->name);
      }
      else {
        fprintf(fp,"a%d",k);
      }
    }
    fprintf(fp,")\n");
    for(k=0;k<ifunc->para_nu[ifn];k++) {
      fprintf(fp,"%s" ,G__type2string(ifunc->param[ifn][k]->type
                                      ,ifunc->param[ifn][k]->p_tagtable
                                      ,ifunc->param[ifn][k]->p_typetable
                                      ,ifunc->param[ifn][k]->reftype
                                      ,ifunc->param[ifn][k]->isconst));
      if(ifunc->param[ifn][k]->name) {
        fprintf(fp," %s;\n",ifunc->param[ifn][k]->name);
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
      fprintf(fp,"%s" ,G__type2string(ifunc->param[ifn][k]->type
                                      ,ifunc->param[ifn][k]->p_tagtable
                                      ,ifunc->param[ifn][k]->p_typetable
                                      ,ifunc->param[ifn][k]->reftype
                                      ,ifunc->param[ifn][k]->isconst));
      if(ifunc->param[ifn][k]->name) {
        fprintf(fp," %s",ifunc->param[ifn][k]->name);
      }
      else {
        fprintf(fp," a%d",k);
      }
    }
    if(ifunc->isconst[ifn]&G__CONSTFUNC) fprintf(fp,") const {\n");
    else fprintf(fp,") {\n");
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
  fprintf(fp,"  snprintf(funccall,G__LONGLINE,\"%s(%s)\"%s);\n"
          ,ifunc->funcname[ifn],pformat(),pbody());
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
void G__cppstub_memfunc(FILE *fp)
{
  int i,j;
  struct G__ifunc_table_internal *ifunc;

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

      /* member function interface */
      fprintf(fp,"\n/* %s */\n",G__fulltagname(i,0));

      while(ifunc) {
        for(j=0;j<ifunc->allifunc;j++) {
          
          if(
             ifunc->hash[j]!=0 &&
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
              G__cppstub_genfunc(fp,i,j,ifunc);
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
void G__cppstub_func(FILE *fp)
{
  int j;
  struct G__ifunc_table_internal *ifunc;

  fprintf(fp,"\n/*********************************************************\n");
  fprintf(fp,"* Global function Stub\n");
  fprintf(fp,"*********************************************************/\n");
  ifunc = &G__ifunc;

  /* member function interface */
  while(ifunc) {
    for(j=0;j<ifunc->allifunc;j++) {
      if((G__CPPSTUB==ifunc->globalcomp[j]||G__CSTUB==ifunc->globalcomp[j])&& 
         ifunc->hash[j]) {
        
        G__cppstub_genfunc(fp,-1,j,ifunc);

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
void G__set_stubflags(G__dictposition *dictpos)
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
    struct G__ifunc_table_internal *ifunc;
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
  }

  if(dictpos->ifunc) {
    struct G__ifunc_table_internal *dictpos_ifunc = G__get_ifunc_internal(dictpos->ifunc);
    struct G__ifunc_table_internal *ifunc = dictpos_ifunc;
    while(ifunc) {
      if(ifunc==dictpos_ifunc) ifn = dictpos->ifn;
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
}

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
