/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file ifunc.c
 ************************************************************************
 * Description:
 *  interpret function and new style compiled function
 ************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto 
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

extern "C" void G__exec_alloc_lock();
extern "C" void G__exec_alloc_unlock();

extern "C" {

extern int G__const_noerror;

static int G__calldepth = 0;

/***********************************************************************
 * G__savestring()
 ***********************************************************************/
char *G__savestring(char **pbuf,char *name)
{
  G__ASSERT(pbuf);
  if(*pbuf) free((void*)(*pbuf));
  *pbuf = (char*)malloc(strlen(name)+1);
  return(strcpy(*pbuf,name));
}

#ifndef G__OLDIMPLEMENTATION1167
/***********************************************************************
 * G__reftypeparam()
***********************************************************************/
void G__reftypeparam(G__ifunc_table *p_ifunc,int ifn,G__param *libp)
{
  int itemp;
  for(itemp=0;itemp<p_ifunc->para_nu[ifn]
        && itemp<libp->paran
        ;itemp++) {
    if(G__PARAREFERENCE==p_ifunc->para_reftype[ifn][itemp] &&
       p_ifunc->para_type[ifn][itemp]!=libp->para[itemp].type) {
      switch(p_ifunc->para_type[ifn][itemp]) {
      case 'c': 
        libp->para[itemp].ref = (long)G__Charref(&libp->para[itemp]);
        break;
      case 's':
        libp->para[itemp].ref = (long)G__Shortref(&libp->para[itemp]);
        break;
      case 'i':
        libp->para[itemp].ref = (long)G__Intref(&libp->para[itemp]);
        break;
      case 'l':
        libp->para[itemp].ref = (long)G__Longref(&libp->para[itemp]);
        break;
      case 'b':
        libp->para[itemp].ref = (long)G__UCharref(&libp->para[itemp]);
        break;
      case 'r':
        libp->para[itemp].ref = (long)G__UShortref(&libp->para[itemp]);
        break;
      case 'h':
        libp->para[itemp].ref = (long)G__UIntref(&libp->para[itemp]);
        break;
      case 'k':
        libp->para[itemp].ref = (long)G__ULongref(&libp->para[itemp]);
        break;
      case 'f':
        libp->para[itemp].ref = (long)G__Floatref(&libp->para[itemp]);
        break;
      case 'd':
        libp->para[itemp].ref = (long)G__Doubleref(&libp->para[itemp]);
        break;
      case 'g':
#ifdef G__BOOL4BYTE
        libp->para[itemp].ref = (long)G__Intref(&libp->para[itemp]);
#else
        libp->para[itemp].ref = (long)G__UCharref(&libp->para[itemp]);
#endif
        break;
      case 'n':
        libp->para[itemp].ref = (long)G__Longlongref(&libp->para[itemp]);
        break;
      case 'm':
        libp->para[itemp].ref = (long)G__ULonglongref(&libp->para[itemp]);
        break;
      case 'q':
        libp->para[itemp].ref = (long)G__Longdoubleref(&libp->para[itemp]);
        break;
      }
    }
  }
}
#endif

/***********************************************************************
 * G__warn_refpromotion
***********************************************************************/
static void G__warn_refpromotion(G__ifunc_table *p_ifunc,int ifn, int itemp
                                 ,G__param *libp) 
{
  if(G__PARAREFERENCE==p_ifunc->para_reftype[ifn][itemp] &&
     'u'!=p_ifunc->para_type[ifn][itemp] &&
     p_ifunc->para_type[ifn][itemp]!=libp->para[itemp].type &&
     0!=libp->para[itemp].obj.i &&
     G__VARIABLE==p_ifunc->para_isconst[ifn][itemp]) {
#ifdef G__OLDIMPLEMENTATION1167
    if(G__dispmsg>=G__DISPWARN) {
      G__fprinterr(G__serr,"Warning: implicit type conversion of non-const reference arg %d",itemp);
      G__printlinenum();
    }
#endif
  }
}


#ifdef G__ASM_WHOLEFUNC
/***********************************************************************
* G__asm_freebytecode()
***********************************************************************/
void G__free_bytecode(G__bytecodefunc *bytecode)
{
  if(bytecode) {
    if(bytecode->asm_name) free((void*)bytecode->asm_name);
    if(bytecode->pstack) free((void*)bytecode->pstack);
    if(bytecode->pinst) free((void*)bytecode->pinst);
    if(bytecode->var) {
      G__destroy(bytecode->var,G__BYTECODELOCAL_VAR) ;
      free((void*)bytecode->var);
    }
    free((void*)bytecode);
  }
}

/***********************************************************************
* G__asm_storebytecodefunc()
***********************************************************************/
void G__asm_storebytecodefunc(G__ifunc_table *ifunc,int ifn,
                              G__var_array *var,G__value *pstack,
                              int sp,long *pinst,int instsize)
{
  struct G__bytecodefunc *bytecode;

  /* check if the function is already compiled, replace old one */
  if(ifunc->pentry[ifn]->bytecode) {
    G__genericerror("Internal error: G__asm_storebytecodefunc duplicated");
  }

  /* allocate bytecode buffer */
  bytecode = (struct G__bytecodefunc*)malloc(sizeof(struct G__bytecodefunc));
  ifunc->pentry[ifn]->bytecode = bytecode;

  /* store function ID */
  bytecode->ifunc = ifunc;
  bytecode->ifn = ifn;

  /* copy local variable table */
  bytecode->var = var;
  bytecode->varsize = G__struct.size[G__tagdefining];

  /* copy instruction */
  bytecode->pinst = (long*)malloc(sizeof(long)*instsize+8);
  memcpy(bytecode->pinst,pinst,sizeof(long)*instsize+1);
  bytecode->instsize = instsize;

  /* copy constant data stack */
  bytecode->stacksize = G__MAXSTACK-sp;
  bytecode->pstack = (G__value*)malloc(sizeof(G__value)*bytecode->stacksize);
  memcpy((void*)bytecode->pstack,(void*)(&pstack[sp])
         ,sizeof(G__value)*bytecode->stacksize);

  /* copy compiled and library function name buffer */
  if(0==G__asm_name_p) {
    if(G__asm_name) free(G__asm_name);
    bytecode->asm_name = (char*)NULL;
  }
  else {
    bytecode->asm_name = G__asm_name;
  }

#ifdef G__OLDIMPLEMENtATION1578 /* Problem  t1048.cxx */
  /* store pointer to function */
  ifunc->pentry[ifn]->tp2f = (void*)bytecode;
#endif
}

/**************************************************************************
* from bc_autoobj.h/cxx
**************************************************************************/
void* G__allocheapobjectstack G__P((struct G__ifunc_table* ifunc,int ifn,int scopelevel));
void G__copyheapobjectstack G__P((void* p,G__value* result,struct G__ifunc_table *ifunc,int ifn));



/***********************************************************************
* G__noclassargument()
*  stops bytecode compilation if class object is passed as argument
***********************************************************************/
int G__noclassargument(G__ifunc_table *ifunc,int iexist)
{
  int i;
  for(i=0;i<ifunc->para_nu[iexist];i++) {
    if('u'==ifunc->para_type[iexist][i] &&
       G__PARAREFERENCE!=ifunc->para_reftype[iexist][i]) {
      /* return false if class/struct object and non-reference type arg */
      return(0);
    }
  }
  return(1);
}

/***********************************************************************
* G__compile_bytecode()
*
***********************************************************************/
int G__compile_bytecode(G__ifunc_table *ifunc,int iexist)
{
  G__value buf;
  struct G__param para; /* This one is only dummy */
  struct G__input_file store_ifile;
  int store_prerun=G__prerun;
  int store_asm_index = G__asm_index;
  int store_no_exec = G__no_exec;
  int store_asm_exec = G__asm_exec;
  int store_tagdefining = G__tagdefining;
  int store_asm_noverflow = G__asm_noverflow;
  int funcstatus;
  long store_globalvarpointer = G__globalvarpointer;
  char funcname[G__ONELINE];
  int store_dispsource = G__dispsource;
  if(G__step||G__stepover) G__dispsource=0;

  if(
     G__xrefflag ||
     (
     ifunc->pentry[iexist]->size<G__ASM_BYTECODE_FUNC_LIMIT
     && 0==G__def_struct_member 
     && ('u'!=ifunc->type[iexist]||G__PARAREFERENCE==ifunc->reftype[iexist])
     && (0==ifunc->para_nu[iexist] ||
         (ifunc->ansi[iexist] && G__noclassargument(ifunc,iexist)))
      )
      ) {

    para.paran=0;
    para.para[0]=G__null;
    G__tagdefining = G__MAXSTRUCT-1;
    G__struct.type[G__tagdefining] = 's';
    G__struct.size[G__tagdefining] = 0;
    G__no_exec = 0;
    G__prerun = 0;
    G__asm_exec = 1;
    G__asm_wholefunction = G__ASM_FUNC_COMPILE;
    G__asm_noverflow = 0;
    store_ifile = G__ifile;
    G__asm_index = iexist;
    ++G__templevel;
    ++G__calldepth;
    strcpy(funcname,ifunc->funcname[iexist]);
    if(-1==ifunc->tagnum) funcstatus = G__TRYNORMAL;
    else                  funcstatus = G__CALLMEMFUNC;
    G__init_jumptable_bytecode();
    G__interpret_func(&buf,funcname,&para
                      ,ifunc->hash[iexist] ,ifunc
                      ,G__EXACT,funcstatus);
    G__init_jumptable_bytecode();
    --G__calldepth;
    --G__templevel;
    G__tagdefining = store_tagdefining;
    G__asm_exec = store_asm_exec;
    G__no_exec = store_no_exec;
    G__prerun = store_prerun;
    G__asm_index = store_asm_index;
    G__asm_wholefunction = G__ASM_FUNC_NOP;
    G__ifile = store_ifile;
    G__asm_noverflow = store_asm_noverflow;
    G__globalvarpointer = store_globalvarpointer;
  }
  else if(G__asm_dbg) {
    G__fprinterr(G__serr,"!!!bytecode compilation %s not tried either because\n"
            ,ifunc->funcname[iexist]);
    G__fprinterr(G__serr,"    function is longer than %d lines\n"
            ,G__ASM_BYTECODE_FUNC_LIMIT);
    G__fprinterr(G__serr,"    function returns class object or reference type\n");
    G__fprinterr(G__serr,"    function is K&R style\n");
    G__printlinenum();
  }
    
  if(ifunc->pentry[iexist]->bytecode) {
    if(0==G__xrefflag) 
      ifunc->pentry[iexist]->bytecodestatus = G__BYTECODE_SUCCESS;
    else
      ifunc->pentry[iexist]->bytecodestatus = G__BYTECODE_ANALYSIS;
  }
  else if(0==G__def_struct_member)
    ifunc->pentry[iexist]->bytecodestatus = G__BYTECODE_FAILURE;

  G__dispsource = store_dispsource;
  return(ifunc->pentry[iexist]->bytecodestatus);
}

/***********************************************************************
*
***********************************************************************/
#define G__MAXGOTOLABEL 30

struct G__gotolabel {
 int pc;
 char *label;
};
static int G__ngoto  = 0 ;
static int G__nlabel = 0 ;
static struct G__gotolabel G__gototable[G__MAXGOTOLABEL];
static struct G__gotolabel G__labeltable[G__MAXGOTOLABEL];

/***********************************************************************
* G__free_gotolabel()
***********************************************************************/
static void G__free_gotolabel(struct G__gotolabel *pgotolabel,int *pn)
{
  while(*pn>0) {
    --(*pn);
    free((char*)pgotolabel[*pn].label);
  }
}

/***********************************************************************
* G__init_jumptable_bytecode()
*
***********************************************************************/
void G__init_jumptable_bytecode()
{
  G__free_gotolabel(G__labeltable,&G__nlabel);
  G__free_gotolabel(G__gototable,&G__ngoto);
}

/***********************************************************************
* G__add_label_bytecode()
*
***********************************************************************/
void G__add_label_bytecode(char *label)
{
  if(G__nlabel<G__MAXGOTOLABEL) {
    int len=strlen(label);
    if(len) {
      G__labeltable[G__nlabel].pc = G__asm_cp;
      label[len-1] = 0;
      G__labeltable[G__nlabel].label = (char*)malloc(strlen(label)+1);
      strcpy(G__labeltable[G__nlabel].label,label);
      ++G__nlabel;
    }
  }
  else {
    G__abortbytecode();
  }
}

/***********************************************************************
* G__add_jump_bytecode()
*
***********************************************************************/
void G__add_jump_bytecode(char *label)
{
  if(G__ngoto<G__MAXGOTOLABEL) {
    int len=strlen(label);
    if(len) {
      G__gototable[G__ngoto].pc = G__asm_cp+1;
      G__asm_inst[G__asm_cp]=G__JMP;
      G__inc_cp_asm(2,0);
      G__gototable[G__ngoto].label = (char*)malloc(strlen(label)+1);
      strcpy(G__gototable[G__ngoto].label,label);
      ++G__ngoto;
    }
  }
  else {
    G__abortbytecode();
  }
}

/***********************************************************************
* G__resolve_jumptable_bytecode()
*
***********************************************************************/
void G__resolve_jumptable_bytecode()
{
  if(G__asm_noverflow) {
    int i,j;
    for(j=0;j<G__nlabel;j++) {
      for(i=0;i<G__ngoto;i++) {
        if(strcmp(G__gototable[i].label,G__labeltable[j].label)==0) {
          G__asm_inst[G__gototable[i].pc] = G__labeltable[j].pc;
        }
      }
    }
  }
  G__init_jumptable_bytecode();
}

#endif /* G__ASM_WHOLEFUNC */

/***********************************************************************
* G__istypename()
*
* true if fundamental type, class, struct, typedef, template class name
***********************************************************************/
int G__istypename(char *temp)
{
  if(strncmp(temp,"class ",6)==0) temp += 6;
  else if(strncmp(temp,"struct ",7)==0) temp += 7;
  else if(strncmp(temp,"enum ",5)==0) temp += 5;
  if(strchr(temp,'(') || strchr(temp,')') || strchr(temp,'|')) return(0);
  /* char *p; */
  /* char buf[G__MAXNAME*2]; */
  if('\0'==temp[0]) return(0);
  if(strcmp(temp,"int")==0||
     strcmp(temp,"short")==0||
     strcmp(temp,"char")==0||
     strcmp(temp,"long")==0||
     strcmp(temp,"float")==0||
     strcmp(temp,"double")==0||
     (strncmp(temp,"unsigned",8)==0 && 
      (strcmp(temp,"unsigned")==0||
       strcmp(temp,"unsignedchar")==0||
       strcmp(temp,"unsignedshort")==0||
       strcmp(temp,"unsignedint")==0||
       strcmp(temp,"unsignedlong")==0))||
     strcmp(temp,"signed")==0||
     strcmp(temp,"const")==0||
     strcmp(temp,"void")==0||
     strcmp(temp,"FILE")==0||
     strcmp(temp,"class")==0||
     strcmp(temp,"struct")==0||
     strcmp(temp,"union")==0||
     strcmp(temp,"enum")==0||
     strcmp(temp,"register")==0||
     strcmp(temp,"bool")==0||
     (G__iscpp && strcmp(temp,"typename")==0)||
     -1!=G__defined_typename(temp)||
     -1!=G__defined_tagname(temp,2)||
     G__defined_templateclass(temp)) {
    return(1);
  }

  if(G__fpundeftype) return(1);

  return(0);
}


/***********************************************************************
* void G__make_ifunctable(funcheader)
*
* Called by
*   G__exec_statement()
*   G__define_var()
*   G__define_var()
*   G__define_var()
*
*  No change will be needed to support ANSI function prototype
* G__interpret_func() should be changed.
*
***********************************************************************/
void G__make_ifunctable(char *funcheader) /* funcheader = 'funcname(' */
{
  int /* ifn=0, */ iin=0;
  int cin='\0';
  char paraname[G__LONGLINE];
  int func_now;
  int iexist;
  struct G__ifunc_table *ifunc;
  char store_type;
  int store_tagnum,store_typenum;
  int isparam;
  int store_access;
  int paranu;
  int dobody=0;
#ifdef G__FRIEND
  struct G__friendtag *friendtag;
#endif
#ifdef G__NEWINHERIT
  int basen;
  struct G__inheritance *baseclass;
#endif
  int isvoid=0;

  /*****************************************************
   * to get type of function parameter
   *****************************************************/
  int iin2;
  fpos_t temppos;
  int store_line_number;  /* bug fix 3 mar 1993 */
  int store_def_struct_member;
  struct G__ifunc_table *store_ifunc;
  struct G__ifunc_table *store_ifunc_tmp;

  /* system check */
  G__ASSERT(G__prerun);
  
  store_ifunc = G__p_ifunc;
  if(G__def_struct_member && G__def_tagnum != -1) {
    /* no need for incremental setup */
    G__p_ifunc = G__struct.memfunc[G__def_tagnum] ;
  }
  
  /* Store ifunc to check if same function already exists */
  ifunc = G__p_ifunc;
  
  /* Get to the last page of interpreted function list */
  while(G__p_ifunc->next) G__p_ifunc = G__p_ifunc->next;
  if(G__p_ifunc->allifunc==G__MAXIFUNC) {
    /* This case is used only when complicated template instantiation is done 
     * during reading argument list 'f(vector<int> &x) { }' */
    G__p_ifunc->next=(struct G__ifunc_table *)malloc(sizeof(struct G__ifunc_table));
    G__p_ifunc->next->allifunc=0;
    G__p_ifunc->next->next=(struct G__ifunc_table *)NULL;
    G__p_ifunc->next->page = G__p_ifunc->page+1;
    G__p_ifunc->next->tagnum = G__p_ifunc->tagnum;
    G__p_ifunc = G__p_ifunc->next;
    {
      int ix;
      for(ix=0;ix<G__MAXIFUNC;ix++) {
        G__p_ifunc->funcname[ix] = (char*)NULL;
        G__p_ifunc->userparam[ix] = 0;
      }
    }
  }
  store_ifunc_tmp = G__p_ifunc;
  
  /* set funcname to G__p_ifunc */
  G__func_now=G__p_ifunc->allifunc;
  G__func_page=G__p_ifunc->page;
  func_now = G__func_now;

#ifndef G__OLDIMPLEMENTATION2027
  if('~'==funcheader[0] && 0==ifunc->hash[0]) {
    G__p_ifunc=ifunc;
    G__func_now=0;
    G__func_page=ifunc->page;
    func_now = G__func_now;
  }
#endif

  if('*'==funcheader[0]) {
    if('*'==funcheader[1]) {
      int numstar=2;
      while('*'==funcheader[numstar]) ++numstar;
      if(strlen(funcheader+2)>G__LONGLINE-1) {
        G__fprinterr(G__serr,
                "Limitation: Function name length overflow strlen(%s)>%d"
                ,funcheader+2,G__LONGLINE-1);
        G__genericerror((char*)NULL);
        funcheader[G__MAXNAME+1]=0;
      }
      G__savestring(&G__p_ifunc->funcname[func_now],funcheader+numstar);
      if(isupper(G__var_type)) {
        switch(G__reftype) {
        case G__PARANORMAL:
          G__reftype=G__PARAP2P2P;
          break;
        default:
          G__reftype += 2;
          break;
        }
      }
      else {
        switch(G__reftype) {
        case G__PARANORMAL:
          G__reftype=G__PARAP2P;
          break;
        case G__PARAP2P:
          G__reftype=G__PARAP2P2P;
          break;
        default:
          G__reftype += 1;
          break;
        }
      }
      G__reftype += numstar-2 ;
    }
    else {
      if(strlen(funcheader+1)>G__LONGLINE-1) {
        G__fprinterr(G__serr,
                "Limitation: Function name length overflow strlen(%s)>%d"
                ,funcheader+1,G__LONGLINE-1);
        G__genericerror((char*)NULL);
        funcheader[G__MAXNAME]=0;
      }
      G__savestring(&G__p_ifunc->funcname[func_now],funcheader+1);
      if(isupper(G__var_type)) {
        switch(G__reftype) {
        case G__PARANORMAL:
          G__reftype=G__PARAP2P;
          break;
        case G__PARAP2P:
          G__reftype=G__PARAP2P2P;
          break;
        default:
          G__reftype += 1;
          break;
        }
      }
    }
    G__var_type = toupper(G__var_type);
  }
  else {
    char *pt1;
    if(strlen(funcheader)>G__LONGLINE-1) {
      funcheader[G__MAXNAME-1]=0;
      G__fprinterr(G__serr,
              "Limitation: Function name length overflow strlen(%s)>%d"
              ,funcheader,G__LONGLINE-1);
      G__genericerror((char*)NULL);
      funcheader[G__MAXNAME-1]=0;
    }
    if(strncmp(funcheader,"operator ",9)==0) {
      char *oprtype= funcheader+9;
      if(strcmp(oprtype,"char")==0 ||
         strcmp(oprtype,"short")==0 ||
         strcmp(oprtype,"int")==0 ||
         strcmp(oprtype,"long")==0 ||
         strcmp(oprtype,"unsigned char")==0 ||
         strcmp(oprtype,"unsigned short")==0 ||
         strcmp(oprtype,"unsigned int")==0 ||
         strcmp(oprtype,"unsigned long")==0 ||
         strcmp(oprtype,"float")==0 ||
         strcmp(oprtype,"double")==0) {
      }
      else {
        int oprtypenum ;
        oprtype[strlen(oprtype)-1]=0;
        oprtypenum = G__defined_typename(oprtype);
        if(-1!=oprtypenum && -1==G__newtype.tagnum[oprtypenum] &&
           -1!=G__newtype.parent_tagnum[oprtypenum]) {
#ifndef G__NEVER
          strcpy(oprtype,G__type2string(G__newtype.type[oprtypenum] ,-1,-1
                                        ,G__newtype.reftype[oprtypenum]
                                        ,G__newtype.isconst[oprtypenum]));
#else
          strcpy(oprtype
                 ,G__fulltagname(G__newtype.parent_tagnum[oprtypenum],1));
          strcat(oprtype,"::");
          strcat(oprtype,G__newtype.name[oprtypenum]);
#endif
        }
        strcat(oprtype,"(");
      }
    }
    G__savestring(&G__p_ifunc->funcname[func_now],funcheader);
    if((strstr(funcheader,">>")!=NULL && strchr(funcheader,'<')!=NULL)
       ||(strstr(funcheader,"<<")!=NULL && strchr(funcheader,'>')!=NULL) ) {

      int maxpt2 = strlen(G__p_ifunc->funcname[func_now])+20; /* allow 20 extra spaces */
      char *pt2 = (char*)malloc(maxpt2);
      strcpy(pt2,G__p_ifunc->funcname[func_now]);
      free((void*)(G__p_ifunc->funcname[func_now]));
      G__p_ifunc->funcname[func_now] = pt2;

      if((char*)NULL!=strstr(funcheader,"operator<<") &&
         (char*)NULL!=strchr(funcheader,'>') ) { 
        /* we might have operator< <> or operator< <double> 
          or operator<< <> or operator<< <double>
          with the space missing */
        pt2 = pt2 + strlen( "operator<" );
        pt1 = funcheader + strlen( "operator<" );
        /*char *pt2 = G__p_ifunc->funcname[func_now] + strlen( "operator<" );*/
        if( *(pt2+1)=='<' ) {
          /* we have operator<< <...> */
          ++pt2;
          ++pt1;
        }
        *pt2 = ' ';
        ++pt2;
        strcpy(pt2,pt1);
      } 
      else if((char*)NULL!=strstr(pt2,"operator>>") &&
              (char*)NULL!=strchr(pt2,'<') ) {
        /* we might have operator>><>  */
        /* we have nothing to do ... yet (we may have to do something 
           for nested templates */
        pt2 = pt2 + strlen("operator>>");
        pt1 = funcheader + strlen( "operator>>" );
      } 
      else {
        pt1 = funcheader;
      }
      while ((char*)NULL!=(pt1=strstr(pt1,">>"))) {
        char *pt3;
        pt3 = strstr(pt2,">>");
        ++pt3;
        *pt3 = ' ';
        ++pt3;
        ++pt1;
        pt2 = pt3;
        strcpy(pt3,pt1);
      }
    }
  }
  G__p_ifunc->funcname[func_now][strlen(G__p_ifunc->funcname[func_now])-1]
    ='\0';

  /******************************************************
   * conv<B>(x) -> conv<ns::B>(x)
   ******************************************************/
  G__p_ifunc->funcname[func_now] = 
    G__rename_templatefunc(G__p_ifunc->funcname[func_now],1);

  G__hash(G__p_ifunc->funcname[func_now],G__p_ifunc->hash[func_now],iin2);

  G__p_ifunc->para_name[func_now][0]=(char*)NULL;




  /*************************************************************
   * check if the function is operator()(), if so, regenerate
   * hash value
   *************************************************************/
  if(G__HASH_OPERATOR==G__p_ifunc->hash[func_now] &&
     strcmp(G__p_ifunc->funcname[func_now],"operator")==0) {
    {
      char *tmpp=(char*)("operator()");
      G__savestring(&G__p_ifunc->funcname[func_now],tmpp);
    }
    G__p_ifunc->hash[func_now] += ('('+')');
  }

  fgetpos(G__ifile.fp,&G__p_ifunc->entry[func_now].pos);
  G__p_ifunc->entry[func_now].p = (void*)G__ifile.fp;
  G__p_ifunc->entry[func_now].line_number = G__ifile.line_number;
  G__p_ifunc->entry[func_now].filenum = G__ifile.filenum;
#ifdef G__TRUEP2F
  G__p_ifunc->entry[func_now].tp2f = (void*)G__p_ifunc->funcname[func_now];
#endif
#ifdef G__ASM_FUNC
  G__p_ifunc->entry[func_now].size = 0;
#endif
#ifdef G__ASM_WHOLEFUNC
  G__p_ifunc->entry[func_now].bytecode = (struct G__bytecodefunc*)NULL;
  G__p_ifunc->entry[func_now].bytecodestatus = G__BYTECODE_NOTYET;
#endif
  G__p_ifunc->pentry[func_now] = &G__p_ifunc->entry[func_now];
  if(-1==G__p_ifunc->tagnum)
    G__p_ifunc->globalcomp[func_now]=G__default_link?G__globalcomp:G__NOLINK;
  else
    G__p_ifunc->globalcomp[func_now]=G__globalcomp;
#ifdef G__FRIEND
  if(-1==G__friendtagnum) {
    G__p_ifunc->friendtag[func_now]=(struct G__friendtag*)NULL;
  }
  else {
    G__p_ifunc->friendtag[func_now]
      =(struct G__friendtag*)malloc(sizeof(struct G__friendtag));
    G__p_ifunc->friendtag[func_now]->next =(struct G__friendtag*)NULL;
    G__p_ifunc->friendtag[func_now]->tagnum=G__friendtagnum;
  }
#endif

  
  /*************************************************************
   * set type struct and typedef information to G__ifile
   *************************************************************/
  if(G__def_struct_member && G__def_tagnum != -1 && 
     strcmp(G__struct.name[G__def_tagnum],G__p_ifunc->funcname[func_now])==0){
    /* constructor */
    /* illegular handling not to instaitiate temp object for return */
    G__p_ifunc->type[func_now] = 'i'; 
    G__p_ifunc->p_tagtable[func_now] = G__def_tagnum;
    G__p_ifunc->p_typetable[func_now] = G__typenum;
    G__struct.isctor[G__def_tagnum] = 1;
  }
  else {
    G__p_ifunc->type[func_now] = G__var_type;
    G__p_ifunc->p_tagtable[func_now] = G__tagnum;
    G__p_ifunc->p_typetable[func_now] = G__typenum;
  }
  
  G__p_ifunc->reftype[func_now]=G__reftype;
  G__p_ifunc->isconst[func_now]=(G__SIGNEDCHAR_T)G__constvar;
  G__p_ifunc->isexplicit[func_now]=(G__SIGNEDCHAR_T)G__isexplicit;
  G__isexplicit = 0;

  G__reftype=G__PARANORMAL;
  /*
  G__constvar=0;
  G__var_type='p';
  G__tagnum = -1;
  G__typenum = -1;
  */
  
  if(funcheader[0]=='~') {
    /* return type is void if destructor */
    G__p_ifunc->type[func_now] = 'y';
    G__p_ifunc->p_tagtable[func_now] = -1;
    G__p_ifunc->p_typetable[func_now] = -1;
  }

#ifndef G__NEWINHERIT
  G__p_ifunc->isinherit[func_now] = 0;
#endif
  
  /*************************************************************
   * member access control
   *************************************************************/
  if(G__def_struct_member) G__p_ifunc->access[func_now]=G__access;
  else                     G__p_ifunc->access[func_now]=G__PUBLIC;
  G__p_ifunc->staticalloc[func_now] = (char)G__static_alloc;
  
  /*************************************************************
   * initiazlize baseoffset
   *************************************************************/
#ifndef G__NEWINHERIT
  G__p_ifunc->baseoffset[func_now] = 0;
  if(-1 != G__def_tagnum) G__p_ifunc->basetagnum[func_now] = G__def_tagnum;
  else G__p_ifunc->basetagnum[func_now] = G__tagdefining;
#endif
  G__p_ifunc->isvirtual[func_now] = G__virtual;
  G__p_ifunc->ispurevirtual[func_now] = 0;
  
  /* for virtual function, allocate virtual identity member.
   * Set offset of the virtual identity member to 
   * G__struct.virtual_offset[G__p_ifunc->basetagnum[func_now]].
   */
#ifdef G__NEWINHERIT
  if(G__virtual && 
     -1==G__struct.virtual_offset[G__tagdefining]) {
#else
  if(G__virtual && 
     -1==G__struct.virtual_offset[G__p_ifunc->basetagnum[func_now]]) {
#endif
    
    store_tagnum = G__tagnum;
    store_typenum = G__typenum;
    store_type=G__var_type;
    G__tagnum = -1;
    G__typenum = -1;
    G__var_type = 'l';
    store_access=G__access;
#ifdef G__DEBUG2
    G__access=G__PUBLIC;
#else
    G__access=G__PRIVATE;
#endif
    G__letvariable("G__virtualinfo",G__null,&G__global,G__p_local);
    G__access=store_access;
    G__var_type=store_type;
    G__tagnum = store_tagnum;
    G__typenum = store_typenum;
    
#ifdef G__NEWINHERIT
    G__struct.virtual_offset[G__tagdefining]
      = G__struct.size[G__tagdefining]-G__LONGALLOC;
#else
    G__struct.virtual_offset[G__p_ifunc->basetagnum[func_now]]
      = G__struct.size[G__p_ifunc->basetagnum[func_now]]-G__LONGALLOC;
#endif
  }
  G__virtual=0; /* this position is not very best */


  G__p_ifunc->comment[func_now].p.com = (char*)NULL;
  G__p_ifunc->comment[func_now].filenum = -1;
  
  /*************************************************************
   * initialize virtual table index
   *  TODO, may need to this this in other places too, need investigation
   *************************************************************/
  G__p_ifunc->vtblindex[func_now] = -1;
  G__p_ifunc->vtblbasetagnum[func_now] = -1;

  /*************************************************************
   * initialize busy flag
   *************************************************************/
  G__p_ifunc->busy[func_now] = 0;
  
  /*************************************************************
   * store C++ or C 
   *************************************************************/
  G__p_ifunc->iscpp[func_now] = (char)G__iscpp;
  
  /*****************************************************
   * to get type of function parameter
   *****************************************************/
  
  /* remember current file position
   *   func(   int   a   ,  double   b )
   *        ^
   *  if this is an ANSI stype header, the file will be rewinded
   */
  fgetpos(G__ifile.fp,&temppos);
  store_line_number = G__ifile.line_number;  /* bub fix 3 mar 1993 */
  
  
  /* Skip parameter field  'param,param,,,)'  until ')' is found */
  
  /* check if the header is written in ANSI format or not 
   *   type func(param,param);
   *             ^
   *   func(   int   a   ,  double   b )
   *         -  -  - - - -
   */

  isparam=0;
  cin=G__fgetname_template(paraname,"<*&,()=");
  if (strlen(paraname) && isspace(cin)) {
    /* There was an argument and the parsing was stopped by a white
    * space rather than on of ",)*&<=", it is possible that 
    * we have a namespace followed by '::' in which case we have
    * to grab more before stopping! */
    int namespace_tagnum;
    char more[G__LONGLINE];
    
    namespace_tagnum = G__defined_tagname(paraname,2);
    while ( ( ( (namespace_tagnum!=-1)
                && (G__struct.type[namespace_tagnum]=='n') )
              || (strcmp("std",paraname)==0)
              || (paraname[strlen(paraname)-1]==':') )
            && isspace(cin) ) {
      cin = G__fgetname(more,"<*&,)=");
      strcat(paraname,more);
      namespace_tagnum = G__defined_tagname(paraname,2);
    }
  }

  if(paraname[0]) {
    if(strcmp("void",paraname)==0) {
      if(isspace(cin)) cin = G__fgetspace();
      switch(cin) {
      case ',':
      case ')':
        G__p_ifunc->ansi[func_now]=1;
        isvoid=1;
        break;
      case '*':
      case '(':
        G__p_ifunc->ansi[func_now]=1;
        isvoid=0;
        break;
      default:
        G__genericerror("Syntax error");
        G__p_ifunc->ansi[func_now]=0;
        isvoid=1;
        break;
      }
    }
    else if (strcmp("register",paraname)==0) {
      G__p_ifunc->ansi[func_now]=1;
      isvoid=0;
    }
    else if(G__istypename(paraname) || strchr(paraname,'[')
            || -1!=G__friendtagnum
            ) {
      G__p_ifunc->ansi[func_now]=1;
      isvoid=0;
    }
    else {
      if(G__def_struct_member) G__genericerror("Syntax error");
      if(G__globalcomp<G__NOLINK&&!G__nonansi_func
#ifdef G__ROOT
         && strncmp(funcheader,"ClassDef",8)!=0
#endif
         ) {
        if(G__dispmsg>=G__DISPWARN) {
          G__fprinterr(G__serr,"Warning: Unknown type %s in function argument"
                       ,paraname);
          G__printlinenum();
        }
      }
      G__p_ifunc->ansi[func_now]=0;
      isvoid=0;
    }
  }
  else {
    if(G__def_struct_member || G__iscpp) G__p_ifunc->ansi[func_now]=1;
    else                     G__p_ifunc->ansi[func_now]=0;
    isvoid=1;
  }
  if(')'!=cin) cin = G__fignorestream(")");

  G__static_alloc=0;

  
  /*****************************************************
   * to get type of function parameter
   *****************************************************/
  /****************************************************************
   * If ANSI style header, rewind file position to 
   *       func(int a ,double b )   ANSI
   *            ^
   * and check type of paramaters and store it into G__ifunc
   ****************************************************************/
  if(G__p_ifunc->ansi[func_now]) {
    
    if(isvoid
#ifndef G__OLDIMPLEMENTATION2027
       || '~'==funcheader[0]
#endif
       ) {
      G__p_ifunc->para_nu[func_now]=0;
      G__p_ifunc->para_def[func_now][0]=(char*)NULL;
      G__p_ifunc->para_default[func_now][0]=(G__value*)NULL;
    }
    else {
      if(G__dispsource) G__disp_mask=1000;
      fsetpos(G__ifile.fp,&temppos);
      G__ifile.line_number = store_line_number; 
      ++G__p_ifunc->allifunc;
      G__readansiproto(G__p_ifunc,func_now);
      if(store_ifunc_tmp!=G__p_ifunc || func_now!=G__p_ifunc->allifunc) {
        /* This is the normal case. This block is skipped only when 
         * compicated template instantiation is done during reading 
         * argument list 'f(vector<int> &x) { }' */
        --G__p_ifunc->allifunc;
      }
      cin=')';
      if(G__dispsource) G__disp_mask=0;
    }
  } /* end of reading ANSI parameter list */
  
  /****************************************************************
   * K&R style function header 
   *
   *
   ****************************************************************/
  else {
    if(isvoid) G__p_ifunc->para_nu[func_now] = 0;
    else       G__p_ifunc->para_nu[func_now] = -1;
  } /* end of reading K&R parameter list */
  
  /* 
   * Set G__no_exec to skip ifunc body 
   * This statement can be placed after endif.
   */
  G__no_exec = 1;
  
  /* skip space character after 
   *   func(param)      int a; {
   *              ^             */

  if(G__isfuncreturnp2f) {
    /* function returning pointer to function
     *   type (*func(param1))(param2)  { } or ;
     *                      ^ -----> ^   */
    cin=G__fignorestream(")");
    cin=G__fignorestream("(");
    cin=G__fignorestream(")");
  }
  
  cin=G__fgetstream_template(paraname,",;{(");
  if('('==cin) {
    int len = strlen(paraname);
    paraname[len++] = cin;
    cin=G__fgetstream(paraname+len,")");
    len = strlen(paraname);
    paraname[len++] = cin;
    cin=G__fgetstream_template(paraname+len,",;{");
  }
  
  /****************************************************************
   * if header ignore following headers
   * else read func body
   ****************************************************************/
  G__mparen=0;
  if((paraname[0]=='\0'
#ifndef G__OLDIMPLEMETATION817
      ||((strncmp(paraname,"throw",5)==0
         ||strncmp(paraname,"const throw",11)==0)&&0==strchr(paraname,'='))
#endif
      ) &&((cin==',')||(cin==';'))
     && strncmp(funcheader,"ClassDef",8)!=0
     ) {
    /* this is ANSI style func proto without param name */
    if(isparam) {
      fsetpos(G__ifile.fp,&temppos);
      G__ifile.line_number = store_line_number; 
      G__readansiproto(G__p_ifunc,func_now);
      cin = G__fignorestream(",;");
    }
    if(cin==',') {
      /* ignore other prototypes */
      G__fignorestream(";");
      if(G__globalcomp<G__NOLINK)
        G__genericerror("Limitation: Items in header must be separately specified");
    }
    /* entry fp = NULL means this is header */
    G__p_ifunc->entry[func_now].p=(void*)NULL;
    G__p_ifunc->entry[func_now].line_number = -1;
    G__p_ifunc->ispurevirtual[func_now]=0;
    /* Key the class comment off of DeclFileLine rather than ClassDef
     * because ClassDef is removed by a preprocessor */
    if(G__fons_comment && G__def_struct_member &&
       (strncmp(G__p_ifunc->funcname[func_now],"DeclFileLine",12)==0 
        || strncmp(G__p_ifunc->funcname[func_now],"DeclFileLine(",13)==0
        || strncmp(G__p_ifunc->funcname[func_now],"DeclFileLine",12)==0 
        || strncmp(G__p_ifunc->funcname[func_now],"DeclFileLine(",13)==0
       )) {
      G__fsetcomment(&G__struct.comment[G__tagdefining]);
    }
  }

  else if(strncmp(paraname,"=",1)==0 ||
          strncmp(paraname,"const =",7)==0 ||
          strncmp(paraname,"const=",6)==0
#ifndef G__OLDIMPLEMETATION817
          ||((strncmp(paraname,"throw",5)==0
              ||strncmp(paraname,"const throw",11)==0)
             &&0!=strchr(paraname,'='))
#endif
          ) {
    char *p;
    p=strchr(paraname,'=');
    if(0!=G__int(G__getexpr(p+1))) {
      G__genericerror("Error: invalid pure virtual function initializer");
    }
    /* this is ANSI style func proto without param name */
    if(0==G__p_ifunc->ansi[func_now]) G__p_ifunc->ansi[func_now]=1;
    if(isparam) {
      fsetpos(G__ifile.fp,&temppos);
      G__ifile.line_number = store_line_number; 
      G__readansiproto(G__p_ifunc,func_now);
      cin = G__fignorestream(",;");
    }
    if(cin==',') {
      /* ignore other prototypes */
      G__fignorestream(";");
      if(G__globalcomp<G__NOLINK)
        G__genericerror(
        "Limitation: Items in header must be separately specified");
    }
    /* entry fp = NULL means this is header */
    G__p_ifunc->entry[func_now].p=(void*)NULL;
    G__p_ifunc->entry[func_now].line_number = -1;
    G__p_ifunc->ispurevirtual[func_now]=1;
    if(G__tagdefining>=0) ++G__struct.isabstract[G__tagdefining];
    if('~'==G__p_ifunc->funcname[func_now][0]) {
      if(G__dispmsg>=G__DISPWARN) {
        G__fprinterr(G__serr,"Warning: Pure virtual destructor may cause problem. Define as 'virtual %s() { }'"
                     ,G__p_ifunc->funcname[func_now]
                     );
        G__printlinenum();
      }
    }
    if(0==strncmp(paraname,"const",5))
      G__p_ifunc->isconst[func_now]|=G__CONSTFUNC;
  }

  else if(strcmp(paraname,"const")==0 ||
          strcmp(paraname,"const ")==0) {
    /* this is ANSI style func proto without param name */
    if(0==G__p_ifunc->ansi[func_now]) G__p_ifunc->ansi[func_now]=1;
    if(isparam) {
      fsetpos(G__ifile.fp,&temppos);
      G__ifile.line_number = store_line_number; 
      G__readansiproto(G__p_ifunc,func_now);
      cin = G__fignorestream(",;{");
    }
    if(cin==',') {/* ignore other prototypes */
      G__fignorestream(";");
      if(G__globalcomp<G__NOLINK)
        G__genericerror(
            "Limitation: Items in header must be separately specified");
    }
    if('{'==cin) {/* it is possible that this is a function body. */
      fseek(G__ifile.fp,-1,SEEK_CUR);
      if(G__dispsource) G__disp_mask=1;
      dobody=1;
    }
    else {/* entry fp = NULL means this is header */
      G__p_ifunc->entry[func_now].p=(void*)NULL;
      G__p_ifunc->entry[func_now].line_number = -1;
    }
    G__p_ifunc->ispurevirtual[func_now]=0;
    G__p_ifunc->isconst[func_now]|=G__CONSTFUNC;
  }


  else if(G__def_struct_member && 
          ('}'==cin || (';'==cin && '\0'!=paraname[0]
                        && ':'!=paraname[0]
                        )
           || (';'==cin && strncmp(funcheader,"ClassDef",8)==0)
           )) {
    /* Function macro as member declaration */
    /* restore file position
     *   func(   int   a   ,  double   b )
     *        ^  <------------------------+
     */
    fsetpos(G__ifile.fp,&temppos);
    G__ifile.line_number = store_line_number; 

    if(G__dispsource) G__disp_mask=1000;
    strcpy(paraname,funcheader);
    cin = G__fgetstream(paraname+strlen(paraname),")");
    iin = strlen(paraname);
    paraname[iin]=')';
    paraname[iin+1]='\0';
    if(G__dispsource) G__disp_mask=0;

    G__no_exec = 0; /* must be set to 1 again after return */
    G__func_now = -1;
    G__p_ifunc = store_ifunc ;

    G__execfuncmacro(paraname,&iin);
    if(!iin) {
      G__genericerror("Error: unrecognized language construct");
    }
    else if(G__fons_comment && G__def_struct_member && 
            (strncmp(paraname,"ClassDef",8)==0 ||
             strncmp(paraname,"ClassDef(",9)==0 ||
             strncmp(paraname,"ClassDefT(",10)==0)) {
      G__fsetcomment(&G__struct.comment[G__tagdefining]);
    }
    
    return; 
  }

  else { 
    /* Body of the function, skip until 
     * 'func(param)  type param;  { '
     *                             ^
     * and rewind file to just before the '{' 
     */
    if('\0'==paraname[0] && isparam) {
      /* Strange case
       *   type f(type) { }; 
       *          ^ <--  ^   */
      fsetpos(G__ifile.fp,&temppos);
      G__ifile.line_number = store_line_number; 
      G__readansiproto(G__p_ifunc,func_now);
      cin = G__fignorestream("{");
    }
    if(G__HASH_MAIN==G__p_ifunc->hash[func_now] &&
       strcmp(G__p_ifunc->funcname[func_now],"main")==0
       && -1==G__def_tagnum
       ) {
      G__ismain=G__MAINEXIST ;
    }
    /* following part is needed to detect inline new/delete in header */
    if(G__CPPLINK==G__globalcomp || R__CPPLINK==G__globalcomp) {
      if(strcmp(G__p_ifunc->funcname[func_now],"operator new")==0&&
         2==G__p_ifunc->para_nu[func_now] &&
         0==(G__is_operator_newdelete&G__MASK_OPERATOR_NEW))
        G__is_operator_newdelete |= G__IS_OPERATOR_NEW;
      if(strcmp(G__p_ifunc->funcname[func_now],"operator delete")==0&&
         0==(G__is_operator_newdelete&G__MASK_OPERATOR_DELETE))
        G__is_operator_newdelete |= G__IS_OPERATOR_DELETE;
    }
    if(':'==paraname[0] && 0==G__p_ifunc->ansi[func_now]) 
      G__p_ifunc->ansi[func_now]=1;
    if(cin!='{') G__fignorestream("{");
    fseek(G__ifile.fp,-1,SEEK_CUR);
    if(G__dispsource) G__disp_mask=1;
    
    /* skip body of the function surrounded by '{' '}'.
     * G__exec_statement() does the job */
    G__p_ifunc->ispurevirtual[func_now]=0;
    
    dobody=1;
  }

  if(G__nonansi_func) G__p_ifunc->ansi[func_now]=0;

#ifdef G__DETECT_NEWDEL
  /****************************************************************
   * operator new(size_t,void*) , operator delete(void*) detection
   * This is only needed for Linux g++
   ****************************************************************/
  if(G__CPPLINK==G__globalcomp || R__CPPLINK==G__globalcomp) {
    if(strcmp(G__p_ifunc->funcname[func_now],"operator new")==0&&
       2==G__p_ifunc->para_nu[func_now] &&
       0==(G__is_operator_newdelete&G__MASK_OPERATOR_NEW))
      G__is_operator_newdelete |= G__IS_OPERATOR_NEW;
    if(strcmp(G__p_ifunc->funcname[func_now],"operator delete")==0&&
       0==(G__is_operator_newdelete&G__MASK_OPERATOR_DELETE))
      G__is_operator_newdelete |= G__IS_OPERATOR_DELETE;
  }
#endif

  if((strcmp(G__p_ifunc->funcname[func_now],"operator delete")==0
      || strcmp(G__p_ifunc->funcname[func_now],"operator delete[]")==0
      ) &&
     -1!=G__p_ifunc->tagnum) {
    G__p_ifunc->staticalloc[func_now] = 1;
  }


  /****************************************************************
   * Set constructor,copy constructor, destructor, operator= flags
   ****************************************************************/
  if(G__def_struct_member && G__def_tagnum != -1) {
    if('~'==G__p_ifunc->funcname[func_now][0]) {
      /* Destructor */
      G__struct.funcs[G__def_tagnum] |= G__HAS_DESTRUCTOR;
    }
    else if(strcmp(G__struct.name[G__def_tagnum]
                   ,G__p_ifunc->funcname[func_now])==0) {
      if(0==G__p_ifunc->para_nu[func_now] || 
         G__p_ifunc->para_default[func_now][0]) {
        /* Default constructor */
        G__struct.funcs[G__def_tagnum] |= G__HAS_DEFAULTCONSTRUCTOR;
      }
      else if((1==G__p_ifunc->para_nu[func_now] || 
               G__p_ifunc->para_default[func_now][1])&&
              G__def_tagnum==G__p_ifunc->para_p_tagtable[func_now][0]&&
              G__p_ifunc->para_reftype[func_now][0]) {
        /* Copy constructor */
        G__struct.funcs[G__def_tagnum] |= G__HAS_COPYCONSTRUCTOR;
      }
      else {
        G__struct.funcs[G__def_tagnum] |= G__HAS_XCONSTRUCTOR;
      }
    }
    else if(strcmp("operator=",G__p_ifunc->funcname[func_now])==0) {
      /* operator= */
        G__struct.funcs[G__def_tagnum] |= G__HAS_ASSIGNMENTOPERATOR;
    }
    else if (!strcmp("operator new", G__p_ifunc->funcname[func_now])) {
      if (G__p_ifunc->para_nu[func_now] == 1) {
        G__struct.funcs[G__def_tagnum] |= G__HAS_OPERATORNEW1ARG;
      } else if (G__p_ifunc->para_nu[func_now] == 2) {
        G__struct.funcs[G__def_tagnum] |= G__HAS_OPERATORNEW2ARG;
      }
    }
    else if(strcmp("operator delete",G__p_ifunc->funcname[func_now])==0) {
      G__struct.funcs[G__def_tagnum] |= G__HAS_OPERATORDELETE;
    }
  }

  /****************************************************************
   * if same function already exists       copy entry
   * else if body exists or ansi header    increment ifunc
   ****************************************************************/
  ifunc=G__ifunc_exist(G__p_ifunc,func_now ,ifunc,&iexist,0xffff);

  if(G__ifile.filenum<G__nfile) {

    if(ifunc
       ) {
#ifdef G__FRIEND
      if(G__p_ifunc->friendtag[func_now]) {
        if(ifunc->friendtag[iexist]) {
          friendtag=ifunc->friendtag[iexist];
          while(friendtag->next) friendtag=friendtag->next;
          friendtag->next = G__p_ifunc->friendtag[func_now];
        }
        else {
          ifunc->friendtag[iexist]=G__p_ifunc->friendtag[func_now];
        }
      }
#endif
      if(
         ((FILE*)G__p_ifunc->entry[func_now].p!=(FILE*)NULL)
         /* C++ precompiled member function must not be overridden  */
         && (0==G__def_struct_member || 
             G__CPPLINK!=G__struct.iscpplink[G__def_tagnum])
         ) {
        ifunc->ansi[iexist]=G__p_ifunc->ansi[func_now];
        if(-1==G__p_ifunc->para_nu[func_now]) paranu=0;
        else paranu=ifunc->para_nu[iexist];
        if(0==ifunc->ansi[iexist]) 
          ifunc->para_nu[iexist] = G__p_ifunc->para_nu[func_now];
        ifunc->type[iexist]=G__p_ifunc->type[func_now];
        ifunc->p_tagtable[iexist]=G__p_ifunc->p_tagtable[func_now];
        ifunc->p_typetable[iexist]=G__p_ifunc->p_typetable[func_now];
        ifunc->reftype[iexist]=G__p_ifunc->reftype[func_now];
        ifunc->isconst[iexist]|=G__p_ifunc->isconst[func_now];
        ifunc->isexplicit[iexist]|=G__p_ifunc->isexplicit[func_now];
        for(iin=0;iin<paranu;iin++) {
          ifunc->para_reftype[iexist][iin]
            =G__p_ifunc->para_reftype[func_now][iin];
          ifunc->para_p_typetable[iexist][iin]
            =G__p_ifunc->para_p_typetable[func_now][iin];
          if(G__p_ifunc->para_default[func_now][iin]) {
            G__genericerror("Error: Redefinition of default argument");
            if(-1!=(long)G__p_ifunc->para_default[func_now][iin])
              free((void*)G__p_ifunc->para_default[func_now][iin]);
            free((void*)G__p_ifunc->para_def[func_now][iin]);
          }
          G__p_ifunc->para_default[func_now][iin]=(G__value*)NULL;
          G__p_ifunc->para_def[func_now][iin]=(char*)NULL;
          if(ifunc->para_name[iexist][iin]) {
            if(G__p_ifunc->para_name[func_now][iin]) {
              if(dobody && 0!=strcmp(ifunc->para_name[iexist][iin]
                                     ,G__p_ifunc->para_name[func_now][iin])) {
                free((void*)ifunc->para_name[iexist][iin]);
                ifunc->para_name[iexist][iin]
                  =G__p_ifunc->para_name[func_now][iin];
              }
              else {
                free((void*)G__p_ifunc->para_name[func_now][iin]);
              }
              G__p_ifunc->para_name[func_now][iin]=(char*)NULL;
            }
          }
          else {
            ifunc->para_name[iexist][iin]=G__p_ifunc->para_name[func_now][iin];
            G__p_ifunc->para_name[func_now][iin]=(char*)NULL;
          }
        }
        ifunc->entry[iexist]=G__p_ifunc->entry[func_now];
        /* The copy in previous get the wrong tp2f ... let's restore it */
        ifunc->entry[iexist].tp2f = (void*)ifunc->funcname[iexist];
        ifunc->pentry[iexist]= &ifunc->entry[iexist];
        if(1==ifunc->ispurevirtual[iexist]) {
          ifunc->ispurevirtual[iexist]=G__p_ifunc->ispurevirtual[func_now];
          if(G__tagdefining>=0) --G__struct.isabstract[G__tagdefining];
        }
        else if(1==G__p_ifunc->ispurevirtual[func_now]) {
          ifunc->ispurevirtual[iexist]=G__p_ifunc->ispurevirtual[func_now];
        }
        if((ifunc!=G__p_ifunc || iexist!=func_now) && 
           G__p_ifunc->funcname[func_now]) {
          free((void*)G__p_ifunc->funcname[func_now]);
          G__p_ifunc->funcname[func_now] = (char*)NULL;
        }
      } /* of if(G__p_ifunc->entry[func_now].p) */
      else {
        /* Entry not used, must free allocated default argument buffer */
        if(1==G__p_ifunc->ispurevirtual[func_now]) {
          if(G__tagdefining>=0) --G__struct.isabstract[G__tagdefining];
        }
        paranu=G__p_ifunc->para_nu[func_now];
        for(iin=0;iin<paranu;iin++) {
          if(G__p_ifunc->para_name[func_now][iin]) {
            free((void*)G__p_ifunc->para_name[func_now][iin]);
            G__p_ifunc->para_name[func_now][iin]=(char*)NULL;
          }
          if(G__p_ifunc->para_default[func_now][iin] && 
             (&G__default_parameter)!=G__p_ifunc->para_default[func_now][iin]) {
            free((void*)G__p_ifunc->para_default[func_now][iin]);
            G__p_ifunc->para_default[func_now][iin]=(G__value*)NULL;
            free((void*)G__p_ifunc->para_def[func_now][iin]);
            G__p_ifunc->para_def[func_now][iin]=(char*)NULL;
          }
        }
        if((ifunc!=G__p_ifunc || iexist!=func_now) && 
           G__p_ifunc->funcname[func_now]) {
          free((void*)G__p_ifunc->funcname[func_now]);
          G__p_ifunc->funcname[func_now] = (char*)NULL;
        }
      }
      G__func_page=ifunc->page;
      G__func_now = iexist;
      G__p_ifunc=ifunc;
    } /* of if(ifunc) */
    else if((G__p_ifunc->entry[func_now].p || G__p_ifunc->ansi[func_now] ||
             G__nonansi_func || 
             G__globalcomp<G__NOLINK || G__p_ifunc->friendtag[func_now])
            /* This block is skipped only when compicated template 
             * instantiation is done during reading argument list 
             * 'f(vector<int> &x) { }' */
            /* with 1706, do not skip this block with template instantiation
             * in function argument. Do not know exactly why... */
            && (store_ifunc_tmp==G__p_ifunc && func_now==G__p_ifunc->allifunc) 
#ifndef G__OLDIMPLEMENTATION2027
            && '~'!=funcheader[0]
#endif
            ) {
      /* increment allifunc */
      ++G__p_ifunc->allifunc;
      
      /* Allocate and initialize function table list if needed */
      if(G__p_ifunc->allifunc==G__MAXIFUNC) {
        G__p_ifunc->next=(struct G__ifunc_table *)malloc(sizeof(struct G__ifunc_table));
        G__p_ifunc->next->allifunc=0;
        G__p_ifunc->next->next=(struct G__ifunc_table *)NULL;
        G__p_ifunc->next->page = G__p_ifunc->page+1;
        {
          int i,j;
          for (i = 0; i < G__MAXIFUNC; i++) {   
            for (j = 0; j < G__MAXFUNCPARA; j++)
              G__p_ifunc->next->para_p_tagtable[i][j] = 0;
          }
        }
#ifdef G__NEWINHERIT
        G__p_ifunc->next->tagnum = G__p_ifunc->tagnum;
#endif
        {
          int ix;
          for(ix=0;ix<G__MAXIFUNC;ix++) {
            G__p_ifunc->next->funcname[ix]=(char*)NULL;
            G__p_ifunc->next->userparam[ix] = 0;
          }
        }
      }
    } /* if(ifunc) */
    /* else: default parameter does not exist in K&R style 
     * no need to free default parameter buffer */
    
  } /* of G__ifile.filenum<G__nfile */
  else {
    G__fprinterr(G__serr,"Limitation: Function can not be defined in a command line or a tempfile\n");
    G__genericerror("You need to write it in a source file");
  }
  
  if(dobody) {
/* #define G__OLDIMPLEMENTATION1770, this line is moved into G__ci.h  */
    store_def_struct_member = G__def_struct_member;
    G__def_struct_member = 0;
    G__exec_statement();
    G__def_struct_member = store_def_struct_member;

#ifdef G__ASM_FUNC
    if(ifunc) {
      ifunc->pentry[iexist]->size = 
        G__ifile.line_number-ifunc->pentry[iexist]->line_number+1;
    }
    else {
      G__p_ifunc->pentry[func_now]->size = 
        G__ifile.line_number-G__p_ifunc->pentry[func_now]->line_number+1;
    }
#endif /* G__ASM_FUNC */
#ifdef G__ASM_WHOLEFUNC
    /***************************************************************
    * compile as bytecode at load time if -O10 or #pragma bytecode
    ***************************************************************/
    if(G__asm_loopcompile>=10 
       ) {
      if(ifunc) G__compile_bytecode(ifunc,iexist);
      else G__compile_bytecode(G__p_ifunc,func_now);
    }
#endif
  }

  if(G__GetShlHandle()) {
    void *shlp2f = G__FindSymbol(G__p_ifunc,func_now);
    if(shlp2f) {
      G__p_ifunc->pentry[func_now]->tp2f = shlp2f;
      G__p_ifunc->pentry[func_now]->p = (void*)G__DLL_direct_globalfunc;
      G__p_ifunc->pentry[func_now]->filenum = G__GetShlFilenum();
      G__p_ifunc->pentry[func_now]->size = -1;
      G__p_ifunc->pentry[func_now]->line_number = -1;
    }
  }

  if(G__fons_comment && G__def_struct_member) {
    if((ifunc && (strncmp(ifunc->funcname[iexist],"ClassDef",8)==0 ||
                  strncmp(ifunc->funcname[iexist],"ClassDef(",9)==0 ||
                  strncmp(ifunc->funcname[iexist],"ClassDefT(",10)==0||
                  strncmp(ifunc->funcname[iexist],"DeclFileLine",12)==0 ||
                  strncmp(ifunc->funcname[iexist],"DeclFileLine(",13)==0) ) ||
       (!ifunc && (strncmp(G__p_ifunc->funcname[func_now],"ClassDef",8)==0 ||
                   strncmp(G__p_ifunc->funcname[func_now],"ClassDef(",9)==0 ||
                   strncmp(G__p_ifunc->funcname[func_now],"ClassDefT(",10)==0||
                   strncmp(G__p_ifunc->funcname[func_now],"DeclFileLine",12)==0 ||
                   strncmp(G__p_ifunc->funcname[func_now],"DeclFileLine(",13)==0) ) ) {
      G__fsetcomment(&G__struct.comment[G__tagdefining]);
    } else {
      if(ifunc) G__fsetcomment(&ifunc->comment[iexist]);
      else      G__fsetcomment(&G__p_ifunc->comment[func_now]);
    }
  }

#ifdef G__NEWINHERIT
  /***********************************************************************
  * If this is a non-pure virtual member function declaration, decrement
  * isabstract flag in G__struct.
  ***********************************************************************/
  if(-1!=G__tagdefining && !ifunc) {
    baseclass = G__struct.baseclass[G__tagdefining];
    for(basen=0;basen<baseclass->basen;basen++) {
      G__incsetup_memfunc(baseclass->basetagnum[basen]);
      ifunc=G__struct.memfunc[baseclass->basetagnum[basen]];
      ifunc=G__ifunc_exist(G__p_ifunc,func_now ,ifunc,&iexist,G__CONSTFUNC);
      if(ifunc) {
        if(ifunc->ispurevirtual[iexist] &&
           G__struct.isabstract[G__tagdefining]) {
          --G__struct.isabstract[G__tagdefining];
        }
        G__p_ifunc->isvirtual[func_now] |= ifunc->isvirtual[iexist]; 
        break; /* revived by Scott Snyder */
      }
    }
  }
#endif

  /* finishing up */
  G__no_exec = 0;
  G__func_now = -1;
  G__p_ifunc = store_ifunc ;

  return; /* end of G__make_ifunctable */
}

/***********************************************************************
* G__readansiproto()
*
*  func(type , type* , ...)
*       ^
*
***********************************************************************/
int G__readansiproto(G__ifunc_table *ifunc,int func_now)
{
  char paraname[G__LONGLINE];
  char name[G__LONGLINE];
  int c=0,iin=0;
  int tagnum,typenum,type=0,pointlevel,reftype;
  int isunsigned,isdefault;
  int ip,itemp;
  int store_var_type;
  int store_tagnum_default=0,store_def_struct_member_default=0;
  int store_exec_memberfunc=0;
  int arydim;

  ifunc->ansi[func_now] = 1;
  while(')'!=c) {
#ifndef G__OLDIMPLEMENTATION824
    if(G__MAXFUNCPARA==iin) {
      G__fprinterr(G__serr,
             "Limitation: cint can not accept more than %d function arguments"
              ,G__MAXFUNCPARA);
      G__printlinenum();
      G__fignorestream(")");
      return(1);
    }
#endif
    arydim=0;
    pointlevel=0;
    reftype=0; /* this reftype has different meaning from G__reftype */
    typenum = -1;
    tagnum = -1;
    isunsigned=0;
    isdefault=0;
    name[0]='\0';
    ifunc->para_isconst[func_now][iin]=G__VARIABLE;

    /* read typename */
    c=G__fgetname_template(paraname,",)&*[(=");
    if (strlen(paraname) && isspace(c)) {
      /* There was an argument and the parsing was stopped by a white
       * space rather than on of ",)*&<=", it is possible that 
       * we have a namespace followed by '::' in which case we have
       * to grab more before stopping! */
      int namespace_tagnum;
      char more[G__LONGLINE];
      
      namespace_tagnum = G__defined_tagname(paraname,2);
      while ( ( ( (namespace_tagnum!=-1)
                  && (G__struct.type[namespace_tagnum]=='n') )
                || (strcmp("std",paraname)==0)
                || (paraname[strlen(paraname)-1]==':') )
              && isspace(c) ) {
        c = G__fgetname(more,",)&*[(=");
        strcat(paraname,more);
        namespace_tagnum = G__defined_tagname(paraname,2);
      }
    }

    /* check const and unsigned keyword */
    if(strcmp(paraname,"...")==0) {
      ifunc->ansi[func_now] = 2;
      break;
    }
    while(strcmp(paraname,"const")==0 || strcmp(paraname,"register")==0 ||
       strcmp(paraname,"auto")==0 || strcmp(paraname,"volatile")==0
          || (G__iscpp && strcmp(paraname,"typename")==0)
          ) {
      if(strcmp(paraname,"const")==0) 
        ifunc->para_isconst[func_now][iin]|=G__CONSTVAR;
      c=G__fgetname_template(paraname,",)&*[(=");
    }
    if(strcmp(paraname,"unsigned")==0||strcmp(paraname,"signed")==0) {
      if('u'==paraname[0]) isunsigned=-1;
      else isunsigned = 0;
      switch(c) {
      case ',':
      case ')':
      case '&':
      case '[':
      case '(':
      case '=':
      case '*':
        strcpy(paraname,"int");
        break;
      default:
      if(isspace(c)) {
        c=G__fgetname(paraname,",)&*[(=");
      }
      else {
        fpos_t pos;
        int store_line = G__ifile.line_number;
        fgetpos(G__ifile.fp,&pos);
        c=G__fgetname(paraname,",)&*[(=");
        if(strcmp(paraname,"int")!=0 && strcmp(paraname,"long")!=0 &&
           strcmp(paraname,"short")!=0) {
          G__ifile.line_number = store_line;
          fsetpos(G__ifile.fp,&pos);
          strcpy(paraname,"int");
          c=' ';
        }
      }
        break;
      }
    }

    /* determine type */
    if(strcmp(paraname,"struct")==0) {
      c=G__fgetname_template(paraname,",)&*[(=");
      tagnum = G__search_tagname(paraname,'s');
      type = 'u';
    }
    else if(strcmp(paraname,"class")==0) {
      c=G__fgetname_template(paraname,",)&*[(=");
      tagnum = G__search_tagname(paraname,'c');
      type = 'u';
    }
    else if(strcmp(paraname,"union")==0) {
      c=G__fgetname_template(paraname,",)&*[(=");
      tagnum = G__search_tagname(paraname,'u');
      type = 'u';
    }
    else if(strcmp(paraname,"enum")==0) {
      c=G__fgetname_template(paraname,",)&*[(=");
      tagnum = G__search_tagname(paraname,'e');
      type = 'i';
    }
    else if(strcmp(paraname,"int")==0) type='i'+isunsigned;
    else if(strcmp(paraname,"char")==0) type='c'+isunsigned;
    else if(strcmp(paraname,"short")==0) type='s'+isunsigned ;
    else if(strcmp(paraname,"long")==0) {
      if(','!=c && ')'!=c
         && '('!=c
         ) {
        fpos_t pos;
        int store_line = G__ifile.line_number;
        int store_c = c;
        fgetpos(G__ifile.fp,&pos);
        c=G__fgetname(paraname,",)&*[(=");
        if(strcmp(paraname,"long")==0 || strcmp(paraname,"double")==0) {
          if(strcmp(paraname,"long")==0) {
            tagnum = -1;
            typenum = -1;
            if(isunsigned) {
              type='m';
            }
            else {
              type='n';
            }
          }
          else {
            type='q';
          }
        }
        else if(strcmp(paraname,"int")==0) {
          type='l'+isunsigned;
        }
        else {
          G__ifile.line_number = store_line;
          fsetpos(G__ifile.fp,&pos);
          c = store_c;
          type='l'+isunsigned;
        }
      }
      else {
        type='l'+isunsigned;
      }
    }
    else if(strcmp(paraname,"float")==0) type='f'+isunsigned;
    else if(strcmp(paraname,"double")==0) type='d'+isunsigned;
    else if(strcmp(paraname,"bool")==0) type='g';
    else if(strcmp(paraname,"void")==0) type='y';
    else if(strcmp(paraname,"FILE")==0) type='e';
    else {
      int store_tagdefining = G__tagdefining;
      int store_def_tagnum = G__def_tagnum;
      if(-1!=G__friendtagnum) {
        G__tagdefining = G__friendtagnum;
        G__def_tagnum = G__friendtagnum;
      }
      typenum=G__defined_typename(paraname);
      if(-1==typenum) {
        tagnum = G__defined_tagname(paraname,1);
        if(-1==tagnum) {
          if(G__fpundeftype) {
            tagnum=G__search_tagname(paraname,'c');
            fprintf(G__fpundeftype,"class %s; /* %s %d */\n",paraname
                    ,G__ifile.name,G__ifile.line_number);
            fprintf(G__fpundeftype,"#pragma link off class %s;\n\n",paraname);
            G__struct.globalcomp[tagnum] = G__NOLINK;
            type='u';
          }
          else {
            /* In case of f(unsigned x,signed y) */
            type='i'+isunsigned;
            if(!isdigit(paraname[0]) && 0==isunsigned) {
              if(G__dispmsg>=G__DISPWARN) {
                G__fprinterr(G__serr,
        "Warning: Unknown type '%s' in function argument handled as int"
                             ,paraname);
                G__printlinenum();
              }
            }
          }
        }
        else if('e'==G__struct.type[tagnum]) {
          type='i';
        }
        else {
          /* re-evaluate typedef name in case of template class */
          if(strchr(paraname,'<')) typenum=G__defined_typename(paraname);
          type = 'u';
        }
      }
      else {
        tagnum=G__newtype.tagnum[typenum];
        type=G__newtype.type[typenum];
#ifndef G__OLDIMPLEMENTATION1329
        pointlevel += G__newtype.nindex[typenum];
#endif
        ifunc->para_isconst[func_now][iin]|=G__newtype.isconst[typenum];
      }
      G__tagdefining = store_tagdefining;
      G__def_tagnum = store_def_tagnum;
    }

    /* determine pointer level */
    while(','!=c&&')'!=c) {
      switch(c) {
      case '&': 
        ++reftype; 
        c=G__fgetspace();
        break;
      case '[': 
        ++arydim;
        if(G__NOLINK>G__globalcomp && (0==name[0] || '['==name[0])) {
          fpos_t tmp_pos;
          int tmp_line;
          int len=strlen(name);
          name[len++] = c;
          name[len++] = ']';
          c=G__fignorestream("],)"); /* <<< */
          /* read 'f(double [][30])' or 'f(double [])' */
          G__disp_mask = 1000;
          fgetpos(G__ifile.fp,&tmp_pos);
          tmp_line = G__ifile.line_number;
          c=G__fgetstream(name+len,"[=,)");
          fsetpos(G__ifile.fp,&tmp_pos);
          G__ifile.line_number = tmp_line;
          G__disp_mask = 0;
          if('['==c) {
            c=G__fgetstream(name+len,"=,)");
            pointlevel = 0;
            break;
          }
          else {
            /* G__fignorestream("],)") already called above <<< */
            name[0] = 0;
          }
        }
        else {
          c=G__fignorestream("],)");
        }
      case '*':  
        ++pointlevel; 
        c=G__fgetspace();
        break;
      case '(': /* func(type (*)(type,...)) */
        if((-1!=typenum && -1!=G__newtype.parent_tagnum[typenum]) ||
           (-1!=tagnum  && -1!=G__struct.parent_tagnum[tagnum])) {
          char *px = strrchr(paraname,' ');
          if(px) ++px;
          else px = paraname;
          strcpy(px,G__type2string(0,tagnum,typenum,0,0));
        }
        ip=strlen(paraname);
        if(reftype) paraname[ip++]='&';
        reftype=0;
        paraname[ip++]=' ';
        for(itemp=0;itemp<pointlevel;itemp++) paraname[ip++]='*';
        pointlevel=0;
        paraname[ip++]='(';
        c = G__fgetstream(paraname+ip,"*)");
        if('*'==c) {
          int ixx;
          paraname[ip++]=c;
          c = G__fgetstream(name,")");
          ixx=0;
          while('*'==name[ixx]) {
            paraname[ip++]='*';
            ++ixx;
          }
          if(ixx) {
            int ixxx=0;
            while(name[ixx]) name[ixxx++] = name[ixx++];
            name[ixxx] = 0;
          }
        }
        if(')'==c) paraname[ip++]=')';
        c = G__fdumpstream(paraname+ip,",)=");
        ip=strlen(paraname);
        paraname[ip++]='\0';
#ifndef G__OLDIMPLEMENTATION2191
        typenum = G__search_typename(paraname,'1',-1,0);
        type='1';
#else
        typenum = G__search_typename(paraname,'Q',-1,0);
        type='Q';
#endif
        tagnum = -1;
        break;
      case '=':
        isdefault=1;
        c=G__fgetstream_template(paraname,",)");
        break;
      case '.':
        ifunc->ansi[func_now] = 2;
        c=G__fignorestream(",)");
        break;
      case EOF:
        return(1);
      default:
        if(isspace(c)) c=G__fgetspace();
        else {
          if(strcmp(name,"long")==0 && strcmp(paraname,"long")==0) {
            type = 'n';
            tagnum = -1;
            typenum = -1;
          }
          name[0] = c;
          c=G__fgetstream(name+1,"[=,)& \t");
          if(strcmp(name,"const")==0) {
            ifunc->para_isconst[func_now][iin]|=G__PCONSTVAR;
            name[0]=0;
          }
          if(strcmp(name,"const*")==0) {
            ifunc->para_isconst[func_now][iin]|=G__CONSTVAR;
            ++pointlevel; 
            name[0]=0;
          }
          else {
            while('['==c || ']'==c) {
              if('['==c) {
                ++pointlevel;
                ++arydim;
                if(G__NOLINK>G__globalcomp && 2==arydim) {
                  int len=strlen(name);
                  if(']'==name[0]) len=0;
                  strcpy(name+len,"[]");
                  pointlevel-=2;
                  len=strlen(name);
                  fseek(G__ifile.fp,-1,SEEK_CUR);
                  if(G__dispsource) G__disp_mask=1;
                  c=G__fgetstream(name+len,"=,)");
                  break;
                }
              }
              c=G__fignorestream("[=,)");
            }
            if('='==c) {
              c=G__fgetstream_template(paraname,",)");
              isdefault=1;
            }
          }
        }
        break;
      }
    }
    ifunc->para_p_tagtable[func_now][iin] = tagnum;
    ifunc->para_p_typetable[func_now][iin] = typenum;
    if(isdefault) {
      int store_def_tagnum = G__def_tagnum;
      int store_tagdefining = G__tagdefining;
      int store_prerun=G__prerun;
      int store_decl=G__decl;
      ifunc->para_def[func_now][iin] = 
        (char*)malloc(strlen(paraname)+1);
      strcpy(ifunc->para_def[func_now][iin],paraname);
      ifunc->para_default[func_now][iin] = (G__value*)malloc(sizeof(G__value));
      store_var_type=G__var_type;
      G__var_type='p';
      if(-1!=G__def_tagnum) {
        store_tagnum_default = G__tagnum;
        store_def_struct_member_default=G__def_struct_member;
        store_exec_memberfunc=G__exec_memberfunc;
        G__tagnum = G__def_tagnum;
        G__exec_memberfunc=1;
        G__def_struct_member=0;
      }
      if('('==paraname[0]) {
        int paranamelen = strlen(paraname);
        if(paranamelen>5 && strcmp(")()",paraname+paranamelen-3)==0 &&
           strchr(paraname,'<')
           && '*'!=paraname[paranamelen-4]
           ) {
          int ix;
          for(ix=1;ix<paranamelen-3;ix++) paraname[ix-1] = paraname[ix];
          strcpy(paraname+ix-1,"()");
        }
      }
      if(G__NOLINK==G__globalcomp) {
        G__prerun=0;
        G__decl=1;
      }
      {
        G__value *tmpx;
        struct G__ifunc_table *store_pifunc = G__p_ifunc;
        G__p_ifunc = &G__ifunc;
        if((G__CPPLINK==G__globalcomp || R__CPPLINK==G__globalcomp) && G__decl && G__prerun)
          G__noerr_defined = 1;
        *ifunc->para_default[func_now][iin] = G__getexpr(paraname);
        if((G__CPPLINK==G__globalcomp || R__CPPLINK==G__globalcomp) && G__decl && G__prerun)
          G__noerr_defined = 0;
        tmpx = ifunc->para_default[func_now][iin];
        if(reftype && (toupper (tmpx->type)!=toupper(type) ||
                       tmpx->tagnum!=tagnum) && 
           0==pointlevel) {
          char tmpy[G__ONELINE];
          sprintf(tmpy,"%s(%s)"
                  ,G__type2string(type,tagnum,-1,0,0),paraname);
          *tmpx = G__getexpr(tmpy);
          if('u'==tmpx->type) tmpx->ref = tmpx->obj.i;
        }
        G__p_ifunc = store_pifunc;
      }
      G__prerun=store_prerun;
      G__decl=store_decl;
      G__def_tagnum = store_def_tagnum;
      G__tagdefining = store_tagdefining;
      if(-1!=G__def_tagnum) {
        G__tagnum = store_tagnum_default;
        G__exec_memberfunc=store_exec_memberfunc;
        G__def_struct_member=store_def_struct_member_default;
      }
      G__var_type=store_var_type;
    }
    else { /* !isdefault */
      ifunc->para_default[func_now][iin] = (G__value*)NULL;
      ifunc->para_def[func_now][iin] = (char*)NULL;
    }
    if(reftype) {
      if(isupper(type)&&pointlevel) pointlevel++;
      if(-1!=typenum && G__newtype.reftype[typenum]>=G__PARAP2P) {
        pointlevel+=G__newtype.reftype[typenum]-G__PARAP2P+2;
        type=tolower(type);
      }
      switch(pointlevel) {
      case 0:
        ifunc->para_type[func_now][iin] = type ;
        ifunc->para_reftype[func_now][iin] = G__PARAREFERENCE ;
        break;
      case 1:
        ifunc->para_type[func_now][iin] = toupper(type) ;
        ifunc->para_reftype[func_now][iin] = G__PARAREFERENCE ;
        break;
      default:
        ifunc->para_type[func_now][iin] = toupper(type) ;
        ifunc->para_reftype[func_now][iin] = pointlevel-2 + G__PARAREFP2P ;
        break;
      }
    }
    else {
      if(isupper(type)&&pointlevel) pointlevel++;
      if(-1!=typenum && G__newtype.reftype[typenum]>=G__PARAP2P) {
        pointlevel+=G__newtype.reftype[typenum]-G__PARAP2P+2;
        type=tolower(type);
      }
      switch(pointlevel) {
      case 0:
        ifunc->para_type[func_now][iin] = type ;
        ifunc->para_reftype[func_now][iin] = G__PARANORMAL ;
        break;
      case 1:
        ifunc->para_type[func_now][iin] = toupper(type) ;
        ifunc->para_reftype[func_now][iin] = G__PARANORMAL ;
        break;
      default:
        ifunc->para_type[func_now][iin] = toupper(type) ;
        ifunc->para_reftype[func_now][iin] = pointlevel-2 + G__PARAP2P ;
      }
    }

    /* paranemter name omitted */
    if(name[0]) {
      ifunc->para_name[func_now][iin] = (char*)malloc(strlen(name)+1);
      strcpy(ifunc->para_name[func_now][iin],name);
    }
    else {
      ifunc->para_name[func_now][iin] = (char*)NULL;
    }
    ++iin;
  } /* while(')'!=c) */

  ifunc->para_nu[func_now]=iin;
  return(0);
}

/***********************************************************************
* int G__matchpointlevel
***********************************************************************/
int G__matchpointlevel(int param_reftype,int formal_reftype)
{
  switch(param_reftype) {
  case G__PARANORMAL:
  case G__PARAREFERENCE:
    if(G__PARANORMAL==formal_reftype||G__PARAREFERENCE==formal_reftype) 
      return(1);
    else
      return(0);
  default:
    return(formal_reftype==param_reftype);
  }
}

/***********************************************************************
* int G__param_match()
***********************************************************************/
int G__param_match(char formal_type,int formal_tagnum
                   ,G__value *default_parameter
                   ,char param_type,int param_tagnum
                   ,G__value *param
                   ,char *parameter
                   ,int funcmatch
                   ,int rewind_arg
                   ,int formal_reftype
                   ,int formal_isconst
                   )
{
  int match;
  static int recursive=0;
  long store_struct_offset; /* used to be int */
  int store_tagnum;
  char conv[G__ONELINE],arg1[G__ONELINE];
  int baseoffset;
  G__value reg;
  int store_oprovld;
  int rewindflag=0;

  if(default_parameter && param_type == '\0') {
    return(2);
  }

  if(funcmatch>=G__EXACT) {
    if(param_type==formal_type && 0==recursive){

      if(tolower(param_type)=='u') {
        /* If struct,class,union, check tagnum */
        if(formal_tagnum != param_tagnum) { /* unmatch */
          match=0; 
        }
        else {  /* match */
          match=1;
        }
      }
      else { /* match */
        match=1;
      }

    }
    else if('I'==param_type&&'U'==formal_type&&param_tagnum==formal_tagnum&&
            -1!=formal_tagnum&&'e'==G__struct.type[formal_tagnum]) {
      match=1;
    }
    else {  /* unmatch */
      match=0;
    }
  }

  if(match==0&&funcmatch>=G__PROMOTION) {
    switch(formal_type) {
    case 'd':
    case 'f':
      switch(param_type) {
      case 'b':
      case 'c':
      case 'r':
      case 's':
      case 'h':
      case 'i':
      case 'k':
      case 'l':
#define G__OLDIMPLEMENTATION1165
      case 'd':
      case 'f':
        match=1;
        break;
      default:
        match=0;
        break;
      }
      break;
    case 'l':
      switch(param_type) {
      case 'b':
      case 'c':
      case 'r':
      case 's':
      /* case 'h': */
      case 'i':
      /* case 'k': */
      case 'l':
        match=1;
        break;
      default:
        match=0;
        break;
      }
      break;
    case 'i':
      switch(param_type) {
      case 'b':
      case 'c':
      case 'r':
      case 's':
      /* case 'h': */
      case 'i':
      /* case 'k': */
      /* case 'l': */
        match=1;
        break;
      case 'u':
        if('e'==G__struct.type[param_tagnum]) {
          if(param->ref) param->obj.i = *(long*)(param->ref);
          match=1;
          break;
        }
      default:
        match=0;
        break;
      }
      break;
    case 's':
      switch(param_type) {
      case 'b':
      case 'c':
      /* case 'r': */
      case 's':
      /* case 'h': */
      /* case 'i': */
      /* case 'k': */
      /* case 'l': */
        match=1;
        break;
      default:
        match=0;
        break;
      }
      break;
    case 'k':
      switch(param_type) {
      case 'b':
      /* case 'c': */
      case 'r':
      /* case 's': */
      case 'h':
      /* case 'i': */
      case 'k':
      /* case 'l': */
        match=1;
        break;
      default:
        match=0;
        break;
      }
      break;
    case 'h':
      switch(param_type) {
      case 'b':
      /* case 'c': */
      case 'r':
      /* case 's': */
      case 'h':
      /* case 'i': */
      /* case 'k': */
      /* case 'l': */
        match=1;
        break;
      default:
        match=0;
        break;
      }
      break;
    case 'r':
      switch(param_type) {
      case 'b':
      /* case 'c': */
      case 'r':
      /* case 's': */
      /* case 'h': */
      /* case 'i': */
      /* case 'k': */
      /* case 'l': */
        match=1;
        break;
      default:
        match=0;
        break;
      }
      break;
    case 'u':
      if(0<=formal_tagnum && 'e'==G__struct.type[formal_tagnum]) {
        switch(param_type) {
        case 'i':
        case 's':
        case 'l':
        case 'c':
        case 'h':
        case 'r':
        case 'k':
        case 'b':
          match=1;
          break;
        default:
          match=0;
          break;
        }
      }
      else {
        match=0;
      }
      break;
    default:
      match=0;
      break;
    }
  }

  if(match==0&&funcmatch>=G__STDCONV) {
    switch(formal_type) {
    case 'b':
    case 'c':
    case 'r':
    case 's':
    case 'h':
    case 'i':
    case 'k':
    case 'l':
    case 'd':
    case 'f':
      switch(param_type) {
      case 'b':
      case 'c':
      case 'r':
      case 's':
      case 'h':
      case 'i':
      case 'k':
      case 'l':
      case 'd':
      case 'f':
        match=1;
        break;
      default:
        match=0;
        break;
      }
      break;
    case 'C':
      switch(param_type) {
      case 'i':
      case 'l':
      if(0==param->obj.i) match=1;
      else match=0;
      break;
      case 'Y':
        match=1;
        break;
      default:
        match=0;
        break;
      }
      break;
    case 'Y':
      if(isupper(param_type)||0==param->obj.i) {
        match=1;
      }
      else {
        match=0;
      }
      break;
#if !defined(G__OLDIMPLEMENTATION2191)
    case '1': /* questionable */
      if('1'==param_type||'C'==param_type ||'Y'==param_type) match=1;
      else  match=0;
      break;
#else
    case 'Q': /* questionable */
      if('Q'==param_type||'C'==param_type
         ||'Y'==param_type
         ) match=1;
      else  match=0;
      break;
#endif
#ifdef G__WORKAROUND000209_1
      /* reference type conversin should not be handled in this way. 
       * difference was found from g++ when activating this part. */
      /* Added condition for formal_reftype and recursive, then things
       * are working 1999/12/5 */
    case 'u':
      if(G__PARAREFERENCE==formal_reftype && recursive) {
        switch(param_type) {
        case 'u':
          /* reference to derived class can be converted to reference to base 
           * class. add offset, modify char *parameter and G__value *param */
          if(-1 != (baseoffset=G__ispublicbase(formal_tagnum,param_tagnum
                                               ,param->obj.i))) {
            param->tagnum = formal_tagnum;
            param->obj.i += baseoffset;
            param->ref += baseoffset;
            match=1;
          }
          else {
            match=0;
          }
          break;
        }
      }
      break;
#endif
    case 'U':
      switch(param_type) {
      case 'U':
        /* Pointer to derived class can be converted to
         * pointer to base class.
         * add offset, modify char *parameter and 
         * G__value *param
         */
#ifdef G__VIRTUALBASE
        if(-1 != (baseoffset=G__ispublicbase(formal_tagnum,param_tagnum
                                             ,param->obj.i))) {
#else
        if(-1 != (baseoffset=G__ispublicbase(formal_tagnum,param_tagnum))) {
#endif
          param->tagnum = formal_tagnum;
          param->obj.i += baseoffset;
          param->ref = 0;
          match=1;
        }
        else {
          match=0;
        }
        break;
      case 'Y':
#ifndef G__OLDIMPLEMENTATION2191
      case '1': /* questionable */
#else
      case 'Q': /* questionable */
#endif
        match=1;
        break;
      case 'i':
      case 0:
        if(0==param->obj.i) match=1;
        else                match=0;
        break;
      default:
        match=0;
        break;
      }
      break;
    default:
      /* questionable */
#ifndef G__OLDIMPLEMENTATION2191
      if((param_type=='Y'||param_type=='1'||0==param->obj.i)&&
         (isupper(formal_type) || 'a'==formal_type)) {
        match=1;
      }
      else {
        match=0;
      }
      if((param_type=='Y'||param_type=='Q'||0==param->obj.i)&&
         (isupper(formal_type)
#ifndef G__OLDIMPLEMENTATION1289
         || 'a'==formal_type
#endif
          )) {
        match=1;
      }
      else {
        match=0;
      }
#endif
      break;
    }
  }

  if(match==0&&funcmatch>=G__USERCONV) {
    if(formal_type=='u' && 0==recursive) {
      /* create temp object buffer */
      if(G__CPPLINK!=G__struct.iscpplink[formal_tagnum]) {
        G__alloc_tempobject(formal_tagnum,-1);
#ifdef G__ASM
        if(G__asm_noverflow) {
#ifdef G__ASM_DBG
          if(G__asm_dbg) {
            G__fprinterr(G__serr,"%3x: ALLOCTEMP %s %d\n"
                    ,G__asm_cp,G__struct.name[formal_tagnum],formal_tagnum);
            G__fprinterr(G__serr,"%3x: SETTEMP\n",G__asm_cp+2);
          }
#endif
          G__asm_inst[G__asm_cp] = G__ALLOCTEMP;
          G__asm_inst[G__asm_cp+1] = formal_tagnum;
          G__inc_cp_asm(2,0);
          G__asm_inst[G__asm_cp] = G__SETTEMP;
          G__inc_cp_asm(1,0);
        }
#endif
      }

      /* try finding constructor */
      if('u'==param_type) {
        if(param->obj.i<0) 
          sprintf(arg1,"(%s)(%ld)"
                  ,G__fulltagname(param_tagnum,1),param->obj.i);
        else
          sprintf(arg1,"(%s)%ld",G__fulltagname(param_tagnum,1),param->obj.i);
      }
      else {
        G__valuemonitor(*param,arg1);
      }
      sprintf(conv,"%s(%s)",G__struct.name[formal_tagnum],arg1);

      if(G__dispsource) {
        G__fprinterr(G__serr, "!!!Trying implicit conversion %s,%d\n"
                ,conv,G__templevel);
      }

      store_struct_offset = G__store_struct_offset;
      G__store_struct_offset = G__p_tempbuf->obj.obj.i;

      store_tagnum = G__tagnum;
      G__tagnum = formal_tagnum;

      /* avoid duplicated argument evaluation in p-code stack */
      store_oprovld = G__oprovld;
      G__oprovld=1;

#ifdef G__ASM
      if(G__asm_noverflow && rewind_arg) {
        rewindflag=1;
#ifdef G__ASM_DBG
        if(G__asm_dbg) G__fprinterr(G__serr,"%3x: REWINDSTACK %d\n"
                               ,G__asm_cp,rewind_arg);
#endif
        G__asm_inst[G__asm_cp] = G__REWINDSTACK;
        G__asm_inst[G__asm_cp+1] = rewind_arg;
        G__inc_cp_asm(2,0);
      }
#endif

      ++recursive;
      if(G__CPPLINK==G__struct.iscpplink[formal_tagnum]) {
        /* in case of pre-compiled class */
        reg=G__getfunction(conv,&match,G__TRYIMPLICITCONSTRUCTOR);
        if(match) {
          G__store_tempobject(reg);
#ifdef G__ASM
          if(G__asm_noverflow) {
#ifdef G__ASM_DBG
            if(G__asm_dbg) G__fprinterr(G__serr,"%3x: STORETEMP\n",G__asm_cp);
#endif
            G__asm_inst[G__asm_cp]=G__STORETEMP;
            G__inc_cp_asm(1,0);
          }
#endif
        }
        else {
          sprintf(conv,"operator %s()",G__fulltagname(formal_tagnum,1));
          G__store_struct_offset = param->obj.i;
          G__tagnum = param->tagnum;
          if(-1!=G__tagnum) reg=G__getfunction(conv,&match,G__TRYMEMFUNC);
          if(!match) G__store_tempobject(G__null);
        }
      }
      else {
        /* in case of interpreted class */
        G__getfunction(conv,&match,G__TRYIMPLICITCONSTRUCTOR);
        if(match) {
          if(G__asm_noverflow) {
#ifdef G__ASM_DBG
            if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPTEMP %d\n"
                                        ,G__asm_cp,formal_tagnum);
#endif
            G__asm_inst[G__asm_cp] = G__POPTEMP;
            G__asm_inst[G__asm_cp+1] = formal_tagnum;
            G__inc_cp_asm(2,0);
          }
        }
        else {
          if(G__asm_noverflow) G__inc_cp_asm(-3,0);
          sprintf(conv,"operator %s()",G__fulltagname(formal_tagnum,1));
          G__store_struct_offset = param->obj.i;
          G__tagnum = param->tagnum;
          reg=G__getfunction(conv,&match,G__TRYMEMFUNC);
          if(!match) {
            if(G__asm_noverflow) {
              if(rewindflag) {
                G__asm_inst[G__asm_cp-2]=G__REWINDSTACK; 
                G__asm_inst[G__asm_cp-1] = rewind_arg;
              }
#ifdef G__ASM_DBG
              if(G__asm_dbg) 
                G__fprinterr(G__serr,"ALLOCTEMP,SETTEMP Cancelled %x\n",G__asm_cp);
#endif
            }
          }
        }
      }
      --recursive;

      G__oprovld=store_oprovld;

      G__tagnum = store_tagnum;
      G__store_struct_offset = store_struct_offset;

      /* if no constructor, try converting to base class */


      if(match==0) {
        if('u'==param_type &&
#ifdef G__VIRTUALBASE
           -1 != (baseoffset=G__ispublicbase(formal_tagnum,param_tagnum
                                             ,param->obj.i))) {
#else
           -1 != (baseoffset=G__ispublicbase(formal_tagnum,param_tagnum))) {
#endif
          if(G__dispsource) {
            G__fprinterr(G__serr, "!!!Implicit conversion from %s to base %s\n"
                    ,G__struct.name[param_tagnum]
                    ,G__struct.name[formal_tagnum]);
          }
          param->typenum = -1;
          param->tagnum = formal_tagnum;
          param->obj.i += baseoffset;
          param->ref += baseoffset;
#ifdef G__ASM
          if(G__asm_noverflow) {
#ifdef G__ASM_DBG
            if(G__asm_dbg) G__fprinterr(G__serr,"%3x: BASECONV %d %d\n"
                                   ,G__asm_cp,formal_tagnum,baseoffset);
#endif
            G__asm_inst[G__asm_cp] = G__BASECONV;
            G__asm_inst[G__asm_cp+1] = formal_tagnum;
            G__asm_inst[G__asm_cp+2] = baseoffset;
            G__inc_cp_asm(3,0);
            if(rewind_arg) {
              rewindflag=1;
#ifdef G__ASM_DBG
              if(G__asm_dbg) G__fprinterr(G__serr,"%3x: REWINDSTACK %d\n"
                                     ,G__asm_cp,-rewind_arg);
#endif
              G__asm_inst[G__asm_cp] = G__REWINDSTACK;
              G__asm_inst[G__asm_cp+1] = -rewind_arg;
              G__inc_cp_asm(2,0);
            }
#endif
            if(param->obj.i<0) 
              sprintf(parameter,"(%s)(%ld)",G__struct.name[formal_tagnum]
                      ,param->obj.i);
            else
              sprintf(parameter,"(%s)%ld",G__struct.name[formal_tagnum]
                      ,param->obj.i);
          }
          match=1;
          G__pop_tempobject();
        }
        else { /* all conversion failed */
          if(G__dispsource) {
            G__fprinterr(G__serr,
                    "!!!Implicit conversion %s,%d tried, but failed\n"
                    ,conv,G__templevel);
          }
          G__pop_tempobject();
#ifdef G__ASM
          if(rewindflag) {
#ifdef G__ASM_DBG
            if(G__asm_dbg) G__fprinterr(G__serr,"REWINDSTACK cancelled\n");
#endif
            G__inc_cp_asm(-2,0);
          }
        }

#else /* ON181 */

        /* all conversion failed */
        if(G__dispsource) {
          G__fprinterr(G__serr,
                  "!!!Implicit conversion %s,%d tried, but failed\n"
                  ,conv,G__templevel);
        }
        G__pop_tempobject();
#ifdef G__ASM
        if(rewindflag) {
#ifdef G__ASM_DBG
          if(G__asm_dbg) G__fprinterr(G__serr,"REWINDSTACK cancelled\n");
#endif
          G__inc_cp_asm(-2,0);
        }
#endif

#endif /* ON181 */
      }
      else { /* match==1, conversion successful */
        if(G__dispsource) {
          if(G__p_tempbuf->obj.obj.i<0) 
            G__fprinterr(G__serr,
              "!!!Create temp object (%s)(%ld),%d for implicit conversion\n"
                    ,conv ,G__p_tempbuf->obj.obj.i ,G__templevel);
          else
            G__fprinterr(G__serr,
                  "!!!Create temp object (%s)%ld,%d for implicit conversion\n"
                    ,conv ,G__p_tempbuf->obj.obj.i ,G__templevel);
        }
#ifdef G__ASM
        if(G__asm_noverflow && rewind_arg) {
          rewindflag=1;
#ifdef G__ASM_DBG
          if(G__asm_dbg) G__fprinterr(G__serr,"%3x: REWINDSTACK %d\n"
                                 ,G__asm_cp,-rewind_arg);
#endif
          G__asm_inst[G__asm_cp] = G__REWINDSTACK;
          G__asm_inst[G__asm_cp+1] = -rewind_arg;
          G__inc_cp_asm(2,0);
        }
#endif
        *param = G__p_tempbuf->obj;
        sprintf(parameter,"(%s)%ld" ,G__struct.name[formal_tagnum]
                ,G__p_tempbuf->obj.obj.i);
      } /* end of if(match==0) */

    }
    else if(-1!=param->tagnum) {
      long store_struct_offset=G__store_struct_offset;
      int store_tagnum=G__tagnum;
      sprintf(conv,"operator %s()"
              ,G__type2string(formal_type,formal_tagnum,-1,0,0));
      G__store_struct_offset = param->obj.i;
      G__tagnum=param->tagnum;
#ifdef G__ASM
      if(G__asm_noverflow) {
        if(rewind_arg) {
          rewindflag=1;
#ifdef G__ASM_DBG
          if(G__asm_dbg) G__fprinterr(G__serr,"%3x: REWINDSTACK %d\n"
                                 ,G__asm_cp,rewind_arg);
#endif
          G__asm_inst[G__asm_cp] = G__REWINDSTACK;
          G__asm_inst[G__asm_cp+1] = rewind_arg;
          G__inc_cp_asm(2,0);
        }
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
#endif
      reg=G__getfunction(conv,&match,G__TRYMEMFUNC);
      if(!match
         && 0!=formal_isconst
         ) {
        sprintf(conv,"operator const %s()"
                ,G__type2string(formal_type,formal_tagnum,-1,0,0));
        G__store_struct_offset = param->obj.i;
        G__tagnum=param->tagnum;
        reg=G__getfunction(conv,&match,G__TRYMEMFUNC);
      }
      G__tagnum=store_tagnum;
      G__store_struct_offset=store_struct_offset;
#ifdef G__ASM
      if(G__asm_noverflow) {
        if(rewind_arg) {
          rewindflag=1;
#ifdef G__ASM_DBG
          if(G__asm_dbg) G__fprinterr(G__serr,"%3x: REWINDSTACK %d\n"
                                  ,G__asm_cp,-rewind_arg);
#endif
          G__asm_inst[G__asm_cp] = G__REWINDSTACK;
          G__asm_inst[G__asm_cp+1] = -rewind_arg;
          G__inc_cp_asm(2,0);
        }
        G__asm_inst[G__asm_cp] = G__POPSTROS;
        G__inc_cp_asm(1,0);
#ifdef G__ASM_DBG
        if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp-1);
#endif
      }
#endif
      /* fixing 'cout<<x' fundamental conversion opr with opr overloading 
      * Not 100% sure if this is OK. */
      if(match) *param = reg;
      else if(rewindflag) {
#ifdef G__ASM_DBG
        if(G__asm_dbg) G__fprinterr(G__serr,"REWINDSTACK~ cancelled\n");
#endif
        G__inc_cp_asm(-7,0);
      }
      else {
#ifdef G__ASM_DBG
        if(G__asm_dbg) G__fprinterr(G__serr,"PUSHSTROS~ cancelled\n");
#endif
        G__inc_cp_asm(-3,0);
      }
    }
    else {
      match=0;
/* #ifdef G__DEBUG */
      if(recursive&&G__dispsource) {
        G__valuemonitor(*param,arg1);
        G__fprinterr(G__serr,"!!!Recursive implicit conversion %s(%s) rejected\n"
                ,G__struct.name[formal_tagnum],arg1);
      }
/* #endif */
    }
  }

  if(match && isupper(param_type) && isupper(formal_type) && 
     'Y'!=param_type && 'Y'!=formal_type
#if !defined(G__OLDIMPLEMENTATION2191)
     && '1'!=param_type 
#else
     && 'Q'!=param_type 
#endif
     ) {
    match=G__matchpointlevel(param->obj.reftype.reftype,formal_reftype);
  }

  return(match);
}

/***********************************************************************
* macro 
**********************************************************************/
#define G__NOMATCH        0xffffffff
#define G__EXACTMATCH     0x00000000
#define G__PROMOTIONMATCH 0x00000100
#define G__STDCONVMATCH   0x00010000
#define G__USRCONVMATCH   0x01000000
#define G__CVCONVMATCH    0x00000001
#define G__BASECONVMATCH  0x00000001
#define G__C2P2FCONVMATCH 0x00000001
#define G__I02PCONVMATCH  0x00000002
#define G__V2P2FCONVMATCH 0x00000002
#define G__TOVOIDPMATCH   0x00000003

/***********************************************************************
 * function overloading resolution
 * G__funclist is defined in common.h
 **********************************************************************/
struct G__funclist* G__funclist_add(G__funclist *last
                                    ,G__ifunc_table *ifunc
                                    ,int ifn,int rate)
{
  struct G__funclist *latest = 
    (struct G__funclist*)malloc(sizeof(struct G__funclist));
  latest->prev = last;
  latest->ifunc = ifunc;
  latest->ifn = ifn;
  latest->rate = rate;
  return(latest);
}

void G__funclist_delete(G__funclist *body)
{
  if(body) {
    if(body->prev) G__funclist_delete(body->prev);
    free((void*)body);
  }
}

/***********************************************************************
* G__rate_inheritance()
***********************************************************************/
unsigned int G__rate_inheritance(int basetagnum,int derivedtagnum)
{
  struct G__inheritance *derived;
  int i,n;

  if(0>derivedtagnum||0>basetagnum) return(G__NOMATCH);
  if(basetagnum==derivedtagnum) return(G__EXACTMATCH);
  derived = G__struct.baseclass[derivedtagnum];
  n = derived->basen;

  for(i=0;i<n;i++) {
    if(basetagnum == derived->basetagnum[i]) {
      if(derived->baseaccess[i]==G__PUBLIC ||
         (G__exec_memberfunc && G__tagnum==derivedtagnum &&
          G__GRANDPRIVATE!=derived->baseaccess[i])) {
        if(G__ISDIRECTINHERIT&derived->property[i]) {
          return(G__BASECONVMATCH);
        }
        else {
          int distance = 1;
          int ii=i; /* i is not 0, because !G__ISDIRECTINHERIT */
          struct G__inheritance *derived2 = derived;
          int derivedtagnum2 = derivedtagnum;
          while(0==(derived2->property[ii]&G__ISDIRECTINHERIT)) {
            ++distance;
            while(ii && 0==(derived2->property[--ii]&G__ISDIRECTINHERIT));
            derivedtagnum2 = derived2->basetagnum[ii];
            derived2 = G__struct.baseclass[derivedtagnum2];
            for(ii=0;ii<derived2->basen;ii++) {
              if(derived2->basetagnum[ii]==basetagnum) break;
            }
            if(ii==derived2->basen) return(G__NOMATCH);
          }
          return(distance*G__BASECONVMATCH);
        }
      }
    }
  }
  return(G__NOMATCH);
}

#ifndef G__OLDIMPLEMENTATION1959
/***********************************************************************
* int G__promotiongrade()
**********************************************************************/
#define G__promotiongrade(f,p) G__PROMOTIONMATCH*(G__igrd(f)-G__igrd(p))

static int G__igrd(int formal_type)
{
  switch(formal_type) {
  case 'g': 
    return(1);
  case 'b':
  case 'c':
    return(2);
  case 'r':
  case 's':
    return(3);
  case 'h':
  case 'i':
    return(4);
  case 'k':
  case 'l':
    return(5);
  case 'n':
  case 'm':
    return(6);
  }
  return(0);
}
#endif

#ifndef __CINT__
struct G__ifunc_table* G__overload_match G__P((char* funcname,struct G__param *libp,int hash,struct G__ifunc_table *p_ifunc,int memfunc_flag,int access,int *pifn,int recursive,int doconvert)) ;
#endif

/***********************************************************************
* int G__rate_parameter_match(libp,ifunc,ifn,i)
**********************************************************************/
void G__rate_parameter_match(G__param *libp,G__ifunc_table *p_ifunc
                             ,int ifn,G__funclist *funclist,int recursive)
{
#ifdef G__DEBUG
  int i=0xa3a3a3a3;
#else
  int i;
#endif
  char param_type,formal_type;
  int param_tagnum,formal_tagnum;
  int param_reftype,formal_reftype;
#ifdef G__DEBUG
  int param_isconst=0xa3a3a3a3,formal_isconst=0xa5a5a5a5;
#else
  int param_isconst=0,formal_isconst=0;
#endif
  funclist->rate = 0;
  for(i=0;i<libp->paran;i++) {
    param_type = libp->para[i].type;
    formal_type = p_ifunc->para_type[ifn][i];
    param_tagnum = libp->para[i].tagnum;
    formal_tagnum = p_ifunc->para_p_tagtable[ifn][i];
    param_reftype = libp->para[i].obj.reftype.reftype;
    formal_reftype = p_ifunc->para_reftype[ifn][i];
    param_isconst = libp->para[i].isconst;
    formal_isconst = p_ifunc->para_isconst[ifn][i];
    funclist->p_rate[i] = G__NOMATCH;

    /* exact match */
    if(param_type==formal_type){
      if(tolower(param_type)=='u') {
        /* If struct,class,union, check tagnum */
        if(formal_tagnum == param_tagnum) { /* match */
          funclist->p_rate[i] = G__EXACTMATCH;
        }
      }
      else if(isupper(param_type)) {
        if(param_reftype==formal_reftype ||
           (param_reftype<=G__PARAREFERENCE &&
            formal_reftype<=G__PARAREFERENCE)) {
          funclist->p_rate[i] = G__EXACTMATCH;
        } else if ( (formal_reftype>G__PARAREF && formal_reftype==param_reftype+G__PARAREF)
           || (param_reftype>G__PARAREF && param_reftype==formal_reftype+G__PARAREF) ) {
          funclist->p_rate[i] = G__STDCONVMATCH;
        }
      }
      else if('i'==param_type && (formal_tagnum!=param_tagnum)) {
        funclist->p_rate[i] = G__PROMOTIONMATCH;
      }
      else { /* match */
        funclist->p_rate[i] = G__EXACTMATCH;
      }
    }
    else if(('I'==param_type||'U'==param_type)&&
            ('I'==formal_type||'U'==formal_type)&&
            param_tagnum==formal_tagnum&&
            -1!=formal_tagnum&&'e'==G__struct.type[formal_tagnum]) {
      funclist->p_rate[i] = G__EXACTMATCH;
    }
    else if(isupper(formal_type)&&
            ('i'==param_type || 'l'==param_type)
            &&0==libp->para[i].obj.i) {
      funclist->p_rate[i] = G__STDCONVMATCH + G__I02PCONVMATCH;
    }
    
    /* promotion */
    if(G__NOMATCH==funclist->p_rate[i]) {
      switch(formal_type) {
      case 'd': 
      case 'f':
        switch(param_type) {
        case 'f':
          funclist->p_rate[i] = G__PROMOTIONMATCH;
          break;
        default:
          break;
        }
        break;
      case 'l':
        switch(param_type) {
        case 'b':
        case 'c':
        case 'r':
        case 's':
          /* case 'h': */
        case 'i':
          /* case 'k': */
        case 'l':
        case 'g':
#ifndef G__OLDIMPLEMENTATION1959
          funclist->p_rate[i] = G__promotiongrade(formal_type,param_type);
#else
          funclist->p_rate[i] = G__PROMOTIONMATCH;
#endif
          break;
        default:
        break;
        }
        break;
      case 'i':
        switch(param_type) {
        case 'b':
        case 'c':
        case 'r':
        case 's':
          /* case 'h': */
        case 'i':
          /* case 'k': */
          /* case 'l': */
        case 'g':
#ifndef G__OLDIMPLEMENTATION1959
          funclist->p_rate[i] = G__promotiongrade(formal_type,param_type);
#else
          funclist->p_rate[i] = G__PROMOTIONMATCH;
#endif
          break;
        case 'u':
          if('e'==G__struct.type[param_tagnum]) {
            funclist->p_rate[i] = G__PROMOTIONMATCH;
          }
          break;
        default:
          break;
        }
        break;
      case 's':
        switch(param_type) {
        case 'b':
        case 'c':
          /* case 'r': */
        case 's':
          /* case 'h': */
          /* case 'i': */
          /* case 'k': */
          /* case 'l': */
        case 'g':
#ifndef G__OLDIMPLEMENTATION1959
          funclist->p_rate[i] = G__promotiongrade(formal_type,param_type);
#else
          funclist->p_rate[i] = G__PROMOTIONMATCH;
#endif
          break;
        default:
          break;
        }
        break;
      case 'k':
        switch(param_type) {
        case 'b':
          /* case 'c': */
        case 'r':
          /* case 's': */
        case 'h':
          /* case 'i': */
        case 'k':
          /* case 'l': */
        case 'g':
#ifndef G__OLDIMPLEMENTATION1959
          funclist->p_rate[i] = G__promotiongrade(formal_type,param_type);
#else
          funclist->p_rate[i] = G__PROMOTIONMATCH;
#endif
          break;
        default:
          break;
        }
        break;
      case 'h':
        switch(param_type) {
        case 'b':
          /* case 'c': */
        case 'r':
          /* case 's': */
        case 'h':
          /* case 'i': */
          /* case 'k': */
          /* case 'l': */
        case 'g':
#ifndef G__OLDIMPLEMENTATION1959
          funclist->p_rate[i] = G__promotiongrade(formal_type,param_type);
#else
          funclist->p_rate[i] = G__PROMOTIONMATCH;
#endif
          break;
        default:
          break;
        }
        break;
      case 'r':
        switch(param_type) {
        case 'b':
          /* case 'c': */
        case 'r':
          /* case 's': */
          /* case 'h': */
          /* case 'i': */
          /* case 'k': */
          /* case 'l': */
        case 'g':
#ifndef G__OLDIMPLEMENTATION1959
          funclist->p_rate[i] = G__promotiongrade(formal_type,param_type);
#else
          funclist->p_rate[i] = G__PROMOTIONMATCH;
#endif
          break;
        default:
          break;
        }
        break;
      case 'u':
        if(0<=formal_tagnum && 'e'==G__struct.type[formal_tagnum]) {
          switch(param_type) {
          case 'i':
          case 's':
          case 'l':
          case 'c':
          case 'h':
          case 'r':
          case 'k':
          case 'b':
            funclist->p_rate[i] = G__PROMOTIONMATCH;
            break;
          default:
            break;
          }
        }
        else {
        }
        break;
      case 'Y':
        if(isupper(param_type)||0==libp->para[i].obj.i
#ifndef G__OLDIMPLEMENTATION2191
           || '1'==param_type
#endif
           ) {
          funclist->p_rate[i] = G__PROMOTIONMATCH+G__TOVOIDPMATCH;
        }
        break;
      default:
        break;
      }
    }

    /* standard conversion */
    if(G__NOMATCH==funclist->p_rate[i]) {
      switch(formal_type) {
      case 'b':
      case 'c':
      case 'r':
      case 's':
      case 'h':
      case 'i':
      case 'k':
      case 'l':
      case 'g':
      case 'n':
      case 'm':
        switch(param_type) {
        case 'd':
        case 'f':
        case 'b':
        case 'c':
        case 'r':
        case 's':
        case 'h':
        case 'i':
        case 'k':
        case 'l':
        case 'g':
        case 'n':
        case 'm':
        case 'q':
          funclist->p_rate[i] = G__STDCONVMATCH;
          break;
        default:
          break;
        }
        break;
      case 'd':
      case 'f':
        switch(param_type) {
        case 'b':
        case 'c':
        case 'r':
        case 's':
        case 'h':
        case 'i':
        case 'k':
        case 'l':
        case 'd':
        case 'f':
        case 'g':
        case 'n':
        case 'm':
          funclist->p_rate[i] = G__STDCONVMATCH;
          break;
        default:
          break;
        }
        break;
      case 'C':
        switch(param_type) {
        case 'i':
        case 'l':
          if(0==libp->para[i].obj.i) 
            funclist->p_rate[i] = G__STDCONVMATCH + G__I02PCONVMATCH;
          break;
        case 'Y':
          if(G__PARANORMAL==param_reftype) {
            funclist->p_rate[i] = G__STDCONVMATCH;
          }
          break;
        default:
          break;
        }
        break;
      case 'Y':
        if(isupper(param_type)||0==libp->para[i].obj.i) {
          funclist->p_rate[i] = G__STDCONVMATCH;
        }
        break;
#ifndef G__OLDIMPLEMENTATION2191
      case '1': /* questionable */
#else
      case 'Q': /* questionable */
#endif
        if(
#ifndef G__OLDIMPLEMENTATION2191
           '1'==param_type
#else
           'Q'==param_type
#endif
           ) 
          funclist->p_rate[i] = G__STDCONVMATCH;
        else if('Y'==param_type) 
          funclist->p_rate[i] = G__STDCONVMATCH+G__V2P2FCONVMATCH;
        else if('C'==param_type) {
          if(
             p_ifunc->pentry[ifn]->size>=0
             ) 
            funclist->p_rate[i] = G__STDCONVMATCH-G__C2P2FCONVMATCH;
          else {
            funclist->p_rate[i] = G__STDCONVMATCH+G__C2P2FCONVMATCH;/*???*/
          }
        }
        break;
      case 'u':
        switch(param_type) {
        case 'u':
          /* reference to derived class can be converted to reference to base 
           * class. add offset, modify char *parameter and G__value *param */
          {
            unsigned int rate_inheritance = 
              G__rate_inheritance(formal_tagnum,param_tagnum);
            if(G__NOMATCH!=rate_inheritance) {
              funclist->p_rate[i] = G__STDCONVMATCH+rate_inheritance;
            }
          }
          break;
        }
        break;
      case 'U':
        switch(param_type) {
        case 'U':
          /* Pointer to derived class can be converted to
           * pointer to base class.
           * add offset, modify char *parameter and 
           * G__value *param
           */
          {
            unsigned int rate_inheritance = 
              G__rate_inheritance(formal_tagnum,param_tagnum);
            if(G__NOMATCH!=rate_inheritance) {
              funclist->p_rate[i] = G__STDCONVMATCH+rate_inheritance;
            }
          }
          break;
        case 'Y':
          if(G__PARANORMAL==param_reftype) {
            funclist->p_rate[i] = G__STDCONVMATCH;
          }
          break;
#ifndef G__OLDIMPLEMENTATION2191
        case '1': /* questionable */
#else
        case 'Q': /* questionable */
#endif
          funclist->p_rate[i] = G__STDCONVMATCH;
          break;
        case 'i':
        case 0:
          if(0==libp->para[0].obj.i) funclist->p_rate[i] = G__STDCONVMATCH;
          break;
        default:
          break;
        }
        break;
      default:
        /* questionable */
#ifndef G__OLDIMPLEMENTATION2191
        if((param_type=='Y'||param_type=='1')&&
           (isupper(formal_type) || 'a'==formal_type)) {
          funclist->p_rate[i] = G__STDCONVMATCH;
        }
#else
        if((param_type=='Y'||param_type=='Q'||0==libp->para[0].obj.i)&&
           (isupper(formal_type) || 'a'==formal_type)) {
          funclist->p_rate[i] = G__STDCONVMATCH;
        }
#endif
        break;
      }
    }

    /* user defined conversion */
    if(0==recursive && G__NOMATCH==funclist->p_rate[i]) {
      if(formal_type=='u') {
        struct G__ifunc_table *ifunc2;
        int ifn2;
        int hash2;
        char funcname2[G__ONELINE];
        struct G__param para;
        G__incsetup_memfunc(formal_tagnum);
        ifunc2 = G__struct.memfunc[formal_tagnum];
        para.paran = 1;
        para.para[0] = libp->para[i];
        strcpy(funcname2,G__struct.name[formal_tagnum]);
        G__hash(funcname2,hash2,ifn2);
        ifunc2 = G__overload_match(funcname2,&para,hash2,ifunc2
                                   ,G__TRYCONSTRUCTOR,G__PUBLIC,&ifn2,1
                                   ,1
                                   );
        if(ifunc2 && -1!=ifn2) 
          funclist->p_rate[i] = G__USRCONVMATCH;
      }
    }

    if(0==recursive && G__NOMATCH==funclist->p_rate[i]) {
      if(param_type=='u' && -1!=param_tagnum) {
        struct G__ifunc_table *ifunc2;
        int ifn2 = -1;
        int hash2;
        char funcname2[G__ONELINE];
        struct G__param para;
        G__incsetup_memfunc(param_tagnum);
        para.paran = 0;
        /* search for  operator type */
        sprintf(funcname2,"operator %s"
                ,G__type2string(formal_type,formal_tagnum,-1,0,0));
        G__hash(funcname2,hash2,ifn2);
        ifunc2 = G__struct.memfunc[param_tagnum];
        ifunc2 = G__overload_match(funcname2,&para,hash2,ifunc2
                                   ,G__TRYMEMFUNC,G__PUBLIC,&ifn2,1
                                   ,1
                                   );
        if(!ifunc2) {
          /* search for  operator const type */
          sprintf(funcname2,"operator %s"
                  ,G__type2string(formal_type,formal_tagnum,-1,0,1));
          G__hash(funcname2,hash2,ifn2);
          ifunc2 = G__struct.memfunc[param_tagnum];
          ifunc2 = G__overload_match(funcname2,&para,hash2,ifunc2
                                     ,G__TRYMEMFUNC,G__PUBLIC,&ifn2,1
                                     ,1
                                     );
        }
        if(ifunc2 && -1!=ifn2) 
          funclist->p_rate[i] = G__USRCONVMATCH;
      }
    }

    /* add up matching rate */
    if(G__NOMATCH==funclist->p_rate[i]) {
      funclist->rate = G__NOMATCH;
      break;
    }
    else {
      if(param_isconst!=formal_isconst) funclist->p_rate[i] += G__CVCONVMATCH;
      /*
      if('u'==param_type && (0!=param_isconst&& 0==formal_isconst)) {
        funclist->p_rate[i]=G__NOMATCH;
        funclist->rate = G__NOMATCH;
      }
      else */ 
      if(G__NOMATCH!=funclist->rate)
        funclist->rate += funclist->p_rate[i];
    }
  }
  if(G__NOMATCH!=funclist->rate && 
     ((0==G__isconst && (p_ifunc->isconst[ifn]&G__CONSTFUNC))
      || (G__isconst && 0==(p_ifunc->isconst[ifn]&G__CONSTFUNC)))
      )
    funclist->rate += G__CVCONVMATCH;
}

/***********************************************************************
* int G__convert_param(libp,p_ifunc,ifn,i)
**********************************************************************/
int G__convert_param(G__param *libp,G__ifunc_table *p_ifunc
                     ,int ifn,G__funclist *pmatch)
{
  int i;
  unsigned int rate;
  char param_type,formal_type;
  int param_tagnum,formal_tagnum;
  int formal_reftype;
  int formal_isconst;
  G__value *param;
#ifdef G__OLDIMPLEMENTATION2195_YET
  int store_asm_cp = G__asm_cp;
#endif

  char conv[G__ONELINE],arg1[G__ONELINE],parameter[G__ONELINE];
  long store_struct_offset; /* used to be int */
  int store_tagnum;
  int store_isconst;
  int baseoffset;
  G__value reg;
  int store_oprovld;
  int rewindflag=0;
  int recursive =0 ;
  int rewind_arg;
  int match = 0;

  for(i=0;i<libp->paran;i++) {
    rate = pmatch->p_rate[i];
    param_type = libp->para[i].type;
    formal_type = p_ifunc->para_type[ifn][i];
    param_tagnum = libp->para[i].tagnum;
    formal_tagnum = p_ifunc->para_p_tagtable[ifn][i];
    param = &libp->para[i];
    formal_reftype = p_ifunc->para_reftype[ifn][i];
#ifndef G__OLDIMPLEMENTATION
    rewind_arg = libp->paran-i-1;
#else
    rewind_arg = p_ifunc->para_nu[ifn]-i-1;
#endif
    formal_isconst = p_ifunc->para_isconst[ifn][i];

    if(rate&G__USRCONVMATCH) {
      if(formal_type=='u') {
        /* create temp object buffer */
        if(G__CPPLINK!=G__struct.iscpplink[formal_tagnum]) {
          G__alloc_tempobject(formal_tagnum,-1);
#ifdef G__ASM
          if(G__asm_noverflow) {
#ifdef G__ASM_DBG
            if(G__asm_dbg) {
              G__fprinterr(G__serr,"%3x: ALLOCTEMP %s %d\n"
                      ,G__asm_cp,G__struct.name[formal_tagnum],formal_tagnum);
              G__fprinterr(G__serr,"%3x: SETTEMP\n",G__asm_cp+2);
            }
#endif
            G__asm_inst[G__asm_cp] = G__ALLOCTEMP;
            G__asm_inst[G__asm_cp+1] = formal_tagnum;
            G__inc_cp_asm(2,0);
            G__asm_inst[G__asm_cp] = G__SETTEMP;
            G__inc_cp_asm(1,0);
          }
#endif
        }
        
        /* try finding constructor */
        if('u'==param_type) {
          if(param->obj.i<0) 
            sprintf(arg1,"(%s)(%ld)"
                    ,G__fulltagname(param_tagnum,1),param->obj.i);
          else
            sprintf(arg1,"(%s)%ld",G__fulltagname(param_tagnum,1),param->obj.i);
        }
        else {
          G__valuemonitor(*param,arg1);
        }
        sprintf(conv,"%s(%s)",G__struct.name[formal_tagnum],arg1);
        
        if(G__dispsource) {
          G__fprinterr(G__serr, "!!!Trying implicit conversion %s,%d\n"
                  ,conv,G__templevel);
        }
        
        store_struct_offset = G__store_struct_offset;
        G__store_struct_offset = G__p_tempbuf->obj.obj.i;
        
        store_tagnum = G__tagnum;
        G__tagnum = formal_tagnum;
        store_isconst = G__isconst;
        G__isconst = formal_isconst;
        
        /* avoid duplicated argument evaluation in p-code stack */
        store_oprovld = G__oprovld;
        G__oprovld=1;
        
#ifdef G__ASM
        if(G__asm_noverflow && rewind_arg) {
          rewindflag=1;
#ifdef G__ASM_DBG
          if(G__asm_dbg) G__fprinterr(G__serr,"%3x: REWINDSTACK %d\n"
                                 ,G__asm_cp,rewind_arg);
#endif
          G__asm_inst[G__asm_cp] = G__REWINDSTACK;
          G__asm_inst[G__asm_cp+1] = rewind_arg;
          G__inc_cp_asm(2,0);
        }
#endif
        
        ++recursive;
        if(G__CPPLINK==G__struct.iscpplink[formal_tagnum]) {
          /* in case of pre-compiled class */
          reg=G__getfunction(conv,&match,G__TRYIMPLICITCONSTRUCTOR);
          if(match) {
            G__store_tempobject(reg);
#ifdef G__ASM
            if(G__asm_noverflow) {
#ifdef G__ASM_DBG
              if(G__asm_dbg) G__fprinterr(G__serr,"%3x: STORETEMP\n",G__asm_cp);
#endif
              G__asm_inst[G__asm_cp]=G__STORETEMP;
              G__inc_cp_asm(1,0);
            }
#endif
          }
          else {
            G__pop_tempobject();
            sprintf(conv,"operator %s()",G__fulltagname(formal_tagnum,1));
            G__store_struct_offset = param->obj.i;
            G__tagnum = param->tagnum;
            if(-1!=G__tagnum) reg=G__getfunction(conv,&match,G__TRYMEMFUNC);
            if(!match) G__store_tempobject(G__null);
          }
        }
        else {
          /* in case of interpreted class */
          G__getfunction(conv,&match,G__TRYIMPLICITCONSTRUCTOR);
          if(match) {
            if(G__asm_noverflow) {
#ifdef G__ASM_DBG
              if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPTEMP %d\n"
                                          ,G__asm_cp,formal_tagnum);
#endif
              G__asm_inst[G__asm_cp] = G__POPTEMP;
              G__asm_inst[G__asm_cp+1] = formal_tagnum;
              G__inc_cp_asm(2,0);
            }
          }
          else {
            G__pop_tempobject();
            if(G__asm_noverflow) G__inc_cp_asm(-3,0);
            sprintf(conv,"operator %s()",G__fulltagname(formal_tagnum,1));
            G__store_struct_offset = param->obj.i;
            G__tagnum = param->tagnum;
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
#endif
            reg=G__getfunction(conv,&match,G__TRYMEMFUNC);
            if(!match) {
              if(G__asm_noverflow) {
                G__inc_cp_asm(-2,0);
                if(rewindflag) {
                  G__asm_inst[G__asm_cp-2]=G__REWINDSTACK; 
                  G__asm_inst[G__asm_cp-1] = rewind_arg;
                }
#ifdef G__ASM_DBG
                if(G__asm_dbg) 
                  G__fprinterr(G__serr,"ALLOCTEMP,SETTEMP Cancelled %x\n",G__asm_cp);
#endif
              }
            }
#ifdef G__ASM
            else if(G__asm_noverflow) {
              G__asm_inst[G__asm_cp] = G__POPSTROS;
              G__inc_cp_asm(1,0);
#ifdef G__ASM_DBG
              if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp-1);
#endif
            }
#endif
          }
        }
        --recursive;
        
        G__oprovld=store_oprovld;
        
        G__isconst = store_isconst;
        G__tagnum = store_tagnum;
        G__store_struct_offset = store_struct_offset;
        
        /* if no constructor, try converting to base class */
        
        
        if(match==0) {
          if('u'==param_type &&
#ifdef G__VIRTUALBASE
             -1 != (baseoffset=G__ispublicbase(formal_tagnum,param_tagnum
                                               ,param->obj.i))) {
#else
             -1 != (baseoffset=G__ispublicbase(formal_tagnum,param_tagnum))) {
#endif
            if(G__dispsource) {
              G__fprinterr(G__serr, "!!!Implicit conversion from %s to base %s\n"
                      ,G__struct.name[param_tagnum]
                      ,G__struct.name[formal_tagnum]);
            }
            param->typenum = -1;
            param->tagnum = formal_tagnum;
            param->obj.i += baseoffset;
            param->ref += baseoffset;
#ifdef G__ASM
            if(G__asm_noverflow) {
#ifdef G__ASM_DBG
              if(G__asm_dbg) G__fprinterr(G__serr,"%3x: BASECONV %d %d\n"
                                   ,G__asm_cp,formal_tagnum,baseoffset);
#endif
              G__asm_inst[G__asm_cp] = G__BASECONV;
              G__asm_inst[G__asm_cp+1] = formal_tagnum;
              G__asm_inst[G__asm_cp+2] = baseoffset;
              G__inc_cp_asm(3,0);
              if(rewind_arg) {
                rewindflag=1;
#ifdef G__ASM_DBG
                if(G__asm_dbg) G__fprinterr(G__serr,"%3x: REWINDSTACK %d\n"
                                       ,G__asm_cp,-rewind_arg);
#endif
                G__asm_inst[G__asm_cp] = G__REWINDSTACK;
                G__asm_inst[G__asm_cp+1] = -rewind_arg;
                G__inc_cp_asm(2,0);
              }
#endif
              if(param->obj.i<0) 
                sprintf(parameter,"(%s)(%ld)",G__struct.name[formal_tagnum]
                        ,param->obj.i);
              else
                sprintf(parameter,"(%s)%ld",G__struct.name[formal_tagnum]
                        ,param->obj.i);
            }
            match=1;
            G__pop_tempobject();
                                               }
          else { /* all conversion failed */
            if(G__dispsource) {
              G__fprinterr(G__serr,
                      "!!!Implicit conversion %s,%d tried, but failed\n"
                      ,conv,G__templevel);
            }
            G__pop_tempobject();
#ifdef G__ASM
            if(rewindflag) {
#ifdef G__ASM_DBG
              if(G__asm_dbg) G__fprinterr(G__serr,"REWINDSTACK cancelled\n");
#endif
              G__inc_cp_asm(-2,0);
            }
          }
          
#else /* ON181 */
          
          /* all conversion failed */
          if(G__dispsource) {
            G__fprinterr(G__serr,
                    "!!!Implicit conversion %s,%d tried, but failed\n"
                    ,conv,G__templevel);
          }
          G__pop_tempobject();
#ifdef G__ASM
          if(rewindflag) {
#ifdef G__ASM_DBG
            if(G__asm_dbg) G__fprinterr(G__serr,"REWINDSTACK cancelled\n");
#endif
            G__inc_cp_asm(-2,0);
          }
#endif
          
#endif /* ON181 */
        }
        else { /* match==1, conversion successful */
          if(G__dispsource) {
            if(G__p_tempbuf->obj.obj.i<0) 
              G__fprinterr(G__serr,
                      "!!!Create temp object (%s)(%ld),%d for implicit conversion\n"
                      ,conv ,G__p_tempbuf->obj.obj.i ,G__templevel);
            else
              G__fprinterr(G__serr,
                      "!!!Create temp object (%s)%ld,%d for implicit conversion\n"
                      ,conv ,G__p_tempbuf->obj.obj.i ,G__templevel);
          }
#ifdef G__ASM
          if(G__asm_noverflow && rewind_arg) {
            rewindflag=1;
#ifdef G__ASM_DBG
            if(G__asm_dbg) G__fprinterr(G__serr,"%3x: REWINDSTACK %d\n"
                                   ,G__asm_cp,-rewind_arg);
#endif
            G__asm_inst[G__asm_cp] = G__REWINDSTACK;
            G__asm_inst[G__asm_cp+1] = -rewind_arg;
            G__inc_cp_asm(2,0);
          }
#endif
          *param = G__p_tempbuf->obj;
          sprintf(parameter,"(%s)%ld" ,G__struct.name[formal_tagnum]
                  ,G__p_tempbuf->obj.obj.i);
        } /* end of if(match==0) */
        
      }
      else if(-1!=param->tagnum) {
        long store_struct_offset=G__store_struct_offset;
        int store_tagnum=G__tagnum;
        int store_isconst=G__isconst;
        sprintf(conv,"operator %s()"
                ,G__type2string(formal_type,formal_tagnum,-1,0,0));
        G__store_struct_offset = param->obj.i;
        G__tagnum=param->tagnum;
#ifdef G__ASM
        if(G__asm_noverflow) {
          if(rewind_arg) {
            rewindflag=1;
#ifdef G__ASM_DBG
            if(G__asm_dbg) G__fprinterr(G__serr,"%3x: REWINDSTACK %d\n"
                                   ,G__asm_cp,rewind_arg);
#endif
            G__asm_inst[G__asm_cp] = G__REWINDSTACK;
            G__asm_inst[G__asm_cp+1] = rewind_arg;
            G__inc_cp_asm(2,0);
          }
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
#endif
        reg=G__getfunction(conv,&match,G__TRYMEMFUNC);
        if(!match
           && 0!=formal_isconst
           ) {
          sprintf(conv,"operator const %s()"
                  ,G__type2string(formal_type,formal_tagnum,-1,0,0));
          G__store_struct_offset = param->obj.i;
          G__tagnum=param->tagnum;
          reg=G__getfunction(conv,&match,G__TRYMEMFUNC);
        }
        G__isconst = store_isconst;
        G__tagnum=store_tagnum;
        G__store_struct_offset=store_struct_offset;
#ifdef G__ASM
        if(G__asm_noverflow) {
          if(rewind_arg) {
            rewindflag=1;
#ifdef G__ASM_DBG
            if(G__asm_dbg) G__fprinterr(G__serr,"%3x: REWINDSTACK %d\n"
                                   ,G__asm_cp,-rewind_arg);
#endif
            G__asm_inst[G__asm_cp] = G__REWINDSTACK;
            G__asm_inst[G__asm_cp+1] = -rewind_arg;
            G__inc_cp_asm(2,0);
          }
          G__asm_inst[G__asm_cp] = G__POPSTROS;
          G__inc_cp_asm(1,0);
#ifdef G__ASM_DBG
          if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp-1);
#endif
        }
#endif
        /* fixing 'cout<<x' fundamental conversion opr with opr overloading 
         * Not 100% sure if this is OK. */
        if(match) *param = reg;
        else if(rewindflag) {
#ifdef G__ASM_DBG
          if(G__asm_dbg) G__fprinterr(G__serr,"REWINDSTACK~ cancelled\n");
#endif
          G__inc_cp_asm(-7,0);
        }
        else {
#ifdef G__ASM_DBG
          if(G__asm_dbg) G__fprinterr(G__serr,"PUSHSTROS~ cancelled\n");
#endif
          G__inc_cp_asm(-3,0);
      }
      }
      else {
        match=0;
        /* #ifdef G__DEBUG */
        if(recursive&&G__dispsource) {
          G__valuemonitor(*param,arg1);
          G__fprinterr(G__serr,"!!!Recursive implicit conversion %s(%s) rejected\n"
                  ,G__struct.name[formal_tagnum],arg1);
        }
        /* #endif */
      }
      continue;
    }

    switch(formal_type) {
    case 'b':
    case 'c':
    case 'r':
    case 's':
    case 'h':
    case 'i':
    case 'k':
    case 'l':
      switch(param_type) {
      case 'd':
      case 'f':
        /* std conv */
        if(G__PARAREFERENCE==formal_reftype) {
          param->obj.i = (long)param->obj.d;
          param->type = formal_type;
          param->ref = 0;
        }
        break;
      }
      break;
    case 'g':
      switch(param_type) {
      case 'd':
      case 'f':
        /* std conv */
        if(G__PARAREFERENCE==formal_reftype) {
          param->obj.i = param->obj.d?1:0;
          param->type = formal_type;
          param->ref = 0;
        }
      case 'l':
      case 'i':
      case 's':
      case 'c':
      case 'h':
      case 'k':
      case 'r':
      case 'b':
        if(G__PARAREFERENCE==formal_reftype) {
          param->obj.i = (long)param->obj.i?1:0;
          param->type = formal_type;
          param->ref = 0;
        }
        break;
      }
      break;
    case 'n': /* long long */
      if(G__PARAREFERENCE==formal_reftype) {
        param->type = formal_type;
        if(param->type!=formal_type) param->ref = 0;
        switch(param_type) {
        case 'd':
        case 'f':
          param->obj.ll = (G__int64)param->obj.d;
          break;
        case 'g':
        case 'c':
        case 's':
        case 'i':
        case 'l':
        case 'b':
        case 'r':
        case 'h':
        case 'k':
          param->obj.ll = param->obj.i;
          break;
        case 'm':
          param->obj.ll = param->obj.ull;
          break;
        case 'q':
          param->obj.ll = (G__int64)param->obj.ld;
          break;
        }
      }
      break;
    case 'm': /* unsigned long long */
      if(G__PARAREFERENCE==formal_reftype) {
        param->type = formal_type;
        if(param->type!=formal_type) param->ref = 0;
        switch(param_type) {
        case 'd':
        case 'f':
          param->obj.ull = (G__uint64)param->obj.d;
          break;
        case 'g':
        case 'c':
        case 's':
        case 'i':
        case 'l':
        case 'b':
        case 'r':
        case 'h':
        case 'k':
          param->obj.ull = param->obj.i;
          break;
        case 'n':
          param->obj.ull = param->obj.ll;
          break;
        case 'q':
          param->obj.ull = (G__int64)param->obj.ld;
          break;
        }
      }
      break;
    case 'q': /* long double */
      if(G__PARAREFERENCE==formal_reftype) {
        param->type = formal_type;
        if(param->type!=formal_type) param->ref = 0;
        switch(param_type) {
        case 'd':
        case 'f':
          param->obj.ld = param->obj.d;
          break;
        case 'g':
        case 'c':
        case 's':
        case 'i':
        case 'l':
        case 'b':
        case 'r':
        case 'h':
        case 'k':
          param->obj.ld = param->obj.i;
          break;
        case 'n':
          param->obj.ld = (long double)param->obj.ll;
          break;
        case 'm':
          param->obj.ld = (long double)param->obj.ld;
          break;
        }
      }
      break;
    case 'd':
    case 'f':
      switch(param_type) {
      case 'b':
      case 'c':
      case 'r':
      case 's':
      case 'h':
      case 'i':
      case 'k':
      case 'l':
      case 'g':
      case 'n':
      case 'm':
        /* std conv */
        if(G__PARAREFERENCE==formal_reftype) {
          param->obj.d = param->obj.i;
          param->type = formal_type;
          param->ref = 0;
        }
        break;
      }
      break;
    case 'u':
      switch(param_type) {
      case 'u':
        if(0==(rate&0xffffff00)) {
          /* exact */
          if('e'==G__struct.type[param_tagnum]) {
            if(param->ref) param->obj.i = *(long*)(param->ref);
          }
        }
        else /* if(G__PARAREFERENCE==formal_reftype) */ {
          if(-1 != (baseoffset=G__ispublicbase(formal_tagnum,param_tagnum
                                               ,param->obj.i))) {
            param->tagnum = formal_tagnum;
            param->obj.i += baseoffset;
            param->ref = param->obj.i;
#ifdef G__ASM
            if(G__asm_noverflow) {
              if(rewind_arg 
                 ) {
#ifdef G__ASM_DBG
                if(G__asm_dbg) G__fprinterr(G__serr,"%3x: REWINDSTACK %d\n"
                                       ,G__asm_cp,rewind_arg);
#endif
                G__asm_inst[G__asm_cp]=G__REWINDSTACK; 
                G__asm_inst[G__asm_cp+1] = rewind_arg;
                G__inc_cp_asm(2,0);
              }
#ifdef G__ASM_DBG
              if(G__asm_dbg) G__fprinterr(G__serr,"%3x: BASECONV %d %d\n"
                                     ,G__asm_cp,formal_tagnum,baseoffset);
#endif
              G__asm_inst[G__asm_cp] = G__BASECONV;
              G__asm_inst[G__asm_cp+1] = formal_tagnum;
              G__asm_inst[G__asm_cp+2] = baseoffset;
              G__inc_cp_asm(3,0);
              if(rewind_arg 
                 ) {
#ifdef G__ASM_DBG
                if(G__asm_dbg) G__fprinterr(G__serr,"%3x: REWINDSTACK %d\n"
                                       ,G__asm_cp,-rewind_arg);
#endif
                G__asm_inst[G__asm_cp] = G__REWINDSTACK;
                G__asm_inst[G__asm_cp+1] = -rewind_arg;
                G__inc_cp_asm(2,0);
              }
            }
#endif
          }
        }
        break;
      }
      break;
    case 'U':
      switch(param_type) {
      case 'U':
        /* Pointer to derived class can be converted to
         * pointer to base class.
         * add offset, modify char *parameter and 
         * G__value *param
         */
        if(-1 != (baseoffset=G__ispublicbase(formal_tagnum,param_tagnum
                                             ,param->obj.i))) {
          param->tagnum = formal_tagnum;
          param->obj.i += baseoffset;
          param->ref += baseoffset;
#ifdef G__ASM
          if(G__asm_noverflow) {
            if(rewind_arg 
               ) {
#ifdef G__ASM_DBG
              if(G__asm_dbg) G__fprinterr(G__serr,"%3x: REWINDSTACK %d\n"
                                     ,G__asm_cp,rewind_arg);
#endif
              G__asm_inst[G__asm_cp]=G__REWINDSTACK; 
              G__asm_inst[G__asm_cp+1] = rewind_arg;
              G__inc_cp_asm(2,0);
            }
#ifdef G__ASM_DBG
            if(G__asm_dbg) G__fprinterr(G__serr,"%3x: BASECONV %d %d\n"
                                   ,G__asm_cp,formal_tagnum,baseoffset);
#endif
            G__asm_inst[G__asm_cp] = G__BASECONV;
            G__asm_inst[G__asm_cp+1] = formal_tagnum;
            G__asm_inst[G__asm_cp+2] = baseoffset;
            G__inc_cp_asm(3,0);
            if(rewind_arg 
               ) {
#ifdef G__ASM_DBG
              if(G__asm_dbg) G__fprinterr(G__serr,"%3x: REWINDSTACK %d\n"
                                     ,G__asm_cp,-rewind_arg);
#endif
              G__asm_inst[G__asm_cp] = G__REWINDSTACK;
              G__asm_inst[G__asm_cp+1] = -rewind_arg;
              G__inc_cp_asm(2,0);
            }
          }
#endif
        }
        break;
      }
#ifndef G__OLDIMPLEMENTATION2191
    case '1':
#else
    case 'Q':
#endif
      if('C'==param_type && 
         p_ifunc->pentry[ifn]->size<0
         ) {
        G__genericerror("Limitation: Precompiled function can not get pointer to interpreted function as argument");
        return(-1);
      }
    }
    
  }
#ifdef G__OLDIMPLEMENTATION2195_YET
  if(G__asm_cp>store_asm_cp) {
    if(G__asm_dbg) G__fprinterr(G__serr,"G__convert_param instructions cancelled\n");
    G__inc_cp_asm(store_asm_cp-G__asm_cp,0);
  }
#endif
  return(0);
}

/***********************************************************************
* G__display_param(scopetagnum,funcname,libp);
**********************************************************************/
void G__display_param(FILE* fp,int scopetagnum
                      ,char *funcname,G__param *libp)
{
  int i;
#ifndef G__OLDIMPLEMENTATION1485
  if(G__serr==fp) {
    if(-1!=scopetagnum) G__fprinterr(G__serr,"%s::",G__fulltagname(scopetagnum,1));
    G__fprinterr(G__serr,"%s(",funcname);
    for(i=0;i<libp->paran;i++) {
      switch(libp->para[i].type) {
      case 'd':
      case 'f':
        G__fprinterr(G__serr,"%s",G__type2string(libp->para[i].type
                                       ,libp->para[i].tagnum
                                       ,libp->para[i].typenum
                                       ,0
                                       ,0));
        break;
      default:
        G__fprinterr(G__serr,"%s",G__type2string(libp->para[i].type
                                       ,libp->para[i].tagnum
                                       ,libp->para[i].typenum
                                       ,libp->para[i].obj.reftype.reftype
                                       ,0));
        break;
      }
      if(i!=libp->paran-1) G__fprinterr(G__serr,",");
    }
    G__fprinterr(G__serr,");\n");
  }
  else {
#endif
    if(-1!=scopetagnum) fprintf(fp,"%s::",G__fulltagname(scopetagnum,1));
    fprintf(fp,"%s(",funcname);
    for(i=0;i<libp->paran;i++) {
      switch(libp->para[i].type) {
      case 'd':
      case 'f':
        fprintf(fp,"%s",G__type2string(libp->para[i].type
                                       ,libp->para[i].tagnum
                                       ,libp->para[i].typenum
                                       ,0
                                       ,0));
        break;
      default:
        fprintf(fp,"%s",G__type2string(libp->para[i].type
                                       ,libp->para[i].tagnum
                                       ,libp->para[i].typenum
                                       ,libp->para[i].obj.reftype.reftype
                                       ,0));
        break;
      }
      if(i!=libp->paran-1) fprintf(fp,",");
    }
    fprintf(fp,");\n");
#ifndef G__OLDIMPLEMENTATION1485
  }
#endif
}

/***********************************************************************
* G__display_func(G__serr,ifunc,ifn);
**********************************************************************/
void G__display_func(FILE *fp,G__ifunc_table *ifunc,int ifn)
{
  int i;
  int store_iscpp = G__iscpp;
  G__iscpp = 1;

  if(!ifunc || !ifunc->pentry[ifn]) return;
  
#ifndef G__OLDIMPLEMENTATION1485
  if(G__serr==fp) {
    if(ifunc->pentry[ifn]->filenum>=0) { /* 2012 must leave this one */
      G__fprinterr(G__serr,"%-10s%4d "
            ,G__stripfilename(G__srcfile[ifunc->pentry[ifn]->filenum].filename)
              ,ifunc->pentry[ifn]->line_number);
    }
    else {
      G__fprinterr(G__serr,"%-10s%4d ","(compiled)",0);
    }
    G__fprinterr(G__serr,"%s ",G__type2string(ifunc->type[ifn]
                                    ,ifunc->p_tagtable[ifn]
                                    ,ifunc->p_typetable[ifn]
                                    ,ifunc->reftype[ifn]
                                    ,ifunc->isconst[ifn]));
    if(-1!=ifunc->tagnum) G__fprinterr(G__serr,"%s::",G__fulltagname(ifunc->tagnum,1));
    G__fprinterr(G__serr,"%s(",ifunc->funcname[ifn]);
    for(i=0;i<ifunc->para_nu[ifn];i++) {
      G__fprinterr(G__serr,"%s",G__type2string(ifunc->para_type[ifn][i]
                                     ,ifunc->para_p_tagtable[ifn][i]
                                     ,ifunc->para_p_typetable[ifn][i]
                                     ,ifunc->para_reftype[ifn][i]
                                     ,ifunc->para_isconst[ifn][i]));
      if(i!=ifunc->para_nu[ifn]-1) G__fprinterr(G__serr,",");
    }
    G__fprinterr(G__serr,");\n");
  } 
  else {
#endif
    if(ifunc->pentry[ifn]->filenum>=0) { /* 2012 must leave this one */
      fprintf(fp,"%-10s%4d "
            ,G__stripfilename(G__srcfile[ifunc->pentry[ifn]->filenum].filename)
              ,ifunc->pentry[ifn]->line_number);
    }
    else {
      fprintf(fp,"%-10s%4d ","(compiled)",0);
    }
    fprintf(fp,"%s ",G__type2string(ifunc->type[ifn]
                                    ,ifunc->p_tagtable[ifn]
                                    ,ifunc->p_typetable[ifn]
                                    ,ifunc->reftype[ifn]
                                    ,ifunc->isconst[ifn]));
    if(-1!=ifunc->tagnum) fprintf(fp,"%s::",G__fulltagname(ifunc->tagnum,1));
    fprintf(fp,"%s(",ifunc->funcname[ifn]);
    for(i=0;i<ifunc->para_nu[ifn];i++) {
      fprintf(fp,"%s",G__type2string(ifunc->para_type[ifn][i]
                                     ,ifunc->para_p_tagtable[ifn][i]
                                     ,ifunc->para_p_typetable[ifn][i]
                                     ,ifunc->para_reftype[ifn][i]
                                     ,ifunc->para_isconst[ifn][i]));
      if(i!=ifunc->para_nu[ifn]-1) fprintf(fp,",");
    }
    fprintf(fp,");\n");
#ifndef G__OLDIMPLEMENTATION1485
  }
#endif

  G__iscpp = store_iscpp;
}

/***********************************************************************
* G__display_ambiguous(funclist,bestmatch);
**********************************************************************/
void G__display_ambiguous(int scopetagnum,char *funcname
                          ,G__param *libp,G__funclist *funclist
                          ,unsigned int bestmatch)
{
  G__fprinterr(G__serr,"Calling : ");
  G__display_param(G__serr,scopetagnum,funcname,libp);
  G__fprinterr(G__serr,"Match rank: file     line  signature\n");
  while(funclist) {
    struct G__ifunc_table *ifunc = funclist->ifunc; 
    int ifn = funclist->ifn;
    if(bestmatch==funclist->rate) G__fprinterr(G__serr,"* %8x ",funclist->rate);
    else                          G__fprinterr(G__serr,"  %8x ",funclist->rate);
    G__display_func(G__serr,ifunc,ifn);
    funclist = funclist->prev;
  }
}

/***********************************************************************
* G__add_templatefunc()
*
* Search matching template function, search by name then parameter.
* If match found, expand template, parse as pre-run 
***********************************************************************/
struct G__funclist* G__add_templatefunc(char *funcnamein,G__param* libp
                                        ,int hash,G__funclist *funclist
                                        ,G__ifunc_table *p_ifunc,int isrecursive)
{
  struct G__Definetemplatefunc *deftmpfunc;
  struct G__Charlist call_para;
  /* int env_tagnum=G__get_envtagnum(); */
  int env_tagnum = p_ifunc->tagnum;
  struct G__inheritance *baseclass;
  int store_friendtagnum = G__friendtagnum;
  struct G__ifunc_table *ifunc; 
  int ifn;
  char *funcname;
#ifndef G__OLDIMPLEMENTATION1560
  char *ptmplt;
#endif
  char *pexplicitarg=0;

  funcname = (char*)malloc(strlen(funcnamein)+1);
  strcpy(funcname,funcnamein);

  if(-1!=env_tagnum) baseclass = G__struct.baseclass[env_tagnum];
  else               baseclass = &G__globalusingnamespace;
  if(0==baseclass->basen) baseclass = (struct G__inheritance*)NULL;


  call_para.string = (char*)NULL;
  call_para.next = (struct G__Charlist*)NULL;
  deftmpfunc = &G__definedtemplatefunc;

#ifndef G__OLDIMPLEMENTATION1560
  ptmplt = strchr(funcname,'<');
  if(ptmplt) {
     if (strncmp("operator",funcname,ptmplt-funcname)==0) {
        /* We have operator< */
        if (ptmplt[1]=='<') ptmplt = strchr(ptmplt+2,'<');
        else ptmplt = strchr(ptmplt+1,'<');
     }
  }
  if(ptmplt) {
    if ((-1!=env_tagnum) && strcmp(funcname,G__struct.name[env_tagnum])==0) {
       /* this is probably a template constructor of a class template */
       ptmplt = (char*)0;
    } 
    else {
      int tmp;
      *ptmplt = 0; 
      if(G__defined_templatefunc(funcname)) {
         G__hash(funcname,hash,tmp);
      }
      else {
         pexplicitarg = ptmplt;
         *ptmplt = '<';
         ptmplt = (char*)0;
      }
    }
  }
#else 
  pexplicitarg = strchr(funcname,'<');
#endif

  if(pexplicitarg) {
    /* funcname="f<int>" ->  funcname="f" , pexplicitarg="int>" */
    int tmp=0;
    *pexplicitarg = 0;
    ++pexplicitarg;
    G__hash(funcname,hash,tmp);
  }
  
  /* Search matching template function name */
  while(deftmpfunc->next) {
    G__freecharlist(&call_para);
#ifndef G__OLDIMPLEMENTATION1560
    if(ptmplt) {
      int itmp = 0;
      int ip = 1;
      int c;
      char buf[G__ONELINE];
      do {
        c = G__getstream_template(ptmplt,&ip,buf,",>");
        G__checkset_charlist(buf,&call_para,++itmp,'u');
      } while(c!='>');
    }
#endif
    if(deftmpfunc->hash==hash && strcmp(deftmpfunc->name,funcname)==0 &&
       (G__matchtemplatefunc(deftmpfunc,libp,&call_para,G__PROMOTION)
#ifndef G__OLDIMPLEMEMTATION2214
        || (pexplicitarg && libp->paran==0)
#else
        || pexplicitarg
#endif
        )) {

      if(-1!=deftmpfunc->parent_tagnum && 
         env_tagnum!=deftmpfunc->parent_tagnum) {
        if(baseclass) {
          int temp;
          for(temp=0;temp<baseclass->basen;temp++) {
            if(baseclass->basetagnum[temp]==deftmpfunc->parent_tagnum) {
              goto match_found;
            }
          }
        }
        deftmpfunc = deftmpfunc->next;
        continue;
      }
    match_found:

      G__friendtagnum = deftmpfunc->friendtagnum;

      if(pexplicitarg) {
        int npara=0;
        G__gettemplatearglist(pexplicitarg,&call_para
                              ,deftmpfunc->def_para,&npara
                              ,-1
                              );
      }

      if(pexplicitarg) {
        int tmp=0;
        G__hash(funcname,hash,tmp);
      }
      
      /* matches funcname and parameter,
       * then expand the template and parse as prerun */
      G__replacetemplate(
                         funcname
                         ,funcnamein
                         ,&call_para /* needs to make this up */
                         ,deftmpfunc->def_fp
                         ,deftmpfunc->line
                         ,deftmpfunc->filenum
                         ,&(deftmpfunc->def_pos)
                         ,deftmpfunc->def_para
                         ,0
                         ,SHRT_MAX /* large enough number */
                         ,deftmpfunc->parent_tagnum
                         );

      G__friendtagnum = store_friendtagnum;


      /* search for instantiated template function */
      ifunc = p_ifunc;
      while(ifunc && ifunc->next && ifunc->next->allifunc) ifunc=ifunc->next;
      if(ifunc) {
        ifn = ifunc->allifunc-1;
        if(
           strcmp(funcnamein,ifunc->funcname[ifn])==0
           ) {
#ifndef G__OLDIMPLEMENTATION1560
          if(ptmplt) {
            int tmp;
            *ptmplt='<';
            free((void*)ifunc->funcname[ifn]);
            ifunc->funcname[ifn] = (char*)malloc(strlen(funcnamein)+1);
            strcpy(ifunc->funcname[ifn],funcnamein);
            G__hash(funcnamein,hash,tmp);
            ifunc->hash[ifn] = hash;
          }
#endif
          if(0==ifunc->pentry[ifn]->p && G__NOLINK==G__globalcomp) {
            /* This was only a prototype template, search for definition
             * template */
            deftmpfunc = deftmpfunc->next;
            continue;
          }
          funclist = G__funclist_add(funclist,ifunc,ifn,0);
          if(ifunc->para_nu[ifn]<libp->paran ||
             (ifunc->para_nu[ifn]>libp->paran&&
              !ifunc->para_default[ifn][libp->paran])) {
            funclist->rate = G__NOMATCH;
          }
          else {
            G__rate_parameter_match(libp,ifunc,ifn,funclist,isrecursive);
          }
        }
      }
      G__freecharlist(&call_para);
    }
    deftmpfunc = deftmpfunc->next;
  }
  G__freecharlist(&call_para);

#ifndef G__OLDIMPLEMENTATION1560
  if(funcname) free((void*)funcname);
#endif

  return(funclist);
}

/***********************************************************************
* G__rate_binary_operator()
**********************************************************************/
struct G__funclist* G__rate_binary_operator(G__ifunc_table *p_ifunc,
   G__param *libp,
   int tagnum,
   char* funcname,
   int hash,
   G__funclist *funclist,
   int isrecursive)
{
  int i;
  struct G__param fpara;
#ifdef G__DEBUG
  {
    int jdbg;
    int sizedbg=sizeof(struct G__param);
    char *pcdbg = (char*)(&fpara);
    for(jdbg=0;jdbg<(int)sizedbg;jdbg++) {
      *(pcdbg+jdbg) = (char)0xa3;
    }
  }
#endif

  /* set 1st argument as the object */
  fpara.para[0].type='u';
  fpara.para[0].tagnum=tagnum;
  fpara.para[0].typenum = -1;
  fpara.para[0].obj.i = G__store_struct_offset;;
  fpara.para[0].ref = G__store_struct_offset;;
  fpara.para[0].isconst = G__isconst; 

  /* set 2nd to n arguments */
  fpara.paran = libp->paran+1;
  for(i=0;i<libp->paran;i++) fpara.para[i+1] = libp->para[i];

  /* Search for name match
   *  if reserved func or K&R, match immediately
   *  check number of arguments and default parameters
   *  rate parameter match */
  while(p_ifunc) {
    int ifn;
    for(ifn=0;ifn<p_ifunc->allifunc;++ifn) {
      if(hash==p_ifunc->hash[ifn]&&strcmp(funcname,p_ifunc->funcname[ifn])==0){
        if(p_ifunc->para_nu[ifn]<fpara.paran ||
           (p_ifunc->para_nu[ifn]>fpara.paran&&
            !p_ifunc->para_default[ifn][fpara.paran])
#ifdef G__OLDIMPLEMENTATION1260_YET
           || (G__isconst && 0==p_ifunc->isconst[ifn])
#endif
           || (isrecursive && p_ifunc->isexplicit[ifn])
           ) {
        }
        else {
          funclist = G__funclist_add(funclist,p_ifunc,ifn,0);
          G__rate_parameter_match(&fpara,p_ifunc,ifn,funclist,isrecursive);
          funclist->ifunc = 0; /* added as dummy */
        }
      }
    }
    p_ifunc = p_ifunc->next;
  }

  return(funclist);
}

/***********************************************************************
* G__identical_function
**********************************************************************/
int G__identical_function(G__funclist *match,G__funclist *func)
{
  int ipara;
  if(!match || !match->ifunc || !func || !func->ifunc) return(0);
  for(ipara=0;ipara<match->ifunc->para_nu[match->ifn];ipara++) {
    if(
       (match->ifunc->para_type[match->ifn][ipara] !=
        func->ifunc->para_type[func->ifn][ipara]) ||
       (match->ifunc->para_p_tagtable[match->ifn][ipara] !=
        func->ifunc->para_p_tagtable[func->ifn][ipara]) ||
       (match->ifunc->para_p_typetable[match->ifn][ipara] !=
        func->ifunc->para_p_typetable[func->ifn][ipara]) ||
       (match->ifunc->para_isconst[match->ifn][ipara] !=
        func->ifunc->para_isconst[func->ifn][ipara]) ||
       (match->ifunc->para_reftype[match->ifn][ipara] !=
        func->ifunc->para_reftype[func->ifn][ipara])
       ) {
      return(0);
    }
  }

  return(1);
}


/***********************************************************************
* G__overload_match(funcname,libp,hash,p_ifunc,memfunc_flag,access,pifn)
**********************************************************************/
struct G__ifunc_table* G__overload_match(char* funcname
                                         ,G__param *libp
                                         ,int hash
                                         ,G__ifunc_table *p_ifunc
                                         ,int memfunc_flag
                                         ,int access
                                         ,int *pifn
                                         ,int isrecursive
                                         ,int doconvert)
{
  struct G__funclist *funclist = (struct G__funclist*)NULL;
  struct G__funclist *match = (struct G__funclist*)NULL;
  unsigned int bestmatch = G__NOMATCH;
  struct G__funclist *func;
  int ambiguous = 0;
  int scopetagnum = p_ifunc->tagnum;
  struct G__ifunc_table *store_ifunc = p_ifunc; 
  int ix=0;


  /* Search for name match
   *  if reserved func or K&R, match immediately
   *  check number of arguments and default parameters
   *  rate parameter match */
  while(p_ifunc) {
    int ifn;
    for(ifn=0;ifn<p_ifunc->allifunc;++ifn) {
      if(hash==p_ifunc->hash[ifn]&&strcmp(funcname,p_ifunc->funcname[ifn])==0){
        if(p_ifunc->ansi[ifn]==0 || /* K&R C style header */
           p_ifunc->ansi[ifn]==2 || /* variable number of args */
           (G__HASH_MAIN==hash && strcmp(funcname,"main")==0)) {
          /* special match */
          *pifn = ifn;
          G__funclist_delete(funclist);
          return(p_ifunc);
        }
        if(-1!=p_ifunc->tagnum && 
           (memfunc_flag==G__TRYNORMAL && doconvert)
           && strcmp(G__struct.name[p_ifunc->tagnum],funcname)==0) {
          continue;
        }
        funclist = G__funclist_add(funclist,p_ifunc,ifn,0);
        if(p_ifunc->para_nu[ifn]<libp->paran ||
           (p_ifunc->para_nu[ifn]>libp->paran&&
            !p_ifunc->para_default[ifn][libp->paran])
#ifdef G__OLDIMPLEMENTATION1260_YET
           || (G__isconst && 0==p_ifunc->isconst[ifn])
#endif
           || (isrecursive && p_ifunc->isexplicit[ifn])
           ) {
          funclist->rate = G__NOMATCH;
        }
        else {
          G__rate_parameter_match(libp,p_ifunc,ifn,funclist,isrecursive);
        }
        if(G__EXACTMATCH==(funclist->rate&0xffffff00)) match = funclist;
      }
    }
    p_ifunc = p_ifunc->next;
    if(!p_ifunc && store_ifunc==G__p_ifunc && 
       ix<G__globalusingnamespace.basen) {
      p_ifunc=G__struct.memfunc[G__globalusingnamespace.basetagnum[ix]];
      ++ix;
    }
  }

  /* If exact match does not exist 
   *    search for template func
   *    rate parameter match */
  if(!match) {
    funclist =  G__add_templatefunc(funcname,libp,hash,funclist
                                    ,store_ifunc,isrecursive);
  }

  if(!match && (G__TRYUNARYOPR==memfunc_flag||G__TRYBINARYOPR==memfunc_flag)) {
    for(ix=0;ix<G__globalusingnamespace.basen;ix++) {
      funclist=G__rate_binary_operator(
                      G__struct.memfunc[G__globalusingnamespace.basetagnum[ix]]
                                       ,libp,G__tagnum,funcname,hash
                                       ,funclist,isrecursive);
    }
    funclist = G__rate_binary_operator(&G__ifunc,libp,G__tagnum,funcname,hash
                                       ,funclist,isrecursive);
  }

  /* if there is no name match, return null */
  if((struct G__funclist*)NULL==funclist) return((struct G__ifunc_table*)NULL);
  /* else  there is function name match */


  /*  choose the best match
   *    display error if the call is ambiguous
   *    display error if there is no parameter match */
  func = funclist;
  ambiguous = 0;
  while(func) {
    if(func->rate<bestmatch) {
      bestmatch = func->rate;
      match = func;
      ambiguous = 0;
    }
    else if(func->rate==bestmatch && bestmatch!=G__NOMATCH) {
      if(0==G__identical_function(match,func)) ++ambiguous;
      match = func;
    }
    func = func->prev;
  }

  if((G__TRYUNARYOPR==memfunc_flag||G__TRYBINARYOPR==memfunc_flag) && 
     match && 0==match->ifunc) {
    G__funclist_delete(funclist);
    return((struct G__ifunc_table*)NULL);
  }

#ifdef G__ASM_DBG
  /* #define G__ASM_DBG2 */
#endif
#ifdef G__ASM_DBG2
  if(G__dispsource) 
    G__display_ambiguous(scopetagnum,funcname,libp,funclist,bestmatch);
#endif

  if(!match) {
#if G__NEVER
    G__genericerror("Error: No appropriate match in the scope");
    *pifn = -1;
#endif
    G__funclist_delete(funclist);
    return((struct G__ifunc_table*)NULL);
  }

  if(ambiguous && G__EXACTMATCH!=bestmatch 
     && !isrecursive
     ) {
    if(!G__mask_error) {
      /* error, ambiguous overloading resolution */
      G__fprinterr(G__serr,"Error: Ambiguous overload resolution (%x,%d)"
                   ,bestmatch,ambiguous+1);
      G__genericerror((char*)NULL);
      G__display_ambiguous(scopetagnum,funcname,libp,funclist,bestmatch);
    }
    *pifn = -1;
    G__funclist_delete(funclist);
    return((struct G__ifunc_table*)NULL);
  }

  /* best match function found */
  p_ifunc = match->ifunc;
  *pifn = match->ifn;

  /*  check private, protected access rights 
   *    display error if no access right
   *    do parameter conversion if needed */
  if(0==(p_ifunc->access[*pifn]&access)&&(!G__isfriend(p_ifunc->tagnum))
     && G__NOLINK==G__globalcomp
     && G__TRYCONSTRUCTOR !=  memfunc_flag
     ) {
    /* no access right */
    G__fprinterr(G__serr,"Error: can not call private or protected function");
    G__genericerror((char*)NULL);
    G__fprinterr(G__serr,"  ");
    G__display_func(G__serr,p_ifunc,*pifn);
    G__display_ambiguous(scopetagnum,funcname,libp,funclist,bestmatch);
    *pifn = -1;
    G__funclist_delete(funclist);
    return((struct G__ifunc_table*)NULL);
  }

  /* convert parameter */
  if(
     doconvert && 
     G__convert_param(libp,p_ifunc,*pifn,match))
    return((struct G__ifunc_table*)NULL);

  G__funclist_delete(funclist);
  return(p_ifunc);
}


/***********************************************************************
* int G__interpret_func(result7,funcname,libp,hash,ifunc,funcmatch)
*
*
*  This function has to be changed to support ANSI style function 
* definition.
*
*
***********************************************************************/
int G__interpret_func(G__value *result7,char* funcname,G__param *libp,
                      int hash,G__ifunc_table *p_ifunc /*local variable overrides global variable*/
                      ,int funcmatch
                      ,int memfunc_flag)
/*  return 1 if function is executed */
/*  return 0 if function isn't executed */
{
  int ifn=0;
  struct G__var_array G_local;
  FILE *prev_fp;
  fpos_t prev_pos /*,temppos */;
  /* paraname[][] is used only for K&R func param. length should be OK */
  char paraname[G__MAXFUNCPARA][G__MAXNAME];
#ifdef G__OLDIMPLEMENTATION1802
  char temp[G__ONELINE];
#endif
  int ipara=0;
  int cin='\0';
  int /* ichar=0,*/ itemp=0;
  /* int store_linenumber; */
  int break_exit_func;
  int store_decl;
  G__value buf;
  int store_var_type;
  int store_tagnum;
  long store_struct_offset; /* used to be int */
  int store_inherit_tagnum;
  long store_inherit_offset;
  struct G__ifunc_table *ifunc;
  int iexist,virtualtag;
  int store_def_struct_member;
  int store_var_typeB;
  int store_doingconstruction;
  int store_func_now;
  int store_func_page;
  int store_iscpp;
  int store_exec_memberfunc;
  G__UINT32 store_security;

#ifdef G__ASM_IFUNC
  long asm_inst_g[G__MAXINST]; /* p-code instruction buffer */
  G__value asm_stack_g[G__MAXSTACK]; /* data stack */
  char asm_name[G__ASM_FUNCNAMEBUF];

  long *store_asm_inst;
  int store_asm_instsize;
  G__value *store_asm_stack;
  char *store_asm_name;
  int store_asm_name_p;
  struct G__param *store_asm_param;
  int store_asm_exec;
  int store_asm_noverflow;
  int store_asm_cp;
  int store_asm_dt;
  int store_asm_index; /* maybe unneccessary */
#endif
#ifdef G__ASM_WHOLEFUNC
  int store_no_exec_compile=0;
  struct G__var_array *localvar=NULL;
#endif
#ifdef G__NEWINHERIT
  int basen=0;
  int isbase;
  int access;
  int memfunc_or_friend=0;
  struct G__inheritance *baseclass=NULL;
#endif
/* #define G__OLDIMPLEMENTATION590 */
  int local_tagnum=0;
  struct G__ifunc_table *store_p_ifunc=p_ifunc;
  int specialflag=0;
  G__value *store_p_tempobject=0;
  int store_memberfunc_struct_offset;
  int store_memberfunc_tagnum;

#ifndef G__OLDIMPLEMENTATION2038
  G_local.enclosing_scope = (struct G__var_array*)NULL;
  G_local.inner_scope = (struct G__var_array**)NULL;
#endif

#ifdef G__NEWINHERIT
  store_inherit_offset = G__store_struct_offset;
  store_inherit_tagnum = G__tagnum;
#endif
  store_asm_noverflow = G__asm_noverflow;
  
#ifdef G__ASM_IFUNC
  if(G__asm_exec) {
    ifn = G__asm_index;
    /* delete 0 ~destructor ignored */
    if(0==G__store_struct_offset && -1!=p_ifunc->tagnum && 
       0==p_ifunc->staticalloc[ifn] && '~'==p_ifunc->funcname[ifn][0]) {
      return(1);
    }
    goto asm_ifunc_start;
  }
#endif
  
  
  /*******************************************************
   * searching function 
   *******************************************************/
#ifdef G__NEWINHERIT
  if((G__exec_memberfunc&&(-1!=G__tagnum||-1!=G__memberfunc_tagnum)) ||
      G__TRYNORMAL!=memfunc_flag) {
    isbase=1;
    basen=0;
    if(G__exec_memberfunc&&-1==G__tagnum) local_tagnum=G__memberfunc_tagnum;
    else                                  local_tagnum=G__tagnum;
    baseclass = G__struct.baseclass[local_tagnum];
    if(G__exec_memberfunc || G__isfriend(G__tagnum)) {
      access = G__PUBLIC_PROTECTED_PRIVATE ;
      memfunc_or_friend = 1;
    }
    else {
      access = G__PUBLIC;
      memfunc_or_friend = 0;
    }
  }
  else {
    access = G__PUBLIC;
    isbase=0;
    if (p_ifunc && p_ifunc == G__p_ifunc) {
      basen=0;
      isbase = 1;
      baseclass = &G__globalusingnamespace;
    }
  }
 next_base:
#endif

  /* FROM HERE */

  p_ifunc = G__overload_match(funcname,libp,hash,p_ifunc,memfunc_flag
                              ,access,&ifn,0
                              ,1
                              );
  /* error */
  if(-1==ifn) {
    *result7 = G__null;
    return(1);
  }


  /* TO HERE */

#ifdef G__NEWINHERIT
  /**********************************************************************
   * iteration for base class member function search
   **********************************************************************/
  if(p_ifunc==NULL || 
     (G__PUBLIC!=p_ifunc->access[ifn] && !G__isfriend(G__tagnum)
      && (0==G__exec_memberfunc || 
          (local_tagnum!=G__memberfunc_tagnum 
           && (G__PROTECTED!=p_ifunc->access[ifn]
               || -1==G__ispublicbase(local_tagnum,G__memberfunc_tagnum
                                      ,store_inherit_offset))
          )))) {
    if(isbase) {
      while(baseclass && basen<baseclass->basen) {
        if(memfunc_or_friend) {
          if((baseclass->baseaccess[basen]&G__PUBLIC_PROTECTED) ||
             baseclass->property[basen]&G__ISDIRECTINHERIT) {
            access = G__PUBLIC_PROTECTED;
            G__incsetup_memfunc(baseclass->basetagnum[basen]);
            p_ifunc = G__struct.memfunc[baseclass->basetagnum[basen]];
#ifdef G__VIRTUALBASE
            if(baseclass->property[basen]&G__ISVIRTUALBASE) {
              G__store_struct_offset = store_inherit_offset + 
                G__getvirtualbaseoffset(store_inherit_offset,G__tagnum
                                        ,baseclass,basen);
              if(G__cintv6) {
                G__bc_VIRTUALADDSTROS(G__tagnum,baseclass,basen);
              }
            }
            else {
              G__store_struct_offset
                = store_inherit_offset + baseclass->baseoffset[basen];
            }
#else
            G__store_struct_offset
              = store_inherit_offset + baseclass->baseoffset[basen];
#endif
            G__tagnum = baseclass->basetagnum[basen];
            ++basen;
            store_p_ifunc=p_ifunc;
            goto next_base; /* I know this is a bad manner */
          }
        }
        else {
          if(baseclass->baseaccess[basen]&G__PUBLIC) {
            access = G__PUBLIC;
            G__incsetup_memfunc(baseclass->basetagnum[basen]);
            p_ifunc = G__struct.memfunc[baseclass->basetagnum[basen]];
#ifdef G__VIRTUALBASE
            if(baseclass->property[basen]&G__ISVIRTUALBASE) {
              G__store_struct_offset = store_inherit_offset + 
                G__getvirtualbaseoffset(store_inherit_offset,G__tagnum
                                        ,baseclass,basen);
              if(G__cintv6) {
                G__bc_VIRTUALADDSTROS(G__tagnum,baseclass,basen);
              }
            }
            else {
              G__store_struct_offset
                = store_inherit_offset + baseclass->baseoffset[basen];
            }
#else
            G__store_struct_offset
              = store_inherit_offset + baseclass->baseoffset[basen];
#endif
            G__tagnum = baseclass->basetagnum[basen];
            ++basen;
            store_p_ifunc=p_ifunc;
            goto next_base; /* I know this is a bad manner */
          }
        }
        ++basen;
      }
      isbase=0;
    }

    if(0==specialflag && 1==libp->paran && -1!=libp->para[0].tagnum &&
       -1!=G__struct.parent_tagnum[libp->para[0].tagnum]) {
      p_ifunc =
        G__struct.memfunc[G__struct.parent_tagnum[libp->para[0].tagnum]];
      switch(G__struct.type[p_ifunc->tagnum]) {
      case 's':
      case 'c':
        store_p_ifunc=p_ifunc;
        specialflag = 1;
        goto next_base;
      }
    }

    /* not found */
    G__store_struct_offset = store_inherit_offset ;
    G__tagnum = store_inherit_tagnum ;
    G__asm_noverflow = store_asm_noverflow;
    return(0);
  }
#else
  /******************************************************************
   * if no such func, return 0
   *******************************************************************/
  if(p_ifunc==NULL) {
    return(0);
  }

  /******************************************************************
   * member access control
   *******************************************************************/
  if(G__PUBLIC!=p_ifunc->access[ifn] && !G__isfriend(G__tagnum)) {
    return(0);
  }
#endif

asm_ifunc_start:   /* loop compilation execution label */

  if(0==p_ifunc->hash[ifn]) return(0);

  /******************************************************************
   * Constructor or destructor call in G__make_ifunctable() parameter
   * type allocation. Return without call.
   * Also, when parameter analysis with -c optoin, return without call.
   *******************************************************************/
  if(G__globalcomp) { /* with -c-1 or -c-2 option */
    result7->obj.d = 0.0;
    result7->ref = 0;
    result7->type=p_ifunc->type[ifn];
    result7->tagnum=p_ifunc->p_tagtable[ifn];
    result7->typenum=p_ifunc->p_typetable[ifn];
#ifndef G__OLDIMPLEMENTATION1259
    result7->isconst = p_ifunc->isconst[ifn];
#endif
    if(isupper(result7->type)) {
      result7->obj.reftype.reftype = p_ifunc->reftype[ifn];
    }
    return(1);
  }
  else if(G__prerun) { /* in G__make_ifunctable parameter allocation */
    result7->obj.i = p_ifunc->type[ifn];
    result7->ref = 0;
    result7->type= G__DEFAULT_FUNCCALL;
    result7->tagnum=p_ifunc->p_tagtable[ifn];
    result7->typenum=p_ifunc->p_typetable[ifn];
#ifndef G__OLDIMPLEMENTATION1259
    result7->isconst = p_ifunc->isconst[ifn];
#endif
    if(isupper(result7->type)) {
      result7->obj.reftype.reftype = p_ifunc->reftype[ifn];
    }
    return(1);
  }

  /******************************************************************
   * error if body not defined
   *******************************************************************/
#ifdef G__ASM_WHOLEFUNC
  if((FILE*)NULL==(FILE*)p_ifunc->pentry[ifn]->p 
     && 0==p_ifunc->ispurevirtual[ifn] 
     && G__ASM_FUNC_NOP==G__asm_wholefunction
     && p_ifunc->hash[ifn]
     ) {
#else
  if((FILE*)NULL==(FILE*)p_ifunc->pentry[ifn]->p 
     && 0==p_ifunc->ispurevirtual[ifn]) {
#endif
    {
      if(0==G__templatefunc(result7,funcname,libp,hash,funcmatch)) {
        if(G__USERCONV==funcmatch) {
          *result7 = G__null;
          G__fprinterr(G__serr,"Error: %s() header declared but not defined"
                  ,funcname);
          G__genericerror((char*)NULL);
          return(1);
        }
        else return(0);
      }
      return(1);
    }
  }

  
  /******************************************************************
   * function was found in interpreted function list
   *******************************************************************/
  /* p_ifunc has found */
  
  /******************************************************************
   * Add baseoffset if calling base class member function.
   * Resolution of virtual function is not done here. There is a
   * separate section down below. Search string 'virtual function'
   * to get there.
   *******************************************************************/
  G__tagnum = p_ifunc->tagnum;
  store_exec_memberfunc=G__exec_memberfunc;
  if(-1==G__tagnum&&-1==G__memberfunc_tagnum) G__exec_memberfunc=0;

#define G__OLDIMPLEMENTATION1101
#ifndef G__OLDIMPLEMENTATION1101
  if(memfunc_flag==G__CALLSTATICMEMFUNC && 0==G__store_struct_offset &&
     -1!=G__tagnum && 0==p_ifunc->staticalloc[ifn] && 0==G__no_exec) {
    G__fprinterr(G__serr,"Error: %s() Illegal non-static member function call"
            ,funcname);
    G__genericerror((char*)NULL);
    *result7 = G__null;
    return(1);
  }
#endif
  
  store_var_typeB = G__var_typeB;
  G__var_typeB='p';

#ifdef G__NEWINHERIT
#ifdef G__ASM
  if(G__asm_noverflow && G__store_struct_offset && 
     G__store_struct_offset!=store_inherit_offset) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"%3x: ADDSTROS %ld\n"
              ,G__asm_cp,G__store_struct_offset-store_inherit_offset);
    }
#endif
    G__asm_inst[G__asm_cp]=G__ADDSTROS;
    G__asm_inst[G__asm_cp+1]=G__store_struct_offset-store_inherit_offset;
    G__inc_cp_asm(2,0);
  }
#endif
#endif

  
  /******************************************************************
   * C++ compiled function
   *******************************************************************/
  if(
     -1 == p_ifunc->pentry[ifn]->size
     ) {

#ifdef G__ROOT
    if(memfunc_flag==G__CALLCONSTRUCTOR ||
       memfunc_flag==G__TRYCONSTRUCTOR  ||
       memfunc_flag==G__TRYIMPLICITCONSTRUCTOR) {
      G__exec_alloc_lock();
#ifdef G__ASM
      if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: ROOTOBJALLOCBEGIN\n" ,G__asm_cp);
#endif
	G__asm_inst[G__asm_cp]=G__ROOTOBJALLOCBEGIN;
	G__inc_cp_asm(1,0);
      }
#endif
      }
#endif
    G__call_cppfunc(result7,libp,p_ifunc,ifn);
#ifdef G__ROOT
    if(memfunc_flag==G__CALLCONSTRUCTOR ||
       memfunc_flag==G__TRYCONSTRUCTOR  ||
       memfunc_flag==G__TRYIMPLICITCONSTRUCTOR) {
      G__exec_alloc_unlock();
#ifdef G__ASM
      if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: ROOTOBJALLOCEND\n" ,G__asm_cp);
#endif
	G__asm_inst[G__asm_cp]=G__ROOTOBJALLOCEND;
	G__inc_cp_asm(1,0);
      }
#endif
    }
#endif

    /* recover tag environment */
    G__store_struct_offset = store_inherit_offset ;
    G__tagnum = store_inherit_tagnum ;
    if(-1!=G__tagnum) {
      G__incsetup_memvar(G__tagnum);
      if(-1!=G__struct.virtual_offset[G__tagnum] && 
         strcmp(funcname,G__struct.name[G__tagnum])==0) {
        long *pvtag
          = (long*)(result7->obj.i+G__struct.virtual_offset[G__tagnum]);
        *pvtag = G__tagnum;
      }
    }
    if('P'==store_var_typeB) G__val2pointer(result7);
#ifdef G__NEWINHERIT
#ifdef G__ASM
    if(G__asm_noverflow && G__store_struct_offset && 
       G__store_struct_offset!=store_inherit_offset) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) 
        G__fprinterr(G__serr,"%3x: ADDSTROS %ld\n"
                ,G__asm_cp, -G__store_struct_offset+store_inherit_offset);
#endif
      G__asm_inst[G__asm_cp]=G__ADDSTROS;
      G__asm_inst[G__asm_cp+1]= -G__store_struct_offset+store_inherit_offset;
      G__inc_cp_asm(2,0);
    }
#endif /* ASM */
#endif /* NEWINHERIT */
    G__exec_memberfunc=store_exec_memberfunc;
    return(1);
  }

  
#ifdef G__ASM
  /******************************************************************
   * create bytecode instruction for calling interpreted function
   *******************************************************************/
  if(G__asm_noverflow) {
    if(G__cintv6) {
      if(p_ifunc->isvirtual[ifn] && !G__fixedscope) {
#ifdef G__ASM_DBG
        if(G__asm_dbg) G__fprinterr(G__serr,"%3x: LD_FUNC virtual %s paran=%d\n"
                               ,G__asm_cp,funcname,libp->paran);
#endif
        G__asm_inst[G__asm_cp]=G__LD_FUNC;
        G__asm_inst[G__asm_cp+1]=p_ifunc->tagnum; /* ??? */
        G__asm_inst[G__asm_cp+2]=(p_ifunc->vtblindex[ifn]&0xffff)
                                 +(p_ifunc->vtblbasetagnum[ifn]*0x10000);
        G__asm_inst[G__asm_cp+3]=libp->paran;
        G__asm_inst[G__asm_cp+4]=(long)G__bc_exec_virtual_bytecode;
        G__inc_cp_asm(5,0);
      }
      else {
#ifdef G__ASM_DBG
        if(G__asm_dbg) G__fprinterr(G__serr,"%3x: LD_FUNC %s paran=%d\n"
                               ,G__asm_cp,funcname,libp->paran);
#endif
        G__asm_inst[G__asm_cp]=G__LD_FUNC;
        G__asm_inst[G__asm_cp+1]=(long)p_ifunc;
        G__asm_inst[G__asm_cp+2]=ifn;
        G__asm_inst[G__asm_cp+3]=libp->paran;
        if(-1!=p_ifunc->tagnum && strcmp(funcname,G__struct.name[p_ifunc->tagnum])==0) {
#ifndef G__OOLDIMPLEMENTATION2150
          G__bc_Baseclassctor_vbase(p_ifunc->tagnum);
#endif
          G__asm_inst[G__asm_cp+4]=(long)G__bc_exec_ctor_bytecode;
        }
        else {
          G__asm_inst[G__asm_cp+4]=(long)G__bc_exec_normal_bytecode;
        }
        G__inc_cp_asm(5,0);
      }
    }
    else {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: LD_IFUNC %s paran=%d\n"
                             ,G__asm_cp,funcname,libp->paran);
#endif
      G__asm_inst[G__asm_cp]=G__LD_IFUNC;
      G__asm_inst[G__asm_cp+1]=(long)p_ifunc->funcname[ifn];
      G__asm_inst[G__asm_cp+2]=hash;
      G__asm_inst[G__asm_cp+3]=libp->paran;
      G__asm_inst[G__asm_cp+4]=(long)p_ifunc;
      G__asm_inst[G__asm_cp+5]=(long)funcmatch;
      G__asm_inst[G__asm_cp+6]=(long)memfunc_flag;
      G__asm_inst[G__asm_cp+7]=(long)ifn;
      G__inc_cp_asm(8,0);
    }
    if(G__store_struct_offset && G__store_struct_offset!=store_inherit_offset){
#ifdef G__ASM_DBG
      if(G__asm_dbg) 
        G__fprinterr(G__serr,"%3x: ADDSTROS %ld\n"
                ,G__asm_cp, -G__store_struct_offset+store_inherit_offset);
#endif
      G__asm_inst[G__asm_cp]=G__ADDSTROS;
      G__asm_inst[G__asm_cp+1]= -G__store_struct_offset+store_inherit_offset;
      G__inc_cp_asm(2,0);
    }
  }
#endif /* G__ASM */

  /* G__oprovld is set when calling operator overload function after 
   * evaluating its' argument to avoid duplication in p-code stack data.
   * This must be reset when calling lower level interpreted function */
  G__oprovld=0;

#ifdef G__ASM
  if(G__no_exec_compile) {
    G__store_struct_offset = store_inherit_offset ;
    G__tagnum = store_inherit_tagnum ;
    result7->tagnum = p_ifunc->p_tagtable[ifn];
    if(-1!=result7->tagnum && 'e'!=G__struct.type[result7->tagnum]) {
      if(isupper(p_ifunc->type[ifn])) result7->type='U';
      else                            result7->type = 'u';
    }
    else {
      result7->type = p_ifunc->type[ifn];
    }
    result7->typenum = p_ifunc->p_typetable[ifn];
    if(-1!=result7->tagnum) result7->ref = 1;
    else result7->ref = 0;
#ifndef G__OLDIMPLEMENTATION1259
    result7->isconst = p_ifunc->isconst[ifn];
#endif
    result7->obj.d = 0.0;
    result7->obj.i = 1;
    if(isupper(result7->type)) {
      result7->obj.reftype.reftype = p_ifunc->reftype[ifn];
    }
    result7->ref = p_ifunc->reftype[ifn];
    if('u'==p_ifunc->type[ifn]&&0==result7->ref&&-1!=result7->tagnum) {
      G__store_tempobject(*result7); /* To free tempobject in pcode */
    }
    /* To be implemented */
    G__exec_memberfunc=store_exec_memberfunc;
    return(1);
  }
#endif /* G__ASM */

  /******************************************************************
   * virtual function
   *  If virtual function flag is set, get actual tag identity by
   * taginfo member at offset of G__struct.virtual_offset[].
   * Then search for virtual function in actual tag. If found, 
   * change p_ifunc,ifn,G__store_struct_offset and G__tagnum.
   * G__store_struct_offset and G__tagnum are already stored above,
   * so no need to store it to temporary here.
   *******************************************************************/
  if(p_ifunc->isvirtual[ifn] && !G__fixedscope) {
    if(-1!=G__struct.virtual_offset[G__tagnum])
      virtualtag= *(long*)(G__store_struct_offset /* NEED TO CHECK THIS PART */
                           +G__struct.virtual_offset[G__tagnum]);
    else {
      virtualtag = G__tagnum;
    }
    if(virtualtag!=G__tagnum) {
      struct G__inheritance *baseclass = G__struct.baseclass[virtualtag];
      int xbase[G__MAXBASE],ybase[G__MAXBASE];
      int nxbase=0,nybase;
      int basen;
      G__incsetup_memfunc(virtualtag);
      ifunc=G__ifunc_exist(p_ifunc,ifn,G__struct.memfunc[virtualtag],&iexist
                           ,0xffff);
      for(basen=0;!ifunc&&basen<baseclass->basen;basen++) {
        virtualtag = baseclass->basetagnum[basen];
        if(0==(baseclass->property[basen]&G__ISDIRECTINHERIT)) continue;
        xbase[nxbase++] = virtualtag;
        G__incsetup_memfunc(virtualtag);
        ifunc
          =G__ifunc_exist(p_ifunc,ifn,G__struct.memfunc[virtualtag],&iexist
                          ,0xffff);
      } 
      while(!ifunc && nxbase) {
        int xxx;
        nybase=0;
        for(xxx=0;!ifunc&&xxx<nxbase;xxx++) {
          baseclass = G__struct.baseclass[xbase[xxx]];
          for(basen=0;!ifunc&&basen<baseclass->basen;basen++) {
            virtualtag = baseclass->basetagnum[basen];
            if(0==(baseclass->property[basen]&G__ISDIRECTINHERIT)) continue;
            ybase[nybase++] = virtualtag;
            G__incsetup_memfunc(virtualtag);
            ifunc
              =G__ifunc_exist(p_ifunc,ifn,G__struct.memfunc[virtualtag]
                              ,&iexist,0xffff);
          } 
        } 
        nxbase=nybase;
        memcpy((void*)xbase,(void*)ybase,sizeof(int)*nybase);
      }
      if(ifunc) {
        if((FILE*)NULL==(FILE*)ifunc->pentry[iexist]->p) {
          G__fprinterr(G__serr,"Error: virtual %s() header found but not defined",funcname);
          G__genericerror((char*)NULL);
          G__exec_memberfunc=store_exec_memberfunc;
          return(1);
        }
        p_ifunc=ifunc;
        ifn=iexist;
        G__store_struct_offset -= G__find_virtualoffset(virtualtag);
        G__tagnum=virtualtag;
        if('~'==funcname[0]) {
          strcpy(funcname+1,G__struct.name[G__tagnum]);
          G__hash(funcname,hash,itemp);
        }
      }
      else if(p_ifunc->ispurevirtual[ifn]) {
        G__fprinterr(G__serr,"Error: pure virtual %s() not defined",funcname);
        G__genericerror((char*)NULL);
        G__exec_memberfunc=store_exec_memberfunc;
        return(1);
      }
    }
  }

  if(G__cintv6 && G__BYTECODE_NOTYET==p_ifunc->pentry[ifn]->bytecodestatus) {
    if(G__BYTECODE_FAILURE==G__bc_compile_function(p_ifunc,ifn)) {
      G__exec_memberfunc=store_exec_memberfunc;
      return(1);
    }
  }

#ifdef G__ASM
#ifdef G__ASM_WHOLEFUNC
  /******************************************************************
   * try bytecode compilation
   *******************************************************************/
  if(G__BYTECODE_NOTYET==p_ifunc->pentry[ifn]->bytecodestatus &&
     G__asm_loopcompile>3 && G__ASM_FUNC_NOP==G__asm_wholefunction && 
#ifndef G__TO_BE_DELETED
     G__CALLCONSTRUCTOR!=memfunc_flag && G__TRYCONSTRUCTOR!=memfunc_flag &&
     G__TRYIMPLICITCONSTRUCTOR!=memfunc_flag && 
     G__TRYDESTRUCTOR!=memfunc_flag && 
#endif
     0==G__step && (G__asm_noverflow||G__asm_exec
                    ||G__asm_loopcompile>4
                    )) {
    G__compile_bytecode(p_ifunc,ifn);
  }
  /******************************************************************
   * if already compiled as bytecode run bytecode
   *******************************************************************/
  if(p_ifunc->pentry[ifn]->bytecode
     && G__BYTECODE_ANALYSIS!=p_ifunc->pentry[ifn]->bytecodestatus
     ) {
    struct G__input_file store_ifile;
    store_ifile=G__ifile;
    G__ifile.filenum=p_ifunc->pentry[ifn]->filenum;
    G__ifile.line_number=p_ifunc->pentry[ifn]->line_number;
    G__exec_bytecode(result7,(char*)p_ifunc->pentry[ifn]->bytecode,libp,hash);
    G__ifile=store_ifile;
    G__tagnum = store_inherit_tagnum;
    G__store_struct_offset = store_inherit_offset;
    return(1);
  }
#endif /* G__ASM_WHOLEFUNC */
#endif /* G__ASM */

#ifndef G__OLDIMPLEMENTATION1167
  G__reftypeparam(p_ifunc,ifn,libp);
#endif

#ifdef G__ASM
#ifdef G__ASM_IFUNC
  /******************************************************************
   * push bytecode environment stack
   *******************************************************************/
  /*
  if(G__asm_noverflow) G__asm_inst[G__asm_cp+1]=(long)p_ifunc->funcname[ifn];
  */
  /* Push loop compilation environment */
  store_asm_inst = G__asm_inst;
  store_asm_stack = G__asm_stack;
  store_asm_name = G__asm_name;
  store_asm_name_p = G__asm_name_p;
  store_asm_param  = G__asm_param ;
  store_asm_exec  = G__asm_exec ;
  store_asm_noverflow  = G__asm_noverflow ;
  store_asm_cp  = G__asm_cp ;
  store_asm_dt  = G__asm_dt ;
  store_asm_index  = G__asm_index ;

  store_asm_instsize = G__asm_instsize;
  G__asm_instsize = 0; /* G__asm_inst is not resizable */

  G__asm_inst = asm_inst_g;
  G__asm_stack = asm_stack_g;
  G__asm_name = asm_name;
  G__asm_name_p = 0;
  /* G__asm_param ; */
  G__asm_exec = 0 ;
#endif /* G__ASM_IFUNC */
#endif /* G__ASM */

#ifdef G__ASM
#ifdef G__ASM_IFUNC
#ifdef G__ASM_WHOLEFUNC
  /******************************************************************
   * bytecode function compilation start
   *******************************************************************/
  if(G__ASM_FUNC_COMPILE&G__asm_wholefunction) {
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"!!!bytecode compilation %s start"
              ,p_ifunc->funcname[ifn]);
      G__printlinenum();
    }
    G__asm_name = (char*)malloc(G__ASM_FUNCNAMEBUF);
    G__asm_noverflow = 1;
    store_no_exec_compile = G__no_exec_compile;
    G__no_exec_compile = 1;
    localvar = (struct G__var_array*)malloc(sizeof(struct G__var_array));
#ifdef G__OLDIMPLEMENTATION1776_YET
    memset(localvar,0,sizeof(struct G__var_array));
#endif
#ifndef G__OLDIMPLEMENTATION2038
    localvar->enclosing_scope = (struct G__var_array*)NULL;
    localvar->inner_scope = (struct G__var_array**)NULL;
#endif

    localvar->prev_local = G__p_local;
    localvar->ifunc = p_ifunc;
    localvar->ifn = ifn;
#ifdef G__VAARG
    localvar->libp = libp;
#endif
    localvar->tagnum=G__tagnum;
    localvar->struct_offset=G__store_struct_offset;
    localvar->exec_memberfunc=G__exec_memberfunc;
    localvar->allvar=0;
    localvar->varlabel[0][0]=0;
    localvar->next=NULL;
    localvar->prev_filenum = G__ifile.filenum;
    localvar->prev_line_number = G__ifile.line_number;
    { 
      int ix;
      for(ix=0;ix<G__MEMDEPTH;ix++) {
        localvar->varnamebuf[ix]=(char*)NULL;
        localvar->p[ix] = 0;
      }
    }
  }
  else {
    G__asm_noverflow = 0;
  }
#else /* G__ASM_WHOLEFUNC */
  G__asm_noverflow = 0;
#endif /* G__ASM_WHOLEFUNC */
  G__asm_cp = 0;
  G__asm_dt = G__MAXSTACK-1;
  /* G__asm_index  ; */
#endif /* G__ASM_IFUNC */
#endif /* G__ASM */

  

  /******************************************************************
   * G__exec_memberfunc and G__memberfunc_tagnum are stored in one 
   * upper level G__getfunction() and G__parenthesisovld() and restored 
   * when exit from these functions.
   *******************************************************************/
  if(-1==p_ifunc->tagnum) G__exec_memberfunc=0;
  else                    G__exec_memberfunc=1;
  G__setclassdebugcond(G__tagnum,0);
  store_memberfunc_tagnum = G__memberfunc_tagnum;
  store_memberfunc_struct_offset = G__memberfunc_struct_offset;
  G__memberfunc_tagnum = G__tagnum;
  G__memberfunc_struct_offset=G__store_struct_offset;
  
  /**********************************************
   * If return value is struct,class,union, 
   * create temp object buffer
   **********************************************/
  if(p_ifunc->type[ifn]=='u' && p_ifunc->reftype[ifn]==G__PARANORMAL
     && G__CPPLINK!=G__struct.iscpplink[p_ifunc->p_tagtable[ifn]]
     && 'e'!=G__struct.type[p_ifunc->p_tagtable[ifn]]
     ) {
    /* create temp object buffer */
    
    G__alloc_tempobject(p_ifunc->p_tagtable[ifn] ,p_ifunc->p_typetable[ifn]);
    store_p_tempobject = &G__p_tempbuf->obj;
    
    if(G__dispsource) {
      G__fprinterr(G__serr,"!!!Create temp object (%s)0x%lx,%d for %s() return\n"
              ,G__struct.name[p_ifunc->p_tagtable[ifn]]
              ,G__p_tempbuf->obj.obj.i ,G__templevel ,p_ifunc->funcname[ifn]);
    }
  }
  
  /**********************************************
   * increment busy flag
   **********************************************/
  p_ifunc->busy[ifn]++;
  
  /**********************************************
   * set global variable G__func_now. This is
   * used in G__malloc() to allocate static
   * variable.
   **********************************************/
  store_func_now=G__func_now;
  store_func_page = G__func_page;
  G__func_now=ifn;
  G__func_page=p_ifunc->page;
  
  /* store old local to prev buffer and allocate new local variable */
#ifdef G__OLDIMPLEMENTATION1776_YET
  memset(&G_local,0,sizeof(struct G__var_array));    
#endif
  G_local.prev_local = G__p_local;
  G_local.ifunc = p_ifunc;
  G_local.ifn = ifn;
#ifdef G__VAARG
  G_local.libp = libp;
#endif
  G_local.tagnum=G__tagnum;
  G_local.struct_offset=G__store_struct_offset;
  G_local.exec_memberfunc=G__exec_memberfunc;
  {
    int ix;
    for(ix=0;ix<G__MEMDEPTH;ix++) {
      G_local.varnamebuf[ix] = (char*)NULL;
      G_local.p[ix] = 0;
    }
  }
#ifdef G__ASM_WHOLEFUNC
  if(G__ASM_FUNC_COMPILE&G__asm_wholefunction) G__p_local = localvar;
  else                     G__p_local = &G_local;
#else
  G__p_local = &G_local;
#endif
  
  break_exit_func=G__break_exit_func;
  
  G__break_exit_func=0;
  G__p_local->allvar=0;
  G__p_local->varlabel[0][0]=0;
  G__p_local->next=NULL;
  
  /* store line number and filename*/
  G_local.prev_filenum = G__ifile.filenum;
  G_local.prev_line_number = G__ifile.line_number;
  G__ifile.line_number = p_ifunc->pentry[ifn]->line_number;
  strcpy(G__ifile.name,G__srcfile[p_ifunc->pentry[ifn]->filenum].filename);
  G__ifile.filenum=p_ifunc->pentry[ifn]->filenum;

  /* check breakpoint */
  if(0==G__nobreak && 0==G__no_exec_compile &&
     G__srcfile[G__ifile.filenum].breakpoint &&
     G__srcfile[G__ifile.filenum].maxline>G__ifile.line_number &&
     G__TESTBREAK&(G__srcfile[G__ifile.filenum].breakpoint[G__ifile.line_number]|=G__TRACED)) {
    G__BREAKfgetc();
  }

  store_security = G__security;
  G__security = G__srcfile[G__ifile.filenum].security;

  /* store file pointer and fpos*/
  /* store_linenumber=G__ifile.line_number; */
  if(G__ifile.fp) fgetpos(G__ifile.fp,&prev_pos);
  prev_fp = G__ifile.fp ;
  
  /* Find the right file pointer */
  if( G__mfp && (FILE*)p_ifunc->pentry[ifn]->p == G__mfp ) {
    /* In case of macro expanded by cint, we use the tmpfile */
    G__ifile.fp = (FILE*)p_ifunc->pentry[ifn]->p ;
  } 
  else if(G__srcfile[G__ifile.filenum].fp) {
    /* The file is already open use that */
    G__ifile.fp = G__srcfile[G__ifile.filenum].fp;
  } 
  else {
    /* The file had been closed, let's reopen the proper file
     * resp from the preprocessor and raw */
    if ( G__srcfile[G__ifile.filenum].prepname ) {
      G__ifile.fp = fopen(G__srcfile[G__ifile.filenum].prepname,"r");
    } else {
      G__ifile.fp = fopen(G__srcfile[G__ifile.filenum].filename,"r");
    }
    G__srcfile[G__ifile.filenum].fp =  G__ifile.fp;
    if(!G__ifile.fp) G__ifile.fp = (FILE*)p_ifunc->pentry[ifn]->p ;
  }
  fsetpos(G__ifile.fp,&p_ifunc->pentry[ifn]->pos);
  
  /* print function header if debug mode */
  
  if(G__dispsource) {
    G__disp_mask=0;
    if((G__debug||G__break||G__step
        ||(strcmp(G__breakfile,G__ifile.name)==0)||(strcmp(G__breakfile,"")==0)
        )&&((G__prerun!=0)||(G__no_exec==0))) {
      if(G__ifile.name&&G__ifile.name[0]) 
        G__fprinterr(G__serr,"\n# %s",G__ifile.name);
      if(-1!=p_ifunc->tagnum) {
        G__fprinterr(G__serr,"\n%-5d%s::%s(" ,G__ifile.line_number
                ,G__struct.name[p_ifunc->tagnum] ,funcname);
      }
      else {
        G__fprinterr(G__serr,"\n%-5d%s(" ,G__ifile.line_number,funcname);
      }
    }
  }

  
  /* now came to         func( para1,para2,,,)
   *                          ^
   */
  
  store_doingconstruction=G__doingconstruction;
  /**************************************************************
   * K&R C
   *
   *   func( para1 ,para2 ,,, )
   *        ^
   **************************************************************/
  if(p_ifunc->ansi[ifn]==0) {
    
    /* read pass parameters , standard C */
    ipara=0;
    while(cin!=')') {
#ifndef G__OLDIMPLEMENTATION1802
      char temp[G__ONELINE];
#endif
      cin=G__fgetstream(temp,",)");
      if(temp[0]!='\0') {
        strcpy(paraname[ipara],temp);
        ipara++;
      }
    }
    
    /* read and exec pass parameter declaration , standard C 
     * G__exec_statement() returns at '{' if G__funcheader==1
     */
    G__funcheader=1;
    G__mparen=0;
    do {
      buf=G__exec_statement();
    } while(buf.type==G__null.type && G__return<G__RETURN_EXIT1);
    
    /* set pass parameters , standard C 
     * Parameters can be constant. When G__funcheader==1,
     * error message of changing const doesn't appear.
     */
    
    for(itemp=0;itemp<ipara;itemp++) {
      G__letvariable(paraname[itemp],libp->para[itemp]
                     ,&G__global,G__p_local);
    }
    
    G__funcheader=0;
    /* fsetpos(G__ifile.fp,&temppos) ; */
    /* G__ifile.line_number=store_linenumber; */
    fseek(G__ifile.fp,-1,SEEK_CUR);
    if(G__dispsource) G__disp_mask=1;
  }
  
  /**************************************************************
   * ANSI C
   *
   *   type func( type para1, type para2 ,,, )
   *             ^
   **************************************************************/
  else {
    G__value store_ansipara;
    store_ansipara=G__ansipara;
    G__ansiheader=1;
    G__funcheader=1;
    
    ipara=0;
    while(G__ansiheader!=0 && G__return<G__RETURN_EXIT1) {
      /****************************************
       * for default parameter
       ****************************************/
      /****************************************
       * if parameter exists, set G__ansipara
       ****************************************/
      if(ipara<libp->paran) {
        G__ansipara=libp->para[ipara];
        /* assigning reference for fundamental type reference argument */
        if(0==G__ansipara.ref) {
          switch(p_ifunc->para_type[ifn][ipara]) {
          case 'f':
            G__Mfloat(libp->para[ipara]);
            G__ansipara.ref = (long)(&libp->para[ipara].obj.fl);
            break;
          case 'd':
            G__ansipara.ref = (long)(&libp->para[ipara].obj.d);
            break;
          case 'c':
            G__Mchar(libp->para[ipara]);
            G__ansipara.ref = (long)(&libp->para[ipara].obj.ch);
            break;
          case 's':
            G__Mshort(libp->para[ipara]);
            G__ansipara.ref = (long)(&libp->para[ipara].obj.sh);
            break;
          case 'i':
            G__Mint(libp->para[ipara]);
            G__ansipara.ref = (long)(&libp->para[ipara].obj.in);
            break;
          case 'l':
            G__ansipara.ref = (long)(&libp->para[ipara].obj.i);
            break;
          case 'b':
          case 'g':
#ifdef G__BOOL4BYTE
            G__Mint(libp->para[ipara]);
            G__ansipara.ref = (long)(&libp->para[ipara].obj.i);
#else
            G__Muchar(libp->para[ipara]);
            G__ansipara.ref = (long)(&libp->para[ipara].obj.uch);
#endif
            break;
          case 'r':
            G__Mushort(libp->para[ipara]);
            G__ansipara.ref = (long)(&libp->para[ipara].obj.ush);
            break;
          case 'h':
            /* G__Muint(libp->para[ipara]); */
            G__ansipara.ref = (long)(&libp->para[ipara].obj.i);
            break;
          case 'k':
            G__ansipara.ref = (long)(&libp->para[ipara].obj.i);
            break;
          case 'u':
            G__ansipara.ref = G__ansipara.obj.i;
            break;
          case 'n':
            G__ansipara.ref = (long)(&libp->para[ipara].obj.ll);
            break;
          case 'm':
            G__ansipara.ref = (long)(&libp->para[ipara].obj.ull);
            break;
          case 'q':
            G__ansipara.ref = (long)(&libp->para[ipara].obj.ld);
            break;
          default:
            G__ansipara.ref = (long)(&libp->para[ipara].obj.i);
            break;
          }
        }
      }
      /****************************************
       * if not, set null.
       * Default value will be used
       *  type func(type paraname=default,...)
       ****************************************/
      else {
        if(
           p_ifunc->para_nu[ifn]>ipara && 
           p_ifunc->para_default[ifn][ipara]) {
          if(p_ifunc->para_default[ifn][ipara]->type==G__DEFAULT_FUNCCALL) {
            G__ASSERT(p_ifunc->para_default[ifn][ipara]->ref);
            *p_ifunc->para_default[ifn][ipara] =
              G__getexpr((char*)p_ifunc->para_default[ifn][ipara]->ref);
            G__ansiheader=1;
            G__funcheader=1;
#define G__OLDIMPLEMENTATION1558 
          }
          G__ansipara = *p_ifunc->para_default[ifn][ipara];
        }
        else
          G__ansipara = G__null;
      }
      G__refansipara = libp->parameter[ipara];
      
      if(G__ASM_FUNC_COMPILE==G__asm_wholefunction &&
         p_ifunc->para_default[ifn][ipara]) {
#ifdef G__ASM_DBG
        if(G__asm_dbg) {
          G__fprinterr(G__serr,"%3x: ISDEFAULTPARA %x\n",G__asm_cp,G__asm_cp+4);
          G__fprinterr(G__serr,"%3x: LD %ld %g\n",G__asm_cp+2
                  ,p_ifunc->para_default[ifn][ipara]->obj.i
                  ,p_ifunc->para_default[ifn][ipara]->obj.d
                  );
        }
#endif
        G__asm_inst[G__asm_cp] = G__ISDEFAULTPARA;
        G__asm_wholefunc_default_cp=G__asm_cp+1;
        G__inc_cp_asm(2,0);

        /* set default param in stack */
        G__asm_inst[G__asm_cp]=G__LD;
        G__asm_inst[G__asm_cp+1]=G__asm_dt;
        G__asm_stack[G__asm_dt] = *p_ifunc->para_default[ifn][ipara];
        G__inc_cp_asm(2,1);

        G__asm_inst[G__asm_wholefunc_default_cp]=G__asm_cp;
        G__suspendbytecode(); /* mask default param evaluation */
        G__exec_statement(); /* Create var entry and ST_LVAR inst */
        G__asm_wholefunc_default_cp=0;
        G__asm_noverflow=1;
      }
      else {
        G__exec_statement();
      }
      ipara++;
    }
    
    G__funcheader=0;
    
    switch(memfunc_flag) {
    case G__CALLCONSTRUCTOR:
    case G__TRYCONSTRUCTOR:
#ifndef G__OLDIMPLEMENTATIO1250
    case G__TRYIMPLICITCONSTRUCTOR:
#endif
      /* read parameters for base constructors and 
       * constructor for base calss and class members 
       * maybe with some parameters
       */
      G__baseconstructorwp();
      G__doingconstruction=1;
    }

    G__ansipara=store_ansipara;
  }

#ifdef G__SECURITY
  if((G__security&G__SECURE_STACK_DEPTH)&&
     G__max_stack_depth && 
     G__calldepth>G__max_stack_depth
     ) {
    G__fprinterr(G__serr,"Error: Stack depth exceeded %d",G__max_stack_depth);
    G__genericerror((char*)NULL);
    G__pause();
    G__return=G__RETURN_EXIT1;
  }

  if(G__return>G__RETURN_EXIT1) {
    G__exec_memberfunc=store_exec_memberfunc;
    G__security = store_security;
    G__memberfunc_tagnum = store_memberfunc_tagnum;
    G__memberfunc_struct_offset = store_memberfunc_struct_offset;
    return(1);
  }
#endif

  G__setclassdebugcond(G__memberfunc_tagnum,1);
  
  /**************************************************************
   * execute ifunction body 
   *
   * common to standard and ANSI
   **************************************************************/

  store_iscpp=G__iscpp;
  G__iscpp=p_ifunc->iscpp[ifn];

  ++G__templevel;
  ++G__calldepth;
#ifdef G__ASM_DBG
  if(G__istrace>1) {
    if(G__istrace>G__templevel) {
      G__debug = 1;
      G__asm_dbg = 1;
    }
    else {
      G__debug = 0;
      G__asm_dbg = 0;
    }
  }
#endif
  
  G__ASSERT(0==G__decl || 1==G__decl);
  store_def_struct_member = G__def_struct_member;
  G__def_struct_member = 0;
  store_decl=G__decl;
  G__decl=0;
  G__no_exec=0;        
  G__mparen=0;
  *result7=G__exec_statement();
  G__decl=store_decl;
  G__def_struct_member = store_def_struct_member;
  G__ASSERT(0==G__decl || 1==G__decl);

  if(G__RETURN_IMMEDIATE==G__return &&
     G__interactivereturnvalue.type && '\0'==result7->type) {
    *result7 = G__interactivereturnvalue;
    G__interactivereturnvalue = G__null;
  }

  --G__calldepth;
  --G__templevel;

#ifdef G__ASM_DBG
  if(G__istrace>1) {
    if(G__istrace>G__templevel) {
      G__debug = 1;
      G__asm_dbg = 1;
    }
    else {
      G__debug = 0;
      G__asm_dbg = 0;
    }
  }
#endif

  G__iscpp=(short)store_iscpp;
  
  G__doingconstruction=store_doingconstruction;

  /**************************************************************
   * Error if goto label not found
   **************************************************************/
  if(G__gotolabel[0]) {
    G__fprinterr(G__serr,"Error: Goto label '%s' not found in %s()"
            ,G__gotolabel,funcname);
    G__genericerror((char*)NULL);
    G__gotolabel[0]='\0';
  }
  
  /**************************************************************
   * return value type conversion
   *
   **************************************************************/
  if(
     0==G__xrefflag &&
     result7->type!='\0' && G__RETURN_EXIT1!=G__return
     && (G__ASM_FUNC_NOP==G__asm_wholefunction||G__asm_noverflow)
     ) {
    switch(p_ifunc->type[ifn]) {
      
      /***************************************************
       * in case of double and float.
       ***************************************************/
    case 'd': /* double */
    case 'f': /* float */
    case 'w': /* logic (original type) */
      G__letdouble(result7,p_ifunc->type[ifn] ,G__double(*result7));
#define G__OLDIMPLEMENTATION753
#ifdef G__OLDIMPLEMENTATION753
      if(p_ifunc->reftype[ifn]==G__PARANORMAL) result7->ref=0;
#endif
#ifndef G__OLDIMPLEMENTATION1259
      result7->isconst = p_ifunc->isconst[ifn];
#endif
      break;

    case 'n':
    case 'm':
      G__letLonglong(result7,p_ifunc->type[ifn],G__Longlong(*result7));
      if(p_ifunc->reftype[ifn]==G__PARANORMAL) result7->ref=0;
      result7->isconst = p_ifunc->isconst[ifn];
      break;

    case 'q':
      G__letLongdouble(result7,p_ifunc->type[ifn],G__Longdouble(*result7));
      if(p_ifunc->reftype[ifn]==G__PARANORMAL) result7->ref=0;
      result7->isconst = p_ifunc->isconst[ifn];
      break;

    case 'g':
      G__letint(result7,p_ifunc->type[ifn],G__int(*result7)?1:0);
#ifdef G__OLDIMPLEMENTATION753
      if(p_ifunc->reftype[ifn]==G__PARANORMAL) result7->ref=0;
#endif
      result7->isconst = p_ifunc->isconst[ifn];
      break;
      
      /***************************************************
       * in case of void, if return(); statement exists
       * it is illegal.
       * Maybe bug if return; statement exists without
       * return value.
       ***************************************************/
    case 'y': /* void */
      if(G__RETURN_NORMAL==G__return) {
        if(G__dispmsg>=G__DISPWARN) {
          G__fprinterr(G__serr,"Warning: Return value of void %s() ignored"
                       ,p_ifunc->funcname[ifn]);
          G__printlinenum();
          /* G__genericerror((char*)NULL); */
        }
      }
      *result7 = G__null ;
      break;
      
      /***************************************************
       * result7 contains pointer to the local variable
       * which will be destroyed right after this.
       ***************************************************/
    case 'u': /* struct, union, class */
      if(
         1
         ) {
#ifndef G__OLDIMPLEMENTATION1802
        char temp[G__ONELINE];
#endif
        
        /* don't call copy constructor if returning reference type */
        if(G__PARANORMAL!=p_ifunc->reftype[ifn]) {
          if(p_ifunc->p_tagtable[ifn]!=result7->tagnum) {
            int offset = G__ispublicbase(p_ifunc->p_tagtable[ifn]
                                         ,result7->tagnum,result7->obj.i);
            if(-1==offset) {
              G__fprinterr(G__serr,"Error: Return type mismatch. %s "
                           ,G__fulltagname(p_ifunc->p_tagtable[ifn],1));
              G__fprinterr(G__serr,"not a public base of %s"
                           ,G__fulltagname(result7->tagnum,1));
              G__genericerror((char*)NULL);
              result7->tagnum = p_ifunc->p_tagtable[ifn];
              break;
            }
            else {
              result7->obj.i += offset;
              if(result7->ref) result7->ref += offset;
              result7->tagnum = p_ifunc->p_tagtable[ifn];
            }
          }
          break;
        }
        if('e'==G__struct.type[p_ifunc->p_tagtable[ifn]]) {
          break;
        }
        
        if(result7->type=='u' || (result7->type=='i'&& -1!=result7->tagnum)){
          if(result7->obj.i<0)
            sprintf(temp,"%s((%s)(%ld))",G__struct.name[p_ifunc->p_tagtable[ifn]]
                    ,G__fulltagname(result7->tagnum,1) ,result7->obj.i);
          else
            sprintf(temp,"%s((%s)%ld)",G__struct.name[p_ifunc->p_tagtable[ifn]]
                    ,G__fulltagname(result7->tagnum,1) ,result7->obj.i);
        }
        else {
          char buf2[G__ONELINE];
          G__valuemonitor(*result7,buf2);
          sprintf(temp,"%s(%s)",G__struct.name[p_ifunc->p_tagtable[ifn]]
                  ,buf2);
        }
        
        store_tagnum = G__tagnum;
        G__tagnum = p_ifunc->p_tagtable[ifn];
        store_var_type = G__var_type; /* bug fix */

          
#ifdef G__SECURITY
        G__castcheckoff = 1;
#endif
        if(G__RETURN_IMMEDIATE>=G__return) G__return=G__RETURN_NON;
        itemp=0;
        store_struct_offset = G__store_struct_offset;

        if(G__CPPLINK!=G__struct.iscpplink[G__tagnum]) {
          /* interpreted class */
          if(store_p_tempobject) 
            G__store_struct_offset = store_p_tempobject->obj.i;
          else 
            G__store_struct_offset=G__p_tempbuf->obj.obj.i;
          if(G__dispsource) {
            G__fprinterr(G__serr,
            "\n!!!Calling copy/conversion constructor for return temp object 0x%lx.%s"
                    ,G__store_struct_offset
                    ,temp);
          }
          G__getfunction(temp,&itemp,G__TRYCONSTRUCTOR);
          if(itemp && 
             store_p_tempobject != (&G__p_tempbuf->obj) &&
             store_struct_offset != G__p_tempbuf->obj.obj.i) {
            ++G__p_tempbuf->level;
            ++G__templevel;
            G__free_tempobject();
            --G__templevel;
          }
        }
        else {
          /* precompiled class */
          long store_globalvarpointer = G__globalvarpointer;
          G__globalvarpointer = G__PVOID;
          G__store_struct_offset=0xffff;
          if(G__dispsource) {
            G__fprinterr(G__serr,
            "\n!!!Calling copy/conversion constructor for return temp object 0x%lx.%s"
                    ,G__store_struct_offset
                    ,temp);
          }
          buf=G__getfunction(temp,&itemp,G__TRYCONSTRUCTOR);
          G__globalvarpointer = store_globalvarpointer;
          if(itemp) {
            G__free_tempobject();
            G__store_tempobject(buf);
          }
#ifdef G__ASM
          /* It is not needed to explicitly create STORETEMP instruction
           * because it is preincluded in the compiled funciton call 
           * interface */
#endif
          if(G__dispsource) {
            G__fprinterr(G__serr,
                    "!!!Create temp object (%s)0x%lx,%d for %s() return\n"
                    ,G__struct.name[G__tagnum] ,G__p_tempbuf->obj.obj.i
                    ,G__templevel ,p_ifunc->funcname[ifn]);
          }
        }
        
        G__store_struct_offset = store_struct_offset;
        G__tagnum = store_tagnum;
        G__var_type = store_var_type; /* bug fix */
        
        /**************************************
         * if no copy constructor, memberwise
         * copy
         **************************************/
        if(itemp==0
           && 0==G__xrefflag
           ) {
          
          long offset=0;
          if(result7->tagnum == p_ifunc->p_tagtable[ifn]) {
            if(store_p_tempobject) 
              memcpy((void*)store_p_tempobject->obj.i
                     ,(void*)(result7->obj.i)
                     ,(size_t)G__struct.size[result7->tagnum]);
            else 
              memcpy((void*)G__p_tempbuf->obj.obj.i
                     ,(void*)(result7->obj.i)
                     ,(size_t)G__struct.size[result7->tagnum]);
          }
          else if(-1!=(offset=G__ispublicbase(p_ifunc->p_tagtable[ifn]
                                              ,result7->tagnum
                                              ,result7->obj.i))){
            sprintf(temp,"%s((%s)(%ld))"
                    ,G__struct.name[p_ifunc->p_tagtable[ifn]]
                    ,G__fulltagname(p_ifunc->p_tagtable[ifn],1)
                    ,result7->obj.i+offset);
            if(G__CPPLINK!=G__struct.iscpplink[G__tagnum]) {
              /* interpreted class */
              if(store_p_tempobject) 
                G__store_struct_offset=store_p_tempobject->obj.i;
              else 
                G__store_struct_offset=G__p_tempbuf->obj.obj.i;
              G__getfunction(temp,&itemp,G__TRYCONSTRUCTOR);
            }
            else {
              /* precompiled class */
              G__store_struct_offset=0xffff;
              buf=G__getfunction(temp,&itemp,G__TRYCONSTRUCTOR);
              if(itemp) G__store_tempobject(buf);
            }
          }
        }
        
        if(store_p_tempobject) *result7 = *store_p_tempobject;
        else                   *result7 = G__p_tempbuf->obj;
        
      }
      else {
        G__fprinterr(G__serr,"Error: Return type mismatch %s()"
                ,p_ifunc->funcname[ifn]);
        G__genericerror((char*)NULL);
      }
#ifndef G__OLDIMPLEMENTATION1259
      result7->isconst = p_ifunc->isconst[ifn];
#endif
      break;

    case 'i':
      /* return value of constructor */
      if(-1!=p_ifunc->p_tagtable[ifn]) {
        if(G__CPPLINK!=G__struct.iscpplink[p_ifunc->p_tagtable[ifn]] &&
           'e'!=G__struct.type[p_ifunc->p_tagtable[ifn]] &&
           0!=G__store_struct_offset && 1!=G__store_struct_offset)
          result7->obj.i=G__store_struct_offset;
        result7->ref = result7->obj.i;
#ifndef G__OLDIMPLEMENTATION1259
        result7->isconst = 0;
#endif
        break;
      }

    case 'U':
      if('U'==p_ifunc->type[ifn] && 'U'==result7->type) {
        if(p_ifunc->p_tagtable[ifn]!=result7->tagnum) {
          int offset = G__ispublicbase(p_ifunc->p_tagtable[ifn]
                                   ,result7->tagnum,result7->obj.i);
          if(-1==offset) {
            G__fprinterr(G__serr,"Error: Return type mismatch. %s "
                         ,G__fulltagname(p_ifunc->p_tagtable[ifn],1));
            G__fprinterr(G__serr,"not a public base of %s"
                         ,G__fulltagname(result7->tagnum,1));
            G__genericerror((char*)NULL);
            result7->tagnum = p_ifunc->p_tagtable[ifn];
            break;
          }
          else {
            result7->obj.i += offset;
            if(result7->ref) result7->ref += offset;
            result7->tagnum = p_ifunc->p_tagtable[ifn];
          }
        }
      }
      /* no break, this case continues to default: */

      /***************************************************
       * Everything else is returned as integer. This
       * includes char,short,int,long,unsigned version
       * of them, pointer and struct/union.
       * If return value is struct/union, malloced memory
       * area will be freed about 20 lines below by
       * G__destroy(). To prevent any data loss, memory
       * area has to be copied to left hand side memory
       * area of assignment (or temp buffer of expression
       * parser which doesn't exist in this version).
       ***************************************************/
    default:
#ifdef G__SECURITY
      if(isupper(p_ifunc->type[ifn])&&islower(result7->type)&&result7->obj.i
         && 0==G__asm_wholefunction
         ) {
        G__fprinterr(G__serr,"Error: Return type mismatch %s()"
                ,p_ifunc->funcname[ifn]);
        G__genericerror((char*)NULL);
        break;
      }
#endif
      G__letint(result7,p_ifunc->type[ifn],G__int(*result7));
#ifdef G__OLDIMPLEMENTATION753
      if(p_ifunc->reftype[ifn]==G__PARANORMAL) result7->ref=0;
#endif
      if(isupper(result7->type)) {
        result7->obj.reftype.reftype = p_ifunc->reftype[ifn];
      }
#ifdef G__SECURITY
      if(isupper(result7->type)&&G__security&G__SECURE_GARBAGECOLLECTION
         && (!G__no_exec_compile)
        ) {
        /* Add reference count to avoid garbage collection when pointer is
         * returned */
        G__add_refcount((void*)result7->obj.i,(void**)NULL);
      }
#endif
#ifndef G__OLDIMPLEMENTATION1259
      result7->isconst = p_ifunc->isconst[ifn];
#endif
      break;
    }
  }
  
  if(G__RETURN_EXIT1!=G__return) { /* if not exit */
    /* return struct and typedef identity */
    result7->tagnum  = p_ifunc->p_tagtable[ifn];
    result7->typenum = p_ifunc->p_typetable[ifn];
  }

  
  if(G__RETURN_TRY!=G__return) {
    /**************************************************************
     * reset no exec flag
     **************************************************************/
    G__no_exec=0;
    /**************************************************************
     * reset return flag
     * G__return is set to 1 if interpreted function returns by
     * return(); statement.  Until G__return is reset to 0, 
     * execution flow exits from G__exec_statment().
     **************************************************************/
    if(G__RETURN_IMMEDIATE>=G__return) G__return=G__RETURN_NON;
  }
  
#ifndef G__NEWINHERIT
  /* recover prev local variable */
  G__p_local = G_local.prev_local; /* same as G__p_local->prev_local */
#endif

#ifdef G__ASM_WHOLEFUNC
  /**************************************************************
   * whole function bytecode compile end
   **************************************************************/
  if(G__ASM_FUNC_COMPILE&G__asm_wholefunction) {
    if(G__NOERROR!=G__security_error) {
      G__resetbytecode();
    }
    if(G__asm_noverflow) {
      int pc=0;
      if(G__asm_dbg) {
#ifdef G__ASM_DBG
        G__fprinterr(G__serr,"%x : RTN_FUNC",G__asm_cp);
#endif
        G__fprinterr(G__serr,"Bytecode compilation of %s successful"
                ,p_ifunc->funcname[ifn]);
        G__printlinenum();
      }
      G__asm_inst[G__asm_cp] = G__RTN_FUNC;
      G__asm_inst[G__asm_cp+1] = 0;
      G__inc_cp_asm(2,0);
      G__asm_inst[G__asm_cp] = G__RETURN;
      G__resolve_jumptable_bytecode();
      if(G__asm_loopcompile>=2) G__asm_optimize(&pc);
      G__resetbytecode();
      G__no_exec_compile = store_no_exec_compile;
      G__asm_storebytecodefunc(p_ifunc,ifn,localvar
                               ,G__asm_stack,G__asm_dt
                               ,G__asm_inst,G__asm_cp);
    }
    else {
      /* destroy temp object, before restoreing G__no_exec_compile */
      if(G__p_tempbuf->level>=G__templevel && G__p_tempbuf->prev) 
        G__free_tempobject();
      free(G__asm_name);
      G__resetbytecode();
      G__no_exec_compile = store_no_exec_compile;
      /* destroy local memory area */
      G__destroy(localvar,G__BYTECODELOCAL_VAR) ;
      free((void*)localvar);
      if(G__asm_dbg) {
        if(G__dispmsg>=G__DISPWARN) {
          G__fprinterr(G__serr,
                       "Warning: Bytecode compilation of %s failed. Maybe slow"
                       ,p_ifunc->funcname[ifn]);
          G__printlinenum();
        }
      }
      if(G__return>=G__RETURN_IMMEDIATE) G__return=G__RETURN_NON;
      G__security_error=G__NOERROR;
    }
  }
  else {
    /**************************************************************
     * destroy malloced local memory area
     **************************************************************/
    int store_security_error=G__security_error;
    G__security_error = 0;
    G__destroy(&G_local,G__LOCAL_VAR) ;
    G__security_error=store_security_error;
  }
#else /* G__ASM_WHOLEFUNC */
  /**************************************************************
   * destroy malloced local memory area
   **************************************************************/
  G__destroy(&G_local,G__LOCAL_VAR) ;
#endif /* G__ASM_WHOLEFUNC */
  
  if(G__TRYDESTRUCTOR==memfunc_flag) {
    /* destructor for base calss and class members */
    G__basedestructor();
  }

#ifdef G__NEWINHERIT
  /* recover prev local variable */
  G__p_local = G_local.prev_local; /* same as G__p_local->prev_local */
#endif
  
  G__tagnum = store_inherit_tagnum;
  G__store_struct_offset = store_inherit_offset;
  
  /* recover line number and filename*/
  G__ifile.line_number = G_local.prev_line_number;
  G__ifile.filenum=G_local.prev_filenum;
  if(-1!=G__ifile.filenum
     && 0!=G__srcfile[G__ifile.filenum].filename 
     ) 
    strcpy(G__ifile.name,G__srcfile[G__ifile.filenum].filename);
  else {
    G__ifile.name[0]='\0';
  }

  if(G__dispsource && G__ifile.name && G__ifile.name[0]) 
    G__fprinterr(G__serr,"\n# %s   ",G__ifile.name);
  
  /* recover file pointer and fpos */
  G__ifile.fp = prev_fp;
  if(G__ifile.fp) fsetpos(G__ifile.fp,&prev_pos);
  if(G__dispsource) {
    if((G__debug||G__break)&& ((G__prerun!=0)||(G__no_exec==0))&&
       (G__disp_mask==0)){
      G__fprinterr(G__serr,"\n");
    }
  }
  
  if(G__break_exit_func!=0) {
    G__break=1;
    G__break_exit_func=0;
    G__setdebugcond();
  }
  G__break_exit_func=break_exit_func;
  
  /**********************************************
   * decrement busy flag
   **********************************************/
  p_ifunc->busy[ifn]--;
  
  if('P'==store_var_typeB) G__val2pointer(result7);


  G__func_page = store_func_page;
  G__func_now=store_func_now;
#ifdef G__ASM_IFUNC
  /* Pop loop compilation environment */
  G__asm_inst = store_asm_inst;
  G__asm_instsize = store_asm_instsize;
  G__asm_stack = store_asm_stack;
  G__asm_name = store_asm_name;
  G__asm_name_p = store_asm_name_p;
  G__asm_param  = store_asm_param ;
  G__asm_exec  = store_asm_exec ;
  G__asm_noverflow  = store_asm_noverflow ;
  G__asm_cp  = store_asm_cp ;
  G__asm_dt  = store_asm_dt ;
  G__asm_index  = store_asm_index ;
#endif /* G__ASM_IFUNC */

  G__exec_memberfunc=store_exec_memberfunc;
  G__security = store_security;
  G__memberfunc_tagnum = store_memberfunc_tagnum;
  G__memberfunc_struct_offset = store_memberfunc_struct_offset;

  return(1);
}





/**************************************************************************
* G__ifunc_exist
*
*  compare  hash,funcname,type,p_tagtype,p_typetable,reftype,para_nu
*           para_type[],para_p_tagtable[],para_p_typetable[],para_reftype[]
*           para_default[]
*
**************************************************************************/
struct G__ifunc_table *G__ifunc_exist(G__ifunc_table *ifunc_now,int allifunc
                                      ,G__ifunc_table *ifunc,int *piexist
                                      ,int mask)
{
  int i,j,paran;
  int ref_diff;
  while(ifunc) {
    for(i=0;i<ifunc->allifunc;i++) {
      if('~'==ifunc_now->funcname[allifunc][0] &&
         '~'==ifunc->funcname[i][0]) { /* destructor matches with ~ */
#ifdef G__OLDIMPLEMENTATION1706_YET 
        /* This change causes problem with virtual func definition */
        if(ifunc!=ifunc_now || allifunc!=i) {
          *piexist = i;
          return(ifunc);
        }
        else {
          return((struct G__ifunc_table*)NULL);
        }
#else
        *piexist = i;
        return(ifunc);
#endif
      }
      if(ifunc_now->hash[allifunc]!=ifunc->hash[i] ||
         strcmp(ifunc_now->funcname[allifunc],ifunc->funcname[i]) != 0 ||
         (ifunc_now->para_nu[allifunc]!=ifunc->para_nu[i] && 
          ifunc_now->para_nu[allifunc]>=0 && ifunc->para_nu[i]>=0)
#ifndef G__OLDIMPLEMENTATION1258
         || ((ifunc_now->isconst[allifunc]&mask) /* 1798 */
             !=(ifunc->isconst[i]&mask)) 
#endif
         ) continue; /* unmatch */

      
      if(ifunc_now->para_nu[allifunc]>=0 && ifunc->para_nu[i]>=0)
        paran=ifunc_now->para_nu[allifunc];
      else
        paran = 0;
      ref_diff=0;
      for(j=0;j<paran;j++) {
        if(ifunc_now->para_type[allifunc][j]!=ifunc->para_type[i][j] ||
           ifunc_now->para_p_tagtable[allifunc][j]!=ifunc->para_p_tagtable[i][j]
           || (ifunc_now->para_reftype[allifunc][j]!=ifunc->para_reftype[i][j]
               && G__PARAREFERENCE !=
               ifunc_now->para_reftype[allifunc][j]+ifunc->para_reftype[i][j]
               )
           || ifunc_now->para_isconst[allifunc][j]!=ifunc->para_isconst[i][j]
           ) {
          break; /* unmatch */
        }
        if(ifunc_now->para_reftype[allifunc][j]!=ifunc->para_reftype[i][j]) 
          ++ref_diff;
      }
      if(j==paran) { /* all matched */
        if(ref_diff) {
          G__fprinterr(G__serr,"Warning: %s(), parameter only differs in reference type or not"
                       ,ifunc->funcname[i]);
          G__printlinenum();
        }
#ifdef G__OLDIMPLEMENTATION1706_YET
        /* This change causes problem with virtual func definition */
        if(ifunc!=ifunc_now || allifunc!=i) {
          *piexist = i;
          return(ifunc);
        }
        else {
          return((struct G__ifunc_table*)NULL);
        }
#else
        *piexist = i;
        return(ifunc);
#endif
      }
    }
    ifunc=ifunc->next;
  }
  return(ifunc); /* not found case */
}

/**************************************************************************
* G__ifunc_ambiguous
*
*
**************************************************************************/
struct G__ifunc_table *G__ifunc_ambiguous(G__ifunc_table *ifunc_now,int allifunc
                                          ,G__ifunc_table *ifunc,int *piexist
                                          ,int derivedtagnum)
{
  int i,j,paran;
  while(ifunc) {
    for(i=0;i<ifunc->allifunc;i++) {
      if('~'==ifunc_now->funcname[allifunc][0] &&
         '~'==ifunc->funcname[i][0]) { /* destructor matches with ~ */
        *piexist = i;
        return(ifunc);
      }
      if(ifunc_now->hash[allifunc]!=ifunc->hash[i] ||
         strcmp(ifunc_now->funcname[allifunc],ifunc->funcname[i]) != 0
         ) continue; /* unmatch */
      if(ifunc_now->para_nu[allifunc] < ifunc->para_nu[i])
        paran=ifunc_now->para_nu[allifunc];
      else
        paran = ifunc->para_nu[i];
      if(paran<0) paran=0;
      for(j=0;j<paran;j++) {
        if(ifunc_now->para_type[allifunc][j]!=ifunc->para_type[i][j]) 
          break; /* unmatch */
        if(ifunc_now->para_p_tagtable[allifunc][j]
           ==ifunc->para_p_tagtable[i][j]) continue; /* match */
#ifdef G__VIRTUALBASE
        if(-1==G__ispublicbase(ifunc_now->para_p_tagtable[allifunc][j]
                               ,derivedtagnum,G__STATICRESOLUTION2) ||
           -1==G__ispublicbase(ifunc->para_p_tagtable[i][j],derivedtagnum
                               ,G__STATICRESOLUTION2))
#else
        if(-1==G__ispublicbase(ifunc_now->para_p_tagtable[allifunc][j]
                               ,derivedtagnum) ||
           -1==G__ispublicbase(ifunc->para_p_tagtable[i][j],derivedtagnum))
#endif
          break; /* unmatch */
        /* else match */
      }
      if((ifunc_now->para_nu[allifunc] < ifunc->para_nu[i] &&
          ifunc->para_default[i][paran]) ||
         (ifunc_now->para_nu[allifunc] > ifunc->para_nu[i] &&
          ifunc_now->para_default[allifunc][paran])) {
        *piexist = i;
        return(ifunc);
      }
      else if(j==paran) { /* all matched */
        *piexist = i;
        return(ifunc);
      }
    }
    ifunc=ifunc->next;
  }
  return(ifunc); /* not found case */
}

/**************************************************************************
* G__get_ifunchandle
*
*
**************************************************************************/
struct G__ifunc_table *G__get_ifunchandle(char *funcname,G__param *libp
                                          ,int hash,G__ifunc_table *p_ifunc
                                          ,long *pifn
                                          ,int access,int funcmatch)
{
  int ifn=0;
  int ipara=0;
  int itemp=0;

  if(-1!=p_ifunc->tagnum) G__incsetup_memfunc(p_ifunc->tagnum);

  /*******************************************************
   * while interpreted function list exists
   *******************************************************/
  while(p_ifunc) {
    while((ipara==0)&&(ifn<p_ifunc->allifunc)) {
      /* if hash (sum of funcname char) matchs */
      if(hash==p_ifunc->hash[ifn]&&strcmp(funcname,p_ifunc->funcname[ifn])==0 
         && (p_ifunc->access[ifn]&access)) {
        /**************************************************
         * for overloading of function and operator
         **************************************************/
        /**************************************************
         * check if parameter type matchs
         **************************************************/
        /* set(reset) match flag ipara temporarily */
        itemp=0;
        ipara=1;
        
        if(p_ifunc->ansi[ifn]==0) break; /* K&R C style header */
        /* main() no overloading */
        if(G__HASH_MAIN==hash && strcmp(funcname,"main")==0) break; 
        
        /* if more actual parameter than formal parameter, unmatch */
        if(p_ifunc->para_nu[ifn]<libp->paran) {
          ipara=0;
          itemp=p_ifunc->para_nu[ifn]; /* end of this parameter */
          ++ifn; /* next function */
        }
        else {
          /* scan each parameter */
          while(itemp<p_ifunc->para_nu[ifn]) {
            if((G__value*)NULL==p_ifunc->para_default[ifn][itemp] && 
               itemp>=libp->paran
               ) {
              ipara = 0;
            }
            else if (p_ifunc->para_default[ifn][itemp] && itemp>=libp->paran) {
              ipara = 2; /* I'm not sure what this is, Fons. */
            }
            else {   
              ipara=G__param_match(p_ifunc->para_type[ifn][itemp]
                                   ,p_ifunc->para_p_tagtable[ifn][itemp]
                                   ,p_ifunc->para_default[ifn][itemp]
                                   ,libp->para[itemp].type
                                   ,libp->para[itemp].tagnum
                                   ,&(libp->para[itemp])
                                   ,libp->parameter[itemp]
                                   ,funcmatch
                                   ,p_ifunc->para_nu[ifn]-itemp-1
                                   ,p_ifunc->para_reftype[ifn][itemp]
                                   ,p_ifunc->para_isconst[ifn][itemp]
                                   /* ,p_ifunc->isexplicit[ifn] */
                                   );
            }
            switch(ipara) {
            case 2: /* default parameter */
#ifdef G__ASM_DBG
              if(G__asm_dbg) {
                G__fprinterr(G__serr," default%d %c tagnum%d %p : %c tagnum%d %d\n"
                        ,itemp
                        ,p_ifunc->para_type[ifn][itemp]
                        ,p_ifunc->para_p_tagtable[ifn][itemp]
                        ,p_ifunc->para_default[ifn][itemp]
                        ,libp->para[itemp].type
                        ,libp->para[itemp].tagnum
                        ,funcmatch);
              }
#endif
              itemp=p_ifunc->para_nu[ifn]; /* end of this parameter */
              break;
            case 1: /* match this one, next parameter */
#ifdef G__ASM_DBG
              if(G__asm_dbg) {
                G__fprinterr(G__serr," match%d %c tagnum%d %p : %c tagnum%d %d\n"
                        ,itemp
                        ,p_ifunc->para_type[ifn][itemp]
                        ,p_ifunc->para_p_tagtable[ifn][itemp]
                        ,p_ifunc->para_default[ifn][itemp]
                        ,libp->para[itemp].type
                        ,libp->para[itemp].tagnum
                        ,funcmatch);
              }
#endif
              if(G__EXACT!=funcmatch)
                G__warn_refpromotion(p_ifunc,ifn,itemp,libp);
              ++itemp; /* next function parameter */
              break;
            case 0: /* unmatch, next function */
#ifdef G__ASM_DBG
              if(G__asm_dbg) {
                G__fprinterr(G__serr," unmatch%d %c tagnum%d %p : %c tagnum%d %d\n"
                        ,itemp
                        ,p_ifunc->para_type[ifn][itemp]
                        ,p_ifunc->para_p_tagtable[ifn][itemp]
                        ,p_ifunc->para_default[ifn][itemp]
                        ,libp->para[itemp].type
                        ,libp->para[itemp].tagnum
                        ,funcmatch);
              }
#endif
              itemp=p_ifunc->para_nu[ifn]; 
              /* exit from while loop */
              break;
            }
            
          } /* end of while(itemp<p_ifunc->para_nu[ifn]) */
          if(ipara==0) { /* parameter doesn't match */
            ++ifn; /* next function */
          }
        }
      }
      else {  /* funcname doesn't match */
        ++ifn;
      }
    }  /* end of while((ipara==0))&&(ifn<p_ifunc->allifunc)) */
    /******************************************************************
     * next page of interpreted function list
     *******************************************************************/
    if(ifn>=p_ifunc->allifunc) {
      p_ifunc=p_ifunc->next;
      ifn=0;
    }
    else {
      break; /* get out from while(p_ifunc) loop */
    }
  } /* end of while(p_ifunc) */


  *pifn = ifn;
  return(p_ifunc);
}

/**************************************************************************
* G__get_ifunchandle_base
*
*
**************************************************************************/
struct G__ifunc_table *G__get_ifunchandle_base(char *funcname,G__param *libp
                                               ,int hash,G__ifunc_table *p_ifunc
                                               ,long *pifn
                                               ,long *poffset
                                               ,int access,int funcmatch
                                               ,int withInheritance
                                               )
{
  int tagnum;
  struct G__ifunc_table *ifunc;
  int basen=0;
  struct G__inheritance *baseclass;

  /* Search for function */
  *poffset = 0;
  ifunc=G__get_ifunchandle(funcname,libp,hash,p_ifunc,pifn,access,funcmatch);
  if(ifunc || !withInheritance) return(ifunc);

  /* Search for base class function if member function */
  tagnum = p_ifunc->tagnum;
  if(-1!=tagnum) {
    baseclass = G__struct.baseclass[tagnum];
    while(basen<baseclass->basen) {
      if(baseclass->baseaccess[basen]&G__PUBLIC) {
#ifdef G__VIRTUALBASE
        /* Can not handle virtual base class member function for ERTTI
         * because pointer to the object is not given  */
#endif
        *poffset = baseclass->baseoffset[basen];
        p_ifunc = G__struct.memfunc[baseclass->basetagnum[basen]];
        ifunc=G__get_ifunchandle(funcname,libp,hash,p_ifunc,pifn
                                 ,access,funcmatch);
        if(ifunc) return(ifunc);
      }
      ++basen;
    }
  }

  /* Not found , ifunc=NULL */
  return(ifunc);
}

/**************************************************************************
* G__argtype2param()
*
*
**************************************************************************/
void G__argtype2param(char *argtype,G__param *libp)
{
  char typenam[G__MAXNAME*2];
  int p=0;
  int c;
  char *endmark=",);";
  
  libp->paran=0;
  libp->para[0]=G__null;

  do {
    c=G__getstream_template(argtype,&p,typenam,endmark);
    if(typenam[0]) {
      libp->para[libp->paran] = G__string2type(typenam);
      ++libp->paran;
    }
  } while(','==c);
}

/**************************************************************************
* G__get_methodhandle
*
*
**************************************************************************/
struct G__ifunc_table *G__get_methodhandle(char *funcname,char *argtype
                                           ,G__ifunc_table *p_ifunc
                                           ,long *pifn,long *poffset
                                           ,int withConversion
                                           ,int withInheritance
                                           )
{
  struct G__ifunc_table *ifunc;
  struct G__param para;
  int hash;
  int temp;
  struct G__funclist *funclist = (struct G__funclist*)NULL;
  int match;

  int store_def_tagnum = G__def_tagnum;
  int store_tagdefining = G__tagdefining;
  G__def_tagnum = p_ifunc->tagnum;
  G__tagdefining = p_ifunc->tagnum;
  G__argtype2param(argtype,&para);
  G__def_tagnum = store_def_tagnum;
  G__tagdefining = store_tagdefining;
  G__hash(funcname,hash,temp);


 if(withConversion) {
   int tagnum = p_ifunc->tagnum;
   int ifn = (int)(*pifn);

   if(-1!=tagnum) G__incsetup_memfunc(tagnum);

   ifunc = G__overload_match(funcname,&para,hash,p_ifunc,G__TRYNORMAL
                             ,G__PUBLIC_PROTECTED_PRIVATE,&ifn,0
                             ,(withConversion&0x2)?1:0) ;
   *poffset = 0;
   *pifn = ifn;
   if(ifunc || !withInheritance) return(ifunc);
   if(-1!=tagnum) {
     int basen=0;
     struct G__inheritance *baseclass = G__struct.baseclass[tagnum];
     while(basen<baseclass->basen) {
       if(baseclass->baseaccess[basen]&G__PUBLIC) {
         G__incsetup_memfunc(baseclass->basetagnum[basen]);
         *poffset = baseclass->baseoffset[basen];
         p_ifunc = G__struct.memfunc[baseclass->basetagnum[basen]];
         ifunc = G__overload_match(funcname,&para,hash,p_ifunc,G__TRYNORMAL
                                   ,G__PUBLIC_PROTECTED_PRIVATE,&ifn,0,0) ;
         *pifn = ifn;
         if(ifunc) return(ifunc);
       }
       ++basen;
     }
   }
 }
 else {
   /* first, search for exact match */
   ifunc=G__get_ifunchandle_base(funcname,&para,hash,p_ifunc,pifn,poffset
                                 ,G__PUBLIC_PROTECTED_PRIVATE,G__EXACT
                                 ,withInheritance
                                 );
   if(ifunc) return(ifunc);
   
   /* if no exact match, try to instantiate template function */
   funclist = G__add_templatefunc(funcname,&para,hash,funclist,p_ifunc,0);
   if(funclist && funclist->rate==G__EXACTMATCH) {
     ifunc = funclist->ifunc;
     *pifn = funclist->ifn;
     G__funclist_delete(funclist);
     return(ifunc);
   }
   G__funclist_delete(funclist);
   
   for(match=G__EXACT;match<=G__STDCONV;match++) {
     ifunc=G__get_ifunchandle_base(funcname,&para,hash,p_ifunc,pifn,poffset
                                   ,G__PUBLIC_PROTECTED_PRIVATE
                                   ,match
                                   ,withInheritance
                                   );
     if(ifunc) return(ifunc);
   }
 }
 
  return(ifunc);
}


/**************************************************************************
* G__get_methodhandle2
*
*
**************************************************************************/
struct G__ifunc_table *G__get_methodhandle2(char *funcname
                                           ,G__param *libp,G__ifunc_table *p_ifunc
                                           ,long *pifn,long *poffset
                                           ,int withConversion
                                           ,int withInheritance
                                           )
{
  struct G__ifunc_table *ifunc;
  int hash;
  int temp;
  struct G__funclist *funclist = (struct G__funclist*)NULL;
  int match;

  int store_def_tagnum = G__def_tagnum;
  int store_tagdefining = G__tagdefining;
  G__def_tagnum = p_ifunc->tagnum;
  G__tagdefining = p_ifunc->tagnum;
  G__def_tagnum = store_def_tagnum;
  G__tagdefining = store_tagdefining;
  G__hash(funcname,hash,temp);


 if(withConversion) {
   int tagnum = p_ifunc->tagnum;
   int ifn = (int)(*pifn);

   if(-1!=tagnum) G__incsetup_memfunc(tagnum);

   ifunc = G__overload_match(funcname,libp,hash,p_ifunc,G__TRYNORMAL
                             ,G__PUBLIC_PROTECTED_PRIVATE,&ifn,0,0) ;
   *poffset = 0;
   *pifn = ifn;
   if(ifunc || !withInheritance) return(ifunc);
   if(-1!=tagnum) {
     int basen=0;
     struct G__inheritance *baseclass = G__struct.baseclass[tagnum];
     while(basen<baseclass->basen) {
       if(baseclass->baseaccess[basen]&G__PUBLIC) {
         G__incsetup_memfunc(baseclass->basetagnum[basen]);
         *poffset = baseclass->baseoffset[basen];
         p_ifunc = G__struct.memfunc[baseclass->basetagnum[basen]];
         ifunc = G__overload_match(funcname,libp,hash,p_ifunc,G__TRYNORMAL
                                   ,G__PUBLIC_PROTECTED_PRIVATE,&ifn,0,0) ;
         *pifn = ifn;
         if(ifunc) return(ifunc);
       }
       ++basen;
     }
   }
 }
 else {
   /* first, search for exact match */
   ifunc=G__get_ifunchandle_base(funcname,libp,hash,p_ifunc,pifn,poffset
                                 ,G__PUBLIC_PROTECTED_PRIVATE,G__EXACT
                                 ,withInheritance
                                 );
   if(ifunc) return(ifunc);
   
   /* if no exact match, try to instantiate template function */
   funclist = G__add_templatefunc(funcname,libp,hash,funclist,p_ifunc,0);
   if(funclist && funclist->rate==G__EXACTMATCH) {
     ifunc = funclist->ifunc;
     *pifn = funclist->ifn;
     G__funclist_delete(funclist);
     return(ifunc);
   }
   G__funclist_delete(funclist);
   
   for(match=G__EXACT;match<=G__STDCONV;match++) {
     ifunc=G__get_ifunchandle_base(funcname,libp,hash,p_ifunc,pifn,poffset
                                   ,G__PUBLIC_PROTECTED_PRIVATE
                                   ,match
                                   ,withInheritance
                                   );
     if(ifunc) return(ifunc);
   }
 }
 
  return(ifunc);
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
