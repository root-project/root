/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file parse.c
 ************************************************************************
 * Description:
 *  Cint parser functions
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

/* 1929 */
#define G__IFDEF_NORMAL       1
#define G__IFDEF_EXTERNBLOCK  2
#define G__IFDEF_ENDBLOCK     4
#ifndef G__OLDIMPLEMENTATION1929
static int G__externblock_iscpp = 0;
#endif

#ifndef G__OLDIMPLEMENTATION1672
#define G__NONBLOCK   0
#define G__IFSWITCH   1
#define G__DOWHILE    8
int G__ifswitch = G__NONBLOCK;
#endif

#ifndef G__OLDIMPLEMENTATION1103
extern int G__const_setnoerror();
extern int G__const_resetnoerror();
#endif

#ifndef G__SECURITY
/**************************************************************************
* G__DEFVAR()
*
*  Variable allocation
**************************************************************************/
#define G__DEFVAR(TYPE)                         \
         G__var_type=TYPE + G__unsigned;        \
         G__define_var(-1,-1);                  \
         spaceflag = -1;                        \
         iout=0;                                \
         if(mparen==0) return(G__null)
/**************************************************************************
* G__DEFREFVAR()
*
*  Variable allocation
**************************************************************************/
#define G__DEFREFVAR(TYPE)                      \
         G__var_type=TYPE + G__unsigned;        \
         G__reftype=G__PARAREFERENCE;           \
         G__define_var(-1,-1);                  \
         G__reftype=G__PARANORMAL;              \
         spaceflag = -1;                        \
         iout=0;                                \
         if(mparen==0) return(G__null)
/**************************************************************************
* G__DEFSTR()
*
*  Variable allocation
**************************************************************************/
#define G__DEFSTR(STRTYPE)                      \
         G__var_type='u';                       \
         G__define_struct(STRTYPE);             \
         spaceflag = -1;                        \
         iout=0;                                \
         if(mparen==0) return(G__null)
#else
/**************************************************************************
* G__DEFVAR()
*
*  Variable allocation
**************************************************************************/
#define G__DEFVAR(TYPE)                         \
         G__var_type=TYPE + G__unsigned;        \
         G__define_var(-1,-1);                  \
         spaceflag = -1;                        \
         iout=0;                                \
         if(mparen==0||G__return>G__RETURN_NORMAL) return(G__null)
/**************************************************************************
* G__DEFREFVAR()
*
*  Variable allocation
**************************************************************************/
#define G__DEFREFVAR(TYPE)                      \
         G__var_type=TYPE + G__unsigned;        \
         G__reftype=G__PARAREFERENCE;           \
         G__define_var(-1,-1);                  \
         G__reftype=G__PARANORMAL;              \
         spaceflag = -1;                        \
         iout=0;                                \
         if(mparen==0||G__return>G__RETURN_NORMAL) return(G__null)
/**************************************************************************
* G__DEFSTR()
*
*  Variable allocation
**************************************************************************/
#define G__DEFSTR(STRTYPE)                      \
         G__var_type='u';                       \
         G__define_struct(STRTYPE);             \
         spaceflag = -1;                        \
         iout=0;                                \
         if(mparen==0||G__return>G__RETURN_NORMAL) return(G__null)
#endif


#ifndef G__OLDIMPLEMENTATION844
/***********************************************************************
* switch statement jump buffer
***********************************************************************/
static int G__prevcase=0;
extern void G__CMP2_equal G__P((G__value*,G__value*));
#endif

#ifdef G__ASM
/***********************************************************************
* G__alloc_breakcontinue_list
*
***********************************************************************/
static struct G__breakcontinue_list* G__alloc_breakcontinue_list()
{
  struct G__breakcontinue_list *p;
  p = G__pbreakcontinue;
  G__pbreakcontinue = (struct G__breakcontinue_list*)NULL;
  return(p);
}

/***********************************************************************
* G__store_breakcontinue_list
*
***********************************************************************/
static void G__store_breakcontinue_list(destination,breakcontinue)
int destination,breakcontinue;
{
  struct G__breakcontinue_list *p;
  p = (struct G__breakcontinue_list*)
    malloc(sizeof(struct G__breakcontinue_list));
  p->prev = G__pbreakcontinue;
  p->destination = destination;
  p->breakcontinue = breakcontinue;
  G__pbreakcontinue = p;
}

/***********************************************************************
* G__free_breakcontinue_list
*
***********************************************************************/
static void G__free_breakcontinue_list(pbreakcontinue)
struct G__breakcontinue_list *pbreakcontinue;
{
  struct G__breakcontinue_list *p;
  while(G__pbreakcontinue) {
    p = G__pbreakcontinue->prev;
    free((void*)G__pbreakcontinue);
    G__pbreakcontinue = p;
  }
  G__pbreakcontinue = pbreakcontinue;
}

/***********************************************************************
* G__set_breakcontinue_destination
*
***********************************************************************/
void G__set_breakcontinue_destination(break_dest,continue_dest,pbreakcontinue)
int break_dest;
int continue_dest;
struct G__breakcontinue_list *pbreakcontinue;
{
  struct G__breakcontinue_list *p;
  /* G__pbreakcontinue = pbreakcontinue; */
  while(G__pbreakcontinue) {
    if(G__pbreakcontinue->breakcontinue) { /* break */
      G__asm_inst[G__pbreakcontinue->destination] = break_dest;
    }
    else { /* continue */
      G__asm_inst[G__pbreakcontinue->destination] = continue_dest;
    }
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"  assigned %x %d JMP %lx  break,continue\n"
			   ,G__pbreakcontinue->destination
			   ,G__pbreakcontinue->breakcontinue
			   ,G__asm_inst[G__pbreakcontinue->destination]);
			   
#endif
    p = G__pbreakcontinue->prev;
    free((void*)G__pbreakcontinue);
    G__pbreakcontinue = p;
  }
  G__pbreakcontinue = pbreakcontinue;
}
#endif /* G__ASM */

#ifndef G__OLDIMPLEMENTATION754
/***********************************************************************
* G__exec_try()
*
***********************************************************************/
int G__exec_try(statement)
char *statement;
{
  G__exec_statement(); 
  if(G__RETURN_TRY==G__return) {
#ifndef G__OLDIMPLEMENTATION1844
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"    G__no_exec_compile reset\n");
#endif
    G__no_exec=0;
    G__return=G__RETURN_NON;
#else
    int store_breaksignal=G__breaksignal;
    G__breaksignal=0;
    G__return=G__RETURN_NON;

    /* exit try block */
    G__no_exec=1;
    G__mparen=1;
    G__exec_statement();
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"    G__no_exec_compile reset\n");
#endif
    G__no_exec=0;
    G__breaksignal=store_breaksignal;

    /* catch block */
#endif
    return(G__exec_catch(statement));
  }
  return(0);
}

/***********************************************************************
* G__exec_catch()
*
***********************************************************************/
int G__exec_catch(statement)
char *statement;
{
  int c;
  while(1) {
    fpos_t fpos;
    int line_number;

    /* catch (ehclass& obj) {  } 
     * ^^^^^^^ */
#ifndef G__OLDIMPLEMENTATION1282
    do {
      c=G__fgetstream(statement,"(};");
    } while('}'==c);
#else
    c=G__fgetstream(statement,"(};");
#endif
    if('('!=c||strcmp(statement,"catch")!=0) return(1);
    fgetpos(G__ifile.fp,&fpos);
    line_number=G__ifile.line_number;

    /* catch (ehclass& obj) {  } 
     *        ^^^^^^^^ */
    c=G__fgetname_template(statement,")&*");

    if('.'==statement[0]) { /* catch all exceptions */
      /* catch(...) {  } */
      if(')'!=c) c=G__fignorestream(")");
      G__exec_statement();
      break;
    }
    else {
      int tagnum;
      tagnum=G__defined_tagname(statement,2);
      if(G__exceptionbuffer.tagnum==tagnum || 
	 -1!=G__ispublicbase(tagnum,G__exceptionbuffer.tagnum
			     ,G__exceptionbuffer.obj.i)) {
        /* catch(ehclass& obj) { match } */
        G__value store_ansipara;
        store_ansipara=G__ansipara;
        G__ansipara=G__exceptionbuffer;
        G__ansiheader=1;
        G__funcheader=1;
        G__ifile.line_number=line_number;
        fsetpos(G__ifile.fp,&fpos);
        G__exec_statement(); /* declare exception handler object */
        G__globalvarpointer=G__PVOID;
        G__ansiheader=0;
        G__funcheader=0;
        G__ansipara=store_ansipara;
        G__exec_statement(); /* exec catch block body */
        break;
      }
      /* catch(ehclass& obj) { unmatch } */
      if(')'!=c) c=G__fignorestream(")");
      G__no_exec = 1;
      G__exec_statement();
      G__no_exec = 0;
    }
  }
  G__free_exceptionbuffer();
  return(0);
}

/***********************************************************************
* G__ignore_catch()
*
***********************************************************************/
int G__ignore_catch()
{
#ifndef G__OLDIMPLEMENTATION1270
  if(G__asm_noverflow) {
    fpos_t fpos1;
    fseek(G__ifile.fp,-1,SEEK_CUR);
    fseek(G__ifile.fp,-1,SEEK_CUR);
    while(fgetc(G__ifile.fp)!='a') {
      fseek(G__ifile.fp,-1,SEEK_CUR);
      fseek(G__ifile.fp,-1,SEEK_CUR);
    }
    while(fgetc(G__ifile.fp)!='c') {
      fseek(G__ifile.fp,-1,SEEK_CUR);
      fseek(G__ifile.fp,-1,SEEK_CUR);
    }
    fseek(G__ifile.fp,-1,SEEK_CUR);
    fgetpos(G__ifile.fp,&fpos1);
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: CATCH\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp]=G__CATCH;
    G__asm_inst[G__asm_cp+1]=G__ifile.filenum;
    G__asm_inst[G__asm_cp+2]=G__ifile.line_number;
#if defined(G__NONSCALARFPOS2)
    G__asm_inst[G__asm_cp+3]=(long)fpos1.__pos;
#elif defined(G__NONSCALARFPOS_QNX)
    G__asm_inst[G__asm_cp+3]=(long)fpos1._Off;
#else
    G__asm_inst[G__asm_cp+3]=(long)fpos1;
#endif
    G__inc_cp_asm(5,0);
    G__fignorestream("(");
  }
#endif

  G__fignorestream(")");
  G__no_exec = 1;
  G__exec_statement();
  G__no_exec = 0;
  return(0);
}

/***********************************************************************
* G__exec_throw()
*
***********************************************************************/
int G__exec_throw(statement)
char* statement;
{
  int iout;
#ifndef G__OLDIMPLEMENTATION1281
  char buf[G__ONELINE];
  G__fgetstream(buf,";");
  if(isdigit(buf[0])||'.'==buf[0]) {
    strcpy(statement,buf);
    iout=5;
  }
  else {
    sprintf(statement,"new %s",buf);
    iout=strlen(statement);
  }
#else
  strcpy(statement,"new ");
  G__fgetstream(statement+4,";");
  iout=strlen(statement);
#endif
#ifdef G__OLDIMPLEMENTATION1270
#ifndef G__OLDIMPLEMENTATION806
  if(G__asm_noverflow) {
#ifndef G__OLDIMPLEMENTATION841
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"bytecode compile aborted by throw statement");
      G__printlinenum();
    }
#endif
    G__abortbytecode();
    if(G__no_exec_compile) return(0);
  }
#endif
#endif
  if(iout>4) {
    int largestep=0;
    if(G__breaksignal && G__beforelargestep(statement,&iout,&largestep)>=1)
      return(1);
    G__exceptionbuffer = G__getexpr(statement);
#ifndef G__OLDIMPLEMENTATION1270
    if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: THROW\n",G__asm_cp);
#endif
      G__asm_inst[G__asm_cp]=G__THROW;
      G__inc_cp_asm(1,0);
    }
#endif
    if(largestep) G__afterlargestep(&largestep);
    G__exceptionbuffer.ref = G__exceptionbuffer.obj.i;
    if('U'==G__exceptionbuffer.type) G__exceptionbuffer.type='u';
  }
  else {
    G__exceptionbuffer = G__null;
  }
#ifndef G__OLDIMPLEMENTATION1281
  if(0==G__no_exec_compile) {
#endif
    G__no_exec=1;
    G__return=G__RETURN_TRY;
#ifndef G__OLDIMPLEMENTATION1281
  }
#endif
  return(0);
}

/***********************************************************************
* G__alloc_exceptionbuffer
*
***********************************************************************/
G__value G__alloc_exceptionbuffer(tagnum) 
int tagnum;
{
  G__value buf;
  /* create class object */
  buf.obj.i = (long)malloc((size_t)G__struct.size[tagnum]);
#ifndef G__OLDIMPLEMENTATION1978
  buf.obj.reftype.reftype = G__PARANORMAL;
#endif
  buf.type = 'u';
  buf.tagnum = tagnum;
  buf.typenum = -1;
  buf.ref = G__p_tempbuf->obj.obj.i;

  return(buf);
}

/***********************************************************************
* G__free_exceptionbuffer
*
***********************************************************************/
int G__free_exceptionbuffer()
{
  if('u'==G__exceptionbuffer.type && G__exceptionbuffer.obj.i &&
     -1!=G__exceptionbuffer.tagnum) {
    char destruct[G__ONELINE];
    int store_tagnum=G__tagnum;
    int store_struct_offset = G__store_struct_offset;
    int dmy=0;
    G__tagnum = G__exceptionbuffer.tagnum;
    G__store_struct_offset = G__exceptionbuffer.obj.i;
    if(G__CPPLINK==G__struct.iscpplink[G__tagnum]) {
      G__globalvarpointer = G__store_struct_offset;
    }
    else G__globalvarpointer = G__PVOID;
    sprintf(destruct,"~%s()",G__fulltagname(G__tagnum,1));
    if(G__dispsource) {
      G__fprinterr(G__serr,"!!!Destructing exception buffer %s %lx"
		   ,destruct,G__exceptionbuffer.obj.i);
      G__printlinenum();
    }
    G__getfunction(destruct,&dmy ,G__TRYDESTRUCTOR);
#ifndef G__OLDIMPLEMENTATION1277
    if(G__CPPLINK!=G__struct.iscpplink[G__tagnum]) 
      free((void*)G__store_struct_offset);
#else
    free((void*)G__store_struct_offset);
#endif
#ifndef G__OLDIMPLEMENTATION2111
    /* do nothing here, exception object shouldn't be stored in legacy temp buf */
#endif
    G__tagnum = store_tagnum;
    G__store_struct_offset = store_struct_offset;
    G__globalvarpointer = G__PVOID;
  }
  G__exceptionbuffer = G__null;
  return(0);
}
#endif

/***********************************************************************
* G__exec_delete()
*
***********************************************************************/
int G__exec_delete(statement,piout,pspaceflag,isarray,mparen)
char *statement;
int *piout;
int *pspaceflag;
int isarray;
int mparen;
{
  int largestep=0;
#ifndef G__G__NODEBUG
  if(G__breaksignal) {
    if(G__beforelargestep(statement ,piout ,&largestep)>1) return(1);
    if(statement[0]=='\0') {
      *pspaceflag = -1;
      *piout=0;
      return(0);
    }
  }
#endif
  G__delete_operator(statement ,isarray);
#ifndef G__G__NODEBUG
  if(largestep) G__afterlargestep(&largestep);
#endif
  if(mparen==0 || G__return>G__RETURN_NORMAL) return(1);
  *pspaceflag = -1;
  *piout=0;
  return(0);
}

/***********************************************************************
* G__exec_function()
*
***********************************************************************/
int G__exec_function(statement,pc,piout,plargestep,presult)
char *statement;
int *pc;
int *piout;
int *plargestep;
G__value *presult;
{
  /* function call */
  if(*pc==';' || G__isoperator(*pc) || *pc==',' || *pc=='.'
#ifndef G__OLDIMPLEMENTATION1001
     || *pc=='['
#endif
     ) {
    if(*pc!=';' && *pc!=',') {
      statement[(*piout)++] = *pc;
      *pc=G__fgetstream_new(statement+(*piout) ,";");
    }
    if(G__breaksignal && G__beforelargestep(statement ,piout ,plargestep)>1) 
      return(1);
#ifdef G__ASM
    if(G__asm_noverflow) G__asm_clear();
#endif
    *presult=G__getexpr(statement);
  }
#ifndef G__OLDIMPLEMENTATION1515
  else if(*pc=='(') {
    int len = strlen(statement);
    statement[len++] = *pc;
    *pc = G__fgetstream_newtemplate(statement+len,")");
    len = strlen(statement);
    statement[len++] = *pc;
#ifndef G__OLDIMPLEMENTATION1711
    statement[len] = 0; 
#endif
#ifndef G__OLDIMPLEMENTATION1876
    *pc=G__fgetspace();
#ifndef G__OLDIMPLEMENTATION1962
    while(*pc!=';') {
#else
    while(*pc=='(') {
#endif
      len = strlen(statement);
      statement[len++] = *pc;
#ifndef G__OLDIMPLEMENTATION1962
      *pc = G__fgetstream_newtemplate(statement+len,");");
      if(*pc==';') break;
#else
      *pc = G__fgetstream_newtemplate(statement+len,")");
#endif
      len = strlen(statement);
      statement[len++] = *pc;
      statement[len] = 0; 
      *pc=G__fgetspace();
    }
    fseek(G__ifile.fp,-1,SEEK_CUR);
    if(G__dispsource) G__disp_mask=1;
#endif
#ifdef G__ASM
    if(G__asm_noverflow) G__asm_clear();
#endif
    *presult=G__getexpr(statement);
  }
#endif
  /* macro function without ';' at the end */
  else {
    if(G__breaksignal&& G__beforelargestep(statement,piout,plargestep)>1) {
      return(1);
    }
    *presult=G__execfuncmacro(statement,piout);
    if(0==(*piout))  {
      if(G__dispmsg>=G__DISPWARN) {
	G__fprinterr(G__serr,"Warning: %s Missing ';'",statement );
	G__printlinenum();
      }
    }
    fseek(G__ifile.fp,-1,SEEK_CUR);
    if(G__dispsource) G__disp_mask=1;
  }

  if(G__p_tempbuf->level>=G__templevel && G__p_tempbuf->prev) 
    G__free_tempobject();
	  
  if(*plargestep) G__afterlargestep(plargestep);

  return(0);
}

#ifndef G__OLDIMPLEMENTATION1785
/***********************************************************************
 * G__IsFundamentalDecl()
 ***********************************************************************/
int G__IsFundamentalDecl()
{
  char typename[G__ONELINE];
  int c;
  fpos_t pos;
  int result=1;
  int tagnum;

  /* store file position */
  int linenum = G__ifile.line_number;
  fgetpos(G__ifile.fp,&pos);
  G__disp_mask = 1000;

  c=G__fgetname_template(typename,"(");
  if(strcmp(typename,"struct")==0 || strcmp(typename,"class")==0 ||
     strcmp(typename,"union")==0) {
    result=0;
  }
  else {
    tagnum = G__defined_tagname(typename,1);
    if(-1!=tagnum) result = 0;
#ifndef G__OLDIMPLEMENTATION2072
    else {
      int typenum = G__defined_typename(typename);	
      if(-1!=typenum) {
	switch(G__newtype.type[typenum]) {
	case 'c':
	case 's':
	case 'i':
	case 'l':
	case 'b':
	case 'r':
	case 'h':
	case 'k':
	  result=1;
	  break;
        default:
	  result=0;
	}
      }
      else {
        if(strcmp(typename,"unsigned")==0 ||
           strcmp(typename,"char")==0 ||
           strcmp(typename,"short")==0 ||
           strcmp(typename,"int")==0 ||
           strcmp(typename,"long")==0) result=1;
        else result=0;
      }
    }
#endif
  }

  /* restore file position */
  G__ifile.line_number = linenum;
  fsetpos(G__ifile.fp,&pos);
  G__disp_mask = 0;
  return result;
}
#endif

/***********************************************************************
* G__keyword_anytime_5()
*
***********************************************************************/
int G__keyword_anytime_5(statement)
char *statement;
{
#ifndef G__OLDIMPLEMENTATION410
  int c=0;
  int iout=0;
#endif

#ifndef G__OLDIMPLEMENTATION986
  if((G__prerun||(G__ASM_FUNC_COMPILE==G__asm_wholefunction&&0==G__ansiheader))
#ifndef G__OLDIMPLEMENTATION1230
     && G__NOLINK == G__globalcomp
#endif
#ifdef G__OLDIMPLEMENTATION1083_YET
     && (G__func_now>=0 || G__def_struct_member )
#else
     && G__func_now>=0 
#endif
     && strcmp(statement,"const")==0
#ifndef G__OLDIMPLEMENTATION1785
     && G__IsFundamentalDecl()
#endif
     ) {
#ifndef G__OLDIMPLEMENTATION1103
    int rslt;
    G__constvar = G__CONSTVAR;
    G__const_setnoerror();
    rslt=G__keyword_anytime_6("static");
    G__const_resetnoerror();
    G__security_error = G__NOERROR ;
    G__return = G__RETURN_NON;
    return(rslt);
#else
    G__constvar = G__CONSTVAR;
    return(G__keyword_anytime_6("static"));
#endif
  }
#endif

  if(statement[0]!='#') return(0);

  /**********************************
   * This part comes in following cases
   * 1)
   *  #ifdef TRUE
   *  #else        <---
   *  #endif
   * Then skip lines untile #endif
   * appears.
   **********************************/
  if(strcmp(statement,"#else")==0) {
    G__pp_skip(1);
    return(1);
  }
  if(strcmp(statement,"#elif")==0) {
    G__pp_skip(1);
    return(1);
  }
#ifndef G__OLDIMPLEMENTATION410
  if(strcmp(statement,"#line")==0) {
    G__setline(statement,c,&iout);
#ifndef G__OLDIMPLEMENTATION1592
    /* restore statement[0] as we found it because the
       callers might look at it! */
    statement[0]='#';
#endif
    return(1);
  }
#endif
  return(0);
}

/***********************************************************************
* G__keyword_anytime_6()
*
***********************************************************************/
int G__keyword_anytime_6(statement)
char *statement;
{
  int store_no_exec;

  if(strcmp(statement,"static")==0){
    /*************************
     * if G__prerun==1
     *  G__static_alloc==1
     *   allocate memory prerun
     *  if G__prerun==0
     *  G__static_alloc==1
     *   get preallocated mem 
     *  int G__func_now
     *************************/
#ifndef G__OLDIMPLEMENTATION610
    struct G__var_array* store_local=G__p_local;
    if(G__p_local && G__prerun && -1!=G__func_now) G__p_local = (struct G__var_array*)NULL;
#endif
    G__static_alloc=1;
    store_no_exec=G__no_exec;
    G__no_exec=0;
    G__exec_statement();
    G__no_exec=store_no_exec;
    G__static_alloc=0;
#ifndef G__OLDIMPLEMENTATION610
    G__p_local = store_local;
#endif
    return(1);
  }

#ifndef G__OLDIMPLEMENTATION702
  if(1==G__no_exec&&strcmp(statement,"return")==0){
    G__fignorestream(";");
    return(1);
  }
#endif

  if(statement[0]!='#') return(0);
  
  /***********************************
   * 1)
   *  #ifdef macro   <---
   *  #endif 
   * 2)
   *  #ifdef macro   <---
   *  #else
   *  #endif
   ***********************************/
  if(strcmp(statement,"#ifdef")==0){
#ifndef G__OLDIMPLEMENTATION1929
    int stat = G__pp_ifdef(1);
    return(stat);
#else
    G__pp_ifdef(1);
#endif
    return(1);
  }
  
  /***********************************
   * This part comes in following cases
   * 1)
   *  #ifdef TRUE
   *  #endif       <---
   * 2)
   *  #ifdef FAULSE
   *  #else
   *  #endif       <---
   ***********************************/
  if(strcmp(statement,"#endif")==0) {
    return(1);
  }
  
  if(strcmp(statement,"#undef")==0){
    G__pp_undef();
    return(1);
  }

#ifndef G__OLDIMPLEMENTATION681
  if(strcmp(statement,"#ident")==0){
    G__fignoreline();
    return(1);
  }
#endif

  return(0);
}

/***********************************************************************
* G__keyword_anytime_7()
*
***********************************************************************/
int G__keyword_anytime_7(statement)
char *statement;
{
  /***********************************
   * 1)
   *  #ifndef macro   <---
   *  #endif 
   * 2)
   *  #ifndef macro   <---
   *  #else
   *  #endif
   ***********************************/
  if(strcmp(statement,"#define")==0){
#ifndef G__OLDIMPLEMENTATION611
    int store_tagnum=G__tagnum;
    int store_typenum=G__typenum;
    struct G__var_array* store_local=G__p_local;
    G__p_local=(struct G__var_array*)NULL;
#endif
    G__var_type='p';
    G__definemacro=1;
    G__define();
    G__definemacro=0;
#ifndef G__OLDIMPLEMENTATION611
    G__p_local=store_local;
    G__tagnum=store_tagnum;
    G__typenum=store_typenum;
#endif
    return(1);
  }
  if(strcmp(statement,"#ifndef")==0){
#ifndef G__OLDIMPLEMENTATION1929
    int stat = G__pp_ifdef(0);
    return(stat);
#else
    G__pp_ifdef(0);
#endif
    return(1);
  }
  if(strcmp(statement,"#pragma")==0){
    G__pragma();
    return(1);
  }
  return(0);
}

#ifndef G__OLDIMPLEMENTATION500
/***********************************************************************
* G__keyword_anytime_8()
*
***********************************************************************/
int G__keyword_anytime_8(statement)
char *statement;
{
  /***********************************
   * template  <class T> class A { ... };
   * template  A<int>;
   * template  class A<int>;
   *          ^
   ***********************************/
  if(strcmp(statement,"template")==0){
    int c;
    fpos_t pos;
    int line_number;
    char tcname[G__ONELINE];
    line_number = G__ifile.line_number;
    fgetpos(G__ifile.fp,&pos);
    c=G__fgetspace();
    if('<'==c) {
      /* if '<' comes, this is an ordinary template declaration */
      G__ifile.line_number = line_number;
      fsetpos(G__ifile.fp,&pos);
      return(0);
    }
    /* template  A<int>; this is a template instantiation */
    tcname[0] = c;
#ifndef G__OLDIMPLEMENTATION1411
#ifndef G__OLDIMPLEMENTATION2209
    fseek(G__ifile.fp,-1,SEEK_CUR);
    G__disp_mask=1;
    c=G__fgetname_template(tcname,";");
#else
    c=G__fgetname_template(tcname+1,";");
#endif
    if(strcmp(tcname,"class")==0 ||
       strcmp(tcname,"struct")==0) {
      c=G__fgetstream_template(tcname,";");
    }
    else if(isspace(c)) {
#ifndef G__OLDIMPLEMENTATION1843
#ifndef G__OLDIMPLEMENTATION2211
      int len = strlen(tcname);
      int store_c;
      while(len && ('&'==tcname[len-1] || '*'==tcname[len-1])) --len;
      store_c = tcname[len];
      tcname[len] = 0;
#endif
      if(G__istypename(tcname)) {
	G__ifile.line_number = line_number;
	fsetpos(G__ifile.fp,&pos);
	G__exec_statement(); 
	return(1);
      }
      else {
#ifndef G__OLDIMPLEMENTATION2211
	tcname[len] = store_c;
#endif
	c=G__fgetstream_template(tcname+strlen(tcname),";");
      }
#else
      c=G__fgetstream_template(tcname+strlen(tcname),";");
#endif
    }
#else
    c=G__fgetstream_template(tcname+1,";");
#endif
    if(!G__defined_templateclass(tcname)) {
      G__instantiate_templateclass(tcname);
    }
    return(1);
  }
#ifndef G__OLDIMPLEMENTATION1201
  if(strcmp(statement,"explicit")==0){
#ifndef G__OLDIMPLEMENTATION1250
    G__isexplicit = 1;
#endif
    return(1);
  }
#endif
  return(0);
}
#endif

/***********************************************************************
* G__keyword_exec_6()
*
***********************************************************************/
int G__keyword_exec_6(statement,piout,pspaceflag,mparen)
char *statement;
int *piout,*pspaceflag,mparen;
{
  if(strcmp(statement,"friend")==0) {
    if(G__parse_friend(piout,pspaceflag,mparen)) return(1);
    return(0);
  }
/* #ifdef G__ROOT */
  if(strcmp(statement,"extern")==0 || strcmp(statement,"EXTERN")==0) {
/* #else */
  /* if(strcmp(statement,"extern")==0) { */
/* #endif */
    if(G__externignore(piout,pspaceflag,mparen)) return(1);
    return(0);
  }
  if(strcmp(statement,"signed")==0) {
    *pspaceflag = -1;
    *piout=0;
    return(0);
  }
  if(strcmp(statement,"inline")==0) {
    *pspaceflag = -1;
    *piout=0;
    return(0);
  }
  if(strcmp(statement,"#error")==0) {
    G__pounderror();
    *pspaceflag = -1;
    *piout=0;
    return(0);
  }
  return(0);
}

#ifndef G__OLDIMPLEMENTATION1584
/***********************************************************************
* G__toLowerString
***********************************************************************/
void G__toUniquePath(s)
char *s;
{
  int i=0,j=0;
  char *d;
  if(!s) return;
  d = (char*)malloc(strlen(s)+1);
  while(s[i]) {
    d[j] = tolower(s[i]);
    if(i && '\\'==s[i] && '\\'==s[i-1]) { /* do nothing */ }
    else { ++j; }
    ++i;
  }
  d[j] = 0;
  strcpy(s,d);
  free((void*)d);
}
#endif

/***********************************************************************
* G__setline()
*
* Called by
*    G__exec_statement()
*
*  Set line number by '# <line> <file>' statement 
***********************************************************************/
int G__setline(statement,c,piout)
char *statement;
int c;
int *piout;
{
#ifndef G__OLDIMPLEMENTATION410
  char *endofline="\n\r";
#else
  char *endofline="\n";
#endif
  /* char *usrinclude="/usr/include/"; */
  /* char *codelibs="/usr/include/codelibs/"; */
  /* char *SC="/usr/include/SC/"; */
  int len,lenstl;
  char sysinclude[G__MAXFILENAME];
  char sysstl[G__MAXFILENAME];
  int hash,temp;
  int i;
  int null_entry;

  if(c!='\n' && c!='\r') {
    c=G__fgetname(statement+1,endofline);
    /*****************************************
     * # [line] "<[filename]>"
     *****************************************/
    if(isdigit(statement[1])) {
      if(c!='\n' && c!='\r') {
	G__ifile.line_number = atoi(statement+1)-1;
	c=G__fgetname(statement,endofline);
	/*****************************************
	 * # [line] "[filename]"
	 *****************************************/
	if(statement[0]=='"') {
	  G__getcintsysdir();
	  sprintf(sysinclude,"%s%sinclude%s",G__cintsysdir,G__psep,G__psep);
	  sprintf(sysstl,"%s%sstl%s",G__cintsysdir,G__psep,G__psep);
	  len=strlen(sysinclude);
	  lenstl=strlen(sysstl);
#ifndef G__OLDIMPLEMENTATION1584
#ifdef G__WIN32
	  G__toUniquePath(sysinclude);
	  G__toUniquePath(sysstl);
	  G__toUniquePath(statement);
#endif
#endif
	  if(strncmp(sysinclude,statement+1,(size_t)len)==0||
	     strncmp(sysstl,statement+1,(size_t)lenstl)==0) {
	    G__globalcomp=G__NOLINK;
	  }
#ifndef G__OLDIMPLEMENTATION1451
	  else if(G__ifile.fp==G__mfp) {
	  }
#endif
	  else {
	    G__globalcomp=G__store_globalcomp;
#ifndef G__OLDIMPLEMENTATION1451
	    {
	      struct G__ConstStringList* sysdir = G__SystemIncludeDir;
	      while(sysdir) {
		if(strncmp(sysdir->string,statement+1,sysdir->hash)==0)
		  G__globalcomp=G__NOLINK;
		sysdir = sysdir->prev;
	      }
	    }
#endif
	  }
	  statement[strlen(statement)-1]='\0';
	  strcpy(G__ifile.name,statement+1);
	  G__hash(G__ifile.name,hash,temp);
	  temp=0;
	  null_entry = -1;
	  for(i=0;i<G__nfile;i++) {
	    if((char*)NULL==G__srcfile[i].filename && -1==null_entry) 
	      null_entry = i;
#ifndef G__OLDIMPLEMENTATION1196
	    if(G__matchfilename(i,G__ifile.name)) {
#else
	    if(hash==G__srcfile[i].hash && 
	       strcmp(G__ifile.name,G__srcfile[i].filename)==0) {
#endif
	      temp=1;
	      break;
	    }
	  }
	  if(temp) {
	    G__ifile.filenum = i;
#ifndef G__OLDIMPLEMENTATION437
	    G__security = G__srcfile[i].security;
#endif
	  }
	  else if(-1 != null_entry) {
	    G__srcfile[null_entry].hash=hash;
	    G__srcfile[null_entry].filename
	      =(char*)malloc(strlen(statement+1)+1);
	    strcpy(G__srcfile[null_entry].filename,statement+1);
	    G__srcfile[null_entry].prepname=(char*)NULL;
	    G__srcfile[null_entry].fp=(FILE*)NULL;
	    G__srcfile[null_entry].maxline=0;
	    G__srcfile[null_entry].breakpoint=(char*)NULL;
#ifndef G__OLDIMPLEMENTATION437
	    G__srcfile[null_entry].security = G__security;
#endif
#ifndef G__PHILIPPE0
	    /* If we are using a preprocessed file, the logical file
	       is actually located in the result file from the preprocessor
	       We need to need to carry this information on.
	       If the previous file (G__ifile) was preprocessed, this one
	       wshould also be. */
	    if( G__cpp && G__srcfile[G__ifile.filenum].prepname[0] ) {
	      G__srcfile[null_entry].prepname = 
		(char*)malloc(strlen(G__srcfile[G__ifile.filenum].prepname)+1);
	      strcpy(G__srcfile[null_entry].prepname,
		     G__srcfile[G__ifile.filenum].prepname);	
	      G__srcfile[null_entry].fp = G__ifile.fp;
	    }
#endif
#ifndef G__PHILIPPE25
#ifndef G__OLDIMPLEMENTATION952
            G__srcfile[null_entry].included_from=G__ifile.filenum;
#endif
#ifndef G__OLDIMPLEMENTATION1207
            G__srcfile[null_entry].ispermanentsl = 0;
            G__srcfile[null_entry].initsl = (G__DLLINIT)NULL;
#endif
#ifndef G__OLDIMPLEMENTATION1273
            G__srcfile[null_entry].hasonlyfunc = (struct G__dictposition*)NULL;
#endif
#endif /* G__PHILIPPE25 */
	    G__ifile.filenum = null_entry;
	  }
	  else {
#ifndef G__OLDIMPLEMENTATION1601
	    if(G__nfile==G__gettempfilenum()+1) {
#else
	    if(G__nfile==G__MAXFILE) {
#endif
	      G__fprinterr(G__serr,
		  "Limitation: Sorry, can not create any more file entry\n");
	    }
	    else {
	      G__srcfile[G__nfile].hash=hash;
	      G__srcfile[G__nfile].filename
		=(char*)malloc(strlen(statement+1)+1);
	      strcpy(G__srcfile[G__nfile].filename,statement+1);
	      G__srcfile[G__nfile].prepname=(char*)NULL;
	      G__srcfile[G__nfile].fp=(FILE*)NULL;
	      G__srcfile[G__nfile].maxline=0;
	      G__srcfile[G__nfile].breakpoint=(char*)NULL;
#ifndef G__OLDIMPLEMENTATION437
	      G__srcfile[G__nfile].security = G__security;
#endif
#ifndef G__PHILIPPE0
	      /* If we are using a preprocessed file, the logical file
		 is actually located in the result file from the preprocessor
		 We need to need to carry this information on.
		 If the previous file (G__ifile) was preprocessed, this one
		 should also be. */
	      if( G__cpp && 
#ifndef G__OLDIMPLEMENTATION1323
		  G__srcfile[G__ifile.filenum].prepname &&
#endif
		  G__srcfile[G__ifile.filenum].prepname[0] ) {
		G__srcfile[G__nfile].prepname = 
	       (char*)malloc(strlen(G__srcfile[G__ifile.filenum].prepname)+1);
		strcpy(G__srcfile[G__nfile].prepname,
		       G__srcfile[G__ifile.filenum].prepname);	
		G__srcfile[G__nfile].fp = G__ifile.fp;
	      }
#endif
#ifndef G__PHILIPPE25
              /* Initilialize more of the data member see loadfile.c:1529 */
#ifndef G__OLDIMPLEMENTATION952
	      G__srcfile[G__nfile].included_from=G__ifile.filenum;
#endif
#ifndef G__OLDIMPLEMENTATION1207
              G__srcfile[G__nfile].ispermanentsl = 0;
              G__srcfile[G__nfile].initsl = (G__DLLINIT)NULL;
#endif
#ifndef G__OLDIMPLEMENTATION1273
              G__srcfile[G__nfile].hasonlyfunc = (struct G__dictposition*)NULL;
#endif
#endif /* G__PHILIPPE25 */
	      G__ifile.filenum = G__nfile;
	      ++G__nfile;
	    }
	  }
	}
	if(c!='\n' && c!='\r') {
	  while((c=G__fgetc())!='\n' && c!='\r');
	}
      }
      else {
	G__ifile.line_number = atoi(statement+1);
      }
    }
    /*****************************************
     * #  define|if|ifdef|ifndef...
     *****************************************/
    else {
      *piout=strlen(statement);
      return(1);
    }
  }
  return(0);
}

/***********************************************************************
* G__skip_comment()
*
***********************************************************************/
int G__skip_comment()
{
  char statement[5];
#ifndef G__OLDIMPLEMENTATION1616
  int c;
#endif
  statement[0]=G__fgetc();
  statement[1]=G__fgetc();
  statement[2]='\0';
  while(strcmp(statement,"*/")!=0) {
#ifdef G__MULTIBYTE
    if(G__IsDBCSLeadByte(statement[0])) {
      statement[0]='\0';
      G__CheckDBCS2ndByte(statement[1]);
    }
    else {
      statement[0]=statement[1];
    }
#else
    statement[0]=statement[1];
#endif
#ifndef G__OLDIMPLEMENTATION1616
    if(EOF==(c=G__fgetc())) {
      G__genericerror("Error: unexpected /* ...EOF");
      if(G__key!=0) system("key .cint_key -l execute");
      G__eof=2;
      return(EOF);
    }
    statement[1] = c;
#else
    if(EOF==(statement[1]=G__fgetc())) {
      G__genericerror("Error: unexpected /* ...EOF");
      if(G__key!=0) system("key .cint_key -l execute");
      G__eof=2;
      return(EOF);
    }
#endif
  }
  return(0);
}

/***********************************************************************
* G__pp_command()
*
*  # if      COND
*  # ifdef   MACRO
*  # ifndef  MACRO
*
*  # elif    COND
*  # else
*
*  # endif
*   ^
*
* to be added
*  # num
*  # define MACRO
*
***********************************************************************/
int G__pp_command()
{
  int c;
  char condition[G__ONELINE];
  c=G__fgetname(condition,"\n\r");
  if(isdigit(condition[0])) {
    if('\n'!=c && '\r'!=c) G__fignoreline();
    G__ifile.line_number=atoi(condition);
  }
  else if(strcmp(condition,"else")==0||
	  strcmp(condition,"elif")==0)   G__pp_skip(1);
  else if(strcmp(condition,"if")==0)     G__pp_if();
  else if(strcmp(condition,"ifdef")==0)  G__pp_ifdef(1);
  else if(strcmp(condition,"ifndef")==0) G__pp_ifdef(0);
  else if('\n'!=c && '\r'!=c)            G__fignoreline();
  return(0);
}

/***********************************************************************
* G__pp_skip()
*
* Called by
*   G__pp_if              if condition is faulse
*   G__pp_ifdef           if condition is faulse
*   G__exec_statement     '#else'
*   G__exec_statement     '#elif'
*
*   #if [condition] , #ifdef, #ifndef
*         // read through this block
*   #else
*   #endif
***********************************************************************/
void G__pp_skip(elifskip)
int elifskip;
{
  char oneline[G__LONGLINE*2];
  char argbuf[G__LONGLINE*2];
  char *arg[G__ONELINE];
  int argn;
  
  FILE *fp;
  int nest=1;
  char condition[G__ONELINE];
  char temp[G__ONELINE];
  int i;
  
  fp=G__ifile.fp;
  
  /* elace traced mark */
  if(0==G__nobreak && 0==G__disp_mask&&
     G__srcfile[G__ifile.filenum].breakpoint&&
     G__srcfile[G__ifile.filenum].maxline>G__ifile.line_number) {
    G__srcfile[G__ifile.filenum].breakpoint[G__ifile.line_number]
      &=G__NOTRACED;
  }
  
  /********************************************************
   * Read lines until end of conditional compilation
   ********************************************************/
  while(nest && G__readline(fp,oneline,argbuf,&argn,arg)!=0) {
    /************************************************
     *  If input line is "abcdefg hijklmn opqrstu"
     *
     *           arg[0]
     *             |
     *     +-------+-------+
     *     |       |       |
     *  abcdefg hijklmn opqrstu
     *     |       |       |
     *   arg[1]  arg[2]  arg[3]    argn=3
     *
     ************************************************/
    ++G__ifile.line_number;
    
    if(argn>0) {
      if(strcmp(arg[1],"#")==0
#ifndef G__OLDIMPLEMENTATION425
	 || strcmp(arg[1],"#pragma")==0
#endif
	 ) {
	if(strcmp(arg[2],"if")==0 ||
	   strcmp(arg[2],"ifdef")==0 ||
	   strcmp(arg[2],"ifndef")==0) {
	  ++nest;
	}
	else if(strcmp(arg[2],"else")==0) {
	  if(nest==1 && elifskip==0) nest=0;
	}
	else if(strcmp(arg[2],"endif")==0) {
	  --nest;
	}
	else if(strcmp(arg[2],"elif")==0) {
	  if(nest==1 && elifskip==0) {
#ifndef G__OLDIMPLEMENTATION878
	    int store_no_exec_compile=G__no_exec_compile;
	    int store_asm_wholefunction=G__asm_wholefunction;
	    int store_asm_noverflow=G__asm_noverflow;
	    G__no_exec_compile=0;
	    G__asm_wholefunction=0;
	    G__abortbytecode();
#endif
	    strcpy(condition,"");
	    for(i=3;i<=argn;i++) {
	      sprintf(temp ,"%s%s" ,condition ,arg[i]);
	      strcpy(condition,temp);
	    }
	    G__noerr_defined=1;
	    if(G__test(condition)) {
	      nest=0;
	    }
#ifndef G__OLDIMPLEMENTATION878
	    G__no_exec_compile=store_no_exec_compile;
	    G__asm_wholefunction=store_asm_wholefunction;
	    G__asm_noverflow=store_asm_noverflow;
#endif
	    G__noerr_defined=0;
	  }
	}
      }
      else if(strcmp(arg[1],"#if")==0 ||
	      strcmp(arg[1],"#ifdef")==0 ||
	      strcmp(arg[1],"#ifndef")==0) {
	++nest;
      }
      else if(strcmp(arg[1],"#else")==0
#ifndef G__OLDIMPLEMENTATION1200
	      || strncmp(arg[1],"#else/*",7)==0
	      || strncmp(arg[1],"#else//",7)==0
#endif
	      ) {
	if(nest==1 && elifskip==0) nest=0;
      }
      else if(strcmp(arg[1],"#endif")==0
#ifndef G__OLDIMPLEMENTATION1200
	      || strncmp(arg[1],"#endif/*",8)==0
	      || strncmp(arg[1],"#endif//",8)==0
#endif
	      ) {
	--nest;
      }
      else if(strcmp(arg[1],"#elif")==0) {
	if(nest==1 && elifskip==0) {
#ifndef G__OLDIMPLEMENTATION878
	  int store_no_exec_compile=G__no_exec_compile;
	  int store_asm_wholefunction=G__asm_wholefunction;
	  int store_asm_noverflow=G__asm_noverflow;
	  G__no_exec_compile=0;
	  G__asm_wholefunction=0;
	  G__abortbytecode();
#endif
	  strcpy(condition,"");
	  for(i=2;i<=argn;i++) {
	    sprintf(temp ,"%s%s" ,condition ,arg[i]);
	    strcpy(condition,temp);
	  }
#ifndef G__OLDIMPLEMENTATION1459
          i = strlen (oneline) - 1;
          while (i >= 0 && (oneline[i] == '\n' || oneline[i] == '\r'))
            --i;
          if (oneline[i] == '\\') {
            int len = strlen (condition);
            while (1) {
              G__fgetstream (condition+len, "\n\r");
              if (condition[len] == '\\' && (condition[len+1] == '\n' ||
                                             condition[len+1] == '\r')) {
                char* p = condition + len;
                memmove (p, p+2, strlen (p+2) + 1);
              }
              len = strlen (condition) - 1;
              while (len>0 && (condition[len]=='\n' || condition[len]=='\r'))
                --len;
              if (condition[len] != '\\') break;
            }
          }
#endif
	  G__noerr_defined=1;
	  if(G__test(condition)) {
	    nest=0;
	  }
#ifndef G__OLDIMPLEMENTATION878
	  G__no_exec_compile=store_no_exec_compile;
	  G__asm_wholefunction=store_asm_wholefunction;
	  G__asm_noverflow=store_asm_noverflow;
#endif
	  G__noerr_defined=0;
	}
      }
    }
  }
  
  /* set traced mark */
  if(0==G__nobreak && 
     0==G__disp_mask&& 0==G__no_exec_compile &&
     G__srcfile[G__ifile.filenum].breakpoint&&
     G__srcfile[G__ifile.filenum].maxline>G__ifile.line_number) {
    G__srcfile[G__ifile.filenum].breakpoint[G__ifile.line_number]
      |=(!G__no_exec);
  }
  
  if(G__dispsource) {
    if((G__debug||G__break||G__step
#ifdef G__OLDIMPLEMENTATION473
	||(strcmp(G__breakfile,G__ifile.name)==0)||(strcmp(G__breakfile,"")==0)
#endif
	)&&
       ((G__prerun!=0)||(G__no_exec==0))&&
       (G__disp_mask==0)){
      G__fprinterr(G__serr, "# conditional interpretation, SKIPPED");
      G__fprinterr(G__serr,"\n%-5d",G__ifile.line_number-1);
      G__fprinterr(G__serr,"%s",arg[0]);
      G__fprinterr(G__serr,"\n%-5d",G__ifile.line_number);
    }
  }
}

#ifndef G__OLDIMPLEMENTATION1929
/***********************************************************************
* G__pp_ifdefextern()
*
*   #ifdef __cplusplus
*   extern "C" {       ^
*   #endif
*
*   #ifdef __cplusplus ^
*   }
*   #endif
***********************************************************************/
int G__pp_ifdefextern(temp) 
char* temp;
{
  int cin;
  fpos_t pos;
  int linenum = G__ifile.line_number;
  fgetpos(G__ifile.fp,&pos);

  cin = G__fgetname(temp,"\"}#");

  if('}'==cin) {
    /******************************
     *   #ifdef __cplusplus
     *   }
     *   #endif
     *****************************/
    G__fignoreline();
    do {
      cin = G__fgetstream(temp,"#");
      cin = G__fgetstream(temp,"\n\r");
    } while(strcmp(temp,"endif")!=0); 
    return(G__IFDEF_ENDBLOCK);
  }

  if('#'!=cin && strcmp(temp,"extern")==0) {
    /******************************
     *   #ifdef __cplusplus
     *   extern "C" {
     *   #endif
     *****************************/
    /******************************
     *   #ifdef __cplusplus
     *   extern "C" {  ...  }
     *   #endif
     *****************************/
    
    G__var_type='p';
    if('{'!=cin) cin = G__fgetspace();
    if('"'==cin) {
      /* extern "C" {  } */
      int flag=0;
      int store_iscpp=G__iscpp;
      int store_externblock_iscpp=G__externblock_iscpp;
      char fname[G__MAXFILENAME];
      cin = G__fgetstream(fname,"\"");

      temp[0] = 0;
      cin = G__fgetstream(temp,"{\r\n");

      if(0!=temp[0] || '{'!=cin)  goto goback;
#ifndef G__OLDIMPLEMENTATION1933
      cin = G__fgetstream(temp,"\n\r");
      if (cin=='}' && 0==strcmp(fname,"C")) {
        goto goback;
      }
#else
      G__fignoreline();
#endif
      cin = G__fgetstream(temp,"#\n\r");
      if('#'!=cin) goto goback;
      cin = G__fgetstream(temp,"\n\r");
      if(strcmp(temp,"endif")!=0) goto goback;

      if(0==strcmp(fname,"C")) {
	G__externblock_iscpp = (G__iscpp||G__externblock_iscpp);
	G__iscpp=0; 
      }
      else {
	G__loadfile(fname);
	G__SetShlHandle(fname);
	flag=1;
      }
      G__mparen = 1;
      G__exec_statement();
      G__iscpp=store_iscpp;
      G__externblock_iscpp = store_externblock_iscpp;
      if(flag) G__ResetShlHandle();
      return(G__IFDEF_EXTERNBLOCK);
    }
  }

 goback:
  fsetpos(G__ifile.fp,&pos);
  G__ifile.line_number = linenum;
  return(G__IFDEF_NORMAL);
}
#endif

/***********************************************************************
* G__pp_if()
*
* Called by
*   G__exec_statement()   '#if'
*
*   #if [condition]
*   #else
*   #endif
***********************************************************************/
int G__pp_if()
{
  char condition[G__LONGLINE];
  int c,len=0;
#ifndef G__OLDIMPLEMENTATION878
  int store_no_exec_compile;
  int store_asm_wholefunction;
  int store_asm_noverflow;
#endif
  
  do {
    c = G__fgetstream(condition+len,"\n\r");
    len = strlen(condition)-1;
    if(len<0) len=0;
#ifndef G__OLDIMPLEMENTATION941
    if(len>0 && (condition[len]=='\n' ||condition[len]=='\r')) --len;
#endif
  } while('\\'==condition[len]);

#ifndef G__OLDIMPLEMENTATION868
  {
    char *p;
    while((p=strstr(condition,"\\\n"))!=0) {
      memmove(p,p+2,strlen(p+2)+1);
    }
  }
#endif
#ifndef G__OLDIMPLEMENTATION1063
  {
    char *p;
    while((p=strstr(condition,"\\\r"))!=0) {
      memmove(p,p+2,strlen(p+2)+1);
    }
  }
#endif
  
  /* This supresses error message when undefined
   * macro is refered in the #if defined(macro) */
  G__noerr_defined=1;
  
  /*************************
   * faulse
   * skip line until #else,
   * #endif or #elif.
   * Then, return to evaluation
   *************************/
#ifndef G__OLDIMPLEMENTATION878
  store_no_exec_compile=G__no_exec_compile;
  store_asm_wholefunction=G__asm_wholefunction;
  store_asm_noverflow=G__asm_noverflow;
  G__no_exec_compile=0;
  G__asm_wholefunction=0;
  G__abortbytecode();
#endif
  if(!G__test(condition)) {
    /********************
     * SKIP
     ********************/
    G__pp_skip(0);
  }
#ifndef G__OLDIMPLEMENTATION1929
  else {
    int stat;
    G__no_exec_compile=store_no_exec_compile;
    G__asm_wholefunction=store_asm_wholefunction;
    G__asm_noverflow=store_asm_noverflow;
    G__noerr_defined=0;
    stat = G__pp_ifdefextern(condition);
    return(stat); /* must be either G__IFDEF_ENDBLOCK or G__IFDEF_NORMAL */
  }
#endif
#ifndef G__OLDIMPLEMENTATION878
  G__no_exec_compile=store_no_exec_compile;
  G__asm_wholefunction=store_asm_wholefunction;
  G__asm_noverflow=store_asm_noverflow;
#endif
  G__noerr_defined=0;

  return(G__IFDEF_NORMAL);
}

/***********************************************************************
* G__defined_macro()
*
* Search for macro symbol
*
***********************************************************************/
int G__defined_macro(macro)
char *macro;
{
  struct G__var_array *var;
  int hash,iout;
  G__hash(macro,hash,iout);
  var = &G__global;
  do {
    for(iout=0;iout<var->allvar;iout++) {
      if((tolower(var->type[iout])=='p' 
#ifndef G__OLDIMPLEMENTATION904
	  || 'T'==var->type[iout]
#endif
	  ) &&
	 hash == var->hash[iout] && strcmp(macro,var->varnamebuf[iout])==0)
	return(1); /* found */
    }
  } while((var=var->next)) ;
  if(682==hash && strcmp(macro,"__CINT__")==0) return(1);
#ifndef G__OLDIMPLEMENTATION1883
  if(!G__cpp && 1704==hash && strcmp(macro,"__CINT_INTERNAL_CPP__")==0) return(1);
#endif
  if(
#ifndef G__OLDIMPLEMENTATION1929
     (G__iscpp || G__externblock_iscpp)
#else
     G__iscpp 
#endif
     && 1193==hash && strcmp(macro,"__cplusplus")==0) return(1);
#ifndef G__OLDIMPLEMENTATION869
  { /* Following fix is not completely correct. It confuses typedef names
     * as macro */
    /* look for typedef names defined by '#define foo int' */
    int stat;
    int save_tagnum = G__def_tagnum;
    G__def_tagnum = -1 ;
    stat = G__defined_typename(macro);
    G__def_tagnum = save_tagnum;
    if(stat>=0) return(1);
  }
#endif
#ifndef G__OLDIMPLEMENTATION2041
  /* search symbol macro table */
  if(macro!=G__replacesymbol(macro)) return(1);
  /* search  function macro table */
  {
    struct G__Deffuncmacro *deffuncmacro;
    deffuncmacro = &G__deffuncmacro;
    while(deffuncmacro->next) {
      if(deffuncmacro->name && strcmp(macro,deffuncmacro->name)==0) {
	return(1);
      }
      deffuncmacro=deffuncmacro->next;
    }
  }
#endif
  return(0); /* not found */
}

/***********************************************************************
* G__pp_ifdef()
*
* Called by
*   G__exec_statement()   '#ifdef'
*   G__exec_statement()   '#ifndef'
*
*   #ifdef [macro]
*   #else
*   #endif
***********************************************************************/
int G__pp_ifdef(def)
int def;  /* 1 for ifdef 0 for ifndef */
{
  char temp[G__LONGLINE];
  int notfound=1;
  
  G__fgetname(temp,"\n\r");
  
  notfound = G__defined_macro(temp) ^ 1; 
  
  /*****************************************************************
   * false macro not found skip line until #else, #endif or #elif.
   * Then, return to evaluation
   *************************************************************/
  if(notfound==def) {
    /* SKIP */
    G__pp_skip(0);
  }
#ifndef G__OLDIMPLEMENTATION1929
  else {
    int stat = G__pp_ifdefextern(temp);
    return(stat); /* must be either G__IFDEF_ENDBLOCK or G__IFDEF_NORMAL */
  }
#endif

  return(G__IFDEF_NORMAL);
}

/***********************************************************************
* G__pp_undef()
*
* Called by
*   G__exec_statement()   '#undef'
*
*   #undef
***********************************************************************/
void G__pp_undef()
{
  int i;
  char temp[G__MAXNAME];
  struct G__var_array *var = &G__global;
  
  G__fgetname(temp,"\n\r");
  
  while(var) {
    for(i=0;i<var->allvar;i++) {
      if((strcmp(temp,var->varnamebuf[i])==0)&&
	 (var->type[i]=='p')) {
	var->hash[i] = 0;
	var->varnamebuf[i][0] = '\0'; /* sprintf(var->varnamebuf[i],""); */
      }
    }
    var = var->next;
  }
}

/***********************************************************************
* G__exec_do()
*
* Called by
*   G__exec_statement()   'do { } while();'
*
*  do { statement list } while(condition);
***********************************************************************/
G__value G__exec_do()
{
  fpos_t store_fpos;
  int    store_line_number;
  G__value result;
  char condition[G__ONELINE];
#ifdef G__ASM
  int asm_exec=0;
  int asm_start_pc;
  int store_asm_noverflow=0;
#endif
  int cond;
  int store_asm_cp;
  struct G__breakcontinue_list *store_pbreakcontinue=NULL;
  int allocflag=0;
  int executed_break=0;
  int store_no_exec_compile;

#ifndef G__OLDIMPLEMENTATION1672
  int store_ifswitch = G__ifswitch;
  G__ifswitch = G__DOWHILE;
#endif

  if(G__p_tempbuf->level>=G__templevel && G__p_tempbuf->prev) {
    G__free_tempobject();
  }

  fgetpos(G__ifile.fp,&store_fpos);
  store_line_number=G__ifile.line_number;
  
  G__no_exec=0;
  G__mparen=0;
#ifdef G__ASM
  if(G__asm_loopcompile) {
    store_asm_noverflow=G__asm_noverflow;
    if(G__asm_noverflow==0) {
      G__asm_noverflow=1;
      G__clear_asm();
    }
    asm_start_pc = G__asm_cp;
#ifdef G__ASM_DBG
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"Loop compile start");
      G__printlinenum();
    }
#endif
#ifndef G__OLDIMPLEMENTATION1019
    G__asm_clear();
#endif
  }
#endif

  /* breakcontinue buffer setup */
  if(G__asm_noverflow) {
    store_pbreakcontinue = G__alloc_breakcontinue_list();
    allocflag=1;
  }

#ifndef G__OLDIMPLEMENTATION744
  store_no_exec_compile=G__no_exec_compile;
  result=G__exec_statement();
  G__no_exec_compile=store_no_exec_compile;
#else
  result=G__exec_statement();
#endif
  if(G__return!=G__RETURN_NON) {
    /* free breakcontinue buffer */
    if(allocflag) G__free_breakcontinue_list(store_pbreakcontinue);
#ifndef G__OLDIMPLEMENTATION1672
    G__ifswitch = store_ifswitch;
#endif
    return(result);
  }
  
  /*************************************************
   * do {
   *    if() break;
   * } while();
   *************************************************/
  if(result.type==G__block_break.type) {
    switch(result.obj.i) {
    case G__BLOCK_BREAK:
#ifdef G__OLDIMPLEMENTATION1625 /* Don't know why following line is here */
      G__fignorestream(";"); 
#endif
      if(result.ref==G__block_goto.ref) {
	/* free breakcontinue buffer */
	if(allocflag) G__free_breakcontinue_list(store_pbreakcontinue);
#ifndef G__OLDIMPLEMENTATION1672
	G__ifswitch = store_ifswitch;
#endif
	return(result);
      }
      else if(!G__asm_noverflow) {
	/* free breakcontinue buffer */
	if(allocflag) G__free_breakcontinue_list(store_pbreakcontinue);
#ifndef G__OLDIMPLEMENTATION1672
	G__ifswitch = store_ifswitch;
#endif
	return(G__null);
      }
      executed_break=1;
      /* No break here, continue to following case */
    case G__BLOCK_CONTINUE:
      if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"    G__no_exec_compile set(%d)\n"
			       ,G__no_exec_compile);
#endif
	/* At this point, G__no_exec_compile must be 1 but not sure */
	store_no_exec_compile=G__no_exec_compile;
	G__no_exec_compile=1;
	G__mparen=1;
	G__exec_statement();
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"    G__no_exec_compile reset\n");
#endif
	G__no_exec_compile=store_no_exec_compile;
      }
      break;
    }
  }
  
  /* spaceflag=0; */
  /* c='\0'; */
  
  /* G__fignorestream("("); */
  G__fgetstream(condition,"(");
  if(strcmp(condition,"while")!=0) {
    G__fprinterr(G__serr,"Syntax error: do{}%s(); Should be do{}while(); FILE:%s LINE:%d\n"
	    ,condition
	    ,G__ifile.name
	    ,G__ifile.line_number);
  }
  /* iout=0; */
  
  
  G__fgetstream(condition,")");
  
  if(G__breaksignal) {
    G__break=0;
    G__setdebugcond();
    G__pause();
    if(G__return>G__RETURN_NORMAL) {
      /* free breakcontinue buffer */
      if(allocflag) G__free_breakcontinue_list(store_pbreakcontinue);
#ifndef G__OLDIMPLEMENTATION1672
      G__ifswitch = store_ifswitch;
#endif
      return(G__null);
    }
  }

  store_asm_cp=G__asm_cp;

  store_no_exec_compile=G__no_exec_compile;
  if(executed_break) G__no_exec_compile=1;
  cond = G__test(condition);
  if(executed_break) G__no_exec_compile=store_no_exec_compile;
#ifndef G__OLDIMPLEMENTATION906
  if(G__p_tempbuf->level>=G__templevel && G__p_tempbuf->prev) {
    G__free_tempobject();
  }
#endif

#ifdef G__ASM
  if(G__asm_noverflow) {
    /****************************
     * CNDJMP
     ****************************/
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: CNDJMP\n" ,G__asm_cp);
#endif
    G__asm_inst[G__asm_cp]=G__CNDJMP;
    G__asm_inst[G__asm_cp+1]=G__asm_cp+4;
    G__inc_cp_asm(2,0);
    /****************************
     * JMP 0
     ****************************/
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: JMP %x\n",G__asm_cp,asm_start_pc);
#endif
    G__asm_inst[G__asm_cp]=G__JMP;
    G__asm_inst[G__asm_cp+1]=asm_start_pc;
    G__inc_cp_asm(2,0);
    /* Check breakcontinue buffer and assign destination 
     *    break     G__asm_cp
     *    continue  store_asm_cp 
     *  Then free breakcontinue buffer */
    if(allocflag) {
      G__set_breakcontinue_destination(G__asm_cp,store_asm_cp
				       ,store_pbreakcontinue);
      allocflag=0;
    }
    G__asm_inst[G__asm_cp]=G__RETURN;
    asm_exec=1;
    G__abortbytecode();
    /* local compile assembler optimize */
    if(G__asm_loopcompile>=2) G__asm_optimize(&asm_start_pc);
  }
  else {
    /* free breakcontinue buffer */
    if(allocflag) {
      G__free_breakcontinue_list(store_pbreakcontinue);
      allocflag=0;
    }
    asm_exec=0;
  }
#endif /* G__ASM */


  if(G__no_exec_compile||executed_break) cond=0;

  while(cond) {
    
#ifdef G__ASM
    if(asm_exec) {
      asm_exec=G__exec_asm(asm_start_pc,/*stack*/0,&result,/*localmem*/0);
      if(G__return!=G__RETURN_NON) { 
#ifndef G__OLDIMPLEMENTATION1672
	G__ifswitch = store_ifswitch;
#endif
	return(result);
      }
      break;
    }
    else {
#endif
      G__ifile.line_number=store_line_number;
      fsetpos(G__ifile.fp,&store_fpos);
      if(G__debug) G__fprinterr(G__serr,"\n%-5d",G__ifile.line_number);
      G__no_exec=0;
      G__mparen=0;
      result=G__exec_statement();
      if(G__return!=G__RETURN_NON) {
#ifndef G__OLDIMPLEMENTATION1672
	G__ifswitch = store_ifswitch;
#endif
	return(result);
      }
      
      /*************************************************
       * do {
       *    if() break;
       * } while();
       *************************************************/
      if(result.type==G__block_break.type) {
	switch(result.obj.i) {
	case G__BLOCK_BREAK:
	  G__fignorestream(";");
#ifndef G__OLDIMPLEMENTATION1672
	  G__ifswitch = store_ifswitch;
#endif
	  if(result.ref==G__block_goto.ref) return(result);
	  else                              return(G__null);
	  /* No break here intentionally */
	case G__BLOCK_CONTINUE:
	  /* G__asm_noverflow never be 1 here */
	  break;
	}
      }
      
      /* spaceflag=0; */
      
      if(G__breaksignal) {
	G__break=0;
	G__setdebugcond();
	G__pause();
	if(G__return>G__RETURN_NORMAL) {
#ifndef G__OLDIMPLEMENTATION1672
	  G__ifswitch = store_ifswitch;
#endif
	  return(G__null);
	}
      }
      
#ifdef G__ASM
    }
#endif
    cond = G__test(condition);
  } /* while(cond) */

  G__disp_mask=10000;
  G__fignorestream(";");
  G__disp_mask=0;
  if(G__debug) G__fprinterr(G__serr,";");
  
#ifdef G__ASM
  G__asm_noverflow = asm_exec && store_asm_noverflow;
#endif
  
  G__no_exec=0;
  
#ifndef G__OLDIMPLEMENTATION1672
  G__ifswitch = store_ifswitch;
  result = G__null;
#endif
  return(result);
}


/***********************************************************************
* G__return_value()
*
* Called by
*    G__exec_statement   'return;'
*    G__exec_statement   'return(result);'
*    G__exec_statement   'return result;'
*
***********************************************************************/
G__value G__return_value(statement)
char *statement;
{
  G__value buf;
						   
  if(G__breaksignal) {
    G__break=0;
    G__setdebugcond();
    G__pause();
    if(G__return>G__RETURN_NORMAL) return(G__null);
  }
  if(G__dispsource) {
    if((G__debug||G__break||G__step)&&
       ((G__prerun!=0)||(G__no_exec==0))&&
       (G__disp_mask==0)){
      G__fprinterr(G__serr,"}\n");
    }
  }

  /* questionable, when to destroy temp object */
  if(G__p_tempbuf->level>=G__templevel && G__p_tempbuf->prev) 
    G__free_tempobject();

  if(statement[0]=='\0') {
    G__no_exec=1;
    buf=G__null;
  }
  else {
    G__no_exec=0;
#ifndef G__OLDIMPLEMENTATION509
    --G__templevel;
    buf=G__getexpr(statement);
    ++G__templevel;
#else
    buf=G__getexpr(statement);
#endif
  }

  if(G__no_exec_compile) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: RTN_FUNC\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp] = G__RTN_FUNC;
    if(statement[0]) G__asm_inst[G__asm_cp+1] = 1;
    else             G__asm_inst[G__asm_cp+1] = 0;
    G__inc_cp_asm(2,0);
  }
  else {
    G__abortbytecode();
    G__return=G__RETURN_NORMAL;
  }
  return(buf);
}

/***********************************************************************
 * G__display_tempobject()
 ***********************************************************************/
void G__display_tempobject(action) 
char* action;
{
  struct G__tempobject_list *ptempbuf = G__p_tempbuf;
  G__fprinterr(G__serr,"\n%s ",action);
  while(ptempbuf) {
    if(ptempbuf->obj.type) {
      G__fprinterr(G__serr,"%d:(%s)0x%p ",ptempbuf->level
		   ,G__type2string(ptempbuf->obj.type,ptempbuf->obj.tagnum
				   ,ptempbuf->obj.typenum
				   ,ptempbuf->obj.obj.reftype.reftype
				   ,ptempbuf->obj.isconst)
		   ,(void*)ptempbuf->obj.obj.i);
    }
    else {
      G__fprinterr(G__serr,"%d:(%s)0x%p ",ptempbuf->level,"NULL",(void*)0);
    }
    ptempbuf = ptempbuf->prev;
  }
  G__fprinterr(G__serr,"\n");
}

/***********************************************************************
* G__free_tempobject()
*
* Called by
*    G__exec_statement()     at ';'
*    G__pause()              'p expr'
*
***********************************************************************/
void G__free_tempobject()
{
  long store_struct_offset; /* used to be int */
  int store_tagnum;
  int iout=0;
  int store_return;
#ifndef G__OLDIMPLEMENTATION1596
   /* The only 2 potential risks of making this static are
    * - a destructor indirectly calls G__free_tempobject
    * - multi-thread application (but CINT is not multi-threadable anyway). */
  static char statement[G__ONELINE];
#else
  char statement[G__ONELINE];
#endif
  struct G__tempobject_list *store_p_tempbuf;

#ifndef G__OLDIMPLEMENTATION1164
  if(G__xrefflag
#ifndef G__OLDIMPLEMENTATION1476
#ifndef G__OLDIMPLEMENTATION1675
     || (G__command_eval && G__DOWHILE!=G__ifswitch)
#else
     || G__command_eval
#endif
#endif
     ) return;
#endif

#ifdef G__ASM_DBG
  if(G__asm_dbg) G__display_tempobject("freetemp");
#endif

  /*****************************************************
   * free temp object buffer
   *****************************************************/
  while(G__p_tempbuf->level >= G__templevel && G__p_tempbuf->prev) {


#ifdef G__ASM_DBG
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"free_tempobject(%d)=0x%lx\n"
	      ,G__p_tempbuf->obj.tagnum,G__p_tempbuf->obj.obj.i);
    }
#endif
    
    store_p_tempbuf = G__p_tempbuf->prev;
    
    /* calling destructor */
    store_struct_offset = G__store_struct_offset;
    G__store_struct_offset = G__p_tempbuf->obj.obj.i;
    
    
#ifdef G__ASM
    if(G__asm_noverflow 
#ifndef G__ASM_IFUNC
       && G__p_tempbuf->cpplink
#endif
       ) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SETTEMP\n" ,G__asm_cp);
#endif
      G__asm_inst[G__asm_cp] = G__SETTEMP;
      G__inc_cp_asm(1,0);
    }
#endif
    
    store_tagnum = G__tagnum;
    G__tagnum = G__p_tempbuf->obj.tagnum;
    
    store_return=G__return;
    G__return=G__RETURN_NON;
    
#ifndef G__OLDIMPLEMENTATION1516
    if(0==G__p_tempbuf->no_exec
#ifndef G__OLDIMPLEMENTATION1626
       || 1==G__no_exec_compile
#endif
       ) {
#endif
      if(G__dispsource) {
	G__fprinterr(G__serr,
		     "!!!Destroy temp object (%s)0x%lx createlevel=%d destroylevel=%d\n"
		     ,G__struct.name[G__tagnum]
		     ,G__p_tempbuf->obj.obj.i
		     ,G__p_tempbuf->level,G__templevel);
      }
      
      sprintf(statement,"~%s()",G__struct.name[G__tagnum]);
      G__getfunction(statement,&iout,G__TRYDESTRUCTOR); 
#ifndef G__OLDIMPLEMENTATION1516
    }
#endif
    
    G__store_struct_offset = store_struct_offset;
    G__tagnum = store_tagnum;
    G__return=store_return;
    
    
    
#ifdef G__ASM
    if(G__asm_noverflow 
#ifndef G__ASM_IFUNC
       && G__p_tempbuf->cpplink
#endif
       ) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: FREETEMP\n" ,G__asm_cp);
#endif
      G__asm_inst[G__asm_cp] = G__FREETEMP;
      G__inc_cp_asm(1,0);
    }
#endif
    
    if(0==G__p_tempbuf->cpplink && G__p_tempbuf->obj.obj.i) {
      free((void *)G__p_tempbuf->obj.obj.i);
    }
    
    if(store_p_tempbuf) {
      free((void*)G__p_tempbuf);
      G__p_tempbuf = store_p_tempbuf;
      if(G__dispsource) {
	if(G__p_tempbuf->obj.obj.i==0) {
	  G__fprinterr(G__serr,"!!!No more temp object\n");
	}
      }
    }
    else {
      if(G__dispsource) {
	G__fprinterr(G__serr,"!!!no more temp object\n");
      }
    }
  }
}

/***********************************************************************
* G__alloc_tempstring()
*
***********************************************************************/
G__value G__alloc_tempstring(string)
char *string;
{
  /* int len; */
  struct G__tempobject_list *store_p_tempbuf;

#ifndef G__OLDIMPLEMENTATION1164
  if(G__xrefflag) return(G__null);
#endif

  /* create temp object buffer */
  store_p_tempbuf = G__p_tempbuf;
  G__p_tempbuf = (struct G__tempobject_list *)malloc(
				     sizeof(struct G__tempobject_list)
						     );
  G__p_tempbuf->prev = store_p_tempbuf;
  G__p_tempbuf->level = G__templevel;
  G__p_tempbuf->cpplink = 0;
#ifndef G__OLDIMPLEMENTATION1516
  G__p_tempbuf->no_exec = 0;
#endif
  
  /* create class object */
  G__p_tempbuf->obj.obj.i = (long)malloc(strlen(string)+1);
  strcpy((char*)G__p_tempbuf->obj.obj.i,string);
  G__p_tempbuf->obj.type = 'C';
  G__p_tempbuf->obj.tagnum = -1;
  G__p_tempbuf->obj.typenum = -1;
  G__p_tempbuf->obj.ref = 0;

#ifdef G__DEBUG
  if(G__asm_dbg)
    G__fprinterr(G__serr,"alloc_tempstring(%s)=0x%lx\n",string
	    ,G__p_tempbuf->obj.obj.i);
#endif

  return(G__p_tempbuf->obj);
}

/***********************************************************************
* G__alloc_tempobject()
*
* Called by
*    G__interpret_func
*    G__param_match
*
*  Used for interpreted classes
*
***********************************************************************/
void G__alloc_tempobject(tagnum,typenum)
int tagnum,typenum;
{
  struct G__tempobject_list *store_p_tempbuf;

  G__ASSERT( 0<=tagnum );

#ifndef G__OLDIMPLEMENTATION1164
  if(G__xrefflag) return;
#endif

  /* create temp object buffer */
  store_p_tempbuf = G__p_tempbuf;
  G__p_tempbuf = (struct G__tempobject_list *)malloc(
				     sizeof(struct G__tempobject_list)
						     );
  G__p_tempbuf->prev = store_p_tempbuf;
  G__p_tempbuf->level = G__templevel;
  G__p_tempbuf->cpplink = 0;
#ifndef G__OLDIMPLEMENTATION1516
  G__p_tempbuf->no_exec = G__no_exec_compile;
#endif
  
  /* create class object */
  G__p_tempbuf->obj.obj.i = (long)malloc((size_t)G__struct.size[tagnum]);
#ifndef G__OLDIMPLEMENTATION1978
  G__p_tempbuf->obj.obj.reftype.reftype = G__PARANORMAL;
#endif
  G__p_tempbuf->obj.type = 'u';
  G__p_tempbuf->obj.tagnum = tagnum;
  G__p_tempbuf->obj.typenum = typenum;
  G__p_tempbuf->obj.ref = G__p_tempbuf->obj.obj.i;

#ifdef G__DEBUG
  if(G__asm_dbg) {
    G__fprinterr(G__serr,"alloc_tempobject(%d,%d)=0x%lx\n",tagnum,typenum,
	    G__p_tempbuf->obj.obj.i);
  }
#endif
#ifdef G__DEBUG
  if(G__asm_dbg) G__display_tempobject("alloctemp");
#endif
}

/***********************************************************************
* G__store_tempobject()
*
* Called by
*    G__interpret_func
*    G__param_match
*
*  Used for precompiled classes
*
***********************************************************************/
void G__store_tempobject(reg)
G__value reg;
{
  struct G__tempobject_list *store_p_tempbuf;

  /* G__ASSERT( 'u'==reg.type || '\0'==reg.type ); */

#ifndef G__OLDIMPLEMENTATION1164
  if(G__xrefflag) return;
#endif

#ifdef G__NEVER
  if('u'!=reg.type) {
#ifndef G__FONS31
    G__fprinterr(G__serr,"%d %d %d %ld\n"
	    ,reg.type,reg.tagnum,reg.typenum,reg.obj.i);
#else
    G__fprinterr(G__serr,"%d %d %d %d\n",reg.type,reg.tagnum,reg.typenum,reg.obj.i);
#endif
  }
#endif

  /* create temp object buffer */
  store_p_tempbuf = G__p_tempbuf;
  G__p_tempbuf = (struct G__tempobject_list *)malloc(
				     sizeof(struct G__tempobject_list)
						     );
  G__p_tempbuf->prev = store_p_tempbuf;
  G__p_tempbuf->level = G__templevel;
  G__p_tempbuf->cpplink = 1;
#ifndef G__OLDIMPLEMENTATION1516
  G__p_tempbuf->no_exec = G__no_exec_compile;
#endif

  /* copy pointer to created class object */
  G__p_tempbuf->obj = reg;

#ifdef G__DEBUG
  if(G__asm_dbg) {
    G__fprinterr(G__serr,"store_tempobject(%d)=0x%lx\n",reg.tagnum,reg.obj.i);
  }
#endif
#ifdef G__DEBUG
  if(G__asm_dbg) G__display_tempobject("storetemp");
#endif
}

/***********************************************************************
* G__pop_tempobject()
*
* Called by
*    G__getfunction
*
***********************************************************************/
int G__pop_tempobject()
{
  struct G__tempobject_list *store_p_tempbuf;

#ifndef G__OLDIMPLEMENTATION1164
  if(G__xrefflag) return(0);
#endif

#ifdef G__DEBUG
  if(G__asm_dbg) {
    G__fprinterr(G__serr,"pop_tempobject(%d)=0x%lx\n"
	    ,G__p_tempbuf->obj.tagnum ,G__p_tempbuf->obj.obj.i);
  }
#endif
#ifdef G__DEBUG
  if(G__asm_dbg) G__display_tempobject("poptemp");
#endif

  store_p_tempbuf = G__p_tempbuf->prev;
  /* free the object buffer only if interpreted classes are stored */
  if(-1!=G__p_tempbuf->cpplink && G__p_tempbuf->obj.obj.i) {
    free((void *)G__p_tempbuf->obj.obj.i);
  }
  free((void *)G__p_tempbuf);
  G__p_tempbuf = store_p_tempbuf;
  return(0);
}

/***********************************************************************
* G__exec_breakcontinue()
*
***********************************************************************/
static int G__exec_breakcontinue(statement,piout,pspaceflag,pmparen
				 ,breakcontinue)
char *statement;
int *piout;
int *pspaceflag;
int *pmparen;
int breakcontinue; /* 0: continue, 1:break */
{
#ifndef G__OLDIMPLEMENTATION1717
  int store_no_exec_compile = G__no_exec_compile;
#endif
#ifdef G__ASM
  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: JMP assigned later\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp] = G__JMP;
    G__store_breakcontinue_list(G__asm_cp+1,breakcontinue);
    G__inc_cp_asm(2,0);
    if(G__no_exec_compile) {
      statement[0]='\0';
      *piout=0;
      *pspaceflag=0;
      return(0);
    }
    else { 
      G__no_exec_compile = 1;
#ifndef G__OLDIMPLEMENTATION744
      if(0==breakcontinue) { /* in case of continue */
	statement[0]='\0';
	*piout=0;
	*pspaceflag=0;
	return(0);
      }
#endif
    }
  }
#endif /* G__ASM */
  /*  Assuming that break,continue appears only within if,switch clause
   *  skip to the end of if,switch conditional clause. If they appear
   *  in plain for,while loop, which make no sense, a strange behavior 
   *  may be observed. */
#ifndef G__OLDIMPLEMENTATION1672
  if(G__DOWHILE!=G__ifswitch) {
#endif
    while(*pmparen) {
#ifndef G__OLDIMPLEMENTATION1672
      char c=G__fignorestream("}");
      if('}'!=c) {
	G__genericerror("Error: Syntax error, possibly too many parenthesis");
      }
#else
      G__fignorestream("}");
#endif
      --(*pmparen);
    }
#ifndef G__OLDIMPLEMENTATION1672
  }
#endif
#ifndef G__OLDIMPLEMENTATION1695
#ifndef G__OLDIMPLEMENTATION1710
  *piout=0;
#endif
#ifndef G__OLDIMPLEMENTATION1717
  if(store_no_exec_compile) return(0);
  else return(1);
#else
  if(G__no_exec_compile) return(0);
  else return(1);
#endif
#else
  return(1);
#endif
}

/***********************************************************************
* G__exec_switch()
*
* Called by
*   G__exec_statement() 
*
***********************************************************************/
G__value G__exec_switch()
{
  char condition[G__ONELINE];
  G__value result,reg;
  int largestep=0,iout;

#ifndef G__OLDIMPLEMENTATION1672
  int store_ifswitch = G__ifswitch;
  G__ifswitch = G__IFSWITCH;
#endif

#ifndef G__OLDIMPLEMENTATION844
  if(G__ASM_FUNC_COMPILE!=G__asm_wholefunction) {
#endif
#ifdef G__ASM
#ifndef G__OLDIMPLEMENTATION841
    if(G__asm_dbg&&G__asm_noverflow) {
      G__fprinterr(G__serr,"bytecode compile aborted by switch statement");
      G__printlinenum();
    }
#endif
    G__abortbytecode();
#endif
#ifndef G__OLDIMPLEMENTATION844
  }
#endif

  /* get switch(condition)
   *            ^^^^^^^^^
   */
  G__fgetstream(condition,")");

  if(G__breaksignal && G__beforelargestep(condition,&iout,&largestep)>1) {
#ifndef G__OLDIMPLEMENTATION1672
    G__ifswitch = store_ifswitch;
#endif
    return(G__null);
  }

#ifndef G__OLDIMPLEMENTATION844
  if(G__asm_noverflow) {
    int allocflag=0;
    int store_prevcase = G__prevcase;
    int store_switch = G__switch;
    struct G__breakcontinue_list *store_pbreakcontinue=NULL;
    reg=G__getexpr(condition);
    G__prevcase = 0;
    if(G__asm_noverflow) {
      store_pbreakcontinue = G__alloc_breakcontinue_list();
      allocflag=1;
    }
    G__switch=1;
    result=G__exec_statement();
    G__switch=store_switch;
    result=G__null;
   if(G__asm_noverflow) {
     if(allocflag) {
       G__set_breakcontinue_destination(G__asm_cp,G__asm_cp
					,store_pbreakcontinue);
       allocflag=0;
     }
   }
   else {
     if(allocflag) {
       G__free_breakcontinue_list(store_pbreakcontinue);
       allocflag=0;
     }
   }
   if(G__prevcase) G__asm_inst[G__prevcase] = G__asm_cp;
#ifdef G__ASM_DBG
   if(G__asm_dbg) G__fprinterr(G__serr,"   %3x: CNDJMP %x assigned\n",G__prevcase-1,G__asm_cp);
#endif
    G__prevcase = store_prevcase;
#ifndef G__OLDIMPLEMENTATION1672
    G__ifswitch = store_ifswitch;
#endif
    return(result);
  } else
#endif
  if(G__no_exec_compile) {
    G__switch=0;
    G__no_exec=1;
    result=G__exec_statement();
    G__no_exec=0;
    result=G__default;
#ifndef G__OLDIMPLEMENTATION1672
    G__ifswitch = store_ifswitch;
#endif
    return(result);
  }
  else {
    result=G__start;
  }
  reg=G__getexpr(condition);
  G__mparen=0;

  if(largestep) {
    G__afterlargestep(&largestep);
  }

  /* testing cases until condition matches */
  while((result.type!=G__null.type)&& ((result.type!=G__default.type)&&
				       (!G__cmp(result,reg)))) { 
    G__switch=1;
    G__no_exec=1;
    result=G__exec_statement();
    G__mparen=1;
    G__switch=0;
  }
  if(result.type!=G__null.type) {
    /* match or default */
    G__no_exec=0;
    G__mparen=1;
    if(0==G__nobreak && 0==G__disp_mask && 0==G__no_exec_compile &&
       G__srcfile[G__ifile.filenum].breakpoint&&
       G__srcfile[G__ifile.filenum].maxline>G__ifile.line_number) {
      G__srcfile[G__ifile.filenum].breakpoint[G__ifile.line_number]|=G__TRACED;
    }
    result=G__exec_statement();
    if(G__return!=G__RETURN_NON) {
      return(result);
#ifndef G__OLDIMPLEMENTATION1672
      G__ifswitch = store_ifswitch;
#endif
    }
    
    /* break found */
    if(result.type==G__block_break.type&&result.obj.i==G__BLOCK_BREAK) {
#ifndef G__OLDIMPLEMENTATION1672
      G__ifswitch = store_ifswitch;
#endif
      if(result.ref==G__block_goto.ref) return(result);
      else                              return(G__null);
    }
  }
  G__mparen=0;
  G__no_exec=0;
#ifndef G__OLDIMPLEMENTATION1672
  G__ifswitch = store_ifswitch;
#endif
  return(result);
}

/***********************************************************************
* G__exec_if()
*
* Called by
*   G__exec_statement() 
*
***********************************************************************/
G__value G__exec_if()
{
#ifndef G__OLDIMPLEMENTATION1802
  char *condition=(char*)malloc(G__LONGLINE);
#else
  char condition[G__LONGLINE];
#endif
  G__value result;
  int false=0;
  fpos_t store_fpos;
  int    store_line_number;
  int c;
  char statement[10]; /* used to identify comment and else */
  int iout;
  int store_no_exec_compile=0;
  int asm_jumppointer=0;
  int largestep=0;

#ifndef G__OLDIMPLEMENTATION1672
  int store_ifswitch = G__ifswitch;
  G__ifswitch = G__IFSWITCH;
#endif
  
#ifndef G__OLDIMPLEMENTATION837
  G__fgetstream_new(condition,")");
#else
  G__fgetstream(condition,")");
#endif

#ifndef G__OLDIMPLEMENTATION1802
  condition = (char*)realloc((void*)condition,strlen(condition)+10);
#endif
  
  if(G__breaksignal &&
     G__beforelargestep(condition,&iout,&largestep)>1) {
#ifndef G__OLDIMPLEMENTATION1672
    G__ifswitch = store_ifswitch;
#endif
#ifndef G__OLDIMPLEMENTATION1802
    free((void*)condition);
#endif
    return(G__null);
  }
  
  result=G__null;

  
  if(G__test(condition)) {
    if(largestep) G__afterlargestep(&largestep);
#ifndef G__OLDIMPLEMENTATION906
    if(G__p_tempbuf->level>=G__templevel && G__p_tempbuf->prev) {
      G__free_tempobject();
    }
#endif
    if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: CNDJMP assigned later\n",G__asm_cp);
#endif
      G__asm_inst[G__asm_cp]=G__CNDJMP;
      asm_jumppointer = G__asm_cp+1;
      G__inc_cp_asm(2,0);
    }
    G__no_exec=0;
    G__mparen=0;
    result=G__exec_statement();
    if(G__return!=G__RETURN_NON) {
#ifndef G__OLDIMPLEMENTATION1672
      G__ifswitch = store_ifswitch;
#endif
#ifndef G__OLDIMPLEMENTATION1802
      free((void*)condition);
#endif
      return(result);
    }
    false=0;
  }

  else {
    if(largestep) G__afterlargestep(&largestep);
#ifndef G__OLDIMPLEMENTATION906
    if(G__p_tempbuf->level>=G__templevel && G__p_tempbuf->prev) {
      G__free_tempobject();
    }
#endif
    store_no_exec_compile=G__no_exec_compile;
    if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"%3x: CNDJMP assigned later\n",G__asm_cp);
	G__fprinterr(G__serr,"     G__no_exec_compile set(G__exec_if:1) %d\n"
		,store_no_exec_compile);
      }
#endif
      G__asm_inst[G__asm_cp]=G__CNDJMP;
      asm_jumppointer = G__asm_cp+1;
      G__inc_cp_asm(2,0);
      
      G__no_exec_compile=1;
    }
    else {
      G__no_exec=1;
    }
    G__mparen=0;
    G__exec_statement();
    /* if(G__return!=G__RETURN_NON)return(result); */ 
#ifdef G__ASM_DBG
    if(G__asm_dbg)
	G__fprinterr(G__serr,"     G__no_exec_compile %d(G__exec_if:1) %d\n"
		,store_no_exec_compile,G__asm_noverflow);
#endif
    G__no_exec_compile=store_no_exec_compile;
    G__no_exec=0;
    false=1;
  }
  
  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: JMP assigned later\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp]=G__JMP;
    G__inc_cp_asm(2,0);
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"     CNDJMP assigned %x\n",G__asm_cp);
#endif
    G__asm_inst[asm_jumppointer] = G__asm_cp;
    asm_jumppointer = G__asm_cp-1; /* to skip else clause */
  }
  
  fgetpos(G__ifile.fp,&store_fpos);
  store_line_number=G__ifile.line_number;
  
  /* reading else keyword */
  c=' ';
  while(isspace(c)) {
    /* increment temp_read */
    c=G__fgetc();
    G__temp_read++;
#ifndef G__OLDIMPLEMENTATION1375
    while(c=='/' || c=='#') {
#endif
      if(c=='/') {
	c=G__fgetc();
	/*****************************
	 * new
	 *****************************/
	switch(c) {
	case '*':
	  if(G__skip_comment()==EOF) {
#ifndef G__OLDIMPLEMENTATION1672
	    G__ifswitch = store_ifswitch;
#endif
#ifndef G__OLDIMPLEMENTATION1802
	    free((void*)condition);
#endif
	    return(G__null);
	  }
	  break;
	case '/':
	  G__fignoreline();
	break;
	default:
	  G__commenterror();
	  break;
      }
	fgetpos(G__ifile.fp,&store_fpos);
	store_line_number=G__ifile.line_number;
	c=G__fgetc();
	G__temp_read=1;
      }
      else if('#'==c) {
	G__pp_command();
	c=G__fgetc();
	G__temp_read=1;
      }
#ifndef G__OLDIMPLEMENTATION1375
    }
#endif
    if(c==EOF) {
      G__genericerror("Error: unexpected if() { } EOF");
      if(G__key!=0) system("key .cint_key -l execute");
      G__eof=2;
#ifndef G__OLDIMPLEMENTATION1672
      G__ifswitch = store_ifswitch;
#endif
#ifndef G__OLDIMPLEMENTATION1802
      free((void*)condition);
#endif
      return(G__null) ;
    }
  }
  statement[0]=c;
  for(iout=1;iout<=3;iout++) {
    c=G__fgetc();
    /* increment temp_read */
    G__temp_read++;
    if(c==EOF) {
      iout=4;
      statement[0]='\0';
    }
    statement[iout] = c;
  }
  statement[4]='\0';
  
  if(strcmp(statement,"else")==0) {
    /* else execute else clause */
    G__temp_read=0;
    G__mparen=0;
    if(false==1
#ifndef G__OLDIMPLEMENTATION882
       || G__asm_wholefunction
#endif
       ) {
      G__no_exec=0;
      /* false=0; */
      result=G__exec_statement();
    }
    else {
      store_no_exec_compile=G__no_exec_compile;
      if(G__asm_noverflow) {
	G__no_exec_compile=1;
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,
			   "     G__no_exec_compile set(G__exec_if:2) %d\n"
			       ,store_no_exec_compile);
#endif
      }
      else {
	G__no_exec=1;
      }
      G__exec_statement();
      G__no_exec_compile=store_no_exec_compile;
#ifdef G__ASM_DBG
    if(G__asm_dbg) 
	G__fprinterr(G__serr,"     G__no_exec_compile %d(G__exec_if:2) %d\n"
		     ,store_no_exec_compile,G__asm_noverflow);
#endif
      G__no_exec=0;
    }
    if(G__return!=G__RETURN_NON){ 
#ifndef G__OLDIMPLEMENTATION1672
      G__ifswitch = store_ifswitch;
#endif
#ifndef G__OLDIMPLEMENTATION1802
      free((void*)condition);
#endif
      return(result);
    }
  }
  else {  /* no else  push back */
    G__ifile.line_number
			=store_line_number;
    fsetpos(G__ifile.fp,&store_fpos);
    statement[0]='\0';
    if(G__dispsource) 
      G__disp_mask=G__temp_read;
    G__temp_read=0;
  }
  G__no_exec=0;

  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"   %x: JMP assigned %x\n",asm_jumppointer-1,G__asm_cp);
#endif
    G__asm_inst[asm_jumppointer] = G__asm_cp;
#ifndef G__OLDIMPLEMENTATION599
    G__asm_cond_cp=G__asm_cp; /* avoid wrong optimization */
#endif
  }

#ifndef G__OLDIMPLEMENTATION1672
  G__ifswitch = store_ifswitch;
#endif
#ifndef G__OLDIMPLEMENTATION1802
  free((void*)condition);
#endif
  return(result);
}

/***********************************************************************
* G__exec_loop()
*
*
***********************************************************************/
G__value G__exec_loop(forinit,condition,naction,foraction)
char *forinit;
char *condition;
int naction;
char **foraction;
{
  G__value result;
  fpos_t store_fpos;
  int    store_line_number;
#ifdef G__ASM
  int asm_exec=0;
  int asm_start_pc;
  int asm_jumppointer=0;
  int store_asm_noverflow=0;
  int store_asm_cp;
  struct G__breakcontinue_list *store_pbreakcontinue=NULL;
  int allocflag=0;
  int executed_break=0;
#endif
  int largestep=0,iout;
  int cond;
#ifndef G__OLDIMPLEMENTATION756
  int dispstat=0;
#endif
#define G__OLDIMPLEMENTATION1256
#ifndef G__OLDIMPLEMENTATION1256
  int zeroloopflag=0;
#endif

#ifndef G__OLDIMPLEMENTATION1672
  int store_ifswitch = G__ifswitch;
  G__ifswitch = G__DOWHILE;
#endif

  if(G__breaksignal && G__beforelargestep(condition,&iout,&largestep)>1) {
#ifndef G__OLDIMPLEMENTATION1672
    G__ifswitch = store_ifswitch;
#endif
    return(G__null);
  }

  fgetpos(G__ifile.fp,&store_fpos);
  store_line_number=G__ifile.line_number;

  if(forinit) G__getexpr(forinit);

  if(G__p_tempbuf->level>=G__templevel && G__p_tempbuf->prev) {
    G__free_tempobject();
  }
  
#ifdef G__ASM
  if(G__asm_loopcompile) {
    store_asm_noverflow=G__asm_noverflow;
    if(G__asm_noverflow==0) {
      G__asm_noverflow=1;
      G__clear_asm();
    }
    asm_start_pc = G__asm_cp;
#ifdef G__ASM_DBG
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"Loop compile start");
      G__printlinenum();
    }
#endif
  }
#endif

  /* breakcontinue buffer setup */
  if(G__asm_noverflow) {
    store_pbreakcontinue = G__alloc_breakcontinue_list();
    allocflag=1;
  }

  cond = G__test(condition);
#ifndef G__OLDIMPLEMENTATION906
  if(G__p_tempbuf->level>=G__templevel && G__p_tempbuf->prev) {
    G__free_tempobject();
  }
#endif

  if(G__no_exec_compile) cond=1;
#ifndef G__OLDIMPLEMENTATION1256
  else if(!cond && G__asm_noverflow && 0==G__asm_wholefunction && !G__no_exec){
    G__no_exec_compile=1;
    cond=1;
    zeroloopflag=1;
  }
#endif

  if(!cond && allocflag) {
    /* free breakcontinue buffer */
    G__free_breakcontinue_list(store_pbreakcontinue);
    allocflag=0;
  }

  
  while(cond) {
#ifndef G__OLDIMPLEMENTATION744
    int store_no_exec_compile = G__no_exec_compile;
#endif

    G__no_exec=0;
    G__mparen=0;
#ifdef G__ASM
    if(G__asm_noverflow) {
      /****************************
       * condition compile successful
       ****************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: CNDJMP assigned later\n",G__asm_cp);
#endif
      G__asm_inst[G__asm_cp]=G__CNDJMP;
      asm_jumppointer = G__asm_cp+1;
      G__inc_cp_asm(2,0);
#ifndef G__OLDIMPLEMENTATION1019
      G__asm_clear();
#endif
    }
#endif
    result=G__exec_statement();
#ifdef G__OLDIMPLEMENTATION1673_YET
    if(G__p_tempbuf->level>=G__templevel && G__p_tempbuf->prev) {
      G__free_tempobject();
    }
#endif
#ifndef G__OLDIMPLEMENTATION744
    G__no_exec_compile = store_no_exec_compile;
#endif
    if(G__return!=G__RETURN_NON) {
      /* free breakcontinue buffer */
      if(allocflag) G__free_breakcontinue_list(store_pbreakcontinue);
#ifndef G__OLDIMPLEMENTATION1672
      G__ifswitch = store_ifswitch;
#endif
      return(result);
    }
    
    /*************************************************
     * for() {
     *    if() break;
     * }
     *************************************************/
    if(result.type==G__block_break.type) {
      switch(result.obj.i) {
      case G__BLOCK_BREAK:
	if(largestep) G__afterlargestep(&largestep);
	if(result.ref==G__block_goto.ref) {
	  /* free breakcontinue buffer */
	  if(allocflag) G__free_breakcontinue_list(store_pbreakcontinue);
#ifndef G__OLDIMPLEMENTATION1672
	  G__ifswitch = store_ifswitch;
#endif
	  return(result);
	}
	else if(!G__asm_noverflow) {
	  /* free breakcontinue buffer */
	  if(allocflag) G__free_breakcontinue_list(store_pbreakcontinue);
#ifndef G__OLDIMPLEMENTATION1672
	  G__ifswitch = store_ifswitch;
#endif
	  return(G__null);
	}
	executed_break=1;
	/* No break here intentionally */
      case G__BLOCK_CONTINUE:
	if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	  if(G__asm_dbg) G__fprinterr(G__serr,"    G__no_exec_compile set(%d)\n"
				 ,G__no_exec_compile);
#endif
	  /* At this point, G__no_exec_compile must be 1 but not sure */
	  G__no_exec_compile=1;
	  G__mparen=1;
	  G__exec_statement();
#ifdef G__ASM_DBG
	  if(G__asm_dbg) G__fprinterr(G__serr,"    G__no_exec_compile reset\n");
#endif
	  G__no_exec_compile=0;
	}
	break;
      }
    }
    
    G__ifile.line_number =store_line_number;
    fsetpos(G__ifile.fp,&store_fpos);

    store_asm_cp = G__asm_cp;
    
    if(executed_break) G__no_exec_compile=1;
    if(naction) {
      for(iout=0;iout<naction;iout++) G__getexpr(foraction[iout]);
    }
    if(executed_break) G__no_exec_compile=0;

#ifdef G__ASM
    if(G__asm_noverflow) {
      /**************************
       * JMP
       **************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: JMP %x\n",G__asm_cp,asm_start_pc);
#endif
      G__asm_inst[G__asm_cp]=G__JMP;
      G__asm_inst[G__asm_cp+1]=asm_start_pc;
      G__inc_cp_asm(2,0);
      /* Check breakcontinue buffer and assign destination 
       *    break     G__asm_cp
       *    continue  store_asm_cp 
       *  Then free breakcontinue buffer */
      if(allocflag) {
	G__set_breakcontinue_destination(G__asm_cp,store_asm_cp
					 ,store_pbreakcontinue);
	allocflag=0;
      }
      G__asm_inst[G__asm_cp]=G__RETURN;
      G__asm_inst[asm_jumppointer]=G__asm_cp;
       /* local compile assembler optimize */
      if(G__asm_loopcompile>=2) G__asm_optimize(&asm_start_pc);
      if(executed_break) break;
      asm_exec=1;
      G__asm_noverflow=0;

#ifndef G__OLDIMPLEMENTATION756
      if(G__asm_dbg) {
        G__fprinterr(G__serr,"Bytecode loop compilation successful");
        G__printlinenum();
      }
#endif

      if(0==G__no_exec_compile) {
	asm_exec=G__exec_asm(asm_start_pc,/*stack*/0,&result,/*localmem*/0);
	if(G__return!=G__RETURN_NON) { 
#ifndef G__OLDIMPLEMENTATION1672
	  G__ifswitch = store_ifswitch;
#endif
	  return(result);
	}
      }
      break;
    }
    else {
      /* free breakcontinue buffer */
      if(allocflag) {
	G__free_breakcontinue_list(store_pbreakcontinue);
	allocflag=0;
      }
      asm_exec=0;
      G__asm_noverflow=0;
#ifndef G__OLDIMPLEMENTATION756
      if(G__asm_dbg && 0==dispstat) {
        G__fprinterr(G__serr,"Bytecode loop compilation failed");
        G__printlinenum();
        dispstat = 1;
      }
#endif
    }
#endif

    if(G__no_exec_compile) cond=0;
    else { 
      cond=G__test(condition);
#ifndef G__OLDIMPLEMENTATION906
      if(G__p_tempbuf->level>=G__templevel && G__p_tempbuf->prev) {
	G__free_tempobject();
      }
#endif
    }
  } /* while(G__test) */

#ifdef G__ASM
  G__asm_noverflow = asm_exec && store_asm_noverflow;
#endif

#ifndef G__OLDIMPLEMENTATION1256
  if(zeroloopflag) {
    G__no_exec_compile=0;
    cond=0;
    zeroloopflag=0;
  }
#endif
  /***********************************************
   * skip last for execution
   ***********************************************/
  G__mparen=0;
  G__no_exec=1;
  result=G__exec_statement();
  if(G__return!=G__RETURN_NON) {
#ifndef G__OLDIMPLEMENTATION1672
    G__ifswitch = store_ifswitch;
#endif
    return(result); 
  }
  G__no_exec=0;
  
  if(largestep) G__afterlargestep(&largestep);
#ifndef G__OLDIMPLEMENTATION1672
  G__ifswitch = store_ifswitch;
#endif
  return(result);
}

/***********************************************************************
* G__exec_while()
*
* Called by
*   G__exec_statement() 
*
***********************************************************************/
G__value G__exec_while()
{
#ifndef G__OLDIMPLEMENTATION1802
  char *condition = (char*)malloc(G__LONGLINE);
#else
  char condition[G__LONGLINE];
#endif
  G__value result;
#ifndef G__OLDIMPLEMENTATION1672
  int store_ifswitch = G__ifswitch;
  G__ifswitch = G__DOWHILE;
#endif

  G__fgetstream(condition,")");

#ifndef G__OLDIMPLEMENTATION1802
  condition = (char*)realloc((void*)condition,strlen(condition)+10);
#endif

  result=G__exec_loop((char*)NULL,condition,0,(char**)NULL);
#ifndef G__OLDIMPLEMENTATION1672
  G__ifswitch = store_ifswitch;
#endif

#ifndef G__OLDIMPLEMENTATION1802
  free((void*)condition);
#endif
  return(result);
}

/***********************************************************************
* G__exec_for()
*
* Called by
*   G__exec_statement() 
*
***********************************************************************/
G__value G__exec_for()
{
  /* char forinit[G__ONELINE]; */
#ifndef G__OLDIMPLEMENTATION1802
  int condlen;
  char *condition;
#else
  char condition[G__LONGLINE];
#endif
  char foractionbuf[G__ONELINE];
  char *foraction[10];
  char *p;
  int naction=0;
  int c;
  G__value result;
#ifndef G__OLDIMPLEMENTATION1672
  int store_ifswitch = G__ifswitch;
  G__ifswitch = G__DOWHILE;
#endif

  /* handling 'for(int i=0;i<xxx;i++)' */
  G__exec_statement();
  if(G__return>G__RETURN_NORMAL) {
#ifndef G__OLDIMPLEMENTATION1672
    G__ifswitch = store_ifswitch;
#endif
    return(G__null);
  }

#ifndef G__OLDIMPLEMENTATION1802
  condition=(char*)malloc(G__LONGLINE);
#endif

#ifndef G__OLDIMPLEMENTATION915
  c=G__fgetstream(condition,";)");
  if(')'==c) {
    G__genericerror("Error: for statement syntax error");
#ifndef G__OLDIMPLEMENTATION1672
    G__ifswitch = store_ifswitch;
#endif
#ifndef G__OLDIMPLEMENTATION1802
    free((void*)condition);
#endif
    return(G__null);
  }
#else
  G__fgetstream(condition,";");
#endif
  if('\0'==condition[0]) strcpy(condition,"1");

#ifndef G__OLDIMPLEMENTATION1802
  condlen = strlen(condition);
  condition = (char*)realloc((void*)condition,condlen+10);
#endif

  p=foractionbuf;
  do {
    c=G__fgetstream(p,"),");
#ifndef G__OLDIMPLEMENTATIONF1086
    if(G__return>G__RETURN_NORMAL) {
      G__fprinterr(G__serr,"Error: for statement syntax error. ';' needed\n");
#ifndef G__OLDIMPLEMENTATION1672
      G__ifswitch = store_ifswitch;
#endif
#ifndef G__OLDIMPLEMENTATION1802
      free((void*)condition);
#endif
      return(G__null);
    }
#endif
    foraction[naction++] = p;
    p += strlen(p)+1;
  } while(')'!=c);

  result=G__exec_loop((char*)NULL,condition,naction,foraction);
#ifndef G__OLDIMPLEMENTATION1672
  G__ifswitch = store_ifswitch;
#endif
#ifndef G__OLDIMPLEMENTATION1802
  free((void*)condition);
#endif
  return(result);
}

/***********************************************************************
* G__exec_else_if()
*
* Called by
*   G__exec_statement() 
*
***********************************************************************/
G__value G__exec_else_if()
{
  int iout;
  fpos_t store_fpos;
  int    store_line_number;
  int c;
  char statement[10]; /* only read commend and else */
  G__value result;
#ifndef G__OLDIMPLEMENTATION1672
  int store_ifswitch = G__ifswitch;
  G__ifswitch = G__IFSWITCH;
#endif

#ifdef G__ASM
#ifndef G__OLDIMPLEMENTATION653
  if(0==G__no_exec_compile) 
    G__abortbytecode(); /* this must be redundant, but just in case */
#else
  G__ASSERT(0==G__asm_noverflow);
  G__abortbytecode(); /* this must be redundant, but just in case */
#endif
#endif

  result = G__null;

  /******************************************************
   * G__no_exec==1 when this function is called.
   * nothing is executed in this function, but just skip
   * else if clause.
   *******************************************************/
  G__fignorestream(")");
  G__mparen=0;
  G__exec_statement();
  
  fgetpos(G__ifile.fp,&store_fpos);
  store_line_number=G__ifile.line_number;
  
  /* reading else keyword */
  c=' ';
  while(isspace(c)) {
    /* increment temp_read */
    c=G__fgetc();
    G__temp_read++;
    if(c=='/') {
      c=G__fgetc();
      /***********************
       *
       ***********************/
      switch(c) {
      case '*':
	if(G__skip_comment()==EOF) {
#ifndef G__OLDIMPLEMENTATION1672
	  G__ifswitch = store_ifswitch;
#endif
	  return(G__null);
	}
	break;
      case '/':
	G__fignoreline();
	break;
      default:
	G__commenterror();
	break;
      }
      fgetpos(G__ifile.fp,&store_fpos);
      store_line_number=G__ifile.line_number;
      c=G__fgetc();
      G__temp_read=1;
    }
    else if('#'==c) {
      G__pp_command();
      c=G__fgetc();
      G__temp_read=1;
    }
    if(c==EOF) {
      G__genericerror("Error: unexpected if() { } EOF");
      if(G__key!=0) system("key .cint_key -l execute");
      G__eof=2;
#ifndef G__OLDIMPLEMENTATION1672
      G__ifswitch = store_ifswitch;
#endif
      return(G__null) ;
    }
  }
  statement[0]=c;
  for(iout=1;iout<=3;iout++) {
    c=G__fgetc();
    /* increment temp_read */
    G__temp_read++;
    if(c==EOF) {
      iout=4;
      statement[0]='\0';
    }
    statement[iout] = c;
  }
  statement[4]='\0';
  
  if(strcmp(statement,"else")==0) {
    /* else found. Skip elseclause */
    G__temp_read=0;
		G__mparen=0;
    result=G__exec_statement();
  }
  else {  /* no else  push back */
    G__ifile.line_number
      =store_line_number;
    fsetpos(G__ifile.fp,&store_fpos);
    statement[0]='\0';
    if(G__dispsource) G__disp_mask=G__temp_read;
    G__temp_read=0;
  }
  G__no_exec=0;
  
#ifndef G__OLDIMPLEMENTATION1672
  G__ifswitch = store_ifswitch;
#endif
  return(result);
}

/***********************************************************************
* G__value G__exec_statement()
*
*
*  Execute statement list  { ... ; ... ; ... ; }
***********************************************************************/

G__value G__exec_statement()
/* struct G__input_file *fin; */
{
  G__value result;
  char statement[G__LONGLINE];
  int c;
  char *conststring=NULL;
  int mparen=0;
  int iout=0;
  int spaceflag=0;
  int single_quote=0,double_quote=0;
  int largestep=0;
  fpos_t start_pos;
  int start_line;
#ifndef G__OLDIMPLEMENTATION439
  int commentflag=0;
#endif
#ifndef G__PHILIPPE12
  int add_fake_space = 0;
  int fake_space = 0;
#endif
#ifndef G__PHILIPPE33
  int discard_space = 0;
  int discarded_space = 0;
#endif

  fgetpos(G__ifile.fp,&start_pos);
  start_line=G__ifile.line_number;

  mparen = G__mparen;
  G__mparen=0;

  result=G__null;

  while(1) {
#ifndef G__PHILIPPE12
    fake_space = 0;
    if (add_fake_space && !double_quote && !single_quote) {
      c = ' ';
      add_fake_space = 0;
      fake_space = 1;
    } else 
#endif
      c=G__fgetc();
#ifndef G__PHILIPPE33
    discard_space = 0;
#endif

/*#define G__OLDIMPLEMENTATION781*/
#ifndef G__OLDIMPLEMENTATION781
  read_again:
#endif
    
    switch( c ) {
#ifdef G__OLDIMPLEMENTATIONxxxx_YET
    case ',' : /* column */
      if(!G__ansiheader) break;
#endif
    case ' ' : /* space */
    case '\t' : /* tab */
    case '\n': /* end of line */
    case '\r': /* end of line */
    case '\f': /* end of line */
#ifndef G__OLDIMPLEMENTATION439
      commentflag=0;
#endif
      /* ignore these character */
      if((single_quote!=0)||(double_quote!=0)) {
	statement[iout++] = c ;
      }
      else {
#ifndef G__OLDIMPLEMENTATION2034
      after_replacement:
#endif
	
#ifndef G__PHILIPPE33
        if (!fake_space) discard_space = 1;
#endif
	if(spaceflag==1) {
	  statement[iout] = '\0' ;
	  /* search keyword */
	G__preproc_again:
#ifndef G__OLDIMPLEMENTATION917
	  if(statement[0]=='#'&&isdigit(statement[1])) {
	    if(G__setline(statement,c,&iout)) goto G__preproc_again;
	    spaceflag = 0;
	    iout=0;
	  }
#endif
	  switch(iout) {
	  case 1: /* # [line] <[filename]> */
	    if(statement[0]=='#') {
	      if(G__setline(statement,c,&iout)) goto G__preproc_again;
	      spaceflag = 0;
	      iout=0;
	    }
	    break;
	  case 2: /* comment '#! xxxxxx' */
	    if(statement[0]=='#') {
	      if(c!='\n' && c!='\r') G__fignoreline();
	      spaceflag = 0;
	      iout=0;
	    }
	    break;
	    
	  case 3:
	    /***********************************
	     * 1)
	     *  #if condition <---
	     *  #endif 
	     * 2)
	     *  #if condition <---
	     *  #else
	     *  #endif
	     ***********************************/
	    if(strcmp(statement,"#if")==0) {
#ifndef G__OLDIMPLEMENTATION1929
	      int stat = G__pp_if();
	      if(stat==G__IFDEF_ENDBLOCK) return(G__null);
#else
	      G__pp_if();
#endif
	      spaceflag = 0;
	      iout=0;
	    }
	    break;
	  case 4:
	    if((mparen==1)&& (strcmp(statement,"case")==0)) {
	      char casepara[G__ONELINE];
	      G__fgetstream(casepara,":");
#ifndef G__OLDIMPLEMENTATION1811
	      c=G__fgetc();
	      while(':'==c) {
		int lenxxx;
		strcat(casepara,"::");
		lenxxx=strlen(casepara);
		G__fgetstream(casepara+lenxxx,":");
		c=G__fgetc();
	      }
	      fseek(G__ifile.fp,-1,SEEK_CUR);
	      G__disp_mask=1;
#endif
	      if(G__switch!=0) {
		int store_no_execXX;
#ifndef G__OLDIMPLEMENTATION844
                int jmp1=0;
		iout=0;
		spaceflag=0;
                if(G__asm_noverflow) {
                  if(G__prevcase) {
#ifdef G__ASM_DBG
                    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: JMP (assigned later)\n" ,G__asm_cp);
#endif
                    G__asm_inst[G__asm_cp]=G__JMP;
                    jmp1=G__asm_cp+1;
                    G__inc_cp_asm(2,0);
                    G__asm_inst[G__prevcase] = G__asm_cp;
#ifdef G__ASM_DBG
                    if(G__asm_dbg) G__fprinterr(G__serr,"   %3x: CNDJMP %x assigned\n",G__prevcase-1,G__asm_cp);
#endif
                  }
#ifdef G__ASM_DBG
                  if(G__asm_dbg) G__fprinterr(G__serr,"%3x: PUSHCPY\n" ,G__asm_cp);
#endif
                  G__asm_inst[G__asm_cp]=G__PUSHCPY;
                  G__inc_cp_asm(1,0);
		  store_no_execXX=G__no_exec;
		  G__no_exec=0;
		  result=G__getexpr(casepara);
		  G__no_exec=store_no_execXX;
#ifdef G__ASM_DBG
                  if(G__asm_dbg) G__fprinterr(G__serr,"%3x: OP2_OPTIMIZED ==\n",G__asm_cp);
#endif
                  G__asm_inst[G__asm_cp]=G__OP2_OPTIMIZED;
                  G__asm_inst[G__asm_cp+1]=(long)G__CMP2_equal;
                  G__inc_cp_asm(2,0);
#ifdef G__ASM_DBG
                  if(G__asm_dbg) G__fprinterr(G__serr,"%3x: CNDJMP (assigned later)\n",G__asm_cp);
#endif
                  G__asm_inst[G__asm_cp]=G__CNDJMP;
                  G__prevcase = G__asm_cp+1;
                  G__inc_cp_asm(2,0);
                  if(jmp1) {
                    G__asm_inst[jmp1]=G__asm_cp;
#ifdef G__ASM_DBG
                    if(G__asm_dbg) G__fprinterr(G__serr,"   %3x: JMP %x assigned\n" ,jmp1-1,G__asm_cp);
#endif
                  }
                }
                else {
		  store_no_execXX=G__no_exec;
		  G__no_exec=0;
		  result=G__getexpr(casepara);
		  G__no_exec=store_no_execXX;
		  return(result);
                }
#else
		iout=0;
		spaceflag=0;
		result=G__getexpr(casepara);
		return(result);
#endif
	      }
	      iout=0;
	      spaceflag=0;
	    }
	    break;

	  case 5:
	    /* #else, #elif */
	    if(G__keyword_anytime_5(statement)) {
#ifndef G__OLDIMPLEMENTATION813
	      if(0==mparen&&'#'!=statement[0]) return(G__null);
#endif
	      spaceflag = 0;
	      iout=0;
	    }
	    break;

	  case 6:
	    /* static, #ifdef,#endif,#undef */
#ifndef G__OLDIMPLEMENTATION1929
	    {
	      int stat=G__keyword_anytime_6(statement);
	      if(stat) {
		if(0==mparen&&'#'!=statement[0]) return(G__null);
		if(stat==G__IFDEF_ENDBLOCK) return(G__null);
		spaceflag = 0;
		iout=0;
	      }
	    }
#else /* 1929 */
	    if(G__keyword_anytime_6(statement)) {
#ifndef G__OLDIMPLEMENTATION813
	      if(0==mparen&&'#'!=statement[0]) return(G__null);
#endif
	      spaceflag = 0;
	      iout=0;
	    }
#endif /* 1929 */
	    break;

	  case 7:
	    if((mparen==1)&&(strcmp(statement,"default")==0)){
	      G__fignorestream(":");
	      if(G__switch!=0) {
#ifndef G__OLDIMPLEMENTATION844
                if(G__asm_noverflow) {
                  if(G__prevcase) {
                    G__asm_inst[G__prevcase] = G__asm_cp;
#ifdef G__ASM_DBG
                    if(G__asm_dbg) G__fprinterr(G__serr,"   %3x: CNDJMP %x assigned\n",G__prevcase-1,G__asm_cp);
#endif
                    G__prevcase=0;
                  }
                }
                else {
                  return(G__default);
                }
#else
                return(G__default);
#endif
              }
	      iout=0;
	      spaceflag=0;
	      break;
	    }
	    
	    /* #ifndef , #pragma */
#ifndef G__OLDIMPLEMENTATION1929
            {
	      int stat=G__keyword_anytime_7(statement);
	      if(stat) {
	        if(0==mparen&&'#'!=statement[0]) return(G__null);
	        if(stat==G__IFDEF_ENDBLOCK) return(G__null);
	        spaceflag = 0;
	        iout=0;
	      }
	    }
#else /* 1929 */
	    if(G__keyword_anytime_7(statement)) {
#ifndef G__OLDIMPLEMENTATION813
	      if(0==mparen&&'#'!=statement[0]) return(G__null);
#endif
	      spaceflag = 0;
	      iout=0;
	    }
#endif /* 1929 */
	    break;

	  case 8:
	    /**********************************
	     * BUGS:
	     *   default:printf("abc");
	     *          ^ 
	     * If there is no space char in
	     * either side of ':', default is
	     * ignored.
	     **********************************/
	    if((mparen==1)&& (strcmp(statement,"default:")==0)){
	      if(G__switch) {
		if(0==G__nobreak && 0==G__disp_mask&& 0==G__no_exec_compile &&
		   G__srcfile[G__ifile.filenum].breakpoint&&
		   G__srcfile[G__ifile.filenum].maxline>G__ifile.line_number) {
		  G__srcfile[G__ifile.filenum].breakpoint[G__ifile.line_number]
		    |=G__TRACED;
		}
#ifndef G__OLDIMPLEMENTATION844
                if(G__asm_noverflow) {
                  if(G__prevcase) {
                    G__asm_inst[G__prevcase] = G__asm_cp;
#ifdef G__ASM_DBG
                    if(G__asm_dbg) G__fprinterr(G__serr,"   %3x: CNDJMP %x assigned\n",G__prevcase-1,G__asm_cp);
#endif
                    G__prevcase=0;
                  }
                }
                else {
                  return(G__default);
                }
#else
		return(G__default);
#endif
	      }
	      iout=0;
	      spaceflag=0;
	      break;
	    }
#ifndef G__OLDIMPLEMENTATION500
	    if(G__keyword_anytime_8(statement)) {
#ifndef G__OLDIMPLEMENTATION813
	      if(0==mparen&&'#'!=statement[0]) return(G__null);
#endif
	      spaceflag = 0;
	      iout=0;
	    }
#endif
	    break;
	  }
	}
	
	if(1==spaceflag&&0==G__no_exec) {
	  statement[iout] = '\0' ;
	  /* search keyword */

	  if(
#ifndef G__OLDIMPLEMENTATION1262
	     iout>3
#else
	     iout>4
#endif
	     &&'*'==statement[iout-2]&&
	     ('*'==statement[iout-1]||'&'==statement[iout-1])) {
	    /* char**, char*&, int**, int*& */
	    statement[iout-2]='\0';
	    iout-=2;
#ifndef G__PHILIPPE12
            if (fake_space) fseek(G__ifile.fp,-2,SEEK_CUR);
            else 
#endif
	      fseek(G__ifile.fp,-3,SEEK_CUR);
	    if(G__dispsource) G__disp_mask=2;
	  }
	  
	  switch(iout){
	  case 2:
	    if(strcmp(statement,"do")==0) {
	    do_do:
	      result=G__exec_do();
	      if(mparen==0||
#ifndef G__OLDIMPLEMENTATION1844
		 G__return>G__RETURN_NON
#else
		 G__return!=G__RETURN_NON
#endif
		 ) return(result); 
	      if(result.type==G__block_goto.type&&
		 result.ref==G__block_goto.ref) {
		if(0==G__search_gotolabel((char*)NULL,&start_pos,start_line
					 ,&mparen))
		   return(G__block_goto);
	      }
	      iout=0;
	      spaceflag= -1;
	    }
	    break;
	  case 3:
	    if(strcmp(statement,"int")==0) {
#if 0
	      G__DEFVAR('i');
#else
	      G__var_type='i' + G__unsigned;
	      G__define_var(-1,-1);          
	      spaceflag = -1;                
	      iout=0;                      
	      if(mparen==0) return(G__null);
#endif
	      break;
	    }
#ifndef G__OLDIMPLEMENTATION698
	    if(strcmp(statement,"new")==0) {
#ifndef G__OLDIMPLEMENTATION784
	      c=G__fgetspace();
	      if('('==c) {
		fseek(G__ifile.fp,-1,SEEK_CUR);
		if(G__dispsource) G__disp_mask=1;
		statement[iout++] = ' ' ;
		spaceflag |= 1;
	      }
	      else {
		statement[0]=c;
		c=G__fgetstream_template(statement+1,";");
		result=G__new_operator(statement);
		spaceflag = -1;
		iout=0;
#ifndef G__OLDIMPLEMENTATION698
		if(0==mparen) return(result);
#endif
	      }
#else
	      statement[iout++] = c ;
	      spaceflag |= 1;
#endif
	      break;
	    }
#endif
#ifndef G__OLDIMPLEMENTATION754
	    if(strcmp(statement,"try")==0) {
              G__exec_try(statement);
              iout=0;
#ifndef G__OLDIMPLEMENTATION1043
	      spaceflag= -1;
#else
	      spaceflag= 0;
#endif
	      break;
	    }
#endif
	    break;
	  case 4:
	    if(strcmp(statement,"char")==0) {
	      G__DEFVAR('c');
	      break;
	    }
	    if(strcmp(statement,"FILE")==0) {
	      G__DEFVAR('e');
	      break;
	    }
	    if(strcmp(statement,"long")==0) {
	      G__DEFVAR('l');
	      break;
	    }
	    if(strcmp(statement,"void")==0) {
	      G__DEFVAR('y');
	      break;
	    }
#ifndef G__OLDIMPLEMENTATION1604
	    if(strcmp(statement,"bool")==0) {
	      G__DEFVAR('g');
	      break;
	    }
#endif
	    if(strcmp(statement,"int*")==0) {
	      G__typepdecl=1;
	      G__DEFVAR('I');
	      G__typepdecl=0;
	      break;
	    }
	    if(strcmp(statement,"int&")==0) {
	      G__DEFREFVAR('i');
	      break;
	    }
	    if(strcmp(statement,"enum")==0) {
	      G__DEFSTR('e');
	      break;
	    }
	    if(strcmp(statement,"auto")==0) {
	      spaceflag = -1;
	      iout=0;
	      break;
	    }
#ifndef G__OLDIMPLEMENTATION1350
	    if(strcmp(statement,"(new")==0) {
	      c=G__fgetspace();
	      if('('==c) {
		fseek(G__ifile.fp,-1,SEEK_CUR);
		statement[iout++] = ' ' ;
		spaceflag |= 1;
	      }
	      else {
		statement[iout++] = ' ' ;
		statement[iout++]=c;
		if(G__dispsource) G__disp_mask=1;
		c=G__fgetstream_template(statement+iout,")");
		spaceflag |= 1;
		iout = strlen(statement);
		statement[iout++]=c;
	      }
	      break;
	    }
#endif
	    if(strcmp(statement,"goto")==0) {
	      G__CHECK(G__SECURE_GOTO,1,return(G__null));
	      c=G__fgetstream(statement,";"); /* get label */
#ifndef G__OLDIMPLEMENTATION842
	      if(G__ASM_FUNC_COMPILE==G__asm_wholefunction) {
		G__add_jump_bytecode(statement);
	      }
	      else {
#endif
#ifndef G__OLDIMPLEMENTATION841
		if(G__asm_dbg&&G__asm_noverflow) {
		  G__fprinterr(G__serr,"bytecode compile aborted by goto statement");
		  G__printlinenum();
		}
#endif
		G__abortbytecode();
#ifndef G__OLDIMPLEMENTATION842
	      }
#endif 
	      if(G__no_exec_compile) {
		if(0==mparen) return(G__null);
#ifndef G__OLDIMPLEMENTATION842
		if(G__ASM_FUNC_COMPILE!=G__asm_wholefunction) 
		  G__abortbytecode();
#else
		G__abortbytecode();
#endif
		spaceflag = -1;
		iout=0;
		break;
	      }
	      if(0==G__search_gotolabel(statement,&start_pos,start_line
				       ,&mparen))
		return(G__block_goto);
	      spaceflag = -1;
	      iout=0;
	    }
	    break;
	    
	    
	  case 5:
	    if(strcmp(statement,"short")==0) {
	      G__DEFVAR('s');
	      break;
	    }
	    if(strcmp(statement,"float")==0) {
	      G__DEFVAR('f');
	      break;
	    }
	    if(strcmp(statement,"char*")==0) {
	      G__typepdecl=1;
	      G__DEFVAR('C');
	      G__typepdecl=0;
	      break;
	    }
	    if(strcmp(statement,"char&")==0) {
	      G__DEFREFVAR('c');
	      break;
	    }
#ifndef G__OLDIMPLEMENTATION1604
	    if(strcmp(statement,"bool&")==0) {
	      G__DEFREFVAR('g');
	      break;
	    }
#endif
	    if(strcmp(statement,"FILE*")==0) {
	      G__typepdecl=1;
	      G__DEFVAR('E');
	      G__typepdecl=0;
	      break;
	    }
	    if(strcmp(statement,"long*")==0) {
	      G__typepdecl=1;
	      G__DEFVAR('L');
	      G__typepdecl=0;
	      break;
	    }
#ifndef G__OLDIMPLEMENTATION1604
	    if(strcmp(statement,"bool*")==0) {
	      G__typepdecl=1;
	      G__DEFVAR('G');
	      G__typepdecl=0;
	      break;
	    }
#endif
	    if(strcmp(statement,"long&")==0) {
	      G__DEFREFVAR('l');
	      break;
	    }
	    if(strcmp(statement,"void*")==0) {
	      G__typepdecl=1;
	      G__DEFVAR('Y');
	      G__typepdecl=0;
	      break;
	    }
	    if(strcmp(statement,"class")==0) {
	      G__DEFSTR('c');
	      break;
	    }
	    if(strcmp(statement,"union")==0) {
	      G__DEFSTR('u');
	      break;
	    }
#ifndef G__OLDIMPLEMENTATION613
	    if(strcmp(statement,"using")==0) {
	      G__using_namespace();
	      spaceflag = -1;
	      iout=0;
	      break;
	    }
#endif
	    if(strcmp(statement,"throw")==0) {
#ifndef G__OLDIMPLEMENTATION754
              G__exec_throw(statement);
#ifndef G__OLDIMPLEMENTATION1844
	      spaceflag = -1;
	      iout=0;
#else
	      return(G__null);
#endif
#else
	      G__fignorestream(";");
	      G__nosupport("Exception handling");
	      G__return=G__RETURN_NORMAL;
	      return(G__null);
#endif
	    }
	    if(strcmp(statement,"const")==0) {
	      G__constvar = G__CONSTVAR;
	      spaceflag = -1;
	      iout=0;
	    }
	    break;
	    
	  case 6:
	    if(strcmp(statement,"double")==0) {
	      G__DEFVAR('d');
	      break;
	    }
	    if(strcmp(statement,"struct")==0) {
	      G__DEFSTR('s');
	      break;
	    }
	    if(strcmp(statement,"short*")==0) {
	      G__typepdecl=1;
	      G__DEFVAR('S');
	      G__typepdecl=0;
	      break;
	    }
	    if(strcmp(statement,"short&")==0) {
	      G__DEFREFVAR('s');
	      break;
	    }
	    if(strcmp(statement,"float*")==0) {
	      G__typepdecl=1;
	      G__DEFVAR('F');
	      G__typepdecl=0;
	      break;
	    }
	    if(strcmp(statement,"float&")==0) {
	      G__DEFREFVAR('f');
	      break;
	    }
	    
	    if(strcmp(statement,"return")==0) {
	      G__fgetstream_new(statement,";");
	      result=G__return_value(statement);
	      if(G__no_exec_compile) {
		spaceflag = -1;
		iout=0;
		if(mparen==0) return(G__null);
		break;
	      }
	      return(result);
	    }
	    if(strcmp(statement,"delete")==0) {
	      G__handle_delete(&iout ,statement); /* iout==1 if 'delete []' */
	      
	      if(G__exec_delete(statement,&iout,&spaceflag,iout,mparen)) {
		return(G__null);
	      }
	      break;
	    }

	    /* friend, extern, signed */
	    if(G__keyword_exec_6(statement,&iout,&spaceflag,mparen)) {
	      return(G__null);
	    }
	    break;
	    
	  case 7:
	    if(strcmp(statement,"typedef")==0) {
	      G__var_type='t';
	      G__define_type();
	      spaceflag = -1;
	      iout=0;
	      if(mparen==0) return(G__null);
	      break;
	    }
	    if(strcmp(statement,"double*")==0) {
	      G__typepdecl=1;
	      G__DEFVAR('D');
	      G__typepdecl=0;
	      break;
	    }
	    if(strcmp(statement,"double&")==0) {
	      G__DEFREFVAR('d');
	      break;
	    }
	    if(strcmp(statement,"virtual")==0) {
	      G__virtual = 1;
	      spaceflag = -1;
	      iout=0;
	      break;
	    }
#ifndef G__OLDIMPLEMENTATION666
	    if (strcmp (statement, "mutable") == 0) {
	      spaceflag = -1;
	      iout = 0;
	      break;
	    }
#endif
#ifdef G__OLDIMPLEMENTATION434
	    if(statement[0]!='#') break;
	    if(strcmp(statement,"#define")==0){
	      G__var_type='p';
	      G__definemacro=1;
	      G__define();
	      G__definemacro=0;
	      spaceflag = -1;
	      iout=0;
	      if(mparen==0) return(G__null);
	    }
#endif
	    break;
	    
	    
	  case 8:
	    if(strcmp(statement,"unsigned")==0){
	      if(G__unsignedintegral(&spaceflag,&iout,mparen)) return(G__null);
	      break;
	    }
	    if(strcmp(statement,"volatile")==0||
	       strcmp(statement,"register")==0){
	      spaceflag = -1;
	      iout=0;
	      break;
	    }
	    if(strcmp(statement,"delete[]")==0){
	      G__fgetstream(statement,";");
	      if(G__exec_delete(statement,&iout,&spaceflag,1,mparen)) {
		return(G__null);
	      }
	      break;
	    }
	    if(strcmp(statement,"operator")==0) {
	      /* type conversion operator */
#ifndef G__OLDIMPLEMENTATION494
	      int store_tagnum;
#endif
	      do {
#ifndef G__OLDIMPLEMENTATION847
		char oprbuf[G__ONELINE];
		iout = strlen(statement);
		c=G__fgetname(oprbuf,"(");
		switch(oprbuf[0]) {
		case '*':
		case '&':
		  strcpy(statement+iout,oprbuf);
		  break;
		default:
		  statement[iout]=' ';
		  strcpy(statement+iout+1,oprbuf);
		}
#else
		iout = strlen(statement);
		statement[iout]=' ';
		c=G__fgetname(statement+iout+1,"(");
#endif
	      } while('('!=c);
	      iout = strlen(statement);
#ifndef G__OLDIMPLEMENTATION494
	      if(' '==statement[iout-1]) --iout;
	      statement[iout]='\0';
	      result=G__string2type(statement+9);
	      store_tagnum=G__tagnum;
	      G__var_type = result.type;
	      G__typenum = result.typenum;
	      G__tagnum = result.tagnum;
	      statement[iout]='(';
	      statement[iout+1]='\0';
	      G__make_ifunctable(statement);
	      G__tagnum=store_tagnum;
#else
	      statement[iout]='(';
	      statement[iout+1]='\0';
	      G__var_type='y'; /* NEED TO CHANGE */
	      G__make_ifunctable(statement);
#endif
	      iout=0;
	      spaceflag = -1;
#ifdef G__SECURITY
	      if(mparen==0) return(G__null);
#else
	      if(mparen==0||G__return>G__RETURN_NORMAL) return(G__null);
#endif
	      break;
	    }
#ifndef G__OLDIMPLEMENTATION666
	    if (strcmp (statement, "typename") == 0) {
	      spaceflag = -1;
	      iout = 0;
              break;
            }
#endif
	    if(statement[0]!='#') break;
	    if(strcmp(statement,"#include")==0){
	      G__include_file();
	      spaceflag = -1;
	      iout=0;
#ifdef G__SECURITY
	      if(mparen==0) return(G__null);
#else
	      if(mparen==0||G__return>G__RETURN_NORMAL) return(G__null);
#endif
	    }
	    break;
	  case 9:
	    if(strcmp(statement,"namespace")==0) {
	      G__DEFSTR('n');
	      break;
	    }
#ifndef G__OLDIMPLEMENTATION1222
	    if(strcmp(statement,"unsigned*")==0) {
	      G__var_type = 'I'-1;
	      G__unsigned = -1;
	      G__define_var(-1,-1);
	      G__unsigned = 0;
	      spaceflag = -1;
	      iout=0;
	    }
	    else if(strcmp(statement,"unsigned&")==0) {
	      G__var_type = 'i'-1;
	      G__unsigned = -1;
	      G__reftype=G__PARAREFERENCE;
	      G__define_var(-1,-1);
	      G__reftype=G__PARANORMAL;
	      G__unsigned = 0;
	      spaceflag = -1;
	      iout=0;
	    }
#endif
#ifndef G__OLDIMPLEMENTATION872
	    if(strcmp(statement,"R__EXTERN")==0) { 
	      if(G__externignore(&iout,&spaceflag,mparen)) return(G__null);
	      break;
	    }
#endif

#ifndef G__OLDIMPLEMENTATION1882
	  case 13:
	    if(strcmp(statement,"__extension__")==0) { 
	      spaceflag = -1;
	      iout = 0;
	      break;
	    }
#endif
	  }
	  
#ifndef G__OLDIMPLEMENTATION1062
          if (iout && G__execvarmacro_noexec (statement)) {
            spaceflag = 0;
            iout = 0;
            break;
          }
#endif
	  if(iout && G__defined_type(statement ,iout)) {
	    spaceflag = -1;
	    iout=0;
#ifdef G__SECURITY
	    if(mparen==0) return(G__null);
#else
	    if(mparen==0||G__return>G__RETURN_NORMAL) return(G__null);
#endif
	  }
#ifndef G__OLDIMPLEMENTATION949
          else 
#ifndef G__OLDIMPLEMENTATION1425
	    if(iout) 
#endif
	    {
            int namespace_tagnum;
#ifdef G__NEVER 
	    /* This part given by Scott Snyder causes problem on Redhat4.2
	     * Linux 2.0.30  gcc -O. If optimizer is turned off, everything
	     * works fine. Other platforms work fine too */
            if (iout) {
              /* For better error recovery.
                 (G__defined_type can truncate statement...) */
              iout = strlen (statement);
	    }
#endif
            /* Allow for spaces after a scope operator. */
            if (iout >= 2 &&
                statement[iout-1] == ':' && statement[iout-2] == ':') {
              spaceflag = 0;
            }
            /* Allow for spaces before a scope operator. */
#ifndef G__OLDIMPLEMENTATION1343
	    if(!strchr(statement,'.') && !strstr(statement,"->")) {
	      namespace_tagnum = G__defined_tagname(statement,2);
	    }
	    else {
	      namespace_tagnum = -1;
	    }
#else
            namespace_tagnum = G__defined_tagname(statement,2);
#endif
#ifndef G__PHILIPPE8
            if (((namespace_tagnum!=-1) && (G__struct.type[namespace_tagnum]=='n'))
		||(strcmp(statement,"std")==0)) {
#else 
            if ((namespace_tagnum!=-1) && (G__struct.type[namespace_tagnum]=='n')) {
#endif
              spaceflag = 0;	    
            }
          }
#endif
#ifndef G__OLDIMPLEMENTATION2034
          {
	    char* replace = (char*)G__replacesymbol(statement);
	    if(replace!=statement) {
	      strcpy(statement,replace);
	      iout = strlen(statement);
	      goto after_replacement;
	    }
          }
#endif
	  ++spaceflag;
	  G__var_type = 'p';
	} /* 1!=spaceflag&&0==G__no_exec */
      }
#ifdef G__SECURITY  
      if(G__return>G__RETURN_NORMAL) return(G__null);
#endif
      break;
      
      
    case ';' : /* semi-column */
      if((single_quote!=0)||(double_quote!=0)) {
	statement[iout++] = c ;
      }
      else {
	statement[iout] = '\0';
	if(G__breaksignal&&G__beforelargestep(statement ,&iout ,&largestep)>1){
	  return(G__null);
	}
	
	if(G__no_exec==0) {
	  switch(iout){
	  case 5:
	    if(strcmp(statement,"break")==0) {
	      /****************************************
	       * switch(),do,while(),for(),if() {
	       * }  come to this parenthesis if any
	       ****************************************/
	      if(G__exec_breakcontinue(statement,&iout,&spaceflag,&mparen,1)){
		return(G__block_break);
	      }
	    }
	    if(strcmp(statement,"throw")==0) {
	      G__nosupport("Exception handling");
	      G__return=G__RETURN_NORMAL;
	      return(G__null);
	    }
	    break;
	  case 6:
	    if(strcmp(statement,"return")==0) {
	      result=G__return_value("");
	      if(G__no_exec_compile) {
		statement[0]='\0';
		spaceflag = -1;
		iout=0;
		if(mparen==0) return(G__null);
		break;
	      }
	      return(result);
	    }
	    break;
	    
	  case 8:
	    if(strcmp(statement,"continue")==0) {
	      /****************************************
	       * switch(),do,while(),for(),if() {
	       * }  come to this parenthesis if any
	       ****************************************/
	      if(G__exec_breakcontinue(statement,&iout,&spaceflag,&mparen,0)){
		return(G__block_continue);
	      }
	    }
	    break;
	    
	  }
	  
	  /* Normal commands 'xxxx;' */
#ifndef G__OLDIMPLEMENTATION1439
	  if(statement[0] && iout) {
#endif
#ifdef G__ASM
	    if(G__asm_noverflow) G__asm_clear();
#endif
	    result=G__getexpr(statement);
	    
	    if(G__p_tempbuf->level>=G__templevel && G__p_tempbuf->prev) {
	      G__free_tempobject();
	    }
#ifndef G__OLDIMPLEMENTATION1439
	  }
#endif
	}
	
	if(largestep) G__afterlargestep(&largestep);
	iout=0;
	spaceflag=0;
	if(mparen==0 || G__return>G__RETURN_NORMAL) return(result);
      }
      break;
      
      
    case '(' : /* parenthesis */
      statement[iout++] = c ;
      statement[iout] = '\0' ;
      if((single_quote!=0)||(double_quote!=0)) break;
      
      /* if((spaceflag==1)&&(G__no_exec==0)) { */
      if(G__no_exec==0) { 

#ifndef G__OLDIMPLEMENTATION942
	if( (0==G__def_struct_member||strncmp(statement,"ClassDef",8)!=0) &&
	    G__execfuncmacro_noexec(statement)) {
	  spaceflag=0;
	  iout =0;
	  break;
	}
#endif
	
	/* make ifunc table at prerun */
	G__ASSERT(0==G__decl || 1==G__decl);
	if(G__prerun && !G__decl) {
	  /* statement='funcname(' */
	  G__var_type='i';
	  G__make_ifunctable(statement);
	  iout=0;
	  spaceflag=0;
	  /* for member func definition */
	  if(mparen==0||G__return>G__RETURN_NORMAL) return(G__null);

	  /**********************************
	   * This break should work with old
	   * implementation
	   **********************************/
	  break; /* for outer switch */
	}
	
	/* search keyword */
	
	switch(iout){
	case 7:
	  if(strcmp(statement,"return(")==0) {
	    
	    /*************************************
	     * following part used to be
	     * G__fgetstream(statement,")");
	     *************************************/
	    fseek(G__ifile.fp,-1,SEEK_CUR);
	    if(G__dispsource) G__disp_mask=1;
	    G__fgetstream_new(statement,";");
	    
	    result=G__return_value(statement);
	    if(G__no_exec_compile) {
	      statement[0]='\0';
	      spaceflag = -1;
	      iout=0;
	      if(mparen==0) return(G__null);
	      break;
	    }
	    return(result);
	  }
	  
	  if(strcmp(statement,"switch(")==0) {
	    result=G__exec_switch();
	    if(
#ifndef G__OLDIMPLEMENTATION1844
	       G__return>G__RETURN_NON 
#else
	       G__return!=G__RETURN_NON 
#endif
	       || mparen==0) return(result);
	    if(result.type==G__block_goto.type &&
	       result.ref==G__block_goto.ref) {
	      if(0==G__search_gotolabel((char*)NULL,&start_pos,start_line
					,&mparen))
		 return(G__block_goto);
	    }
#ifndef G__OLDIMPLEMENTATION1877
	    if(result.type==G__block_break.type) {
	      if(result.obj.i==G__BLOCK_CONTINUE) {
		return(result);
	      }
	    }
#endif
	    iout=0;
	    spaceflag=0;
	  }

	  break;
	  
	case 3:
	  if(strcmp(statement,"if(")==0) {
	    result=G__exec_if();
	    if(
#ifndef G__OLDIMPLEMENTATION1844
	       G__return>G__RETURN_NON 
#else
	       G__return!=G__RETURN_NON 
#endif
	       || mparen==0) return(result);
	    /************************************
	     * handling break statement
	     * switch(),do,while(),for() {
	     *    if(cond) break; or continue;
	     *    if(cond) {break; or continue;}
	     * } G__fignorestream() skips until here
	     *************************************/
	    if(result.type==G__block_break.type) {
	      if(result.ref==G__block_goto.ref) {
		if(0==G__search_gotolabel((char*)NULL,&start_pos,start_line
					  ,&mparen))
		   return(G__block_goto);
	      }
	      else {
		if(!G__asm_noverflow) {
		  G__no_exec=1;
		  G__fignorestream("}");
		  G__no_exec=0;
		}
		return(result);
	      }
	    }
	    iout=0;
	    spaceflag=0;
	  }
	  break;
	  
	case 6:
	  if(strcmp(statement,"while(")==0){
	    result=G__exec_while();
	    if(
#ifndef G__OLDIMPLEMENTATION1844
	       G__return>G__RETURN_NON 
#else
	       G__return!=G__RETURN_NON 
#endif
	       || mparen==0) return(result); 
	    if(result.type==G__block_goto.type &&
	       result.ref==G__block_goto.ref) {
	      if(0==G__search_gotolabel((char*)NULL,&start_pos,start_line
					,&mparen))
		 return(G__block_goto);
	    }
	    iout=0;
	    spaceflag=0;
	  }
	  if(strcmp(statement,"catch(")==0) {
            G__ignore_catch();
	    iout=0;
	    spaceflag=0;
	  }
#ifndef G__OLDIMPLEMENTATION1544
	  if(strcmp(statement,"throw(")==0) {
            c=G__fignorestream(")");
	    iout=0;
	    spaceflag=0;
	  }
#endif
	  break;
	  
	case 4:
	  if(strcmp(statement,"for(")==0){
	    result=G__exec_for();
	    if(
#ifndef G__OLDIMPLEMENTATION1844
	       G__return>G__RETURN_NON 
#else
	       G__return!=G__RETURN_NON 
#endif
	       || mparen==0) return(result); 
	    if(result.type==G__block_goto.type &&
	       result.ref==G__block_goto.ref) {
	      if(0==G__search_gotolabel((char*)NULL,&start_pos,start_line
					,&mparen))
		 return(G__block_goto);
	    }
	    iout=0;
	    spaceflag=0;
	  }
	  break;
	}
	/*********************************************
	 * G__no_exec==0 and not if,for,while,switch,
	 * this must be a function call.
	 * classtype a(1); G__define_var does
	 * a = b();        another case in the switch
	 *********************************************/
	if(iout>1 && isalpha(statement[0])&&(char*)NULL==strchr(statement,'[')
#ifndef G__OLDIMPLEMENTATION701
	   && strncmp(statement,"cout<<",6)!=0
#endif
#ifndef G__OLDIMPLEMENTATION1718
	   && (0==strstr(statement,"<<")||0==strcmp(statement,"operator<<("))
	   && (0==strstr(statement,">>")||0==strcmp(statement,"operator>>("))
#endif
	   && single_quote==0 && double_quote==0) {
	  /***********************************
	   * read to 'func(xxxxxx)'
	   *               ^^^^^^^
	   ***********************************/
	  c=G__fgetstream_new(statement+iout ,")");
	  iout=strlen(statement);
	  statement[iout]=c;
	  statement[++iout]='\0';
	  c=G__fgetspace();
	  /***********************************
	   * if 'func(xxxxxx) \n    nextexpr'   macro
	   * if 'func(xxxxxx) ;'                func call
	   * if 'func(xxxxxx) operator xxxxx'   func call + operator
	   * if 'new (arena) type'              new operator with arena
	   ***********************************/
#ifndef G__OLDIMPLEMENTATION721
	  if(strncmp(statement,"new ",4)==0||strncmp(statement,"new(",4)==0) {
	    char *pnew;
	    statement[iout++]=c;
	    c=G__fgetstream(statement+iout,";");
	    pnew=strchr(statement,'(');
	    G__ASSERT(pnew);
	    result=G__new_operator(pnew);
	  }
	  else if(G__exec_function(statement,&c,&iout,&largestep,&result)) 
	    return(G__null);
#else
	  if(G__exec_function(statement,&c,&iout,&largestep,&result)) 
	    return(G__null);
#endif
	  if((mparen==0&&c==';')||
#ifndef G__OLDIMPLEMENTATION1844
	       G__return>G__RETURN_NON 
#else
	       G__return!=G__RETURN_NON 
#endif
	     ) return(result); 
	  iout=0;
	  spaceflag=0;
	}
#ifndef G__OLDIMPLEMENTATION1177
	else if(iout>3) {
	  c=G__fgetstream_new(statement+iout ,")");
	  iout = strlen(statement);
	  statement[iout++] = c;
	}
#endif
	
      }
      else if((mparen==0)&&(iout==3)) {
	
	/**************************************************
	 * G__no_exec is set 
	 *   if(true)  ;
	 *   else if() ;  skip this
	 *          ^
	 *        else ;  skip this too
	 ***************************************************/
	if(strcmp(statement,"if(")==0) {
	  result=G__exec_else_if();
	  if(mparen==0||
#ifndef G__OLDIMPLEMENTATION1844
	     G__return>G__RETURN_NON 
#else
	     G__return!=G__RETURN_NON 
#endif
	     ) return(result); 
	  iout=0;
	  spaceflag=0;
	}
      }
      else {
	c=G__fignorestream(")");
	iout=0;
	spaceflag=0;
#ifndef G__OLDIMPLEMENTATION1215
	if(c!=')') G__genericerror("Error: Parenthesis does not match");
#endif
      }
      break;
      
    case '=' : /* assignement */
      if((G__no_exec==0 )&& (iout<8 || 9<iout ||
			     strncmp(statement,"operator",8)!=0) &&
	 (single_quote==0 && double_quote==0)) {
	statement[iout]='=';
#ifndef G__OLDIMPLEMENTATION944
	c=G__fgetstream_new(statement+iout+1,";,{}");
	if('}'==c || '{'==c) {
#else
	c=G__fgetstream_new(statement+iout+1,";,}");
	if('}'==c) {
#endif
#ifndef G__OLDIMPLEMENTATION1146
	  G__syntaxerror(statement);
#else
	  G__missingsemicolumn(statement);
#endif
	  --mparen;
	  c=';';
	}
	iout=0;
	spaceflag=0;
	if((G__breaksignal&&G__beforelargestep(statement,&iout,&largestep)>1)||
	   G__return>G__RETURN_NORMAL) {
	  return(G__null);
	}
#ifdef G__ASM
	if(G__asm_noverflow) G__asm_clear();
#endif
	result=G__getexpr(statement);
	if(G__p_tempbuf->level>=G__templevel && G__p_tempbuf->prev) {
	  G__free_tempobject();
	}
	if(largestep) G__afterlargestep(&largestep);
	if((mparen==0&&c==';')||G__return>G__RETURN_NORMAL) return(result);
      }
#ifndef G__OLDIMPLEMENTATION926
      else if(G__prerun && single_quote==0 && double_quote==0) {
        c = G__fignorestream(";,}"); 
#ifndef G__OLDIMPLEMENTATION1023
	if('}'==c && mparen) --mparen;
#endif
	iout=0;
	spaceflag=0;
      }
#endif
      else {
	statement[iout++]=c;
	spaceflag |= 1;
      }
      break;
      
    case ')' :
      if(G__ansiheader!=0) {
	G__ansiheader=0;
	return(G__null);
      }
      statement[iout++] = c ;
      spaceflag |= 1;
      break;
      
    case '{' : 
      if(G__funcheader==1) {
	/* return if called from G__interpret_func()
	 * as pass parameter declaratioin
	 */
	G__unsigned = 0;
	G__constvar = G__VARIABLE;
	return(G__start);
      }
      if((single_quote)||(double_quote)) {
	statement[iout++] = c ;
      }
      else {
#ifndef G__OLDIMPLEMENTATION615
	G__constvar = G__VARIABLE;
#endif
#ifndef G__OLDIMPLEMENTATION1841
	G__static_alloc = 0;
#endif
	statement[iout] = '\0' ;
	++mparen;
	if(0==G__no_exec) {
	  if(2==iout&&strcmp(statement,"do")==0) {
	    fseek(G__ifile.fp,-1,SEEK_CUR);
	    --mparen;
	    if(G__dispsource) G__disp_mask=1;
	    goto do_do;
	  }
	  if((4==iout&&strcmp(statement,"enum")==0)  ||
	     (5==iout&&strcmp(statement,"class")==0) ||
	     (6==iout&&strcmp(statement,"struct")==0)) {
	    fseek(G__ifile.fp,-1,SEEK_CUR);
	    --mparen;
	    if(G__dispsource) G__disp_mask=1;
	    G__DEFSTR(statement[0]);
	    spaceflag=0;
	    break;
	  }
#ifndef G__OLDIMPLEMENTATION612
	  if(8==iout&&strcmp(statement,"namespace")==0) {
            /* unnamed namespace treat as global scope.
	     * This implementation may be wrong. 
	     * Should fix later with using directive in global scope */
	  }
#endif
	}
	iout=0;
	spaceflag=0;
      }
      break;
      
    case '}' :
      if((single_quote!=0)||(double_quote!=0)) {
	statement[iout++] = c ;
      }
      else {
	if((--mparen)<=0) {
	  if(iout
#ifndef G__OLDIMPLEMENTATION1924
	     && G__NOLINK==G__globalcomp
#endif
	     ) {
	    statement[iout]='\0';
	    G__missingsemicolumn(statement);
	  }
	  if(mparen<0) G__genericerror("Error: Too many '}'");
	  return(result);
	}
	iout=0;
	spaceflag=0;
      }
      break;
      
      
    case '"' : /* double quote */
#ifndef G__OLDIMPLEMENTATION1051
      if(8==iout && strcmp(statement,"#include")==0) {
	fseek(G__ifile.fp,-1,SEEK_CUR);
	if(G__dispsource) G__disp_mask=1;
	G__include_file();
	iout=0;
	spaceflag=0;
	if(mparen==0 || G__return>G__RETURN_NORMAL) return(G__null);
      }
#endif
      statement[iout++] = c ;
      spaceflag=5;
      if(single_quote==0) {
	if(double_quote==0) {
	  double_quote=1;
	  if(G__prerun==1) conststring=statement+(iout-1);
	}
	else {
	  double_quote=0;
	  if(G__prerun==1) {
	    statement[iout]='\0';
	    G__strip_quotation(conststring);
	  }
	}
      }
      break;
      
    case '\'' : /* single quote */
      if(double_quote==0) single_quote ^= 1;
      statement[iout++] = c ;
      spaceflag=5;
      break;
      
    case '/' :
#ifndef G__OLDIMPLEMENTATION439
      if(iout>0 && double_quote==0 && statement[iout-1]=='/' && commentflag) {
#else
      if(iout>0 && double_quote==0 && statement[iout-1]=='/') {
#endif
	iout--;
	if(iout==0) spaceflag=0;
	G__fignoreline();
      }
      else {
#ifndef G__OLDIMPLEMENTATION439
	commentflag=1;
#endif
	statement[iout++] = c ;
	spaceflag |= 1;
      }
      break;
      
    case '*' :  /* comment */
#ifndef G__OLDIMPLEMENTATION439
      if(iout>0 && double_quote==0 && statement[iout-1]=='/' && commentflag) {
#else
      if(iout>0 && double_quote==0 && statement[iout-1]=='/') {
#endif
	/* start commenting out*/
	iout--;
	if(iout==0) spaceflag=0;
	if(G__skip_comment()==EOF) return(G__null);
      }
      else {
	statement[iout++] = c ;
	spaceflag |= 1;
#ifndef G__PHILIPPE12
	if(!double_quote && !single_quote) add_fake_space = 1;
#endif
      }
      break;

#ifndef G__PHILIPPE12
    case '&' :  /* this cut a symbols! */
      statement[iout++] = c ;
      spaceflag |= 1;
      if(!double_quote && !single_quote) add_fake_space = 1;
      break;
#endif
      
    case ':' :
      statement[iout++] = c ;
      spaceflag |= 1;
      if((double_quote==0)&&(single_quote==0)) {
	statement[iout] = '\0';
	if(G__label_access_scope(statement,&iout ,&spaceflag,&mparen))
	  return(G__null);
      }
      
      break;

#ifdef G__TEMPLATECLASS
    case '<': /* case 1,11) */
      statement[iout]='\0';
      if(8==iout && strcmp(statement,"template")==0) {
	G__declare_template();
	iout=0;
	spaceflag=0;
	if(mparen==0 || G__return>G__RETURN_NORMAL) return(G__null);
      }
#ifndef G__OLDIMPLEMENTATION1051
      else if(8==iout && strcmp(statement,"#include")==0) {
	fseek(G__ifile.fp,-1,SEEK_CUR);
	if(G__dispsource) G__disp_mask=1;
	G__include_file();
	iout=0;
	spaceflag=0;
	if(mparen==0 || G__return>G__RETURN_NORMAL) return(G__null);
      }
#endif
      else 
#ifndef G__OLDIMPLEMENTATION870
      {
	char *s=strchr(statement,'=');
	if(s) ++s;
	else  s=statement;
	if((1==spaceflag||2==spaceflag) && 
#ifndef G__PHILIPPE15
           (G__no_exec==0)&& 
#endif
	   (('~'==statement[0] && G__defined_templateclass(s+1)) ||
	    G__defined_templateclass(s))) {
#else
	if((1==spaceflag||2==spaceflag) && 
	   (('~'==statement[0] && G__defined_templateclass(statement+1)) ||
	    G__defined_templateclass(statement))) {
#endif
	  spaceflag=1;
	  /*   X  if(a<b) ;
	   *   X  func(a<b);
	   *   X  a=a<b;
	   *   x  cout<<a;   this is rare.
	   *   O  Vector<Array<T> >  This has to be handled here */
	  statement[iout++]=c;
	  c=G__fgetstream_template(statement+iout,">");
	  G__ASSERT('>'==c);
	  iout = strlen(statement);
	  if('>'==statement[iout-1]) statement[iout++]=' ';
	  statement[iout++]=c;
	  spaceflag=1;
#ifndef G__OLDIMPLEMENTATION781
	  /* Try to accept statements with no space between the closing
	   * > and the identifier. Ugly, though. */
	  {
	    fpos_t xpos;
	    fgetpos(G__ifile.fp,&xpos);
	    c = G__fgetspace();
	    fsetpos(G__ifile.fp,&xpos);
	    if(isalpha(c) || '_'==c) {
	      c = ' ';
	      goto read_again;
	    }
	  }
#endif
	}
	else {
	  statement[iout++] = c ;
	  spaceflag |= 1;
	}
#ifndef G__OLDIMPLEMENTATION870
      }
#endif
      break;
#endif /* G__TEMPLATECLASS */
      
    case EOF : /* end of file */
      statement[iout]='\0';
      G__handleEOF(statement,mparen ,single_quote,double_quote);
      return(G__null);
      
    case '\\':
      statement[iout++] = c ;
      spaceflag |= 1;
      c=G__fgetc();
      /* continue to default case */
      
    default:
      /* G__CHECK(G__SECURE_BUFFER_SIZE,iout==G__LONGLINE,return(G__null)); */
#ifndef G__PHILIPPE33
       /* Make sure that the delimiters that have not been treated
        * in the switch statement do drop the discarded_space */
      /* if (c!='[' && c!=']' && discarded_space && iout) { */
      if (c!='[' && c!=']' && discarded_space && iout 
          && statement[iout-1]!=':') {
	/* since the character following a discarded space is NOT a
	   separator, we have to keep the space */
	statement[iout++] = ' ';
      }
#endif
      statement[iout++] = c ;
      spaceflag |= 1;
#ifdef G__MULTIBYTE
      if(G__IsDBCSLeadByte(c)) {
	c = G__fgetc();
	G__CheckDBCS2ndByte(c);
	statement[iout++] = c ;
      }
#endif
      break;
    } /* end of switch */
#ifndef G__PHILIPPE33
    discarded_space = discard_space;
#endif
  } /* end of infinite loop */
  
}
/***********************************************************************
* end of G__exec_statement()
***********************************************************************/


/**************************************************************************
* G__readpointer2function()
*
*   'type (*func[n])(type var1,...);'
*   'type (*ary)[n];'
**************************************************************************/
int G__readpointer2function(new_name,pvar_type)
char *new_name;
char *pvar_type;
{
  int c;
  int n;
  char *p;
  fpos_t pos;
  int line;
  int ispointer=0;
  fpos_t pos2;
  int line2;
  /* static int warning=1; */
#ifndef G__OLDIMPLEMENTATIOn175
  int isp2memfunc=G__POINTER2FUNC;
  char tagname[G__ONELINE];

  tagname[0]='\0';
#endif

  if('*'==new_name[0]) ispointer=1;
  
  /* pointer of function
   *   type ( *funcpointer(type arg))( type var1,.....)
   *   type ( *funcpointer[n])( type var1,.....)
   *         ^                    */

  /* read variable name of function pointer */
  fgetpos(G__ifile.fp,&pos2);
  line2=G__ifile.line_number;
  c=G__fgetstream(new_name,"()");

  if('*'!=new_name[0]&&!strstr(new_name,"::*")) {
    fsetpos(G__ifile.fp,&pos2);
    G__ifile.line_number=line2;
    return(G__CONSTRUCTORFUNC);
  }

  if('('==c) {
    fgetpos(G__ifile.fp,&pos2);
    line2=G__ifile.line_number;
    c=G__fignorestream(")");
    c=G__fignorestream(")");
  }
  else {
    line2=0;
  }

  p=strstr(new_name,"::*");
  if(p) {
    isp2memfunc = G__POINTER2MEMFUNC;
    /* (A::*p)(...)  => new_name="p" , tagname="A::" */
    strcpy(tagname,new_name);
    p=strstr(tagname,"::*");
    strcpy(new_name,p+3);
    *(p+2)='\0';
  }
  
  /* pointer of function
   *   type ( *funcpointer[n])( type var1,.....)
   *                          ^             */

  c=G__fignorestream("([");

  /* pointer of function
   *   type ( *funcpointer[n])( type var1,.....)
   *                           ^             */

  if('['==c) {
    /***************************************************************
     * type (*pary)[n]; pointer to array
     *             ^
     ***************************************************************/
    char temp[G__ONELINE];
    n=0;
    while('['==c) {
      c=G__fgetstream(temp,"]");
      G__p2arylabel[n++]=G__int(G__getexpr(temp));
      c=G__fgetstream(temp,"[;,)=");
    }
    G__p2arylabel[n]=0;
    fseek(G__ifile.fp,-1,SEEK_CUR);
    if(G__dispsource) G__disp_mask=1;
  }
  else {
    /***************************************************************
    * type (*pfunc)(...); pointer to function
    *              ^
    ***************************************************************/
    /* Set newtype for pointer to function */
    char temp[G__ONELINE];
    line=G__ifile.line_number;
    fgetpos(G__ifile.fp,&pos);
    if(G__dispsource) G__disp_mask=1000;
    if(ispointer)
      sprintf(temp,"%s *(%s*)(",
	      G__type2string(G__var_type,G__tagnum,G__typenum,G__reftype
			     ,G__constvar)
	      ,tagname);
    else
      sprintf(temp,"%s (%s*)(",
	      G__type2string(G__var_type,G__tagnum,G__typenum,G__reftype
			     ,G__constvar)
	      ,tagname);
    c=G__fdumpstream(temp+strlen(temp),")");
    temp[strlen(temp)+1]='\0';
    temp[strlen(temp)]=c;
    G__tagnum = -1;
    if(G__POINTER2MEMFUNC==isp2memfunc) {
      G__typenum = G__search_typename(temp,'a',-1,0);
      sprintf(temp,"G__p2mf%d",G__typenum);
      G__typenum = G__search_typename(temp,'a',-1,0);
      G__var_type = 'a';
      *pvar_type = 'a';
    }
    else {
#ifndef G__OLDIMPLEMENTATION2191
      G__typenum = G__search_typename(temp,'1',-1,0);
      G__var_type = '1';
      *pvar_type = '1';
#else
      G__typenum = G__search_typename(temp,'Q',-1,0);
      G__var_type = 'Q';
      *pvar_type = 'Q';
#endif
    }
    G__ifile.line_number=line;
    fsetpos(G__ifile.fp,&pos);
    if(G__dispsource) G__disp_mask=0;
    if(G__asm_dbg) {
      if(G__dispmsg>=G__DISPNOTE) {
	G__fprinterr(G__serr,"Note: pointer to function exists");
	G__printlinenum();
      }
    }
    if(line2) {
      /* function returning pointer to function 
       *   type ( *funcpointer(type arg))( type var1,.....)
       *                       ^ <------- ^    */
       fsetpos(G__ifile.fp,&pos2);
       G__ifile.line_number = line2;
       return(G__FUNCRETURNP2F);
    }
    G__fignorestream(")");
  }
  return(isp2memfunc);
}



/**************************************************************************
* G__search_gotolabel()
*
*    Searches for goto label from given fpos and line_number. If label is
*    found, it returns label of {} nesting as mparen. fpos and line_numbers
*    are set inside this function. If label is not found 0 is returned.
**************************************************************************/
int G__search_gotolabel(label,pfpos,line,pmparen)
char *label; /* NULL if upper level call */
fpos_t *pfpos;
int line;
int *pmparen;
{
  int mparen=0;
  int c;
  char token[G__LONGLINE];
  /* int store_no_exec; */

  if(label) strcpy(G__gotolabel,label);


  if(G__breaksignal) {
    G__beforelargestep(G__gotolabel,&c,&mparen);
    if('\0'==G__gotolabel[0]) return(-1); /* ignore goto */
    if(mparen) {
      /* 'S' command  */
      G__step=1;
      G__setdebugcond();
    }
  }

  mparen=0;

  /* set file position, line number and source code display mask */
  fsetpos(G__ifile.fp,pfpos);
  G__ifile.line_number=line;
  G__no_exec=1;

  do {
    c=G__fgetstream(token,"{};:()");
    switch(c) {
    case '{':
      ++mparen;
      break;
    case '}':
      --mparen;
      break;
    case ':':
      if(strcmp(G__gotolabel,token)==0) {
	/* goto label found */
	if(G__dispsource) G__disp_mask=0;
#ifndef G__OLDIMPLEMENTATION311
	if(0==G__nobreak && 0==G__disp_mask && 0==G__no_exec_compile &&
	   G__srcfile[G__ifile.filenum].breakpoint&&
	   G__srcfile[G__ifile.filenum].maxline>G__ifile.line_number) {
	  G__srcfile[G__ifile.filenum].breakpoint[G__ifile.line_number]
	    |=G__TRACED;
	}
#else
	if(0==G__nobreak && 0==G__disp_mask && 0==G__no_exec_compile &&
	   G__breakpoint[G__ifile.filenum]&&
	   G__maxline[G__ifile.filenum]>G__ifile.line_number) {
	  G__breakpoint[G__ifile.filenum][G__ifile.line_number]|=G__TRACED;
	}
#endif
	G__gotolabel[0]='\0';
	G__no_exec=0;
	*pmparen = mparen;
	return(mparen);
      }
    default: /* ;,(,) just punctuation */
      break;
    }
  } while(mparen);

  /* goto label not found */
  return(0);
}

#ifndef G__OLDIMPLEMENTATION1596
void G__settemplevel(val)
int val; 
{
   G__templevel += val;
}

void G__clearstack() 
{
   int store_command_eval = G__command_eval;
   ++G__templevel;
   G__command_eval = 0;

   G__free_tempobject();

   G__command_eval = store_command_eval;
   --G__templevel;
}
#endif
   


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
