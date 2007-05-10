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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"

extern "C" {

/* 1929 */
#define G__IFDEF_NORMAL       1
#define G__IFDEF_EXTERNBLOCK  2
#define G__IFDEF_ENDBLOCK     4
static int G__externblock_iscpp = 0;

#define G__NONBLOCK   0
#define G__IFSWITCH   1
#define G__DOWHILE    8
int G__ifswitch = G__NONBLOCK;

extern int G__const_setnoerror();
extern int G__const_resetnoerror();

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


/***********************************************************************
* switch statement jump buffer
***********************************************************************/
static int G__prevcase=0;
extern void G__CMP2_equal G__P((G__value*,G__value*));

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
static void G__store_breakcontinue_list(int destination,int breakcontinue)
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
static void G__free_breakcontinue_list(G__breakcontinue_list *pbreakcontinue)
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
void G__set_breakcontinue_destination(int break_dest,int continue_dest
                                      ,G__breakcontinue_list *pbreakcontinue)
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

/***********************************************************************
* G__exec_try()
*
***********************************************************************/
int G__exec_try(char *statement)
{
  G__exec_statement(); 
  if(G__RETURN_TRY==G__return) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"    G__no_exec_compile reset\n");
#endif
    G__no_exec=0;
    G__return=G__RETURN_NON;
    return(G__exec_catch(statement));
  }
  return(0);
}

/***********************************************************************
* G__exec_catch()
*
***********************************************************************/
int G__exec_catch(char *statement)
{
  int c;
  while(1) {
    fpos_t fpos;
    int line_number;

    /* catch (ehclass& obj) {  } 
     * ^^^^^^^ */
    do {
      c=G__fgetstream(statement,"(};");
    } while('}'==c);
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
int G__exec_throw(char* statement)
{
  int iout;
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
  if(iout>4) {
    int largestep=0;
    if(G__breaksignal && G__beforelargestep(statement,&iout,&largestep)>=1)
      return(1);
    G__exceptionbuffer = G__getexpr(statement);
    if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: THROW\n",G__asm_cp);
#endif
      G__asm_inst[G__asm_cp]=G__THROW;
      G__inc_cp_asm(1,0);
    }
    if(largestep) G__afterlargestep(&largestep);
    G__exceptionbuffer.ref = G__exceptionbuffer.obj.i;
    if('U'==G__exceptionbuffer.type) G__exceptionbuffer.type='u';
  }
  else {
    G__exceptionbuffer = G__null;
  }
  if(0==G__no_exec_compile) {
    G__no_exec=1;
    G__return=G__RETURN_TRY;
  }
  return(0);
}

/***********************************************************************
* G__alloc_exceptionbuffer
*
***********************************************************************/
G__value G__alloc_exceptionbuffer(int tagnum) 
{
  G__value buf;
  /* create class object */
  buf.obj.i = (long)malloc((size_t)G__struct.size[tagnum]);
  buf.obj.reftype.reftype = G__PARANORMAL;
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
    if(G__CPPLINK!=G__struct.iscpplink[G__tagnum]) 
      free((void*)G__store_struct_offset);
    /* do nothing here, exception object shouldn't be stored in legacy temp buf */
    G__tagnum = store_tagnum;
    G__store_struct_offset = store_struct_offset;
    G__globalvarpointer = G__PVOID;
  }
  G__exceptionbuffer = G__null;
  return(0);
}

/***********************************************************************
* G__exec_delete()
*
***********************************************************************/
int G__exec_delete(char *statement,int *piout,int *pspaceflag,int isarray,int mparen)
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
int G__exec_function(char *statement,int *pc,int *piout,int *plargestep,G__value *presult)
{
  /* function call */
  if(*pc==';' || G__isoperator(*pc) || *pc==',' || *pc=='.'
     || *pc=='['
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
  else if(*pc=='(') {
    int len = strlen(statement);
    statement[len++] = *pc;
    *pc = G__fgetstream_newtemplate(statement+len,")");
    len = strlen(statement);
    statement[len++] = *pc;
    statement[len] = 0; 
    *pc=G__fgetspace();
    while(*pc!=';') {
      len = strlen(statement);
      statement[len++] = *pc;
      *pc = G__fgetstream_newtemplate(statement+len,");");
      if(*pc==';') break;
      len = strlen(statement);
      statement[len++] = *pc;
      statement[len] = 0; 
      *pc=G__fgetspace();
    }
    fseek(G__ifile.fp,-1,SEEK_CUR);
    if(G__dispsource) G__disp_mask=1;
#ifdef G__ASM
    if(G__asm_noverflow) G__asm_clear();
#endif
    *presult=G__getexpr(statement);
  }
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

/***********************************************************************
 * G__IsFundamentalDecl()
 ***********************************************************************/
int G__IsFundamentalDecl()
{
  char type_name[G__ONELINE];
  int c;
  fpos_t pos;
  int result=1;
  int tagnum;

  /* store file position */
  int linenum = G__ifile.line_number;
  fgetpos(G__ifile.fp,&pos);
  G__disp_mask = 1000;

  c=G__fgetname_template(type_name,"(");
  if(strcmp(type_name,"struct")==0 || strcmp(type_name,"class")==0 ||
     strcmp(type_name,"union")==0) {
    result=0;
  }
  else {
    tagnum = G__defined_tagname(type_name,1);
    if(-1!=tagnum) result = 0;
    else {
      int typenum = G__defined_typename(type_name);        
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
        if(strcmp(type_name,"unsigned")==0 ||
           strcmp(type_name,"char")==0 ||
           strcmp(type_name,"short")==0 ||
           strcmp(type_name,"int")==0 ||
           strcmp(type_name,"long")==0) result=1;
        else result=0;
      }
    }
  }

  /* restore file position */
  G__ifile.line_number = linenum;
  fsetpos(G__ifile.fp,&pos);
  G__disp_mask = 0;
  return result;
}

/***********************************************************************
* G__keyword_anytime_5()
*
***********************************************************************/
int G__keyword_anytime_5(char *statement)
{
  int c=0;
  int iout=0;

  if((G__prerun||(G__ASM_FUNC_COMPILE==G__asm_wholefunction&&0==G__ansiheader))
     && G__NOLINK == G__globalcomp
#ifdef G__OLDIMPLEMENTATION1083_YET
     && (G__func_now>=0 || G__def_struct_member )
#else
     && G__func_now>=0 
#endif
     && strcmp(statement,"const")==0
     && G__IsFundamentalDecl()
     ) {
    int rslt;
    G__constvar = G__CONSTVAR;
    G__const_setnoerror();
    rslt=G__keyword_anytime_6("static");
    G__const_resetnoerror();
    G__security_error = G__NOERROR ;
    G__return = G__RETURN_NON;
    return(rslt);
  }

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
  if(strcmp(statement,"#line")==0) {
    G__setline(statement,c,&iout);
    /* restore statement[0] as we found it because the
       callers might look at it! */
    statement[0]='#';
    return(1);
  }
  return(0);
}

/***********************************************************************
* G__keyword_anytime_6()
*
***********************************************************************/
int G__keyword_anytime_6(char *statement)
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
    struct G__var_array* store_local=G__p_local;
    if(G__p_local && G__prerun && -1!=G__func_now) G__p_local = (struct G__var_array*)NULL;
    G__static_alloc=1;
    store_no_exec=G__no_exec;
    G__no_exec=0;
    G__exec_statement();
    G__no_exec=store_no_exec;
    G__static_alloc=0;
    G__p_local = store_local;
    return(1);
  }

  if(1==G__no_exec&&strcmp(statement,"return")==0){
    G__fignorestream(";");
    return(1);
  }

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
    int stat = G__pp_ifdef(1);
    return(stat);
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

  if(strcmp(statement,"#ident")==0){
    G__fignoreline();
    return(1);
  }

  return(0);
}

/***********************************************************************
* G__keyword_anytime_7()
*
***********************************************************************/
int G__keyword_anytime_7(char *statement)
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
    int store_tagnum=G__tagnum;
    int store_typenum=G__typenum;
    struct G__var_array* store_local=G__p_local;
    G__p_local=(struct G__var_array*)NULL;
    G__var_type='p';
    G__definemacro=1;
    G__define();
    G__definemacro=0;
    G__p_local=store_local;
    G__tagnum=store_tagnum;
    G__typenum=store_typenum;
    return(1);
  }
  if(strcmp(statement,"#ifndef")==0){
    int stat = G__pp_ifdef(0);
    return(stat);
  }
  if(strcmp(statement,"#pragma")==0){
    G__pragma();
    return(1);
  }
  return(0);
}

/***********************************************************************
* G__keyword_anytime_8()
*
***********************************************************************/
int G__keyword_anytime_8(char* statement)
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
    fseek(G__ifile.fp,-1,SEEK_CUR);
    G__disp_mask=1;
    c=G__fgetname_template(tcname,";");
    if(strcmp(tcname,"class")==0 ||
       strcmp(tcname,"struct")==0) {
      c=G__fgetstream_template(tcname,";");
    }
    else if(isspace(c)) {
      int len = strlen(tcname);
      int store_c;
      while(len && ('&'==tcname[len-1] || '*'==tcname[len-1])) --len;
      store_c = tcname[len];
      tcname[len] = 0;
      if(G__istypename(tcname)) {
        G__ifile.line_number = line_number;
        fsetpos(G__ifile.fp,&pos);
        G__exec_statement(); 
        return(1);
      }
      else {
        tcname[len] = store_c;
        c=G__fgetstream_template(tcname+strlen(tcname),";");
      }
    }
    if(!G__defined_templateclass(tcname)) {
      G__instantiate_templateclass(tcname,0);
    }
    return(1);
  }
  if(strcmp(statement,"explicit")==0){
    G__isexplicit = 1;
    return(1);
  }
  return(0);
}

/***********************************************************************
* G__keyword_exec_6()
*
***********************************************************************/
int G__keyword_exec_6(char *statement,int *piout,int *pspaceflag,int mparen)
{
  if(strcmp(statement,"friend")==0) {
    if(G__parse_friend(piout,pspaceflag,mparen)) return(1);
    return(0);
  }
/* #ifdef G__ROOT */
  if(strcmp(statement,"extern")==0 || strcmp(statement,"EXTERN")==0) {
/* #else */
  /* if(strcmp(statement,"extern")==0) {} */
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

/***********************************************************************
* G__toLowerString
***********************************************************************/
void G__toUniquePath(char *s)
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

/***********************************************************************
* G__setline()
*
* Called by
*    G__exec_statement()
*
*  Set line number by '# <line> <file>' statement 
***********************************************************************/
int G__setline(char *statement,int c,int *piout)
{
  char *endofline="\n\r";
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
#ifdef G__WIN32
          G__toUniquePath(sysinclude);
          G__toUniquePath(sysstl);
          G__toUniquePath(statement);
#endif
          if(strncmp(sysinclude,statement+1,(size_t)len)==0||
             strncmp(sysstl,statement+1,(size_t)lenstl)==0) {
            G__globalcomp=G__NOLINK;
          }
          else if(G__ifile.fp==G__mfp) {
          }
          else {
            G__globalcomp=G__store_globalcomp;
            {
              struct G__ConstStringList* sysdir = G__SystemIncludeDir;
              while(sysdir) {
                if(strncmp(sysdir->string,statement+1,sysdir->hash)==0)
                  G__globalcomp=G__NOLINK;
                sysdir = sysdir->prev;
              }
            }
          }
          statement[strlen(statement)-1]='\0';
          strcpy(G__ifile.name,statement+1);
          G__hash(G__ifile.name,hash,temp);
          temp=0;
          null_entry = -1;
          for(i=0;i<G__nfile;i++) {
            if((char*)NULL==G__srcfile[i].filename && -1==null_entry) 
              null_entry = i;
            if(G__matchfilename(i,G__ifile.name)) {
              temp=1;
              break;
            }
          }
          if(temp) {
            G__ifile.filenum = i;
            G__security = G__srcfile[i].security;
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
            G__srcfile[null_entry].security = G__security;
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
            G__srcfile[null_entry].included_from=G__ifile.filenum;
            G__srcfile[null_entry].ispermanentsl = 0;
            G__srcfile[null_entry].initsl = 0;
            G__srcfile[null_entry].hasonlyfunc = (struct G__dictposition*)NULL;
            G__ifile.filenum = null_entry;
          }
          else {
            if(G__nfile==G__gettempfilenum()+1) {
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
              G__srcfile[G__nfile].security = G__security;
              /* If we are using a preprocessed file, the logical file
                 is actually located in the result file from the preprocessor
                 We need to need to carry this information on.
                 If the previous file (G__ifile) was preprocessed, this one
                 should also be. */
              if( G__cpp && 
                  G__srcfile[G__ifile.filenum].prepname &&
                  G__srcfile[G__ifile.filenum].prepname[0] ) {
                G__srcfile[G__nfile].prepname = 
               (char*)malloc(strlen(G__srcfile[G__ifile.filenum].prepname)+1);
                strcpy(G__srcfile[G__nfile].prepname,
                       G__srcfile[G__ifile.filenum].prepname);        
                G__srcfile[G__nfile].fp = G__ifile.fp;
              }
              /* Initilialize more of the data member see loadfile.c:1529 */
              G__srcfile[G__nfile].included_from=G__ifile.filenum;
              G__srcfile[G__nfile].ispermanentsl = 0;
              G__srcfile[G__nfile].initsl = 0;
              G__srcfile[G__nfile].hasonlyfunc = (struct G__dictposition*)NULL;
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
  int c;
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
    if(EOF==(c=G__fgetc())) {
      G__genericerror("Error: unexpected /* ...EOF");
      if(G__key!=0) system("key .cint_key -l execute");
      G__eof=2;
      return(EOF);
    }
    statement[1] = c;
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
  else if(strncmp(condition,"el",2)==0)     G__pp_skip(1);
  else if(strncmp(condition,"ifdef",5)==0)  G__pp_ifdef(1);
  else if(strncmp(condition,"ifndef",6)==0) G__pp_ifdef(0);
  else if(strncmp(condition,"if",2)==0)     G__pp_if();
  else if('\n'!=c && '\r'!=c)               G__fignoreline();
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
void G__pp_skip(int elifskip)
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
    
    if(argn>0 && arg[1][0]=='#') {
      const char* directive = arg[1]+1; // with "#if" directive will point to "if"
      int directiveArgI = 1;
      if(arg[1][1]==0
         || strcmp(arg[1],"#pragma")==0
         ) {
         directive = arg[2];
         directiveArgI = 2;
      }

      if(strncmp(directive,"if",2)==0) {
        ++nest;
      }
      else if(strncmp(directive,"else",4)==0) {
        if(nest==1 && elifskip==0) nest=0;
      }
      else if(strncmp(directive,"endif",5)==0) {
        --nest;
      }
      else if(strncmp(directive,"elif",4)==0) {
        if(nest==1 && elifskip==0) {
          int store_no_exec_compile=G__no_exec_compile;
          int store_asm_wholefunction=G__asm_wholefunction;
          int store_asm_noverflow=G__asm_noverflow;
          G__no_exec_compile=0;
          G__asm_wholefunction=0;
          G__abortbytecode();
          strcpy(condition,"");
          for(i=directiveArgI+1;i<=argn;i++) 
            strcat(condition, arg[i]);
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

          /* remove comments */
          char* posComment = strstr(condition, "/*");
          if (!posComment) posComment = strstr(condition, "//");
          while (posComment) {
             if (posComment[1]=='*') {
                char* posCXXComment = strstr(condition, "//");
                if (posCXXComment && posCXXComment < posComment)
                   posComment = posCXXComment;
             }
             if (posComment[1]=='*') {
                const char* posCommentEnd = strstr(posComment+2,"*/");
                // we can have
                // #if A /*
                //   comment */ || B
                // #endif
                if (!posCommentEnd) {
                  if (G__skip_comment()) 
                     break;
                  if (G__fgetstream (posComment, "\r\n") == EOF)
                     break;
                } else {
                   strcpy(temp, posCommentEnd+2);
                   strcpy(posComment, temp);
                }
                posComment = strstr(posComment, "/*");
                if (!posComment) posComment = strstr(condition, "//");
             } else {
                posComment[0]=0;
                posComment=0;
             }
          }

          G__noerr_defined=1;
          if(G__test(condition)) {
            nest=0;
          }
          G__no_exec_compile=store_no_exec_compile;
          G__asm_wholefunction=store_asm_wholefunction;
          G__asm_noverflow=store_asm_noverflow;
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
int G__pp_ifdefextern(char* temp)
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
      do {
         cin = G__fgetstream(temp,"{\r\n");
      } while (0==temp[0] && (cin == '\r' || cin == '\n'));
      if(0!=temp[0] || '{'!=cin)  goto goback;

      cin = G__fgetstream(temp,"\n\r");
      if (cin=='}' && 0==strcmp(fname,"C")) {
        goto goback;
      }
      cin = G__fgetstream(temp,"#\n\r");
      if ( (cin=='\n'||cin=='\r') && temp[0]==0) {
         cin = G__fgetstream(temp,"#\n\r");
      }
      if('#'!=cin) goto goback;
      cin = G__fgetstream(temp,"\n\r");
      if ( (cin=='\n'||cin=='\r') && temp[0]==0) {
         cin = G__fgetstream(temp,"#\n\r");
      }
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
  int store_no_exec_compile;
  int store_asm_wholefunction;
  int store_asm_noverflow;
  
  do {
    c = G__fgetstream(condition+len,"\n\r");
    len = strlen(condition)-1;
    if(len<0) len=0;
    if(len>0 && (condition[len]=='\n' ||condition[len]=='\r')) --len;
  } while('\\'==condition[len]);

  {
    char *p;
    while((p=strstr(condition,"\\\n"))!=0) {
      memmove(p,p+2,strlen(p+2)+1);
    }
  }
  
  /* This supresses error message when undefined
   * macro is refered in the #if defined(macro) */
  G__noerr_defined=1;
  
  /*************************
   * faulse
   * skip line until #else,
   * #endif or #elif.
   * Then, return to evaluation
   *************************/
  store_no_exec_compile=G__no_exec_compile;
  store_asm_wholefunction=G__asm_wholefunction;
  store_asm_noverflow=G__asm_noverflow;
  G__no_exec_compile=0;
  G__asm_wholefunction=0;
  G__abortbytecode();
  if(!G__test(condition)) {
    /********************
     * SKIP
     ********************/
    G__pp_skip(0);
  }
  else {
    int stat;
    G__no_exec_compile=store_no_exec_compile;
    G__asm_wholefunction=store_asm_wholefunction;
    G__asm_noverflow=store_asm_noverflow;
    G__noerr_defined=0;
    stat = G__pp_ifdefextern(condition);
    return(stat); /* must be either G__IFDEF_ENDBLOCK or G__IFDEF_NORMAL */
  }
  G__no_exec_compile=store_no_exec_compile;
  G__asm_wholefunction=store_asm_wholefunction;
  G__asm_noverflow=store_asm_noverflow;
  G__noerr_defined=0;

  return(G__IFDEF_NORMAL);
}

/***********************************************************************
* G__defined_macro()
*
* Search for macro symbol
*
***********************************************************************/
int G__defined_macro(char *macro)
{
  struct G__var_array *var;
  int hash,iout;
  G__hash(macro,hash,iout);
  var = &G__global;
  do {
    for(iout=0;iout<var->allvar;iout++) {
      if((tolower(var->type[iout])=='p' 
          || 'T'==var->type[iout]
          ) &&
         hash == var->hash[iout] && strcmp(macro,var->varnamebuf[iout])==0)
        return(1); /* found */
    }
  } while((var=var->next)) ;
  if(682==hash && strcmp(macro,"__CINT__")==0) return(1);
  if(!G__cpp && 1704==hash && strcmp(macro,"__CINT_INTERNAL_CPP__")==0) return(1);
  if(
     (G__iscpp || G__externblock_iscpp)
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
int G__pp_ifdef(int def)  /* 1 for ifdef 0 for ifndef */
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
  else {
    int stat = G__pp_ifdefextern(temp);
    return(stat); /* must be either G__IFDEF_ENDBLOCK or G__IFDEF_NORMAL */
  }

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

  int store_ifswitch = G__ifswitch;
  G__ifswitch = G__DOWHILE;

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
    G__asm_clear();
  }
#endif

  /* breakcontinue buffer setup */
  if(G__asm_noverflow) {
    store_pbreakcontinue = G__alloc_breakcontinue_list();
    allocflag=1;
  }

  store_no_exec_compile=G__no_exec_compile;
  result=G__exec_statement();
  G__no_exec_compile=store_no_exec_compile;
  if(G__return!=G__RETURN_NON) {
    /* free breakcontinue buffer */
    if(allocflag) G__free_breakcontinue_list(store_pbreakcontinue);
    G__ifswitch = store_ifswitch;
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
        G__ifswitch = store_ifswitch;
        return(result);
      }
      else if(!G__asm_noverflow) {
        /* free breakcontinue buffer */
        if(allocflag) G__free_breakcontinue_list(store_pbreakcontinue);
        G__ifswitch = store_ifswitch;
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
      G__ifswitch = store_ifswitch;
      return(G__null);
    }
  }

  store_asm_cp=G__asm_cp;

  store_no_exec_compile=G__no_exec_compile;
  if(executed_break) G__no_exec_compile=1;
  cond = G__test(condition);
  if(executed_break) G__no_exec_compile=store_no_exec_compile;
  if(G__p_tempbuf->level>=G__templevel && G__p_tempbuf->prev) {
    G__free_tempobject();
  }

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
        G__ifswitch = store_ifswitch;
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
        G__ifswitch = store_ifswitch;
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
          G__ifswitch = store_ifswitch;
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
          G__ifswitch = store_ifswitch;
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
  
  G__ifswitch = store_ifswitch;
  result = G__null;
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
G__value G__return_value(char* statement)
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
    --G__templevel;
    buf=G__getexpr(statement);
    ++G__templevel;
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
void G__display_tempobject(char* action)
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
   /* The only 2 potential risks of making this static are
    * - a destructor indirectly calls G__free_tempobject
    * - multi-thread application (but CINT is not multi-threadable anyway). */
  static char statement[G__ONELINE];
  struct G__tempobject_list *store_p_tempbuf;

  if(G__xrefflag
     || (G__command_eval && G__DOWHILE!=G__ifswitch)
     ) return;

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
    
    if(0==G__p_tempbuf->no_exec
       || 1==G__no_exec_compile
       ) {
      if(G__dispsource) {
        G__fprinterr(G__serr,
                     "!!!Destroy temp object (%s)0x%lx createlevel=%d destroylevel=%d\n"
                     ,G__struct.name[G__tagnum]
                     ,G__p_tempbuf->obj.obj.i
                     ,G__p_tempbuf->level,G__templevel);
      }
      
      sprintf(statement,"~%s()",G__struct.name[G__tagnum]);
      G__getfunction(statement,&iout,G__TRYDESTRUCTOR); 
    }
    
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
G__value G__alloc_tempstring(char *string)
{
  /* int len; */
  struct G__tempobject_list *store_p_tempbuf;

  if(G__xrefflag) return(G__null);

  /* create temp object buffer */
  store_p_tempbuf = G__p_tempbuf;
  G__p_tempbuf = (struct G__tempobject_list *)malloc(
                                     sizeof(struct G__tempobject_list)
                                                     );
  G__p_tempbuf->prev = store_p_tempbuf;
  G__p_tempbuf->level = G__templevel;
  G__p_tempbuf->cpplink = 0;
  G__p_tempbuf->no_exec = 0;
  
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
void G__alloc_tempobject(int tagnum,int typenum)
{
  struct G__tempobject_list *store_p_tempbuf;

  G__ASSERT( 0<=tagnum );

  if(G__xrefflag) return;

  /* create temp object buffer */
  store_p_tempbuf = G__p_tempbuf;
  G__p_tempbuf = (struct G__tempobject_list *)malloc(
                                     sizeof(struct G__tempobject_list)
                                                     );
  G__p_tempbuf->prev = store_p_tempbuf;
  G__p_tempbuf->level = G__templevel;
  G__p_tempbuf->cpplink = 0;
  G__p_tempbuf->no_exec = G__no_exec_compile;
  
  /* create class object */
  G__p_tempbuf->obj.obj.i = (long)malloc((size_t)G__struct.size[tagnum]);
  G__p_tempbuf->obj.obj.reftype.reftype = G__PARANORMAL;
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
void G__store_tempobject(G__value reg)
{
  struct G__tempobject_list *store_p_tempbuf;

  /* G__ASSERT( 'u'==reg.type || '\0'==reg.type ); */

  if(G__xrefflag) return;

#ifdef G__NEVER
  if('u'!=reg.type) {
    G__fprinterr(G__serr,"%d %d %d %ld\n"
            ,reg.type,reg.tagnum,reg.typenum,reg.obj.i);
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
  G__p_tempbuf->no_exec = G__no_exec_compile;

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
* G__pop_tempobject_imp()
*
* Called by
*    G__pop_tempobj[_nodel]()
*
***********************************************************************/
static int G__pop_tempobject_imp(bool delobj)
{
  struct G__tempobject_list *store_p_tempbuf;

  if(G__xrefflag) return(0);

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
  if(delobj && -1!=G__p_tempbuf->cpplink && G__p_tempbuf->obj.obj.i) {
    free((void *)G__p_tempbuf->obj.obj.i);
  }
  free((void *)G__p_tempbuf);
  G__p_tempbuf = store_p_tempbuf;
  return(0);
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
   return G__pop_tempobject_imp(true);
   
}
/***********************************************************************
* G__pop_tempobject_nodel()
*
* Called by
*    G__getfunction
*
***********************************************************************/
int G__pop_tempobject_nodel()
{
   return G__pop_tempobject_imp(false);
   
}
/***********************************************************************
* G__exec_breakcontinue()
*
***********************************************************************/
static int G__exec_breakcontinue(char *statement,int *piout,int *pspaceflag,int *pmparen
                                 ,int breakcontinue) /* 0: continue, 1:break */
{
  int store_no_exec_compile = G__no_exec_compile;
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
      if(0==breakcontinue) { /* in case of continue */
        statement[0]='\0';
        *piout=0;
        *pspaceflag=0;
        return(0);
      }
    }
  }
#endif /* G__ASM */
  /*  Assuming that break,continue appears only within if,switch clause
   *  skip to the end of if,switch conditional clause. If they appear
   *  in plain for,while loop, which make no sense, a strange behavior 
   *  may be observed. */
  if(G__DOWHILE!=G__ifswitch) {
    while(*pmparen) {
      char c=G__fignorestream("}");
      if('}'!=c) {
        G__genericerror("Error: Syntax error, possibly too many parenthesis");
      }
      --(*pmparen);
    }
  }
#ifndef G__OLDIMPLEMENTATION1695
  *piout=0;
  if(store_no_exec_compile) return(0);
  else return(1);
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

  int store_ifswitch = G__ifswitch;
  G__ifswitch = G__IFSWITCH;

  if(G__ASM_FUNC_COMPILE!=G__asm_wholefunction) {
#ifdef G__ASM
    if(G__asm_dbg&&G__asm_noverflow) {
      G__fprinterr(G__serr,"bytecode compile aborted by switch statement");
      G__printlinenum();
    }
    G__abortbytecode();
#endif
  }

  /* get switch(condition)
   *            ^^^^^^^^^
   */
  G__fgetstream(condition,")");

  if(G__breaksignal && G__beforelargestep(condition,&iout,&largestep)>1) {
    G__ifswitch = store_ifswitch;
    return(G__null);
  }

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
    G__ifswitch = store_ifswitch;
    return(result);
  } else
  if(G__no_exec_compile) {
    G__switch=0;
    G__no_exec=1;
    result=G__exec_statement();
    G__no_exec=0;
    result=G__default;
    G__ifswitch = store_ifswitch;
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
      G__ifswitch = store_ifswitch;
      return(result);
    }
    
    /* break found */
    if(result.type==G__block_break.type&&result.obj.i==G__BLOCK_BREAK) {
      G__ifswitch = store_ifswitch;
      if(result.ref==G__block_goto.ref) return(result);
      else                              return(G__null);
    }
  }
  G__mparen=0;
  G__no_exec=0;
  G__ifswitch = store_ifswitch;
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
  int localFALSE=0;
  fpos_t store_fpos;
  int    store_line_number;
  int c;
  char statement[10]; /* used to identify comment and else */
  int iout;
  int store_no_exec_compile=0;
  int asm_jumppointer=0;
  int largestep=0;

  int store_ifswitch = G__ifswitch;
  G__ifswitch = G__IFSWITCH;
  
  G__fgetstream_new(condition,")");

#ifndef G__OLDIMPLEMENTATION1802
  condition = (char*)realloc((void*)condition,strlen(condition)+10);
#endif
  
  if(G__breaksignal &&
     G__beforelargestep(condition,&iout,&largestep)>1) {
    G__ifswitch = store_ifswitch;
#ifndef G__OLDIMPLEMENTATION1802
    free((void*)condition);
#endif
    return(G__null);
  }
  
  result=G__null;

  
  if(G__test(condition)) {
    if(largestep) G__afterlargestep(&largestep);
    if(G__p_tempbuf->level>=G__templevel && G__p_tempbuf->prev) {
      G__free_tempobject();
    }
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
      G__ifswitch = store_ifswitch;
#ifndef G__OLDIMPLEMENTATION1802
      free((void*)condition);
#endif
      return(result);
    }
    localFALSE=0;
  }

  else {
    if(largestep) G__afterlargestep(&largestep);
    if(G__p_tempbuf->level>=G__templevel && G__p_tempbuf->prev) {
      G__free_tempobject();
    }
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
    localFALSE=1;
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
    while(c=='/' || c=='#') {
      if(c=='/') {
        c=G__fgetc();
        /*****************************
         * new
         *****************************/
        switch(c) {
        case '*':
          if(G__skip_comment()==EOF) {
            G__ifswitch = store_ifswitch;
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
    }
    if(c==EOF) {
      G__genericerror("Error: unexpected if() { } EOF");
      if(G__key!=0) system("key .cint_key -l execute");
      G__eof=2;
      G__ifswitch = store_ifswitch;
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
    if(localFALSE==1
       || G__asm_wholefunction
       ) {
      G__no_exec=0;
      /* localFALSE=0; */
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
      G__ifswitch = store_ifswitch;
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
    G__asm_cond_cp=G__asm_cp; /* avoid wrong optimization */
  }

  G__ifswitch = store_ifswitch;
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
G__value G__exec_loop(char *forinit,char *condition,int naction,char **foraction)
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
  int dispstat=0;
#define G__OLDIMPLEMENTATION1256

  int store_ifswitch = G__ifswitch;
  G__ifswitch = G__DOWHILE;

  if(G__breaksignal && G__beforelargestep(condition,&iout,&largestep)>1) {
    G__ifswitch = store_ifswitch;
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
  if(G__p_tempbuf->level>=G__templevel && G__p_tempbuf->prev) {
    G__free_tempobject();
  }

  if(G__no_exec_compile) cond=1;

  if(!cond && allocflag) {
    /* free breakcontinue buffer */
    G__free_breakcontinue_list(store_pbreakcontinue);
    allocflag=0;
  }

  
  while(cond) {
    int store_no_exec_compile = G__no_exec_compile;

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
      G__asm_clear();
    }
#endif
    result=G__exec_statement();
#ifdef G__OLDIMPLEMENTATION1673_YET
    if(G__p_tempbuf->level>=G__templevel && G__p_tempbuf->prev) {
      G__free_tempobject();
    }
#endif
    G__no_exec_compile = store_no_exec_compile;
    if(G__return!=G__RETURN_NON) {
      /* free breakcontinue buffer */
      if(allocflag) G__free_breakcontinue_list(store_pbreakcontinue);
      G__ifswitch = store_ifswitch;
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
          G__ifswitch = store_ifswitch;
          return(result);
        }
        else if(!G__asm_noverflow) {
          /* free breakcontinue buffer */
          if(allocflag) G__free_breakcontinue_list(store_pbreakcontinue);
          G__ifswitch = store_ifswitch;
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

      if(G__asm_dbg) {
        G__fprinterr(G__serr,"Bytecode loop compilation successful");
        G__printlinenum();
      }

      if(0==G__no_exec_compile) {
        asm_exec=G__exec_asm(asm_start_pc,/*stack*/0,&result,/*localmem*/0);
        if(G__return!=G__RETURN_NON) { 
          G__ifswitch = store_ifswitch;
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
      if(G__asm_dbg && 0==dispstat) {
        G__fprinterr(G__serr,"Bytecode loop compilation failed");
        G__printlinenum();
        dispstat = 1;
      }
    }
#endif

    if(G__no_exec_compile) cond=0;
    else { 
      cond=G__test(condition);
      if(G__p_tempbuf->level>=G__templevel && G__p_tempbuf->prev) {
        G__free_tempobject();
      }
    }
  } /* while(G__test) */

#ifdef G__ASM
  G__asm_noverflow = asm_exec && store_asm_noverflow;
#endif

  /***********************************************
   * skip last for execution
   ***********************************************/
  G__mparen=0;
  G__no_exec=1;
  result=G__exec_statement();
  if(G__return!=G__RETURN_NON) {
    G__ifswitch = store_ifswitch;
    return(result); 
  }
  G__no_exec=0;
  
  if(largestep) G__afterlargestep(&largestep);
  G__ifswitch = store_ifswitch;
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
  int store_ifswitch = G__ifswitch;
  G__ifswitch = G__DOWHILE;

  G__fgetstream(condition,")");

#ifndef G__OLDIMPLEMENTATION1802
  condition = (char*)realloc((void*)condition,strlen(condition)+10);
#endif

  result=G__exec_loop((char*)NULL,condition,0,(char**)NULL);
  G__ifswitch = store_ifswitch;

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
  int store_ifswitch = G__ifswitch;
  G__ifswitch = G__DOWHILE;

  /* handling 'for(int i=0;i<xxx;i++)' */
  G__exec_statement();
  if(G__return>G__RETURN_NORMAL) {
    G__ifswitch = store_ifswitch;
    return(G__null);
  }

#ifndef G__OLDIMPLEMENTATION1802
  condition=(char*)malloc(G__LONGLINE);
#endif

  c=G__fgetstream(condition,";)");
  if(')'==c) {
    G__genericerror("Error: for statement syntax error");
    G__ifswitch = store_ifswitch;
#ifndef G__OLDIMPLEMENTATION1802
    free((void*)condition);
#endif
    return(G__null);
  }
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
      G__ifswitch = store_ifswitch;
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
  G__ifswitch = store_ifswitch;
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
  int store_ifswitch = G__ifswitch;
  G__ifswitch = G__IFSWITCH;

#ifdef G__ASM
  if(0==G__no_exec_compile) 
    G__abortbytecode(); /* this must be redundant, but just in case */
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
          G__ifswitch = store_ifswitch;
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
      G__ifswitch = store_ifswitch;
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
  
  G__ifswitch = store_ifswitch;
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
  int commentflag=0;
  int add_fake_space = 0;
  int fake_space = 0;
  int discard_space = 0;
  int discarded_space = 0;

  fgetpos(G__ifile.fp,&start_pos);
  start_line=G__ifile.line_number;

  mparen = G__mparen;
  G__mparen=0;

  result=G__null;

  while(1) {
     if (iout>=G__LONGLINE) {
        G__genericerror("Error: Line too long in G__exec_statement when processing ");
        return(result);
     }
    fake_space = 0;
    if (add_fake_space && !double_quote && !single_quote) {
      c = ' ';
      add_fake_space = 0;
      fake_space = 1;
    } else { 
      c=G__fgetc();
    }
    discard_space = 0;

/*#define G__OLDIMPLEMENTATION781*/
  read_again:
    
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
      commentflag=0;
      /* ignore these character */
      if((single_quote!=0)||(double_quote!=0)) {
        statement[iout++] = c ;
      }
      else {
      after_replacement:
        
        if (!fake_space) discard_space = 1;
        if(spaceflag==1) {
          statement[iout] = '\0' ;
          /* search keyword */
        G__preproc_again:
          if(statement[0]=='#'&&isdigit(statement[1])) {
            if(G__setline(statement,c,&iout)) goto G__preproc_again;
            spaceflag = 0;
            iout=0;
          }
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
              int stat = G__pp_if();
              if(stat==G__IFDEF_ENDBLOCK) return(G__null);
              spaceflag = 0;
              iout=0;
            }
            break;
          case 4:
            if((mparen==1)&& (strcmp(statement,"case")==0)) {
              char casepara[G__ONELINE];
              G__fgetstream(casepara,":");
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
              if(G__switch!=0) {
                int store_no_execXX;
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
              }
              iout=0;
              spaceflag=0;
            }
            break;

          case 5:
            /* #else, #elif */
            if(G__keyword_anytime_5(statement)) {
              if(0==mparen&&'#'!=statement[0]) return(G__null);
              spaceflag = 0;
              iout=0;
            }
            break;

          case 6:
            /* static, #ifdef,#endif,#undef */
            {
              int stat=G__keyword_anytime_6(statement);
              if(stat) {
                if(0==mparen&&'#'!=statement[0]) return(G__null);
                if(stat==G__IFDEF_ENDBLOCK) return(G__null);
                spaceflag = 0;
                iout=0;
              }
            }
            break;

          case 7:
            if((mparen==1)&&(strcmp(statement,"default")==0)){
              G__fignorestream(":");
              if(G__switch!=0) {
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
              }
              iout=0;
              spaceflag=0;
              break;
            }
            
            /* #ifndef , #pragma */
            {
              int stat=G__keyword_anytime_7(statement);
              if(stat) {
                if(0==mparen&&'#'!=statement[0]) return(G__null);
                if(stat==G__IFDEF_ENDBLOCK) return(G__null);
                spaceflag = 0;
                iout=0;
              }
            }
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
              }
              iout=0;
              spaceflag=0;
              break;
            }
            if(G__keyword_anytime_8(statement)) {
              if(0==mparen&&'#'!=statement[0]) return(G__null);
              spaceflag = 0;
              iout=0;
            }
            break;
          }
        }
        
        if(1==spaceflag&&0==G__no_exec) {
          statement[iout] = '\0' ;
          /* search keyword */

          if(
             iout>3
             &&'*'==statement[iout-2]&&
             ('*'==statement[iout-1]||'&'==statement[iout-1])) {
            /* char**, char*&, int**, int*& */
            statement[iout-2]='\0';
            iout-=2;
            if (fake_space) fseek(G__ifile.fp,-2,SEEK_CUR);
            else 
              fseek(G__ifile.fp,-3,SEEK_CUR);
            if(G__dispsource) G__disp_mask=2;
          }
          
          switch(iout){
          case 2:
            if(strcmp(statement,"do")==0) {
            do_do:
              result=G__exec_do();
              if(mparen==0||
                 G__return>G__RETURN_NON
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
              G__var_type='i' + G__unsigned;
              G__define_var(-1,-1);          
              spaceflag = -1;                
              iout=0;                      
              if(mparen==0) return(G__null);
              break;
            }
            if(strcmp(statement,"new")==0) {
              c=G__fgetspace();
              if('('==c) {
                fseek(G__ifile.fp,-1,SEEK_CUR);
                if(G__dispsource) G__disp_mask=1;
                statement[iout++] = ' ' ;
                spaceflag |= 1;
      /* a little later this string will be passed to subfunctions
         that expect the string to be terminated */
      statement[iout] = '\0'; 
              }
              else {
                statement[0]=c;
                c=G__fgetstream_template(statement+1,";");
                result=G__new_operator(statement);
                spaceflag = -1;
                iout=0;
                if(0==mparen) return(result);
              }
              break;
            }
            if(strcmp(statement,"try")==0) {
              G__exec_try(statement);
              iout=0;
              spaceflag= -1;
              break;
            }
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
            if(strcmp(statement,"bool")==0) {
              G__DEFVAR('g');
              break;
            }
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
            if(strcmp(statement,"(new")==0) {
              c=G__fgetspace();
              if('('==c) {
                fseek(G__ifile.fp,-1,SEEK_CUR);
                statement[iout++] = ' ' ;
      /* a little later this string will be passed to subfunctions
         that expect the string to be terminated */
      statement[iout] = '\0'; 
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
      /* a little later this string will be passed to subfunctions
         that expect the string to be terminated */
      statement[iout] = '\0'; 
              }
              break;
            }
            if(strcmp(statement,"goto")==0) {
              G__CHECK(G__SECURE_GOTO,1,return(G__null));
              c=G__fgetstream(statement,";"); /* get label */
              if(G__ASM_FUNC_COMPILE==G__asm_wholefunction) {
                G__add_jump_bytecode(statement);
              }
              else {
                if(G__asm_dbg&&G__asm_noverflow) {
                  G__fprinterr(G__serr,"bytecode compile aborted by goto statement");
                  G__printlinenum();
                }
                G__abortbytecode();
              }
              if(G__no_exec_compile) {
                if(0==mparen) return(G__null);
                if(G__ASM_FUNC_COMPILE!=G__asm_wholefunction) 
                  G__abortbytecode();
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
            if(strcmp(statement,"bool&")==0) {
              G__DEFREFVAR('g');
              break;
            }
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
            if(strcmp(statement,"bool*")==0) {
              G__typepdecl=1;
              G__DEFVAR('G');
              G__typepdecl=0;
              break;
            }
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
            if(strcmp(statement,"using")==0) {
              G__using_namespace();
              spaceflag = -1;
              iout=0;
              break;
            }
            if(strcmp(statement,"throw")==0) {
              G__exec_throw(statement);
              spaceflag = -1;
              iout=0;
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
            if (strcmp (statement, "mutable") == 0) {
              spaceflag = -1;
              iout = 0;
              break;
            }
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
              int store_tagnum;
              do {
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
              } while('('!=c);
              iout = strlen(statement);
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
              iout=0;
              spaceflag = -1;
#ifdef G__SECURITY
              if(mparen==0) return(G__null);
#else
              if(mparen==0||G__return>G__RETURN_NORMAL) return(G__null);
#endif
              break;
            }
            if (strcmp (statement, "typename") == 0) {
              spaceflag = -1;
              iout = 0;
              break;
            }
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
            if(strcmp(statement,"R__EXTERN")==0) { 
              if(G__externignore(&iout,&spaceflag,mparen)) return(G__null);
              break;
            }

          case 13:
            if(strcmp(statement,"__extension__")==0) { 
              spaceflag = -1;
              iout = 0;
              break;
            }
          }
          
          if(iout && G__defined_type(statement ,iout)) {
            spaceflag = -1;
            iout=0;
#ifdef G__SECURITY
            if(mparen==0) return(G__null);
#else
            if(mparen==0||G__return>G__RETURN_NORMAL) return(G__null);
#endif
          }
          else 
            if(iout) 
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
            if(!strchr(statement,'.') && !strstr(statement,"->")) {
              namespace_tagnum = G__defined_tagname(statement,2);
            }
            else {
              namespace_tagnum = -1;
            }
            if (((namespace_tagnum!=-1) && (G__struct.type[namespace_tagnum]=='n'))
                ||(strcmp(statement,"std")==0)) {
              spaceflag = 0;            
            }
          }
          {
            char* replace = (char*)G__replacesymbol(statement);
            if(replace!=statement) {
              strcpy(statement,replace);
              iout = strlen(statement);
              goto after_replacement;
            }
          }
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

          if(strncmp(statement,"return\"",7)==0 ||
             strncmp(statement,"return'",7)==0) {
            result=G__return_value(statement+6);
            if(G__no_exec_compile) {
              statement[0]='\0';
              spaceflag = -1;
              iout=0;
              if(mparen==0) return(G__null);
              break;
            }
            return(result);
          }
          
          /* Normal commands 'xxxx;' */
          if(statement[0] && iout) {
#ifdef G__ASM
            if(G__asm_noverflow) G__asm_clear();
#endif
            result=G__getexpr(statement);
            
            if(G__p_tempbuf->level>=G__templevel && G__p_tempbuf->prev) {
              G__free_tempobject();
            }
          }
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
      
      /* if((spaceflag==1)&&(G__no_exec==0)) {} */
      if(G__no_exec==0) { 

        if( (0==G__def_struct_member||strncmp(statement,"ClassDef",8)!=0) &&
            G__execfuncmacro_noexec(statement)) {
          spaceflag=0;
          iout =0;
          break;
        }
        
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
               G__return>G__RETURN_NON 
               || mparen==0) return(result);
            if(result.type==G__block_goto.type &&
               result.ref==G__block_goto.ref) {
              if(0==G__search_gotolabel((char*)NULL,&start_pos,start_line
                                        ,&mparen))
                 return(G__block_goto);
            }
            if(result.type==G__block_break.type) {
              if(result.obj.i==G__BLOCK_CONTINUE) {
                return(result);
              }
            }
            iout=0;
            spaceflag=0;
          }

          break;
          
        case 3:
          if(strcmp(statement,"if(")==0) {
            result=G__exec_if();
            if(
               G__return>G__RETURN_NON 
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
               G__return>G__RETURN_NON 
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
          if(strcmp(statement,"throw(")==0) {
            c=G__fignorestream(")");
            iout=0;
            spaceflag=0;
          }
          break;
          
        case 4:
          if(strcmp(statement,"for(")==0){
            result=G__exec_for();
            if(
               G__return>G__RETURN_NON 
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
           && strncmp(statement,"cout<<",6)!=0
           && (0==strstr(statement,"<<")||0==strcmp(statement,"operator<<("))
           && (0==strstr(statement,">>")||0==strcmp(statement,"operator>>("))
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
          if((mparen==0&&c==';')||
               G__return>G__RETURN_NON 
             ) return(result); 
          iout=0;
          spaceflag=0;
        }
        else if(iout>3) {
          c=G__fgetstream_new(statement+iout ,")");
          iout = strlen(statement);
          statement[iout++] = c;
        }
        
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
             G__return>G__RETURN_NON 
             ) return(result); 
          iout=0;
          spaceflag=0;
        }
      }
      else {
        c=G__fignorestream(")");
        iout=0;
        spaceflag=0;
        if(c!=')') G__genericerror("Error: Parenthesis does not match");
      }
      break;
      
    case '=' : /* assignment */
      if((G__no_exec==0 )&& (iout<8 || 9<iout ||
                             strncmp(statement,"operator",8)!=0) &&
         (single_quote==0 && double_quote==0)) {
        statement[iout]='=';
        c=G__fgetstream_new(statement+iout+1,";,{}");
        if('}'==c || '{'==c) {
          G__syntaxerror(statement);
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
      else if(G__prerun && single_quote==0 && double_quote==0) {
        c = G__fignorestream(";,}"); 
        if('}'==c && mparen) --mparen;
        iout=0;
        spaceflag=0;
      }
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
        G__constvar = G__VARIABLE;
        G__static_alloc = 0;
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
          if(8==iout&&strcmp(statement,"namespace")==0) {
            /* unnamed namespace treat as global scope.
             * This implementation may be wrong. 
             * Should fix later with using directive in global scope */
          }
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
             && G__NOLINK==G__globalcomp
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
      if(8==iout && strcmp(statement,"#include")==0) {
        fseek(G__ifile.fp,-1,SEEK_CUR);
        if(G__dispsource) G__disp_mask=1;
        G__include_file();
        iout=0;
        spaceflag=0;
        if(mparen==0 || G__return>G__RETURN_NORMAL) return(G__null);
      }
      if(6==iout && strcmp(statement,"return")==0) {
        fseek(G__ifile.fp,-1,SEEK_CUR);
        if(G__dispsource) G__disp_mask=1;
        G__fgetstream_new(statement,";");
        result=G__return_value(statement);
        if(G__no_exec_compile) {
          spaceflag = -1;
          iout=0;
          if(mparen==0) return(G__null);
          break;
        }
        if(G__prerun==0) return(result);
        G__fignorestream("}");
        return(result);
      }
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
      if(iout>0 && double_quote==0 && statement[iout-1]=='/' && commentflag) {
        iout--;
        if(iout==0) spaceflag=0;
        G__fignoreline();
      }
      else {
        commentflag=1;
        statement[iout++] = c ;
        spaceflag |= 1;
      }
      break;
      
    case '*' :  /* comment */
      if(iout>0 && double_quote==0 && statement[iout-1]=='/' && commentflag) {
        /* start commenting out*/
        iout--;
        if(iout==0) spaceflag=0;
        if(G__skip_comment()==EOF) return(G__null);
      }
      else {
        statement[iout++] = c ;
        spaceflag |= 1;
        if(!double_quote && !single_quote) add_fake_space = 1;
      }
      break;

    case '&' :  /* this cut a symbols! */
      statement[iout++] = c ;
      spaceflag |= 1;
      if(!double_quote && !single_quote) add_fake_space = 1;
      break;
      
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
      else if(8==iout && strcmp(statement,"#include")==0) {
        fseek(G__ifile.fp,-1,SEEK_CUR);
        if(G__dispsource) G__disp_mask=1;
        G__include_file();
        iout=0;
        spaceflag=0;
        if(mparen==0 || G__return>G__RETURN_NORMAL) return(G__null);
      }
      else 
      {
        char *s=strchr(statement,'=');
        if(s) ++s;
        else  s=statement;
        if((1==spaceflag||2==spaceflag) && 
           (G__no_exec==0)&& 
           (('~'==statement[0] && G__defined_templateclass(s+1)) ||
            G__defined_templateclass(s))) {
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
        }
        else {
          statement[iout++] = c ;
          spaceflag |= 1;
        }
      }
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
       /* Make sure that the delimiters that have not been treated
        * in the switch statement do drop the discarded_space */
      /* if (c!='[' && c!=']' && discarded_space && iout) {} */
      if (c!='[' && c!=']' && discarded_space && iout 
          && statement[iout-1]!=':') {
        /* since the character following a discarded space is NOT a
           separator, we have to keep the space */
        statement[iout++] = ' ';
      }
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
    discarded_space = discard_space;
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
int G__readpointer2function(char *new_name,char *pvar_type)
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
int G__search_gotolabel(char *label,fpos_t *pfpos,int line,int *pmparen)
    /* label: NULL if upper level call */
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
        if(0==G__nobreak && 0==G__disp_mask && 0==G__no_exec_compile &&
           G__srcfile[G__ifile.filenum].breakpoint&&
           G__srcfile[G__ifile.filenum].maxline>G__ifile.line_number) {
          G__srcfile[G__ifile.filenum].breakpoint[G__ifile.line_number]
            |=G__TRACED;
        }
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

void G__settemplevel(int val)
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
