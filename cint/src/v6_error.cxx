/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file error.c
 ************************************************************************
 * Description:
 *  Show error message 
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


#ifndef G__OLDIMPLEMENTATION1103
int G__const_noerror=0;
/******************************************************************
* G__const_setnoerror()
******************************************************************/
int G__const_setnoerror() {
  G__const_noerror = 1;
  return(G__const_noerror);
}
/******************************************************************
* G__const_resetnoerror()
******************************************************************/
int G__const_resetnoerror() {
  G__const_noerror = 0;
  return(G__const_noerror);
}
#endif

#ifndef G__OLDIMPLEMENTATION1528
/******************************************************************
* G__const_whatnoerror()
******************************************************************/
int G__const_whatnoerror() {
  return(G__const_noerror);
}
#endif

/******************************************************************
* G__nosupport()
*
*  print out error message for unsupported capability.
******************************************************************/
void G__nosupport(name)
/* struct G__input_file *fin; */
char *name;
{

  G__fprinterr(G__serr, "Limitation: %s is not supported", name);
  G__printlinenum();
  G__CHECK(G__SECURE_EXIT_AT_ERROR,1,G__return=G__RETURN_EXIT1);
}


#ifdef G__NEVER
/******************************************************************
* void G__error_clear()
*
* Called by
*   nothing
******************************************************************/
void G__error_clear()
{
	G__error_flag=0;
}
#endif



/******************************************************************
* G__malloc_error(varname)
*
******************************************************************/
void G__malloc_error(varname)
char *varname;
{
  G__fprinterr(G__serr,"Internal Error: malloc failed for %s", varname);
  G__printlinenum();
  G__CHECK(G__SECURE_EXIT_AT_ERROR,1,G__return=G__RETURN_EXIT1);
#ifdef G__SECURITY
  G__security_error = G__DANGEROUS;
#endif
}


/******************************************************************
* G__arrayindexerror()
*
*
******************************************************************/
void G__arrayindexerror(ig15 ,var ,item,p_inc)
int ig15;
struct G__var_array *var;
char *item;
int p_inc;
{
  int ig25;
  
  G__fprinterr(G__serr,"Error: Array index out of range %s -> [%d] " ,item,p_inc);
  G__fprinterr(G__serr," valid upto %s",var->varnamebuf[ig15]);
  if(var->varlabel[ig15][1]!=0)
    G__fprinterr(G__serr,"[%d]",(var->varlabel[ig15][1]+1)/var->varlabel[ig15][0]-1);
  ig25=2;
  while(ig25<var->paran[ig15]+1) {
    G__fprinterr(G__serr,"[%d]",var->varlabel[ig15][ig25]-1);
    ig25++;
  }
  G__printlinenum();
  G__CHECK(G__SECURE_EXIT_AT_ERROR,1,G__return=G__RETURN_EXIT1);
#ifdef G__SECURITY
  G__security_error = G__RECOVERABLE;
#endif
}



#ifdef G__ASM
/**************************************************************************
* G__asm_execerr()
**************************************************************************/
int G__asm_execerr(message,num)
char *message;
int num;
{
  G__fprinterr(G__serr,"Loop Compile Internal Error: %s %d ",message,num);
  G__genericerror((char*)NULL);
  G__asm_exec=0;
  return(0);
}
#endif

/**************************************************************************
* G__assign_error()
**************************************************************************/
int G__assign_error(item,pbuf)
char *item;
G__value *pbuf;
{
  if(0==G__prerun) {
    if(pbuf->type) {
      G__fprinterr(G__serr,"Error: Incorrect assignment to %s, wrong type '%s'"
	      ,item ,G__type2string(pbuf->type ,pbuf->tagnum ,pbuf->typenum
				    ,pbuf->obj.reftype.reftype,0));
    }
    else {
      G__fprinterr(G__serr,"Error: Incorrect assignment to %s ",item);
    }
    G__genericerror((char*)NULL);
  }
#ifdef G__SECURITY
  G__security_error = G__RECOVERABLE;
#endif
  return(0);
}

/**************************************************************************
* G__reference_error()
**************************************************************************/
int G__reference_error(item)
char *item;
{
  G__fprinterr(G__serr,"Error: Incorrect referencing of %s ",item);
  G__genericerror((char*)NULL);
#ifdef G__SECURITY
  G__security_error = G__RECOVERABLE;
#endif
  return(0);
}

#ifndef G__OLDIMPELMENTATION1174
#ifndef G__OLDIMPELMENTATION1186
/**************************************************************************
* G__splitmessage()
**************************************************************************/
char* G__findrpos(s1,s2)
char* s1;
char* s2;
{
  char c;
  int nest=0, double_quote=0, single_quote=0;
  int i = strlen(s1);
  int s2len = strlen(s2);

  if(!s1 || !s2) return(0);
  while(i--) {
    c = s1[i];
    switch(c) {
    case '[':
    case '(':
    case '{':
     if(!double_quote && !single_quote) nest--;
     break;
    case ']':
    case ')':
    case '}':
     if(!double_quote && !single_quote) nest++;
     break;
    }
    if(!nest && !double_quote && !single_quote) {
      if(0==strncmp(s1+i,s2,s2len)) return(s1+i);
    }
  }
  return(0);
}
#endif

/**************************************************************************
* G__splitmessage()
**************************************************************************/
int G__splitmessage(item) 
char *item;
{
  int stat=0;
  char *dot;
  char *point;
  char *buf;
  char *p;
  buf = (char*)malloc(strlen(item)+1);
  strcpy(buf,item);

#ifndef G__OLDIMPELMENTATION1186
  dot = G__findrpos(buf,".");
  point = G__findrpos(buf,"->");
#else
  dot = strrchr(buf,'.');
  point = G__strrstr(buf,"->");
#endif
  if(dot || point) {
    G__value result;
    if(!dot || (point && point>dot) ) {
      p=point;
      *p = 0;
      p += 2;
    }
    else {
      p=dot;
      *p = 0;
      p += 1;
    }
    stat = 1;
    result = G__getexpr(buf);
    if(result.type) {
      G__fprinterr(G__serr,
	 "Error: Failed to evaluate class member '%s' (%s)",p,item[0]=='$'?item+1:item);
    }
    else {
      G__fprinterr(G__serr,
	 "Error: Failed to evaluate %s",item[0]=='$'?item+1:item);
    }
  }
  free((void*)buf);
  return(stat);
}
#endif

/**************************************************************************
* G__warnundefined()
**************************************************************************/
int G__warnundefined(item)
char *item;
{
#ifndef G__OLDIMPLEMENTATION997
  if(G__prerun&&G__static_alloc&&G__func_now>=0) return(0);
#endif
#ifndef G__OLDIMPLEMENTATION997
  if(G__no_exec_compile && 0==G__asm_noverflow) return(0);
#endif
  if(G__in_pause) return(0);
  if(
#ifndef G__OLDIMPLEMENTATION2105
     !G__cintv6 &&
#endif
     G__ASM_FUNC_COMPILE&G__asm_wholefunction) {
    G__CHECK(G__SECURE_PAUSE,1,G__pause());
    G__CHECK(G__SECURE_EXIT_AT_ERROR,1,G__return=G__RETURN_EXIT1);
  }
  else {
#ifndef G__OLDIMPLEMENTATION1103
    if(0==G__const_noerror
#ifndef G__OLDIMPELMENTATION1174
       && !G__splitmessage(item)
#endif
       ) {
#endif
#ifndef G__OLDIMPLEMENTATION1571
      char *p = strchr(item,'(');
      if(p) {
	char tmp[G__ONELINE];
	strcpy(tmp,item);
	p = G__strrstr(tmp,"::");
	if(p) {
	  *p = 0; p+=2;
	  G__fprinterr(G__serr,
		       "Error: Function %s is not defined in %s ",p,tmp);
	}
	else {
	  G__fprinterr(G__serr,
		       "Error: Function %s is not defined in current scope "
		       ,item[0]=='$'?item+1:item);
	}
      }
      else {
	char tmp[G__ONELINE];
	strcpy(tmp,item);
	if(p) {
	  *p = 0; p+=2;
	  G__fprinterr(G__serr,
		       "Error: Symbol %s is not defined in %s ",p,tmp);
	}
	else {
	  G__fprinterr(G__serr,
		       "Error: Symbol %s is not defined in current scope ",item[0]=='$'?item+1:item);
	}
      }
#else
#ifdef G__ROOT
      G__fprinterr(G__serr,
	      "Error: No symbol %s in current scope ",item[0]=='$'?item+1:item);
#else
      G__fprinterr(G__serr,
	      "Error: No symbol %s in current scope ",item);
#endif
#endif
#ifndef G__OLDIMPLEMENTATION1519
      G__genericerror((char*)NULL);
#endif
#ifndef G__OLDIMPLEMENTATION1103
    }
#endif
#ifdef G__OLDIMPLEMENTATION1519
    G__genericerror((char*)NULL);
#endif
  }
#ifdef G__SECURITY
  G__security_error = G__RECOVERABLE;
#endif
  return(0);
}

/**************************************************************************
* G__unexpectedEOF()
**************************************************************************/
int G__unexpectedEOF(message)
char *message;
{
  G__eof=2;
  G__fprinterr(G__serr,"Error: Unexpected EOF %s",message);
#ifndef G__OLDIMPLEMENTATION1086
  G__genericerror((char*)NULL);
#else
  G__printlinenum();
#endif
  if(0==G__cpp) 
    G__fprinterr(G__serr,"Advice: You may need to use +P or -p option\n");
  G__CHECK(G__SECURE_EXIT_AT_ERROR,1,G__return=G__RETURN_EXIT1);
#ifdef G__SECURITY
  G__security_error = G__RECOVERABLE;
#endif
#ifndef G__OLDIMPLEMENTATION1771
  if(G__NOLINK!=G__globalcomp && (G__steptrace||G__stepover)) 
    while(0==G__pause()) ;
#else
  if(G__steptrace||G__stepover) while(0==G__pause()) ;
#endif
  return(0);
}

/**************************************************************************
* G__shl_load_error()
**************************************************************************/
int G__shl_load_error(shlname,message)
char *shlname,*message;
{
  G__fprinterr(G__serr,"%s: Failed to load Dynamic link library %s\n",message,shlname);
  G__CHECK(G__SECURE_EXIT_AT_ERROR,1,G__return=G__RETURN_EXIT1);
#ifdef G__SECURITY
  G__security_error = G__RECOVERABLE;
#endif
  return(0);
}

/**************************************************************************
* G__getvariable_error()
**************************************************************************/
int G__getvariable_error(item)
char *item;
{
  G__fprinterr(G__serr,"Error: G__getvariable: expression %s",item);
  G__printlinenum();
  G__CHECK(G__SECURE_EXIT_AT_ERROR,1,G__return=G__RETURN_EXIT1);
#ifdef G__SECURITY
  G__security_error = G__RECOVERABLE;
#endif
  return(0);
}

/**************************************************************************
* G__referenceytypeerror()
**************************************************************************/
int G__referencetypeerror(new_name)
char *new_name;
{
  G__fprinterr(G__serr,"Error: Can't take address for reference type %s",new_name);
  G__printlinenum();
  G__CHECK(G__SECURE_EXIT_AT_ERROR,1,G__return=G__RETURN_EXIT1);
#ifdef G__SECURITY
  G__security_error = G__RECOVERABLE;
#endif
  return(0);
}


/**************************************************************************
* G__err_pointer2pointer(item);
**************************************************************************/
int G__err_pointer2pointer(item)
char *item;
{
  G__fprinterr(G__serr,"Limitation: Pointer to pointer %s not supported",item);
  G__printlinenum();
  G__CHECK(G__SECURE_EXIT_AT_ERROR,1,G__return=G__RETURN_EXIT1);
#ifdef G__SECURITY
  G__security_error = G__DANGEROUS;
#endif
  return(0);
}

/**************************************************************************
* G__syntaxerror()
**************************************************************************/
int G__syntaxerror(expr)
char *expr;
{
  G__fprinterr(G__serr,"Syntax Error: %s",expr);
  G__genericerror((char*)NULL);
#ifdef G__SECURITY
  G__security_error = G__RECOVERABLE;
#endif
  return(0);
}

/**************************************************************************
* G__assignmenterror()
**************************************************************************/
int G__assignmenterror(item)
char *item;
{
  G__fprinterr(G__serr,"Error: trying to assign to %s",item);
  G__printlinenum();
  G__CHECK(G__SECURE_EXIT_AT_ERROR,1,G__return=G__RETURN_EXIT1);
#ifdef G__SECURITY
  G__security_error = G__RECOVERABLE;
#endif
  return(0);
}

/**************************************************************************
* G__parenthesiserror()
**************************************************************************/
int G__parenthesiserror(expression,funcname)
char *expression,*funcname;
{
  G__fprinterr(G__serr,"Syntax error: %s: Parenthesis or quotation unmatch %s"
	  ,funcname ,expression);
  G__printlinenum();
  G__CHECK(G__SECURE_EXIT_AT_ERROR,1,G__return=G__RETURN_EXIT1);
#ifdef G__SECURITY
  G__security_error = G__RECOVERABLE;
#endif
  return(0);
}

/**************************************************************************
* G__commenterror()
**************************************************************************/
int G__commenterror()
{
  G__fprinterr(G__serr, "Syntax error: Unexpected '/' Comment?");
  G__printlinenum();
  G__CHECK(G__SECURE_EXIT_AT_ERROR,1,G__return=G__RETURN_EXIT1);
#ifdef G__SECURITY
  G__security_error = G__RECOVERABLE;
#endif
  return(0);
}

/**************************************************************************
* G__changeconsterror()
**************************************************************************/
int G__changeconsterror(item,categ)
char *item,*categ;
{
  if(G__dispmsg>=G__DISPWARN) {
    G__fprinterr(G__serr,"Warning: Re-initialization %s %s",categ,item);
    G__printlinenum();
  }
  if(0==G__prerun) {
    G__CHECK(G__SECURE_EXIT_AT_ERROR,1,G__return=G__RETURN_EXIT1);
#ifdef G__SECURITY
    G__security_error = G__RECOVERABLE;
#endif
  }
  return(0);
}

/**************************************************************************
* G__printlinenum()
**************************************************************************/
int G__printlinenum()
{
#ifndef G__OLDIMPLEMENTATION1196
  G__fprinterr(G__serr," FILE:%s LINE:%d\n" 
	  ,G__stripfilename(G__ifile.name),G__ifile.line_number);
#else
#ifndef G__OLDIMPLEMENTATION1171
  char *p;
  p = strstr(G__ifile.name,"./");
  if(!p) p = G__ifile.name;
  G__fprinterr(G__serr," FILE:%s LINE:%d\n" ,p,G__ifile.line_number);
#else
  G__fprinterr(G__serr," FILE:%s LINE:%d\n" ,G__ifile.name,G__ifile.line_number);
#endif
#endif
  return(0);
}

/**************************************************************************
* G__get_security_error()
**************************************************************************/
int G__get_security_error()
{
  return(G__security_error);
}


/**************************************************************************
* G__genericerror()
**************************************************************************/
int G__genericerror(message)
char *message;
{
#ifndef G__OLDIMPLEMENTATION1164
  if(G__xrefflag) return(1);
#endif

  if(
#ifndef G__OLDIMPLEMENTATION2105
     G__cintv6 ||
#endif
     G__ASM_FUNC_NOP==G__asm_wholefunction) {
#ifndef G__OLDIMPLEMENTATION1103
    if(0==G__const_noerror) {
#endif
      if(message) G__fprinterr(G__serr,"%s",message);
      G__printlinenum();
#ifndef G__OLDIMPLEMENTATION1198
      G__storelasterror();
#endif
#ifndef G__OLDIMPLEMENTATION1103
    }
#endif
  }

  G__CHECK(G__SECURE_PAUSE,1,G__pause());
  G__CHECK(G__SECURE_EXIT_AT_ERROR,1,G__return=G__RETURN_EXIT1);
#ifdef G__SECURITY
  G__security_error = G__RECOVERABLE;
#endif

#ifndef G__OLDIMPLEMENTATION875
  if(G__aterror) {
    int store_return=G__return;
    G__return=G__RETURN_NON;
    G__p2f_void_void((void*)G__aterror);
    G__return=store_return;
  }
#endif

#ifndef G__OLDIMPLEMENTATION2117
  if(G__cintv6) {
    if(G__cintv6&G__BC_COMPILEERROR) G__bc_throw_compile_error();
    if(G__cintv6&G__BC_RUNTIMEERROR) G__bc_throw_runtime_error();
  }
#endif

  return(0);
}

#ifndef G__FRIEND
/**************************************************************************
* G__friendignore
*
**************************************************************************/
int G__friendignore(piout,pspaceflag,mparen)
int *piout,*pspaceflag;
int mparen;
{
#ifdef G__FRIEND
  int friendtagnum,envtagnum;
  struct G__friendtag *friendtag;
  int tagtype=0;
#else
  static int state=1;
#endif
  int store_tagnum,store_def_tagnum,store_def_struct_member;
  int store_tagdefining,store_access;
  fpos_t pos;
  int line_number;
  char classname[G__ONELINE];
  int c;

#ifndef G__FRIEND
  if(G__NOLINK==G__store_globalcomp&&G__NOLINK==G__globalcomp) {
    if(state) {
      G__genericerror("Limitation: friend privilege not supported");
      state=0;
    }
  }
#endif

  fgetpos(G__ifile.fp,&pos);
  line_number = G__ifile.line_number;
  c = G__fgetname_template(classname,";");
  if(isspace(c)) {
    if(strcmp(classname,"class")==0) {
      c=G__fgetname_template(classname,";");
      tagtype='c';
    }
    else if(strcmp(classname,"struct")==0) {
      c=G__fgetname_template(classname,";");
      tagtype='s';
    }
  }

#ifdef G__FRIEND
  envtagnum = G__get_envtagnum();
  if(-1==envtagnum) {
    G__genericerror("Error: friend keyword appears outside class definition");
  }
#endif
  
  
#ifdef G__FRIEND
  store_tagnum = G__tagnum;
  store_def_tagnum = G__def_tagnum;
  store_def_struct_member = G__def_struct_member;
  store_tagdefining = G__tagdefining;
  store_access = G__access;
  
  G__friendtagnum=envtagnum;
  G__tagnum = -1;
  G__def_tagnum = -1;
  G__tagdefining = -1;
  G__def_struct_member = 0;
  G__access = G__PUBLIC;
  G__var_type='p';

  if(tagtype) {
    friendtagnum=G__search_tagname(classname,tagtype);
    /* friend class ... ; */
    if(-1!=envtagnum) {
      friendtag = G__struct.friendtag[friendtagnum];
      if(friendtag) {
	while(friendtag->next) friendtag=friendtag->next;
	friendtag->next
	  =(struct G__friendtag*)malloc(sizeof(struct G__friendtag));
	friendtag->next->next=(struct G__friendtag*)NULL;
	friendtag->next->tagnum=envtagnum;
      }
      else {
	G__struct.friendtag[friendtagnum]
	  =(struct G__friendtag*)malloc(sizeof(struct G__friendtag));
	friendtag = G__struct.friendtag[friendtagnum];
	friendtag->next=(struct G__friendtag*)NULL;
	friendtag->tagnum=envtagnum;
      }
    }
#else
  if(-1!=G__defined_tagname(classname,1)) {
#endif
    if(';'!=c) c = G__fignorestream(";");
  }
  else {
    /* friend type f() {  } ; */
    fsetpos(G__ifile.fp,&pos);
    G__ifile.line_number = line_number;

#ifndef G__FRIEND
    store_tagnum = G__tagnum;
    store_def_tagnum = G__def_tagnum;
    store_def_struct_member = G__def_struct_member;
    store_tagdefining = G__tagdefining;
    store_access = G__access;
    G__tagnum = -1;
    G__def_tagnum = -1;
    G__tagdefining = -1;
    G__def_struct_member = 0;
    G__access = G__PUBLIC;
    G__var_type='p';
#endif
    
    G__exec_statement();
    
#ifndef G__FRIEND
    G__tagnum = store_tagnum;
    G__def_tagnum = store_def_tagnum;
    G__def_struct_member = store_def_struct_member;
    G__tagdefining = store_tagdefining;
    G__access = store_access;
#endif
  }
    
#ifdef G__FRIEND
  G__tagnum = store_tagnum;
  G__def_tagnum = store_def_tagnum;
  G__def_struct_member = store_def_struct_member;
  G__tagdefining = store_tagdefining;
  G__access = store_access;
  G__friendtagnum = -1;
#endif
    
  *pspaceflag = -1;
  *piout=0;
  return(!mparen);
}
#endif

/**************************************************************************
* G__externignore
*
**************************************************************************/
int G__externignore(piout,pspaceflag,mparen)
int *piout,*pspaceflag;
int mparen;
{
  int flag=0;
  int c;
  int store_iscpp;

  G__var_type='p';
  c = G__fgetspace();
  if('"'==c) {
    /* extern "C" {  } */
#ifndef G__OLDIMPLEMENTATION1908
    char fname[G__MAXFILENAME];
    int flag=0;
    c = G__fgetstream(fname,"\"");
    store_iscpp=G__iscpp;
    /* if('{'==c) G__iscpp=0; */
    if(0==strcmp(fname,"C")) {
      G__iscpp=0;
    }
    else {
      G__loadfile(fname);
      G__SetShlHandle(fname);
      flag=1;
    }
    *pspaceflag = -1;
    *piout=0;
    c=G__fgetspace();
#else
    c=G__fignorestream("\"");
    *pspaceflag = -1;
    *piout=0;
    c=G__fgetspace();
    store_iscpp=G__iscpp;
    if('{'==c) G__iscpp=0;
#endif
    fseek(G__ifile.fp,-1,SEEK_CUR);
    if(G__dispsource) G__disp_mask=1;
    G__exec_statement();
    G__iscpp=store_iscpp;
#ifndef G__OLDIMPLEMENTATION1908
    if(flag) G__ResetShlHandle();
#endif
    return(0);
  }
  else {
    fseek(G__ifile.fp,-1,SEEK_CUR);
    if(c=='\n' /* ||c=='\r' */) --G__ifile.line_number;
    if(G__dispsource) G__disp_mask=1;
    if(G__globalcomp==G__NOLINK && 0==flag && 0==G__parseextern) {
      G__fignorestream(";");
    }
    *pspaceflag = -1;
    *piout=0;
    return(!mparen);
  }
}


/**************************************************************************
* G__handleEOF()
*
*  separated from G__exec_statement()
**************************************************************************/
int G__handleEOF(statement,mparen,single_quote,double_quote)
char *statement;
int mparen,single_quote,double_quote;
{
  G__eof=1;
  if((mparen!=0)||(single_quote!=0)||(double_quote!=0)){
    G__unexpectedEOF("G__exec_statement()");
  }
#ifndef G__OLDIMPLEMENTATION632
  if(strcmp(statement,"")!=0 && strcmp(statement,"#endif")!=0 &&
     statement[0]!=0x1a) {
#else
  if(strcmp(statement,"")!=0 && strcmp(statement,"#endif")) {
#endif
    G__fprinterr(G__serr,"Report: Unrecognized string '%s' ignored",statement);
    G__printlinenum();
  }
  return(0);
}

#ifdef G__SECURITY
/******************************************************************
* G__check_drange()
* check for double
******************************************************************/
int G__check_drange(p,low,up,d,result7,funcname)
int p;
double low;
double up;
double d;
G__value *result7;
char *funcname;
{
  if(d<low||up<d) { 
    G__fprinterr(G__serr,"Error: %s param[%d]=%g up:%g low:%g out of range"
	    ,funcname,p,d,up,low); 
    G__genericerror((char*)NULL); 
    *result7=G__null; 
    return(1); 
  } 
  else {
    return(0);
  }
}

/******************************************************************
* G__check_lrange()
* check for long
******************************************************************/
int G__check_lrange(p,low,up,l,result7,funcname)
int p;
long low;
long up;
long l;
G__value *result7;
char *funcname;
{
  if(l<low||up<l) { 
#ifndef G__FONS31
    G__fprinterr(G__serr,"Error: %s param[%d]=%ld up:%ld low:%ld out of range"
	    ,funcname,p,l,up,low); 
#else
    G__fprinterr(G__serr,"Error: %s param[%d]=%d up:%d low:%d out of range"
	    ,funcname,p,l,up,low); 
#endif
    G__genericerror((char*)NULL); 
    *result7=G__null; 
    return(1); 
  } 
  else {
    return(0);
  }
}

/******************************************************************
* G__check_type()
* check for NULL pointer
******************************************************************/
int G__check_type(p,t1,t2,para,result7,funcname)
int p;
int t1;
int t2;
G__value *para;
G__value *result7;
char *funcname;
{
  if(para->type!=t1 && para->type!=t2) {
    G__fprinterr(G__serr,"Error: %s param[%d] type mismatch",funcname,p); 
    G__genericerror((char*)NULL); 
    *result7=G__null; 
    return(1);
  }
  return(0);
}

/******************************************************************
* G__check_nonull()
* check for NULL pointer
******************************************************************/
int G__check_nonull(p
#ifndef G__OLDIMPLEMENTATION575
		    ,t,para
#else
		    ,l
#endif
		    ,result7,funcname)
int p;
#ifndef G__OLDIMPLEMENTATION575
int t;
G__value *para;
#else
long l;
#endif
G__value *result7;
char *funcname;
{
#ifndef G__OLDIMPLEMENTATION575
  long l;
  l = G__int(*para);
#endif
  if(0==l) { 
    G__fprinterr(G__serr,"Error: %s param[%d]=%ld must not be 0",funcname,p,l); 
    G__genericerror((char*)NULL); 
    *result7=G__null; 
    return(1); 
  } 
#ifndef G__OLDIMPLEMENTATION575
  else if(t!=para->type) {
    if('Y'!=t){
      G__fprinterr(G__serr,"Error: %s parameter mismatch param[%d] %c %c"
	      ,funcname,p,t,para->type); 
      G__genericerror((char*)NULL); 
      *result7=G__null; 
      return(1); 
    }
    return(0);
  }
#endif
  else {
    return(0);
  }
}

#endif


/**************************************************************************
* G__printerror()
*
**************************************************************************/
void G__printerror(funcname,ipara,paran)
char *funcname;
int ipara,paran;
{
  if(G__dispmsg>=G__DISPWARN) {
    G__fprinterr(G__serr
		 ,"Warning: %s() expects %d parameters, %d parameters given"
		 ,funcname,ipara,paran);
    G__printlinenum();
  }
  G__CHECK(G__SECURE_EXIT_AT_ERROR,1,G__return=G__RETURN_EXIT1);
#ifdef G__SECURITY
  G__security_error = G__RECOVERABLE;
#endif
}


/**************************************************************************
* G__pounderror()
*
*  #error xxx
*
**************************************************************************/
int G__pounderror()
{
  char buf[G__ONELINE];
  char *p;
  fgets(buf,G__ONELINE,G__ifile.fp);
  p = strchr(buf,'\n');
  if(p) *p='\0';
  p = strchr(buf,'\r');
  if(p) *p='\0';
  G__fprinterr(G__serr,"#error %s\n",buf);
  G__CHECK(G__SECURE_EXIT_AT_ERROR,1,G__return=G__RETURN_EXIT1);
#ifdef G__SECURITY
#ifndef G__OLDIMPLEMENTATION1199
  G__security_error = G__RECOVERABLE;
#else
  G__security_error = G__DANGEROUS;
#endif
#endif
  return(0);
}

/**************************************************************************
* G__missingsemicolumn()
*
**************************************************************************/
int G__missingsemicolumn(item)
char *item;
{
  G__fprinterr(G__serr,"Syntax Error: %s Maybe missing ';'",item);
  G__genericerror((char*)NULL);
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
