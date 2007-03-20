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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"

extern "C" {

int G__const_noerror = 0;

/******************************************************************
* G__const_setnoerror()
******************************************************************/
int G__const_setnoerror()
{
  G__const_noerror = 1;
  return 1;
}

/******************************************************************
* G__const_resetnoerror()
******************************************************************/
int G__const_resetnoerror()
{
  G__const_noerror = 0;
  return 0;
}

/******************************************************************
* G__const_whatnoerror()
******************************************************************/
int G__const_whatnoerror()
{
  return G__const_noerror;
}

/******************************************************************
* G__nosupport()
*
*  print out error message for unsupported capability.
******************************************************************/
void G__nosupport(char* name)
{
  G__fprinterr(G__serr, "Limitation: %s is not supported", name);
  G__printlinenum();
  G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return=G__RETURN_EXIT1);
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
  G__error_flag = 0;
}
#endif

/******************************************************************
* G__malloc_error(varname)
*
******************************************************************/
void G__malloc_error(char* varname)
{
  G__fprinterr(G__serr, "Internal Error: malloc failed for %s", varname);
  G__printlinenum();
  G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return=G__RETURN_EXIT1);
#ifdef G__SECURITY
  G__security_error = G__DANGEROUS;
#endif
}

/******************************************************************
* G__arrayindexerror()
*
*
******************************************************************/
void G__arrayindexerror(int varid, G__var_array* var, char* name, int index)
{
  G__fprinterr(G__serr, "Error: Array index out of range %s -> [%d] ", name, index);
  G__fprinterr(G__serr, " valid upto %s", var->varnamebuf[varid]);
  const int num_of_elements = var->varlabel[varid][1];
  const int stride = var->varlabel[varid][0];
  if (num_of_elements) {
    G__fprinterr(G__serr, "[%d]", (num_of_elements / stride) - 1);
  }
  const int num_of_dimensions = var->paran[varid];
  for (int j = 2; j <= num_of_dimensions; ++j) {
    G__fprinterr(G__serr, "[%d]", var->varlabel[varid][j] - 1);
  }
  G__printlinenum();
  G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return=G__RETURN_EXIT1);
#ifdef G__SECURITY
  G__security_error = G__RECOVERABLE;
#endif
}

#ifdef G__ASM
/**************************************************************************
* G__asm_execerr()
**************************************************************************/
int G__asm_execerr(char* message, int num)
{
  G__fprinterr(G__serr, "Loop Compile Internal Error: %s %d ", message, num);
  G__genericerror(0);
  G__asm_exec = 0;
  return 0;
}
#endif

/**************************************************************************
* G__assign_error()
**************************************************************************/
int G__assign_error(char* item, G__value* pbuf)
{
  if (!G__prerun) {
    if (pbuf->type) {
      G__fprinterr(G__serr, "Error: Incorrect assignment to %s, wrong type '%s'", item, G__type2string(pbuf->type, pbuf->tagnum, pbuf->typenum, pbuf->obj.reftype.reftype, 0));
    }
    else {
      G__fprinterr(G__serr, "Error: Incorrect assignment to %s ", item);
    }
    G__genericerror(0);
  }
#ifdef G__SECURITY
  G__security_error = G__RECOVERABLE;
#endif
  return 0;
}

/**************************************************************************
* G__reference_error()
**************************************************************************/
int G__reference_error(char* item)
{
  G__fprinterr(G__serr, "Error: Incorrect referencing of %s ", item);
  G__genericerror(0);
#ifdef G__SECURITY
  G__security_error = G__RECOVERABLE;
#endif
  return 0;
}

#ifndef G__OLDIMPELMENTATION1174
#ifndef G__OLDIMPELMENTATION1186
/**************************************************************************
* G__splitmessage()
**************************************************************************/
char* G__findrpos(char* s1, char* s2)
{
  if (!s1 || !s2) {
    return 0;
  }
  int i = strlen(s1);
  int s2len = strlen(s2);
  int nest = 0;
  int double_quote = 0;
  int single_quote = 0;
  char c = '\0';
  while (i--) {
    c = s1[i];
    switch (c) {
    case '[':
    case '(':
    case '{':
      if (!double_quote && !single_quote) {
        --nest;
      }
      break;
    case ']':
    case ')':
    case '}':
      if (!double_quote && !single_quote) {
        ++nest;
      }
      break;
    }
    if (!nest && !double_quote && !single_quote) {
      if (!strncmp(s1 + i, s2, s2len)) {
        return s1 + i;
      }
    }
  }
  return 0;
}
#endif

/**************************************************************************
* G__splitmessage()
**************************************************************************/
int G__splitmessage(char* item)
{
  int stat = 0;
  char* dot;
  char* point;
  char* p;
  char* buf = (char*) malloc(strlen(item) + 1);
  strcpy(buf,item);
#ifndef G__OLDIMPELMENTATION1186
  dot = G__findrpos(buf, ".");
  point = G__findrpos(buf, "->");
#else
  dot = strrchr(buf, '.');
  point = G__strrstr(buf, "->");
#endif
  if (dot || point) {
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
         "Error: Failed to evaluate class member '%s' (%s)\n",p,item[0]=='$'?item+1:item);
    }
    else {
      G__fprinterr(G__serr,
         "Error: Failed to evaluate %s\n",item[0]=='$'?item+1:item);
    }
  }
  free((void*) buf);
  return stat;
}
#endif

/**************************************************************************
* G__warnundefined()
**************************************************************************/
int G__warnundefined(char* item)
{
  if(G__prerun&&G__static_alloc&&G__func_now>=0) return(0);
  if(G__no_exec_compile && 0==G__asm_noverflow) return(0);
  if(G__in_pause) return(0);
  if(
     !G__cintv6 &&
     G__ASM_FUNC_COMPILE&G__asm_wholefunction) {
    G__CHECK(G__SECURE_PAUSE,1,G__pause());
    G__CHECK(G__SECURE_EXIT_AT_ERROR,1,G__return=G__RETURN_EXIT1);
  }
  else {
    if(0==G__const_noerror
#ifndef G__OLDIMPELMENTATION1174
       && !G__splitmessage(item)
#endif
       ) {
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
      G__genericerror((char*)NULL);
    }
  }
#ifdef G__SECURITY
  G__security_error = G__RECOVERABLE;
#endif
  return(0);
}

/**************************************************************************
* G__unexpectedEOF()
**************************************************************************/
int G__unexpectedEOF(char* message)
{
  G__eof=2;
  G__fprinterr(G__serr,"Error: Unexpected EOF %s",message);
  G__genericerror((char*)NULL);
  if(0==G__cpp)
    G__fprinterr(G__serr,"Advice: You may need to use +P or -p option\n");
  G__CHECK(G__SECURE_EXIT_AT_ERROR,1,G__return=G__RETURN_EXIT1);
#ifdef G__SECURITY
  G__security_error = G__RECOVERABLE;
#endif
  if(G__NOLINK!=G__globalcomp && (G__steptrace||G__stepover))
    while(0==G__pause()) ;
  return(0);
}

/**************************************************************************
* G__shl_load_error()
**************************************************************************/
int G__shl_load_error(char* shlname, char* message)
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
int G__getvariable_error(char* item)
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
int G__referencetypeerror(char* new_name)
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
int G__err_pointer2pointer(char* item)
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
int G__syntaxerror(char* expr)
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
int G__assignmenterror(char* item)
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
int G__parenthesiserror(char* expression, char* funcname)
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
int G__changeconsterror(char* item, char* categ)
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
  return 0;
}

/**************************************************************************
* G__printlinenum()
**************************************************************************/
int G__printlinenum()
{
  char* format=" FILE:%s LINE:%d\n";
#ifdef VISUAL_CPLUSPLUS
  // make error msg Visual Studio compatible
  format=" %s(%d)\n";
#elif defined(G__ROOT)
  // make error msg GCC compatible
  format=" %s:%d:\n";
#endif
  G__fprinterr(G__serr, format
          ,G__stripfilename(G__ifile.name),G__ifile.line_number);
  return(0);
}

/**************************************************************************
* G__get_security_error()
**************************************************************************/
int G__get_security_error()
{
  return G__security_error;
}

/**************************************************************************
* G__genericerror()
**************************************************************************/
int G__genericerror(const char* message)
{
  if (G__xrefflag) {
    return 1;
  }
  if (!G__const_noerror && (G__cintv6 || (G__asm_wholefunction == G__ASM_FUNC_NOP))) {
    if (message) {
      G__fprinterr(G__serr, "%s", message);
    }
    G__printlinenum();
    G__storelasterror();
  }
  G__CHECK(G__SECURE_PAUSE, 1, G__pause());
  G__CHECK(G__SECURE_EXIT_AT_ERROR, 1, G__return=G__RETURN_EXIT1);
#ifdef G__SECURITY
  G__security_error = G__RECOVERABLE;
#endif
  if (G__aterror) {
    int store_return = G__return;
    G__return = G__RETURN_NON;
    G__p2f_void_void((void*) G__aterror);
    G__return = store_return;
  }
  if (G__cintv6) {
    if (G__cintv6 & G__BC_COMPILEERROR) {
      G__bc_throw_compile_error();
    }
    if (G__cintv6 & G__BC_RUNTIMEERROR) {
      G__bc_throw_runtime_error();
    }
  }
  return 0;
}

#ifndef G__FRIEND
/**************************************************************************
* G__friendignore
*
**************************************************************************/
int G__friendignore(int* piout, int* pspaceflag, int mparen)
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
int G__externignore(int* piout, int* pspaceflag, int mparen)
{
  int flag=0;
  int c;
  int store_iscpp;

  G__var_type='p';
  c = G__fgetspace();
  if('"'==c) {
    /* extern "C" {  } */
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
    fseek(G__ifile.fp,-1,SEEK_CUR);
    if(G__dispsource) G__disp_mask=1;
    G__exec_statement();
    G__iscpp=store_iscpp;
    if(flag) G__ResetShlHandle();
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
int G__handleEOF(char* statement, int mparen, int single_quote, int double_quote)
{
  G__eof=1;
  if((mparen!=0)||(single_quote!=0)||(double_quote!=0)){
    G__unexpectedEOF("G__exec_statement()");
  }
  if(strcmp(statement,"")!=0 && strcmp(statement,"#endif")!=0 &&
     statement[0]!=0x1a) {
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
int G__check_drange(int p, double low, double up, double d, G__value* result7, char* funcname)
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
int G__check_lrange(int p, long low, long up, long l, G__value* result7, char* funcname)
{
  if(l<low||up<l) {
    G__fprinterr(G__serr,"Error: %s param[%d]=%ld up:%ld low:%ld out of range"
            ,funcname,p,l,up,low);
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
int G__check_type(int p, int t1, int t2, G__value* para, G__value* result7, char* funcname)
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
int G__check_nonull(int p, int t, G__value* para, G__value* result7, char* funcname)
{
  long l;
  l = G__int(*para);
  if(0==l) {
    G__fprinterr(G__serr,"Error: %s param[%d]=%ld must not be 0",funcname,p,l);
    G__genericerror((char*)NULL);
    *result7=G__null;
    return(1);
  }
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
  else {
    return(0);
  }
}
#endif

/**************************************************************************
* G__printerror()
*
**************************************************************************/
void G__printerror(char* funcname, int ipara, int paran)
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
  G__security_error = G__RECOVERABLE;
#endif
  return(0);
}

/**************************************************************************
* G__missingsemicolumn()
*
**************************************************************************/
int G__missingsemicolumn(char* item)
{
  G__fprinterr(G__serr,"Syntax Error: %s Maybe missing ';'",item);
  G__genericerror((char*)NULL);
  return(0);
}

} // extern "C"

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
