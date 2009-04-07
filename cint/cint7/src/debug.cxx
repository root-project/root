/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file debug.c
 ************************************************************************
 * Description:
 *  Debugger capability
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"

#include "Dict.h"
#include "Reflex/Tools.h"

using namespace Cint::Internal;

/*********************************************************
* G__setdebugcond()
*
*  Pre-set trace/debug condition for speed up
*********************************************************/

extern "C" void G__setdebugcond()
{
  G__dispsource=G__step+G__break+G__debug;
  if(G__dispsource==0) G__disp_mask=0;
  if((G__break||G__step)&&0==G__prerun) G__breaksignal=1;
  else                                  G__breaksignal=0;
}


/****************************************************************
* G__findposition
*
*  return   0    source line not found
*  return   1    source file exists but line not exact
*  return   2    source line exactly found
****************************************************************/
int Cint::Internal::G__findposition(char *string,G__input_file view,int *pline, int *pfnum)
{
  int i=0;

  /* preset current position */
  *pline=view.line_number;
  *pfnum=view.filenum;

  /* skip space */
  while(isspace(string[i])) i++;

  if('\0'==string[i]) {
    if('\0'==view.name[0]) return(0);
    *pline=view.line_number;
    if(view.line_number<1||G__srcfile[view.filenum].maxline<=view.line_number)
      return(1);
    else
      return(2);
  }
  else if(isdigit(string[i])) {
    if('\0'==view.name[0]) return(0);
    *pline=atoi(string+i);
  }
  else {
    return(G__findfuncposition(string+i,pline,pfnum));
  }

  if(*pfnum<0 || G__nfile <= *pfnum) {
    *pfnum=view.filenum;
    *pline=view.line_number;
    return(0);
  }
  else if(*pline<1) {
    *pline=1;
    return(1);
  }         
  else if(G__srcfile[*pfnum].maxline<*pline) {
    *pline=G__srcfile[*pfnum].maxline-1;
    return(1);
  }
  return(2);
}

/****************************************************************
* G__findfuncposition()
****************************************************************/
int Cint::Internal::G__findfuncposition(char *func,int *pline,int *pfnum)
{
   if (func==0) return 0;

   ::Reflex::Scope scope( ::Reflex::Scope::GlobalScope() );

   size_t len = strlen(func);
   if (len>2 && func[len-1]==':' && func[len-2]==':') {
      std::string scopename(func,0,len-2);
      scope = scope.ByName( scopename );
      if (scope) {
         *pline = G__get_properties(scope)->linenum;
         *pfnum = G__get_properties(scope)->filenum;
         return(2);
      }
   }

   G__incsetup_memfunc(scope);

   ::Reflex::Member ifunc = scope.MemberByName(func);
   if (ifunc) {
      *pline = G__get_properties(ifunc)->linenum;
      *pfnum = G__get_properties(ifunc)->filenum;
      return(2);
   }
   return 0;
}

/****************************************************************
* G__display_proto()
****************************************************************/
int Cint::Internal::G__display_proto(FILE *fp,const char *func)
{
   return G__display_proto_pretty(fp,func,0);
}

/****************************************************************
* G__display_proto_pretty()
****************************************************************/
int Cint::Internal::G__display_proto_pretty(FILE *fp,const char *func, char friendlyStyle)
{
   ::Reflex::Scope scope( ::Reflex::Scope::GlobalScope() );

   size_t i = 0;
   while(isspace(func[i])) ++i;

   std::string scopename = ::Reflex::Tools::GetScopeName( func+i );
   ::Reflex::Scope ifunc = scope.ByName( scopename ) ;
   if (!ifunc) ifunc = scope;

   int tagnum = G__get_tagnum(ifunc);
   if (tagnum>0) G__incsetup_memfunc(tagnum);

   std::string funcname;

   size_t len = strlen(func);
   if (len>2 && func[len-1]==':' && func[len-2]==':') {
      i = 0;
   } else {
      funcname = ::Reflex::Tools::GetBaseName( func+i );
      i = funcname.length();
      while(i&&(isspace(funcname[i-1])||'('==funcname[i-1])) funcname[--i]='\0';
   }

   if(i) {
      if(G__listfunc_pretty(fp,G__PUBLIC_PROTECTED_PRIVATE,funcname.c_str(),ifunc,friendlyStyle)) return(1);
   }
   else  {
      if(G__listfunc_pretty(fp,G__PUBLIC_PROTECTED_PRIVATE,0,ifunc,friendlyStyle))return(1);
   }

   if(ifunc.IsClass()) {
      size_t i1;
      struct G__inheritance *baseclass = G__struct.baseclass[tagnum];
      for(i1=0;i1<baseclass->vec.size();i1++) {
         ifunc = G__Dict::GetDict().GetScope(baseclass->vec[i1].basetagnum);
         if(i) {
            if(G__listfunc_pretty(fp,G__PUBLIC_PROTECTED_PRIVATE,funcname.c_str(),ifunc,friendlyStyle)) 
               return(1);
         }
         else  {
            if(G__listfunc_pretty(fp,G__PUBLIC_PROTECTED_PRIVATE,0,ifunc,friendlyStyle))
               return(1);
         }
      }
   }
   return(0);
}

static int G__tempfilenum = G__MAXFILE-1;

/****************************************************************
* G__gettempfilenum()
****************************************************************/
extern "C" int G__gettempfilenum()
{
  return(G__tempfilenum);
}


/****************************************************************
* G__exec_tempfile_core()
****************************************************************/
static G__value G__exec_tempfile_core(const char *file,FILE *fp)
{
#ifdef G__EH_SIGNAL
  void (*fpe)();
  void (*segv)();
#ifdef SIGILL
  void (*ill)();
#endif
#ifdef SIGEMT
  void (*emt)();
#endif
#ifdef SIGBUS
  void (*bus)();
#endif
#endif

  long asm_inst_g[G__MAXINST]; /* p-code instruction buffer */
  G__value asm_stack_g[G__MAXSTACK]; /* data stack */
  G__StrBuf asm_name_sb(G__ASM_FUNCNAMEBUF);
  char *asm_name = asm_name_sb;

  long *store_asm_inst;
  G__value *store_asm_stack;
  char *store_asm_name;
  int store_asm_name_p;
  struct G__param *store_asm_param;
  /* int store_asm_exec; */
  int store_asm_noverflow;
  int store_asm_cp;
  int store_asm_dt;
  ::Reflex::Member store_asm_index; /* maybe unneccessary */

  int len;

  fpos_t pos;
  char store_var_type;
  struct G__input_file ftemp,store_ifile;
  G__value buf = G__null;
#ifdef G__ASM
  G__ALLOC_ASMENV;
#endif

  G__LockCriticalSection();

  /*************************************************
  * delete space chars at the end of filename
  *************************************************/
  char *filename = 0;
  if(file) {
    len = strlen(file);
    filename = new char[len+1];
    strcpy(filename,file);
    while(len>1&&isspace(filename[len-1])) {
      filename[--len]='\0';
    }
  
#ifndef G__WIN32
    ftemp.fp = fopen(filename,"r");
#else
    ftemp.fp = fopen(filename,"rb");
#endif
  }
  else {
    fseek(fp,0L,SEEK_SET);
    ftemp.fp = fp;
  }

  if(ftemp.fp) {
    ftemp.line_number = 1;
    if(file) { strcpy(ftemp.name,filename); delete [] filename; }
    else     strcpy(ftemp.name,"(tmpfile)");
    ftemp.filenum = G__tempfilenum;
    G__srcfile[G__tempfilenum].fp = ftemp.fp;
    G__srcfile[G__tempfilenum].filename=ftemp.name;
    G__srcfile[G__tempfilenum].hash=0;
    G__srcfile[G__tempfilenum].maxline=0;
    G__srcfile[G__tempfilenum].breakpoint = (char*)NULL;
    --G__tempfilenum;
    if(G__ifile.fp && G__ifile.filenum>=0) {
      fgetpos(G__ifile.fp,&pos);
    }
    store_ifile = G__ifile;
    G__ifile = ftemp;
    
    /**********************************************
     * interrpret signal handling during inner loop asm exec
     **********************************************/
#ifdef G__ASM
    G__STORE_ASMENV;
#endif
    store_var_type = G__var_type;
    
    G__var_type='p';

#ifdef G__EH_SIGNAL
    fpe = signal(SIGFPE,G__error_handle);
    segv = signal(SIGSEGV,G__error_handle);
#ifdef SIGILL
    ill = signal(SIGILL,G__error_handle);
#endif
#ifdef SIGEMT
    emt = signal(SIGEMT,G__error_handle);
#endif
#ifdef SIGBUS
    bus = signal(SIGBUS,G__error_handle);
#endif
#endif

  store_asm_inst = G__asm_inst;
  store_asm_stack = G__asm_stack;
  store_asm_name = G__asm_name;
  store_asm_name_p = G__asm_name_p;
  store_asm_param  = G__asm_param ;
  /* store_asm_exec  = G__asm_exec ; */
  store_asm_noverflow  = G__asm_noverflow ;
  store_asm_cp  = G__asm_cp ;
  store_asm_dt  = G__asm_dt ;
  store_asm_index  = G__asm_index ;

  G__asm_inst = asm_inst_g;
  G__asm_stack = asm_stack_g;
  G__asm_name = asm_name;
  G__asm_name_p = 0;
  /* G__asm_param ; */
  G__asm_exec = 0 ;

  /* execution */
  int brace_level = 0;
  buf = G__exec_statement(&brace_level);

  G__asm_inst = store_asm_inst;
  G__asm_stack = store_asm_stack;
  G__asm_name = store_asm_name;
  G__asm_name_p = store_asm_name_p;
  G__asm_param  = store_asm_param ;
  /* G__asm_exec  = store_asm_exec ; */
  G__asm_noverflow  = store_asm_noverflow ;
  G__asm_cp  = store_asm_cp ;
  G__asm_dt  = store_asm_dt ;
  G__asm_index  = store_asm_index ;

    /**********************************************
     * restore interrpret signal handling
     **********************************************/
#ifdef G__EH_SIGNAL
    signal(SIGFPE,fpe);
    signal(SIGSEGV,segv);
#ifdef SIGILL
    signal(SIGSEGV,ill);
#endif
#ifdef SIGEMT
    signal(SIGEMT,emt);
#endif
#ifdef SIGBUS
    signal(SIGBUS,bus);
#endif
#endif
    
#ifdef G__ASM
    G__RECOVER_ASMENV;
#endif
    G__var_type = store_var_type;
    
    /* print out result */
    G__ifile = store_ifile;
    if(G__ifile.fp && G__ifile.filenum>=0) {
      fsetpos(G__ifile.fp,&pos);
    }
    /* Following is intentionally commented out. This has to be selectively
     * done for 'x' and 'E' command  but not for { } command */
    /* G__security = G__srcfile[G__ifile.filenum].security; */
    if(file) fclose(ftemp.fp);
    ++G__tempfilenum;
    G__srcfile[G__tempfilenum].fp = (FILE*)NULL;
    G__srcfile[G__tempfilenum].filename=(char*)NULL;
    if(G__srcfile[G__tempfilenum].breakpoint)
      free(G__srcfile[G__tempfilenum].breakpoint);
    if(G__RETURN_IMMEDIATE>=G__return) G__return=G__RETURN_NON;
    G__no_exec=0;
    G__UnlockCriticalSection();

    return(buf);
  }
  else {
    G__fprinterr(G__serr,"Error: can not open file '%s'\n",file);
    G__UnlockCriticalSection();
    return(G__null);
  }
}

/****************************************************************
* G__exec_tempfile()
****************************************************************/
extern "C" G__value G__exec_tempfile(const char *file)
{
  return(G__exec_tempfile_core(file,(FILE*)NULL));
}



/**************************************************************************
* G__exec_text()
**************************************************************************/
extern "C" G__value G__exec_text(const char *unnamedmacro)
{
#ifndef G__TMPFILE
  char tname[L_tmpnam+10], sname[L_tmpnam+10];
#else
  char tname[G__MAXFILENAME], sname[G__MAXFILENAME];
#endif
  int nest=0,single_quote=0,double_quote=0;
  int ccomment=0,cppcomment=0;
  G__value buf;
  FILE *fp;
  int i,len;
  int addmparen=0;
  int addsemicolumn =0;
  int istmpnam=0;

  i=0;
  while(unnamedmacro[i] && isspace(unnamedmacro[i])) ++i;
  if(unnamedmacro[i]!='{') addmparen = 1;

  i = strlen(unnamedmacro)-1;
  while(i && isspace(unnamedmacro[i])) --i;
  if(unnamedmacro[i]=='}')       addsemicolumn = 0;
  else if(unnamedmacro[i]==';')  addsemicolumn = 0;
  else                   addsemicolumn = 1;

  len = (int)strlen(unnamedmacro);
  for(i=0;i<len;i++) {
    switch(unnamedmacro[i]) {
    case '(': 
    case '[': 
    case '{':
      if(!single_quote&&!double_quote&&!ccomment&&!cppcomment) ++nest; 
      break;
    case ')': 
    case ']': 
    case '}':
      if(!single_quote&&!double_quote&&!ccomment&&!cppcomment) --nest; 
      break;
    case '\'': 
      if(!double_quote&&!ccomment&&!cppcomment) single_quote^=1;
      break;
    case '"': 
      if(!single_quote&&!ccomment&&!cppcomment) double_quote^=1;
      break;
    case '/': 
      switch(unnamedmacro[i+1]) {
      case '/': cppcomment=1; ++i; break;
      case '*': ccomment=1; ++i; break;
      default: break;
      }
      break;
    case '\n': 
    case '\r': 
      if(cppcomment) {cppcomment=0;++i;}
      break;
    case '*': 
      if(ccomment && unnamedmacro[i+1]=='/') {ccomment=0;++i;}
      break;
    case '\\': ++i; break;
    default: break;
    }
  }
  if(nest!=0 || single_quote!=0 || double_quote!=0) {
    G__fprinterr(G__serr,"!!!Error in given statement!!! \"%s\"\n",unnamedmacro);
    return(G__null);
  }
  
  fp = tmpfile();
  if(!fp) {
    G__tmpnam(tname);  /* not used anymore 0 */
    fp = fopen(tname,"w");
    istmpnam=1;
  }
  if(!fp) return G__null;
  if(addmparen) fprintf(fp,"{\n");
  fprintf(fp,"%s",unnamedmacro);
  if(addsemicolumn) fprintf(fp,";");
  fprintf(fp,"\n");
  if(addmparen) fprintf(fp,"}\n");
  if(!istmpnam) fseek(fp,0L,SEEK_SET);
  else          fclose(fp);

  if(!istmpnam) {
    G__storerewindposition();
    buf=G__exec_tempfile_fp(fp);
    G__security_recover(G__serr);
    fclose(fp);
  }
  else {
    strcpy(sname,tname);
    G__storerewindposition();
    buf=G__exec_tempfile(sname);
    G__security_recover(G__serr);
    remove(sname);
  }

  return(buf);
}

#ifndef G__OLDIMPLEMENTATION1867
/**************************************************************************
* G__exec_text_str()
**************************************************************************/
extern "C" char* G__exec_text_str(const char *unnamedmacro,char *result)
{
  G__value buf = G__exec_text(unnamedmacro);
  G__valuemonitor(buf,result);
  return(result);
}
#endif

#ifndef G__OLDIMPLEMENTATION1546
/**************************************************************************
* G__load_text()
**************************************************************************/
extern "C" const char* G__load_text(const char *namedmacro)
{
  int fentry;
  const char* result = 0;
  FILE *fp;
  int istmpnam=0;
#ifndef G__TMPFILE
  static char tname[L_tmpnam+10];
#else
  static char tname[G__MAXFILENAME];
#endif

  fp = tmpfile();
  if(!fp) {
    G__tmpnam(tname);  /* not used anymore */
    strcat(tname,G__NAMEDMACROEXT);
    fp = fopen(tname,"w");
    if(!fp) return((char*)NULL);
    istmpnam=1;
  }
  fprintf(fp,"%s",namedmacro);
  fprintf(fp,"\n");

  if(!istmpnam) {
    fseek(fp,0L,SEEK_SET);
    fentry=G__loadfile_tmpfile(fp);
  }
  else {
    fclose(fp);
    fentry=G__loadfile(tname);
  }

  switch(fentry) {
  case G__LOADFILE_SUCCESS:
    if(!istmpnam) result = "(tmpfile)";
    else          result = tname;
    break;
  case G__LOADFILE_DUPLICATE:
  case G__LOADFILE_FAILURE:
  case G__LOADFILE_FATAL:
    if(!istmpnam) fclose(fp);
    else          remove(tname);
    result = (char*)NULL;
    break;
  default:
    result = G__srcfile[fentry-2].filename;
    break;
  }
  return(result);


}
#endif

/**************************************************************************
* G__beforelargestep()
**************************************************************************/
int Cint::Internal::G__beforelargestep(char *statement,int *piout,int *plargestep)
{
  G__break=0;
  G__setdebugcond();
  switch(G__pause()) {
  case 1: /* ignore */
    statement[0]='\0';
    *piout=0;
    break;
  case 3: /* largestep */
    if(strcmp(statement,"break")!=0 &&
       strcmp(statement,"continue")!=0 &&
       strcmp(statement,"return")!=0) {
      *plargestep=1;
      G__step=0;
      G__setdebugcond();
    }
    break;
  }
  return(G__return);
}

/**************************************************************************
* G__afterlargestep()
**************************************************************************/
void Cint::Internal::G__afterlargestep(int *plargestep)
{
        G__step = 1;
        *plargestep=0;
        G__setdebugcond();
}



/**************************************************************************
* G__EOFfgetc()
**************************************************************************/
void Cint::Internal::G__EOFfgetc()
{
  G__eof_count++;
  if(G__eof_count>10) {
    G__unexpectedEOF("G__fgetc()");
    if(G__steptrace||G__stepover||G__break||G__breaksignal||G__debug) 
      G__pause();
    G__exit(EXIT_FAILURE);
  }
  if(G__dispsource) {
    if((G__debug||G__break||G__step
        )&&
       ((G__prerun!=0)||(G__no_exec==0))&&
       (G__disp_mask==0)){
      G__fprinterr(G__serr,"EOF\n");
    }
    if(G__disp_mask>0) G__disp_mask-- ;
  }
  if(G__NOLINK==G__globalcomp && 
     NULL==G__srcfile[G__ifile.filenum].breakpoint) {
    G__srcfile[G__ifile.filenum].breakpoint
      =(char*)calloc((size_t)G__ifile.line_number,1);
    G__srcfile[G__ifile.filenum].maxline=G__ifile.line_number;
  }
}

/**************************************************************************
* G__DEBUGfgetc()
**************************************************************************/
void Cint::Internal::G__BREAKfgetc()
{
#ifdef G__ASM
  if(G__no_exec_compile) {
    G__abortbytecode();
  }
  else {
    G__break=1;
    G__setdebugcond();
    if(G__srcfile[G__ifile.filenum].breakpoint) {
      G__srcfile[G__ifile.filenum].breakpoint[G__ifile.line_number] 
        &= G__NOCONTUNTIL;
    }
  }
#else
  G__break=1;
  G__setdebugcond();
  G__breakpoint[G__ifile.filenum][G__ifile.line_number] &= G__NOCONTUNTIL;
#endif
}

/**************************************************************************
* G__DISPNfgetc()
**************************************************************************/
void Cint::Internal::G__DISPNfgetc()
{
  if((G__debug||G__break||G__step
      )&&
     ((G__prerun)||(G__no_exec==0))&&(G__disp_mask==0)){
    
    G__fprinterr(G__serr,"\n%-5d",G__ifile.line_number);
    
  }
  if(G__disp_mask>0) G__disp_mask-- ;
}

/**************************************************************************
* G__DISPfgetc()
**************************************************************************/
void Cint::Internal::G__DISPfgetc(int c)
{
  if((G__debug||G__break||G__step
      )&&
     ((G__prerun!=0)||(G__no_exec==0))&& (G__disp_mask==0)){
#ifndef G__OLDIMPLEMENTATION1485
    G__fputerr(c);
#else
    fputc(c,G__serr);
#endif
  }
  if(G__disp_mask>0) G__disp_mask-- ;
}


/**************************************************************************
* G__lockedvariable()
**************************************************************************/
void Cint::Internal::G__lockedvariable(const char *item)
{
  if(G__dispmsg>=G__DISPWARN) {
    G__fprinterr(G__serr,"Warning: Assignment to %s locked FILE:%s LINE:%d\n"
                 ,item
                 ,G__ifile.name,G__ifile.line_number);
  }
}


/**************************************************************************
* G__lock_variable()
**************************************************************************/
int Cint::Internal::G__lock_variable(const char *varname)
{
   int hash,ig15;
   ::Reflex::Member var;

   if(G__dispmsg>=G__DISPWARN) {
      G__fprinterr(G__serr,"Warning: lock variable obsolete feature");
      G__printlinenum();
   }

   G__hash(varname,hash,ig15);
   var = G__getvarentry(varname,hash,::Reflex::Scope::GlobalScope(),G__p_local);        

   if(var) {
      G__get_properties(var)->lock = true;
      G__fprinterr(G__serr,"Variable %s locked FILE:%s LINE:%d\n"
         ,varname,G__ifile.name,G__ifile.line_number);
      return(0);
   }
   else {
      G__fprinterr(G__serr,"Warining: failed locking %s FILE:%s LINE:%d\n"
         ,varname,G__ifile.name,G__ifile.line_number);
      return(1);
   }
}

/**************************************************************************
* G__unlock_variable()
**************************************************************************/
int Cint::Internal::G__unlock_variable(const char *varname)
{
   int hash,ig15;
   ::Reflex::Member var;

   if(G__dispmsg>=G__DISPWARN) {
      G__fprinterr(G__serr,"Warning: lock variable obsolete feature");
      G__printlinenum();
   }

   G__hash(varname,hash,ig15);
   var = G__getvarentry(varname,hash,::Reflex::Scope::GlobalScope(),G__p_local);

   if(var) {
      G__get_properties(var)->lock = false;
      G__fprinterr(G__serr,"Variable %s unlocked FILE:%s LINE:%d\n"
         ,varname,G__ifile.name,G__ifile.line_number);
      return(0);
   }
   else {
      G__fprinterr(G__serr,"Warining: failed unlocking %s FILE:%s LINE:%d\n"
         ,varname,G__ifile.name,G__ifile.line_number);
      return(1);
   }
}


/**************************************************************************
* G__setbreakpoint()
*
**************************************************************************/
extern "C" int G__setbreakpoint(char *breakline,char *breakfile)
{
  int ii;
  int line;
  
  if(isdigit(breakline[0])) {
    line=atoi(breakline);
    
    if(NULL==breakfile || '\0'==breakfile[0]) {
      G__fprinterr(G__serr," -b : break point on line %d every file\n",line);
      for(ii=0;ii<G__nfile;ii++) {
        if(G__srcfile[ii].breakpoint && G__srcfile[ii].maxline>line)
          G__srcfile[ii].breakpoint[line] |= G__BREAK;
      }
    }
    else {
      for(ii=0;ii<G__nfile;ii++) {
        if(G__srcfile[ii].filename&&
           G__matchfilename(ii,breakfile)
           ) break;
      }
      if(ii<G__nfile) {
        G__fprinterr(G__serr," -b : break point on line %d file %s\n"
                ,line,breakfile);
        if(G__srcfile[ii].breakpoint && G__srcfile[ii].maxline>line)
          G__srcfile[ii].breakpoint[line] |= G__BREAK;
      }
      else {
        G__fprinterr(G__serr,"File %s is not loaded\n",breakfile);
        return(1);
      }
    }

  }
  else {
    if(1<G__findfuncposition(breakline,&line,&ii)) {
      if(G__srcfile[ii].breakpoint) {
        G__fprinterr(G__serr," -b : break point on line %d file %s\n"
                ,line,G__srcfile[ii].filename);
        G__srcfile[ii].breakpoint[line] |= G__BREAK;
      }
      else {
        G__fprinterr(G__serr,"unable to put breakpoint in %s (included file)\n"
                ,breakline);
      }
    }
    else {
      G__fprinterr(G__serr,"function %s is not loaded\n",breakline);
      return(1);
    }
  }
  return(0);
}


/**************************************************************************
* G__interactivereturn()
*
**************************************************************************/
G__value Cint::Internal::G__interactivereturn()
{
  G__value result;
  result=G__null;
  if(G__interactive) {
    G__interactive=0;
    fprintf(G__sout,"!!!Return arbitrary value by 'return [value]' command");
    G__interactive_undefined=1;
    G__pause();
    G__interactive_undefined=0;
    G__interactive=1;
    result=G__interactivereturnvalue;
  }
  G__interactivereturnvalue=G__null;
  return(result);
}

/**************************************************************************
* G__set_tracemode()
*
**************************************************************************/
void Cint::Internal::G__set_tracemode(const char *name)
{
   int tagnum;
   int i=0;
   while(name[i]&&isspace(name[i])) i++;
   if('\0'==name[i]) {
      fprintf(G__sout,"trace all source code\n");
      G__istrace = 1;
      tagnum = -1;
   }
   else {
      G__StrBuf buffer(strlen(name)+1);
      char *p;
      char *s = buffer;
      strcpy(s,name+i);

      while(s) {
         p = strchr(s,' ');
         if(p) *p = '\0';
         tagnum = G__defined_tagname(s,0);
         if(-1!=tagnum) { 
            G__struct.istrace[tagnum] = 1;
            fprintf(G__sout,"trace %s object on\n",s);
         }
         if(p) s = p+1;
         else  s = p;
      }
   }
   G__setclassdebugcond(G__get_tagnum(G__memberfunc_tagnum),0);
}

/**************************************************************************
* G__del_tracemode()
*
**************************************************************************/
void Cint::Internal::G__del_tracemode(const char *name)
{
   int tagnum;
   int i=0;
   while(name[i]&&isspace(name[i])) i++;
   if('\0'==name[i]) {
      G__istrace = 0;
      tagnum = -1;
      fprintf(G__sout,"trace all source code off\n");
   }
   else {
      G__StrBuf buffer(strlen(name)+1);
      char *p;
      char *s = buffer;
      strcpy(s,name+i);

      while(s) {
         p = strchr(s,' ');
         if(p) *p = '\0';
         tagnum = G__defined_tagname(s,0);
         if(-1!=tagnum) {
            G__struct.istrace[tagnum] = 0;
            fprintf(G__sout,"trace %s object off\n",s);
         }
         if(p) s = p+1;
         else  s = p;
      }
   }
   G__setclassdebugcond(G__get_tagnum(G__memberfunc_tagnum),0);
}

/**************************************************************************
* G__set_classbreak()
*
**************************************************************************/
void Cint::Internal::G__set_classbreak(const char *name)
{
   int tagnum;
   int i=0;
   while(name[i]&&isspace(name[i])) i++;
   if(name[i]) {
      G__StrBuf buffer(strlen(name)+1);
      char *p;
      char *s = buffer;
      strcpy(s,name+i);

      while(s) {
         p = strchr(s,' ');
         if(p) *p = '\0';
         tagnum = G__defined_tagname(s,0);
         if(-1!=tagnum) {
            G__struct.isbreak[tagnum] = 1;
            fprintf(G__sout,"set break point at every %s member function\n",s);
         }
         if(p) s = p+1;
         else  s = p;
      }
   }
}

/**************************************************************************
* G__del_classbreak()
*
**************************************************************************/
void Cint::Internal::G__del_classbreak(const char *name)
{
   int tagnum;
   int i=0;
   while(name[i]&&isspace(name[i])) i++;
   if(name[i]) {
      G__StrBuf buffer(strlen(name)+1);
      char *p;
      char *s = buffer;
      strcpy(s,name+i);

      while(s) {
         p = strchr(s,' ');
         if(p) *p = '\0';
         tagnum = G__defined_tagname(s,0);
         if(-1!=tagnum) {
            G__struct.isbreak[tagnum] = 0;
            fprintf(G__sout,"delete break point at every %s member function\n",s);
         }
         if(p) s = p+1;
         else  s = p;
      }
   }
}

/**************************************************************************
* G__setclassdebugcond()
*
**************************************************************************/
void Cint::Internal::G__setclassdebugcond(int tagnum,int brkflag)
{
  if(-1==tagnum) {
    G__debug = G__istrace;
  }
  else {
    G__debug = G__struct.istrace[tagnum] | G__istrace;
    G__break |= G__struct.isbreak[tagnum];
  }
  G__dispsource=G__step+G__break+G__debug;
  if(G__dispsource==0) G__disp_mask=0;
  if(brkflag) {
    if((G__break||G__step)&&0==G__prerun) G__breaksignal=1;
    else                                  G__breaksignal=0;
  }
}

/****************************************************************
* G__exec_tempfile()
****************************************************************/
extern "C" G__value G__exec_tempfile_fp(FILE *fp)
{
   return(G__exec_tempfile_core((char*)NULL,fp));
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
