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
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#include "common.h"


/*********************************************************
* G__setdebugcond()
*
*  Pre-set trace/debug condition for speed up
*********************************************************/

void G__setdebugcond()
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
int G__findposition(string,view,pline,pfnum)
char *string;
struct G__input_file view;
int *pline,*pfnum;
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
int G__findfuncposition(func,pline,pfnum)
char *func;
int *pline,*pfnum;
{
  char funcname[G__ONELINE];
  char scope[G__ONELINE];
  char temp[G__ONELINE];
  char *pc;
  int temp1;
  int tagnum;
  struct G__ifunc_table *ifunc;

  strcpy(funcname,func);

  pc=strstr(funcname,"::");

  /* get appropreate scope */
  if(pc) {
    *pc = '\0';
    strcpy(scope,funcname);
    strcpy(temp,pc+2);
    strcpy(funcname,temp);
    tagnum = G__defined_tagname(scope,0);
    if('\0'==funcname[0] && -1!=tagnum) {
      /* display class declaration */
      *pline = G__struct.line_number[tagnum];
      *pfnum = G__struct.filenum[tagnum];
      return(2);
    }
    else {
      /* class scope A::func , global scope ::func */
      if(-1==tagnum) ifunc = &G__ifunc;  /* global scope, ::func */
      else {
	G__incsetup_memfunc(tagnum);
	ifunc = G__struct.memfunc[tagnum]; /* specific class */
      }
    }
  }
  else {
    /* global scope */
    ifunc = &G__ifunc;
  }

  while(ifunc) {
    temp1=0;
    while(temp1<ifunc->allifunc) {
      if(strcmp(ifunc->funcname[temp1],funcname)==0) {
	*pline = ifunc->pentry[temp1]->line_number;
	*pfnum = ifunc->pentry[temp1]->filenum;
	return(2);
      }
      ++temp1;
    }
    ifunc=ifunc->next;
  }
  return(0);
}

#ifndef G__OLDIMPLEMENTATION553
/****************************************************************
* G__display_proto()
****************************************************************/
int G__display_proto(fp,func)
FILE *fp;
char *func;
{
  char funcname[G__LONGLINE];
  char scope[G__LONGLINE];
  char temp[G__LONGLINE];
  char *pc;
  /* int temp1; */
  int tagnum;
  struct G__ifunc_table *ifunc;
  int i=0;

  while(isspace(func[i])) ++i;
  strcpy(funcname,func+i);

  pc=strstr(funcname,"::");

  /* get appropreate scope */
  if(pc) {
    *pc = '\0';
    strcpy(scope,funcname);
    strcpy(temp,pc+2);
    strcpy(funcname,temp);
#ifndef G__OLDIMPLEMENTATION1030
    if(0==scope[0]) tagnum = -1;
    else tagnum = G__defined_tagname(scope,0);
#else
    tagnum = G__defined_tagname(scope,0);
#endif
    /* class scope A::func , global scope ::func */
    if(-1==tagnum) ifunc = &G__ifunc;  /* global scope, ::func */
    else {
      G__incsetup_memfunc(tagnum);
      ifunc = G__struct.memfunc[tagnum]; /* specific class */
    }
  }
  else {
    /* global scope */
    tagnum = -1;
    ifunc = &G__ifunc;
  }

  i=strlen(funcname);
  while(i&&(isspace(funcname[i-1])||'('==funcname[i-1])) funcname[--i]='\0';
  if(i) {
    if(G__listfunc(fp,G__PUBLIC_PROTECTED_PRIVATE,funcname,ifunc)) return(1);
  }
  else  {
    if(G__listfunc(fp,G__PUBLIC_PROTECTED_PRIVATE,(char*)NULL,ifunc))return(1);
  }
#ifndef G__OLDIMPLEMENTATION1079
  if(-1!=tagnum) {
    int i1;
    struct G__inheritance *baseclass = G__struct.baseclass[tagnum];
    for(i1=0;i1<baseclass->basen;i1++) {
      ifunc = G__struct.memfunc[baseclass->basetagnum[i1]];
      if(i) {
	if(G__listfunc(fp,G__PUBLIC_PROTECTED_PRIVATE,funcname,ifunc)) 
	  return(1);
      }
      else  {
	if(G__listfunc(fp,G__PUBLIC_PROTECTED_PRIVATE,(char*)NULL,ifunc))
	  return(1);
      }
    }
  }
#endif
  return(0);
}
#endif

#ifndef G__OLDIMPLEMENTATION1601
static int G__tempfilenum = G__MAXFILE-1;

/****************************************************************
* G__gettempfilenum()
****************************************************************/
int G__gettempfilenum()
{
  return(G__tempfilenum);
}
#endif


#ifndef G__OLDIMPLEMENTATION1794
/****************************************************************
* G__exec_tempfile_core()
****************************************************************/
G__value G__exec_tempfile_core(file,fp)
char *file;
FILE *fp;
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

#ifndef G__OLDIMPLEMENTATION1247
  long asm_inst_g[G__MAXINST]; /* p-code instruction buffer */
  G__value asm_stack_g[G__MAXSTACK]; /* data stack */
  char asm_name[G__ASM_FUNCNAMEBUF];

  long *store_asm_inst;
  G__value *store_asm_stack;
  char *store_asm_name;
  int store_asm_name_p;
  struct G__param *store_asm_param;
  /* int store_asm_exec; */
  int store_asm_noverflow;
  int store_asm_cp;
  int store_asm_dt;
  int store_asm_index; /* maybe unneccessary */
#endif

  int len;

#ifdef G__OLDIMPLEMENTATION1601
  static int filenum = G__MAXFILE-1;
#endif
  fpos_t pos;
  char store_var_type;
  struct G__input_file ftemp,store_ifile;
  G__value buf = G__null;
#ifdef G__ASM
  G__ALLOC_ASMENV;
#endif

#ifndef G__OLDIMPLEMENTATION1035
  G__LockCriticalSection();
#endif

  /*************************************************
  * delete space chars at the end of filename
  *************************************************/
  if(file) {
    len = strlen(file);
    while(len>1&&isspace(file[len-1])) {
      file[--len]='\0';
    }
  
#ifndef G__WIN32
    ftemp.fp = fopen(file,"r");
#else
    ftemp.fp = fopen(file,"rb");
#endif
  }
  else {
    fseek(fp,0L,SEEK_SET);
    ftemp.fp = fp;
  }

  if(ftemp.fp) {
    ftemp.line_number = 1;
    if(file) sprintf(ftemp.name,file);
    else     strcpy(ftemp.name,"(tmpfile)");
#ifndef G__OLDIMPLEMENTATION1601
    ftemp.filenum = G__tempfilenum;
    G__srcfile[G__tempfilenum].fp = ftemp.fp;
    G__srcfile[G__tempfilenum].filename=ftemp.name;
    G__srcfile[G__tempfilenum].hash=0;
    G__srcfile[G__tempfilenum].maxline=0;
    G__srcfile[G__tempfilenum].breakpoint = (char*)NULL;
    --G__tempfilenum;
#else
    ftemp.filenum = filenum;
    G__srcfile[filenum].fp = ftemp.fp;
    G__srcfile[filenum].filename=ftemp.name;
    G__srcfile[filenum].hash=0;
    G__srcfile[filenum].maxline=0;
    G__srcfile[filenum].breakpoint = (char*)NULL;
    --filenum;
#endif
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

#ifndef G__OLDIMPLEMENTATION1247
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
#endif

  /* execution */
  buf = G__exec_statement();

#ifndef G__OLDIMPLEMENTATION1247
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
#endif

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
#ifndef G__OLDIMPLEMENTATION1601
    ++G__tempfilenum;
    G__srcfile[G__tempfilenum].fp = (FILE*)NULL;
    G__srcfile[G__tempfilenum].filename=(char*)NULL;
#ifndef G__OLDIMPLEMENTATION1899
    if(G__srcfile[G__tempfilenum].breakpoint)
      free(G__srcfile[G__tempfilenum].breakpoint);
#endif
#else
    ++filenum;
    G__srcfile[filenum].fp = (FILE*)NULL;
    G__srcfile[filenum].filename=(char*)NULL;
#endif
#ifndef G__OLDIMPLEMENTATION630
    if(G__RETURN_IMMEDIATE>=G__return) G__return=G__RETURN_NON;
#else
    if(G__RETURN_NORMAL==G__return) G__return=G__RETURN_NON;
#endif
    G__no_exec=0;
#ifndef G__OLDIMPLEMENTATION1035
    G__UnlockCriticalSection();
#endif

    return(buf);
  }
  else {
    G__fprinterr(G__serr,"Error: can not open file '%s'\n",file);
#ifndef G__OLDIMPLEMENTATION1035
    G__UnlockCriticalSection();
#endif
    return(G__null);
  }
}

/****************************************************************
* G__exec_tempfile()
****************************************************************/
G__value G__exec_tempfile_fp(fp)
FILE *fp;
{
  return(G__exec_tempfile_core((char*)NULL,fp));
}

/****************************************************************
* G__exec_tempfile()
****************************************************************/
G__value G__exec_tempfile(file)
char *file;
{
  return(G__exec_tempfile_core(file,(FILE*)NULL));
}


#else /* 1794 */

/****************************************************************
* G__exec_tempfile()
****************************************************************/
G__value G__exec_tempfile(file)
char *file;
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

#ifndef G__OLDIMPLEMENTATION1247
  long asm_inst_g[G__MAXINST]; /* p-code instruction buffer */
  G__value asm_stack_g[G__MAXSTACK]; /* data stack */
  char asm_name[G__ASM_FUNCNAMEBUF];

  long *store_asm_inst;
  G__value *store_asm_stack;
  char *store_asm_name;
  int store_asm_name_p;
  struct G__param *store_asm_param;
  /* int store_asm_exec; */
  int store_asm_noverflow;
  int store_asm_cp;
  int store_asm_dt;
  int store_asm_index; /* maybe unneccessary */
#endif

  int len;

#ifdef G__OLDIMPLEMENTATION1601
  static int filenum = G__MAXFILE-1;
#endif
  fpos_t pos;
  char store_var_type;
  struct G__input_file ftemp,store_ifile;
  G__value buf;
#ifdef G__ASM
  G__ALLOC_ASMENV;
#endif

#ifndef G__OLDIMPLEMENTATION1035
  G__LockCriticalSection();
#endif

  /*************************************************
  * delete space chars at the end of filename
  *************************************************/
  len = strlen(file);
  while(len>1&&isspace(file[len-1])) {
    file[--len]='\0';
  }
  
#ifndef G__WIN32
  ftemp.fp = fopen(file,"r");
#else
  ftemp.fp = fopen(file,"rb");
#endif
  if(ftemp.fp) {
    ftemp.line_number = 1;
    sprintf(ftemp.name,file);
#ifndef G__OLDIMPLEMENTATION1601
    ftemp.filenum = G__tempfilenum;
    G__srcfile[G__tempfilenum].fp = ftemp.fp;
    G__srcfile[G__tempfilenum].filename=ftemp.name;
    G__srcfile[G__tempfilenum].hash=0;
    G__srcfile[G__tempfilenum].maxline=0;
    G__srcfile[G__tempfilenum].breakpoint = (char*)NULL;
    --G__tempfilenum;
#else
    ftemp.filenum = filenum;
    G__srcfile[filenum].fp = ftemp.fp;
    G__srcfile[filenum].filename=ftemp.name;
    G__srcfile[filenum].hash=0;
    G__srcfile[filenum].maxline=0;
    G__srcfile[filenum].breakpoint = (char*)NULL;
    --filenum;
#endif
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

#ifndef G__OLDIMPLEMENTATION1247
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
#endif

    /* execution */
    buf = G__exec_statement();

#ifndef G__OLDIMPLEMENTATION1247
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
#endif

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
    fclose(ftemp.fp);
#ifndef G__OLDIMPLEMENTATION1601
    ++G__tempfilenum;
    G__srcfile[G__tempfilenum].fp = (FILE*)NULL;
    G__srcfile[G__tempfilenum].filename=(char*)NULL;
#else
    ++filenum;
    G__srcfile[filenum].fp = (FILE*)NULL;
    G__srcfile[filenum].filename=(char*)NULL;
#endif
#ifndef G__OLDIMPLEMENTATION630
    if(G__RETURN_IMMEDIATE>=G__return) G__return=G__RETURN_NON;
#else
    if(G__RETURN_NORMAL==G__return) G__return=G__RETURN_NON;
#endif
    G__no_exec=0;
#ifndef G__OLDIMPLEMENTATION1035
    G__UnlockCriticalSection();
#endif

    return(buf);
  }
  else {
    G__fprinterr(G__serr,"Error: can not open file '%s'\n",file);
#ifndef G__OLDIMPLEMENTATION1035
    G__UnlockCriticalSection();
#endif
    return(G__null);
  }
}

#endif /* 1794 */

#ifndef G__OLDIMPLEMENTATION1348
/**************************************************************************
* G__exec_text()
**************************************************************************/
G__value G__exec_text(unnamedmacro)
char *unnamedmacro;
{
#if !defined(G__OLDIMPLEMENTATION2092)
#ifndef G__TMPFILE
  char tname[L_tmpnam+10], sname[L_tmpnam+10];
#else
  char tname[G__MAXFILENAME], sname[G__MAXFILENAME];
#endif
#elif !defined(G__OLDIMPLEMENTATION1794)
#else
#ifndef G__TMPFILE
  char tname[L_tmpnam+10], sname[L_tmpnam+10];
#else
  char tname[G__MAXFILENAME], sname[G__MAXFILENAME];
#endif
#endif /* 1794 */
  int nest=0,single_quote=0,double_quote=0;
#ifndef G__OLDIMPLEMENTATION1783
  int ccomment=0,cppcomment=0;
#endif
  G__value buf;
  FILE *fp;
  int i,len;
  int addmparen=0;
  int addsemicolumn =0;
#ifndef G__OLDIMPLEMENTATION2092
  int istmpnam=0;
#endif

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
#ifndef G__OLDIMPLEMENTATION1783
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
#else
    case '(': case '[': case '{':
      if(!single_quote && !double_quote) ++nest; break;
    case ')': case ']': case '}':
      if(!single_quote && !double_quote) --nest; break;
    case '\'': if(!double_quote) single_quote ^= 1; break;
    case '"': if(!single_quote) double_quote ^= 1; break;
#endif
    case '\\': ++i; break;
    default: break;
    }
  }
  if(nest!=0 || single_quote!=0 || double_quote!=0) {
    G__fprinterr(G__serr,"!!!Error in given statement!!! \"%s\"\n",unnamedmacro);
    return(G__null);
  }
  
#if !defined(G__OLDIMPLEMENTATION2092)
  fp = tmpfile();
  if(!fp) {
    G__tmpnam(tname);  /* not used anymore 0 */
    fp = fopen(tname,"w");
    istmpnam=1;
  }
#elif !defined(G__OLDIMPLEMENTATION1794)
  fp = tmpfile();
#else
  G__tmpnam(tname);  /* not used anymore 0 */
  fp = fopen(tname,"w");
#endif
  if(!fp) return G__null;
  if(addmparen) fprintf(fp,"{\n");
  fprintf(fp,"%s",unnamedmacro);
  if(addsemicolumn) fprintf(fp,";");
  fprintf(fp,"\n");
  if(addmparen) fprintf(fp,"}\n");
#if !defined(G__OLDIMPLEMENTATION2092)
  if(!istmpnam) fseek(fp,0L,SEEK_SET);
  else          fclose(fp);
#elif !defined(G__OLDIMPLEMENTATION1794)
  fseek(fp,0L,SEEK_SET);
#else
  fclose(fp);
#endif

#if !defined(G__OLDIMPLEMENTATION2092)
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
#elif !defined(G__OLDIMPLEMENTATION1794)
  G__storerewindposition();
  buf=G__exec_tempfile_fp(fp);
  G__security_recover(G__serr);
  fclose(fp);
#else
  strcpy(sname,tname);
  G__storerewindposition();
  buf=G__exec_tempfile(sname);
  G__security_recover(G__serr);
  remove(sname);
#endif

  return(buf);
}
#endif

#ifndef G__OLDIMPLEMENTATION1867
/**************************************************************************
* G__exec_text_str()
**************************************************************************/
char* G__exec_text_str(unnamedmacro,result)
char *unnamedmacro;
char *result;
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
char* G__load_text(namedmacro)
char *namedmacro;
{
#ifndef G__OLDIMPLEMENTATION1919
  int fentry;
  char* result = (char*)NULL;
  FILE *fp;
#ifndef G__OLDIMPLEMENTATION2092
  int istmpnam=0;
#ifndef G__TMPFILE
  static char tname[L_tmpnam+10];
#else
  static char tname[G__MAXFILENAME];
#endif
#endif

  fp = tmpfile();
#ifndef G__OLDIMPLEMENTATION2092
  if(!fp) {
    G__tmpnam(tname);  /* not used anymore */
    strcat(tname,G__NAMEDMACROEXT);
    fp = fopen(tname,"w");
    if(!fp) return((char*)NULL);
    istmpnam=1;
  }
#else
  if(!fp) return((char*)NULL);
#endif
  fprintf(fp,"%s",namedmacro);
  fprintf(fp,"\n");

#ifndef G__OLDIMPLEMENTATION2092
  if(!istmpnam) {
    fseek(fp,0L,SEEK_SET);
    fentry=G__loadfile_tmpfile(fp);
  }
  else {
    fclose(fp);
    fentry=G__loadfile(tname);
  }
#else
  fseek(fp,0L,SEEK_SET);
  fentry=G__loadfile_tmpfile(fp);
#endif

  switch(fentry) {
  case G__LOADFILE_SUCCESS:
#ifndef G__OLDIMPLEMENTATION2092
    if(!istmpnam) result = "(tmpfile)";
    else          result = tname;
#else
    result = "(tmpfile)";
#endif
    break;
  case G__LOADFILE_DUPLICATE:
  case G__LOADFILE_FAILURE:
  case G__LOADFILE_FATAL:
#ifndef G__OLDIMPLEMENTATION2092
    if(!istmpnam) fclose(fp);
    else          remove(tname);
#else
    fclose(fp);
#endif
    result = (char*)NULL;
    break;
  default:
    result = G__srcfile[fentry-2].filename;
    break;
  }
  return(result);

#else

  char* result = (char*)NULL;
#ifndef G__TMPFILE
  static char tname[L_tmpnam+10];
#else
  static char tname[G__MAXFILENAME];
#endif
  FILE *fp;
  
  G__tmpnam(tname);  /* not used anymore */
  strcat(tname,G__NAMEDMACROEXT);
  fp = fopen(tname,"w");
  if(!fp) return((char*)NULL);
  fprintf(fp,"%s",namedmacro);
  fprintf(fp,"\n");
  fclose(fp);

  switch(G__loadfile(tname)) {
  case G__LOADFILE_SUCCESS:
    result = tname;
    break;
  case G__LOADFILE_DUPLICATE:
  case G__LOADFILE_FAILURE:
  case G__LOADFILE_FATAL:
    remove(tname);
    result = (char*)NULL;
    break;
  }
  return(result);
#endif

}
#endif

/**************************************************************************
* G__beforelargestep()
**************************************************************************/
int G__beforelargestep(statement,piout,plargestep)
char *statement;
int *piout;
int *plargestep;
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
void G__afterlargestep(plargestep)
int *plargestep;
{
	G__step = 1;
	*plargestep=0;
	G__setdebugcond();
}



/**************************************************************************
* G__EOFfgetc()
**************************************************************************/
void G__EOFfgetc()
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
#ifdef G__OLDIMPLEMENTATION473
	||strcmp(G__breakfile,G__ifile.name)==0||strcmp(G__breakfile,"")==0
#endif
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
void G__BREAKfgetc()
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
void G__DISPNfgetc()
{
  if((G__debug||G__break||G__step
#ifdef G__OLDIMPLEMENTATION473
      ||strcmp(G__breakfile,G__ifile.name)==0||strcmp(G__breakfile,"")==0
#endif
      )&&
     ((G__prerun)||(G__no_exec==0))&&(G__disp_mask==0)){
    
    G__fprinterr(G__serr,"\n%-5d",G__ifile.line_number);
    
  }
  if(G__disp_mask>0) G__disp_mask-- ;
}

/**************************************************************************
* G__DISPfgetc()
**************************************************************************/
void G__DISPfgetc(c)
int c;
{
  if((G__debug||G__break||G__step
#ifdef G__OLDIMPLEMENTATION473
      ||strcmp(G__breakfile,G__ifile.name)==0||strcmp(G__breakfile,"")==0
#endif
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
void G__lockedvariable(item)
char *item;
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
int G__lock_variable(varname)
char *varname;
{
  int hash,ig15;
  struct G__var_array *var;

#ifndef G__OLDIMPLEMENTATION1119
  if(G__dispmsg>=G__DISPWARN) {
    G__fprinterr(G__serr,"Warning: lock variable obsolete feature");
    G__printlinenum();
  }
#endif
  
  G__hash(varname,hash,ig15)
  var = G__getvarentry(varname,hash,&ig15,&G__global,G__p_local);	
  
  if(var) {
    var->constvar[ig15] |= G__LOCKVAR;
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
int G__unlock_variable(varname)
char *varname;
{
  int hash,ig15;
  struct G__var_array *var;

#ifndef G__OLDIMPLEMENTATION1119
  if(G__dispmsg>=G__DISPWARN) {
    G__fprinterr(G__serr,"Warning: lock variable obsolete feature");
    G__printlinenum();
  }
#endif
  
  G__hash(varname,hash,ig15)
    var = G__getvarentry(varname,hash,&ig15,&G__global,G__p_local);	
  
  if(var) {
    var->constvar[ig15] &= ~G__LOCKVAR;
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
int G__setbreakpoint(breakline,breakfile)
char *breakline,*breakfile;
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
#ifndef G__OLDIMPLEMENTATION1196
	   G__matchfilename(ii,breakfile)
#else
	   strcmp(breakfile,G__srcfile[ii].filename)==0
#endif
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
G__value G__interactivereturn()
{
  G__value result;
  result=G__null;
  if(G__interactive) {
    G__interactive=0;
    fprintf(G__sout,"!!!Return arbitrary value by 'return [value]' command");
#ifndef G__OLDIMPLEMENTATION630
    G__interactive_undefined=1;
    G__pause();
    G__interactive_undefined=0;
#else
    G__pause();
#endif
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
void G__set_tracemode(name)
char *name;
{
  int tagnum;
  int i=0;
  char *p,*s;
  while(name[i]&&isspace(name[i])) i++;
  if('\0'==name[i]) {
    fprintf(G__sout,"trace all source code\n");
    G__istrace = 1;
    tagnum = -1;
  }
  else {
    s = name+i;
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
  G__setclassdebugcond(G__memberfunc_tagnum,0);
}

/**************************************************************************
* G__del_tracemode()
*
**************************************************************************/
void G__del_tracemode(name)
char *name;
{
  int tagnum;
  int i=0;
  char *p,*s;
  while(name[i]&&isspace(name[i])) i++;
  if('\0'==name[i]) {
    G__istrace = 0;
    tagnum = -1;
    fprintf(G__sout,"trace all source code off\n");
  }
  else {
    s = name+i;
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
  G__setclassdebugcond(G__memberfunc_tagnum,0);
}

/**************************************************************************
* G__set_classbreak()
*
**************************************************************************/
void G__set_classbreak(name)
char *name;
{
  int tagnum;
  int i=0;
  char *p,*s;
  while(name[i]&&isspace(name[i])) i++;
  if(name[i]) {
    s = name+i;
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
void G__del_classbreak(name)
char *name;
{
  int tagnum;
  int i=0;
  char *p,*s;
  while(name[i]&&isspace(name[i])) i++;
  if(name[i]) {
    s = name+i;
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
void G__setclassdebugcond(tagnum,brkflag)
int tagnum;
int brkflag;
{
#ifndef G__OLDIMPLEMENTATION2135
  if(G__cintv6) return;
#endif
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
