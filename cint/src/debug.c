/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file debug.c
 ************************************************************************
 * Description:
 *  Debugger capability
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto (MXJ02154@niftyserve.or.jp)
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

  static int filenum = G__MAXFILE-1;
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
    ftemp.filenum = filenum;
    G__srcfile[filenum].fp = ftemp.fp;
    G__srcfile[filenum].filename=ftemp.name;
    G__srcfile[filenum].hash=0;
    G__srcfile[filenum].maxline=0;
    G__srcfile[filenum].breakpoint = (char*)NULL;
    --filenum;
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
    ++filenum;
    G__srcfile[filenum].fp = (FILE*)NULL;
    G__srcfile[filenum].filename=(char*)NULL;
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
    fprintf(G__serr,"Error: file %s can not open\n",file);
#ifndef G__OLDIMPLEMENTATION1035
    G__UnlockCriticalSection();
#endif
    return(G__null);
  }
}


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
      fprintf(G__serr,"EOF\n");
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
    
    fprintf(G__serr,"\n%-5d",G__ifile.line_number);
    
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
    fputc(c,G__serr);
  }
  if(G__disp_mask>0) G__disp_mask-- ;
}


/**************************************************************************
* G__lockedvariable()
**************************************************************************/
void G__lockedvariable(item)
char *item;
{
  fprintf(G__serr,"Warning: Assignment to %s locked FILE:%s LINE:%d\n"
	  ,item
	  ,G__ifile.name,G__ifile.line_number);
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
  fprintf(G__serr,"Warning: lock variable obsolete feature");
  G__printlinenum();
#endif
  
  G__hash(varname,hash,ig15)
  var = G__getvarentry(varname,hash,&ig15,&G__global,G__p_local);	
  
  if(var) {
    var->constvar[ig15] |= G__LOCKVAR;
    fprintf(G__serr,"Variable %s locked FILE:%s LINE:%d\n"
	    ,varname,G__ifile.name,G__ifile.line_number);
    return(0);
  }
  else {
    fprintf(G__serr,"Warining: failed locking %s FILE:%s LINE:%d\n"
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
  fprintf(G__serr,"Warning: lock variable obsolete feature");
  G__printlinenum();
#endif
  
  G__hash(varname,hash,ig15)
    var = G__getvarentry(varname,hash,&ig15,&G__global,G__p_local);	
  
  if(var) {
    var->constvar[ig15] &= ~G__LOCKVAR;
    fprintf(G__serr,"Variable %s unlocked FILE:%s LINE:%d\n"
	    ,varname,G__ifile.name,G__ifile.line_number);
    return(0);
  }
  else {
    fprintf(G__serr,"Warining: failed unlocking %s FILE:%s LINE:%d\n"
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
      fprintf(G__serr," -b : break point on line %d every file\n",line);
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
	fprintf(G__serr," -b : break point on line %d file %s\n"
		,line,breakfile);
	if(G__srcfile[ii].breakpoint && G__srcfile[ii].maxline>line)
	  G__srcfile[ii].breakpoint[line] |= G__BREAK;
      }
      else {
	fprintf(G__serr,"File %s not loaded\n",breakfile);
	return(1);
      }
    }

  }
  else {
    if(1<G__findfuncposition(breakline,&line,&ii)) {
      if(G__srcfile[ii].breakpoint) {
	fprintf(G__serr," -b : break point on line %d file %s\n"
		,line,G__srcfile[ii].filename);
	G__srcfile[ii].breakpoint[line] |= G__BREAK;
      }
      else {
	fprintf(G__serr,"function %s in include file, can not put breakpoint\n"
		,breakline);
      }
    }
    else {
      fprintf(G__serr,"function %s not loaded\n",breakline);
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
