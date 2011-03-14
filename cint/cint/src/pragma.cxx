/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file pragma.c
 ************************************************************************
 * Description:
 *  #pragma support
 ************************************************************************
 * Copyright(c) 1995~2002  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"

/**************************************************************************
* G__addpragma()
**************************************************************************/
G__AppPragma::G__AppPragma(char *comname, G__AppPragma_func_t in_p2f) : 
   name(comname), p2f(in_p2f), next(0)
{
   // Normal constructor.
}

extern "C" void G__addpragma(char *comname, G__AppPragma_func_t p2f)
{
   struct G__AppPragma *paddpragma;
   
   if(G__paddpragma) {
      paddpragma = G__paddpragma;
      while(paddpragma->next) paddpragma = paddpragma->next;
      paddpragma->next = new G__AppPragma( comname, p2f); 
      paddpragma = paddpragma->next;
   }
   else {
      G__paddpragma = new G__AppPragma( comname, p2f); 
      paddpragma=G__paddpragma;
   }
}

/**************************************************************************
* G__execpragma()
**************************************************************************/
int G__execpragma(const char *comname,char *args)
{
  struct G__AppPragma *paddpragma;
  void (*p2f) G__P((char*));

  paddpragma=G__paddpragma;
  while(paddpragma) {
    if(strcmp(paddpragma->name,comname)==0) {
      p2f = (void (*)(char *))paddpragma->p2f;
      if(p2f) (*p2f)(args);
      else    G__fprinterr(G__serr,"p2f null\n");
      return(0);
    }
    paddpragma=paddpragma->next;
  }
  return(0);
}

/**************************************************************************
* G__freepragma()
**************************************************************************/
G__AppPragma::~G__AppPragma()
{
   delete next;
}
void G__freepragma(G__AppPragma *paddpragma)
{
   delete paddpragma;
}

extern "C" {

/**************************************************************************
* G__read_setmode()
**************************************************************************/
int G__read_setmode(int *pmode)
{
  int c;
  G__FastAllocString command(G__ONELINE);
  c=G__fgetstream(command, 0, ";\n\r");
  if(strcmp(command,"on")==0||'\0'==command[0]) *pmode=1;
  else if(strcmp(command,"ON")==0)              *pmode=1;
  else if(strcmp(command,"off")==0)             *pmode=0;
  else if(strcmp(command,"OFF")==0)             *pmode=0;
#ifdef G__NEVER
  else if(strcmp(command,"always")==0)          *pmode=2;
  else if(strcmp(command,"ALWAYS")==0)          *pmode=2;
  else if(strcmp(command,"all")==0)             *pmode=3;
  else if(strcmp(command,"ALL")==0)             *pmode=3;
#endif
  else                              *pmode=(int)G__int(G__getexpr(command));
  return(c);
}

/**************************************************************************
* G__addpreprocessfile
**************************************************************************/
static int G__addpreprocessfile()
{
  int c;
  struct G__Preprocessfilekey *pkey;
  G__FastAllocString keystring(G__ONELINE);

  /* Get the key string for preprocessed header file group */
  c=G__fgetstream(keystring, 0, ";\n\r");

  /* Get to the end of the preprocessed file key list */
  pkey = &G__preprocessfilekey;
  while(pkey->next) pkey=pkey->next;

  /* Add the list */
  pkey->keystring = (char*)malloc(strlen(keystring)+1);
  strcpy(pkey->keystring,keystring); // Okay we allocated enough space
  pkey->next
    =(struct G__Preprocessfilekey*)malloc(sizeof(struct G__Preprocessfilekey));
  pkey->next->next=(struct G__Preprocessfilekey*)NULL;
  pkey->next->keystring=(char*)NULL;

  return(c);
}

extern int G__rootCcomment; /* used and defined in sizeof.c */

/**************************************************************************
* G__do_not_include
**************************************************************************/
static void G__do_not_include()
{
  int c;
  G__FastAllocString fnameorig(G__ONELINE);
  char *fname;
  size_t len;
  int hash;
  int i;

  /* if(!G__IsInMacro()) return; */

  /* Get the key string for preprocessed header file group */
  c=G__fgetstream(fnameorig, 0, ";\n\r");

  switch(fnameorig[0]) {
  case '\'':
  case '"':
  case '<':
    fname = fnameorig+1;
    break;
  default:
    fname = fnameorig;
    break;
  }
  len = strlen(fname);
  if(len) {
    switch(fname[len-1]) {
    case '\'':
    case '"':
    case '>':
      fname[len-1] = 0;
      break;
    }
  }

  G__hash(fname,hash,i);

  for(i=0;i<G__nfile;i++) {
    if((hash==G__srcfile[i].hash&&strcmp(G__srcfile[i].filename,fname)==0)){
      return;
    }
  }

  G__srcfile[G__nfile].hash = hash;
  G__srcfile[G__nfile].filename = (char*)malloc(strlen(fname)+1);
  strcpy(G__srcfile[G__nfile].filename,fname); // Okay we allocated enough space
  G__srcfile[G__nfile].included_from = -1;

  ++G__nfile;
  ++G__srcfile_serial;

  return;
}

#ifdef G__OLDIMPLEMENTATION1781_YET
/**************************************************************************
* G__force_bytecode_compilation();
**************************************************************************/
void G__force_bytecode_compilation()
{
}
#endif

/**************************************************************************
* G__pragma()
**************************************************************************/
int G__pragma()
{
  G__FastAllocString command(G__ONELINE);
  int c;
  int store_no_exec_compile;
  /* static int store_asm_loopcompile=4; */

  c = G__fgetname(command, 0, ";\n\r");


  if(strcmp(command,"include")==0) {
    G__include_file();
    c='\n';
  }
  else if(strcmp(command,"include_noerr")==0) {
#ifndef __CINT__
    G__ispragmainclude = 1;
    G__include_file();
    G__ispragmainclude = 0;
#else
    G__fignoreline();
#endif
    c='\n';
  }
  else if(strcmp(command,"permanent_link")==0) {
    c=G__fgetstream(command, 0, ";\n\r");
    G__ispragmainclude = 1;
    G__loadsystemfile(command);
    G__ispragmainclude = 0;
    c='\n';
  }
  else if(strcmp(command,"includepath")==0) {
    c=G__fgetstream(command, 0, ";\n\r");
    G__add_ipath(command);
  }
  else if(strcmp(command,"preprocessor")==0) {
    /* #pragma preprocessor on/off */
    c=G__read_setmode(&G__include_cpp);
  }
  else if(strcmp(command,"preprocess")==0) {
    /* #pragma preprocess [String]
     * #pragma preprocess X11/X     */
    c=G__addpreprocessfile();
  }

  else if(strcmp(command,"Ccomment")==0) {
    /* ROOT C comment on/off */
    c=G__read_setmode(&G__rootCcomment);
  }

  else if(strcmp(command,"setstdio")==0) {
    G__set_stdio();
  }

  else if(strcmp(command,"setstream")==0) {
    struct G__input_file store_ifile = G__ifile;
    G__ifile.filenum = -1;
    G__ifile.line_number = -1;
    G__cpp_setupG__stream();
    G__ifile = store_ifile;
  }

  else if(strcmp(command,"setertti")==0) {
    struct G__input_file store_ifile = G__ifile;
    G__ifile.filenum = -1;
    G__ifile.line_number = -1;
#ifndef G__BUILDING_CINTTMP
    G__cpp_setupG__API();
#else
    G__fprinterr(G__serr,"ERROR: You cannot invoke the pragma setertti when running cint_tmp!");
#endif
    G__ifile = store_ifile;
  }

  else if(strcmp(command,"setstdstruct")==0) {
#ifndef G__TESTMAIN
    G__c_setupG__stdstrct();
#endif
  }

  else if(strcmp(command,"link")==0) {
    G__specify_link(G__SPECIFYLINK); /* look into newlink.c file for detail */
  }
  else if(strcmp(command,"stub")==0) {
    G__specify_link(G__SPECIFYSTUB); /* look into newlink.c file for detail */
  }

  else if(strcmp(command,"mask_newdelete")==0) {
    c = G__fgetstream(command, 0, ";\n\r");
    G__is_operator_newdelete |= (int)G__int(G__calc_internal(command));
  }

#ifdef G__SECURITY
  else if(strcmp(command,"security")==0) {
    c = G__fgetstream(command, 0, ";\n\r");
    G__security = G__getsecuritycode(command);
    /* if('\n'!=c&&'\r'!=c) G__fignoreline(); */
  }
#endif

#ifdef G__ASM_WHOLEFUNC
  else if(strcmp(command,"optimize")==0) {
    c = G__fgetstream(command, 0, ";\n\r");
    G__asm_loopcompile = G__int(G__calc_internal(command));
    G__asm_loopcompile_mode = G__asm_loopcompile; 
    /* if('\n'!=c&&'\r'!=c) G__fignoreline(); */
  }
  else if(strcmp(command,"bytecode")==0) {
#ifdef G__OLDIMPLEMENTATION1781_YET
    G__force_bytecode_compilation();
#else
    if(G__asm_dbg) {
      if(G__dispmsg>=G__DISPWARN) {
        G__fprinterr(G__serr,"Warning: #pragma bytecode obsoleted");
        G__printlinenum();
      }
    }
#ifdef G__DEBUG
    else {
      if(G__dispmsg>=G__DISPWARN) {
        G__fprinterr(G__serr,"Warning: #pragma bytecode obsoleted");
        G__printlinenum();
      }
    }
#endif
    /*
    store_asm_loopcompile=G__asm_loopcompile;
    G__asm_loopcompile = 10;
    */
#endif
  }
  else if(strcmp(command,"endbytecode")==0) {
    /*
    G__asm_loopcompile = store_asm_loopcompile;
    */
  }
#endif

  else if(strcmp(command,"K&R")==0) {
    G__nonansi_func=1;
    if(!G__globalcomp)
      G__genericerror(
        "Error: #pragma K&R only legal in parameter information file"
                      );
  }
  else if(strcmp(command,"ANSI")==0) {
    G__nonansi_func=0;
  }
  else if(strcmp(command,"extra_include")==0) {
    G__specify_extra_include();
    c='\n';
  }

  else if(strcmp(command,"do_not_include")==0) {
    G__do_not_include();
  }

  else if(0==strcmp(command,"define")) {
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
    c='\n';
  }

  else if(0==strcmp(command,"ifdef")) {
    G__pp_ifdef(1);
    c='\n';
  }
  else if(0==strcmp(command,"ifndef")) {
    G__pp_ifdef(0);
    c='\n';
  }
  else if(0==strcmp(command,"if")) {
    G__pp_if();
    c='\n';
  }
  else if(0==strcmp(command,"else")||
          0==strcmp(command,"elif")) {
    G__pp_skip(1);
    c='\n';
  }
  else if(0==strcmp(command,"endif")) {
    if('\n'!=c&&'\r'!=c) G__fignoreline();
    return(1);
  }

  else if(strcmp(command,"message")==0) {
    c=G__fgetline(command);
    G__fprinterr(G__serr,"%s\n",command());
  }

  else if(strcmp(command,"eval")==0) {
    store_no_exec_compile = G__no_exec_compile;
    G__no_exec_compile=0;
    c=G__fgetstream(command, 0, ";");
    fprintf(G__sout," evaluate (%d) ",store_no_exec_compile);
    G__calc_internal(command);
    G__no_exec_compile=store_no_exec_compile;
  }

#ifdef G__AUTOCOMPILE
  else if(strcmp(command,"endcompile")==0) {
    /* do nothing */
  }

  else if(strcmp(command,"autocompile")==0) {
    c=G__read_setmode(&G__compilemode);
  }
  else if(G__compilemode&&strcmp(command,"compile")==0) {
    /* if('\n'!=c&&'\r'!=c) G__fignoreline(); */
    if(0==G__prerun || -1 != G__func_now) {
      G__genericerror(
       "Error: '#pragma compile' must be placed outside of function in normal source file"
                      );
    }
    else {
      if((FILE*)NULL==G__fpautocc) {
        if(G__setautoccnames()) {
          G__compilemode = 0;
          if(G__dispmsg>=G__DISPWARN) {
            G__fprinterr(G__serr,"Warning: auto-compile disabled. Can not open tmp file");
            G__printlinenum();
          }
          return(1);
        }
        G__fpautocc=fopen(G__autocc_c,"w");
        if((FILE*)NULL==G__fpautocc) {
          if(G__dispmsg>=G__DISPWARN) {
            G__fprinterr(G__serr,"Warning: auto-compile disabled. Can not open tmp file");
            G__printlinenum();
          }
          G__compilemode = 0;
          return(1);
        }
      }
      G__appendautocc(G__fpautocc);
    }
  }
#endif

  else {
    int c2;
    G__FastAllocString args(4096*4);
    args[0]='\0';
    if('\n'!=c&&'\r'!=c) c = G__fgetline(args);
    /* Back up before a line terminator, to get errors reported correctly. */
    fseek (G__ifile.fp, -1, SEEK_CUR);
    c2 = G__fgetc ();
    if (c2 == '\n') {
      fseek (G__ifile.fp, -1, SEEK_CUR);
      G__ifile.line_number -= 2;
    }
    G__execpragma(command,args);
  }

  if('\n'!=c&&'\r'!=c) G__fignoreline();
  return(0);
}

/**************************************************************************
* G__sequrity_handle()
**************************************************************************/
int G__security_handle(G__UINT32 category)
{
  if(category==G__SECURE_EXIT_AT_ERROR) {
    G__security_error |= G__DANGEROUS;
    G__return = G__RETURN_EXIT1;
  }
  else {
    if(category&G__SECURE_POINTER_INIT) {
      G__security_error |= G__NOERROR;
      return(1);
    }
    if(category&G__SECURE_STANDARDLIB) {
      G__security_error |= G__NOERROR;
      return(1);
    }
    if(category&G__SECURE_BUFFER_SIZE) {
      G__genericerror("Limitation: Statement too long");
      G__security_error |= G__DANGEROUS;
    }
    if(category&G__SECURE_STACK_DEPTH) {
      G__genericerror("Function nesting too deep");
      G__security_error |= G__DANGEROUS;
    }

    G__fprinterr(G__serr,"cint: Security mode 0x%lx:0x%lx ",G__security,category);
    if(category&G__SECURE_POINTER_TYPE) {
      G__genericerror("Assignment to pointer from different type protected");
      G__security_error |= G__RECOVERABLE;
    }
    if(category&G__SECURE_POINTER_CALC) {
      G__genericerror("Pointer arithmetic protected");
      G__security_error |= G__RECOVERABLE;
    }
    if(category&G__SECURE_CAST2P) {
      G__genericerror("Casting to pointer protected");
      G__security_error |= G__RECOVERABLE;
    }
    if(category&G__SECURE_GOTO) {
      G__genericerror("Can not use goto statement");
      G__security_error |= G__RECOVERABLE;
    }
    if(category&G__SECURE_POINTER_AS_ARRAY) {
      G__genericerror("Can not use array index to a pointer");
      G__security_error |= G__RECOVERABLE;
    }


    if(category&G__SECURE_CASTING) {
      G__genericerror("Casting protected");
      G__security_error |= G__RECOVERABLE;
    }

    if(category&G__SECURE_MALLOC) {
      G__genericerror("Dynamic memory allocation protected");
      G__security_error |= G__RECOVERABLE;
    }
    if(category&G__SECURE_POINTER_OBJECT) {
      G__genericerror("Can not use pointer except for FILE*");
      G__security_error |= G__RECOVERABLE;
    }
    if(category&G__SECURE_POINTER_INSTANTIATE) {
      G__genericerror("Can not create pointer except for FILE*");
      G__security_error |= G__DANGEROUS;
    }
    if(category&G__SECURE_POINTER_ASSIGN) {
      G__genericerror("Can not assign throuth pointer");
      G__security_error |= G__RECOVERABLE;
    }
    if(category&G__SECURE_POINTER_REFERENCE) {
      G__genericerror("Can not reference through pointer");
      G__security_error |= G__RECOVERABLE;
    }
#if 0
    // G__SECURE_ARRAY is not used
    if(category&G__SECURE_ARRAY) {
      G__genericerror("Can not instantiate array");
      G__security_error |= G__DANGEROUS;
    }
#endif
    if(category&G__SECURE_FILE_POINTER) {
      G__genericerror("Can not use FILE pointer");
      G__security_error |= G__DANGEROUS;
    }
  }
  return(1);
}

#ifdef G__AUTOCOMPILE
/**************************************************************************
* G__setautoccnames()
**************************************************************************/
int G__setautoccnames()
{
  G__FastAllocString backup(G__MAXFILE);
  char *p;
  FILE *fpto;
  FILE *fpfrom;

  if(G__ifile.filenum<0) {
    return(1);
  }
  p = strrchr(G__srcfile[G__ifile.filenum].filename,'/');
  if(!p) p = strrchr(G__srcfile[G__ifile.filenum].filename,'\\');
  if(!p) p = strrchr(G__srcfile[G__ifile.filenum].filename,':');
  if(!p) p = G__srcfile[G__ifile.filenum].filename;
  else   ++p;
  G__FastAllocString fname(p);
  p = strrchr(fname,'.');
  if(p) *p = '\0';

  /* assign autocc filenames */
  if(G__iscpp) 
     G__snprintf(G__autocc_c,G__MAXNAME,"G__AC%s%s",fname(),G__getmakeinfo1("CPPSRCPOST"));
  else
     G__snprintf(G__autocc_c,G__MAXNAME,"G__AC%s%s",fname(),G__getmakeinfo1("CSRCPOST"));
  G__snprintf(G__autocc_h,G__MAXNAME,"G__AC%s",fname());
#ifdef G__WIN32
  G__snprintf(G__autocc_sl,G__MAXNAME,"G__AC%s%s",fname(),G__getmakeinfo1("DLLPOST"));
#else
  G__snprintf(G__autocc_sl,G__MAXNAME,"./G__AC%s%s",fname(),G__getmakeinfo1("DLLPOST"));
#endif
  G__snprintf(G__autocc_mak,G__MAXNAME,"G__AC%s.mak",fname());

  /* copy autocc file backup */
  backup.Format("G__%s",G__autocc_c);
  fpfrom=fopen(G__autocc_c,"r");
  if(fpfrom) {
    fpto=fopen(backup,"w");
    if(fpto) {
      G__copyfile(fpto,fpfrom);
      fclose(fpto);
    }
    else {/* error */
      fclose(fpfrom);
      return(1);
    }
    fclose(fpfrom);
  }
  else {
    fpto=fopen(backup,"w");
    if(fpto) {
      fprintf(fpto,"new autocc file\n");
      fclose(fpto);
    }
    else {/* error */
      return(1);
    }
  } 
  G__autoccfilenum = G__ifile.filenum;
  return(0);
}

/**************************************************************************
* G__autocc()
*
*  #pragma compiled appears in source code
**************************************************************************/
int G__autocc()
{
  G__FastAllocString temp(G__LONGLINE);
  char ansi[10];
  char cpp[10];

  fclose(G__fpautocc);
  G__fpautocc=(FILE*)NULL;
  G__autoccfilenum = -1;

  /* Compile shared library if updated */
  if(G__isautoccupdate()) {
    G__fprinterr(G__serr,"Compiling #pragma compile ...\n");
    ansi[0]='\0';
    if(G__cpp)  G__strlcpy(cpp,"-p",sizeof(cpp));
    else        cpp[0]='\0';

    if(G__iscpp) {
       temp.Format("makecint -mk %s %s %s %s %s -dl %s -H %s"
              ,G__autocc_mak
              ,ansi,cpp,G__allincludepath,G__macros,G__autocc_sl,G__autocc_c);
    }
    else {
       temp.Format("makecint -mk %s %s %s %s %s -dl %s -h %s"
              ,G__autocc_mak
              ,ansi,cpp,G__allincludepath,G__macros,G__autocc_sl,G__autocc_c);
    }
    if(G__asm_dbg) G__fprinterr(G__serr,"%s\n",temp());
    if (system(temp)) {
       G__fprinterr(G__serr, "Error running \"%s\"\n", temp());
    }

#if defined(G__SYMANTEC) && !defined(G__HAVE_CONFIG)
    temp.Format("smake -f %s",G__autocc_mak);
    if(G__asm_dbg) G__fprinterr(G__serr,"%s\n",temp());
    system(temp);
#elif defined(G__BORLAND) && !defined(G__HAVE_CONFIG)
    temp.Format("make.exe -f %s",G__autocc_mak);
    if(G__asm_dbg) G__fprinterr(G__serr,"%s\n",temp());
    system(temp);
#elif defined(G__VISUAL) && !defined(G__HAVE_CONFIG)
    temp.Format("nmake /f %s CFG=\"%s - Win32 Release\""
            ,G__autocc_mak,G__autocc_h);
    if(G__asm_dbg) G__fprinterr(G__serr,"%s\n",temp());
    system(temp);
    FILE *fp = fopen(G__autocc_sl,"r");
    if(fp) {
      fclose(fp);
      temp.Format("del %s",G__autocc_sl);
      if(G__asm_dbg) G__fprinterr(G__serr,"%s\n",temp());
      system(temp);
    }
    temp.Format("move Release\\%s %s",G__autocc_sl,G__autocc_sl);
    if(G__asm_dbg) G__fprinterr(G__serr,"%s\n",temp());
    system(temp);
#else
    temp.Format("make -f %s",G__autocc_mak);
    if(G__asm_dbg) G__fprinterr(G__serr,"%s\n",temp());
    if (system(temp)) {
       G__fprinterr(G__serr, "Error running \"%s\"\n", temp());
    }
#endif

    G__fprinterr(G__serr,"Finish compiling #pragma compile ...\n");
  }
  /* load automatically compiled shard library */
  G__shl_load(G__autocc_sl);
  return(0);
}

/**************************************************************************
* G__appendautocc()
**************************************************************************/
int G__appendautocc(FILE *fp)
{
  G__FastAllocString G__oneline(G__LONGLINE*2);
  G__FastAllocString G__argbuf(G__LONGLINE*2);
  char* arg[G__ONELINE];
  int argn;
  FILE *G__fp;

  G__fp=G__ifile.fp;

  while(G__readline_FastAlloc(G__fp,G__oneline,G__argbuf,&argn,arg)!=0) {
    ++G__ifile.line_number;
    if((argn>=3 && strcmp(arg[1],"#")==0 && strcmp(arg[2],"pragma")==0 &&
        strcmp(arg[3],"endcompile")==0) ||
       (argn>=2 && strcmp(arg[1],"#pragma")==0 && 
        strcmp(arg[2],"endcompile")==0)) {
      return(EXIT_SUCCESS);
    }
    else if(argn>=2 && strcmp(arg[1],"#")==0 && strcmp(arg[2],"pragma")==0) {
      if(argn>=3 && strcmp(arg[3],"include")==0) 
        fprintf(fp,"#include \"%s\"\n",arg[4]);
      else if(argn>=3 && strcmp(arg[3],"define")==0)
        fprintf(fp,"#%s\n",strstr(arg[0],"define"));
    }
    else if(argn>=1 && strcmp(arg[1],"#pragma")==0) {
      if(argn>=3 && strcmp(arg[2],"include")==0) 
        fprintf(fp,"#include \"%s\"\n",arg[3]);
      else if(argn>=2 && strcmp(arg[2],"define")==0)
        fprintf(fp,"#%s\n",strstr(arg[0],"define"));
    }
    else if(argn>=2 && strcmp(arg[1],"#")==0 && isdigit(arg[2][0])) {
    }
    else {
      fprintf(fp,"%s\n",arg[0]);
    }
  }
#ifndef G__OLDIMPLEMENTATON1724
  return(EXIT_SUCCESS);
#else
  G__genericerror("Error: '#pragma endcompile' not found");
  return(EXIT_FAILURE);
#endif
}
/**************************************************************************
* G__isautoccupdate()
**************************************************************************/
int G__isautoccupdate()
{
  G__FastAllocString backup(G__MAXFILE);
  int result;
  FILE *fp;
  backup.Format("G__%s",G__autocc_c);
  result=G__difffile(G__autocc_c,backup);
  remove(backup);
  if(0==result) {
    fp=fopen(G__autocc_sl,"r");
    if(!fp) result=1;
    else    fclose(fp);
  }
  return(result);
}
#endif

/**************************************************************************
* G__getsecuritycode()
*
**************************************************************************/
G__UINT32 G__getsecuritycode(const char *string)
{
  G__UINT32 code;
  int level;
  size_t len;
  if(string[0]) {
    if(isdigit(string[0])) {
      code = G__int(G__calc_internal(string));
    }
    else {
      len = strlen(string)-1;
      level = string[len] - '0';
      if(level>3) {
        if(G__dispmsg>=G__DISPWARN) {
          G__fprinterr(G__serr,
                   "Warning: Security level%d only experimental, High risk\n"
                       ,level);
        }
      }
      switch(level) {
      case 0: code = G__SECURE_LEVEL0; break;
      case 1: code = G__SECURE_LEVEL1; break;
      case 2: code = G__SECURE_LEVEL2; break;
      case 3: code = G__SECURE_LEVEL3; break;
      case 4: code = G__SECURE_LEVEL4; break;
      case 5: code = G__SECURE_LEVEL5; break;
      case 6: code = G__SECURE_LEVEL6; break;
      default:
        G__fprinterr(G__serr,"Error: Unknown seciruty code %s",string);
        G__genericerror((char*)NULL);
        code = G__security;
        break;
      }
    }
  }
  else {
    G__fprinterr(G__serr,"Error: Unknown seciruty code");
    G__genericerror((char*)NULL);
    code = G__security;
  }

  /* Prevent 2 #pragma security in one file */
  /* Let's not complain if the security requested is the same as
     before */
  /* In case of preprocessed file, the same logical file might actually
     be processed more than once. */
  if((G__security&G__SECURE_NO_CHANGE) &&(G__security!=code) ) {
    if(G__dispmsg>=G__DISPWARN) {
      G__fprinterr(G__serr,"Warning: security level locked, can't change");
      G__printlinenum();
    }
    code = G__security;
  }
  else if(G__security&G__SECURE_NO_RELAX) {
    if(G__dispmsg>=G__DISPWARN) {
      G__fprinterr(G__serr,"Warning: security level locked, can't relax");
      G__printlinenum();
    }
    code |= G__security;
  }

  if(-1!=G__ifile.filenum&&G__prerun) {
    /* Let's not complain if the security requested is the same as
       before */
    if((G__srcfile[G__ifile.filenum].security&G__SECURE_NO_CHANGE) &&
       (G__srcfile[G__ifile.filenum].security!=code)) {
      if(G__dispmsg>=G__DISPWARN) {
        G__fprinterr(G__serr,"Warning: security level locked, can't change");
        G__printlinenum();
      }
    }
    else {
      G__srcfile[G__ifile.filenum].security = code | G__SECURE_NO_CHANGE;
    }
  }

  return(code);
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
