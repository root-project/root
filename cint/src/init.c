/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file init.c
 ************************************************************************
 * Description:
 *  Entry functions
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

#ifndef __CINT__
void G__setCINTLIBNAME G__P((char *cintlib));
void G__setcopyflag G__P((int flag));
#endif

#ifndef G__OLDIMPLEMENTATION1207
extern int G__ispermanentsl;
extern G__DLLINIT G__initpermanentsl;
#endif

#ifndef G__OLDIMPLEMENTATION1817
#if defined(G__ROOT) && !defined(G__NATIVELONGLONG)
void G__cpp_setuplongif();
#endif
#endif

/**************************************************************************
* For C++ dictionary setup
**************************************************************************/
typedef struct {
   char              *libname;
   G__incsetup        func;
   int                inited;
} G__setup_func_struct;


static G__setup_func_struct **G__setup_func_list;
static int G__max_libs;
static int G__nlibs;
#ifndef G__OLDIMPLEMENTATION953
typedef void G__parse_hook_t ();
static G__parse_hook_t* G__afterparse_hook;
static G__parse_hook_t* G__beforeparse_hook;
#endif

/**************************************************************************
* G__add_setup_func(char *libname, G__incsetup func)
*
* Called by
*    G__cpp_setupXXX initializer ctor.
*
**************************************************************************/
void G__add_setup_func(libname, func)
char *libname;
G__incsetup func;
{
   int i, islot = -1;

   if (!G__setup_func_list) {
      G__max_libs = 10;
      G__setup_func_list = (G__setup_func_struct**)calloc(G__max_libs,sizeof(G__setup_func_struct*));
   }
   if (G__nlibs >= G__max_libs) {
      G__max_libs += 10;
      G__setup_func_list = (G__setup_func_struct**)realloc(G__setup_func_list,
                                   G__max_libs*sizeof(G__setup_func_struct*));
      for (i = G__nlibs; i < G__max_libs; i++)
         G__setup_func_list[i] = 0;
   }

   /* if already in table: ignore (could also print warning) */
   for (i = 0; i < G__nlibs; i++)
      if (G__setup_func_list[i] &&
          !strcmp(G__setup_func_list[i]->libname, libname)) return;

   /* find empty slot */
   for (i = 0; i < G__nlibs; i++)
      if (!G__setup_func_list[i]) {
         islot = i;
         break;
      }
   if (islot == -1) islot = G__nlibs++;

   G__setup_func_list[islot] = (G__setup_func_struct*)malloc(sizeof(G__setup_func_struct));
   G__setup_func_list[islot]->libname = malloc(strlen(libname)+1);
   G__setup_func_list[islot]->func    = func;
   G__setup_func_list[islot]->inited  = 0;
   strcpy(G__setup_func_list[islot]->libname, libname);
}

/**************************************************************************
* G__remove_setup_func(char *libname)
*
* Called by
*    G__cpp_setupXXX initializer dtor.
*
**************************************************************************/
void G__remove_setup_func(libname)
char *libname;
{
   int i;

   for (i = 0; i < G__nlibs; i++)
      if (G__setup_func_list[i] &&
          !strcmp(G__setup_func_list[i]->libname, libname)) {
         free(G__setup_func_list[i]->libname);
         free(G__setup_func_list[i]);
         G__setup_func_list[i] = 0;
         if (i == G__nlibs-1) G__nlibs--;
         return;
      }
}

/**************************************************************************
* G__call_setup_funcs(void)
*
* Called by
*    G__init_cint() and G__dlmod().
*
**************************************************************************/
int G__call_setup_funcs()
{
  int i, k = 0;
#ifndef G__OLDIMPLEMENTATION1035
  G__LockCriticalSection();
#endif

  for (i = 0; i < G__nlibs; i++)
    if (G__setup_func_list[i] && !G__setup_func_list[i]->inited) {
      (G__setup_func_list[i]->func)();
      G__setup_func_list[i]->inited = 1;
#ifndef G__OLDIMPLEMENTATION1207
      G__initpermanentsl = G__setup_func_list[i]->func;
#endif
      k++;
#ifdef G__DEBUG
      fprintf(G__sout,"Dictionary for %s initialized\n", G__setup_func_list[i]->libname); /* only for debug */
#endif
    }
#ifndef G__OLDIMPLEMENTATION1035
  G__UnlockCriticalSection();
#endif
  return k;
}

/**************************************************************************
* G__reset_setup_funcs(void)
*
* Called by
*    G__scratch_all().
*
**************************************************************************/
void G__reset_setup_funcs()
{
   int i;

   for (i = 0; i < G__nlibs; i++)
      if (G__setup_func_list[i])
         G__setup_func_list[i]->inited = 0;
}


#ifndef G__OLDIMPLEMENTATION442
/**************************************************************************
* Windows-NT Symantec C++ setup
**************************************************************************/
struct G__libsetup_list {
  void (*p2f)();
  struct G__libsetup_list *next;
} ;

static struct G__libsetup_list G__p2fsetup;

/**************************************************************************
* G__set_p2fsetup()
**************************************************************************/
void G__set_p2fsetup(p2f)
void (*p2f)();
{
  struct G__libsetup_list *setuplist;
  setuplist = &G__p2fsetup;
  /* get to the end of list */
  while(setuplist->next) setuplist = setuplist->next;
  /* add given entry to the list */
  setuplist->p2f = p2f;
  /* allocation new list entry */
  setuplist->next
    = (struct G__libsetup_list*)malloc(sizeof(struct G__libsetup_list));
  setuplist->next->next=(struct G__libsetup_list*)NULL;
}

/**************************************************************************
* G__free_p2fsetuplist()
**************************************************************************/
static void G__free_p2fsetuplist(setuplist)
struct G__libsetup_list *setuplist;
{
  if(setuplist->next) {
    G__free_p2fsetuplist(setuplist->next);
    free(setuplist->next);
    setuplist->next = (struct G__libsetup_list*)NULL;
  }
  setuplist->p2f = (void (*)())NULL;
}

/**************************************************************************
* G__free_p2fsetup()
**************************************************************************/
void G__free_p2fsetup()
{
  G__free_p2fsetuplist(&G__p2fsetup);
}

/**************************************************************************
* G__do_p2fsetup()
**************************************************************************/
static void G__do_p2fsetup()
{
  struct G__libsetup_list *setuplist;
  setuplist = &G__p2fsetup;
  while(setuplist->next) {
    (*setuplist->p2f)();
    setuplist=setuplist->next;
  }
#ifndef G__OLDIMPLEMENTATION874
#ifdef G__OLDIMPLEMENTATION1707
  G__free_p2fsetup();
#endif
#endif
}
#endif

#ifndef G__OLDIMPLEMENATTION1090
/**************************************************************************
* G__read_include_env()
**************************************************************************/
static void G__read_include_env(envname)
char* envname;
{
  char *env = getenv(envname);
  if(env) {
    char *p,*pc;
    char* tmp = (char*)malloc(strlen(env)+2);
    strcpy(tmp,env);
    p=tmp;
    while( (pc=strchr(p,';')) || (pc=strchr(p,',')) ) {
      *pc=0;
      if(p[0]) {
        if(strncmp(p,"-I",2)==0) G__add_ipath(p+2);
        else G__add_ipath(p);
      }
      p=pc+1;
    }
    if(p[0]) {
      if(strncmp(p,"-I",2)==0) G__add_ipath(p+2);
      else G__add_ipath(p);
    }
    free((void*)tmp);
  }
}
#endif

/**************************************************************************
* G__init_cint(char *command)
*
* Called by
*    nothing. User host application should call this
*
* Return:
* G__CINT_INIT_FAILURE   (-1) : If initialization failed
* G__CINT_INIT_SUCCESS      0 : Initialization success.
*                               no main() found in given source.
* G__CINT_INIT_SUCCESS_MAIN 1 : Initialization + main() executed successfully.
*
*  Initialize cint, read source file(if any) and return to host program.
* Return 0 if success, -1 if failed.
* After calling this function, G__calc() or G__pause() has to be explicitly
* called to start interpretation or interactive interface.
*
*  Example user source code
*
*    G__init_cint("cint source.c"); // read source.c and allocate globals
*    G__calc("func1()");   // interpret func1() which should be in source.c
*    while(G__pause()==0); // start interactive interface
*    G__calc("func2()");   // G__calc(),G__pause() can be called multiple times
*    G__scratch_all();     // terminate cint
*
*    G__init_cint("cint source2.c"); // you can start cint again
*    G__calc("func3()");
*          .
*    G__scratch_all();     // terminate cint
*
**************************************************************************/
int G__init_cint(command)
char *command;
{
  int argn=0,i;
  int result;
  char *arg[G__MAXARG];
  char argbuf[G__LONGLINE];

#ifndef G__OLDIMPLEMENTATION1035
    G__LockCriticalSection();
#endif

  /***************************************
  * copy command to argbuf. argbuf will
  * be modified by G__split().
  ***************************************/
  if(G__commandline!=command) strcpy(G__commandline,command);
  strcpy(argbuf,command);

  /***************************************
  * split arguments as follows
  *
  *        arg[0]
  *   <--------------->
  *  "cint -s source1.c"
  *    ^   ^        ^
  * arg[1] arg[2]  arg[3]
  ***************************************/
  G__split(G__commandline,argbuf,&argn,arg);

  /***************************************
  * shift argument as follows
  *
  *  "cint -s source1.c"
  *    ^   ^        ^
  * arg[0] arg[1]  arg[2]
  ***************************************/
  for(i=0;i<argn;i++) arg[i]=arg[i+1];
  while(i<G__MAXARG) {
    arg[i]=(char*)NULL;
    ++i;
  }

  /***************************************
  * G__othermain is set to 2 so that
  * G__main() returns without start
  * interpretation.
  *
  *  G__main() returns 0 if it is OK,
  * -1 if error occur.
  ***************************************/
  G__othermain=2;
  result=G__main(argn,arg);


  if(G__MAINEXIST==G__ismain) {
#ifndef G__OLDIMPLEMENTATION1035
    G__UnlockCriticalSection();
#endif
    return(G__INIT_CINT_SUCCESS_MAIN);
  }
  else if(result==EXIT_SUCCESS) {
#ifndef G__OLDIMPLEMENTATION1035
    G__UnlockCriticalSection();
#endif
    return(G__INIT_CINT_SUCCESS);
  }
  else {
#ifndef G__OLDIMPLEMENTATION1035
    G__UnlockCriticalSection();
#endif
    return(G__INIT_CINT_FAILURE);
  }

}



/**************************************************************************
* USING THIS FUNCTION IS NOT RECOMMENDED
*
* G__load()
*
* Called by
*    nothing. User host application should call this
*
* If main() exists in user compiled function, G__load() is an entry.
* Command file has to contain
*    cint < [options] > [file1] < [file2] < [file3] <...>>>
**************************************************************************/
int G__load(commandfile)
char *commandfile;
{
  int argn=0,i;
  char *arg[G__MAXARG],line[G__LONGLINE*2],argbuf[G__LONGLINE*2];
  FILE *cfp;


  cfp=fopen(commandfile,"r");
  if(cfp==NULL) {
    fprintf(stderr,"Error: command file \"%s\" doesn't exist\n"
	    ,commandfile);
    fprintf(stderr,"  Make command file : [comID] <[cint options]> [file1] <[file2]<[file3]<...>>>\n");
    return(-1);
  }


  /**************************************************************
   * Read command file line by line.
   **************************************************************/
  while(G__readline(cfp,line,argbuf,&argn,arg)!=0) {

    for(i=0;i<argn;i++) arg[i]=arg[i+1];
    arg[argn]=NULL;

    /*******************************************************
     * Ignore line start with # and blank lines.
     * else call G__main()
     *******************************************************/
    if((argn>0)&&(arg[0][0]!='#')) {
      G__othermain=1;
      G__main(argn,arg);
      if(G__return>G__RETURN_EXIT1) return(0);
      G__return=G__RETURN_NON;
    }
  }

  fclose(cfp);

  return(0);
}

#ifndef G__OLDIMPLEMENTATION563
/**************************************************************************
* G__getcintready()
*
**************************************************************************/
int G__getcintready()
{
  return(G__cintready);
}
#endif

/**************************************************************************
* G__setothermain()
*
**************************************************************************/
void G__setothermain(othermain)
int othermain;
{
  G__othermain = (short)othermain;
}

/**************************************************************************
* G__setglobalcomp()
*
**************************************************************************/
void G__setglobalcomp(globalcomp)
int globalcomp;
{
  G__globalcomp = globalcomp;
}

#ifndef G__OLDIMPLEMENTATION953
/**************************************************************************
 * G__set_afterparse_hook()
**************************************************************************/
G__parse_hook_t* G__set_afterparse_hook (G__parse_hook_t* hook)
{
  G__parse_hook_t* old= G__afterparse_hook;
  G__afterparse_hook = hook;
  return old;
}

/**************************************************************************
* G__set_beforeparse_hook()
**************************************************************************/
G__parse_hook_t* G__set_beforeparse_hook (G__parse_hook_t* hook)
{
  G__parse_hook_t* old= G__beforeparse_hook;
  G__beforeparse_hook = hook;
  return old;
}
#endif

/**************************************************************************
* G__display_note()
**************************************************************************/
void G__display_note() {
  G__more(G__sout,"\n");
  G__more(G__sout,"Note1: Cint is not aimed to be a 100%% ANSI/ISO compliant C/C++ language\n");
  G__more(G__sout," processor. It rather is a portable script language environment which\n");
  G__more(G__sout," is close enough to the standard C++.\n");
  G__more(G__sout,"\n");
  G__more(G__sout,"Note2: Regulary check either of /tmp /usr/tmp /temp /windows/temp directory\n");
  G__more(G__sout," and remove temp-files which are accidentally left by cint.\n");
  G__more(G__sout,"\n");
  G__more(G__sout,"Note3: Cint reads source file on-the-fly from the file system. Do not change\n");
  G__more(G__sout," the active source during cint run. Use -C option or C1 command otherwise.\n");
  G__more(G__sout,"\n");
  G__more(G__sout,"Note4: In source code trace mode, cint sometimes displays extra-characters.\n");
  G__more(G__sout," This is harmless. Please ignore.\n");
  G__more(G__sout,"\n");
}

#ifndef G__OLDIMPLEMENTATION889
/***********************************************************************
* G__getopt();
*
***********************************************************************/
int G__optind=1;
char *G__optarg;
#define optind G__optind
#define optarg G__optarg
#define getopt G__getopt

int G__getopt(argc, argv,optlist)
int argc;
char **argv;
char *optlist;
{
  int optkey;
  char *p;
  if(optind<argc) {
    if('-'==argv[optind][0]) {
      optkey = argv[optind][1] ;
      p = optlist;
      while(*p) {
	if( (*p) == optkey ) {
	  ++p;
	  if(':'==(*p)) { /* option with argument */
	    if(argv[optind][2]) { /* -aARGUMENT */
	      optarg=argv[optind]+2;
	      optind+=1;
	      return(argv[optind-1][1]);
	    }
	    else { /* -a ARGUMENT */
	      optarg=argv[optind+1];
	      optind+=2;
	      return(argv[optind-2][1]);
	    }
	  }
	  else { /* option without argument */
	    ++optind;
	    optarg=(char*)NULL;
	    return(argv[optind-1][1]);
	  }
	}
	++p;
      }
      G__fprinterr(G__serr,"Error: Unknown option %s\n",argv[optind]);
      ++optind;
      return(0);
    }
    else {
      return(EOF);
    }
  }
  else {
    return (EOF);
  }
}
#endif

#ifndef G__OLDIMPLEMENTATION1078
extern int G__quiet;
#endif

/**************************************************************************
* G__main()
*
* Called by
*    G__init_cint()
*    G__load          G__ONLINELOAD is defined
*    main()
*
* Main entry of the C/C++ interpreter
**************************************************************************/
int G__main(argc,argv)
int  argc ;
char *argv[] ;
{
  int stepfrommain=0;
  int  ii ;
  char *forceassignment=NULL;
  int xfileflag=0;
  char sourcefile[G__MAXFILENAME];
  /*************************************************************
   * C/C++ interpreter option related variables
   *************************************************************/
  extern int optind;
  extern char *optarg;
  static char usage[]="Usage: %s [options] [sourcefiles|suboptions] [arguments]\n";
  static char *progname;
  char *ppc;

  /*************************************************************
   * Interpreted code option related variables
   *************************************************************/
  int c,iarg;
  char temp[G__ONELINE];
  long G__argpointer[G__MAXARG];
  char dumpfile[G__MAXFILENAME];
  G__value result;
  char *linkfilename=(char*)NULL;
  int linkflag=0;
  char *dllid=(char*)NULL;
  struct G__dictposition stubbegin;
#ifndef G__OLDIMPLEMENTATION1996
  char *icom=(char*)NULL;
#endif

#ifndef G__OLDIMPLEMENTATION2028
  stubbegin.ptype = (char*)G__PVOID;
#endif


  /*****************************************************************
  * Setting STDIOs.  May need to modify here.
  *  init.c, end.c, scrupto.c, pause.c
  *****************************************************************/
  if((FILE*)NULL==G__stdout) G__stdout=stdout;
  if((FILE*)NULL==G__stderr) G__stderr=stderr;
  if((FILE*)NULL==G__stdin) G__stdin=stdin;
  G__serr = G__stderr;
#ifndef G__OLDIMPLEMENTATION463
  G__sout = G__stdout;
  G__sin = G__stdin;
#endif

#ifdef G__MEMTEST
  G__memanalysis();
#endif

  G__ifunc.allifunc = 0;
  G__ifunc.next = (struct G__ifunc_table *)NULL;
#ifdef G__NEWINHERIT
  G__ifunc.tagnum = -1;
#endif
#ifndef G__OLDIMPLEMENTATION1543
  {
    int ix;
    for(ix=0;ix<G__MAXIFUNC;ix++) G__ifunc.funcname[ix] = (char*)NULL;
  }
#endif
  G__p_ifunc = &G__ifunc ;


  /*************************************************************
   * Inialization before running C/C++ interpreter.
   * Clear ifunc table,global variables and local variable pointer
   *************************************************************/
  G__scratch_all();            /* scratch all malloc memory */

#ifndef G__OLDIMPLEMENTATION953
  if (G__beforeparse_hook) G__beforeparse_hook ();
#endif

#ifdef G__SECURITY
  G__init_garbagecollection();
#endif

  /*************************************************************
   * Inialization of readline dumpfile pointer
   *************************************************************/
  for(ii=0;ii<=5;ii++) {
    G__dumpreadline[ii]=NULL;
    G__Xdumpreadline[ii]=0;
  }

  if(argv[0]) sprintf(G__nam,"%s",argv[0]); /* get command name */
  else        strcpy(G__nam,"cint");

  /*************************************************************
   * Set stderr,stdin,stdout,NULL pointer values to global
   *************************************************************/
  G__macros[0]='\0';
  G__set_stdio();


#ifndef G__OLDIMPLEMENTATION1817
#if defined(G__ROOT) && !defined(G__NATIVELONGLONG)
  {
    int xtagnum,xtypenum;
    G__cpp_setuplongif();
    xtagnum=G__defined_tagname("G__longlong",2);
    xtypenum=G__search_typename("long long",'u',xtagnum,G__PARANORMAL);
    xtagnum=G__defined_tagname("G__ulonglong",2);
    xtypenum=G__search_typename("unsigned long long",'u',xtagnum,G__PARANORMAL);
    xtagnum=G__defined_tagname("G__longdouble",2);
    xtypenum=G__search_typename("long double",'u',xtagnum,G__PARANORMAL);
  }
#endif
#endif

  /* Signal handling moved after getopt to enable core dump with 'E' */

#ifdef G__HSTD
  /*************************************************************
   * TEE software error handling
   *************************************************************/
  if(setjmp(EH_env)) {
    G__errorprompt("TEE software error");
    G__scratch_all();
    return(EXIT_FAILURE);
  }
#endif

  /*************************************************************
   * set compile/interpret environment
   *************************************************************/
  G__breakline[0]='\0';
  G__breakfile[0]='\0';
#ifndef G__OLDIMPLEMENTATION928
  if(G__allincludepath) free(G__allincludepath);
  G__allincludepath=(char*)malloc(1);
#endif
  G__allincludepath[0]='\0';

  /*************************************************************
   * Set compiled global variable pointer to interpreter variable
   * pointer.  Global variables in compiled code are tied to
   * global variables in interpreted code.
   *************************************************************/
  G__prerun=1;
  G__setdebugcond();
  G__globalsetup();
  G__call_setup_funcs();
#ifndef G__OLDIMPLEMENTATION442
  G__do_p2fsetup();
#endif
  G__prerun=0;
  G__setdebugcond();

  /*************************************************************
   * Get command name
   *************************************************************/
  progname=argv[0];

  /*************************************************************
   * Initialize dumpfile name. If -d option is given, dumpfile
   * is set to some valid file name.
   *************************************************************/
  dumpfile[0]='\0';
  /* sprintf(dumpfile,""); */

#ifndef G__TESTMAIN
  optind=1;
#endif

  /*************************************************************
   * Get command options
   *************************************************************/
  while((c=getopt(argc,argv
  ,"a:b:c:d:ef:gij:kl:mn:pq:rstu:vw:x:y:z:AB:CD:EF:G:H:I:J:KM:N:O:P:QRSTU:VW:X:Y:Z:-:@+:"))
	!=EOF) {
    switch(c) {

#ifndef G__OLDIMPLEMENTATION2226
    case '+':
      G__setmemtestbreak(atoi(optarg)/10000,atoi(optarg)%10000);
      break;
#endif

#ifndef G__OLDIMPLEMENTATION2068
#ifdef G__CINT_VER6
    case '@':
      if(G__cintv6==0) G__cintv6=G__CINT_VER6;
      else if(G__cintv6==G__CINT_VER6) G__cintv6|=G__BC_DEBUG;
      break;
#endif
#endif

#ifndef G__OLDIMPLEMENTATION1725
    case 'H': /* level of inclusion for dictionary generation */
      G__gcomplevellimit = atoi(optarg);
      break;
#endif

    case 'J': /* 0 - 5 */
      G__dispmsg = atoi(optarg);
      break;

#ifndef G__OLDIMPLEMENTATION1525
    case 'j':
      G__multithreadlibcint = atoi(optarg);
      break;
#endif

#ifndef G__OLDIMPLEMENTATION1480
    case 'm':
      G__lang = G__ONEBYTE;
      break;
#endif

#ifndef G__OLDIMPLEMENTATION1078
    case 'Q':
      G__quiet=1;
      break;
#endif

#ifndef G__OLDIMPLEMENTATION453
    case 'B':
      G__setInitFunc(optarg);
      break;
#endif

#ifndef G__OLDIMPLEMENTATION970
    case 'C':
      G__setcopyflag(1);
      break;
#endif

    case 'y':
      G__setCINTLIBNAME(optarg);
      break;

    case 'z':
      G__setPROJNAME(optarg);
      break;

    case 'w': /* set Windows-NT DLL mode */
      G__setDLLflag(atoi(optarg));
      break;

    case 'M':
      G__is_operator_newdelete = (int)G__int(G__calc_internal((optarg)));
#ifndef G__OLDIMPLEMENTATION1513
      if(G__NOT_USING_2ARG_NEW&G__is_operator_newdelete) {
	G__fprinterr(G__serr,"!!!-M option may not be needed any more. A new scheme has been implemented.\n");
	G__fprinterr(G__serr,"!!!Refer to $CINTSYSDIR/doc/makecint.txt for the detail.\n");
      }
#endif
      break;

    case 'V': /* Create precompiled private member dictionary */
      G__precomp_private=1;
      break;

    case 'q':
#ifdef G__SECURITY
      G__security = G__getsecuritycode(optarg);
#endif
      break;

    case 'e': /* Parse all extern declaration */
      G__parseextern=1;
      break;
    case 'i': /* interactive return */
      G__interactive=1;
      break;

    case 'x':
      if(xfileflag) {
	G__fprinterr(G__serr,"Error: only one -x option is allowed\n");
	G__exit(EXIT_FAILURE);
      }
      else {
#ifndef G__OLDIMPLEMENATTION1919
	xfileflag=optind-1;
#else
#ifndef G__OLDIMPLEMENATTION1564
	G__tmpnam(G__xfile); /* not used anymore */
#else
	tmpnam(G__xfile);
#endif
	G__ifile.fp=fopen(G__xfile,"w");
	fprintf(G__ifile.fp,"%s\n",optarg);
	fclose(G__ifile.fp);
	xfileflag=1;
#endif
      }
      break;

    case 'I': /* Include path */
#ifndef G__OLDIMPLEMENATTION1090
      /*************************************************************
       * INCLUDE environment variable. Implemented but must not activate this.
       *************************************************************/
      if(','==optarg[0]) G__read_include_env(optarg+1);
      else G__add_ipath(optarg);
#else
      G__add_ipath(optarg);
#endif
      break;

    case 'F': /* force assignment is evaluated after pre-RUN */
      forceassignment = optarg ;
      break;

#ifndef G__OLDIMPLEMENTATION1285
    case 'Y': /* Do not ignore std namespace or not */
      G__ignore_stdnamespace = atoi(optarg);
      break;
#endif

    case 'Z': /* Generate archived header */
#ifndef G__OLDIMPLEMENTATION1278
      G__autoload_stdheader = atoi(optarg);
#else
      G__genericerror("Sorry, this feature has been eliminated");
      G__exit(EXIT_SUCCESS);
#endif
      break;

    case 'n': /* customize G__cpplink file name
	       *   G__cppXXX?.C , G__cXXX.c */
      if(linkflag) {
        G__fprinterr(G__serr,"Warning: option -n[linkname] must be given prior to -c[linklevel]\n");
      }
      linkfilename = optarg;
      break;

    case 'N': /* customize DLL identification name */
      if(linkflag) {
        G__fprinterr(G__serr,"Warning: option -N[DLL_Name] must be given prior to -c[linklevel]\n");
      }
      dllid = optarg;
      break;

#ifndef G__OLDIMPLEMENTATION1451
    case 'U':
      G__SystemIncludeDir=G__AddConstStringList(G__SystemIncludeDir,optarg,1);
      break;
#endif

#ifndef G__OLDIMPLEMENTATION411
    case 'u':
      if(!G__fpundeftype) {
	G__security = 0;
	G__fpundeftype = fopen(optarg,"w");
	fprintf(G__fpundeftype,"/* List of possible undefined type names. This list is not perfect.\n");
  	fprintf(G__fpundeftype,"* It is user's responsibility to modify this file. \n");
  	fprintf(G__fpundeftype,"* There are cases that the undefined type names can be\n");
  	fprintf(G__fpundeftype,"*   class name\n");
  	fprintf(G__fpundeftype,"*   struct name\n");
  	fprintf(G__fpundeftype,"*   union name\n");
  	fprintf(G__fpundeftype,"*   enum name\n");
  	fprintf(G__fpundeftype,"*   typedef name\n");
  	fprintf(G__fpundeftype,"*   not a typename but object name by CINT's mistake\n");
  	fprintf(G__fpundeftype,"* but CINT can not distinguish between them. So it outputs 'class xxx;' \n");
  	fprintf(G__fpundeftype,"* for all.\n");
  	fprintf(G__fpundeftype,"*/\n");
      }
      break;
#endif

#ifdef G__SHAREDLIB
    case 'l': /* dynamic link file, shared library file */
      if(
#ifndef G__OLDIMPLEMENTATION1908
	 -1==G__shl_load(optarg)
#else
	 G__shl_load(optarg)==EXIT_FAILURE
#endif
	 ) {
	if(G__key!=0) system("key .cint_key -l execute");
	G__scratch_all();
	return(EXIT_FAILURE);
      }
      break;
#else
    case 'l': /* error message for dynamic link option */
      G__fprinterr(G__serr,"Error: %s is not compiled with dynamic link capability\n",argv[0]);
      break;
#endif
    case 'p': /* use preprocessor */
      G__cpp = 1;
      break;
    case 'P': /* specify what preprocessor to use */
    case 'W':
      switch(optarg[1]) {
      case 'p':
	strcpy(G__ppopt,optarg+2);
	ppc=G__ppopt;
	while((ppc=strchr(ppc,','))) *ppc=' ';
	break;
      case 'l':
	break;
      }
      break;
    case 'A': /* C++ mode lock */
      G__cpplock = 1;
      G__iscpp = 1;
      break;
    case 'K': /* C mode lock */
      G__clock = 0; /* Why I did this? Why not 1? */
      G__iscpp = 0;
#ifndef G__OLDIMPLEMENTATION402
      if(G__CLINK==G__globalcomp) G__clock=1;
#endif
      break;
    case 'g': /* whole function compile off */
      G__asm_loopcompile = 3;
#ifndef G__OLDIMPLEMENTATION1155
      G__asm_loopcompile_mode = G__asm_loopcompile; 
#endif
      break;
    case 'O': /* loop compiler on */
      G__asm_loopcompile = atoi(optarg);
#ifndef G__OLDIMPLEMENTATION1155
      G__asm_loopcompile_mode = G__asm_loopcompile; 
#endif
      break;
    case 'v': /* loop compiler debug mode */
      G__asm_dbg = 1;
      break;
    case 'D': /* define macro */
      G__add_macro(optarg);
      break;
    case 'E': /* Dump core at error */
#ifndef G__OLDIMPLEMENTATION1947
      if(1==G__catchexception) G__catchexception = 0;
      else if(0==G__catchexception) G__catchexception=2;
      else ++G__catchexception;
#endif
#ifndef G__OLDIMPLEMENTATION1946
      ++G__coredump;
#else
      G__coredump = 1;
#endif
      break;
    case 'X': /* readline dumpfile execution */
      G__dumpreadline[0]=fopen(optarg,"r");
      if(G__dumpreadline[0]) {
	G__Xdumpreadline[0]=1;
	G__fprinterr(G__serr," -X : readline dumpfile %s executed\n",
		optarg);
      }
      else {
	G__fprinterr(G__serr,
		"Readline dumpfile %s can not open\n"
		,optarg);
	return(EXIT_FAILURE);
      }
      break;
    case 'd': /* dump file */
#ifdef G__DUMPFILE
      G__fprinterr(G__serr," -d : dump function call history to %s\n",
	      optarg);
      if(strcmp(optarg+strlen(optarg)-2,".c")==0 ||
	 strcmp(optarg+strlen(optarg)-2,".C")==0 ||
	 strcmp(optarg+strlen(optarg)-2,".h")==0 ||
	 strcmp(optarg+strlen(optarg)-2,".H")==0) {
	G__fprinterr(G__serr,"-d %s : Improper history dump file name\n"
		,optarg);
      }
      else sprintf(dumpfile,"%s",optarg);
#else
      G__fprinterr(G__serr,
	      " -d : func call dump not supported now\n");
#endif
      break;
    case 'k': /* user function key */
      system("key .cint_key -l pause");
      G__key=1;
      break;
    case 'c': /* global compile */
      /* G__CPPLINK1
       * c-1 : create global variable & function information from
       *      C++ header file. Default link on for pure cint, off for ROOT
       * c-2 : create global variable & function information from
       *      C header file. Default link on for pure cint, off for ROOT
       * c-10 : create global variable & function information from
       *      C++ header file. Default link off
       * c-20 : create global variable & function information from
       *      C header file. Default link off
       * c-11 : create global variable & function information from
       *      C++ header file. Default link on
       * c-21 : create global variable & function information from
       *      C header file. Default link on
       */
      G__globalcomp=atoi(optarg);
#ifndef G__OLDIMPLEMENTATION1700
      if(abs(G__globalcomp)>=10) {
	G__default_link = abs(G__globalcomp)%10;
	G__globalcomp /= 10;
      }
#endif
      linkflag=1;
      if(!linkfilename) {
        switch(G__globalcomp) {
        case G__CPPLINK: 
	  linkfilename = "G__cpplink.C";
#ifndef G__OLDIMPLEMENTATION772
	  G__iscpp=1;
	  G__cpplock=1;
#endif
	  break;
        case G__CLINK:
	  linkfilename = "G__clink.c";
#ifndef G__OLDIMPLEMENTATION772
	  G__iscpp=0;
	  G__clock=1;
#endif
	  break;
        default:
	  linkfilename = "G__cpplink.cxx"; break;
        }
      }
      if(!dllid) {
	dllid = "";
      }
      G__set_globalcomp(optarg,linkfilename,dllid);
      break;
    case 's': /* step into mode */
      G__fprinterr(G__serr," -s : Step into function/loop mode\n");
      G__steptrace = 1;
      break;
    case 'S': /* step over mode */
      G__stepover=3;
      G__fprinterr(G__serr," -S : Step over function/loop mode\n");
      stepfrommain = 1;
      break;
    case 'b': /* break line */
      strcpy(G__breakline,optarg);
      break;
    case 'f': /* break file */
      strcpy(G__breakfile,optarg);
      break;
    case 'a': /* assertion */
      strcpy(G__assertion,optarg);
#ifdef G__OLDIMPLEMENTATION418
      G__fprinterr(G__serr," -a : break at assertion %s\n" ,G__assertion);
#endif
      break;
    case 'T': /* trace of input file */
      /* sprintf(monitorfile,"%s",optarg); */
      G__fprinterr(G__serr," -T : trace from pre-run\n");
      G__debugtrace=G__istrace=G__debug = 1;
      G__setdebugcond();
      break;
    case 'G': /* trace dump */
      G__serr = fopen(optarg,"w");
    case 't': /* trace of input file */
      /* sprintf(monitorfile,"%s",optarg); */
      G__fprinterr(G__serr," -t : trace execution\n");
      G__istrace=G__debugtrace = 1;
      break;
    case 'R': /* displays input file at the break point*/
      /* sprintf(monitorfile,"%s",optarg); */
      G__fprinterr(G__serr," -d : display at break point mode\n");
      G__breakdisp = 1;
      break;
    case 'r': /* revision */
      G__revprint(G__sout);
      if(G__key!=0) system("key .cint_key -l execute");
      return(EXIT_SUCCESS);
      /* break; */

#ifndef G__OLDIMPLEMENTATION1996
    case '-':
      icom = optarg;
      break;
#endif

    default:
#ifndef G__SMALLOBJECT
      G__more_pause((FILE*)NULL,0);
      fprintf(G__sout,usage,progname);
      G__more_pause(G__sout,0);

      G__display_note();

      G__more(G__sout,"options     (* : used only with makecint or -c option)\n");
#ifdef G__OLDIMPLEMENTATION418
      G__more(G__sout,"  -a [assertion] : set assertion for break condition\n");
#endif
      G__more(G__sout,"  -A : ANSI C++ mode(default)\n");
      G__more(G__sout,"  -b [line] : set break line\n");
      G__more(G__sout,"* -c -1: make C++ precompiled interface method files\n");
      G__more(G__sout,"* -c -2: make C precompiled interface method files\n");
      G__more(G__sout,"* -c -10: make C++ precompiled interface method files. Default link off\n");
      G__more(G__sout,"* -c -20: make C precompiled interface method files. Default link off\n");
      G__more(G__sout,"* -c -11: make C++ precompiled interface method files. Default link on\n");
      G__more(G__sout,"* -c -21: make C precompiled interface method files. Default link on\n");
#ifndef G__OLDIMPLEMENTATION970
      G__more(G__sout,"  -C : copy src to $TMPDIR so that src can be changed during cint run\n");
#endif
#ifdef G__DUMPFILE
      G__more(G__sout,"  -d [dumpfile] : dump function call history\n");
#endif
      G__more(G__sout,"  -D [macro] : define macro symbol for #ifdef\n");
      G__more(G__sout,"  -e : Not ignore extern declarations\n");
      G__more(G__sout,"  -E : Dump core at error\n");
#ifndef G__OLDIMPLEMENTATION1946
      G__more(G__sout,"  -E -E : Exit process at error and uncaught exception\n");
#endif
      G__more(G__sout,"  -f [file] : set break file\n");
      G__more(G__sout,"  -F [assignement] : set global variable\n");
      G__more(G__sout,"  -G [tracedmp] : dump exec trace into file\n");
      G__more(G__sout,"* -H[1-100] : level of header inclusion activated for dictionary generation\n");
      G__more(G__sout,"  -i : interactively return undefined symbol value\n");
      G__more(G__sout,"  -I [includepath] : set include file search path\n");
#ifndef G__OLDIMPLEMENTATION1525
      G__more(G__sout,"* -j [0|1]: Create multi-thread safe DLL(experimental)\n");
#endif
      G__more(G__sout,"  -J[0-4] : Display nothing(0)/error(1)/warning(2)/note(3)/all(4)\n");
      /* G__more(G__sout,"  -k : function key on\n"); */
      G__more(G__sout,"  -K : C mode\n");
#ifdef G__SHAREDLIB
      G__more(G__sout,"  -l [dynamiclinklib] : link dynamic link library\n");
#endif
      G__more(G__sout,"  -m : Support ISO-8859-x Eurpoean char set (disabling multi-byte char)\n");
      G__more(G__sout,"* -M [newdelmask] : operator new/delete mask for precompiled interface method\n");
      G__more(G__sout,"* -n [linkname] : Specify precompiled interface method filename\n");
      G__more(G__sout,"* -N [DLL_name] : Specify DLL interface method name\n");
      G__more(G__sout,"  -O [0~4] : Loop compiler on(1~5) off(0). Default on(4)\n");
      G__more(G__sout,"  -p : use preprocessor prior to interpretation\n");
      G__more(G__sout,"  -q [security] : Set security level(default 0)\n");
      G__more(G__sout,"  -Q : Quiet mode (no prompt)\n");
      G__more(G__sout,"  -r : revision and linked function/global info\n");
      G__more(G__sout,"  -R : display input file at break point\n");
      G__more(G__sout,"  -s : step execution mode\n");
      G__more(G__sout,"  -S : step execution mode, First stop in main()\n");
      G__more(G__sout,"  -t : trace execution\n");
      G__more(G__sout,"  -T : trace from pre-run\n");
#ifndef G__OLDIMPLEMENTATION411
      G__more(G__sout,"  -u [undefout] : listup undefined typenames\n");
#endif
#ifndef G__OLDIMPLEMENTATION1451
      G__more(G__sout,"* -U [dir] : directory to disable interface method generation\n");
#endif
      G__more(G__sout,"* -V : Generate symbols for non-public member with -c option\n");
      G__more(G__sout,"  -v : Bytecode compiler debug mode\n");
      G__more(G__sout,"  -X [readlinedumpfile] : Execute readline dumpfile\n");
      G__more(G__sout,"  -x 'main() {...}' : Execute argument as source code\n");
#ifndef G__OLDIMPLEMENTATION1285
      G__more(G__sout,"  -Y [0|1]: ignore std namespace (default=1:ignore)\n"); 
#endif
      G__more(G__sout,"  -Z [0|1]: automatic loading of standard header files with DLL\n"); 
      G__more(G__sout,"  --'command': Execute interactive command and terminate Cint\n"); 
      G__more(G__sout,"suboptions\n");
      G__more(G__sout,"  +V : turn on class title comment mode for following source fies\n");
      G__more(G__sout,"  -V : turn off class title comment mode for following source fies\n");
      G__more(G__sout,"  +P : turn on preprocessor for following source files\n");
      G__more(G__sout,"  -P : turn off preprocessor for following source files\n");
      G__more(G__sout,"* +STUB : stub function header begin\n");
      G__more(G__sout,"* -STUB : stub function header end\n");
      G__more(G__sout,"sourcefiles\n");
      G__more(G__sout,"  Any valid C/C++ source or header files\n");
      G__more(G__sout,"EXAMPLES\n");
      G__more(G__sout,"  $ cint prog.c main.c\n");
      G__more(G__sout,"  $ cint -S prog.c main.c\n");
      G__more(G__sout,"\n");
      if(G__key!=0) system("key .cint_key -l execute");
#endif
      return(EXIT_FAILURE);
    }
  }

  /*************************************************************
   * Signal handling
   *************************************************************/
#ifndef G__ROOT  /* This is only defined for ROOT */
#ifndef G__DONT_CATCH_SIGINT
  signal(SIGINT,G__breakkey);
#endif /* G__DONT_CACH_SIGINT */
  if(G__coredump==0) {
    signal(SIGFPE,G__floatexception);
    signal(SIGSEGV,G__segmentviolation);
#ifdef SIGEMT
    signal(SIGEMT,G__outofmemory);
#endif
#ifdef SIGBUS
    signal(SIGBUS,G__buserror);
#endif
  }
#ifndef G__OLDIMPLEMENTATION1946
  else if(G__coredump>=2) {
    signal(SIGFPE,G__errorexit);
    signal(SIGSEGV,G__errorexit);
#ifdef SIGEMT
    signal(SIGEMT,G__errorexit);
#endif
#ifdef SIGBUS
    signal(SIGBUS,G__errorexit);
#endif
  }
#endif
#endif /* G__ROOT */


#ifndef G__OLDIMPLEMENATTION175
  if(0==G__sizep2memfunc) G__sizep2memfunc=G__SHORTALLOC*2+G__LONGALLOC;
#endif


  /* normal option */


#ifdef G__DUMPFILE
  /*************************************************************
   * Open dumpfile if specified.
   *************************************************************/
  /* G__dumpfile=fopen("/dev/null","w"); */
  if(strcmp(dumpfile,"")!=0) G__dumpfile=fopen(dumpfile,"w");
#ifdef G__MEMTEST
  /* else G__dumpfile = G__memhist; */
#endif
#endif

  if(G__NOLINK > G__globalcomp) {
    G__gen_cppheader((char*)NULL);
  }


  /*************************************************************
   * pre run , read whole ifuncs to allocate global variables and
   * make ifunc table.
   *************************************************************/
  while( (G__MAINEXIST!=G__ismain&&(optind<argc)) || xfileflag ) {

    if(xfileflag) {
#ifndef G__OLDIMPLEMENATTION1919
      FILE *tmpf = tmpfile();
      if(tmpf) {
	fprintf(tmpf,"%s\n",argv[xfileflag]);
	xfileflag=0;
	fseek(tmpf,0L,SEEK_SET);
	if(G__loadfile_tmpfile(tmpf) || G__eof==2) {
	  /* file not found or unexpected EOF */
	  if(G__CPPLINK==G__globalcomp||G__CLINK==G__globalcomp) {
	    G__cleardictfile(-1);
	  }
	  G__scratch_all();
	  return(EXIT_FAILURE);
	}
	continue;
      }
      else {
	xfileflag=0;
      }
#else
      sprintf(sourcefile,G__xfile);
      xfileflag=0;
#endif
    }
    else {
      strcpy(sourcefile,argv[optind]);
      optind++;
    }

    if(strcmp(sourcefile,"+P")==0) {
      G__cpp=1;
      continue;
    }
    else if(strcmp(sourcefile,"-P")==0) {
      G__cpp=0;
      continue;
    }
#ifdef G__FONS_COMMENT
    else if(strcmp(sourcefile,"+V")==0) {
      G__fons_comment = 1;
      continue;
    }
    else if(strcmp(sourcefile,"-V")==0) {
      G__fons_comment = 0;
      continue;
    }
#endif
    else if(strcmp(sourcefile,"+STUB")==0) {
#ifndef G__OLDIMPLEMENTATION2024
      stubbegin.ptype = (char*)G__PVOID;
#endif
      G__store_dictposition(&stubbegin);
      continue;
    }
    else if(strcmp(sourcefile,"-STUB")==0) {
      G__set_stubflags(&stubbegin);
      continue;
    }

    if(G__NOLINK > G__globalcomp) {
      G__gen_cppheader(sourcefile);
    }

    if(G__loadfile(sourcefile)<0 || G__eof==2) {
      /* file not found or unexpected EOF */
#ifndef G__OLDIMPLEMENTATION1197
      if(G__CPPLINK==G__globalcomp||G__CLINK==G__globalcomp) {
	G__cleardictfile(-1);
      }
#endif
      G__scratch_all();
      return(EXIT_FAILURE);
    }
    if(G__return>G__RETURN_NORMAL) {
      G__scratch_all();
      return(EXIT_SUCCESS);
    }
  }

#ifndef G__OLDIMPLEMENTATION1996
  if(icom) {
    int more = 0;
    G__redirect_on();
    G__init_process_cmd();
    G__process_cmd(icom, "cint>", &more,(int*)NULL,(G__value*)NULL);
    G__scratch_all();
    return(EXIT_SUCCESS);
  }
#endif

#ifndef G__OLDIMPLEMENTATION953
  if (G__afterparse_hook) G__afterparse_hook ();
#endif

#ifndef G__OLDIMPLEMENTATION1162
  if(G__security_error) {
    G__fprinterr(G__serr,"Warning: Error occured during reading source files\n");
  }
#endif

#ifndef G__PHILIPPE30
  G__gen_extra_include();
#endif

  if(G__CPPLINK == G__globalcomp) { /* C++ header */
    if(G__steptrace||G__stepover) while(0==G__pause()) ;
    G__gen_cpplink();
#if !defined(G__ROOT) && !defined(G__D0)
    G__scratch_all();
#endif
#ifndef G__OLDIMPLEMENTATION996
    if(G__security_error) {
#ifndef G__OLDIMPLEMENTATION1162
      G__fprinterr(G__serr,"Warning: Error occured during dictionary source generation\n");
#endif
#ifndef G__OLDIMPLEMENTATION1197
      G__cleardictfile(-1);
#endif
      return(-1);
    }
#endif
#ifndef G__OLDIMPLEMENTATION1197
    G__cleardictfile(EXIT_SUCCESS);
#endif
    return(EXIT_SUCCESS);
  }
  else if(G__CLINK == G__globalcomp) { /* C header */
    if(G__steptrace||G__stepover) while(0==G__pause()) ;
    G__gen_clink();
#if !defined(G__ROOT) && !defined(G__D0)
    G__scratch_all();
#endif
#ifndef G__OLDIMPLEMENTATION996
    if(G__security_error) {
#ifndef G__OLDIMPLEMENTATION1162
      G__fprinterr(G__serr,"Warning: Error occured during dictionary source generation\n");
#endif
#ifndef G__OLDIMPLEMENTATION1197
      G__cleardictfile(-1);
#endif
      return(-1);
    }
#endif
#ifndef G__OLDIMPLEMENTATION1197
    G__cleardictfile(EXIT_SUCCESS);
#endif
    return(EXIT_SUCCESS);
  }


#ifdef G__OLDIMPLEMENTATION487
#ifdef G__AUTOCOMPILE
  /*************************************************************
   * if '#pragma compile' appears in source code.
   *************************************************************/
  if(G__fpautocc) G__autocc();
#endif
#endif

  optind--;
  if(G__debugtrace!=0) G__fprinterr(G__serr,"PRE-RUN END\n");

  /*************************************************************
   * set debug conditon after prerun
   *************************************************************/
  if(G__breakline[0]) {
    G__setbreakpoint(G__breakline,G__breakfile);
  }
  G__eof = 0;
  G__prerun = 0;
  G__debug=G__debugtrace;
  G__step=G__steptrace;
  if(stepfrommain) {
    G__step=1;
  }
  G__setdebugcond();


  /*************************************************************
   * Forceassignment is given by -F command line option.
   *    cint -F(expr1,expr2,...)
   * Expressions are evaluated after pre-RUN.
   *************************************************************/
  if(forceassignment) {
    G__calc_internal(forceassignment);
  }



  /*************************************************************
   * If G__main() is called from G__init_cint() then return
   * from G__main().  G__calc() has to be explicitly called
   * to start interpretation.
   *************************************************************/
#ifndef G__OLDIMPLEMENTATION563
  G__cintready=1; /* set ready flag for embedded use */
#endif
  switch(G__othermain) {
  case 2:
    if(G__NOMAIN==G__ismain) return(EXIT_SUCCESS);
    break;
  case 3: /* for drcc */
    return(EXIT_SUCCESS);
  }

#ifndef G__OLDIMPLEMENTATION411
  if(G__fpundeftype) {
    fclose(G__fpundeftype);
    G__fpundeftype=(FILE*)NULL;
    return(EXIT_SUCCESS);
  }
#endif


  /*************************************************************
   * Interpretation begin. If there were main() in input file,
   * main() is used as an entry. If not, G__pause() is called and
   * wait for user input.
   *************************************************************/

  if(G__MAINEXIST==G__ismain) {
#ifndef G__OLDIMPLEMENTATION575
    struct G__param para;
#endif
    /*********************************/
    /* If main() exists , execute it */
    /*********************************/

    /********************************
     * Set interpreter argument
     ********************************/
    for(iarg=optind;iarg<argc;iarg++) {
      G__argpointer[iarg-optind] = (long)argv[iarg];
    }
    while(iarg<G__MAXARG) {
      G__argpointer[iarg-optind] = (long)NULL;
      ++iarg;
    }

#ifndef G__OLDIMPLEMENTATION575
    /********************************
     * Call main(argc,argv)
     ********************************/
    if(G__breaksignal) G__fprinterr(G__serr,"\nCALL main()\n");
    sprintf(temp,"main");
    para.paran=2;
    G__letint(&para.para[0],'i',argc-optind);
    para.para[0].tagnum= -1;
    para.para[0].typenum= -1;
    G__letint(&para.para[1],'C',(long)G__argpointer);
    para.para[1].tagnum= -1;
    para.para[1].typenum= -1;
    para.para[1].obj.reftype.reftype = G__PARAP2P;
    G__interpret_func(&result,temp,&para,G__HASH_MAIN,G__p_ifunc
		      ,G__EXACT,G__TRYNORMAL);

#else /* ON575 */
    sprintf(temp,"main(%d,%ld)",argc-optind,G__argpointer);

    /********************************
     * Call main(argc,argv)
     ********************************/
    if(G__breaksignal) G__fprinterr(G__serr,"\nCALL main()\n");
    result=G__calc_internal(temp);
#endif /* ON575 */

    if(0==result.type) result=G__null;

    /*************************************
     * After main() function, break if
     * step or break mode
     *************************************/
#ifndef G__OLDIMPLEMENTATION743
    if(G__breaksignal || G__RETURN_EXIT1==G__return) {
      G__return=G__RETURN_NON;
#else
    if(G__breaksignal) {
#endif
      G__break=0;
      G__setdebugcond();
      G__fprinterr(G__serr,"!!! return from main() function\n");
#ifndef G__OLDIMPLEMENTATION2158
#ifdef SIGALRM
      if(G__RETURN_EXIT1==G__return) {
	G__fprinterr(G__serr,
	    "Press return or process will be terminated in %dsec by timeout\n"
		     ,G__TIMEOUT);
	signal(SIGALRM,G__timeout);
	alarm(G__TIMEOUT);
      }
#endif
#endif
      G__pause();
#ifndef G__OLDIMPLEMENTATION2158
#ifdef SIGALRM
      if(G__RETURN_EXIT1==G__return) {
	alarm(0);
	G__fprinterr(G__serr,"Time out cancelled\n");
      }
#endif
#endif
      G__pause();
    }
    if(G__stepover) {
      G__step=0;
      G__setdebugcond();
    }

    /*************************************************************
     * If G__main() is called from G__init_cint() then return
     * from G__main() before destroying data.
     *************************************************************/
    if(G__othermain==2) {
      return(G__int(result));
    }

    /*************************************
     * atexit()
     * global destruction
     * file close
     *************************************/
    G__interpretexit();


    return(G__int(result));
  }
#ifdef G__TCLTK
  else if(G__TCLMAIN==G__ismain) {
    if(G__othermain==2) {
      return(EXIT_SUCCESS);
    }
    G__interpretexit();
    return(EXIT_SUCCESS);
  }
#endif
  else {
    /*************************************
     * If no main() ,
     * Print out message and
     * invoke debugger front end
     *************************************/

    /*************************************
     * print out message
     *************************************/
    if(0==G__quiet) { /* ON1078 */
      G__cintrevision(G__sout);
      fprintf(G__sout,"No main() function found in given source file. Interactive interface started.\n");
#ifndef G__OLDIMPLEMENTATION1795
      switch(G__ReadInputMode()) {
      case G__INPUTROOTMODE:
	fprintf(G__sout,"'?':help, '.q':quit, 'statement','{statements;}' or '.p [expr]' to evaluate\n");
	break;
      case G__INPUTCXXMODE:
	fprintf(G__sout,"'?':help, '.q':quit, 'statement;','{statements;}' or '.p [expr]' to evaluate\n");
	break;
      case G__INPUTCINTMODE:
      default:
	fprintf(G__sout,"'h':help, 'q':quit, '{statements;}' or 'p [expr]' to evaluate\n");
	break;
      }
#else
#ifdef G__ROOT
      fprintf(G__sout,"'?' for help, '.q' to quit, '{ statements }' or '.p [expression]' to evaluate\n");
#else
      fprintf(G__sout,"'h' for help, 'q' to quit, '{ statements }' or 'p [expression]' to evaluate\n");
#endif
#endif
    } /* ON1078 */


    /*************************************
    * invoke debugger front end
    *************************************/
    while(1) {
      G__pause();
#ifndef G__OLDIMPLEMENTATION743
      if(G__return>G__RETURN_NORMAL&&G__RETURN_EXIT1!=G__return) {
	G__return=G__RETURN_NON;
#else
      if(G__return>G__RETURN_NORMAL) {
#endif
	G__scratch_all();
	return(EXIT_SUCCESS);
      }
    }
  }

}



/**************************************************************************
* G__init_globals()
*
*  Explicit initialization of all necessary global variables
**************************************************************************/
int G__init_globals()
{
  int i;
#ifndef G__OLDIMPLEMENTATION1599
  if (G__init) return(1);
  G__init = 1;
#endif
  /* G__p_ifunc = &G__ifunc ; */

  G__exec_memberfunc = 0;
  G__memberfunc_tagnum = -1;
  G__memberfunc_struct_offset=0;

  G__atpause = NULL ;

#ifdef G__ASM
  /***************************************************
   * loop compiler variables initialization
   ***************************************************/
  G__asm_name_p=0;
#if defined(G__ROOT)
  G__asm_loopcompile=4;
#else
  G__asm_loopcompile=4;
#endif
#ifndef G__OLDIMPLEMENTATION1155
  G__asm_loopcompile_mode = G__asm_loopcompile; 
#endif
#ifdef G__ASM_WHOLEFUNC
  G__asm_wholefunction = G__ASM_FUNC_NOP;
#endif
#ifndef G__OLDIMPLEMENTATION517
  G__asm_wholefunc_default_cp=0;
#endif
  G__abortbytecode();
  G__asm_dbg=0;
  G__asm_cp=0;               /* compile program counter */
  G__asm_dt=G__MAXSTACK-1;   /* constant data address */
#ifdef G__ASM_IFUNC
  G__asm_inst = G__asm_inst_g;
#ifndef G__OLDIMPLEMENTATION2116
  G__asm_instsize = 0; /* 0 means G__asm_inst is not resizable */
#endif
  G__asm_stack = G__asm_stack_g;
  G__asm_name = G__asm_name_g;
  G__asm_name_p = 0;
#endif
#endif

  G__debug=0;            /* trace input file */
  G__breakdisp=0;        /* display input file at break point */
  G__break=0;            /* break flab */
  G__step=0;             /* step execution flag */
  G__charstep=0;         /* char step flag */
  G__break_exit_func=0;  /* break at function exit */
  /* G__no_exec_stack = -1; */ /* stack for G__no_exec */
  G__no_exec=0;          /* no execution(ignore) flag */
  G__no_exec_compile=0;
  G__var_type='p';      /* variable decralation type */
  G__var_typeB='p';
  G__prerun=0;           /* pre-run flag */
  G__funcheader=0;       /* function header mode */
  G__return=G__RETURN_NON; /* return flag of i function */
  /* G__extern_global=0;    number of globals defined in c function */
  G__disp_mask=0;        /* temporary read count */
  G__temp_read=0;        /* temporary read count */
  G__switch=0;           /* switch case control flag */
  G__mparen=0;           /* switch case break nesting control */
  G__eof_count=0;        /* end of file error flag */
  G__ismain=G__NOMAIN;   /* is there a main function */
  G__globalcomp=G__NOLINK;  /* make compiled func's global table */
  G__store_globalcomp=G__NOLINK;
  G__globalvarpointer=G__PVOID; /* make compiled func's global table */
  G__nfile=0;
  G__key=0;              /* user function key on/off */

  G__xfile[0]='\0';
  G__tempc[0]='\0';

  G__doingconstruction=0;

#ifdef G__DUMPFILE
  G__dumpfile=NULL;
  G__dumpspace=0;
#endif

  G__nobreak=0;

#ifdef G__FRIEND
  G__friendtagnum = -1;
#endif
  G__def_struct_member=0;
#ifndef G__OLDIMPLEMENTATION440
  G__tmplt_def_tagnum = -1;
#endif
  G__def_tagnum = -1;
  G__tagdefining = -1;
  G__tagnum = -1;
  G__typenum = -1;
  G__iscpp = 1;
  G__cpplock=0;
  G__clock=0;
  G__constvar = 0;
#ifndef G__OLDIMPLEMENTATION1250
  G__isexplicit = 0;
#endif
  G__unsigned = 0;
  G__ansiheader = 0;
  G__enumdef=0;
  G__store_struct_offset=0;
  G__decl=0;
#ifdef G__OLDIMPLEMENTATION435
  G__allstring=0;
#endif
  G__longjump=0;
  G__coredump=0;
  G__definemacro=0;

  G__noerr_defined=0;

  G__static_alloc=0;
  G__func_now = -1;
  G__func_page = 0;
  G__varname_now=NULL;
  G__twice=0;
  /* G__othermain; */
  G__cpp=0;
#ifndef G__OLDIMPLEMENTATOIN136
  G__include_cpp=0;
#endif
  G__ccom[0]='\0';
  G__cppsrcpost[0]='\0';
  G__csrcpost[0]='\0';
  G__cpphdrpost[0]='\0';
  G__chdrpost[0]='\0';
  G__dispsource=0;
  G__breaksignal=0;

  G__bitfield=0;

  G__atexit = NULL;

#ifdef G__SIGNAL
  G__SIGINT = NULL;
  G__SIGILL = NULL;
  G__SIGFPE = NULL;
  G__SIGABRT = NULL;
  G__SIGSEGV = NULL;
  G__SIGTERM = NULL;
#ifdef SIGHUP
  G__SIGHUP = NULL;
#endif
#ifdef SIGQUIT
  G__SIGQUIT = NULL;
#endif
#ifdef SIGTSTP
  G__SIGTSTP = NULL;
#endif
#ifdef SIGTTIN
  G__SIGTTIN = NULL;
#endif
#ifdef SIGTTOU
  G__SIGTTOU = NULL;
#endif
#ifdef SIGALRM
  G__SIGALRM = NULL;
#endif
#ifdef SIGUSR1
  G__SIGUSR1 = NULL;
#endif
#ifdef SIGUSR2
  G__SIGUSR2 = NULL;
#endif
#endif

#ifdef G__SHAREDLIB
  G__allsl=0;
#endif

  G__p_tempbuf = &G__tempbuf;
  G__tempbuf.level = 0;
  G__tempbuf.obj = G__null;
  G__tempbuf.prev = NULL;
  G__templevel = 0;

  G__reftype=G__PARANORMAL;

  G__access=G__PUBLIC;

  G__stepover=0;

  G__newarray.next = (struct G__newarylist *)NULL;
  /* G__newarray.point = 0; */

  /*************************************
   * Bug fix for SUNOS
   *************************************/
  G__ifile.fp = NULL;
  G__ifile.name[0] = '\0';
  G__ifile.line_number = 0;
  G__ifile.filenum = -1;

  G__steptrace = 0;
  G__debugtrace = 0;

  G__ipathentry.pathname = (char *)NULL;
  G__ipathentry.next = (struct G__includepath *)NULL;

  G__mfp=NULL;

  G__deffuncmacro.hash = 0;
  G__deffuncmacro.callfuncmacro.call_fp = (FILE *)NULL;
#ifndef G__OLDIMPLEMENTATION1179
  G__deffuncmacro.callfuncmacro.call_filenum = -1;
#endif
  G__deffuncmacro.callfuncmacro.next = (struct G__Callfuncmacro *)NULL;
  G__deffuncmacro.next = (struct G__Deffuncmacro *)NULL;

  G__cintsysdir[0]='*';
  G__cintsysdir[1]='\0';

  G__p_local = NULL ;    /* local variable array */
#ifndef G__OLDIMPLEMENTATION686
  G__globalusingnamespace.basen = 0;
#endif

  G__global.varlabel[0][0]=0;  /* initialize global variable */
  G__global.next=NULL;         /* start from one G__var_table */
  G__global.ifunc = (struct G__ifunc_table *)NULL;
  G__global.prev_local = (struct G__var_array *)NULL;
  G__global.prev_filenum = -1;
  G__global.tagnum = -1;
#ifndef G__OLDIMPLEMENTATION2053
  G__global.allvar = 0;
#endif
#ifndef G__OLDIMPLEMENTATION1543
  {
    int ix;
    for(ix=0;ix<G__MEMDEPTH;ix++) {
#ifndef G__OLDIMPLEMENTATION2053
      G__global.hash[ix] = 0;
#endif
      G__global.varnamebuf[ix] = (char*)NULL;
    }
  }
#endif
  G__cpp_aryconstruct=0;
  G__cppconstruct=0;

  G__newtype.alltype=0;        /* initialize typedef table */

  G__breakfile[0]='\0';        /* default breakfile="" */

  G__assertion[0]='\0';        /* initialize debug assertion */

  /**************************************************************
   * initialize constant values
   **************************************************************/
  G__null.obj.d = 0.0;
  G__letint(&G__null,'\0',0);   /* set null value */
  G__null.tagnum = -1;
  G__null.typenum = -1;
  G__null.ref = 0;
#ifndef G__OLDIMPLEMENTATION1259
  G__null.isconst = G__CONSTVAR;
#endif

  G__one.obj.d = 0.0;
  G__letint(&G__one,'i',1); /* set default value */
  G__one.tagnum = -1;
  G__one.typenum = -1;
  G__one.ref = 0;
#ifndef G__OLDIMPLEMENTATION1259
  G__one.isconst = G__CONSTVAR;
#endif

  G__letint(&G__start,'a',G__SWITCH_START);   /* set start value */
  G__start.tagnum = -1;
  G__start.typenum = -1;
  G__start.ref = 0;
#ifndef G__OLDIMPLEMENTATION1259
  G__start.isconst = G__CONSTVAR;
#endif

  G__letint(&G__default,'z',G__SWITCH_DEFAULT); /* set default value */
  G__default.tagnum = -1;
  G__default.typenum = -1;
  G__default.ref = 0;
#ifndef G__OLDIMPLEMENTATION1259
  G__default.isconst = G__CONSTVAR;
#endif

  G__letint(&G__block_break,'Z',G__BLOCK_BREAK); /* set default value */
  G__block_break.tagnum = -1;
  G__block_break.typenum = -1;
  G__block_break.ref = 0;
#ifndef G__OLDIMPLEMENTATION1259
  G__block_break.isconst = G__CONSTVAR;
#endif

  G__letint(&G__block_continue,'Z',G__BLOCK_CONTINUE); /* set default value */
  G__block_continue.tagnum = -1;
  G__block_continue.typenum = -1;
  G__block_continue.ref = 0;
#ifndef G__OLDIMPLEMENTATION1259
  G__block_continue.isconst = G__CONSTVAR;
#endif

  G__letint(&G__block_goto,'Z',G__BLOCK_BREAK); /* set default value */
  G__block_goto.tagnum = -1;
  G__block_goto.typenum = -1;
  G__block_goto.ref = 1; /* identical to G__block_break except for this */
#ifndef G__OLDIMPLEMENTATION1259
  G__block_goto.isconst = G__CONSTVAR;
#endif
  G__gotolabel[0]='\0';

#ifndef G__OLDIMPLEMENTATION954
  G__exceptionbuffer = G__null;
#endif

  G__virtual=0;
#ifdef G__AUTOCOMPILE
  G__fpautocc=(FILE*)NULL;
  G__compilemode=1;
#endif
  G__typedefnindex = 0;
  G__oprovld = 0;
  G__p2arylabel[0]=0;

  G__interactive=0;
  G__interactivereturnvalue=G__null;

#ifdef G__TEMPLATECLASS
  G__definedtemplateclass.next = (struct G__Definedtemplateclass *)NULL;
  G__definedtemplateclass.def_para = (struct G__Templatearg *)NULL;
  G__definedtemplateclass.def_fp = (FILE*)NULL;
#ifdef G__TEMPLATEMEMFUNC
  G__definedtemplateclass.memfunctmplt.next
    =(struct G__Definedtemplatememfunc*)NULL;
#endif /* G__TEMPLATEMEMFUNC */
#ifndef G__OLDIMPLEMENTATION682
  G__definedtemplateclass.parent_tagnum = -1;
#endif
#ifndef G__OLDIMPLEMENTATION691
  G__definedtemplateclass.isforwarddecl=0;
  G__definedtemplateclass.instantiatedtagnum=(struct G__IntList*)NULL;
#endif
#ifndef G__OLDIMPLEMENTATION1587
  G__definedtemplateclass.specialization=(struct G__Definedtemplateclass*)NULL;
  G__definedtemplateclass.spec_arg=(struct G__Templatearg*)NULL;
#endif

#ifdef G__TEMPLATEFUNC
  G__definedtemplatefunc.next = (struct G__Definetemplatefunc *)NULL;
  G__definedtemplatefunc.def_para = (struct G__Templatearg*)NULL;
  G__definedtemplatefunc.name = (char*)NULL;
#ifndef G__OLDIMPLEMENTATION727
  for(i=0;i<G__MAXFUNCPARA;i++) {
    G__definedtemplatefunc.func_para.ntarg[i]=(int*)NULL;
    G__definedtemplatefunc.func_para.nt[i]=0;
  }
#endif
#endif /* G__TEMPLATEFUNC */
#endif /* G__TEMPLATECLASS */

  G__isfuncreturnp2f=0;

  G__typepdecl=0;

  G__macroORtemplateINfile=0;

  G__macro_defining=0;

  G__nonansi_func = 0;

  G__parseextern=0;

  G__istrace = 0;

  G__pbreakcontinue = (struct G__breakcontinue_list*)NULL;


#ifdef G__FONS_COMMENT
  G__fons_comment = 0;
  G__setcomment = (char*)NULL;
#endif

#ifdef G__SECURITY
  G__security = G__SECURE_LEVEL0;
  G__castcheckoff=0;
  G__security_error = G__NOERROR;
  G__max_stack_depth = G__DEFAULT_MAX_STACK_DEPTH;
#endif

  G__preprocessfilekey.keystring=(char*)NULL;
  G__preprocessfilekey.next=(struct G__Preprocessfilekey*)NULL;

  G__precomp_private=0;

  /* The first entry in the const string is a blank string
   * which is never used */
  G__conststringlist.string = "";
  G__conststringlist.hash = 0;
  G__conststringlist.prev = (struct G__ConstStringList*)NULL;
  G__plastconststring = &G__conststringlist;

  /* $xxx search function , set default. */
#ifdef G__ROOT
  if (!G__GetSpecialObject) G__GetSpecialObject = G__getreserved;
#else
  G__GetSpecialObject = G__getreserved;
#endif

#if defined(_MSC_VER) && (_MSC_VER<=1100)
#define G__OLDIMPLEMENTATION1423
#endif

#ifndef G__OLDIMPLEMENTATION1423
#if defined(G__ROOT) && !defined(G__EXPERIMENTAL1423)
  G__is_operator_newdelete = G__MASK_OPERATOR_NEW |
                             G__MASK_OPERATOR_DELETE |
                             G__NOT_USING_2ARG_NEW ;
#else
  G__is_operator_newdelete = G__DUMMYARG_NEWDELETE 
                          /* | G__DUMMYARG_NEWDELETE_STATIC */
                             | G__NOT_USING_2ARG_NEW ;
#endif
#else /* 1423 */
#if defined(G__ROOT) 
  G__is_operator_newdelete = G__MASK_OPERATOR_NEW |
                             G__MASK_OPERATOR_DELETE |
                             G__NOT_USING_2ARG_NEW ;
#elif defined(G__BORLAND)
  G__is_operator_newdelete = G__NOT_USING_2ARG_NEW ;
#elif (__SUNPRO_C>=1280)
  G__is_operator_newdelete = G__MASK_OPERATOR_NEW |
                             G__MASK_OPERATOR_DELETE |
                             G__NOT_USING_2ARG_NEW ;
#else
  G__is_operator_newdelete = 0;
#endif
#endif /* 1423 */

#ifndef G__OLDIMPLEMENTATION411
  G__fpundeftype=(FILE*)NULL;
#endif

#ifndef G__OLDIMPLEMENTATION1207
  G__ispermanentsl=0;
#endif

#ifndef G__OLDIMPLEMENTATION1593
  G__boolflag=0;
#endif

  return(0);
}

#ifndef G__OLDIMPLEMENTATION1689
void G__initcxx(); 
#endif

#ifndef G__OLDIMPLEMENTATION893
/******************************************************************
* G__platformMacro
******************************************************************/
void G__platformMacro() 
{
  char temp[G__ONELINE];
#ifdef G__CINTVERSION
  sprintf(temp,"G__CINTVERSION=%ld",(long)G__CINTVERSION); G__add_macro(temp);
#endif
  /***********************************************************************
   * operating system
   ***********************************************************************/
#if defined(__linux__)  /* Linux */
  sprintf(temp,"G__LINUX=%ld",(long)__linux__); G__add_macro(temp);
#elif defined(__linux) 
  sprintf(temp,"G__LINUX=%ld",(long)__linux); G__add_macro(temp);
#elif defined(linux)
  sprintf(temp,"G__LINUX=%ld",(long)linux); G__add_macro(temp);
#endif
#ifdef __FreeBSD__   /* FreeBSD */
  sprintf(temp,"G__FBSD=%ld",(long)__FreeBSD__); G__add_macro(temp);
#endif
#ifdef __OpenBSD__   /* OpenBSD */
  sprintf(temp,"G__OBSD=%ld",(long)__OpenBSD__); G__add_macro(temp);
#endif
#ifdef __hpux        /* HP-UX */
  sprintf(temp,"G__HPUX=%ld",(long)__hpux); G__add_macro(temp);
#endif
#ifdef __sun         /* SunOS and Solaris */
  sprintf(temp,"G__SUN=%ld",(long)__sun); G__add_macro(temp);
#endif
#ifdef _WIN32        /* Windows 32bit */
  sprintf(temp,"G__WIN32=%ld",(long)_WIN32); G__add_macro(temp);
#endif
#ifdef _WINDOWS_     /* Windows */
  sprintf(temp,"G__WINDOWS=%ld",(long)_WINDOWS_); G__add_macro(temp);
#endif
#ifdef __APPLE__     /* Apple MacOS X */
  sprintf(temp,"G__APPLE=%ld",(long)__APPLE__); G__add_macro(temp);
#endif
#ifdef __VMS         /* DEC/Compac VMS */
  sprintf(temp,"G__VMS=%ld",(long)__VMS); G__add_macro(temp);
#endif
#ifdef _AIX          /* IBM AIX */
  sprintf(temp,"G__AIX=%ld",(long)_AIX); G__add_macro(temp);
#endif
#ifdef __sgi         /* SGI IRIX */
  sprintf(temp,"G__SGI=%ld",(long)__sgi); G__add_macro(temp);
#endif
#if defined(__alpha) && !defined(__linux) && !defined(__linux__) && !defined(linux) /* DEC/Compac Alpha-OSF operating system */
  sprintf(temp,"G__ALPHA=%ld",(long)__alpha); G__add_macro(temp);
#endif
#ifdef __QNX__         /* QNX realtime OS */
  sprintf(temp,"G__QNX=%ld",(long)__QNX__); G__add_macro(temp);
#endif
  /***********************************************************************
   * compiler and library
   ***********************************************************************/
#ifdef G__MINGW /* Mingw */
  sprintf(temp,"G__MINGW=%ld",(long)G__MINGW); G__add_macro(temp);
#endif
#ifdef G__CYGWIN /* Cygwin */
  sprintf(temp,"G__CYGWIN=%ld",(long)G__CYGWIN); G__add_macro(temp);
#endif
#ifdef __GNUC__  /* gcc/g++  GNU C/C++ compiler major version */
  sprintf(temp,"G__GNUC=%ld",(long)__GNUC__); G__add_macro(temp);
#endif
#ifdef __GNUC_MINOR__  /* gcc/g++ minor version */
  sprintf(temp,"G__GNUC_MINOR=%ld",(long)__GNUC_MINOR__); G__add_macro(temp);
#endif
#if defined(__GNUC__) && defined(__GNUC_MINOR__)
  sprintf(temp,"G__GNUC_VER=%ld",(long)__GNUC__*1000+__GNUC_MINOR__); 
  G__add_macro(temp);
#endif
#ifdef __GLIBC__   /* GNU C library major version */
  sprintf(temp,"G__GLIBC=%ld",(long)__GLIBC__); G__add_macro(temp);
#endif
#ifdef __GLIBC_MINOR__  /* GNU C library minor version */
  sprintf(temp,"G__GLIBC_MINOR=%ld",(long)__GLIBC_MINOR__); G__add_macro(temp);
#endif
#ifdef __HP_aCC     /* HP aCC C++ compiler */
  sprintf(temp,"G__HP_aCC=%ld",(long)__HP_aCC); G__add_macro(temp);
#if __HP_aCC > 15000
  sprintf(temp,"G__ANSIISOLIB=1"); G__add_macro(temp);
#endif
#endif
#ifdef __SUNPRO_CC  /* Sun C++ compiler */
  sprintf(temp,"G__SUNPRO_CC=%ld",(long)__SUNPRO_CC); G__add_macro(temp);
#endif
#ifdef __SUNPRO_C   /* Sun C compiler */
  sprintf(temp,"G__SUNPRO_C=%ld",(long)__SUNPRO_C); G__add_macro(temp);
#endif
#ifdef G__VISUAL    /* Microsoft Visual C++ compiler */
  sprintf(temp,"G__VISUAL=%ld",(long)G__VISUAL); G__add_macro(temp);
#endif
#ifdef _MSC_VER     /* Microsoft Visual C++ version */
  sprintf(temp,"G__MSC_VER=%ld",(long)_MSC_VER); G__add_macro(temp);
#endif
#ifdef __SC__       /* Symantec C/C++ compiler */
  sprintf(temp,"G__SYMANTEC=%ld",(long)__SC__); G__add_macro(temp);
#endif
#ifdef __BORLANDC__ /* Borland C/C++ compiler */
  sprintf(temp,"G__BORLAND=%ld",(long)__BORLANDC__); G__add_macro(temp);
#endif
#ifdef __BCPLUSPLUS__  /* Borland C++ compiler */
  sprintf(temp,"G__BCPLUSPLUS=%ld",(long)__BCPLUSPLUS__); G__add_macro(temp);
#endif
#ifdef G__BORLANDCC5 /* Borland C/C++ compiler 5.5 */
  sprintf(temp,"G__BORLANDCC5=%ld",(long)505); G__add_macro(temp);
#endif
#ifdef __KCC        /* KCC  C++ compiler */
  sprintf(temp,"G__KCC=%ld",(long)__KCC); G__add_macro(temp);
#endif
#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER<810) /* icc and ecc C++ compilers */
  sprintf(temp,"G__INTEL_COMPILER=%ld",(long)__INTEL_COMPILER); G__add_macro(temp);
#endif
#ifndef _AIX
#ifdef G__OLDIMPLEMENTATION2095
#ifdef __xlC__ /* IBM xlC compiler */
  sprintf(temp,"G__XLC=%ld",(long)__xlC__); G__add_macro(temp); 
#endif
#endif
#ifdef __xlc__ /* IBM xlc compiler */
  sprintf(temp,"G__XLC=%ld",(long)__xlc__); G__add_macro(temp);
#ifndef G__OLDIMPLEMENTATION2095
  sprintf(temp,"G__GNUC=%ld",(long)3 /*__GNUC__*/); G__add_macro(temp);
  sprintf(temp,"G__GNUC_MINOR=%ld",(long)3 /*__GNUC_MINOR__*/); G__add_macro(temp);
#endif
#endif
#endif
#ifndef G__OLDIMPLEMENTATION1689
  G__initcxx(); 
#endif
  /***********************************************************************
   * micro processor
   ***********************************************************************/
#ifdef __hppa__ /* HP-PA , Hewlett Packard Precision Architecture */
  sprintf(temp,"G__hppa=%ld",(long)__hppa__); G__add_macro(temp);
#endif
#ifdef __i386__ /* Intel 386,486,586 */
  sprintf(temp,"G__i386=%ld",(long)__i386__); G__add_macro(temp);
#endif
#ifdef __i860__ /* Intel 860 */
  sprintf(temp,"G__i860=%ld",(long)__i860__); G__add_macro(temp);
#endif
#ifdef __i960__ /* Intel 960 */
  sprintf(temp,"G__i860=%ld",(long)__i960__); G__add_macro(temp);
#endif
#ifdef __ia64__ /* Intel Itanium */
  sprintf(temp,"G__ia64=%ld",(long)__ia64__); G__add_macro(temp);
#endif
#ifdef __m88k__ /* Motorola 88000 */
  sprintf(temp,"G__m88k=%ld",(long)__m88k__); G__add_macro(temp);
#endif
#ifdef __m68k__ /* Motorola 68000 */
  sprintf(temp,"G__m68k=%ld",(long)__m68k__); G__add_macro(temp);
#endif
#ifdef __ppc__  /* Motorola Power-PC */
  sprintf(temp,"G__ppc=%ld",(long)__ppc__); G__add_macro(temp);
#endif
#ifdef __PPC__  /* IBM Power-PC */
  sprintf(temp,"G__PPC=%ld",(long)__PPC__); G__add_macro(temp);
#endif
#ifdef __mips__ /* MIPS architecture */
  sprintf(temp,"G__mips=%ld",(long)__mips__); G__add_macro(temp);
#endif
#ifdef __alpha__ /* DEC/Compac Alpha */
  sprintf(temp,"G__alpha=%ld",(long)__alpha__); G__add_macro(temp);
#endif
#if defined(__sparc) /* Sun Microsystems SPARC architecture */
  sprintf(temp,"G__SPARC=%ld",(long)__sparc); G__add_macro(temp);
  sprintf(temp,"G__sparc=%ld",(long)__sparc); G__add_macro(temp);
#elif  defined(__sparc__)
  sprintf(temp,"G__SPARC=%ld",(long)__sparc__); G__add_macro(temp);
  sprintf(temp,"G__sparc=%ld",(long)__sparc__); G__add_macro(temp);
#endif
#ifdef __arc__  /* ARC architecture */
  sprintf(temp,"G__arc=%ld",(long)__arc__); G__add_macro(temp);
#endif
#ifdef __M32R__
  sprintf(temp,"G__m32r=%ld",(long)__M32R__); G__add_macro(temp);
#endif
#ifdef __sh__   /* Hitachi SH micro-controller */
  sprintf(temp,"G__sh=%ld",(long)__SH__); G__add_macro(temp);
#endif
#ifdef __arm__  /* ARM , Advanced Risk Machines */
  sprintf(temp,"G__arm=%ld",(long)__arm__); G__add_macro(temp);
#endif
#ifdef __s390__ /* IBM S390 */
  sprintf(temp,"G__s390=%ld",(long)__s390__); G__add_macro(temp);
#endif
  /***********************************************************************
   * application environment
   ***********************************************************************/
#ifdef G__ROOT
  sprintf(temp,"G__ROOT=%ld",(long)G__ROOT); G__add_macro(temp);
#endif
#ifdef G__NO_STDLIBS
  sprintf(temp,"G__NO_STDLIBS=%ld",(long)G__NO_STDLIBS); G__add_macro(temp);
#endif
#ifdef G__NATIVELONGLONG
  sprintf(temp,"G__NATIVELONGLONG=%ld",(long)G__NATIVELONGLONG); G__add_macro(temp);
#endif
  
  sprintf(temp,"int& G__cintv6=*(int*)(%ld);",(long)(&G__cintv6)); G__exec_text(temp);
}
#endif

/******************************************************************
* void G__set_stdio()
*
* Called by
*   G__main()
*
******************************************************************/
void G__set_stdio()
{
  char temp[G__ONELINE];

  G__globalvarpointer = G__PVOID ;

#ifndef G__OLDIMPLEMENTATION713
  G__intp_sout = G__sout;
  G__intp_serr = G__serr;
  G__intp_sin = G__sin;
#endif


  G__var_type='E';
#ifndef G__OLDIMPLEMENTATION713
  sprintf(temp,"stdout=(FILE*)(%ld)",(long)G__intp_sout);
#else
  sprintf(temp,"stdout=(FILE*)(%ld)",G__sout);
#endif
  G__getexpr(temp);

  G__var_type='E';
#ifndef G__OLDIMPLEMENTATION713
  sprintf(temp,"stderr=(FILE*)(%ld)",(long)G__intp_serr);
#else
  sprintf(temp,"stderr=(FILE*)(%ld)",G__serr);
#endif
  G__getexpr(temp);

  G__var_type='E';
#ifndef G__OLDIMPLEMENTATION713
  sprintf(temp,"stdin=(FILE*)(%ld)",(long)G__intp_sin);
#else
  sprintf(temp,"stdin=(FILE*)(%ld)",G__sin);
#endif
  G__getexpr(temp);

  G__definemacro=1;
#ifndef G__FONS31
  sprintf(temp,"EOF=%ld",(long)EOF); G__getexpr(temp);
  sprintf(temp,"NULL=%ld",(long)NULL); G__getexpr(temp);
#else
  sprintf(temp,"EOF=%d",EOF); G__getexpr(temp);
  sprintf(temp,"NULL=%d",NULL); G__getexpr(temp);
#endif
#ifdef G__SHAREDLIB
  sprintf(temp,"G__SHAREDLIB=1"); G__getexpr(temp);
#endif
#if defined(G__P2FCAST) || defined(G__P2FDECL)
  sprintf(temp,"G__P2F=1"); G__getexpr(temp);
#endif
#ifdef G__NEWSTDHEADER
  sprintf(temp,"G__NEWSTDHEADER=1"); G__getexpr(temp);
#endif
#ifndef G__OLDIMPLEMENTATION893
  G__platformMacro();
#endif
  G__definemacro=0;

#ifndef G__OLDIMPLEMENTATION1604
  /* G__constvar=G__CONSTVAR; G__var_type='g'; G__getexpr("TRUE=1"); */
  /* G__constvar=G__CONSTVAR; G__var_type='g'; G__getexpr("FALSE=0"); */
  G__constvar=G__CONSTVAR; G__var_type='g'; G__getexpr("true=1");
  G__constvar=G__CONSTVAR; G__var_type='g'; G__getexpr("false=0");
  G__constvar = 0;
#endif

#ifdef G__DUMPFILE
  G__globalvarpointer = (long)(&G__dumpfile);
  G__var_type='E';
  G__getexpr("G__dumpfile=0");
#endif
  G__globalvarpointer = G__PVOID;

  G__var_type = 'p';
  G__tagnum = -1;
  G__typenum = -1;
}

/******************************************************************
* void G__set_stdio_handle()
*
*
******************************************************************/
void G__set_stdio_handle(sout,serr,sin)
FILE* sout;
FILE* serr;
FILE* sin;
{
#ifndef G__OLDIMPLEMENTATION1812
  char temp[G__ONELINE];
#endif

  G__sout = G__stdout = sout;
  G__serr = G__stderr = serr;
  G__sin  = G__stdin  = sin;

#ifndef G__OLDIMPLEMENTATION1812
  G__var_type='E';
  sprintf(temp,"stdout=(FILE*)(%ld)",(long)G__intp_sout);
  G__getexpr(temp);

  G__var_type='E';
  sprintf(temp,"stderr=(FILE*)(%ld)",(long)G__intp_serr);
  G__getexpr(temp);

  G__var_type='E';
  sprintf(temp,"stdin=(FILE*)(%ld)",(long)G__intp_sin);
  G__getexpr(temp);
#else
  G__set_stdio();
#endif
}


#ifndef G__FONS1
/**************************************************************************
* G__cint_version()
*
* Called by
*    G__cintrevision
*
**************************************************************************/
char *G__cint_version()
{
  return(G__CINTVERSIONSTR);
  /* return "5.14.34, Mar 10 2000"; */
}
#endif

/**************************************************************************
* G__cintrevision()
*
* Called by
*    G__main()  '-r' option
*    G__main()  if main() is not found in interpreted source file
*
* revision print out
*
**************************************************************************/
int G__cintrevision(fp)
FILE *fp;
{
  fprintf(fp,"\n");
  fprintf(fp,"cint : C/C++ interpreter  (mailing list 'cint@root.cern.ch')\n");
  fprintf(fp,"   Copyright(c) : 1995~2005 Masaharu Goto (gotom@hanno.jp)\n");
  fprintf(fp,"   revision     : %s by M.Goto\n\n",G__cint_version());

#ifdef G__DEBUG
  fprintf(fp,"   MEMORY LEAK TEST ACTIVATED!!! MAYBE SLOW.\n\n");
#endif
  return(0);
}

#ifndef G__OLDIMPLEMENTATION1451
/**************************************************************************
* G__AddConstStringList()
*
**************************************************************************/
struct G__ConstStringList* G__AddConstStringList(current,str,islen)
struct G__ConstStringList* current;
char* str;
int islen;
{
  int itemp;
  struct G__ConstStringList* next;

  next=(struct G__ConstStringList*)malloc(sizeof(struct G__ConstStringList));

  next->string = (char*)malloc(strlen(str)+1);
  strcpy(next->string,str);

  if(islen) {
    next->hash = strlen(str);
  }
  else {
    G__hash(str,next->hash,itemp);
  }

  next->prev = current;

  return(next);
}

/**************************************************************************
* G__DeleteConstStringList()
*
**************************************************************************/
void G__DeleteConstStringList(current)
struct G__ConstStringList* current;
{
  struct G__ConstStringList* tmp;
  while(current) {
    if(current->string) free((void*)current->string);
    tmp = current->prev;
    free((void*)current);
    current = tmp;
  }
}

#endif

/**************************************************************************
* G__LockCpp()
*
**************************************************************************/
void G__LockCpp() 
{
  /* Same as option -A */
  G__cpplock=1;
  G__iscpp=1;
}

#ifndef G__OLDIMPLEMENTATION1815
/**************************************************************************
* G__SetCatchException()
*
**************************************************************************/
void G__SetCatchException(mode)
int mode;
{
  G__catchexception = mode;
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
