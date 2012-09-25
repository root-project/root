/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file init.c
 ************************************************************************
 * Description:
 *  Entry functions
 ************************************************************************
 * Copyright(c) 1995~2010  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "Api.h"
#include <list>
#include <typeinfo>
#include <string>

#if defined(G__NOSTUBS) && defined(__APPLE__)
#include <libgen.h> //needed for basename
#endif

#ifdef G__SHAREDLIB
extern std::list<G__DLLINIT>* G__initpermanentsl;
#endif //G__SHAREDLIB

extern "C" {

#ifndef __CINT__
void G__setCINTLIBNAME G__P((char *cintlib));
void G__setcopyflag G__P((int flag));
#endif

extern int G__ispermanentsl;

#ifndef G__OLDIMPLEMENTATION1817
#if defined(G__ROOT) && !defined(G__NATIVELONGLONG)
void G__cpp_setuplongif();
#endif
#endif

#include "rflx_gendict.h"

//______________________________________________________________________________
//
//  For C++ dictionary setup
//
struct G__setup_func_struct {
   std::string libname;
   G__incsetup func;
   bool inited;
   bool registered;

   G__setup_func_struct() : libname(), func(), inited(false), registered(false) {}
   G__setup_func_struct(const char *name, G__incsetup functions, bool isregistered) : libname(name), func(functions), inited(false), registered(isregistered) {}

};

//______________________________________________________________________________
static std::list<G__setup_func_struct> *G__setup_func_list;
static char G__memsetup_init;
typedef void G__parse_hook_t();
static G__parse_hook_t* G__afterparse_hook;
static G__parse_hook_t* G__beforeparse_hook;
void G__platformMacro();

//______________________________________________________________________________
void G__add_setup_func(const char* libname, G__incsetup func)
{
   if (!G__memsetup_init) {
      for (int i = 0; i < G__MAXSTRUCT; ++i) {
         G__struct.incsetup_memvar[i] = 0;
         G__struct.incsetup_memfunc[i] = 0;
         G__memsetup_init = 1;
      }
   }
   if ( !G__setup_func_list ) {
      G__setup_func_list = new std::list<G__setup_func_struct>;
   }

   /* if already in table: ignore (could also print warning) */
   std::list<G__setup_func_struct>::iterator begin = G__setup_func_list->begin();
   std::list<G__setup_func_struct>::iterator end = G__setup_func_list->end();
   std::list<G__setup_func_struct>::iterator j;
   for (j = begin ; j != end; ++j) {
      if ( j->libname == libname ) {
         return;
      }
   }
   G__setup_func_list->push_back( G__setup_func_struct( libname, func, true ) );

   ++G__nlibs;

   G__RegisterLibrary(func);
}

//______________________________________________________________________________
void G__remove_setup_func(const char* libname)
{
   std::list<G__setup_func_struct>::iterator begin = G__setup_func_list->begin();
   std::list<G__setup_func_struct>::iterator end = G__setup_func_list->end();
   std::list<G__setup_func_struct>::iterator i;

   for (i = begin ; i != end; ++i) {
       if ( i->libname == libname ) {
          G__UnregisterLibrary( i->func );
          G__setup_func_list->erase(i);
         --G__nlibs;
         return;
      }
   }
}

//______________________________________________________________________________
int G__call_setup_funcs()
{
   if (!G__ifunc.inited || !G__init) {
      // Veto any dictionary uploading until at least G__ifunc is initialized
      // (because it's initialization will be wipe out 'some' of the work done.
      return 0;
   }
   int k = 0;
   G__var_array* store_p_local = G__p_local; // changed by setupfuncs
   G__LockCriticalSection();
#ifdef G__SHAREDLIB
   if (!G__initpermanentsl) {
      G__initpermanentsl = new std::list<G__DLLINIT>;
   }
#endif //G__SHAREDLIB

   if (!G__struct.namerange)
      G__struct.namerange = new NameMap;
   if (!G__newtype.namerange)
      G__newtype.namerange = new NameMap;

   // Call G__RegisterLibrary() again, after it got called already
   // in G__init_setup_funcs(), because G__scratchall might have been
   // called in between.
   
   // Register libCint itself.
   G__RegisterLibrary( (void (*)()) G__call_setup_funcs);
   
   // Do a separate loop so we don't re-load because of A->B->A
   // dependencies introduced by autoloading during dictionary
   // initialization
   if (G__setup_func_list) {
      std::list<G__setup_func_struct>::iterator begin = G__setup_func_list->begin();
      std::list<G__setup_func_struct>::iterator end = G__setup_func_list->end();
      std::list<G__setup_func_struct>::iterator i;
      for (i = begin ; i != end; ++i) {
         if (!i->registered) {
            G__RegisterLibrary(i->func);
            i->registered = true;
         }
      }

      int count = 0;
      for (i = begin ; i != end; ++i, ++count) {
         if (count < G__nlibs_highwatermark) continue;
         if (!i->inited) {
            (i->func)();
            // We setup inited to one only __after__ the execution because the execution
            // can trigger (in particular via ROOT's TCint::UpdateClassInfo)) code that
            // requires the dictionary to be fully initiliazed .. which is done by coming
            // back and expecting the setup function to be run.
            i->inited = 1;
#ifdef G__SHAREDLIB
            G__initpermanentsl->push_back(i->func);
#endif //G__SHAREDLIB
            k++;
#ifdef G__DEBUG
            fprintf(G__sout, "Dictionary for %s initialized\n", i->libname.c_str()); /* only for debug */
#endif
         }
      }
   }
   G__UnlockCriticalSection();
   G__p_local = store_p_local;
   return k;
}

//______________________________________________________________________________
void G__reset_setup_funcs()
{
   if (G__setup_func_list) {

      std::list<G__setup_func_struct>::iterator begin = G__setup_func_list->begin();
      std::list<G__setup_func_struct>::iterator end = G__setup_func_list->end();
      std::list<G__setup_func_struct>::iterator i;

      for (i = begin ; i != end; ++i) {
         i->inited = false;
         i->registered = false;
      }
   }
}

//______________________________________________________________________________
//
//  Windows-NT Symantec C++ setup
//
struct G__libsetup_list {
   void(*p2f)();
   struct G__libsetup_list* next;
};

static struct G__libsetup_list G__p2fsetup;

//______________________________________________________________________________
void G__set_p2fsetup(void(*p2f)())
{
   struct G__libsetup_list *setuplist;
   setuplist = &G__p2fsetup;
   /* get to the end of list */
   while (setuplist->next) setuplist = setuplist->next;
   /* add given entry to the list */
   setuplist->p2f = p2f;
   /* allocation new list entry */
   setuplist->next
   = (struct G__libsetup_list*)malloc(sizeof(struct G__libsetup_list));
   setuplist->next->next = (struct G__libsetup_list*)NULL;
}

//______________________________________________________________________________
static void G__free_p2fsetuplist(G__libsetup_list *setuplist)
{
   if (setuplist->next) {
      G__free_p2fsetuplist(setuplist->next);
      free(setuplist->next);
      setuplist->next = (struct G__libsetup_list*)NULL;
   }
   setuplist->p2f = (void(*)())NULL;
}

//______________________________________________________________________________
void G__free_p2fsetup()
{
   G__free_p2fsetuplist(&G__p2fsetup);
}

//______________________________________________________________________________
static void G__do_p2fsetup()
{
   struct G__libsetup_list *setuplist;
   setuplist = &G__p2fsetup;
   while (setuplist->next) {
      (*setuplist->p2f)();
      setuplist = setuplist->next;
   }
#ifdef G__OLDIMPLEMENTATION1707
   G__free_p2fsetup();
#endif
}

#ifndef G__OLDIMPLEMENATTION1090
//______________________________________________________________________________
static void G__read_include_env(char* envname)
{
   char *env = getenv(envname);
   if (env) {
      char *p, *pc;
      char* tmp = (char*)malloc(strlen(env) + 2);
      strcpy(tmp, env); // Okay we allocated enough memory
      p = tmp;
      while ((pc = strchr(p, ';')) || (pc = strchr(p, ','))) {
         *pc = 0;
         if (p[0]) {
            if (strncmp(p, "-I", 2) == 0) G__add_ipath(p + 2);
            else G__add_ipath(p);
         }
         p = pc + 1;
      }
      if (p[0]) {
         if (strncmp(p, "-I", 2) == 0) G__add_ipath(p + 2);
         else G__add_ipath(p);
      }
      free((void*)tmp);
   }
}
#endif

//______________________________________________________________________________
int G__init_cint(const char* command)
{
   //
   //  Called by
   //     nothing. User host application should call this
   //
   //  Return:
   //  G__CINT_INIT_FAILURE   (-1) : If initialization failed
   //  G__CINT_INIT_SUCCESS      0 : Initialization success.
   //                                no main() found in given source.
   //  G__CINT_INIT_SUCCESS_MAIN 1 : Initialization + main() executed successfully.
   //
   //   Initialize cint, read source file(if any) and return to host program.
   //  Return 0 if success, -1 if failed.
   //  After calling this function, G__calc() or G__pause() has to be explicitly
   //  called to start interpretation or interactive interface.
   //
   //   Example user source code
   //
   //     G__init_cint("cint source.c"); // read source.c and allocate globals
   //     G__calc("func1()");   // interpret func1() which should be in source.c
   //     while(G__pause()==0); // start interactive interface
   //     G__calc("func2()");   // G__calc(),G__pause() can be called multiple times
   //     G__scratch_all();     // terminate cint
   //
   //     G__init_cint("cint source2.c"); // you can start cint again
   //     G__calc("func3()");
   //           .
   //     G__scratch_all();     // terminate cint
   //
   int argn = 0, i;
   int result;
   char *arg[G__MAXARG];
   char argbuf[G__LONGLINE];

   G__LockCriticalSection();

   /***************************************
    * copy command to argbuf. argbuf will
    * be modified by G__split().
    ***************************************/
   if (G__commandline != command) G__strlcpy(G__commandline, command, G__LONGLINE);
   G__strlcpy(argbuf, command, G__LONGLINE);

   /***************************************
    * split arguments as follows
    *
    *        arg[0]
    *   <--------------->
    *  "cint -s source1.c"
    *    ^   ^        ^
    * arg[1] arg[2]  arg[3]
    ***************************************/
   G__split(G__commandline, argbuf, &argn, arg);

   /***************************************
    * shift argument as follows
    *
    *  "cint -s source1.c"
    *    ^   ^        ^
    * arg[0] arg[1]  arg[2]
    ***************************************/
   for (i = 0;i < argn;i++) arg[i] = arg[i+1];
   while (i < G__MAXARG) {
      arg[i] = (char*)NULL;
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
   G__othermain = 2;
   result = G__main(argn, arg);


   if (G__MAINEXIST == G__ismain) {
      G__UnlockCriticalSection();
      return(G__INIT_CINT_SUCCESS_MAIN);
   }
   else if (result == EXIT_SUCCESS) {
      G__UnlockCriticalSection();
      return(G__INIT_CINT_SUCCESS);
   }
   else {
      G__UnlockCriticalSection();
      return(G__INIT_CINT_FAILURE);
   }
}

//______________________________________________________________________________
int G__load(char *commandfile)
{
   //  USING THIS FUNCTION IS NOT RECOMMENDED
   //
   //  Called by
   //     nothing. User host application should call this
   //
   //  If main() exists in user compiled function, G__load() is an entry.
   //  Command file has to contain
   //     cint < [options] > [file1] < [file2] < [file3] <...>>>
   //
   int argn = 0, i;
   char *arg[G__MAXARG], line[G__LONGLINE*2], argbuf[G__LONGLINE*2];
   FILE *cfp;
   cfp = fopen(commandfile, "r");
   if (!cfp) {
      fprintf(stderr, "Error: command file \"%s\" doesn't exist\n", commandfile);
      fprintf(stderr, "  Make command file : [comID] <[cint options]> [file1] <[file2]<[file3]<...>>>\n");
      return -1;
   }
   /**************************************************************
    * Read command file line by line.
    **************************************************************/
   while (G__readline(cfp, line, argbuf, &argn, arg)) {
      for (i = 0; i < argn; i++) {
         arg[i] = arg[i+1];
      }
      arg[argn] = 0;
      /*******************************************************
       * Ignore line start with # and blank lines.
       * else call G__main()
       *******************************************************/
      if ((argn > 0) && (arg[0][0] != '#')) {
         G__othermain = 1;
         G__main(argn, arg);
         if (G__return > G__RETURN_EXIT1) {
            fclose(cfp);
            return 0;
         }
         G__return = G__RETURN_NON;
      }
   }
   fclose(cfp);
   return 0;
}

//______________________________________________________________________________
int G__getcintready()
{
   return(G__cintready);
}

//______________________________________________________________________________
void G__setothermain(int othermain)
{
   G__othermain = (short)othermain;
}

//______________________________________________________________________________
int G__setglobalcomp(int globalcomp)
{
   int ret = G__globalcomp;
   G__globalcomp = globalcomp;
   return ret;
}

//______________________________________________________________________________
void G__setisfilebundled(int isfilebundled)
{
  G__isfilebundled = isfilebundled;
}

//______________________________________________________________________________
G__parse_hook_t* G__set_afterparse_hook(G__parse_hook_t* hook)
{
   G__parse_hook_t* old = G__afterparse_hook;
   G__afterparse_hook = hook;
   return old;
}

//______________________________________________________________________________
G__parse_hook_t* G__set_beforeparse_hook(G__parse_hook_t* hook)
{
   G__parse_hook_t* old = G__beforeparse_hook;
   G__beforeparse_hook = hook;
   return old;
}

//______________________________________________________________________________
void G__display_note()
{
   G__more(G__sout, "\n");
   G__more(G__sout, "Note1: Cint is not aimed to be a 100% ANSI/ISO compliant C/C++ language\n");
   G__more(G__sout, " processor. It rather is a portable script language environment which\n");
   G__more(G__sout, " is close enough to the standard C++.\n");
   G__more(G__sout, "\n");
   G__more(G__sout, "Note2: Regulary check either of /tmp /usr/tmp /temp /windows/temp directory\n");
   G__more(G__sout, " and remove temp-files which are accidentally left by cint.\n");
   G__more(G__sout, "\n");
   G__more(G__sout, "Note3: Cint reads source file on-the-fly from the file system. Do not change\n");
   G__more(G__sout, " the active source during cint run. Use -C option or C1 command otherwise.\n");
   G__more(G__sout, "\n");
   G__more(G__sout, "Note4: In source code trace mode, cint sometimes displays extra-characters.\n");
   G__more(G__sout, " This is harmless. Please ignore.\n");
   G__more(G__sout, "\n");
}

//______________________________________________________________________________
int G__optind = 1;
char* G__optarg;

#define optind G__optind
#define optarg G__optarg
#define getopt G__getopt

//______________________________________________________________________________
int G__getopt(int argc, char** argv, char* optlist)
{
   int optkey;
   char *p;
   if (optind < argc) {
      if ('-' == argv[optind][0]) {
         optkey = argv[optind][1] ;
         p = optlist;
         while (*p) {
            if ((*p) == optkey) {
               ++p;
               if (':' == (*p)) { /* option with argument */
                  if (argv[optind][2]) { /* -aARGUMENT */
                     optarg = argv[optind] + 2;
                     optind += 1;
                     return(argv[optind-1][1]);
                  }
                  else { /* -a ARGUMENT */
                     optarg = argv[optind+1];
                     optind += 2;
                     return(argv[optind-2][1]);
                  }
               }
               else { /* option without argument */
                  ++optind;
                  optarg = (char*)NULL;
                  return(argv[optind-1][1]);
               }
            }
            ++p;
         }
         G__fprinterr(G__serr, "Error: Unknown option %s\n", argv[optind]);
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

//______________________________________________________________________________
extern int G__quiet;
const char *G__libname;

//______________________________________________________________________________
int G__main(int argc, char** argv)
{
   // -- Main entry of the C/C++ interpreter.
   int stepfrommain = 0;
   int  ii;
   char* forceassignment = 0;
   int xfileflag = 0;
   G__FastAllocString sourcefile(G__MAXFILENAME);
   /*************************************************************
    * C/C++ interpreter option related variables
    *************************************************************/
   extern int optind;
   extern char *optarg;
   static const char usage[] = "Usage: %s [options] [sourcefiles|suboptions] [arguments]\n";
   static char *progname;
   char *ppc;
   /*************************************************************
    * Interpreted code option related variables
    *************************************************************/
   int c, iarg;
   G__FastAllocString temp(G__ONELINE);
   long G__argpointer[G__MAXARG];
   char dumpfile[G__MAXFILENAME];
   G__value result;
   const char* linkfilename = 0;
   int linkflag = 0;
   char* dllid = 0;
   static char clnull[1] = "";
   struct G__dictposition stubbegin;
   char* icom = 0;
   /*****************************************************************
    * Setting STDIOs.  May need to modify here.
    *  init.c, end.c, scrupto.c, pause.c
    *****************************************************************/
   if (!G__stdout) G__stdout = stdout;
   if (!G__stderr) G__stderr = stderr;
   if (!G__stdin) G__stdin = stdin;
   G__serr = G__stderr;
   G__sout = G__stdout;
   G__sin = G__stdin;
#ifdef G__MEMTEST
   G__memanalysis();
#endif
   G__free_ifunc_table(&G__ifunc);
   G__ifunc.allifunc = 0;
   G__ifunc.next = 0;
#ifdef G__NEWINHERIT
   G__ifunc.tagnum = -1;
#endif
   for (int ix = 0; ix < G__MAXIFUNC; ++ix) {
      G__ifunc.funcname[ix] = 0;
   }
   G__p_ifunc = &G__ifunc;
   /*************************************************************
    * Inialization before running C/C++ interpreter.
    * Clear ifunc table,global variables and local variable pointer
    *************************************************************/
   G__scratch_all();
   if (G__beforeparse_hook) {
      G__beforeparse_hook();
   }
#ifdef G__SECURITY
   G__init_garbagecollection();
#endif
   /*************************************************************
    * Inialization of readline dumpfile pointer
    *************************************************************/
   for (ii = 0; ii <= 5; ++ii) {
      G__dumpreadline[ii] = 0;
      G__Xdumpreadline[ii] = 0;
   }
   if (argv[0]) {
      G__strlcpy(G__nam, argv[0], sizeof(G__nam)); // get command name
   }
   else {
      G__strlcpy(G__nam, "cint", sizeof(G__nam));
   }
   /*************************************************************
    * Set stderr,stdin,stdout,NULL pointer values to global
    *************************************************************/
   G__macros[0] = '\0';
   G__set_stdio();
#ifndef G__OLDIMPLEMENTATION1817
#if defined(G__ROOT) && !defined(G__NATIVELONGLONG)
   //
   {
      int xtagnum, xtypenum;
      G__cpp_setuplongif();
      xtagnum = G__defined_tagname("G__longlong", 2);
      xtypenum = G__search_typename("long long", 'u', xtagnum, G__PARANORMAL);
      xtagnum = G__defined_tagname("G__ulonglong", 2);
      xtypenum = G__search_typename("unsigned long long", 'u', xtagnum, G__PARANORMAL);
      xtagnum = G__defined_tagname("G__longdouble", 2);
      xtypenum = G__search_typename("long double", 'u', xtagnum, G__PARANORMAL);
   }
#endif
#endif

   /* Signal handling moved after getopt to enable core dump with 'E' */
#ifdef G__HSTD
   /*************************************************************
    * TEE software error handling
    *************************************************************/
   if (setjmp(EH_env)) {
      G__errorprompt("TEE software error");
      G__scratch_all();
      return EXIT_FAILURE;
   }
#endif

   /*************************************************************
    * set compile/interpret environment
    *************************************************************/
   G__breakline[0] = '\0';
   G__breakfile[0] = '\0';
   if (G__allincludepath) {
      free(G__allincludepath);
      G__allincludepath = 0;
   }
   G__allincludepath = (char*) malloc(1);
   if (G__allincludepath) {
      G__allincludepath[0] = '\0';
   }
   /*************************************************************
    * Set compiled global variable pointer to interpreter variable
    * pointer.  Global variables in compiled code are tied to
    * global variables in interpreted code.
    *************************************************************/
   G__prerun = 1;
   G__setdebugcond();
   G__globalsetup();
   G__call_setup_funcs();
   G__do_p2fsetup();
   G__prerun = 0;
   G__setdebugcond();
   /*************************************************************
    * Get command name
    *************************************************************/
   progname = argv[0];
   /*************************************************************
    * Initialize dumpfile name. If -d option is given, dumpfile
    * is set to some valid file name.
    *************************************************************/
   dumpfile[0] = '\0';
#ifndef G__TESTMAIN
   optind = 1;
#endif
   // Keep track of the +STUB -STUB argument
   bool startedSTUB = false;
   /*************************************************************
    * Get command options
    *************************************************************/
   char magicchars[100];
   G__strlcpy(magicchars,".:a:b:c:d:ef:gij:kl:mn:pq:rstu:vw:x:y:z:AB:CD:EF:G:H:I:J:L:KM:N:O:P:QRSTU:VW:X:Y:Z:-:@+:",100);
   while ((c = getopt(argc, argv, magicchars)) != EOF) {
      switch (c) {
#ifndef G__OLDIMPLEMENTATION2226
         case '+':
            G__setmemtestbreak(atoi(optarg) / 10000, atoi(optarg) % 10000);
            break;
#endif
#ifdef G__CINT_VER6
         case '@':
            if (G__cintv6 == 0) G__cintv6 = G__CINT_VER6;
            else if (G__cintv6 == G__CINT_VER6) G__cintv6 |= G__BC_DEBUG;
            break;
#endif
         case 'H': /* level of inclusion for dictionary generation */
            G__gcomplevellimit = atoi(optarg);
            break;
         case 'J': /* 0 - 5 */
            G__dispmsg = atoi(optarg);
            break;
         case 'j':
            G__multithreadlibcint = atoi(optarg);
            break;
         // 03-07-07 new option to include the library name (it's actually the .nm file)
         case 'L':
            G__libname = optarg;
            break;
         case '.':
            // 09-07-07 new option to separate the dictionaries
            // If G__dicttype==0 write everything (like in the old times)
            // If G__dicttype==1 the write the ShowMembers
            // If G__dicttype==2 write only the pointer to inline functions
            // If G__dicttype==3 write all the memfunc_setup rubbish
            // do we still need to fill up the structures and all that?
            G__dicttype = (G__dictgenmode) atoi(optarg);
            break;
         case 'm':
            G__lang = G__ONEBYTE;
            break;
         case 'Q':
            G__quiet = 1;
            break;
         case 'B':
            G__setInitFunc(optarg);
            break;
         case 'C':
            G__setcopyflag(1);
            break;
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
            if (G__NOT_USING_2ARG_NEW&G__is_operator_newdelete) {
               G__fprinterr(G__serr, "!!!-M option may not be needed any more. A new scheme has been implemented.\n");
               G__fprinterr(G__serr, "!!!Refer to $CINTSYSDIR/doc/makecint.txt for the detail.\n");
            }
            break;
         case 'V': /* Create precompiled private member dictionary */
            G__precomp_private = 1;
            break;
         case 'q':
#ifdef G__SECURITY
            G__security = G__getsecuritycode(optarg);
#endif
            break;
         case 'e': /* Parse all extern declaration */
            G__parseextern = 1;
            break;
         case 'i': /* interactive return */
            G__interactive = 1;
            break;
         case 'x':
            if (xfileflag) {
               G__fprinterr(G__serr, "Error: only one -x option is allowed\n");
               G__exit(EXIT_FAILURE);
            }
            else {
               xfileflag = optind - 1;
            }
            break;
         case 'I': /* Include path */
#ifndef G__OLDIMPLEMENATTION1090
            /*************************************************************
             * INCLUDE environment variable. Implemented but must not activate this.
             *************************************************************/
            if (',' == optarg[0]) G__read_include_env(optarg + 1);
            else G__add_ipath(optarg);
#else
            G__add_ipath(optarg);
#endif
            break;
         case 'F': /* force assignment is evaluated after pre-RUN */
            forceassignment = optarg ;
            break;
         case 'Y': /* Do not ignore std namespace or not */
            G__ignore_stdnamespace = atoi(optarg);
            break;
         case 'Z': /* Generate archived header */
            G__autoload_stdheader = atoi(optarg);
            break;
         case 'n': /* customize G__cpplink file name
                                         *   G__cppXXX?.C , G__cXXX.c */
            if (linkflag) {
               G__fprinterr(G__serr, "Warning: option -n[linkname] must be given prior to -c[linklevel]\n");
            }
            linkfilename = optarg;
            break;
         case 'N': /* customize DLL identification name */
            if (linkflag) {
               G__fprinterr(G__serr, "Warning: option -N[DLL_Name] must be given prior to -c[linklevel]\n");
            }
            dllid = optarg;
            break;
         case 'U':
            G__SystemIncludeDir = G__AddConstStringList(G__SystemIncludeDir, optarg, 1);
            break;
         case 'u':
            if (!G__fpundeftype) {
               G__security = 0;
               G__fpundeftype = fopen(optarg, "w");
               fprintf(G__fpundeftype, "/* List of possible undefined type names. This list is not perfect.\n");
               fprintf(G__fpundeftype, "* It is user's responsibility to modify this file. \n");
               fprintf(G__fpundeftype, "* There are cases that the undefined type names can be\n");
               fprintf(G__fpundeftype, "*   class name\n");
               fprintf(G__fpundeftype, "*   struct name\n");
               fprintf(G__fpundeftype, "*   union name\n");
               fprintf(G__fpundeftype, "*   enum name\n");
               fprintf(G__fpundeftype, "*   typedef name\n");
               fprintf(G__fpundeftype, "*   not a typename but object name by CINT's mistake\n");
               fprintf(G__fpundeftype, "* but CINT can not distinguish between them. So it outputs 'class xxx;' \n");
               fprintf(G__fpundeftype, "* for all.\n");
               fprintf(G__fpundeftype, "*/\n");
            }
            break;
#ifdef G__SHAREDLIB
         case 'l': /* dynamic link file, shared library file */
            if (
               -1 == G__shl_load(optarg)
            ) {
               if (G__key != 0) {
                  if (system("key .cint_key -l execute"))
                     G__fprinterr(G__serr, "Error running \"key .cint_key -l execute\"\n");
               }
               G__scratch_all();
               return(EXIT_FAILURE);
            }
            break;
#else
         case 'l': /* error message for dynamic link option */
            G__fprinterr(G__serr, "Error: %s is not compiled with dynamic link capability\n", argv[0]);
            break;
#endif
         case 'p': /* use preprocessor */
            G__cpp = 1;
            break;
         case 'P': /* specify what preprocessor to use */
         case 'W':
            switch (optarg[1]) {
               case 'p':
                  G__ppopt = optarg + 2;
                  ppc = G__ppopt;
                  while ((ppc = strchr(ppc, ','))) * ppc = ' ';
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
            if (G__CLINK == G__globalcomp) G__clock = 1;
            break;
         case 'g': /* whole function compile off */
            G__asm_loopcompile = 3;
            G__asm_loopcompile_mode = G__asm_loopcompile;
            break;
         case 'O': /* loop compiler on */
            G__asm_loopcompile = atoi(optarg);
            G__asm_loopcompile_mode = G__asm_loopcompile;
            break;
         case 'v': /* loop compiler debug mode */
            G__asm_dbg = 1;
            break;
         case 'D': /* define macro */
            G__add_macro(optarg);
            break;
         case 'E': /* Dump core at error */
            if (1 == G__catchexception) G__catchexception = 0;
            else if (0 == G__catchexception) G__catchexception = 2;
            else ++G__catchexception;
            ++G__coredump;
            break;
         case 'X': /* readline dumpfile execution */
            G__dumpreadline[0] = fopen(optarg, "r");
            if (G__dumpreadline[0]) {
               G__Xdumpreadline[0] = 1;
               G__fprinterr(G__serr, " -X : readline dumpfile %s executed\n",
                            optarg);
            }
            else {
               G__fprinterr(G__serr,
                            "Readline dumpfile %s can not open\n"
                            , optarg);
               return(EXIT_FAILURE);
            }
            break;
         case 'd': /* dump file */
#ifdef G__DUMPFILE
            G__fprinterr(G__serr, " -d : dump function call history to %s\n",
                         optarg);
            if (strcmp(optarg + strlen(optarg) - 2, ".c") == 0 ||
                  strcmp(optarg + strlen(optarg) - 2, ".C") == 0 ||
                  strcmp(optarg + strlen(optarg) - 2, ".h") == 0 ||
                  strcmp(optarg + strlen(optarg) - 2, ".H") == 0) {
               G__fprinterr(G__serr, "-d %s : Improper history dump file name\n"
                            , optarg);
            }
            else G__strlcpy(dumpfile, optarg, sizeof(dumpfile));
#else
            G__fprinterr(G__serr,
                         " -d : func call dump not supported now\n");
#endif
            break;
         case 'k': /* user function key */
            if (system("key .cint_key -l pause")) {
               G__fprinterr(G__serr, "Error running \"key .cint_key -l pause\"\n");
            }
            G__key = 1;
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
            G__globalcomp = atoi(optarg);
            if (abs(G__globalcomp) >= 10) {
               G__default_link = abs(G__globalcomp) % 10;
               G__globalcomp /= 10;
            }
            linkflag = 1;
            if (!linkfilename) {
               switch (G__globalcomp) {
                  case G__CPPLINK:
                     linkfilename = "G__cpplink.C";
                     G__iscpp = 1;
                     G__cpplock = 1;
                     break;
                  case G__CLINK:
                     linkfilename = "G__clink.c";
                     G__iscpp = 0;
                     G__clock = 1;
                     break;
                  case R__CPPLINK:
                     linkfilename = "G__cpplink_rflx.cxx";
                     G__iscpp = 1;
                     G__cpplock = 1;
                     break;
                  default:
                     linkfilename = "G__cpplink.cxx";
                     break;
               }
            }
            if (!dllid) {
               dllid = clnull;
            }
            G__set_globalcomp(optarg, linkfilename, dllid);
            break;
         case 's': /* step into mode */
            G__fprinterr(G__serr, " -s : Step into function/loop mode\n");
            G__steptrace = 1;
            break;
         case 'S': /* step over mode */
            G__stepover = 3;
            G__fprinterr(G__serr, " -S : Step over function/loop mode\n");
            stepfrommain = 1;
            break;
         case 'b': /* break line */
            G__strlcpy(G__breakline, optarg, sizeof(G__breakline));
            break;
         case 'f': /* break file */
            G__strlcpy(G__breakfile, optarg, sizeof(G__breakfile));
            break;
         case 'a': /* assertion */
            G__strlcpy(G__assertion, optarg, sizeof(G__assertion));
            break;
         case 'T': /* trace of input file */
            G__fprinterr(G__serr, " -T : trace from pre-run\n");
            G__debugtrace = G__istrace = G__debug = 1;
            G__setdebugcond();
            break;
         case 'G': { /* trace dump */
            FILE *newerr =  fopen(optarg, "w");
            if (newerr==0) {
               G__fprinterr(G__serr, " -G : unable to open file %s.\n",optarg);
            } else {
               G__serr = newerr;
            }
            break;
         }
         case 't': /* trace of input file */
            G__fprinterr(G__serr, " -t : trace execution\n");
            G__istrace = G__debugtrace = 1;
            break;
         case 'R': /* displays input file at the break point*/
            G__fprinterr(G__serr, " -d : display at break point mode\n");
            G__breakdisp = 1;
            break;
         case 'r': /* revision */
            G__revprint(G__sout);
            if (G__key != 0) {
               if (system("key .cint_key -l execute")) {
                  G__fprinterr(G__serr, "Error running \"key .cint_key -l execute\"\n");
               }
            }
            return(EXIT_SUCCESS);
            /* break; */
         case '-':
            icom = optarg;
            break;
         default:
#ifndef G__SMALLOBJECT
            G__more_pause((FILE*)NULL, 0);
            fprintf(G__sout, usage, progname);
            G__more_pause(G__sout, 0);

            G__display_note();

            G__more(G__sout, "options     (* : used only with makecint or -c option)\n");
            G__more(G__sout, "  -A : ANSI C++ mode(default)\n");
            G__more(G__sout, "  -b [line] : set break line\n");
            G__more(G__sout, "* -c -1: make C++ precompiled interface method files\n");
            G__more(G__sout, "* -c -2: make C precompiled interface method files\n");
            G__more(G__sout, "* -c -10: make C++ precompiled interface method files. Default link off\n");
            G__more(G__sout, "* -c -20: make C precompiled interface method files. Default link off\n");
            G__more(G__sout, "* -c -11: make C++ precompiled interface method files. Default link on\n");
            G__more(G__sout, "* -c -21: make C precompiled interface method files. Default link on\n");
            G__more(G__sout, "  -C : copy src to $TMPDIR so that src can be changed during cint run\n");
#ifdef G__DUMPFILE
            G__more(G__sout, "  -d [dumpfile] : dump function call history\n");
#endif
            G__more(G__sout, "  -D [macro] : define macro symbol for #ifdef\n");
            G__more(G__sout, "  -e : Not ignore extern declarations\n");
            G__more(G__sout, "  -E : Dump core at error\n");
            G__more(G__sout, "  -E -E : Exit process at error and uncaught exception\n");
            G__more(G__sout, "  -f [file] : set break file\n");
            G__more(G__sout, "  -F [assignement] : set global variable\n");
            G__more(G__sout, "  -G [tracedmp] : dump exec trace into file\n");
            G__more(G__sout, "* -H[1-100] : level of header inclusion activated for dictionary generation\n");
            G__more(G__sout, "  -i : interactively return undefined symbol value\n");
            G__more(G__sout, "  -I [includepath] : set include file search path\n");
            G__more(G__sout, "* -j [0|1]: Create multi-thread safe DLL(experimental)\n");
            G__more(G__sout, "  -J[0-4] : Display nothing(0)/error(1)/warning(2)/note(3)/all(4)\n");
            /* G__more(G__sout,"  -k : function key on\n"); */
            G__more(G__sout, "  -K : C mode\n");
#ifdef G__SHAREDLIB
            G__more(G__sout, "  -l [dynamiclinklib] : link dynamic link library\n");
#endif
            G__more(G__sout, "  -m : Support ISO-8859-x Eurpoean char set (disabling multi-byte char)\n");
            G__more(G__sout, "* -M [newdelmask] : operator new/delete mask for precompiled interface method\n");
            G__more(G__sout, "* -n [linkname] : Specify precompiled interface method filename\n");
            G__more(G__sout, "* -N [DLL_name] : Specify DLL interface method name\n");
            G__more(G__sout, "  -O [0~4] : Loop compiler on(1~5) off(0). Default on(4)\n");
            G__more(G__sout, "  -p : use preprocessor prior to interpretation\n");
            G__more(G__sout, "  -q [security] : Set security level(default 0)\n");
            G__more(G__sout, "  -Q : Quiet mode (no prompt)\n");
            G__more(G__sout, "  -r : revision and linked function/global info\n");
            G__more(G__sout, "  -R : display input file at break point\n");
            G__more(G__sout, "  -s : step execution mode\n");
            G__more(G__sout, "  -S : step execution mode, First stop in main()\n");
            G__more(G__sout, "  -t : trace execution\n");
            G__more(G__sout, "  -T : trace from pre-run\n");
            G__more(G__sout, "  -u [undefout] : listup undefined typenames\n");
            G__more(G__sout, "* -U [dir] : directory to disable interface method generation\n");
            G__more(G__sout, "* -V : Generate symbols for non-public member with -c option\n");
            G__more(G__sout, "  -v : Bytecode compiler debug mode\n");
            G__more(G__sout, "  -X [readlinedumpfile] : Execute readline dumpfile\n");
            G__more(G__sout, "  -x 'main() {...}' : Execute argument as source code\n");
            G__more(G__sout, "  -Y [0|1]: ignore std namespace (default=1:ignore)\n");
            G__more(G__sout, "  -Z [0|1]: automatic loading of standard header files with DLL\n");
            G__more(G__sout, "  --'command': Execute interactive command and terminate Cint\n");
            G__more(G__sout, "suboptions\n");
            G__more(G__sout, "  +V : turn on class title comment mode for following source fies\n");
            G__more(G__sout, "  -V : turn off class title comment mode for following source fies\n");
            G__more(G__sout, "  +P : turn on preprocessor for following source files\n");
            G__more(G__sout, "  -P : turn off preprocessor for following source files\n");
            G__more(G__sout, "* +STUB : stub function header begin\n");
            G__more(G__sout, "* -STUB : stub function header end\n");
            G__more(G__sout, "sourcefiles\n");
            G__more(G__sout, "  Any valid C/C++ source or header files\n");
            G__more(G__sout, "EXAMPLES\n");
            G__more(G__sout, "  $ cint prog.c main.c\n");
            G__more(G__sout, "  $ cint -S prog.c main.c\n");
            G__more(G__sout, "\n");
            if (G__key) {
               if (system("key .cint_key -l execute")) {
                  G__fprinterr(G__serr, "Error running \"key .cint_key -l execute\"\n");
               }
            }
#endif
            return EXIT_FAILURE;
      }
   }
   {
      if (G__globalcomp != G__NOLINK) {
         G__platformMacro(); // second round, now taking G__globalcomp into account
      }
      int oldglobalcomp = G__globalcomp;
      G__globalcomp = G__NOLINK;
      if (G__cintv6) {
         temp.Format("G__CINTVERSION=%ld", (long) G__CINTVERSION_V6);
      }
      else {
         temp.Format("G__CINTVERSION=%ld", (long) G__CINTVERSION_V5);
      }
      G__add_macro(temp);
      G__globalcomp = oldglobalcomp;
   }
   /*************************************************************
    * Signal handling
    *************************************************************/
#ifndef G__ROOT  /* This is only defined for ROOT */
#ifndef G__DONT_CATCH_SIGINT
   signal(SIGINT, G__breakkey);
#endif /* G__DONT_CACH_SIGINT */
   if (!G__coredump) {
      signal(SIGFPE, G__floatexception);
      signal(SIGSEGV, G__segmentviolation);
#ifdef SIGEMT
      signal(SIGEMT, G__outofmemory);
#endif
#ifdef SIGBUS
      signal(SIGBUS, G__buserror);
#endif
   }
   else if (G__coredump >= 2) {
      signal(SIGFPE, G__errorexit);
      signal(SIGSEGV, G__errorexit);
#ifdef SIGEMT
      signal(SIGEMT, G__errorexit);
#endif
#ifdef SIGBUS
      signal(SIGBUS, G__errorexit);
#endif
   }
#endif /* G__ROOT */
#ifndef G__OLDIMPLEMENATTION175
   if (!G__sizep2memfunc) {
      G__sizep2memfunc = G__SHORTALLOC * 2 + G__LONGALLOC;
   }
#endif
#ifdef G__DUMPFILE
   /*************************************************************
    * Open dumpfile if specified.
    *************************************************************/
   /* G__dumpfile=fopen("/dev/null","w"); */
   if (strcmp(dumpfile, "") != 0) G__dumpfile = fopen(dumpfile, "w");
#ifdef G__MEMTEST
   /* else G__dumpfile = G__memhist; */
#endif
#endif

#ifdef G__NOSTUBS
   int includes_printed = 0;
   std::string linkfilename_h;
#endif //G__NOSTUBS

  // 15-01-08
  // Translate an ifdef into a normal global variable..
  // sligthly more convenient.
  // This is the variable used to check if the stubs must be
  // written in the dictionary (can only be changed in conf. time)
#ifdef G__NOSTUBS
  G__nostubs = 1;
#endif

#ifdef G__NOSTUBSTEST
  G__nostubs = 0;
#endif

   if (G__globalcomp < G__NOLINK) {
      G__gen_cppheader(0);
   }

   /*************************************************************
    * prerun, read whole ifuncs to allocate global variables and
    * make ifunc table.
    *************************************************************/
   while ((G__MAINEXIST != G__ismain && (optind < argc)) || xfileflag) {
      if (xfileflag) {
         // coverity[secure_temp]: we don't care about predictable names.
         FILE *tmpf = tmpfile();
         if (tmpf) {
            fprintf(tmpf, "%s\n", argv[xfileflag]);
            xfileflag = 0;
            fseek(tmpf, 0L, SEEK_SET);
            if (G__loadfile_tmpfile(tmpf) || G__eof == 2) {
               /* file not found or unexpected EOF */
               if (G__CPPLINK == G__globalcomp || G__CLINK == G__globalcomp) {
                  G__cleardictfile(-1);
               }
               G__scratch_all();
               return EXIT_FAILURE;
            }
            continue;
         }
         else {
            xfileflag = 0;
         }
      }
      else {
         sourcefile = argv[optind];
         ++optind;
      }
      if (strcmp(sourcefile, "+P") == 0) {
         G__cpp = 1;
         continue;
      }
      else if (strcmp(sourcefile, "-P") == 0) {
         G__cpp = 0;
         continue;
      }
      else if (strcmp(sourcefile, "+V") == 0) {
         G__fons_comment = 1;
         continue;
      }
      else if (strcmp(sourcefile, "-V") == 0) {
         G__fons_comment = 0;
         continue;
      }
      else if (strcmp(sourcefile, "+STUB") == 0) {
         G__store_dictposition(&stubbegin);
         startedSTUB = true;
         continue;
      }
      else if (strcmp(sourcefile, "-STUB") == 0) {
         if (startedSTUB) {
            G__set_stubflags(&stubbegin);
            startedSTUB = false;
         } else {
            G__more(G__sout, "Warning: -STUB needs to be after a +STUB\n");
         }
         continue;
      }

      if (G__NOLINK > G__globalcomp) {
#ifdef G__NOSTUBS
         if (!includes_printed && !G__isfilebundled) {
            linkfilename_h = linkfilename;
            std::string::size_type in = linkfilename_h.rfind(".");
            if(in != std::string::npos) {
               int l = in;
               linkfilename_h[l+1] = 'h';
               linkfilename_h[l+2] = '\0';
            }

            std::string headerb(basename(dllid));
            std::string::size_type idx = headerb.rfind("Tmp");
            int l;
            if(idx != std::string::npos) {
               l = idx;
               headerb[l] = '\0';
            }
            else {
               idx = headerb.rfind(".");
               if(idx != std::string::npos) {
                  l = idx;
                  headerb[l] = '\0';
               }
            }

            // 12-11-07
            // put protection against multiple includes of dictionaries' .h
            FILE *fp = fopen(linkfilename_h.c_str(),"a");
            if(fp) {
               fprintf(fp,"#ifndef G__includes_dict_%s\n", headerb.c_str());
               fprintf(fp,"#define G__includes_dict_%s\n", headerb.c_str(), sourcefile());
               fclose(fp);
            }
            includes_printed = 1;
         }
#endif // G__NOSTUBS
         G__gen_cppheader(sourcefile);
      }

      if (G__loadfile(sourcefile) < 0 || G__eof == 2) {
         /* file not found or unexpected EOF */
         if (G__CPPLINK == G__globalcomp || G__CLINK == G__globalcomp) {
            G__cleardictfile(-1);
         }
         G__scratch_all();
         return(EXIT_FAILURE);
      }
      if (G__return > G__RETURN_NORMAL) {
         G__scratch_all();
         return(EXIT_SUCCESS);
      }
   }
#ifdef G__NOSTUBS
   if (G__globalcomp < G__NOLINK && includes_printed && !G__isfilebundled) {
      FILE *fp = fopen(linkfilename_h.c_str(),"a");
      if(fp){
         fprintf(fp,"#endif\n");
         fclose(fp);
      }
   }
#endif // G__NOSTUBS

   if (icom) {
      int more = 0;
      G__redirect_on();
      G__init_process_cmd();
      char prompt[G__ONELINE];
      G__strlcpy(prompt, "cint>", G__ONELINE);
      G__process_cmd(icom, prompt, &more, 0, 0);
      G__scratch_all();
      return EXIT_SUCCESS;
   }
   if (G__afterparse_hook) {
      G__afterparse_hook();
   }
   if (G__security_error) {
      G__fprinterr(G__serr, "Warning: Error occurred during reading source files\n");
   }

#ifdef G__NOSTUBS
   // Try to differentiate the different kinds of tmp dicts.
   // (although we used the all here so it's a bit silly)
   if(G__dicttype==kCompleteDictionary || G__dicttype==kFunctionSymbols || G__dicttype==kNoWrappersDictionary) {
      G__init_process_cmd();
      G__gen_extra_include();
   }
#else
   G__gen_extra_include();
#endif //G__NOSTUBS

   if (G__globalcomp == G__CPPLINK) {
      // -- C++ header.
      if (G__steptrace || G__stepover) {
         while (!G__pause()) {}
      }

#ifdef G__NOSTUBS
      // Try to differentiate the different kinds of tmp dicts.
      // (although we used the all here so it's a bit silly)
      if(G__dicttype==kCompleteDictionary || G__dicttype==kFunctionSymbols || G__dicttype==kNoWrappersDictionary)
         G__gen_cpplink();
#else
      G__gen_cpplink();
#endif //G__NOSTUBS

#if !defined(G__ROOT) && !defined(G__D0)
      G__scratch_all();
#endif
      if (G__security_error) {
         G__fprinterr(G__serr, "Warning: Error occurred during dictionary source generation\n");
         G__cleardictfile(-1);
         return -1;
      }
      G__cleardictfile(EXIT_SUCCESS);
      return EXIT_SUCCESS;
   }
   else if (G__globalcomp == G__CLINK) {
      // -- C header.
      if (G__steptrace || G__stepover) {
         while (!G__pause()) {}
      }

#ifdef G__NOSTUBS
      // Try to differentiate the different kinds of tmp dicts.
      // (although we used the all here so it's a bit silly)
      if(G__dicttype==kCompleteDictionary || G__dicttype==kFunctionSymbols || G__dicttype==kNoWrappersDictionary)
         G__gen_clink();
#else
      G__gen_clink();
#endif //G__NOSTUBS

#if !defined(G__ROOT) && !defined(G__D0)
      G__scratch_all();
#endif
      if (G__security_error) {
         G__fprinterr(G__serr, "Warning: Error occured during dictionary source generation\n");
         G__cleardictfile(-1);
         return -1;
      }
      G__cleardictfile(EXIT_SUCCESS);
      return EXIT_SUCCESS;
   }
#ifdef G__ROOT
   else if (G__globalcomp == R__CPPLINK) {
      if (linkfilename == 0) {
         linkfilename = "";
      }
#ifdef G__NOSTUBS
      // Try to differentiate the different kinds of tmp dicts.
      // (although we used the all here so it's a bit silly)
      if(G__dicttype==kCompleteDictionary || G__dicttype==kFunctionSymbols || G__dicttype==kNoWrappersDictionary)
         rflx_gendict(linkfilename, sourcefile);
#else
      rflx_gendict(linkfilename, sourcefile);
#endif //G__NOSTUBS

      return EXIT_SUCCESS;
   }
#endif
   --optind;
   if (G__debugtrace) {
      G__fprinterr(G__serr, "PRE-RUN END\n");
   }
   /*************************************************************
    * set debug conditon after prerun
    *************************************************************/
   if (G__breakline[0]) {
      G__setbreakpoint(G__breakline, G__breakfile);
   }
   G__eof = 0;
   G__prerun = 0;
   G__debug = G__debugtrace;
   G__step = G__steptrace;
   if (stepfrommain) {
      G__step = 1;
   }
   G__setdebugcond();
   /*************************************************************
    * Forceassignment is given by -F command line option.
    *    cint -F(expr1,expr2,...)
    * Expressions are evaluated after pre-RUN.
    *************************************************************/
   if (forceassignment) {
      G__calc_internal(forceassignment);
   }
   //
   // If G__main() is called from G__init_cint() then return
   // from G__main().  G__calc() has to be explicitly called
   // to start interpretation.
   //
   G__cintready = 1; // set ready flag for embedded use
   switch (G__othermain) {
      case 2:
         if (G__ismain == G__NOMAIN) {
            return EXIT_SUCCESS;
         }
         break;
      case 3: /* for drcc */
         return EXIT_SUCCESS;
   }
   if (G__fpundeftype) {
      fclose(G__fpundeftype);
      G__fpundeftype = 0;
      return EXIT_SUCCESS;
   }
   /*************************************************************
    * Interpretation begin. If there were main() in input file,
    * main() is used as an entry. If not, G__pause() is called and
    * wait for user input.
    *************************************************************/
   if (G__ismain == G__MAINEXIST) {
      G__param* para = new G__param;
      /*********************************/
      /* If main() exists , execute it */
      /*********************************/
      /********************************
       * Set interpreter argument
       ********************************/
      for (iarg = optind;iarg < argc;iarg++) {
         G__argpointer[iarg-optind] = (long)argv[iarg];
      }
      while (iarg < G__MAXARG) {
         G__argpointer[iarg-optind] = 0;
         ++iarg;
      }
      /********************************
       * Call main(argc, argv)
       ********************************/
      if (G__breaksignal) {
         G__fprinterr(G__serr, "\nCALL main()\n");
      }
      temp = "main";
      para->paran = 2;
      G__letint(&para->para[0], 'i', argc - optind);
      para->para[0].tagnum = -1;
      para->para[0].typenum = -1;
      para->para[0].ref = 0;
      G__letint(&para->para[1], 'C', (long) G__argpointer);
      para->para[1].tagnum = -1;
      para->para[1].typenum = -1;
      para->para[1].ref = 0;
      para->para[1].obj.reftype.reftype = G__PARAP2P;
      G__interpret_func(&result, temp, para, G__HASH_MAIN, G__p_ifunc, G__EXACT, G__TRYNORMAL);
      delete para;
      if (!result.type) {
         result = G__null;
      }
      /*************************************
       * After main() function, break if
       * step or break mode
       *************************************/
      if (G__breaksignal || G__RETURN_EXIT1 == G__return) {
         G__return = G__RETURN_NON;
         G__break = 0;
         G__setdebugcond();
         G__fprinterr(G__serr, "!!! return from main() function\n");
#ifdef SIGALRM
         if (G__RETURN_EXIT1 == G__return) {
            G__fprinterr(G__serr,
                         "Press return or process will be terminated in %dsec by timeout\n"
                         , G__TIMEOUT);
            signal(SIGALRM, G__timeout);
            alarm(G__TIMEOUT);
         }
#endif
         if (G__catchexception != 2) G__pause();
#ifdef SIGALRM
         if (G__RETURN_EXIT1 == G__return) {
            alarm(0);
            G__fprinterr(G__serr, "Time out cancelled\n");
         }
#endif
         if (G__catchexception != 2) G__pause();
      }
      if (G__stepover) {
         G__step = 0;
         G__setdebugcond();
      }
      /*************************************************************
       * If G__main() is called from G__init_cint() then return
       * from G__main() before destroying data.
       *************************************************************/
      if (G__othermain == 2) {
         return (int)G__int(result);
      }
      /*************************************
       * atexit()
       * global destruction
       * file close
       *************************************/
      G__interpretexit();
      return (int)G__int(result);
   }
#ifdef G__TCLTK
   else if (G__ismain == G__TCLMAIN) {
      if (G__othermain == 2) {
         return EXIT_SUCCESS;
      }
      G__interpretexit();
      return EXIT_SUCCESS;
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
      if (!G__quiet) {
         G__cintrevision(G__sout);
         fprintf(G__sout, "No main() function found in given source file. Interactive interface started.\n");
         switch (G__ReadInputMode()) {
            case G__INPUTROOTMODE:
               fprintf(G__sout, "'?':help, '.q':quit, 'statement','{statements;}' or '.p [expr]' to evaluate\n");
               break;
            case G__INPUTCXXMODE:
               fprintf(G__sout, "'?':help, '.q':quit, 'statement;','{statements;}' or '.p [expr]' to evaluate\n");
               break;
            case G__INPUTCINTMODE:
            default:
               fprintf(G__sout, "'h':help, 'q':quit, '{statements;}' or 'p [expr]' to evaluate\n");
               break;
         }
      }
      /*************************************
       * invoke debugger front end
       *************************************/
      while (1) {
         G__pause();
         if (G__return > G__RETURN_NORMAL && G__RETURN_EXIT1 != G__return) {
            G__return = G__RETURN_NON;
            G__scratch_all();
            return EXIT_SUCCESS;
         }
      }
   }
}

//______________________________________________________________________________
int G__init_globals()
{
   // -- Explicit initialization of all necessary global variables.
   int i;
   if (G__init) {
      return 1;
   }
   G__init = 1;

   G__exec_memberfunc = 0;
   G__memberfunc_tagnum = -1;
   G__memberfunc_struct_offset = 0;

   G__atpause = 0;

#ifdef G__ASM
   /***************************************************
    * loop compiler variables initialization
    ***************************************************/
   G__asm_name_p = 0;
#if defined(G__ROOT)
   G__asm_loopcompile = 4;
#else
   G__asm_loopcompile = 4;
#endif
   G__asm_loopcompile_mode = G__asm_loopcompile;
#ifdef G__ASM_WHOLEFUNC
   G__asm_wholefunction = G__ASM_FUNC_NOP;
#endif
   G__asm_wholefunc_default_cp = 0;
   G__abortbytecode();
   G__asm_dbg = 0;
   G__asm_cp = 0;               /* compile program counter */
   G__asm_dt = G__MAXSTACK - 1;   /* constant data address */
#ifdef G__ASM_IFUNC
   G__asm_inst = G__asm_inst_g;
   G__asm_instsize = 0; /* 0 means G__asm_inst is not resizable */
   G__asm_stack = G__asm_stack_g;
   G__asm_name = G__asm_name_g;
   G__asm_name_p = 0;
#endif
#endif

   G__debug = 0;            /* trace input file */
   G__breakdisp = 0;        /* display input file at break point */
   G__break = 0;            /* break flag */
   G__step = 0;             /* step execution flag */
   G__charstep = 0;         /* char step flag */
   G__break_exit_func = 0;  /* break at function exit */
   /* G__no_exec_stack = -1; */ /* stack for G__no_exec */
   G__no_exec = 0;          /* no execution(ignore) flag */
   G__no_exec_compile = 0;
   G__var_type = 'p';      /* variable decralation type */
   G__var_typeB = 'p';
   G__prerun = 0;           /* pre-run flag */
   G__funcheader = 0;       /* function header mode */
   G__return = G__RETURN_NON; /* return flag of i function */
   /* G__extern_global=0;    number of globals defined in c function */
   G__disp_mask = 0;        /* temporary read count */
   G__temp_read = 0;        /* temporary read count */
   G__switch = 0;           /* in a switch, parser should evaluate case expressions */
   G__switch_searching = 0; /* in a switch, parser should return after evaluating a case expression */
   G__eof_count = 0;        /* end of file error flag */
   G__ismain = G__NOMAIN;   /* is there a main function */
   G__globalcomp = G__NOLINK;  /* make compiled func's global table */
   G__store_globalcomp = G__NOLINK;
   G__globalvarpointer = G__PVOID; /* make compiled func's global table */
   // This is already set to zero by the compiler __and__ it may already have incremented for
   // some library load:
   //    G__nfile = 0;
   G__key = 0;              /* user function key on/off */

   G__xfile[0] = '\0';
   G__tempc[0] = '\0';

   G__doingconstruction = 0;

#ifdef G__DUMPFILE
   G__dumpfile = 0;
   G__dumpspace = 0;
#endif

   G__nobreak = 0;

#ifdef G__FRIEND
   G__friendtagnum = -1;
#endif
   G__def_struct_member = 0;
   G__tmplt_def_tagnum = -1;
   G__def_tagnum = -1;
   G__tagdefining = -1;
   G__tagnum = -1;
   G__typenum = -1;
   G__iscpp = 1;
   G__cpplock = 0;
   G__clock = 0;
   G__isconst = 0;
   G__constvar = 0;
   G__isexplicit = 0;
   G__unsigned = 0;
   G__ansiheader = 0;
   G__enumdef = 0;
   G__store_struct_offset = 0;
   G__decl = 0;
   G__longjump = 0;
   G__coredump = 0;
   G__definemacro = 0;

   G__noerr_defined = 0;

   G__using_alloc = 0;
   G__static_alloc = 0;
   G__func_now = -1;
   G__func_page = 0;
   G__varname_now = 0;
   G__twice = 0;
   /* G__othermain; */
   G__cpp = 0;
#ifndef G__OLDIMPLEMENTATOIN136
   G__include_cpp = 0;
#endif
   G__ccom[0] = '\0';
   G__cppsrcpost[0] = '\0';
   G__csrcpost[0] = '\0';
   G__cpphdrpost[0] = '\0';
   G__chdrpost[0] = '\0';
   G__dispsource = 0;
   G__breaksignal = 0;

   G__bitfield = 0;

   G__atexit = 0;

#ifdef G__SIGNAL
   G__SIGINT = 0;
   G__SIGILL = 0;
   G__SIGFPE = 0;
   G__SIGABRT = 0;
   G__SIGSEGV = 0;
   G__SIGTERM = 0;
#ifdef SIGHUP
   G__SIGHUP = 0;
#endif
#ifdef SIGQUIT
   G__SIGQUIT = 0;
#endif
#ifdef SIGTSTP
   G__SIGTSTP = 0;
#endif
#ifdef SIGTTIN
   G__SIGTTIN = 0;
#endif
#ifdef SIGTTOU
   G__SIGTTOU = 0;
#endif
#ifdef SIGALRM
   G__SIGALRM = 0;
#endif
#ifdef SIGUSR1
   G__SIGUSR1 = 0;
#endif
#ifdef SIGUSR2
   G__SIGUSR2 = 0;
#endif
#endif //G__SIGNAL

#ifdef G__SHAREDLIB
   G__allsl = 0;
#endif

   G__p_tempbuf = &G__tempbuf;
   G__tempbuf.level = 0;
   G__tempbuf.obj = G__null;
   G__tempbuf.prev = 0;
   G__templevel = 0;

   G__reftype = G__PARANORMAL;

   G__access = G__PUBLIC;

   G__stepover = 0;

   G__newarray.next = 0;
   /* G__newarray.point = 0; */

   /*************************************
    * Bug fix for SUNOS
    *************************************/
   G__ifile.fp = 0;
   G__ifile.name[0] = '\0';
   G__ifile.line_number = 0;
   G__ifile.filenum = -1;

   G__steptrace = 0;
   G__debugtrace = 0;

   G__ipathentry.pathname = 0;
   G__ipathentry.next = 0;

   G__mfp = 0;

   G__deffuncmacro.hash = 0;
   G__deffuncmacro.callfuncmacro.call_fp = 0;
   G__deffuncmacro.callfuncmacro.call_filenum = -1;
   G__deffuncmacro.callfuncmacro.next = 0;
   G__deffuncmacro.next = 0;

   G__cintsysdir[0] = '*';
   G__cintsysdir[1] = '\0';

   G__p_local = 0 ;    /* local variable array */
   G__globalusingnamespace.basen = 0;

   G__global.varlabel[0][0] = 0;  /* initialize global variable */
   G__global.next = 0;         /* start from one G__var_table */
   G__global.ifunc = 0;
   G__global.prev_local = 0;
   G__global.prev_filenum = -1;
   G__global.tagnum = -1;
   G__global.allvar = 0;
   {
      int ix;
      for (ix = 0;ix < G__MEMDEPTH;ix++) {
         G__global.hash[ix] = 0;
         G__global.varnamebuf[ix] = 0;
         G__global.is_init_aggregate_array[ix] = 0;
      }
   }
   G__cpp_aryconstruct = 0;
   G__cppconstruct = 0;

   G__newtype.alltype = 0;        /* initialize typedef table */

   G__breakfile[0] = '\0';        /* default breakfile="" */

   G__assertion[0] = '\0';        /* initialize debug assertion */

   /**************************************************************
    * initialize constant values
    **************************************************************/
   G__null.obj.d = 0.0;
   G__letint(&G__null, '\0', 0);   /* set null value */
   G__null.tagnum = -1;
   G__null.typenum = -1;
   G__null.ref = 0;
#ifndef G__OLDIMPLEMENTATION1259
   G__null.isconst = G__CONSTVAR;
#endif

   G__one.obj.d = 0.0;
   G__letint(&G__one, 'i', 1); /* set default value */
   G__one.tagnum = -1;
   G__one.typenum = -1;
   G__one.ref = 0;
#ifndef G__OLDIMPLEMENTATION1259
   G__one.isconst = G__CONSTVAR;
#endif

   G__letint(&G__start, 'a', G__SWITCH_START);   /* set start value */
   G__start.tagnum = -1;
   G__start.typenum = -1;
   G__start.ref = 0;
#ifndef G__OLDIMPLEMENTATION1259
   G__start.isconst = G__CONSTVAR;
#endif

   G__letint(&G__default, 'z', G__SWITCH_DEFAULT); /* set default value */
   G__default.tagnum = -1;
   G__default.typenum = -1;
   G__default.ref = 0;
#ifndef G__OLDIMPLEMENTATION1259
   G__default.isconst = G__CONSTVAR;
#endif

   G__letint(&G__block_break, 'Z', G__BLOCK_BREAK); /* set default value */
   G__block_break.tagnum = -1;
   G__block_break.typenum = -1;
   G__block_break.ref = 0;
#ifndef G__OLDIMPLEMENTATION1259
   G__block_break.isconst = G__CONSTVAR;
#endif

   G__letint(&G__block_continue, 'Z', G__BLOCK_CONTINUE); /* set default value */
   G__block_continue.tagnum = -1;
   G__block_continue.typenum = -1;
   G__block_continue.ref = 0;
#ifndef G__OLDIMPLEMENTATION1259
   G__block_continue.isconst = G__CONSTVAR;
#endif

   G__letint(&G__block_goto, 'Z', G__BLOCK_BREAK); /* set default value */
   G__block_goto.tagnum = -1;
   G__block_goto.typenum = -1;
   G__block_goto.ref = 1; /* identical to G__block_break except for this */
#ifndef G__OLDIMPLEMENTATION1259
   G__block_goto.isconst = G__CONSTVAR;
#endif
   G__gotolabel[0] = '\0';

   G__exceptionbuffer = G__null;

   G__virtual = 0;
#ifdef G__AUTOCOMPILE
   G__fpautocc = 0;
   G__compilemode = 1;
#endif
   G__typedefnindex = 0;
   G__oprovld = 0;
   G__p2arylabel[0] = 0;

   G__interactive = 0;
   G__interactivereturnvalue = G__null;

#ifdef G__TEMPLATECLASS
   G__definedtemplateclass.next = 0;
   G__definedtemplateclass.def_para = 0;
   G__definedtemplateclass.def_fp = 0;
#ifdef G__TEMPLATEMEMFUNC
   G__definedtemplateclass.memfunctmplt.next = 0;
#endif /* G__TEMPLATEMEMFUNC */
   G__definedtemplateclass.parent_tagnum = -1;
   G__definedtemplateclass.isforwarddecl = 0;
   G__definedtemplateclass.instantiatedtagnum = 0;
   G__definedtemplateclass.specialization = 0;
   G__definedtemplateclass.spec_arg = 0;

#ifdef G__TEMPLATEFUNC
   G__definedtemplatefunc.next = 0;
   G__definedtemplatefunc.def_para = 0;
   G__definedtemplatefunc.name = 0;
   for (i = 0;i < G__MAXFUNCPARA;i++) {
      G__definedtemplatefunc.func_para.ntarg[i] = 0;
      G__definedtemplatefunc.func_para.nt[i] = 0;
   }
#endif /* G__TEMPLATEFUNC */
#endif /* G__TEMPLATECLASS */

   G__isfuncreturnp2f = 0;

   G__typepdecl = 0;

   G__macroORtemplateINfile = 0;

   G__macro_defining = 0;

   G__nonansi_func = 0;

   G__parseextern = 0;

   G__istrace = 0;

   G__pbreakcontinue = 0;


   G__fons_comment = 0;
   G__setcomment = 0;

#ifdef G__SECURITY
   G__security = G__SECURE_LEVEL0;
   G__castcheckoff = 0;
   G__security_error = G__NOERROR;
   G__max_stack_depth = G__DEFAULT_MAX_STACK_DEPTH;
#endif

   G__preprocessfilekey.keystring = 0;
   G__preprocessfilekey.next = 0;

   G__precomp_private = 0;

   /* The first entry in the const string is a blank string */
   static char clnull[1] = "";
   G__conststringlist.string = clnull;
   G__conststringlist.hash = 0;
   G__conststringlist.prev = 0;
   G__plastconststring = &G__conststringlist;

   /* $xxx search function , set default. */
#ifdef G__ROOT
   if (!G__GetSpecialObject) G__GetSpecialObject = G__getreserved;
#else
   G__GetSpecialObject = G__getreserved;
#endif

   G__is_operator_newdelete = G__DUMMYARG_NEWDELETE | G__NOT_USING_2ARG_NEW;

   G__fpundeftype = 0;

   G__ispermanentsl = 0;

   G__boolflag = 0;

   return 0;
}

//______________________________________________________________________________
void G__initcxx();

//______________________________________________________________________________
static void G__defineMacro(const char* name, long value, const char* cintname = 0, bool cap = true, bool compiler = false)
{
   //
   //  Add a macro called name with its value to the known macros.
   //  Also add a CINT version, which transforms
   //  [_]*xyz[_]* to G__XYZ
   //  If called with cap=false, capitalization does not happen,
   //  i.e. [_]*xyz[_]* is transformed to G__xyz.
   //  If cintname is given, it will be used instead of the
   //  converted name G__XYZ.
   //
   char temp[G__ONELINE];

   if ((G__globalcomp != G__NOLINK) && !compiler) {
      // -- Not a compiler, and !=G__NOLINK - already dealt with in first pass
      return;
   }

   snprintf(temp + 2, G__ONELINE-2, "!%s=%ld", name, value);

   if (!compiler || G__globalcomp != G__NOLINK) {
      // add system version, which starts with a '!'
      G__add_macro(temp + 2);
   }

   if (G__globalcomp != G__NOLINK) {
      // already dealt with in first pass
      return;
   }

   char* start = temp;
   if (cintname) {
      start +=3;
      snprintf(start, G__ONELINE - (start-temp), "%s=%ld", cintname, value);
   }
   else {
      // generate CINT name:
      // leading '_' are skipped:
      char* end = start + 3 + strlen(name) - 1;
      while (start[3] == '_') {
         ++start;
      }
      // it starts with a "G__":
      memcpy(start, "G__", 3);
      // trailing '_' are removed
      while (*end == '_') {
         --end;
      }

      snprintf(end + 1, G__ONELINE - (end-temp), "=%ld", value);
      while (cap && end != start) {
         // capitalize the CINT macro name
         *end = (char)toupper(*end);
         --end;
      }
   }
   // add the CINT version of the macro
   G__add_macro(start);
}

//______________________________________________________________________________
/* Define macro with value, both system macro and CINT macro */
#define G__DEFINE_MACRO(macro) \
   G__defineMacro(#macro, (long)macro)

//______________________________________________________________________________
/* Define compiler macro with value, both system macro and CINT macro */
#define G__DEFINE_MACRO_C(macro) \
   G__defineMacro(#macro, (long)macro, 0, true, true)

//______________________________________________________________________________
/* Define macro with value, both system macro and CINT macro,
  specifying the CINT macro name */
#define G__DEFINE_MACRO_N(macro, name) \
   G__defineMacro(#macro, (long)macro, name)

//______________________________________________________________________________
/* Define compiler macro with value, both system macro and CINT macro,
  specifying the CINT macro name */
#define G__DEFINE_MACRO_N_C(macro, name) \
   G__defineMacro(#macro, (long)macro, name, true, true)

//______________________________________________________________________________
/* Define macro with value, both system macro and CINT macro,
  preventing capitalization of the CINT macro name */
#define G__DEFINE_MACRO_S(macro) \
   G__defineMacro(#macro, (long)macro, 0, false)

//______________________________________________________________________________
/* Define compiler macro with value, both system macro and CINT macro,
  preventing capitalization of the CINT macro name */
#define G__DEFINE_MACRO_S_C(macro) \
   G__defineMacro(#macro, (long)macro, 0, false, true)

// cross-compiling for iOS and iOS simulator (assumes host is Intel Mac OS X)
#if defined(R__IOSSIM) || defined(R__IOS)
#ifdef __x86_64__
#undef __x86_64__
#endif
#ifdef __i386__
#undef __i386__
#endif
#ifdef R__IOSSIM
#define __i386__ 1
#endif
#ifdef R__IOS
#define __arm__ 1
#endif
#endif

//______________________________________________________________________________
void G__platformMacro()
{
   //
   //  (G__globalcomp == G__NOLINK) means first pass, before
   //  G__globalcomp has been defined
   //
   char temp[G__ONELINE];
   /***********************************************************************
    * operating system
    ***********************************************************************/
#if defined(__linux__)  /* Linux */
   G__DEFINE_MACRO(__linux__);
#elif defined(__linux)
   G__DEFINE_MACRO(__linux);
#elif defined(linux)
   G__DEFINE_MACRO(linux);
#endif
#if defined(_FreeBSD_) && !defined(__FreeBSD__)
# define __FreeBSD__ _FreeBSD_
#endif
#ifdef __FreeBSD__   /* FreeBSD */
   G__DEFINE_MACRO_N(__FreeBSD__, "G__FBSD");
#endif
#ifdef __OpenBSD__   /* OpenBSD */
   G__DEFINE_MACRO_N(__OpenBSD__, "G__OBSD");
#endif
#ifdef __hpux        /* HP-UX */
   G__DEFINE_MACRO(__hpux__);
#endif
#ifdef __sun         /* SunOS and Solaris */
   G__DEFINE_MACRO(__sun);
#endif
#ifdef _WIN32        /* Windows 32bit */
   G__DEFINE_MACRO(_WIN32);
#endif
#ifdef _WINDOWS_     /* Windows */
   G__DEFINE_MACRO(_WINDOWS_);
#endif
#ifdef __APPLE__     /* Apple MacOS X */
   G__DEFINE_MACRO(__APPLE__);
#endif
#ifdef _AIX          /* IBM AIX */
   G__DEFINE_MACRO(_AIX);
#endif
#ifdef __sgi         /* SGI IRIX */
   G__DEFINE_MACRO(__sgi);
#endif
#if defined(__alpha) && !defined(__linux) && !defined(__linux__) && !defined(linux) /* DEC/Compac Alpha-OSF operating system */
   G__DEFINE_MACRO(__alpha);
#endif
#ifdef __QNX__         /* QNX realtime OS */
   G__DEFINE_MACRO(__QNX__);
#endif
   /***********************************************************************
    * compiler and library
    ***********************************************************************/
#ifdef G__MINGW /* Mingw */
   G__DEFINE_MACRO_C(G__MINGW);
#endif
#ifdef __CYGWIN__ /* Cygwin */
   G__DEFINE_MACRO(__CYGWIN__);
#endif
#ifdef __GNUC__  /* gcc/g++  GNU C/C++ compiler major version */
   G__DEFINE_MACRO_C(__GNUC__);
#endif
#ifdef __GNUC_MINOR__  /* gcc/g++ minor version */
   G__DEFINE_MACRO_C(__GNUC_MINOR__);
#endif
#if defined(__GNUC__) && defined(__GNUC_MINOR__)
   if (G__globalcomp == G__NOLINK) {
      G__snprintf(temp, sizeof(temp), "G__GNUC_VER=%ld", (long)__GNUC__*1000 + __GNUC_MINOR__);
      G__add_macro(temp);
   }
#endif
#ifdef __GLIBC__   /* GNU C library major version */
   G__DEFINE_MACRO(__GLIBC__);
#endif
#ifdef __GLIBC_MINOR__  /* GNU C library minor version */
   G__DEFINE_MACRO(__GLIBC_MINOR__);
#endif
#ifdef __HP_aCC     /* HP aCC C++ compiler */
   if (G__globalcomp == G__NOLINK) {
      G__snprintf(temp, sizeof(temp), "G__HP_aCC=%ld", (long)__HP_aCC);
      G__add_macro(temp);
   }
   G__DEFINE_MACRO_S_C(__HP_aCC);
#if __HP_aCC > 15000
   if (G__globalcomp == G__NOLINK) {
      G__snprintf(temp, sizeof(temp), "G__ANSIISOLIB=1");
      G__add_macro(temp);
   }
#endif
#endif
#ifdef __SUNPRO_CC  /* Sun C++ compiler */
   G__DEFINE_MACRO_C(__SUNPRO_CC);
#endif
#ifdef __SUNPRO_C   /* Sun C compiler */
   G__DEFINE_MACRO_C(__SUNPRO_C);
#endif
#ifdef _STLPORT_VERSION
   // stlport version, used on e.g. SUN
   G__DEFINE_MACRO_C(_STLPORT_VERSION);
#endif
#ifdef G__VISUAL    /* Microsoft Visual C++ compiler */
   if (G__globalcomp == G__NOLINK) {
      G__snprintf(temp, sizeof(temp), "G__VISUAL=%ld", (long)G__VISUAL);
      G__add_macro(temp);
   }
#endif
#ifdef _MSC_VER     /* Microsoft Visual C++ version */
   if (G__globalcomp == G__NOLINK) {
      G__snprintf(temp, sizeof(temp), "G__VISUAL=%ld", (long)G__VISUAL);
      G__add_macro(temp);
   }
   G__DEFINE_MACRO_C(_MSC_VER);
   if (G__globalcomp == G__NOLINK) {
#ifdef _HAS_ITERATOR_DEBUGGING
      G__snprintf(temp, sizeof(temp), "G__HAS_ITERATOR_DEBUGGING=%d", _HAS_ITERATOR_DEBUGGING);
      G__add_macro(temp);
#endif
#ifdef _SECURE_SCL
      G__snprintf(temp, sizeof(temp), "G__SECURE_SCL=%d", _SECURE_SCL);
      G__add_macro(temp);
#endif
   }
#endif
#ifdef __SC__       /* Symantec C/C++ compiler */
   G__DEFINE_MACRO_N_C(__SC__, "G__SYMANTEC");
#endif
#ifdef __BORLANDC__ /* Borland C/C++ compiler */
   G__DEFINE_MACRO_C(__BORLANDC__);
#endif
#ifdef __BCPLUSPLUS__  /* Borland C++ compiler */
   G__DEFINE_MACRO_C(__BCPLUSPLUS__);
#endif
#ifdef G__BORLANDCC5 /* Borland C/C++ compiler 5.5 */
   G__DEFINE_MACRO_C(__BORLANDCC5__);
#endif
#ifdef __KCC        /* KCC  C++ compiler */
   G__DEFINE_MACRO_C(__KCC__);
#endif
#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER<810) /* icc and ecc C++ compilers */
   G__DEFINE_MACRO_C(__INTEL_COMPILER);
#endif
#ifndef _AIX
#ifdef __xlc__ /* IBM xlc compiler */
   if (G__globalcomp == G__NOLINK) {
      G__snprintf(temp, sizeof(temp), "G__GNUC=%ld", (long)3 /*__GNUC__*/);
      G__add_macro(temp);
      G__snprintf(temp, sizeof(temp), "G__GNUC_MINOR=%ld", (long)3 /*__GNUC_MINOR__*/);
      G__add_macro(temp);
   }
   G__DEFINE_MACRO_C(__xlc__);
#endif
#endif

   if (G__globalcomp == G__NOLINK) {
      G__initcxx();
   }

   /***********************************************************************
    * micro processor
    ***********************************************************************/
#ifdef __hppa__ /* HP-PA , Hewlett Packard Precision Architecture */
   G__DEFINE_MACRO_S(__hppa__);
#endif
#ifdef __i386__ /* Intel 386,486,586 */
   G__DEFINE_MACRO_S(__i386__);
#endif
#ifdef __x86_64__ /* Intel / AMD 64 */
   G__DEFINE_MACRO_S(__x86_64__);
#endif
#ifdef __arm__ /* ARM iOS */
   G__DEFINE_MACRO_S(__arm__);
#endif
#ifdef __amd64 /* Intel / AMD 64 */
   G__DEFINE_MACRO_S(__amd64);
#endif
#ifdef __i860__ /* Intel 860 */
   G__DEFINE_MACRO_S(__i860__);
#endif
#ifdef __i960__ /* Intel 960 */
   G__DEFINE_MACRO_N(__i960__, "G__i860");
#endif
#ifdef __ia64__ /* Intel Itanium */
   G__DEFINE_MACRO_S(__ia64__);
#endif
#ifdef __m88k__ /* Motorola 88000 */
   G__DEFINE_MACRO_S(__m88k__);
#endif
#ifdef __m68k__ /* Motorola 68000 */
   G__DEFINE_MACRO_S(__m68k__);
#endif
#ifdef __ppc__  /* Motorola Power-PC */
   G__DEFINE_MACRO_S(__ppc__);
#endif
#ifdef __PPC__  /* IBM Power-PC */
   G__DEFINE_MACRO(__PPC__);
#endif
#ifdef __mips__ /* MIPS architecture */
   G__DEFINE_MACRO_S(__mips__);
#endif
#ifdef __alpha__ /* DEC/Compac Alpha */
   G__DEFINE_MACRO_S(__alpha__);
#endif
#if defined(__sparc) /* Sun Microsystems SPARC architecture */
   G__DEFINE_MACRO(__sparc);
#elif  defined(__sparc__)
   G__DEFINE_MACRO(__sparc__);
#endif
#ifdef __arc__  /* ARC architecture */
   G__DEFINE_MACRO_S(__arc__);
#endif
#ifdef __M32R__
   G__DEFINE_MACRO_S(__M32R__);
#endif
#ifdef __sh__   /* Hitachi SH micro-controller */
   G__DEFINE_MACRO_S(__sh__);
#endif
#ifdef __arm__  /* ARM , Advanced Risk Machines */
   G__DEFINE_MACRO_S(__arm__);
#endif
#ifdef __s390__ /* IBM S390 */
   G__DEFINE_MACRO_S(__s390__);
#endif

   // Avoid any problem with __attribute__ and __asm
   G__value (*store__GetSpecialObject) (G__CONST char *name,void **ptr,void** ppdict) = G__GetSpecialObject;
   G__GetSpecialObject = 0;
   G__add_macro("__attribute__(X)=");
   G__add_macro("__asm(X)=");
   G__GetSpecialObject = store__GetSpecialObject;

   if (G__globalcomp != G__NOLINK)
      return;

   /***********************************************************************
    * application environment
    ***********************************************************************/
#ifdef G__ROOT
   G__snprintf(temp, sizeof(temp), "G__ROOT=%ld", (long)G__ROOT);
   G__add_macro(temp);
#endif
#ifdef G__NO_STDLIBS
   G__snprintf(temp, sizeof(temp), "G__NO_STDLIBS=%ld", (long)G__NO_STDLIBS);
   G__add_macro(temp);
#endif
#ifdef G__NATIVELONGLONG
   G__snprintf(temp, sizeof(temp), "G__NATIVELONGLONG=%ld", (long)G__NATIVELONGLONG);
   G__add_macro(temp);
#endif

   G__snprintf(temp, sizeof(temp), "int& G__cintv6=*(int*)(%ld);", (long)(&G__cintv6));
   G__exec_text(temp);

   // setup size_t, ssize_t
   int size_t_type = 0;
   if (typeid(size_t) == typeid(unsigned int))
      size_t_type = 'i';
   else if (typeid(size_t) == typeid(unsigned long))
      size_t_type = 'l';
   else if (typeid(size_t) == typeid(unsigned long long))
      size_t_type = 'n';
   else G__fprinterr(G__serr, "On your platform, size_t has a weird typeid of %s which is not handled yet!\n",
                        typeid(size_t).name());

   G__search_typename2("size_t", size_t_type - 1, -1, 0, -1);
   G__setnewtype(-1, NULL, 0);

   G__search_typename2("ssize_t", size_t_type, -1, 0, -1);
   G__setnewtype(-1, NULL, 0);

#if  defined(__APPLE__) && defined(__GNUC__)
   // Apple MacOS X gcc header use directly __builtin_va_list, let's
   // make sure that rootcint does not complain about not knowing what it is.
   G__linked_taginfo G__a_cxx_ACLiC_dictLN_va_list = { "va_list" , 115 , -1 };
   // G__a_cxx_ACLiC_dictLN_va_list.tagnum = -1 ;
   G__get_linked_tagnum_fwd(&G__a_cxx_ACLiC_dictLN_va_list);
   G__search_typename2("__builtin_va_list",117,G__get_linked_tagnum(&G__a_cxx_ACLiC_dictLN_va_list),0,-1);
   G__setnewtype(-1,NULL,0);
#endif
}

//______________________________________________________________________________
void G__set_stdio()
{
   char temp[G__ONELINE];

   G__globalvarpointer = G__PVOID;

   G__intp_sout = G__sout;
   G__intp_serr = G__serr;
   G__intp_sin = G__sin;


   G__var_type = 'E';
   G__globalvarpointer = (long) & G__intp_sout;
   G__snprintf(temp, sizeof(temp), "stdout=(FILE*)(%ld)", (long)G__intp_sout);
   G__getexpr(temp);
   G__globalvarpointer = G__PVOID;

   G__var_type = 'E';
   G__globalvarpointer = (long) & G__intp_serr;
   G__snprintf(temp, sizeof(temp), "stderr=(FILE*)(%ld)", (long)G__intp_serr);
   G__getexpr(temp);
   G__globalvarpointer = G__PVOID;

   G__var_type = 'E';
   G__globalvarpointer = (long) & G__intp_sin;
   G__snprintf(temp, sizeof(temp), "stdin=(FILE*)(%ld)", (long)G__intp_sin);
   G__getexpr(temp);
   G__globalvarpointer = G__PVOID;

   G__definemacro = 1;
   G__snprintf(temp, sizeof(temp), "EOF=%ld", (long)EOF);
   G__getexpr(temp);
   G__snprintf(temp, sizeof(temp), "NULL=%ld", (long)NULL);
   G__getexpr(temp);
#ifdef G__SHAREDLIB
   G__snprintf(temp, sizeof(temp), "G__SHAREDLIB=1");
   G__getexpr(temp);
#endif
#if defined(G__P2FCAST) || defined(G__P2FDECL)
   G__snprintf(temp, sizeof(temp), "G__P2F=1");
   G__getexpr(temp);
#endif
#ifdef G__NEWSTDHEADER
   G__snprintf(temp, sizeof(temp), "G__NEWSTDHEADER=1");
   G__getexpr(temp);
#endif
   G__platformMacro();
   G__definemacro = 0;

   /* G__constvar=G__CONSTVAR; G__var_type='g'; G__getexpr("TRUE=1"); */
   /* G__constvar=G__CONSTVAR; G__var_type='g'; G__getexpr("FALSE=0"); */
   G__constvar = G__CONSTVAR;
   G__var_type = 'g';
   G__getexpr("true=1");
   G__constvar = G__CONSTVAR;
   G__var_type = 'g';
   G__getexpr("false=0");
   G__constvar = 0;

#ifdef G__DUMPFILE
   G__globalvarpointer = (long) & G__dumpfile;
   G__var_type = 'E';
   G__getexpr("G__dumpfile=0");
#endif
   G__globalvarpointer = G__PVOID;

   G__var_type = 'p';
   G__tagnum = -1;
   G__typenum = -1;
}

//______________________________________________________________________________
void G__set_stdio_handle(FILE *sout, FILE *serr, FILE *sin)
{
   char temp[G__ONELINE];

   G__sout = G__stdout = sout;
   G__serr = G__stderr = serr;
   G__sin  = G__stdin  = sin;

   G__var_type = 'E';
   G__globalvarpointer = (long) & G__intp_sout;
   G__snprintf(temp, sizeof(temp), "stdout=(FILE*)(%ld)", (long)G__intp_sout);
   G__getexpr(temp);
   G__globalvarpointer = G__PVOID;

   G__var_type = 'E';
   G__globalvarpointer = (long) & G__intp_serr;
   G__snprintf(temp, sizeof(temp), "stderr=(FILE*)(%ld)", (long)G__intp_serr);
   G__getexpr(temp);
   G__globalvarpointer = G__PVOID;

   G__var_type = 'E';
   G__globalvarpointer = (long) & G__intp_sin;
   G__snprintf(temp, sizeof(temp), "stdin=(FILE*)(%ld)", (long)G__intp_sin);
   G__getexpr(temp);
   G__globalvarpointer = G__PVOID;
}

//______________________________________________________________________________
const char *G__cint_version()
{
   if (G__cintv6) return(G__CINTVERSIONSTR_V6);
   else           return(G__CINTVERSIONSTR_V5);
   /* return "5.14.34, Mar 10 2000"; */
}

//______________________________________________________________________________
int G__cintrevision(FILE* fp)
{
   fprintf(fp, "\n");
   fprintf(fp, "cint : C/C++ interpreter  (mailing list 'root-cint@cern.ch')\n");
   fprintf(fp, "   Copyright(c) : 1995~2010 Masaharu Goto (gotom@hanno.jp)\n");
   fprintf(fp, "   revision     : %s by M.Goto\n\n", G__cint_version());

#ifdef G__DEBUG
   fprintf(fp, "   MEMORY LEAK TEST ACTIVATED!!! MAYBE SLOW.\n\n");
#endif
   return(0);
}

//______________________________________________________________________________
G__ConstStringList* G__AddConstStringList(G__ConstStringList* current, char* str, int islen)
{
   int itemp;
   struct G__ConstStringList* next;

   next = (struct G__ConstStringList*)malloc(sizeof(struct G__ConstStringList));

   next->string = (char*)malloc(strlen(str) + 1);
   strcpy(next->string, str); // Okay we allocated enough memory

   if (islen) {
      next->hash = (int)strlen(str);
   }
   else {
      G__hash(str, next->hash, itemp);
   }

   next->prev = current;

   return(next);
}

//______________________________________________________________________________
void G__DeleteConstStringList(G__ConstStringList* current)
{
   struct G__ConstStringList* tmp;
   while (current) {
      if (current->string) free((void*)current->string);
      tmp = current->prev;
      free((void*)current);
      current = tmp;
   }
}

//______________________________________________________________________________
void G__LockCpp()
{
   /* Same as option -A */
   G__cpplock = 1;
   G__iscpp = 1;
}

//______________________________________________________________________________
void G__SetCatchException(int mode)
{
   G__catchexception = mode;
}

int G__GetCatchException()
{
   return G__catchexception;
}

} /* extern "C" */

/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:3
 * c-continued-statement-offset:3
 * c-brace-offset:-3
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-3
 * compile-command:"make -k"
 * End:
 */
