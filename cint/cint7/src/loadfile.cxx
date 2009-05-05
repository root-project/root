/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file loadfile.c
 ************************************************************************
 * Description:
 *  Loading source file
 ************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

/* Define one of following */
#define G__OLDIMPLEMENTATION1922 /* keep opening all header files for +V +P */

#ifdef _WIN32
#include "process.h"
#endif // _WIN32

#ifdef G__ROOT
#include "RConfigure.h"
#endif // G__ROOT

#ifdef G__HAVE_CONFIG
#include "configcint.h"
#endif // G__HAVE_CONFIG

#include "common.h"
#include "Dict.h"

#ifndef G__TESTMAIN
#include <sys/stat.h>
#endif // G__TESTMAIN

#ifdef G__WIN32
#include <windows.h>
#endif // G__WIN32

#define G__OLDIMPLEMENTATION1849

#include <cstring>
#include <string>
#include <list>

using namespace Cint::Internal;
using namespace std;

//______________________________________________________________________________
//
//  Define G__EDU_VERSION for CINT C++ educational version.
//  If G__EDU_VERSION is defined, CINT will search ./include and
//  ./stl directory for standard header files.
//
// #define G__EDU_VERSION

//______________________________________________________________________________
static G__IgnoreInclude G__ignoreinclude = (G__IgnoreInclude)NULL;

//______________________________________________________________________________
void G__set_ignoreinclude(G__IgnoreInclude ignoreinclude)
{
   G__ignoreinclude = ignoreinclude;
}

//______________________________________________________________________________
static int G__kindofheader = G__USERHEADER; // Current header file type, user or system.

//______________________________________________________________________________
static int G__copyflag = 0; // Copy source file to a temporary file.

//______________________________________________________________________________
int (*G__ScriptCompiler)(G__CONST char*, G__CONST char*) = 0;

//______________________________________________________________________________
extern "C" void G__RegisterScriptCompiler(int (*p2f)(G__CONST char*, G__CONST char*))
{
   G__ScriptCompiler = p2f;
}

//______________________________________________________________________________
static FILE* G__copytotmpfile(char* prepname)
{
   FILE* ifp = fopen(prepname, "rb");
   if (!ifp) {
      G__genericerror("Internal error: G__copytotmpfile() 1\n");
      return 0;
   }
   FILE* ofp = fopen(G__tmpnam(0), "w+b");
   if (!ofp) {
      G__genericerror("Internal error: G__copytotmpfile() 2\n");
      fclose(ifp);
      return 0;
   }
   G__copyfile(ofp, ifp);
   fclose(ifp);
   fseek(ofp, 0L, SEEK_SET);
   return ofp;
}

//______________________________________________________________________________
static void G__copysourcetotmp(char* prepname, G__input_file* pifile, int fentry)
{
   if (!G__copyflag) {
      return;
   }
   if (prepname[0]) { // FIXME: Check that prepname is not nil.
      return;
   }
   FILE* fpout = fopen(G__tmpnam(0), "w+b");
   if (!fpout) {
      G__genericerror("Internal error: can not open tmpfile.");
      return;
   }
   sprintf(prepname, "(tmp%d)", fentry);
   G__copyfile(fpout, pifile->fp);
   fseek(fpout, 0L, SEEK_SET);
   G__srcfile[fentry].prepname = (char*) malloc(strlen(prepname) + 1);
   strcpy(G__srcfile[fentry].prepname, prepname);
   G__srcfile[fentry].fp = fpout;
   fclose(pifile->fp);
   pifile->fp = fpout;
}

//______________________________________________________________________________
void Cint::Internal::G__setcopyflag(int flag)
{
   G__copyflag = flag;
}

//______________________________________________________________________________
static int G__ispreprocessfilekey(char* filename)
{
   // Check match of the keystring.
   struct G__Preprocessfilekey* pkey = &G__preprocessfilekey;
   for (; pkey->next; pkey = pkey->next) {
      if (pkey->keystring && strstr(filename, pkey->keystring)) {
         return 1;
      }
   }
   // No match.
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__include_file()
{
   // Parse include file name, and then load the include file.
   //
   //  #include   <stdio.h>    \n
   //           ^---------------^       do nothing
   //
   //  #include  comment   "header.h" comment   \n
   //           ^--------------------------------^    load "header.h"
   //
   int result;
   G__StrBuf filename_sb(G__MAXFILENAME);
   char* filename = filename_sb;
   int i = 0;
   int storeit = 0; // collect filename characters flag, valid values: -1, 0, 1
   int store_cpp;
   int store_globalcomp;
   int expandflag = 0;
   static int include_depth = 0; // number of nested includes, that is, include file depth
   int c = G__fgetc();
   for (; (c != '\n') && (c != '\r') && (c != '#'); c = G__fgetc()) { // FIXME: Test for EOF here.
      switch (c) {
         case '<': // Begin of system header name.
            if (!storeit) {
               storeit = 1;
            }
            break;
         case '>': // End of system header name.
            storeit = -1;
            G__kindofheader = G__SYSHEADER;
            break;
         case '\"': // Begin or end of user header name.
            switch (storeit) {
               case -1: // Ignorechar , we are done with the filename.
                  break;
               case 0: // Start of user header name.
                  storeit = 1;
                  break;
               case 1: // End of user header name.
                  storeit = -1;
                  G__kindofheader = G__USERHEADER;
                  break;
            }
            break;
         default:
            if (isspace(c)) {
               if (expandflag) { // At end of macro filename, flag it.
                  storeit = -1;
               }
            }
            else {
               if (storeit == -1) { // Ignore char, we have finished with the filename.
                  // Ignore character, we are finished with the filename.
               }
               else if (storeit == 0) { // We have a macro as the filename, flag it.
                  expandflag = 1;
                  storeit = 1;
                  filename[i++] = c;
                  filename[i] = '\0';
               }
               else if (storeit == 1) { // Collecting filename chars.
                  filename[i++] = c;
                  filename[i] = '\0';
               }
            }
            break;
      }
   }
   if (expandflag) { // Filename was a macro, expand it.
      int ig15 = 0;
      int hash = 0;
      G__hash(filename, hash, ig15);
      ::Reflex::Member var = G__getvarentry(filename, hash,::Reflex::Scope::GlobalScope(), G__p_local);
      if (var) {
         strcpy(filename, *(char**)G__get_offset(var));
         G__kindofheader = G__USERHEADER;
      }
      else {
         G__fprinterr(G__serr, "Error: cannot expand #include %s", filename);
         G__genericerror(0);
         if (c == '#') {
            G__fignoreline();
         }
         return G__LOADFILE_FAILURE;
      }
   }
   store_cpp = G__cpp;
   G__cpp = G__include_cpp;
   store_globalcomp = G__globalcomp;
   if (++include_depth >= G__gcomplevellimit) {
      G__globalcomp = G__NOLINK;
   }
   result = G__loadfile(filename);
   --include_depth;
   G__globalcomp = store_globalcomp;
   G__kindofheader = G__USERHEADER;
   G__cpp = store_cpp;
   if (c == '#') {
      if ((result == G__LOADFILE_FAILURE) && G__ispragmainclude) {
         G__ispragmainclude = 0;
         c = G__fgetname(filename, "\n\r");
         store_globalcomp = G__globalcomp;
         if (++include_depth >= G__gcomplevellimit) {
            G__globalcomp = G__NOLINK;
         }
         if ((c != '\n') && (c != '\r')) {
            result = G__include_file();
         }
         --include_depth;
         G__globalcomp = store_globalcomp;
      }
      else {
         G__fignoreline();
      }
   }
   return result;
}

//______________________________________________________________________________
extern "C" const char* G__getmakeinfo(const char* item)
{
   G__StrBuf makeinfo_sb(G__MAXFILENAME);
   char* makeinfo = makeinfo_sb;
   G__StrBuf line_sb(G__LARGEBUF);
   char* line = line_sb;
   G__StrBuf argbuf_sb(G__LARGEBUF);
   char* argbuf = argbuf_sb;
   static char buf[G__ONELINE];
   buf[0] = '\0';
   char* arg[G__MAXARG];
   int argn;
   char* p;
   FILE* fp;
#ifdef G__HAVE_CONFIG
   if (!strcmp(item, "CPP")) {
      return G__CFG_CXX;
   }
   else if (!strcmp(item, "CC")) {
      return G__CFG_CC;
   }
   else if (!strcmp(item, "DLLPOST")) {
      return G__CFG_SOEXT;
   }
   else if (!strcmp(item, "CSRCPOST")) {
      return ".c";
   }
   else if (!strcmp(item, "CPPSRCPOST")) {
      return ".cxx";
   }
   else if (!strcmp(item, "CHDRPOST")) {
      return ".h";
   }
   else if (!strcmp(item, "CPPHDRPOST")) {
      return ".h";
   }
   else if (!strcmp(item, "INPUTMODE")) {
      return G__CFG_INPUTMODE;
   }
   else if (!strcmp(item, "INPUTMODELOCK")) {
      return G__CFG_INPUTMODELOCK;
   }
   else if (!strcmp(item, "CPREP")) {
      return G__CFG_CPP;
   }
   else if (!strcmp(item, "CPPPREP")) {
      return G__CFG_CPP;
   }
   else {
      printf("G__getmakeinfo for G__HAVE_CONFIG: %s not implemented yet!\n", item);
      return "";
   }
#elif defined(G__NOMAKEINFO)
   return "";
#endif // G__HAVE_CONFIG, G__NOMAKEINFO
   //
   //  Environment variable overrides MAKEINFO file if exists.
   //
   if ((p = getenv(item)) && p[0] && !isspace(p[0])) {
      strcpy(buf, p);
      return buf;
   }
   //
   //  Get information from MAKEINFO file.
   //
   // Get $CINTSYSDIR/MAKEINFO file name
   if (G__getcintsysdir()) {
      return buf;
   }
#ifdef G__VMS
   sprintf(makeinfo, "%sMAKEINFO.txt", G__cintsysdir);
#else // G__VMS
   sprintf(makeinfo, "%s/MAKEINFO", G__cintsysdir);
#endif // G__VMS
   // Open MAKEINFO file.
   fp = fopen(makeinfo, "r");
   if (!fp) {
      G__fprinterr(G__serr, "Error: cannot open %s\n", makeinfo);
      G__fprinterr(G__serr, "!!! There are examples of MAKEINFO files under %s/platform/ !!!\n", G__cintsysdir);
      G__fprinterr(G__serr, "Please refer to these examples and create for your platform\n");
      return buf;
   }
   // Read the MAKEINFO file
   while (G__readline(fp, line, argbuf, &argn, arg)) {
      if ((argn > 2) && !strcmp(arg[1], item)) {
         p = strchr(arg[0], '=');
         if (!p) {
            G__fprinterr(G__serr, "MAKEINFO syntax error\n");
         }
         else {
            do {
               ++p;
            }
            while (isspace(*p));
            strcpy(buf, p);
            fclose(fp);
            return buf;
         }
      }
   }
   fclose(fp);
   return(buf);
}

//______________________________________________________________________________
extern "C" const char* G__getmakeinfo1(const char* item)
{
   return G__getmakeinfo(item);
}

//______________________________________________________________________________
extern "C" void G__SetCINTSYSDIR(const char *cintsysdir)
{
   strcpy(G__cintsysdir, cintsysdir);
}

//______________________________________________________________________________
static int G__UseCINTSYSDIR = 0;

//______________________________________________________________________________
extern "C" void G__SetUseCINTSYSDIR(int UseCINTSYSDIR)
{
   G__UseCINTSYSDIR = UseCINTSYSDIR;
}

//______________________________________________________________________________
int Cint::Internal::G__getcintsysdir()
{
   const char* env = 0;
   if (G__cintsysdir[0] != '*') {
      return EXIT_SUCCESS;
   }


#ifdef G__ROOT
# ifdef ROOTBUILD
   env = "cint";
# else // ROOTBUILD
#  ifdef CINTINCDIR
   env = CINTINCDIR;
#  else // CINTINCDIR
   if (G__UseCINTSYSDIR) {
      env = getenv("CINTSYSDIR");
   }
   else {
      env = getenv("ROOTSYS");
   }
#  endif // CINTINCDIR
# endif // ROOTBUILD
#elif defined(G__WILDC)
   env = getenv("WILDCDIR");
   if (!env) {
      env = getenv("CINTSYSDIR");
   }
   if (!env) {
      env = "C:\\WILDC";
   }
#else // G__ROOT, G__WILDC
   env = getenv("CINTSYSDIR");
# ifdef CINTINCDIR
   if (!env || !env[0]) {
      env = CINTINCDIR;
   }
# endif // CINTINCDIR
# ifdef CINTSYSDIR
   if (!env || !env[0]) {
      env = CINTSYSDIR;
   }
# endif // CINTSYSDIR
# ifdef G__CFG_DATADIRCINT
   if (!env || !env[0]) {
      env = G__CFG_DATADIRCINT;
   }
# endif // G__CFG_DATADIRCINT
#endif // G__ROOT, G__WILDC


   if (env) {
      // --
#ifdef G__ROOT
# ifdef ROOTBUILD
      sprintf(G__cintsysdir, "%s", env);
# else // ROOTBUILD
#  ifdef CINTINCDIR
      sprintf(G__cintsysdir, "%s", CINTINCDIR);
#  else // CINTINCDIR
      if (G__UseCINTSYSDIR) {
         strcpy(G__cintsysdir, env);
      }
      else {
         sprintf(G__cintsysdir, "%s/cint", env);
      }
#  endif // CINTINCDIR
# endif // ROOTBUILD
#else // G__ROOT
      strcpy(G__cintsysdir, env);
#endif // G__ROOT
      return EXIT_SUCCESS;
   }



#ifdef G__EDU_VERSION
   sprintf(G__cintsysdir, ".");
   return EXIT_SUCCESS;
#endif // G__EDU_VERSION


#ifdef G__WIN32
   HMODULE hmodule = 0;
   if (GetModuleFileName(hmodule, G__cintsysdir, G__MAXFILENAME)) {
      char* p = G__strrstr(G__cintsysdir, (char*) G__psep);
      if (p) {
         *p = 0;
      }
# ifdef G__ROOT
      p = G__strrstr(G__cintsysdir, (char*) G__psep);
      if (p) {
         *p = 0;
      }
      strcat(G__cintsysdir, G__psep);
      strcat(G__cintsysdir, "cint");
# endif // G__ROOT
      return EXIT_SUCCESS;
   }
#endif // G__WIN32


#if defined(G__ROOT)
   G__fprinterr(G__serr, "Warning: environment variable ROOTSYS is not set. Standard include files ignored\n");
#elif defined(G__WILDC)
   G__fprinterr(G__serr, "Warning: environment variable WILDCDIR is not set. Standard include files ignored\n");
#else // G__ROOT, G__WILDC
   G__fprinterr(G__serr, "Warning: environment variable CINTSYSDIR is not set. Standard include files ignored\n");
#endif // G__ROOT, G__WILDC


   G__cintsysdir[0] = '\0';
   return EXIT_FAILURE;
}

//______________________________________________________________________________
int Cint::Internal::G__isfilebusy(int ifn)
{
   int flag = 0;
   //
   //  Check global function busy status.
   //
   ::Reflex::Scope ifunc = ::Reflex::Scope::GlobalScope();
   for (
      ::Reflex::Member_Iterator mbr_iter = ifunc.FunctionMember_Begin();
      mbr_iter != ifunc.FunctionMember_End();
      ++mbr_iter
   ) {
      if (
         G__get_funcproperties(*mbr_iter)->entry.busy &&
         (G__get_funcproperties(*mbr_iter)->filenum >= ifn)
      ) {
         G__fprinterr(G__serr, "Function %s() busy. loaded after \"%s\"\n", mbr_iter->Name().c_str(), G__srcfile[ifn].filename);
         ++flag;
      }
   }
   //
   //  Check member function busy status.
   //
   if (
      !G__nfile ||
      (ifn < 0) ||
      (G__nfile <= ifn) ||
      !G__srcfile[ifn].dictpos ||
      (G__srcfile[ifn].dictpos->tagnum == -1)
   ) {
      return flag;
   }
   for (int i2 = G__srcfile[ifn].dictpos->tagnum; i2 < G__struct.alltag; ++i2) {
      ifunc = G__Dict::GetDict().GetScope(i2);
      for (
         ::Reflex::Member_Iterator mbr_iter = ifunc.FunctionMember_Begin();
         mbr_iter != ifunc.FunctionMember_End();
         ++mbr_iter
      ) {
         if (
            G__get_funcproperties(*mbr_iter)->entry.busy &&
            (G__get_funcproperties(*mbr_iter)->filenum >= ifn)
         ) {
            G__fprinterr(G__serr, "Function %s() busy. loaded after \"%s\"\n", mbr_iter->Name().c_str(), G__srcfile[ifn].filename);
            ++flag;
         }
      }
   }
   return flag;
}

//______________________________________________________________________________
int Cint::Internal::G__matchfilename(int i1, const char* filename)
{
   // --
#ifndef __CINT__
# ifdef G__WIN32
   char i1name[_MAX_PATH];
   char fullfile[_MAX_PATH];
# else // G__WIN32
   struct stat statBufItem;
   struct stat statBuf;
# endif // G__WIN32
   if (G__srcfile[i1].filename==0) {
      return 0;
   }
   if (!strcmp(filename, G__srcfile[i1].filename)) {
      return 1;
   }
# ifdef G__WIN32
   _fullpath(i1name, G__srcfile[i1].filename, _MAX_PATH);
   _fullpath(fullfile, filename, _MAX_PATH);
   if (!stricmp(i1name, fullfile)) {
      return 1;
   }
# else // G__WIN32
   if (
      !stat(filename, &statBufItem) &&
      !stat(G__srcfile[i1].filename, &statBuf) &&
      (statBufItem.st_dev == statBuf.st_dev) && // Files on same device
      (statBufItem.st_ino == statBuf.st_ino) && // Files on same inode (but this is not unique on AFS so we need the next 2 test
      (statBufItem.st_size == statBuf.st_size) && // Files of same size
      (statBufItem.st_mtime == statBuf.st_mtime) // Files modified at the same time
   ) {
      return 1;
   }
# endif // G__WIN32
   return 0;
#else // __CINT__
   if (!strcmp(G__srcfile[i1].filename, filename)) {
      return 1;
   }
   char* filenamebase = G__strrstr(G__srcfile[i1].filename, "./");
   if (filenamebase) {
      char* parentdir = G__strrstr(G__srcfile[i1].filename, "../");
      if (!parentdir && !strcmp(filename, filenamebase + 2)) {
         G__StrBuf buf_sb(G__ONELINE);
         char* buf = buf_sb;
# ifdef G__WIN32
         char* p;
# endif // G__WIN32
         if (filenamebase == G__srcfile[i1].filename) {
            return 1;
         }
# ifdef G__WIN32
         GetCurrentDirectory(G__ONELINE, buf);
         p = buf;
         while ((p = strchr(p, '\\'))) {
            *p = '/';
         }
         if ((strlen(buf) > 1) && (buf[1] == ':')) {
            G__StrBuf buf2_sb(G__ONELINE);
            char* buf2 = buf2_sb;
            strcpy(buf2, buf + 2);
            strcpy(buf, buf2);
         }
# elif defined(G__POSIX) || defined(G__ROOT)
         getcwd(buf, G__ONELINE);
# else // G__WIN32, G__POSIX || G__ROOT
         buf[0] = 0;
# endif // G__WIN32, G__POSIX || G__ROOT
         if (!strncmp(buf, G__srcfile[i1].filename, filenamebase - G__srcfile[i1].filename - 1)) {
            return 1;
         }
      }
   }
   return 0;
#endif // __CINT__
   // --
}

//______________________________________________________________________________
extern "C" const char* G__stripfilename(const char* filename)
{
   if (!filename) {
      return "";
   }
   const char* filenamebase = G__strrstr(filename, "./");
   if (filenamebase) {
      const char* parentdir = G__strrstr(filename, "../");
      G__StrBuf buf_sb(G__ONELINE);
      char* buf = buf_sb;
#ifdef G__WIN32
      char *p;
#endif // G__WIN32
      if (parentdir) {
         return filename;
      }
      if (filenamebase == filename) {
         return filenamebase + 2;
      }
#ifdef G__WIN32
      GetCurrentDirectory(G__ONELINE, buf);
      p = buf;
      while ((p = strchr(p, '\\'))) {
         *p = '/';
      }
      if ((strlen(buf) > 1) && (buf[1] == ':')) {
         G__StrBuf buf2_sb(G__ONELINE);
         char* buf2 = buf2_sb;
         strcpy(buf2, buf + 2);
         strcpy(buf, buf2);
      }
#elif defined(G__POSIX) || defined(G__ROOT)
      getcwd(buf, G__ONELINE);
#else // G__WIN32, G__POSIX || G__ROOT
      buf[0] = 0;
#endif // G__WIN32, G__POSIX || G__ROOT
      if (!strncmp(buf, filename, filenamebase - filename - 1)) {
         return filenamebase + 2;
      }
   }
   return filename;
}

//______________________________________________________________________________
void Cint::Internal::G__smart_unload(int ifn)
{
   struct G__dictposition* dictpos = G__srcfile[ifn].dictpos;
   struct G__dictposition* hasonlyfunc = G__srcfile[ifn].hasonlyfunc;
   if (G__nfile == hasonlyfunc->nfile) {
      ::Reflex::Scope var = ::Reflex::Scope::GlobalScope();
      if ((var == hasonlyfunc->var) && ( ((int)var.DataMemberSize()) == hasonlyfunc->ig15)) {
         G__scratch_upto(G__srcfile[ifn].dictpos);
         return;
      }
   }
   // disable functions
   ::Reflex::Scope ifunc = dictpos->ifunc;
   ifn = dictpos->ifn;
   while (ifunc && ((ifunc != hasonlyfunc->ifunc) || (ifn != hasonlyfunc->ifn)) && ifunc.FunctionMemberAt(ifn)) {
      ifunc.RemoveFunctionMember(ifunc.FunctionMemberAt(ifn));
      ++ifn;
   }
   // disable file entry
   for (int i = dictpos->nfile; i < hasonlyfunc->nfile; ++i) {
      G__srcfile[i].hash = 0;
      G__srcfile[i].filename[0] = 0;
      if (G__srcfile[i].fp) {
         fclose(G__srcfile[i].fp);
      }
      G__srcfile[i].fp = 0;
   }
   // unload shared library
   for (int i = dictpos->allsl; i < hasonlyfunc->allsl; ++i) {
      G__smart_shl_unload(i); // TODO: Replace this tail recursion with non-recursive implementation.
   }
}

//______________________________________________________________________________
extern "C" int G__unloadfile(const char* filename)
{
   // Check if function is busy, if so return -1, otherwise unload file and return 0.
   int ifn;
   int i1 = 0;
   int flag;
   char* scope;
   int envtagnum;
   G__StrBuf buf_sb(G__MAXFILENAME);
   char* buf = buf_sb;
   G__LockCriticalSection();
   strcpy(buf, filename);
   char* fname = G__strrstr(buf, "::");
   if (!fname) {
      fname = (char*) filename;
      envtagnum = -1;
   }
   else {
      scope = buf;
      *fname = 0;
      fname += 2;
      if (!scope[0]) {
         envtagnum = -1;
      }
      else {
         envtagnum = G__defined_tagname(scope, 2);
         if (envtagnum == -1) {
            G__fprinterr(G__serr, "Error: G__unloadfile() File \"%s\" scope not found ", scope);
            G__genericerror(0);
            G__UnlockCriticalSection();
            return G__UNLOADFILE_FAILURE;
         }
      }
   }
   //
   //  Check if file is already loaded, if not return.
   //
   int hash = 0;
   int i2 = 0;
   G__hash(fname, hash, i2)
   flag = 0;
   while (i1 < G__nfile) {
      if (
         G__matchfilename(i1, fname) &&
         (
            (envtagnum == -1) ||
            (envtagnum == G__srcfile[i1].parent_tagnum)
         )
      ) {
         flag = 1;
         break;
      }
      ++i1;
   }
   if (!flag) {
      G__fprinterr(G__serr, "Error: G__unloadfile() File \"%s\" not loaded ", filename);
      G__genericerror(0);
      G__UnlockCriticalSection();
      return G__UNLOADFILE_FAILURE;
   }
   //
   //  Set G__ifile index number to ifn.
   //
   ifn = i1;
   //
   //  If function in unloaded files are busy, cancel unloading.
   //
   if (G__isfilebusy(ifn)) {
      G__fprinterr(G__serr, "Error: G__unloadfile() Can not unload \"%s\", file busy ", filename);
      G__genericerror(0);
      G__UnlockCriticalSection();
      return G__UNLOADFILE_FAILURE;
   }
   if (G__srcfile[ifn].hasonlyfunc && G__do_smart_unload) {
      G__smart_unload(ifn);
   }
   else {
      G__scratch_upto(G__srcfile[ifn].dictpos);
   }
   if (G__debug) {
      G__fprinterr(G__serr, "File=%s unloaded\n", filename);
   }
   G__UnlockCriticalSection();
   return G__UNLOADFILE_SUCCESS;
}

//______________________________________________________________________________
static int G__isbinaryfile(char* filename)
{
   int c;
   int prev = 0;
   int i;
   int badflag = 0;
   int comflag = 0;
#ifdef G__VISUAL
   char buf[11];
#endif // G__VISUAL
   int unnamedmacro = 0;
   int alphaflag = 0;
   int store_lang = G__lang;
   if (G__lang != G__ONEBYTE) {
      G__lang = G__UNKNOWNCODING;
   }
   // Read 10 byte from beginning of the file.
   // Set badflag if unprintable char is found.
   for (i = 0; i < 10; ++i) {
      c = fgetc(G__ifile.fp);
      if (G__IsDBCSLeadByte(c)) {
         c = fgetc(G__ifile.fp);
         if (c != EOF) {
            G__CheckDBCS2ndByte(c);
         }
      }
      else {
         if (
            (c != EOF) &&
            (c != '\t') &&
            (c != '\n') &&
            (c != '\r') &&
            !isprint(c) &&
            !comflag // not comment flag
         ) {
            ++badflag;
         }
         else if (
            (prev == '/') &&
            (
               (c == '/') ||
               (c == '*')
            )
         ) {
            comflag = 1; // Set comment flag.
         }
         else if (
            (c == '{') &&
            !alphaflag &&
            !comflag
         ) {
            unnamedmacro = 1;
         }
         else if (isalpha(c)) {
            ++alphaflag;
         }
      }
      prev = c;
      if (c == EOF) {
         break;
      }
#ifdef G__VISUAL
      buf[i] = c;
#endif // G__VISUAL
      // --
   }
   if (badflag) {
      G__fprinterr(G__serr, "Error: Bad source file(binary) %s", filename);
      G__genericerror(0);
      G__return = G__RETURN_EXIT1;
      G__lang = store_lang;
      return 1;
   }
   else if (unnamedmacro) {
      G__fprinterr(G__serr, "Error: Bad source file(unnamed macro) %s", filename);
      G__genericerror(0);
      G__fprinterr(G__serr, "  unnamed macro has to be executed by 'x' command\n");
      G__return = G__RETURN_EXIT1;
      G__lang = store_lang;
      return 1;
   }
   else {
      // --
#ifdef G__VISUAL
      buf[10] = 0;
      if (!strncmp(buf, "Microsoft ", 10)) {
         // Skip this compiler message:
         //
         //      Microsoft (R) 32-bit C/C++ Optimizing Compiler Version 11.00.7022 for 80x86
         //      Copyright (C) Microsoft Corp 1984-1997. All rights reserved.
         //
         //      2.c
         //
         G__fignoreline();
         c = G__fgetc(); // \r\n
         G__fignoreline();
         c = G__fgetc(); // \r\n
         G__fignoreline();
         c = G__fgetc(); // \r\n
         G__fignoreline();
         c = G__fgetc(); // \r\n
      }
      else {
         fseek(G__ifile.fp, SEEK_SET, 0);
      }
#else // G__VISUAL
      fseek(G__ifile.fp, SEEK_SET, 0);
#endif // G__VISUAL
      // --
   }
   G__lang = store_lang;
   return 0;
}

//______________________________________________________________________________
static void G__checkIfOnlyFunction(int fentry)
{
   G__dictposition* dictpos = G__srcfile[fentry].dictpos;

   // Sum the number of G__struct slot used by any of the file
   // we enclosed:
   int nSubdefined = 0;
   for(int filecur = fentry+1; filecur < G__nfile; ++filecur) {
     nSubdefined += G__srcfile[filecur].definedStruct;
   }
   G__srcfile[fentry].definedStruct = G__struct.nactives - dictpos->nactives - nSubdefined;
   
   int tagflag = ( G__srcfile[fentry].definedStruct == 0 ); 
  
   int varflag = 1;
   ::Reflex::Scope var = ::Reflex::Scope::GlobalScope();
   if (dictpos->var == var && dictpos->ig15 == ((int)var.DataMemberSize())) {
      varflag = 1;
   }
   else {
      ::Reflex::Scope var2 = dictpos->var;
      unsigned int ig152 = dictpos->ig15;
      while (var2 && ((var2 != var) || (ig152 != var.DataMemberSize()))) {
         if (G__get_type(var2.DataMemberAt(ig152).TypeOf()) != 'p') {
            varflag = 0;
            break;
         }
         if (ig152 > var2.DataMemberSize()) {
            break;
         }
         ++ig152;
      }
   }
   G__Deffuncmacro* deffuncmacro = &G__deffuncmacro;
   while (deffuncmacro->next) {
      deffuncmacro = deffuncmacro->next;
   }
   G__Definedtemplateclass* definedtemplateclass = &G__definedtemplateclass;
   while (definedtemplateclass->next) {
      definedtemplateclass = definedtemplateclass->next;
   }
   G__Definetemplatefunc* definedtemplatefunc = &G__definedtemplatefunc;
   while (definedtemplatefunc->next) {
      definedtemplatefunc = definedtemplatefunc->next;
   }
   if (
      tagflag &&
      !dictpos->typenum &&
      varflag &&
      (deffuncmacro == dictpos->deffuncmacro) &&
      (definedtemplateclass == dictpos->definedtemplateclass) &&
      (definedtemplatefunc == dictpos->definedtemplatefunc)
   ) {
      G__srcfile[fentry].hasonlyfunc = (G__dictposition*) malloc(sizeof(G__dictposition));
      G__store_dictposition(G__srcfile[fentry].hasonlyfunc);
   }
}

//______________________________________________________________________________
int Cint::Internal::G__loadfile_tmpfile(FILE* fp)
{
   int store_prerun;
   ::Reflex::Scope store_p_local;
   int store_var_type;
   ::Reflex::Scope store_tagnum;
   ::Reflex::Type store_typenum;
   int fentry;
   int store_nobreak;
   int store_step;
   struct G__input_file store_file;
   int store_macroORtemplateINfile;
   short store_iscpp;
   G__UINT32 store_security;
   ::Reflex::Member store_func_now;
   int pragmacompile_iscpp;
   int pragmacompile_filenum;
   int store_asm_noverflow;
   int store_no_exec_compile;
   int store_asm_exec;
   int store_return;
   char* store_struct_offset;
   int hash;
   int temp;
   char hdrprop = G__NONCINTHDR;
   //
   //  Check if number of loaded file exceeds G__MAXFILE.
   //  If so, restore G__ifile reset G__eof and return.
   //
   if (G__nfile == G__MAXFILE) {
      G__fprinterr(G__serr, "Limitation: Sorry, can not load any more files\n");
      return G__LOADFILE_FATAL;
   }
   if (!fp) {
      G__genericerror("Internal error: G__loadfile_tmpfile((FILE*)NULL)");
      return G__LOADFILE_FATAL;
   }
   G__LockCriticalSection();
   //
   //  Store current input file information.
   //
   store_file = G__ifile;
   store_step = G__step;
   G__step = 0;
   G__setdebugcond();
   //
   //  prerun, read whole ifuncs to allocate global
   //  variables and make ifunc table.
   //
   //  Store iscpp flag. This flag is modified
   //  in G__preprocessor() function and restored
   //  in G__loadfile() before return.
   //
   store_iscpp = G__iscpp;
   //
   //  filenum and line_number.
   //
   G__ifile.line_number = 1;
   G__ifile.fp = fp;
   G__ifile.filenum = G__nfile;
   fentry = G__nfile;
   sprintf(G__ifile.name, "(tmp%d)", fentry);
   G__hash(G__ifile.name, hash, temp);
   G__srcfile[fentry].dictpos = (G__dictposition*) malloc(sizeof(G__dictposition));
   G__store_dictposition(G__srcfile[fentry].dictpos);
   G__srcfile[fentry].hdrprop = hdrprop;
   store_security = G__security;
   G__srcfile[fentry].security = G__security;
   G__srcfile[fentry].prepname = 0;
   G__srcfile[fentry].hash = hash;
   G__srcfile[fentry].filename = (char*) malloc(strlen(G__ifile.name) + 1);
   strcpy(G__srcfile[fentry].filename, G__ifile.name);
   G__srcfile[fentry].fp = G__ifile.fp;
   G__srcfile[fentry].included_from = store_file.filenum;
   G__srcfile[fentry].ispermanentsl = G__ispermanentsl;
   G__srcfile[fentry].initsl = 0;
   G__srcfile[fentry].hasonlyfunc = 0;
   G__srcfile[fentry].parent_tagnum = G__get_tagnum(G__get_envtagnum());
   G__srcfile[fentry].slindex = -1;
   G__srcfile[fentry].breakpoint = 0;
   ++G__nfile;
   if (G__debugtrace) {
      G__fprinterr(G__serr, "LOADING tmpfile\n");
   }
   if (G__debug) {
      G__fprinterr(G__serr, "%-5d", G__ifile.line_number);
   }
   store_prerun = G__prerun;
   store_p_local = G__p_local;
   if (
      !G__def_struct_member ||
      !G__tagdefining ||
      (
         (G__struct.type[G__get_tagnum(G__tagdefining)] != 'n') &&
         (G__struct.type[G__get_tagnum(G__tagdefining)] != 'c') &&
         (G__struct.type[G__get_tagnum(G__tagdefining)] != 's')
      )
   ) {
      G__p_local = 0;
   }
   G__eof = 0;
   G__prerun = 1;
   store_nobreak = G__nobreak;
   G__nobreak = 1;
   store_var_type = G__var_type;
   store_tagnum = G__tagnum;
   store_typenum = G__typenum;
   store_func_now = G__func_now;
   G__func_now = ::Reflex::Member();
   store_macroORtemplateINfile = G__macroORtemplateINfile;
   G__macroORtemplateINfile = 0;
   store_asm_noverflow = G__asm_noverflow;
   store_no_exec_compile = G__no_exec_compile;
   store_asm_exec = G__asm_exec;
   G__asm_noverflow = 0;
   G__no_exec_compile = 0;
   G__asm_exec = 0;
   store_return = G__return;
   G__return = G__RETURN_NON;
   store_struct_offset = G__store_struct_offset;
   G__store_struct_offset = 0;
   //
   //  Load and parse source file in prerun environment.
   //
   while (!G__eof && (G__return < G__RETURN_EXIT1)) {
      int brace_level = 0;
      G__exec_statement(&brace_level);
   }
   G__store_struct_offset = store_struct_offset;
   pragmacompile_filenum = G__ifile.filenum;
   pragmacompile_iscpp = G__iscpp;
   G__func_now = store_func_now;
   G__macroORtemplateINfile = store_macroORtemplateINfile;
   G__var_type = store_var_type;
   G__tagnum = store_tagnum;
   G__typenum = store_typenum;
   G__nobreak = store_nobreak;
   G__prerun = store_prerun;
   G__p_local = store_p_local;
   G__asm_noverflow = store_asm_noverflow;
   G__no_exec_compile = store_no_exec_compile;
   G__asm_exec = store_asm_exec;
   G__ifile = store_file;
   G__eof = 0;
   G__step = store_step;
   G__setdebugcond();
   G__globalcomp = G__store_globalcomp;
   G__iscpp = store_iscpp;
#ifdef G__SECURITY
   G__security = store_security;
#endif // G__SECURITY
   if (G__return > G__RETURN_NORMAL) {
      G__UnlockCriticalSection();
      return G__LOADFILE_FAILURE;
   }
#ifdef G__AUTOCOMPILE
   //
   //  If '#pragma compile' appears in source code.
   //
   if (G__fpautocc && (G__autoccfilenum == pragmacompile_filenum)) {
      store_iscpp = G__iscpp;
      G__iscpp = pragmacompile_iscpp;
      G__autocc();
      G__iscpp = store_iscpp;
   }
#endif // G__AUTOCOMPILE
   G__checkIfOnlyFunction(fentry);
   G__UnlockCriticalSection();
   return fentry + 2;
}

//______________________________________________________________________________
int Cint::Internal::G__statfilename(const char *filenamein, struct stat *statBuf)
{
   // Use the same search algorithm as G__loadfile to do a 'stat' on the file.
   char filename[G__ONELINE];
   char workname[G__ONELINE];
   int hash,temp;
   char addpost[3][8];
   int res = -1;
   
   strcpy(filename,filenamein);
   
   /*************************************************
    * delete space chars at the end of filename
    *************************************************/
   int len = strlen(filename);
   while(len>1&&isspace(filename[len-1])) {
      filename[--len]='\0';
   }
   
   G__hash(filename,hash,temp);

   strcpy(addpost[0],"");
   strcpy(addpost[1],".h");
   
   strcpy(addpost[2],"");
   for(int i2=0;i2<3;i2++) {
      if(2==i2) {
         if((len>3&& (strcmp(filename+len-3,".sl")==0 ||
                      strcmp(filename+len-3,".dl")==0 ||
                      strcmp(filename+len-3,".so")==0))) {
            strcpy(filename+len-3,G__getmakeinfo1("DLLPOST"));
         }
         else if((len>4&& (strcmp(filename+len-4,".dll")==0 ||
                           strcmp(filename+len-4,".DLL")==0))) {
            strcpy(filename+len-4,G__getmakeinfo1("DLLPOST"));
         }
         else if((len>2&& (strcmp(filename+len-2,".a")==0 ||
                           strcmp(filename+len-2,".A")==0))) {
            strcpy(filename+len-2,G__getmakeinfo1("DLLPOST"));
         }
#if defined(R__FBSD) || defined(R__OBSD)
         else if (len>strlen(soext) &&
                  strcmp(filename+len-strlen(soext),soext)==0) {
            strcpy(filename+len-strlen(soext),G__getmakeinfo1("DLLPOST"));
         }
#endif
      }
      
      /**********************************************
       * If it's a "" header with a relative path, first
       * try relative to the current input file.
       * (This corresponds to the behavior of gcc.)
       **********************************************/
      if (G__USERHEADER == G__kindofheader &&
#ifdef G__WIN32
          filename[0] != '/' &&
          filename[0] != '\\' &&
#else
          filename[0] != G__psep[0] &&
#endif
          G__ifile.name[0] != '\0') {
         char* p;
         strcpy(workname,G__ifile.name);
#ifdef G__WIN32
         p = strrchr (workname, '/');
         {
            char* q = strrchr (workname, '\\');
            if (q && q > p)
               p = q;
         }
#else
         p = strrchr (workname, G__psep[0]);
#endif
         if (p == 0) p = workname;
         else ++p;
         strcpy (p, filename);
         strcat (p, addpost[i2]);
         
         res = stat( workname, statBuf );
         
         if (res==0) return res;
      }
      /**********************************************
       * try ./filename
       **********************************************/
      if(G__USERHEADER==G__kindofheader) {
         sprintf(workname,"%s%s",filename,addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }  else {
         // Do we need that or not?
         // G__kindofheader = G__USERHEADER;
      }
      /**********************************************
       * try includepath/filename
       **********************************************/
      struct G__includepath *ipath = &G__ipathentry;
      while(res!=0 && ipath->pathname) {
         sprintf(workname,"%s%s%s%s"
                 ,ipath->pathname,G__psep,filename,addpost[i2]);
         res = stat( workname, statBuf );         
         ipath = ipath->next;
      }
      if (res==0) return res;
      
      G__getcintsysdir();

      /**********************************************
       * try $CINTSYSDIR/stl
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
         sprintf(workname,"%s%s%s%sstl%s%s%s",G__cintsysdir,G__psep,G__CFG_COREVERSION
                 ,G__psep,G__psep,filename,addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }

      /**********************************************
       * try $CINTSYSDIR/lib
       **********************************************/
      /* G__getcintsysdir(); */
      if('\0'!=G__cintsysdir[0]) {
         sprintf(workname,"%s%s%s%slib%s%s%s",G__cintsysdir,G__psep,G__CFG_COREVERSION
                 ,G__psep,G__psep,filename,addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }

#ifdef G__EDU_VERSION
      /**********************************************
       * try include/filename
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
         sprintf(workname,"include%s%s%s"
                 ,G__psep,filename,addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }
      /**********************************************
       * try stl
       **********************************************/
      G__getcintsysdir();
      if('\0'!=G__cintsysdir[0]) {
         sprintf(workname,"stl%s%s%s"
                 ,G__psep,filename,addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }         
#endif /* G__EDU_VERSION */
      
#ifdef G__VISUAL
      /**********************************************
       * try /msdev/include
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
         sprintf(workname,"/msdev/include/%s%s",filename,addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }
#endif /* G__VISUAL */
         
#ifdef G__SYMANTEC
      /**********************************************
       * try /sc/include
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
         sprintf(workname,"/sc/include/%s%s",filename,addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }
#endif // G__SYMANTEC
         
#ifndef G__WIN32
      /**********************************************
       * try /usr/include/filename
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
         sprintf(workname,"/usr/include/%s%s",filename,addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }
#endif
      
#ifdef __GNUC__
      /**********************************************
       * try /usr/include/g++/filename
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
         sprintf(workname,"/usr/include/g++/%s%s",filename,addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }
#endif /* __GNUC__ */
      
#ifndef G__WIN32
      /* #ifdef __hpux */
      /**********************************************
       * try /usr/include/CC/filename
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
         sprintf(workname,"/usr/include/CC/%s%s",filename,addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }         
#endif
         
#ifndef G__WIN32
      /**********************************************
       * try /usr/include/codelibs/filename
       **********************************************/
      if('\0'!=G__cintsysdir[0]) {
         sprintf(workname,"/usr/include/codelibs/%s%s"
                 ,filename,addpost[i2]);
         res = stat( workname, statBuf );         
         if (res==0) return res;
      }
#endif
   }
   return -1;
}
   
   
//______________________________________________________________________________
extern "C" int G__loadfile(const char* filenamein)
{
   //  0) If .sl .dl .so .dll .DLL call G__shl_load()
   //  1) check G__MAXFILE                       return -2 if fail(fatal)
   //  2) check if file is already loaded        return 1 if already loaded
   //  3) Open filename
   //  4) If fp==NULL, search include path
   //  5) Set filename and linenumber
   //  6) If still fp==NULL                      return -1
   //  7) LOAD filename
   //  8) If G__return>G__RETURN_NORMAL          return -1
   //  9)                                        return 0
   FILE* tmpfp;
   int external_compiler = 0;
   const char* compiler_option = "";
   int store_prerun;
   int i1 = 0;
   ::Reflex::Scope store_p_local;
   int store_var_type;
   ::Reflex::Scope store_tagnum;
   ::Reflex::Type store_typenum;
   int fentry;
   struct G__includepath* ipath;
   int store_nobreak;
#ifdef G__TMPFILE
   G__StrBuf prepname_sb(G__MAXFILENAME);
   char* prepname = prepname_sb;
#else // G__TMPFILE
   char prepname[L_tmpnam+10];
#endif // G__TMPFILE
   int store_step;
   int null_entry = -1;
   struct G__input_file store_file;
   int hash;
   int temp;
   int store_macroORtemplateINfile;
   int len;
#ifdef G__SHAREDLIB
   int len1;
   const char* dllpost;
#endif
   short store_iscpp;
   G__UINT32 store_security;
   char addpost[3][8];
   int i2;
   ::Reflex::Member store_func_now;
   int pragmacompile_iscpp;
   int pragmacompile_filenum;
   int store_asm_noverflow;
   int store_no_exec_compile;
   int store_asm_exec;
#if defined(R__FBSD) || defined(R__OBSD) // Free BSD or Open BSD
   char soext[] = SOEXT;
#endif // R__FBSD || R__OBSD
   char hdrprop = G__NONCINTHDR;
   G__StrBuf filename_sb(G__ONELINE);
   char* filename = filename_sb;
   strcpy(filename, filenamein);
   //
   // delete space chars at the end of filename
   //
   len = strlen(filename);
   while ((len > 1) && isspace(filename[len-1])) {
      filename[--len] = '\0';
   }
   //
   // Check if the filename has an extension ending
   // in double plus, e.g., script.cxx++ or script.C++.
   // Ending with only one + means to keep the shared
   // library after the end of this process.
   // The + or ++ can also be followed by either a 'g'
   // or an 'O' which means respectively to compile
   // in debug or optimized mode.
   //
   compiler_option = 0;
   if (
      (len > 2) &&
      !strncmp(filename + len - 2, "+", 1) &&
      (
         !strcmp(filename + len - 1, "O") ||
         !strcmp(filename + len - 1, "g")
      )
   ) {
      compiler_option = filename + len - 1;
      len -= 1;
   }
   if (
      (len > 1) &&
      !strncmp(filename + len - 1, "+", 1)
   ) {
      if (
         (len > 2) &&
         !strncmp(filename + len - 2, "++", 2)
      ) {
         if (compiler_option) {
            switch (compiler_option[0]) {
               case 'O':
                  compiler_option = "kfO";
                  break;
               case 'g':
                  compiler_option = "kfg";
                  break;
               default:
                  G__genericerror("Should not have been reached!");
            }
         }
         else {
            compiler_option = "kf";
         }
         len -= 2;
      }
      else {
         if (compiler_option) {
            switch (compiler_option[0]) {
               case 'O':
                  compiler_option = "kO";
                  break;
               case 'g':
                  compiler_option = "kg";
                  break;
               default:
                  G__genericerror("Should not have been reached!");
            }
         }
         else {
            compiler_option = "k";
         }
         len -= 1;
      }
      filename[len] = '\0';
      external_compiler = 1; // Request external compilation if available (in ROOT)
      if (G__ScriptCompiler) {
         int noterr = (*G__ScriptCompiler)(filename, compiler_option);
         if (noterr) {
            return G__LOADFILE_SUCCESS;
         }
         return G__LOADFILE_FAILURE;
      }
   }
   G__LockCriticalSection();
   store_file = G__ifile;
   store_step = G__step;
   G__step = 0;
   G__setdebugcond();
   //
   // prerun, read whole ifuncs to allocate global variables and make ifunc table.
   //
   // check if number of loaded file exceeds G__MAXFILE
   // if so, restore G__ifile reset G__eof and return.
   //
   if (G__nfile == G__MAXFILE) {
      G__fprinterr(G__serr, "Limitation: Sorry, can not load any more files\n");
      G__ifile = store_file;
      G__eof = 0;
      G__step = store_step;
      G__setdebugcond();
      G__UnlockCriticalSection();
      return G__LOADFILE_FATAL;
   }
   G__hash(filename, hash, temp);
   //
   // check if file is already loaded.
   // if so, restore G__ifile reset G__eof and return.
   //
   for (; i1 < G__nfile; ++i1) {
      if (!G__srcfile[i1].filename) {
         // This entry was unloaded by G__unloadfile()
         // Then remember the entry index into 'null_entry'.
         if (null_entry == -1) {
            null_entry = i1;
         }
      }
      //
      // check if alreay loaded
      //
      if (
         G__matchfilename(i1, filename) &&
         (G__get_envtagnum() == G__Dict::GetDict().GetScope(G__srcfile[i1].parent_tagnum))
      ) {
         if ((!G__prerun || G__debugtrace) && (G__dispmsg >= G__DISPNOTE)) {
            static const char* excludelist [] = {
               "climits.dll"
               "complex.dll",
               "deque.dll",
               "exception.dll",
               "ipc.dll",
               "list.dll",
               "map.dll",
               "map2.dll",
               "multimap.dll",
               "multimap2.dll",
               "multiset.dll",
               "posix.dll"
               "posix.dll",
               "queue.dll",
               "set.dll",
               "stack.dll",
               "stdcxxfunc.dll",
               "stdexcept.dll",
               "stdfunc.dll",
               "string.dll",
               "valarray.dll",
               "vector.dll",
               "vectorbool.dll",
            };
            static const unsigned int excludelistsize = sizeof(excludelist) / sizeof(excludelist[0]);
            static int excludelen[excludelistsize] = { -1 };
            // Initialize excludelen array.
            if (excludelen[0] == -1) {
               for (unsigned int i = 0; i < excludelistsize; ++i) {
                  excludelen[i] = strlen(excludelist[i]);
               }
            }
            bool cintdlls = false;
            int local_len = strlen(filename);
            for (unsigned int i = 0; !cintdlls && (i < excludelistsize); ++i) {
               if (local_len >= excludelen[i]) {
                  cintdlls = !strncmp(filename + local_len - excludelen[i], excludelist[i], excludelen[i]);
               }
            }
            if (!cintdlls) {
               G__fprinterr(G__serr, "Note: File \"%s\" already loaded\n", filename);
            }
         }
         G__ifile = store_file;
         G__eof = 0;
         G__step = store_step;
         G__setdebugcond();
         G__UnlockCriticalSection();
         return G__LOADFILE_DUPLICATE;
      }
   }
   //
   // Get actual open file name.
   //
   if (!G__cpp) {
      G__cpp = G__ispreprocessfilekey(filename);
   }
   //
   // store iscpp (is C++) flag. This flag is modified in G__preprocessor()
   // function and restored in G__loadfile() before return.
   //
   store_iscpp = G__iscpp;
   //
   // Get actual open file name.
   //
   int pres = G__preprocessor(prepname, filename, G__cpp, G__macros, /*FIXME*/(char*)G__undeflist, G__ppopt, G__allincludepath);
   if (pres) {
      G__fprinterr(G__serr, "Error: external preprocessing failed.");
      G__genericerror(0);
      G__UnlockCriticalSection();
      return G__LOADFILE_FAILURE;
   }
   //
   // open file
   //
   if (prepname[0]) {
      //
      // -p option. open preprocessed tmpfile
      //
      sprintf(G__ifile.name, "%s", filename);
#ifndef G__OLDIMPLEMENTATION1922
      if (G__fons_comment && G__cpp && (G__globalcomp != G__NOLINK)) {
         // --
#ifndef G__WIN32
         G__ifile.fp = fopen(prepname, "r");
#else // G__WIN32
         G__ifile.fp = fopen(prepname, "rb");
#endif // G__WIN32
         // --
      }
      else {
         G__ifile.fp = G__copytotmpfile(prepname);
         if (G__ifile.fp) {
            remove(prepname);
            strcpy(prepname, "(tmpfile)");
         }
         else {
            // --
#ifndef G__WIN32
            G__ifile.fp = fopen(prepname, "r");
#else // G__WIN32
            G__ifile.fp = fopen(prepname, "rb");
#endif // G__WIN32
            // --
         }
      }
#else // G__OLDIMPLEMENTATION1922
      G__ifile.fp = G__copytotmpfile(prepname);
      remove(prepname);
      strcpy(prepname, "(tmpfile)");
#endif // G__OLDIMPLEMENTATION1922
      G__kindofheader = G__USERHEADER;
   }
   else {
      strcpy(addpost[0], "");
      strcpy(addpost[1], ".h");
      strcpy(addpost[2], "");
      for (i2 = 0; i2 < 3; ++i2) {
         if (i2 == 2) {
            if (
               (len > 3) &&
               (
                  !strcmp(filename + len - 3, ".sl") ||
                  !strcmp(filename + len - 3, ".dl") ||
                  !strcmp(filename + len - 3, ".so")
               )
            ) {
               strcpy(filename + len - 3, G__getmakeinfo1("DLLPOST"));
            }
            else if (
               (len > 4) &&
               (
                  !strcmp(filename + len - 4, ".dll") ||
                  !strcmp(filename + len - 4, ".DLL")
               )
            ) {
               strcpy(filename + len - 4, G__getmakeinfo1("DLLPOST"));
            }
            else if (
               (len > 2) &&
               (
                  !strcmp(filename + len - 2, ".a") ||
                  !strcmp(filename + len - 2, ".A")
               )
            ) {
               strcpy(filename + len - 2, G__getmakeinfo1("DLLPOST"));
            }
#if defined(R__FBSD) || defined(R__OBSD) // Free BSD || Open BSD
            else if (
               (len > strlen(soext)) &&
               !strcmp(filename + len - strlen(soext), soext)
            ) {
               strcpy(filename + len - strlen(soext), G__getmakeinfo1("DLLPOST"));
            }
#endif // R__FBSD || R__OBSD
            // --
         }
         G__ifile.fp = 0;
         //
         // If it's a user header with a relative path, first
         // try relative to the current input file.
         // (This corresponds to the behavior of gcc.)
         //
         if (
            (G__kindofheader == G__USERHEADER) &&
#ifdef G__WIN32
            (filename[0] != '/') &&
            (filename[0] != '\\') &&
#else // G__WIN32
            (filename[0] != G__psep[0]) &&
#endif // G__WIN32
            store_file.name[0]
         ) {
            char* p = 0;
            strcpy(G__ifile.name, store_file.name);
#ifdef G__WIN32
            p = strrchr(G__ifile.name, '/');
            {
               char* q = strrchr(G__ifile.name, '\\');
               if (q && (q > p)) {
                  p = q;
               }
            }
#else // G__WIN32
            p = strrchr(G__ifile.name, G__psep[0]);
#endif // G__WIN32
            if (!p) {
               p = G__ifile.name;
            }
            else {
               ++p;
            }
            strcpy(p, filename);
            strcat(p, addpost[i2]);
#ifndef G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "r");
#else // G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "rb");
#endif // G__WIN32
            // --
         }
         if (G__ifile.fp) {
            break;
         }
         //
         // try ./filename
         //
         if (G__kindofheader == G__USERHEADER) {
            // --
#ifdef G__VMS
            sprintf(G__ifile.name, "%s", filename);
#else // G__VMS
            sprintf(G__ifile.name, "%s%s", filename, addpost[i2]);
#endif // G__VMS
#ifndef G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "r");
#else // G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "rb");
#endif // G__WIN32
            // --
         }
         else {
            G__ifile.fp = 0;
            G__kindofheader = G__USERHEADER;
         }
         if (G__ifile.fp) {
            break;
         }
         //
         // try includepath/filename
         //
         ipath = &G__ipathentry;
         while (!G__ifile.fp && ipath->pathname) {
            sprintf(G__ifile.name, "%s%s%s%s", ipath->pathname, G__psep, filename, addpost[i2]);
#ifndef G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "r");
#else // G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "rb");
#endif // G__WIN32
            ipath = ipath->next;
            {
               G__ConstStringList* sysdir = G__SystemIncludeDir;
               while (sysdir) {
                  if (!strncmp(sysdir->string, G__ifile.name, sysdir->hash)) {
                     G__globalcomp = G__NOLINK;
                     hdrprop = G__CINTHDR;
                  }
                  sysdir = sysdir->prev;
               }
            }
         }
         if (G__ifile.fp) {
            break;
         }
         //
         // try $CINTSYSDIR/include/filename
         //
         G__getcintsysdir();
         if (G__cintsysdir[0]) {
            sprintf(G__ifile.name,"%s%s%s%sinclude%s%s%s",G__cintsysdir,G__psep,G__CFG_COREVERSION
                    ,G__psep,G__psep,filename,addpost[i2]);
#ifndef G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "r");
#else // G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "rb");
#endif // G__WIN32
            if (G__ifile.fp && G__autoload_stdheader) {
               G__globalcomp = G__store_globalcomp;
               G__gen_linksystem(filename);
            }
            hdrprop = G__CINTHDR;
            G__globalcomp = G__NOLINK;
         }
         if (G__ifile.fp) {
            break;
         }
         //
         // try $CINTSYSDIR/stl
         //
         G__getcintsysdir();
         if (G__cintsysdir[0]) {
            sprintf(G__ifile.name,"%s%s%s%sstl%s%s%s",G__cintsysdir, G__psep, G__CFG_COREVERSION,
                    G__psep, G__psep, filename,addpost[i2]);
#ifndef G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "r");
#else // G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "rb");
#endif // G__WIN32
            if (G__ifile.fp && G__autoload_stdheader) {
               G__globalcomp = G__store_globalcomp;
               G__gen_linksystem(filename);
            }
            hdrprop = G__CINTHDR;
            G__globalcomp = G__NOLINK;
         }
         if (G__ifile.fp) {
            break;
         }
         //
         // try $CINTSYSDIR/lib
         //
         if (G__cintsysdir[0]) {
            sprintf(G__ifile.name,"%s%s%s%slib%s%s%s",G__cintsysdir, G__psep, G__CFG_COREVERSION,
                    G__psep, G__psep,filename,addpost[i2]);
#ifndef G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "r");
#else // G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "rb");
#endif // G__WIN32
            // --
         }
         if (G__ifile.fp) {
            break;
         }
#ifdef G__EDU_VERSION
         //
         // try include/filename
         //
         G__getcintsysdir();
         if (G__cintsysdir[0]) {
            sprintf(G__ifile.name, "include%s%s%s", G__psep, filename, addpost[i2]);
#ifndef G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "r");
#else // G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "rb");
#endif // G__WIN32
            if (G__ifile.fp && G__autoload_stdheader) {
               G__globalcomp = G__store_globalcomp;
               G__gen_linksystem(filename);
            }
            hdrprop = G__CINTHDR;
            G__globalcomp = G__NOLINK;
         }
         if (G__ifile.fp) {
            break;
         }
         //
         // try stl
         //
         G__getcintsysdir();
         if (G__cintsysdir[0]) {
            sprintf(G__ifile.name, "stl%s%s%s", G__psep, filename, addpost[i2]);
#ifndef G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "r");
#else // G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "rb");
#endif // G__WIN32
            if (G__ifile.fp && G__autoload_stdheader) {
               G__globalcomp = G__store_globalcomp;
               G__gen_linksystem(filename);
            }
            hdrprop = G__CINTHDR;
            G__globalcomp = G__NOLINK;
         }
         if (G__ifile.fp) {
            break;
         }
#endif // G__EDU_VERSION
#ifdef G__VISUAL
         //
         // try /msdev/include
         //
         if (G__cintsysdir[0]) {
            sprintf(G__ifile.name, "/msdev/include/%s%s", filename, addpost[i2]);
#ifndef G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "r");
#else // G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "rb");
#endif // G__WIN32
            G__globalcomp = G__store_globalcomp;
         }
         if (G__ifile.fp) {
            break;
         }
#endif // G__VISUAL
#ifdef G__SYMANTEC
         //
         // try /sc/include
         //
         if (G__cintsysdir[0]) {
            sprintf(G__ifile.name, "/sc/include/%s%s", filename, addpost[i2]);
#ifndef G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "r");
#else // G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "rb");
#endif // G__WIN32
            G__globalcomp = G__store_globalcomp;
         }
         if (G__ifile.fp) {
            break;
         }
#endif // G__SYMANTEC
#ifdef G__VMS
         //
         // try $ROOTSYS[include]
         //
         if (G__cintsysdir[0]) {
            sprintf(G__ifile.name, "%s[include]%s", getenv("ROOTSYS"), filename);
            G__ifile.fp = fopen(G__ifile.name, "r");
         }
         if (G__ifile.fp) {
            break;
         }
         //
         // try $ROOTSYS[cint.include]
         //
         if (G__cintsysdir[0]) {
            sprintf(G__ifile.name, "%s[include]%s", G__cintsysdir, filename);
            G__ifile.fp = fopen(G__ifile.name, "r");
            hdrprop = G__CINTHDR;
            G__globalcomp = G__NOLINK;
         }
         if (G__ifile.fp) {
            break;
         }
         //
         // try sys$common:[decc$lib.reference.decc$rtldef..]
         //
         sprintf(G__ifile.name, "sys$common:decc$lib.reference.decc$rtdef]%s", filename);
         printf("Trying to open %s\n", G__ifile.name, "r");
         G__ifile.fp = fopen(G__ifile.name, "r");
         G__globalcomp = G__store_globalcomp;
         if (G__ifile.fp) {
            break;
         }
#endif // G__VMS
#ifndef G__WIN32
         //
         // try /usr/include/filename
         //
         if (G__cintsysdir[0]) {
            sprintf(G__ifile.name, "/usr/include/%s%s", filename, addpost[i2]);
            G__ifile.fp = fopen(G__ifile.name, "r");
            G__globalcomp = G__store_globalcomp;
         }
         if (G__ifile.fp) {
            break;
         }
#endif // G__WIN32
#ifdef __GNUC__
         //
         // try /usr/include/g++/filename
         //
         if (G__cintsysdir[0]) {
            sprintf(G__ifile.name, "/usr/include/g++/%s%s", filename, addpost[i2]);
#ifndef G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "r");
#else // G__WIN32
            G__ifile.fp = fopen(G__ifile.name, "rb");
#endif // G__WIN32
            G__globalcomp = G__store_globalcomp;
         }
         if (G__ifile.fp) {
            break;
         }
#endif // __GNUC__
#ifndef G__WIN32
         //
         // try /usr/include/CC/filename
         //
         if (G__cintsysdir[0]) {
            sprintf(G__ifile.name, "/usr/include/CC/%s%s", filename, addpost[i2]);
            G__ifile.fp = fopen(G__ifile.name, "r");
            G__globalcomp = G__store_globalcomp;
         }
         if (G__ifile.fp) {
            break;
         }
#endif // G__WIN32
#ifndef G__WIN32
         //
         // try /usr/include/codelibs/filename
         //
         if (G__cintsysdir[0]) {
            sprintf(G__ifile.name, "/usr/include/codelibs/%s%s", filename, addpost[i2]);
            G__ifile.fp = fopen(G__ifile.name, "r");
            G__globalcomp = G__store_globalcomp;
         }
         if (G__ifile.fp) {
            break;
         }
#endif // G__WIN32
         // --
      }
   }
   //
   // check if alreay loaded
   //
   for (int otherfile = 0; otherfile < G__nfile; ++otherfile) {
      if (
         G__matchfilename(otherfile, G__ifile.name) &&
         (G__get_envtagnum() == G__Dict::GetDict().GetScope(G__srcfile[otherfile].parent_tagnum))
      ) {
         if ((!G__prerun || G__debugtrace) && (G__dispmsg >= G__DISPNOTE)) {
            G__fprinterr(G__serr, "Note: File \"%s\" already loaded\n", G__ifile.name);
         }
         fclose(G__ifile.fp);
         G__ifile = store_file;
         G__eof = 0;
         G__step = store_step;
         G__setdebugcond();
         G__UnlockCriticalSection();
         return G__LOADFILE_DUPLICATE;
      }
   }
   //
   // filenum and line_number.
   //
   G__ifile.line_number = 1;
   //
   // if there is null_entry which has been unloaded,
   // use that index. null_entry is found above.
   //
   if (null_entry != -1) {
      G__ifile.filenum = null_entry;
   }
   else {
      G__ifile.filenum = G__nfile;
   }
   if (!G__ifile.fp) {
      G__ifile = store_file;
      G__eof = 0;
      G__step = store_step;
      G__globalcomp = G__store_globalcomp;
      if (!G__ispragmainclude) {
         G__fprinterr(G__serr, "Error: cannot open file \"%s\" ", filename);
         G__genericerror(0);
      }
      G__iscpp = store_iscpp;
      G__UnlockCriticalSection();
      return G__LOADFILE_FAILURE;
   }
   if (G__ignoreinclude && (*G__ignoreinclude)(filename, G__ifile.name)) {
      // Close file for process max file open limitation with -cN option
      fclose(G__ifile.fp);
      // since we ignore the file, we can assume that it has no template nor any references.
      G__ifile = store_file;
      G__eof = 0;
      G__step = store_step;
      G__setdebugcond();
      G__globalcomp = G__store_globalcomp;
      G__iscpp = store_iscpp;
      G__UnlockCriticalSection();
      return G__LOADFILE_SUCCESS;
   }
   //
   // if there is null_entry which has been unloaded,
   // use that index. null_entry is found above.
   //
   if (null_entry != -1) {
      fentry = null_entry;
   }
   else {
      fentry = G__nfile;
   }
   // G__ignoreinclude might have loaded some more libs,
   // so update G__ifile to point to fentry:
   G__ifile.filenum = fentry;
   G__srcfile[fentry].dictpos = (G__dictposition*) malloc(sizeof(G__dictposition));
   G__store_dictposition(G__srcfile[fentry].dictpos);
   if (null_entry == -1) {
      ++G__nfile;
   }
   G__srcfile[fentry].hdrprop = hdrprop;
#ifdef G__SECURITY
   store_security = G__security;
   G__srcfile[fentry].security = G__security;
#endif // G__SECURITY
   G__srcfile[fentry].hash = hash;
   G__srcfile[fentry].prepname = 0;
   if (prepname[0]) {
      G__srcfile[fentry].prepname = (char*) malloc(strlen(prepname) + 1);
      strcpy(G__srcfile[fentry].prepname, prepname);
   }
   if (G__globalcomp < G__NOLINK) {
      G__srcfile[fentry].filename = (char*) malloc(strlen(G__ifile.name) + 1);
      strcpy(G__srcfile[fentry].filename, G__ifile.name);
   }
   else {
      G__srcfile[fentry].filename = (char*) malloc(strlen(filename) + 1);
      strcpy(G__srcfile[fentry].filename, filename);
   }
   G__srcfile[fentry].fp = G__ifile.fp;
   G__srcfile[fentry].included_from = store_file.filenum;
   G__srcfile[fentry].ispermanentsl = G__ispermanentsl;
   G__srcfile[fentry].initsl = 0;
   G__srcfile[fentry].hasonlyfunc = 0;
   G__srcfile[fentry].parent_tagnum = G__get_tagnum(G__get_envtagnum());
   G__srcfile[fentry].slindex = -1;
   G__srcfile[fentry].breakpoint = 0;
   if (G__debugtrace) {
      G__fprinterr(G__serr, "LOADING file=%s:%s:%s\n", filename, G__ifile.name, prepname);
   }
   if (G__debug) {
      G__fprinterr(G__serr, "%-5d", G__ifile.line_number);
   }
   store_prerun = G__prerun;
   store_p_local = G__p_local;
   if (
      !G__def_struct_member ||
      !G__tagdefining ||
      (
         (G__struct.type[G__get_tagnum(G__tagdefining)] != 'n') &&
         (G__struct.type[G__get_tagnum(G__tagdefining)] != 'c') &&
         (G__struct.type[G__get_tagnum(G__tagdefining)] != 's')
      )
   ) {
      G__p_local = 0;
   }
   G__eof = 0;
   G__prerun = 1;
   store_nobreak = G__nobreak;
   G__nobreak = 1;
   store_var_type = G__var_type;
   store_tagnum = G__tagnum;
   store_typenum = G__typenum;
   store_func_now = G__func_now;
   G__func_now = ::Reflex::Member();
   store_macroORtemplateINfile = G__macroORtemplateINfile;
   G__macroORtemplateINfile = 0;
   store_asm_noverflow = G__asm_noverflow;
   store_no_exec_compile = G__no_exec_compile;
   store_asm_exec = G__asm_exec;
   G__asm_noverflow = 0;
   G__no_exec_compile = 0;
   G__asm_exec = 0;
#ifdef G__SHAREDLIB
   len = strlen(filename);
   dllpost = G__getmakeinfo1("DLLPOST");
   if ( // Shared library filename.
      (
         (len > 3) &&
         (
            !strcmp(filename + len - 3, ".sl") ||
            !strcmp(filename + len - 3, ".dl") ||
            !strcmp(filename + len - 3, ".so")
         )
      ) ||
      (
         (len > 4) &&
         (
            !strcmp(filename + len - 4, ".dll") ||
            !strcmp(filename + len - 4, ".DLL")
         )
      ) ||
#if defined(R__FBSD) || defined(R__OBSD) // Free BSD or Open BSD
      (
         (len > strlen(soext)) &&
         !strcmp(filename + len - strlen(soext), soext)
      ) ||
#endif // R__FBSD || R__OBSD
      (
         dllpost[0] &&
         (len > (len1 = strlen(dllpost))) &&
         !strcmp(filename + len - len1, dllpost)
      ) ||
      (
         (len > 2) &&
         (
            !strcmp(filename + len - 2, ".a") ||
            !strcmp(filename + len - 2, ".A")
         )
      )
   ) { // Shared library filename.
      //
      //  This is a shared library filename.
      //
      //  Close source file.
      //
      //  Caution, G__ifile.fp is left opened,
      //  this may cause trouble in future.
      //
      fclose(G__srcfile[fentry].fp);
      if (G__ifile.fp == G__srcfile[fentry].fp) {
         // Since the file is closed, the FILE* pointer is now invalid and thus
         // we have to remove it from G__ifile!
         G__ifile.fp = 0;
      }
      G__srcfile[fentry].fp = 0;
      //
      //  Load the shared library.
      //
      {
         // --
#if !defined(ROOTBUILD) && !defined(G__BUILDING_CINTTMP)
         G__StrBuf nm_sb(G__MAXFILENAME);
         char* nm = nm_sb;
         strcpy(nm, G__ifile.name);
         //int allsl = G__shl_load(G__ifile.name);
         int allsl = G__shl_load(nm);
#else // !ROOTBUILD && !G__BUILDING_CINTTMP
         int allsl = -1; // don't load any shared libs
#endif // !ROOTBUILD && !G__BUILDING_CINTTMP
         if (allsl != -1) {
            G__srcfile[fentry].slindex = allsl;
         }
      }
      if (G__initpermanentsl) {
         if (G__ispermanentsl) {
            if (!G__srcfile[fentry].initsl) {
               G__srcfile[fentry].initsl = new std::list<G__DLLINIT>;
            }
            G__srcfile[fentry].initsl->insert(G__srcfile[fentry].initsl->end(), G__initpermanentsl->begin(), G__initpermanentsl->end());
         }
         G__initpermanentsl->clear();
      }
   }
   else {
      if ((G__globalcomp > 1) && !strcmp(filename + strlen(filename) - 4, ".sut")) {
         G__ASSERT(G__sutpi && G__ifile.fp);
         G__copyfile(G__sutpi, G__ifile.fp);
      }
      else {
         char* store_struct_offset = G__store_struct_offset;
         G__store_struct_offset = 0;
         if (G__isbinaryfile(filename)) { // Error, binary file.
            G__iscpp = store_iscpp;
#ifdef G__SECURITY
            G__security = store_security;
#endif // G__SECURITY
            G__UnlockCriticalSection();
            return(G__LOADFILE_FAILURE);
         }
         if (G__copyflag) {
            G__copysourcetotmp(prepname, &G__ifile, fentry);
         }
         //
         //  Load and parse file in prerun environment.
         //
         while (!G__eof && (G__return < G__RETURN_EXIT1)) {
            int brace_level = 0;
            G__exec_statement(&brace_level);
         }
         G__store_struct_offset = store_struct_offset;
      }
   }
#else // G__SHAREDLIB
   if ((G__globalcomp > 1) && !strcmp(filename + strlen(filename) - 4, ".sut")) {
      G__ASSERT(G__sutpi && G__ifile.fp);
      G__copyfile(G__sutpi, G__ifile.fp);
   }
   else {
      if (G__isbinaryfile(filename)) { // Error, binary file.
         G__iscpp = store_iscpp;
#ifdef G__SECURITY
         G__security = store_security;
#endif // G__SECURITY
         G__UnlockCriticalSection();
         return G__LOADFILE_FAILURE;
      }
      //
      //  Load and parse file in prerun environment.
      //
      while (!G__eof && (G__return < G__RETURN_EXIT1)) {
         int brace_level = 0;
         G__exec_statement(&brace_level);
      }
   }
#endif // G__SHAREDLIB
   //
   // Avoid file array overflow when G__globalcomp
   //
   if ((G__globalcomp != G__NOLINK) && G__srcfile[fentry].fp) {
      if (!G__macroORtemplateINfile
#ifndef G__OLDIMPLEMENTATION1923
            && (!G__fons_comment || !G__cpp)
#endif // G__OLDIMPLEMENTATION1923
      ) {
         // After closing the file let's make sure that all reference to
         // the file pointer are reset. When preprocessor is used, we
         // will have several logical file packed in one file.
         tmpfp = G__srcfile[fentry].fp;
         for (i1 = 0; i1 < G__nfile; ++i1) {
            if (G__srcfile[i1].fp == tmpfp) {
               G__srcfile[i1].fp = 0;
            }
         }
         // Close file for process max file open limitation with -cN option
         fclose(tmpfp);
      }
   }
   pragmacompile_filenum = G__ifile.filenum;
   pragmacompile_iscpp = G__iscpp;
   G__func_now = store_func_now;
   G__macroORtemplateINfile = store_macroORtemplateINfile;
   G__var_type = store_var_type;
   G__tagnum = store_tagnum;
   G__typenum = store_typenum;
   G__nobreak = store_nobreak;
   G__prerun = store_prerun;
   G__p_local = store_p_local;
   G__asm_noverflow = store_asm_noverflow;
   G__no_exec_compile = store_no_exec_compile;
   G__asm_exec = store_asm_exec;
   G__ifile = store_file;
   G__eof = 0;
   G__step = store_step;
   G__setdebugcond();
   G__globalcomp = G__store_globalcomp;
   G__iscpp = store_iscpp;
#ifdef G__SECURITY
   G__security = store_security;
#endif // G__SECURITY
   if (G__return > G__RETURN_NORMAL) {
      G__UnlockCriticalSection();
      return G__LOADFILE_FAILURE;
   }
#ifdef G__AUTOCOMPILE
   //
   // if '#pragma compile' appears in source code.
   //
   if (G__fpautocc && (G__autoccfilenum == pragmacompile_filenum)) {
      store_iscpp = G__iscpp;
      G__iscpp = pragmacompile_iscpp;
      G__autocc();
      G__iscpp = store_iscpp;
   }
#endif // G__AUTOCOMPILE
   G__checkIfOnlyFunction(fentry);
   G__UnlockCriticalSection();
   return G__LOADFILE_SUCCESS;
}

//______________________________________________________________________________
int G__setfilecontext(const char* filename, G__input_file* ifile)
{
   // Set the current G__ifile to filename, allocate an entry in G__srcfiles
   // if necessary. Set ifile to the previous G__ifile to it can be popped.
   // This function does not load any file, it just provides a valid G__ifile
   // context for type manipulations.
   //
   // Returns 0 in case of error.
   // Returns 1 if file entry is newly allocated, 2 otherwise.
   if (!filename) {
      return 0;
   }
   int null_entry = -1;
   int found_entry = -1;
   // find G__srcfile index matching filename
   for (int i = 0; (found_entry == -1) && (i < G__nfile); ++i) {
      if (G__srcfile[i].filename) {
         if (!strcmp(G__srcfile[i].filename, filename)) {
            found_entry = i;
         }
      }
      else if (null_entry == -1) {
         null_entry = i;
      }
   }
   if (found_entry == -1) {
      int fentry = null_entry;
      if (fentry == -1) {
         fentry = G__nfile;
      }
      G__srcfile[fentry].dictpos = (G__dictposition*) malloc(sizeof(G__dictposition));
      G__store_dictposition(G__srcfile[fentry].dictpos);
      if (null_entry == -1) {
         ++G__nfile;
      }
#ifdef G__SECURITY
      G__srcfile[fentry].security = G__security;
#endif // G__SECURITY
      int hash = 0;
      int temp = 0;
      G__hash(filename, hash, temp)
      G__srcfile[fentry].hash = hash;
      G__srcfile[fentry].prepname = 0;
      G__srcfile[fentry].filename = (char*) malloc(strlen(filename) + 1);
      strcpy(G__srcfile[fentry].filename, filename);
      G__srcfile[fentry].fp = 0;
      G__srcfile[fentry].included_from = G__ifile.filenum;
      G__srcfile[fentry].ispermanentsl = 1;
      G__srcfile[fentry].initsl = 0;
      G__srcfile[fentry].hasonlyfunc =  0;
      G__srcfile[fentry].parent_tagnum = G__get_tagnum(G__get_envtagnum());
      G__srcfile[fentry].slindex = -1;
      G__srcfile[fentry].breakpoint = 0;
      found_entry = fentry;
   }
   if (ifile) {
      *ifile = G__ifile;
   }
   G__ifile.fp = G__srcfile[found_entry].fp;
   G__ifile.filenum = found_entry;
   strcpy(G__ifile.name, G__srcfile[found_entry].filename);
   G__ifile.line_number = 0;
   G__ifile.str = 0;
   G__ifile.pos = 0;
   G__ifile.vindex = 0;
   if (found_entry != -1) {
      return 2;
   }
   return 1;
}

//______________________________________________________________________________
int Cint::Internal::G__preprocessor(char* outname, char* inname, int cppflag, char* macros, char* undeflist, char* ppopt, char* includepath)
{
   // Use C/C++ preprocessor prior to interpretation.
   // Name of preprocessor must be defined in $CINTSYSDIR/MAKEINFO file as
   // CPPPREP and CPREP.
   G__StrBuf tmpfilen_sb(G__MAXFILENAME);
   char* tmpfilen = tmpfilen_sb;
   int tmplen;
   FILE* fp;
   int flag = 0;
   int inlen;
   char* post;
   inlen = strlen(inname);
   post = strrchr(inname, '.');
   if (post && (inlen > 2)) {
      if (!strcmp(inname + strlen(inname) - 2, ".c")) {
         if (!G__cpplock) {
            G__iscpp = 0;
         }
         flag = 1;
      }
      else if (!strcmp(inname + strlen(inname) - 2, ".C")) {
         if (!G__cpplock) {
            G__iscpp = 1;
         }
         flag = 1;
      }
      else if (!strcmp(inname + strlen(inname) - 2, ".h")) {
         flag = 1;
      }
      else if (!strcmp(inname + strlen(inname) - 2, ".H")) {
         flag = 1;
      }
   }
   if (!flag && post && (inlen > 3)) {
      if (
         !strcmp(inname + strlen(inname) - 3, ".cc") ||
         !strcmp(inname + strlen(inname) - 3, ".CC") ||
         !strcmp(inname + strlen(inname) - 3, ".hh") ||
         !strcmp(inname + strlen(inname) - 3, ".HH") ||
         !strcmp(inname + strlen(inname) - 3, ".wc") ||
         !strcmp(inname + strlen(inname) - 3, ".WC")
      ) {
         G__iscpp = 1;
         flag = 1;
      }
   }
   if (!flag && post && (inlen > 4)) {
      if (
          !strcmp(inname + strlen(inname) - 4, ".cxx") ||
          !strcmp(inname + strlen(inname) - 4, ".CXX") ||
          !strcmp(inname + strlen(inname) - 4, ".cpp") ||
          !strcmp(inname + strlen(inname) - 4, ".CPP") ||
          !strcmp(inname + strlen(inname) - 4, ".hxx") ||
          !strcmp(inname + strlen(inname) - 4, ".HXX") ||
          !strcmp(inname + strlen(inname) - 4, ".hpp") ||
          !strcmp(inname + strlen(inname) - 4, ".HPP")
      ) {
         G__iscpp = 1;
         flag = 1;
      }
   }
   if (!flag && post) {
      if (!G__cppsrcpost[0]) {
         strcpy(G__cppsrcpost, G__getmakeinfo1("CPPSRCPOST"));
      }
      if (!G__csrcpost[0]) {
         strcpy(G__csrcpost, G__getmakeinfo1("CSRCPOST"));
      }
      if (!G__cpphdrpost[0]) {
         strcpy(G__cpphdrpost, G__getmakeinfo1("CPPHDRPOST"));
      }
      if (!G__chdrpost[0]) {
         strcpy(G__chdrpost, G__getmakeinfo1("CHDRPOST"));
      }
      if (!strcmp(inname + strlen(inname) - strlen(G__cppsrcpost), G__cppsrcpost)) {
         if (!G__clock) {
            G__iscpp = 1;
         }
         flag = 1;
      }
      else if (!strcmp(inname + strlen(inname) - strlen(G__csrcpost), G__csrcpost)) {
         if (!G__cpplock) {
            G__iscpp = 0;
         }
         flag = 1;
      }
      else if (!strcmp(inname + strlen(inname) - strlen(G__cpphdrpost), G__cpphdrpost)) {
         flag = 1;
      }
      else if (!strcmp(inname + strlen(inname) - strlen(G__chdrpost), G__chdrpost)) {
         if (!G__cpplock) {
            G__iscpp = 0;
         }
         flag = 1;
      }
   }
   else if (!flag && !post) {
      if (!G__clock) {
         G__iscpp = 1;
      }
      flag = 1;
   }
   // If using C preprocessor '-p' option'
   if (!cppflag || !flag) {
      // Didn't use C preprocessor because cppflag is
      // not set or file name suffix does not match.
      outname[0] = '\0';
   }
   else {
      // Determine what C/C++ preprocessor to use
      if (!G__ccom[0]) {
         switch (G__globalcomp) {
            case G__CPPLINK: // C++ link
               strcpy(G__ccom, G__getmakeinfo("CPPPREP"));
               break;
            case G__CLINK: // C link
               strcpy(G__ccom, G__getmakeinfo("CPREP"));
               break;
            default:
               if (G__iscpp) {
                  strcpy(G__ccom, G__getmakeinfo("CPPPREP"));
               }
               else {
                  strcpy(G__ccom, G__getmakeinfo("CPREP"));
               }
               break;
         }
         if (!G__ccom[0]) {
            // --
#ifdef __GNUC__
            sprintf(G__ccom, "g++ -E");
#else // __GNUC__
            sprintf(G__ccom, "CC -E");
#endif // __GNUC__
            // --
         }
      }
      // Get tmpfile name if necessary
      if (
         (
            (strlen(inname) > 2) &&
            (
               !strcmp(inname + strlen(inname) - 2, ".H") ||
               !strcmp(inname + strlen(inname) - 2, ".h")
            )
         ) ||
         (
            (strlen(inname) > 3) &&
            (
               !strcmp(inname + strlen(inname) - 3, ".hh") ||
               !strcmp(inname + strlen(inname) - 3, ".HH")
            )
         ) ||
         (
            (strlen(inname) > 4) &&
            (
               !strcmp(inname + strlen(inname) - 4, ".hpp") ||
               !strcmp(inname + strlen(inname) - 4, ".HPP") ||
               !strcmp(inname + strlen(inname) - 4, ".hxx") ||
               !strcmp(inname + strlen(inname) - 4, ".HXX")
            )
         ) ||
         !strchr(inname, '.')
      ) {
         // if header file, create tmpfile name as xxx.C
         do {
            G__tmpnam(tmpfilen); // can't replace this with tmpfile()
            tmplen = strlen(tmpfilen);
            if ((G__globalcomp == G__CPPLINK) || G__iscpp) {
               if (!G__cppsrcpost[0]) {
                  strcpy(G__cppsrcpost, G__getmakeinfo1("CPPSRCPOST"));
               }
               strcpy(tmpfilen + tmplen, G__cppsrcpost);
            }
            else {
               if (!G__csrcpost[0]) {
                  strcpy(G__csrcpost, G__getmakeinfo1("CSRCPOST"));
               }
               strcpy(tmpfilen + tmplen, G__csrcpost);
            }
            fp = fopen(tmpfilen, "w");
         }
         while (!fp && G__setTMPDIR(tmpfilen));
         if (fp) {
            fprintf(fp, "#include \"%s\"\n\n\n", inname);
            fclose(fp);
         }
      }
      else {
         // otherwise, simply copy the inname
         strcpy(tmpfilen, inname);
         tmplen = 0;
      }
      // Get output file name
      G__getcintsysdir();
      G__tmpnam(outname); // can't replace this with tmpfile()
#if defined(G__SYMANTEC) && (!defined(G__TMPFILE))
      // NEVER DONE
      {
         int len_outname = strlen(outname);
         outname[len_outname] = '.';
         outname[len_outname+1] = '\0';
      }
#endif // G__SYMANTEC && !G__TMPFILE

      G__StrBuf temp_sb(G__LARGEBUF);
#ifdef G__SYMANTEC
      //
      // preprocessor statement for Symantec C++
      //
      if (G__cintsysdir[0]) {
         temp_sb.Format("%s %s %s -I. %s %s -D__CINT__ -I%s/include -I%s/stl -I%s/lib %s -o%s", G__ccom, macros, undeflist, ppopt, includepath, G__cintsysdir, G__cintsysdir, G__cintsysdir, tmpfilen, outname);
      }
      else {
         temp_sb.Format("%s %s %s %s -I. %s -D__CINT__ %s -o%s", G__ccom, macros, undeflist, ppopt, includepath, tmpfilen, outname);
      }
#elif defined(G__BORLAND)
      strcat(outname, ".i");
      if (G__cintsysdir[0]) {
         std::string corecintsysdir(G__cintsysdir);
         corecintsysdir += "/";
         corecintsysdir += G__CFG_COREVERSION;
         corecintsysdir += "/";
         temp_sb.Format("%s %s %s -I. %s %s -D__CINT__ -I%s/include -I%s/stl -I%s/lib -o%s %s", G__ccom, macros, undeflist, ppopt, includepath, corecintsysdir.c_str(), corecintsysdir.c_str(), corecintsysdir.c_str(), outname, tmpfilen);
      }
      else {
         temp_sb.Format("%s %s %s %s -I. %s -D__CINT__ -o%s %s" , G__ccom, macros, undeflist, ppopt, includepath, outname, tmpfilen);
      }
#else
      //
      // preprocessor statement for UNIX
      //
      if (G__cintsysdir[0]) {
         std::string corecintsysdir(G__cintsysdir);
         corecintsysdir += "/";
         corecintsysdir += G__CFG_COREVERSION;
         corecintsysdir += "/";
         temp_sb.Format("%s %s %s -I. %s %s -D__CINT__ -I%s/include -I%s/stl -I%s/lib %s > %s", G__ccom, macros, undeflist, ppopt , includepath, corecintsysdir.c_str(), corecintsysdir.c_str(), corecintsysdir.c_str(), tmpfilen, outname);
      }
      else {
         temp_sb.Format("%s %s %s %s -I. %s -D__CINT__ %s > %s", G__ccom, macros, undeflist, ppopt, includepath, tmpfilen, outname);
      }
#endif
      if (G__debugtrace || G__steptrace || G__step || G__asm_dbg) {
         G__fprinterr(G__serr, " %s\n", temp_sb.data());
      }
      int pres = system(temp_sb);
      if (tmplen) {
         remove(tmpfilen);
      }
      return pres;
   }
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__difffile(char* file1, char* file2)
{
   int unmatch = 0;
   FILE* fp1 = fopen(file1, "r");
   FILE* fp2 = fopen(file2, "r");
   if (!fp1 || !fp2) {
      unmatch = 1;
   }
   else {
      int c1 = 0;
      int c2 = 0;
      do {
         c1 = fgetc(fp1);
         c2 = fgetc(fp2);
         if (c1 != c2) {
            ++unmatch;
            break;
         }
      }
      while ((c1 != EOF) && (c2 != EOF));
      if (c1 != c2) {
         ++unmatch;
      }
   }
   if (fp1) {
      fclose(fp1);
   }
   if (fp2) {
      fclose(fp2);
   }
   return unmatch;
}

//______________________________________________________________________________
int Cint::Internal::G__copyfile(FILE* to, FILE* from)
{
   int c = fgetc(from);
   for (; c != EOF; c = fgetc(from)) {
      fputc(c, to);
   }
   return 0;
}

//______________________________________________________________________________
#ifdef G__TMPFILE
static char G__tmpdir[G__MAXFILENAME];
static char G__mfpname[G__MAXFILENAME];
#else // G__TMPFILE
static char G__mfpname[G__MAXFILENAME];
#endif // G__TMPFILE

//______________________________________________________________________________
extern "C" int G__setTMPDIR(char* badname)
{
   // --
#ifndef G__TMPFILE
   G__fprinterr(G__serr, "CAUTION: tempfile %s can't open\n", badname);
   return 0;
#else // G__TMPFILE
   G__fprinterr(G__serr, "CINT CAUTION: tempfile %s can't open\n", badname);
   G__fprinterr(G__serr, "Input another temp directory or '*' to give up\n");
   G__fprinterr(G__serr, "(Setting CINTTMPDIR environment variable avoids this interrupt)\n");
   strcpy(G__tmpdir, G__input("Input TMPDIR > "));
   char* p = strchr(G__tmpdir, '\r');
   if (p) {
      *p = '\0';
   }
   p = strchr(G__tmpdir, '\n');
   if (p) {
      *p = '\0';
   }
   if (G__tmpdir[0] == '*') {
      G__tmpdir[0] = '\0';
      return 0;
   }
   return 1;
#endif // G__TMPFILE
   // --
}

//______________________________________________________________________________
namespace {

class G__Tmpnam_Files
{
public:
   G__Tmpnam_Files()
   {
   }
   ~G__Tmpnam_Files()
   {
      std::list<std::string>::iterator iter = fFiles.begin();
      for (; iter != fFiles.end(); ++iter) {
         unlink(iter->c_str());
      }
   }
   void Add(const char* name)
   {
      fFiles.push_back(name);
   }
public:
   std::list<std::string> fFiles;
};

} // unnamed namespace

//______________________________________________________________________________
extern "C" char* G__tmpnam(char* name)
{
   static G__Tmpnam_Files G__tmpfiles;
#ifdef G__TMPFILE
   const char* appendix = "_cint";
   static char tempname[G__MAXFILENAME];
   int pid = getpid();
   int now = clock();
   char* tmp = 0;
   if (!G__tmpdir[0]) {
      tmp = getenv("CINTTMPDIR");
      if (!tmp) {
         tmp = getenv("TEMP");
      }
      if (!tmp) {
         tmp = getenv("TMP");
      }
      if (tmp) {
         strcpy(G__tmpdir, tmp);
      }
      else {
         strcpy(G__tmpdir, ".");
      }
   }
   if (name) {
      tmp = tempnam(G__tmpdir, "");
      strcpy(name, tmp);
      free(tmp);
      if (strlen(name) < (G__MAXFILENAME - 10)) {
         sprintf(name + strlen(name), "%d%d", pid % 10000, now % 10000);
      }
      if (strlen(name) < (G__MAXFILENAME - 6)) {
         strcat(name, appendix);
      }
      G__tmpfiles.Add(name);
      return name;
   }
   tmp = tempnam(G__tmpdir, "");
   strcpy(tempname, tmp);
   free(tmp);
   size_t lentemp = strlen(tempname);
   if (lentemp < (G__MAXFILENAME - 10)) {
      sprintf(tempname + lentemp, "%d%d", pid % 10000, now % 10000);
   }
   if (strlen(tempname) < (G__MAXFILENAME - strlen(appendix) - 1)) {
      strcat(tempname, appendix);
   }
   G__tmpfiles.Add(tempname);
   return tempname;
#elif defined(__CINT__)
   static char tempname[G__MAXFILENAME];
   const char* appendix = "_cint";
   if (!name) {
      name = tempname;
   }
   tmpnam(name);
   if (strlen(name) < (G__MAXFILENAME - 6)) {
      strcat(name, appendix);
   }
   G__tmpfiles.Add(name);
   return name;
#elif ((__GNUC__ >= 3) || ((__GNUC__ >= 2) && (__GNUC_MINOR__ >= 96))) && (defined(__linux) || defined(__linux__))
   // After all, mkstemp creates more problem than a solution.
   static char tempname[G__MAXFILENAME];
   const char* appendix = "_cint";
   static char tmpdir[G__MAXFILENAME];
   char* tmp;
   if (!tmpdir[0]) {
      tmp = getenv("CINTTMPDIR");
      if (!tmp) {
         tmp = getenv("TEMP");
      }
      if (!tmp) {
         tmp = getenv("TMP");
      }
      if (tmp) {
         strcpy(tmpdir, tmp);
      }
      else {
         strcpy(tmpdir, "/tmp");
      }
   }
   if (!name) {
      name = tempname;
   }
   strcpy(name, tmpdir);
   strcat(name, "/XXXXXX");
   close(mkstemp(name)); // mkstemp not only generate file name but also opens the file.
   remove(name); // mkstemp creates this file anyway.  Delete it.  Questionable.
   if (strlen(name) < (G__MAXFILENAME - 6)) {
      strcat(name, appendix);
   }
   G__tmpfiles.Add(name);
   return name;
#else
   static char tempname[G__MAXFILENAME];
   const char* appendix = "_cint";
   if (!name) {
      name = tempname;
   }
   tmpnam(name);
   if (strlen(name) < (G__MAXFILENAME - 6)) {
      strcat(name, appendix);
   }
   G__tmpfiles.Add(name);
   return name;
#endif
   // --
}

static int G__istmpnam = 0;

//______________________________________________________________________________
void Cint::Internal::G__openmfp()
{
   // --
#ifndef G__TMPFILE
   G__mfp = tmpfile();
   if (!G__mfp) {
      do {
         G__tmpnam(G__mfpname); // Only VC++ uses this.
         G__mfp = fopen(G__mfpname, "wb+");
      }
      while (!G__mfp && G__setTMPDIR(G__mfpname));
      G__istmpnam = 1;
   }
#else // G__TMPFILE
   do {
      G__tmpnam(G__mfpname); // Only VC++ uses this.
      G__mfp = fopen(G__mfpname, "wb+");
   }
   while (!G__mfp && G__setTMPDIR(G__mfpname));
#endif // G__TMPFILE
   // --
}

//______________________________________________________________________________
int Cint::Internal::G__closemfp()
{
   // --
#ifndef G__TMPFILE
   int result = 0;
   if (!G__istmpnam) {
      if (G__mfp) {
         result = fclose(G__mfp);
      }
      G__mfp = 0;
   }
   else {
      if (G__mfp) {
         fclose(G__mfp);
      }
      G__mfp = 0;
      if (G__mfpname[0]) {
         result = remove(G__mfpname);
      }
      G__mfpname[0] = 0;
      G__istmpnam = 0;
   }
   return result;
#else // G__TMPFILE
   int result = 0;
   if (G__mfp) {
      fclose(G__mfp);
   }
   G__mfp = 0;
   if (G__mfpname[0]) {
      result = remove(G__mfpname);
   }
   G__mfpname[0] = 0;
   return result;
#endif // G__TMPFILE
   // --
}

//______________________________________________________________________________
extern "C" G__input_file* G__get_ifile()
{
   return &G__ifile;
}

//______________________________________________________________________________
int Cint::Internal::G__register_sharedlib(const char *libname)
{
   // Register (if not already registered) in G__srcfile a library that
   // is indirectly loaded (via a hard link) and has a CINT dictionary
   // and return the filenum (index in G__srcfile).
   
   int null_entry = -1;
   int i1 = 0;
   int hash,temp;
   
   G__LockCriticalSection();
   
   /*************************************************
    * store current input file information
    *************************************************/
   G__setdebugcond();
   
   /******************************************************************
    * check if number of loaded file exceeds G__MAXFILE
    * if so, restore G__ifile reset G__eof and return.
    ******************************************************************/
   if(G__nfile==G__MAXFILE) {
      G__fprinterr(G__serr,"Limitation: Sorry, can not load any more files\n");
      G__setdebugcond();
      G__UnlockCriticalSection();
      return(G__LOADFILE_FATAL);
   }
   
   G__hash(libname,hash,temp);
   
   
   /******************************************************************
    * check if file is already loaded.
    * if so, restore G__ifile reset G__eof and return.
    ******************************************************************/
   while(i1<G__nfile) {
      /***************************************************
       * This entry was unloaded by G__unloadfile()
       * Then remember the entry index into 'null_entry'.
       ***************************************************/
      if((char*)NULL==G__srcfile[i1].filename) {
         if(null_entry == -1) {
            null_entry = i1;
         }
      }
      /***************************************************
       * check if alreay loaded
       ***************************************************/
      if(G__matchfilename(i1,libname)
         &&G__get_tagnum(G__get_envtagnum())==G__srcfile[i1].parent_tagnum
         ){
         /******************************************************
          * restore input file information to G__ifile
          * and reset G__eof to 0.
          ******************************************************/
         G__UnlockCriticalSection();
         return i1;
      }
      else {
         ++i1;
      }
   }
   
   int fentry;
   if (null_entry != -1) {
      fentry = null_entry;
   } else {
      fentry = G__nfile;
      ++G__nfile;
   }
   
   G__srcfile[fentry].dictpos
   = (struct G__dictposition*)malloc(sizeof(struct G__dictposition));
   G__store_dictposition(G__srcfile[fentry].dictpos);
   
   G__srcfile[fentry].hdrprop = G__NONCINTHDR;
   
   G__srcfile[fentry].security = G__security;
   
   G__srcfile[fentry].prepname = (char*)NULL;
   G__srcfile[fentry].hash = hash;
   G__srcfile[fentry].filename = (char*)malloc(strlen(libname)+1);
   strcpy(G__srcfile[fentry].filename,libname);
   G__srcfile[fentry].fp=0;
   
   G__srcfile[fentry].included_from = G__ifile.filenum;
   
   G__srcfile[fentry].ispermanentsl = 2;
   G__srcfile[fentry].initsl = 0;
   G__srcfile[fentry].hasonlyfunc = (struct G__dictposition*)NULL;
   G__srcfile[fentry].parent_tagnum = G__get_tagnum(G__get_envtagnum());
   G__srcfile[fentry].slindex = -1;
   
   G__UnlockCriticalSection();
   return(fentry);
}

//______________________________________________________________________________
int Cint::Internal::G__unregister_sharedlib(const char *libname)
{
   // Unregister (if and only if it has been registered by G__register_sharedlib) 
   // in G__srcfile a library.
   
   G__LockCriticalSection();
   
   int envtagnum = -1;
   
   /******************************************************************
    * check if file is already loaded.
    * if not so, return
    ******************************************************************/
   int i2;
   int hash;
   G__hash(libname,hash,i2);
   
   bool flag = false;
   int ifn;
   for(ifn = G__nfile-1; ifn>0; --ifn) {
      if(G__srcfile[ifn].ispermanentsl == 2 
         && G__matchfilename(ifn,libname)
         && (-1==envtagnum||(envtagnum==G__srcfile[ifn].parent_tagnum))){
         flag = true;
         break;
      }
   }
   
   if (flag) {
      // File found

      // No check for busy-ness is need.  If we get there the library
      // is already being unloaded.

      // No active unload to do, since we did not load this file directly.
      
      if (G__srcfile[ifn].dictpos) {
         free((void*) G__srcfile[ifn].dictpos);
         G__srcfile[ifn].dictpos = 0;
      }
      if (G__srcfile[ifn].hasonlyfunc) {
         free((void*) G__srcfile[ifn].hasonlyfunc);
         G__srcfile[ifn].hasonlyfunc = 0;
      }
      if (G__srcfile[ifn].filename) {
         // --
#ifndef G__OLDIMPLEMENTATION1546
         unsigned int len = strlen(G__srcfile[ifn].filename);
         if (
             (len > strlen(G__NAMEDMACROEXT2)) &&
             !strcmp(G__srcfile[ifn].filename+len - strlen(G__NAMEDMACROEXT2), G__NAMEDMACROEXT2)
             ) {
            remove(G__srcfile[ifn].filename);
         }
#endif // G__OLDIMPLEMENTATION1546
         free((void*) G__srcfile[ifn].filename);
         G__srcfile[ifn].filename = 0;
      }
      G__srcfile[ifn].hash = 0;      
      
      if(G__debug) {
         G__fprinterr(G__serr,"File=%s unregistered\n",libname);
      }
      while(G__nfile && G__srcfile[G__nfile-1].filename==0) {
         --G__nfile;
      }
   }
   
   G__UnlockCriticalSection(); 
   return G__UNLOADFILE_SUCCESS;
   
}


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
