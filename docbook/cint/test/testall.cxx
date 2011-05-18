#! ../bin/cint
/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * C++ Script testcint.cxx
 ************************************************************************
 * Description:
 *  Automatic test suite of cint
 ************************************************************************
 * Copyright(c) 2002~2004  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// Usage:
//  $ cint testall.cxx

#include <stdio.h>
#ifdef G__ROOT
#include "../../include/configcint.h"
#else
#include "../inc/configcint.h"
#endif

#ifndef G__VISUAL // ??? fprintf crashes if stdfunc.dll is loaded ???
#include <stdlib.h>
#include <string.h>
#endif

#ifdef DEBUG2
#ifndef DEBUG
#define DEBUG
#endif
#endif

char* debug = 0;
bool bKeepOnGoing = false;
bool bIgnoreDiffErrors = false;
bool bHideKnownErrors = false;
#ifndef DEBUG
bool debugMode = false;
#else
bool debugMode = true;
#endif

#include <string>

const char *shellSeparator = "&&";

#ifdef G__ROOT
// ROOT disable the autoloading of the standard header from the compiled
// dictionary (see G__autoload_stdheader) however the cint test requires it
std::string mkcintoption = " -Z1 ";
std::string cintoption = " -Z1";
#else
std::string mkcintoption = "";
std::string cintoption = "";
#endif // G__ROOT
std::string compileroption = "";
std::string prefixcmd = "";

enum ELanguage {
   kLangUnknown,
   kLangC,
   kLangCXX
};

//______________________________________________________________________________
int clear(const char* fname)
{
   // -- Erase a file.
   FILE* fp = fopen(fname, "w");
   fclose(fp);
   return 0;
}

//______________________________________________________________________________
int exist(const char* fname)
{
   // -- Check if a file exists.
   FILE* fp = fopen(fname, "r");
   if (fp) {
      fclose(fp);
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
int rm(const char* fname)
{
   // -- Delete a file.
   int stat;
   do {
      stat = remove(fname);
   }
   while (exist(fname));
   return stat;
}

//______________________________________________________________________________
char *cleanAndCreateDirectory(const char *tname) 
{
   // Create an empty directory with the same name, without extension
   // as the test.

   char com[4096];

   char *dir = new char[strlen(tname)+1];
   strcpy(dir,tname);
   char *cursor = strstr(dir,".");
   if (cursor) {
      *cursor = '\0';
   };
   
   char com[4096];

   sprintf(com,"rm -rf %s",dir);
   run(com);
   sprintf(com,"mkdir %s",dir);
   run(com);
   
   return dir;
}

//______________________________________________________________________________
void cleanDirectory(const char* dir)
{
   // Delete a test directory
   char com[4096];
   sprintf(com,"rm -rf %s",dir);
   run(com);
}

//______________________________________________________________________________
int cat(FILE* fout, const char* fname)
{
   // -- Display a file on a given output file.
   FILE* fp = fopen(fname, "r");
   char b1[4096];
   while (fgets(b1, 4096, fp)) {
      fprintf(fout, "%s", b1);
   }
   fclose(fp);
   return 0;
}

//______________________________________________________________________________
int run(const char* com, bool show_when_debug = true)
{
   // -- Run a system command.
   if (debugMode && show_when_debug) printf("%s\n", com);

   //printf("%s\n", com);
   fflush(stdout);
   int ret = system(com);
   if (ret) {
      printf("FAILED with code %d: %s\n", ret, com);
      if (!bKeepOnGoing) {
         exit(ret);
      }
   }
   return ret;
}

//______________________________________________________________________________
int readahead(FILE* fp, const char* b, const int ahead = 10)
{
   // -- FIXME: Describe this function!
   if (!fp) {
      return 0;
   }
   int result = 0;
   fpos_t p;
   fgetpos(fp, &p);
   char buf[4096];
   int a = 0;
   for (int i = 0; i < ahead; ++i) {
      char* c = fgets(buf, 4096, fp);
      ++a;
      if (!c) {
         break;
      }
      if (!strcmp(b, buf)) {
         result = a;
         break;
      }
   }
   fsetpos(fp, &p);
   return result;
}

//______________________________________________________________________________
void outdiff(FILE* fp, FILE* fpi, int a, const char* b, int& l, const char* m)
{
   // -- FIXME: Describe this function!
   for (int i = 0; i < a; i++) {
      fprintf(fp, "%3d%s %s", l, m, b);
      if (!fpi) {
         break;
      }
      char* c = fgets((char*) b, 400, fpi);
      ++l;
      if (!c) {
         break;
      }
   }
}

//______________________________________________________________________________
void checkdiff(FILE* fp, FILE* fp1, FILE* fp2, char* b1, const char* b2, int& l1, int& l2, const char* m1, const char* m2)
{
   // -- FIXME: Describe this function!
   int a1 = readahead(fp1, b2);
   int a2 = readahead(fp2, b1);
   if (a1 && a2) {
      if (a1 <= a2) {
         outdiff(fp, fp1, a1, b1, l1, m1);
      }
      else {
         outdiff(fp, fp2, a2, b2, l2, m2);
      }
   }
   else if (a1) {
      outdiff(fp, fp1, a1, b1, l1, m1);
   }
   else if (a2) {
      outdiff(fp, fp2, a2, b2, l2, m2);
   }
   else {
      fprintf(fp, "%3d%s %s", l1, m1, b1);
      fprintf(fp, "%3d%s %s", l2, m2, b2);
   }
}

//______________________________________________________________________________
int diff(const char* title, const char* f1, const char* f2, const char* out, const char* macro = "", const char* m1 = ">", const char* m2 = "<")
{
   // -- FIXME: Describe this function!
   FILE* fp = fopen(out, "a");
   FILE* fp1 = fopen(f1, "r");
   FILE* fp2 = fopen(f2, "r");
   char b1[4096];
   char b2[4096];
   char* c1 = 0;
   char* c2 = 0;
   int l1 = 0;
   int l2 = 0;
   fprintf(fp, "%s %s\n", title, macro);
   for (;;) {
      if (fp1) {
         c1 = fgets(b1, 4096, fp1);
         ++l1;
      }
      else {
         c1 = 0;
      }
      if (fp2) {
         c2 = fgets(b2, 4096, fp2);
         ++l2;
      }
      else {
         c2 = 0;
      }
      if (c1 && c2) {
         if (strcmp(b1, b2)) {
            // --
#ifndef G__VISUAL
            checkdiff(fp, fp1, fp2, b1, b2, l1, l2, m1, m2);
#else // G__VISUAL
            fprintf(fp, "%3d%s %s", l1, m1, b1);
            fprintf(fp, "%3d%s %s", l2, m2, b2);
#endif // G__VISUAL
            // --
         }
      }
      else if (c1) {
         fprintf(fp, "%3d%s %s", l1, m1, b1);
      }
      else if (c2) {
         fprintf(fp, "%3d%s %s", l2, m2, b2);
      }
      else {
         break;
      }
   }
   if (fp2) {
      fclose(fp2);
   }
   if (fp1) {
      fclose(fp1);
   }
   if (fp) {
      fclose(fp);
   }
   return 0;
}

//______________________________________________________________________________
int ediff(const char* title, const char* macro, const char* dfile, const char* compiled = "compiled")
{
   // -- Call the system diff to compare 2 files (dfile and compiled).
   FILE* fp = fopen(dfile, "a");
   fprintf(fp, "%s %s\n", title, macro);
   fclose(fp);
   char com[4096];
#if G__VISUAL
  const char* extraoption = "--strip-trailing-cr ";
#else // G__VISUAL
  const char* extraoption = "";
#endif // G__VISUAL
   sprintf(com, "diff %s --old-group-format=\"%s %s:%%c'\\012'%%<\" --new-group-format=\"%s interpreted:%%c'\\012'%%>\" --unchanged-line-format=\"\" --old-line-format=\" %%3dn: %%L\" --new-line-format=\" %%3dn: %%L\" %s interpreted>> %s", extraoption, title, compiled, title, compiled, dfile);
   int ret;
   if (bIgnoreDiffErrors) {
      // for now we want to continue despite diffs in the text
      ret = system(com);
   } else {
      ret = run(com,false);
   }
   return ret;
}

//______________________________________________________________________________
bool check_skip(const char* sname)
{
   // -- FIXME: Describe this function!
   if (debug) {
      if (debug[0] == '+') {
         if (debug[1] == '+') {
            if (!strcmp(debug + 2, sname)) {
               debug = 0;
            }
            return true;
         }
         else {
            if (strcmp(debug + 1, sname)) {
               return true;
            }
            else {
               debug = 0;
            }
         }
      }
      else
         if (strcmp(debug, sname)) {
            return true;
         }
#ifdef CONTINUE_TEST
         else {
            debug = 0;
         }
#endif // CONTINUE_TEST
   }
   return false;
}

//______________________________________________________________________________
int buildDictionaryLibrary(ELanguage lang, const char *dir, const char* sname, const char* macro, const char* src, const char *suffix = "")
{
   // Generate the shared library with a dictionary for the test.
   // Run makecint then run the generate makefile.

   char com[4096];

   const char *hopt = lang == kLangCXX ? "H" : "h";

   sprintf(com, "cd %s %s makecint -mk Makefile%s %s -dl test%s.dll %s -%s ../%s %s", 
           dir, shellSeparator, suffix, mkcintoption.c_str(), suffix, macro, hopt, sname, src);
   run(com);
   sprintf(com, "cd %s %s make -f Makefile%s",dir,shellSeparator,suffix);
   run(com);

}


//______________________________________________________________________________
int ci(ELanguage lang, const char* sname, const char* dfile, const char* cflags = "", const char* exsname = "", const char* cintopt = "", const char *dir = ".")
{
   // -- Compare compiled and interpreted result.
   if (check_skip(sname)) {
      return 0;
   }
   printf("%s %s %s\n", sname, cflags, exsname);
   char exename[4096];
   strcpy(exename, sname);
   char* posExt = strrchr(exename, '.');
   if (posExt) {
      strcpy(posExt, ".exe");
   }
   // compile source
   const char* comp = 0;
   const char* flags = 0;
   const char* macros = 0;
   const char* ldflags = 0;
   const char* link = "";
   if (lang == kLangC) {
      comp = G__CFG_CC;
      flags = G__CFG_CFLAGS;
      ldflags = G__CFG_LDFLAGS;
      macros = G__CFG_CMACROS;
   }
   else if (lang == kLangCXX) {
      comp = G__CFG_CXX;
      flags = G__CFG_CXXFLAGS;
      ldflags = G__CFG_LDFLAGS;
      macros = G__CFG_CXXMACROS;
   }
   else {
      printf("ERROR in ci: language is not set!\n");
      return 0;
   }
#if defined(G__WIN32)
   link = "/link";
#endif
   char com[4096];
   sprintf(com, "%s -Dcompiled %s %s %s %s %s %s %s%s %s %s", comp, cflags, 
      compileroption.c_str(),
      flags, macros, sname, exsname, G__CFG_COUTEXE, exename, link, ldflags);
   //fprintf(stderr, "ci: run: %s\n", com);
   run(com);
   // run compiled program
#ifdef G__WIN32
   sprintf(com, ".\\%s > compiled", exename);
#else // G__WIN32
   sprintf(com, "./%s > compiled", exename);
#endif // G__WIN32
   //fprintf(stderr, "ci: run: %s\n", com);
   run(com);
#ifdef DEBUG2
   //fprintf(stderr, "ci: run: %s\n", exename);
   run(exename);
#endif // DEBUG2
#ifndef DEBUG
   //fprintf(stderr, "ci: rm %s\n", exename);
   rm(exename);
#endif // DEBUG
#if defined(G__WIN32) || defined(G__CYGWIN)
   if (posExt) {
      strcpy(posExt, ".obj");
   }
   rm(exename);
   if (posExt) {
      strcpy(posExt, ".exe.manifest");
   }
   rm(exename);
   if (posExt) {
      strcpy(posExt, ".pdb");
   }
   rm(exename);
#endif // G__WIN32 || G__CYGWIN
#ifdef G__BORLAND
   if (posExt) {
      strcpy(posExt, ".tds");
   }
   rm(exename);
#endif
   // run interpreted program
   sprintf(com, "%s cint -I%s %s %s -Dinterp %s %s %s %s > interpreted", prefixcmd.c_str(), dir, cintoption.c_str(), debug ? "-DDEBUG" : "", cintopt, cflags, exsname, sname);
   //fprintf(stderr, "ci: run: %s\n", com);
   run(com);
   int ret = ediff(sname, cflags, dfile);
#ifndef DEBUG
   //fprintf(stderr, "ci: rm %s\n", "compiled");
   rm("compiled");
   //fprintf(stderr, "ci: rm %s\n", "interpreted");
   rm("interpreted");
#endif // DEBUG
   return ret;
}

//______________________________________________________________________________
int io(const char* sname, const char* old, const char*dfile, const char* macro = "")
{
   // -- Check output of interpreted program.
   if (check_skip(sname)) {
      return 0;
   }
   printf("%s\n", sname);
   // run interpreted program
   char com[4096];
   sprintf(com, "%s cint %s %s %s > interpreted", prefixcmd.c_str(), cintoption.c_str(), macro, sname);
   run(com);
   int ret = ediff(sname, "", dfile, old);
   //diff(sname, old, "interpreted", dfile, "", "o", "i");
   rm("interpreted");
   return ret;
}

//______________________________________________________________________________
int mkc(ELanguage lang, const char* sname, const char* dfile, const char* macro = "", const char* src = "")
{
   // -- Check difference in output between compiled and interpreted code, with dictionary.
   if (check_skip(sname)) {
      return 0;
   }
   printf("%s\n", sname);

   char *dir = cleanAndCreateDirectory(sname);
   buildDictionaryLibrary(kLangCXX, dir, sname, macro, src);

   char com[4096];
   // run interpreted program
   sprintf(com, "-DHNAME=\\\"%s\\\" -DDNAME=\\\"%s/test.dll\\\"", sname,dir);
   int ret = ci(lang, "mkcmain.cxx", dfile, com, "", "", dir);
#ifndef DEBUG
   cleanDirectory(dir);
#endif // DEBUG
   delete [] dir;
   return ret;
}

//______________________________________________________________________________
int mkco(ELanguage lang, const char* sname, const char* hname, const char* old, const char* dfile, const char* macro = "", const char* src = "", const char* cintopt = "")
{
   // -- Check output of interpreted code, with dictionary.
   if (check_skip(sname)) {
      return 0;
   }
   printf("%s\n", sname);
   char *dir = cleanAndCreateDirectory(sname);
   buildDictionaryLibrary(lang, dir, hname, macro, src);

   // run interpreted program
   char imacro[4096];
   sprintf(imacro, "%s -Dmakecint", macro);
   int ret = io(sname, old, dfile, imacro);

#ifndef DEBUG
   cleanDirectory(dir);
#endif // DEBUG
   delete [] dir;
   return ret;
}

//______________________________________________________________________________
int mkci(ELanguage lang, const char* sname, const char* hname, const char* dfile, const char* macro = "", const char* src = "", const char* cintopt = "")
{
   // -- Check difference in output of compiled and interpreted code, with dictionary.
   if (check_skip(sname)) {
      return 0;
   }
   printf("%s\n", sname);
   char *dir = cleanAndCreateDirectory(sname);
   buildDictionaryLibrary(lang, dir, hname, macro, src);

   // run interpreted program
   char imacro[4096];
   sprintf(imacro, "%s -Dmakecint", macro);
   int ret = ci(lang, sname, dfile, imacro, "", cintopt, dir);

#ifndef DEBUG
   cleanDirectory(dir);
 #endif // DEBUG
   delete [] dir;
   return ret;
}

//______________________________________________________________________________
int mkciN(ELanguage lang, const char* sname, const char* hname1, const char* dfile, const char* macro = "", const char* hname2 = "", const char* hname3 = "")
{
   // -- Check difference in output between compiled and interpreted code, with up to three dictionaries.
   if (check_skip(sname)) {
      return 0;
   }
   printf("%s\n", sname);
   
   char *dir = cleanAndCreateDirectory( sname );
   buildDictionaryLibrary(lang, dir, hname1, macro, "", "1");
   if (hname2[0]) buildDictionaryLibrary(lang, dir, hname2, macro, "", "2");
   if (hname3[0]) buildDictionaryLibrary(lang, dir, hname3, macro, "", "3");

   // run interpreted program
   char imacro[4096];
   sprintf(imacro, "%s -Dmakecint2", macro);
   int ret = ci(lang, sname, dfile, imacro, "", "", dir);
#ifndef DEBUG
   cleanDirectory( dir );
#endif // DEBUG
   delete [] dir;
   return ret;
}

//______________________________________________________________________________
int testn(ELanguage lang, const char* hdr, int* num, const char* ext, const char* dfile, const char* macro = "")
{
   // -- Test series of files with enumerated suffix.
   char sname[4096];
   int ret = 0;
   int i = 0;
   while (num[i] != -1) {
      sprintf(sname, "%s%d%s", hdr, num[i], ext);
      ret += ci(lang, sname, dfile, macro);
      ++i;
   }
   return ret;
}

//______________________________________________________________________________
int main(int argc, char** argv)
{
   // Put this default first so it can be 
   // over-ridden
   cintoption += " -O0 ";

   const char* difffile = "testdiff.txt";
   int ret = 0;
   for (int i = 1; i < argc; i++) {
      if (!strcmp("-d", argv[i]) && !strstr(argv[i+1], ".cxx")) {
         difffile = argv[++i];
      }
      else if (!strcmp("-c", argv[i]) && !strstr(argv[i+1], ".cxx")) {
         cintoption = argv[++i];
      }
      else if (!strcmp("-m", argv[i]) && !strstr(argv[i+1], ".cxx")) {
         mkcintoption = argv[++i];
      }
      else if (!strcmp("-k", argv[i])) {
         bKeepOnGoing = true;
      }
      else if (!strcmp("--ignore-diff-errors", argv[i])) {
         bIgnoreDiffErrors = true;
      }
      else if (!strcmp("--hide-known-errors", argv[i]) || 
               !strcmp("--hide-known-defects", argv[i]) ||
               !strcmp("--hide", argv[i])) {
         bHideKnownErrors = true;
      }
      else if (!strcmp("--time", argv[i])) {
         prefixcmd = "time";
      }
      else if (!strcmp("--valgrind", argv[i])) {
         prefixcmd = "valgrind";
      }
      else if (!strcmp("--prefix", argv[i])) {
         if ((i+1)<argc) {
            ++i;
            prefixcmd = argv[i];
         }
      }
      else if (!strcmp("-?", argv[i]) || (argv[i][0]=='-')) {
         if (argv[i][1]!='?') {
            fprintf(stderr,"Unknown parameter: %s\n",argv[i]);
         }
         fprintf(stderr,"%s -k <-d [difffile]> <-c [cintoption]> <-m [makecintoption]> <[testcase.cxx]>\n",argv[0]);
         fprintf(stderr,"   use -k to keep on going even if errors are encountered,\n");
         fprintf(stderr,"   use --ignore-diff-errors to downgraded the output difference to a warning,\n");
         fprintf(stderr,"   use --hide-known-defect or --hide to only test full supported feastures,\n");
         fprintf(stderr,"   use --prefix 'command' to have the cint command line passed to command,\n");
         fprintf(stderr,"   use --time to bring the time spend in each test,\n");
         fprintf(stderr,"   use '+testcase.cxx' to skip all before testcase.cxx,\n");
         fprintf(stderr,"   use '++testcase.cxx' to skip all before and including testcase.cxx\n");
         return 0;
      }
      else {
         debug = argv[i];
      }
   }
   if (!debugMode) {
      mkcintoption += " -q ";
   }
   if (bHideKnownErrors) {
      cintoption += " -DCINT_HIDE_FAILURE ";
      compileroption += " -DCINT_HIDE_FAILURE ";
      mkcintoption += " -DCINT_HIDE_FAILURE ";
   }

#ifdef TARGET
    debug = TARGET;
    //bKeepOnGoing = true;
#endif
   clear(difffile);

   ret += io("simple01.cxx","simple01.ref",difffile,"-DTARGET=\\\"simple01.cxx\\\"");
   ret += ci(kLangCXX, "simple10.cxx", difffile);
   ret += ci(kLangCXX, "simple11.cxx", difffile);
   ret += mkci(kLangCXX, "simple11.cxx", "simple11.cxx", difffile);
   ret += io("simple12.cxx", "simple12.ref", difffile);
   ret += io("simple13.cxx", "simple13.ref", difffile);
   ret += ci(kLangCXX, "simple14.cxx", difffile);
   ret += ci(kLangCXX, "simple15.cxx", difffile);
   ret += ci(kLangCXX, "simple16.cxx", difffile);
   ret += ci(kLangCXX, "simple17.cxx", difffile);
   ret += ci(kLangCXX, "simple18.cxx", difffile);
   ret += ci(kLangCXX, "simple19.cxx", difffile);
   ret += io("simple20.cxx", "simple20.ref", difffile);
   ret += io("simple21.cxx", "simple21.ref", difffile);
   ret += io("simple22.cxx", "simple22.ref", difffile);
   ret += io("simple23.cxx", "simple23.ref", difffile);
   ret += io("simple24.cxx", "simple24.ref", difffile);
   ret += mkci(kLangCXX,"simple25.cxx","simple25.h",difffile);
   int cpp[] = { 0, 1, 2, 3, 4, 5, 6, 8, -1 };
   ret += testn(kLangCXX, "cpp", cpp, ".cxx", difffile);
   ret += ci(kLangCXX, "bool01.cxx", difffile);
   ret += ci(kLangCXX, "switch.cxx", difffile);
   ret += ci(kLangCXX, "refassign.cxx", difffile);
   ret += ci(kLangCXX, "ostream.cxx", difffile);  // cout << pointer
   ret += ci(kLangCXX, "setw0.cxx", difffile);    // VC++6.0 setbase()
   int inherit[] = { 0, 1, 2, -1 };
   ret += testn(kLangCXX, "inherit", inherit, ".cxx", difffile);
   int virtualfunc[] = { 0, 1, 2, -1 };
   ret += testn(kLangCXX, "virtualfunc", virtualfunc, ".cxx", difffile);
   int oprovld[] = { 0, 2, -1 };
   ret += testn(kLangCXX, "oprovld", oprovld, ".cxx", difffile);
   ret += ci(kLangCXX, "constary.cxx", difffile);
   ret += ci(kLangCXX, "const.cxx", difffile);
   ret += ci(kLangCXX, "scope0.cxx", difffile);
   ret += ci(kLangCXX, "idxscope0.cxx", difffile);
   ret += ci(kLangCXX, "access0.cxx", difffile);
   ret += ci(kLangCXX, "staticmem0.cxx", difffile);
   ret += ci(kLangCXX, "staticmem1.cxx", difffile);
   ret += ci(kLangCXX, "staticary.cxx", difffile);
   ret += ci(kLangCXX, "static_object.cxx", difffile);
   ret += ci(kLangCXX, "static_string.cxx", difffile);
   ret += ci(kLangCXX, "static_call.cxx", difffile);
   ret += ci(kLangCXX, "minexam.cxx", difffile);
   ret += ci(kLangCXX, "btmplt.cxx", difffile);
   int loopcompile[] = { 1, 2, 3, 4, 5, -1 };
   ret += testn(kLangCXX, "loopcompile", loopcompile, ".cxx", difffile);
   ret += ci(kLangCXX, "mfstatic.cxx", difffile);
   ret += ci(kLangCXX, "new0.cxx", difffile);
#if defined(G__MSC_VER)&&(G__MSC_VER<=1200)
   int template_tests[] = { 0, 1, 2, 4, 6, -1 };
#else // G__MSC__VER && (G__MSC_VER <= 1200)
   int template_tests[] = { 0, 1, 2, 4, 5, 6, -1 };
#endif // G__MSC__VER && (G__MSC_VER <= 1200)
   ret += testn(kLangCXX, "template", template_tests, ".cxx", difffile);
   ret += io("template3.cxx", "template3.ref", difffile);
   ret += ci(kLangCXX, "minherit0.cxx", difffile);
   ret += ci(kLangCXX, "enumscope.cxx", difffile);
   ret += ci(kLangCXX, "baseconv0.cxx", difffile);
   ret += ci(kLangCXX, "friend0.cxx", difffile);
   ret += ci(kLangCXX, "anonunion.cxx", difffile);
   ret += ci(kLangCXX, "init1.cxx", difffile);
   ret += ci(kLangCXX, "init2.cxx", difffile);
   ret += ci(kLangCXX, "include.cxx", difffile);
   ret += ci(kLangCXX, "eh1.cxx", difffile);
   ret += ci(kLangCXX, "ifs.cxx", difffile);
   ret += ci(kLangCXX, "bitfield.cxx", difffile);
   ret += ci(kLangCXX, "cout1.cxx", difffile);
   ret += ci(kLangCXX, "longlong.cxx", difffile);
   ret += ci(kLangCXX, "explicitdtor.cxx", difffile);//fails due to base class dtor
   int nick[] = { 3, 4, -1 };
   ret += testn(kLangCXX, "nick", nick, ".cxx", difffile);
   ret += ci(kLangCXX, "nick4.cxx", difffile, "-DDEST");
   int telea[] = { 0, 1, 2, 3, 5, 6, 7, -1 };
   ret += testn(kLangCXX, "telea", telea, ".cxx", difffile);
   ret += ci(kLangCXX, "fwdtmplt.cxx", difffile);
   ret += ci(kLangCXX, "VPersonTest.cxx", difffile);
   ret += ci(kLangCXX, "convopr0.cxx", difffile);
   ret += ci(kLangCXX, "nstmplt1.cxx", difffile);
   ret += ci(kLangCXX, "aoki0.cxx", difffile);
   ret += ci(kLangCXX, "borg1.cxx", difffile);
   ret += ci(kLangCXX, "borg2.cxx", difffile);
   // This test currently fails because the char** argument is
   // registered as a char *&
   ret += ci(kLangCXX, "bruce1.cxx", difffile);
   ret += ci(kLangCXX, "fons3.cxx", difffile);
   ret += ci(kLangCXX, "Test0.cxx", difffile, "", "MyString.cxx");
   ret += ci(kLangCXX, "Test1.cxx", difffile, "", "Complex.cxx MyString.cxx");
   ret += ci(kLangCXX, "delete0.cxx", difffile);
   ret += ci(kLangCXX, "pb19.cxx", difffile);
#ifdef AUTOCC
   ret += ci(kLangCXX, "autocc.cxx", difffile);
   system("rm G__*");
#endif // AUTOCC
   ret += ci(kLangCXX, "maincmplx.cxx", difffile, "", "complex1.cxx");
   ret += ci(kLangCXX, "funcmacro.cxx", difffile);
   ret += ci(kLangCXX, "template.cxx", difffile);
   ret += mkci(kLangCXX, "template.cxx", "template.h", difffile);
   ret += ci(kLangCXX, "vbase.cxx", difffile);
   ret += mkci(kLangCXX, "vbase.cxx", "vbase.h", difffile);
   ret += ci(kLangCXX, "vbase1.cxx", difffile);
   ret += mkci(kLangCXX, "vbase1.cxx", "vbase1.h", difffile);


#define PROBLEM
#if defined(PROBLEM) && (!defined(G__WIN32) || defined(FORCEWIN32))
   ret += mkci(kLangCXX, "t674.cxx", "t674.h", difffile); // Problem with VC++6.0
   ret += ci(kLangCXX, "t648.cxx", difffile); // long long has problem with BC++5.5
                                    // also with VC++6.0 bug different
   ret += mkci(kLangCXX,"t977.cxx","t977.h",difffile); // VC++ problem is known
   ret += ci(kLangCXX, "t980.cxx", difffile); // problem with BC++5.5
#if (G__GNUC==2)
   ret += mkci(kLangCXX, "t1030.cxx", "t1030.h", difffile); // works only with gcc2.96
   ret += mkci(kLangCXX, "t1031.cxx", "t1031.h", difffile); // works only with gcc2.96
   //ret += mkci(kLangCXX,"t1030.cxx","t1030.h",difffile,"","","-Y0");
   //ret += mkci(kLangCXX,"t1031.cxx","t1031.h",difffile,"","","-Y0");
#endif // G__GNUC == 2
#endif // PROBLEM && (!G__WIN32 || FORCEWIN32)
   ret += ci(kLangCXX, "t215.cxx", difffile);
   ret += ci(kLangCXX, "t358.cxx", difffile);
   ret += ci(kLangCXX, "t488.cxx", difffile);
   ret += ci(kLangCXX, "t516.cxx", difffile);
   ret += ci(kLangCXX, "t603.cxx", difffile);
   ret += ci(kLangCXX, "t627.cxx", difffile);
   ret += mkci(kLangCXX, "t627.cxx", "t627.h", difffile);
   ret += ci(kLangCXX, "t630.cxx", difffile);
   ret += ci(kLangCXX, "t633.cxx", difffile);
   ret += mkci(kLangCXX, "t633.cxx", "t633.h", difffile);
   ret += ci(kLangCXX, "t634.cxx", difffile);
   ret += ci(kLangCXX, "t674.cxx", difffile, "-DINTERPRET");
#if !defined(G__WIN32) && !defined(G__CYGWIN) && !defined(G__APPLE)
   ret += ci(kLangCXX, "t676.cxx", difffile); //recursive call stack too deep for Visual C++
#endif // !G__WIN32 && !G__CYGWIN && !G__APPLE
   ret += mkci(kLangCXX, "t694.cxx", "t694.h", difffile);
   ret += ci(kLangCXX, "t694.cxx", difffile, "-DINTERPRET"); //fails due to default param
   ret += ci(kLangCXX, "t695.cxx", difffile); //fails due to tmplt specialization
   ret += mkci(kLangCXX, "t705.cxx", "t705.h", difffile);
   ret += ci(kLangCXX, "t705.cxx", difffile, "-DINTERPRET");
   ret += ci(kLangCXX, "t714.cxx", difffile);
   ret += io("t733.cxx", "t733.ref", difffile);
#if !defined(G__WIN32) || defined(FORCEWIN32)
   //NOT WORKING: in debug mode on WINDOWS!
   ret += ci(kLangCXX,"t749.cxx",difffile);
#endif // !G__WIN32 || FORCEWIN32
   ret += ci(kLangCXX, "t751.cxx", difffile);
   ret += ci(kLangCXX, "t764.cxx", difffile);
   ret += ci(kLangCXX, "t767.cxx", difffile);
   ret += ci(kLangCXX, "t776.cxx", difffile);
   ret += ci(kLangCXX, "t777.cxx", difffile);
   ret += ci(kLangCXX, "t784.cxx", difffile);
   ret += ci(kLangCXX, "t825.cxx", difffile);
   ret += ci(kLangCXX, "t910.cxx", difffile);
   ret += ci(kLangCXX, "t916.cxx", difffile);
#if !defined(G__VISUAL) || defined(FORCEWIN32)
#if G__CINTVERSION < 70000000
   ret += io("t927.cxx","t927.ref5",difffile);
#else
   ret += io("t927.cxx","t927.ref",difffile);
#endif
#endif // !G__VISUAL || FORCEWIN32
#if !defined(G__WIN32) || defined(FORCEWIN32)
   ret += mkciN(kLangCXX, "t928.cxx", "t928.h", difffile, "", "t928a.h", "t928b.h");
#endif // !G__WIN32 | FORCEWIN32
   ret += ci(kLangCXX, "t930.cxx", difffile);
   ret += ci(kLangCXX, "t938.cxx", difffile);
   ret += ci(kLangCXX, "t958.cxx", difffile);
   ret += ci(kLangCXX, "t959.cxx", difffile);
   ret += mkci(kLangCXX, "t961.cxx", "t961.h", difffile); //mkc(kLangCXX,"t961.h",difffile);
                                                //Borland C++5.5 has a problem
                                                //with reverse_iterator::reference
   ret += ci(kLangCXX, "t963.cxx", difffile);
#ifdef G__P2F
   ret += mkci(kLangCXX, "t966.cxx", "t966.h", difffile);
#endif // G__P2F
   ret += mkci(kLangCXX, "t968.cxx", "t968.h", difffile); // problem with BC++5.5 & VC++6.0
   ret += mkci(kLangCXX, "t970.cxx", "t970.h", difffile);
   ret += mkciN(kLangCXX, "t972.cxx", "t972a.h", difffile, "", "t972b.h");
#if !defined(G__WIN32) || defined(FORCEWIN32)
   ret += mkci(kLangCXX, "t980.cxx", "t980.h", difffile);
#endif // !G__WIN32 | FORCEWIN32
#if !defined(G__WIN32) || defined(FORCEWIN32)
   ret += ci(kLangCXX, "t986.cxx", difffile, "-DTEST");
#endif // !G__WIN32 | FORCEWIN32
   ret += mkci(kLangCXX, "t987.cxx", "t987.h", difffile);
   ret += mkciN(kLangCXX, "t991.cxx", "t991a.h", difffile, "", "t991b.h", "t991c.h");
   ret += mkci(kLangCXX, "t992.cxx", "t992.h", difffile);  // problem gcc3.2
   ret += mkci(kLangCXX, "maptest.cxx", "maptest.h", difffile); // problem icc
   ret += mkci(kLangC, "t993.c", "t993.h", difffile);
   ret += mkci(kLangCXX, "t995.cxx", "t995.h", difffile);
   ret += mkci(kLangCXX, "t996.cxx", "t996.h", difffile);
   ret += ci(kLangCXX, "t998.cxx", difffile);
   ret += mkci(kLangCXX, "t1002.cxx", "t1002.h", difffile);
   ret += ci(kLangCXX, "t1004.cxx", difffile);
   ret += ci(kLangCXX, "t1011.cxx", difffile);
   ret += mkci(kLangCXX, "t1011.cxx", "t1011.h", difffile);
   ret += ci(kLangCXX, "t1015.cxx", difffile);
   ret += ci(kLangCXX, "t1016.cxx", difffile);
   ret += mkci(kLangCXX, "t1016.cxx", "t1016.h", difffile);
   ret += ci(kLangCXX, "t1023.cxx", difffile);
   ret += ci(kLangCXX, "t1024.cxx", difffile);
   ret += mkci(kLangCXX, "t1024.cxx", "t1024.h", difffile);
#if !defined(G__WIN32) || defined(FORCEWIN32)
   ret += mkci(kLangCXX, "t1025.cxx", "t1025.h", difffile);
#endif // !G__WIN32 | FORCEWIN32
   ret += ci(kLangCXX, "t1026.cxx", difffile); // problem with BC++5.5
   ret += mkci(kLangCXX, "t1026.cxx", "t1026.h", difffile);
   ret += io("t1027.cxx", "t1027.ref", difffile);
   //ret += ci(kLangCXX,"t1027.cxx",difffile); // problem with BC++5.5
   //ret += mkci(kLangCXX,"t1027.cxx","t1027.h",difffile);
   ret += ci(kLangCXX, "t1032.cxx", difffile);
   ret += ci(kLangCXX, "t1032.cxx", difffile);
   
   ret += ci(kLangCXX, "t1034a.cxx", difffile);
   
#if !defined(G__WIN32) || defined(FORCEWIN32)
   if (sizeof(long double)==16) {
      ret += io("t1034.cxx", "t1034.ref64", difffile); // sizeof(long double)==16      
   } else if (sizeof(long)==4) {
      ret += io("t1034.cxx", "t1034.ref", difffile); // sizeof(long double)==12
   } else if (sizeof(void*)==8) {
      ret += io("t1034.cxx", "t1034.ref64", difffile); // sizeof(long double)==16
   } else {
      ret += io("t1034.cxx", "t1034.refXX", difffile); // sizeof(long double)==12
   }
#endif // !G__WIN32 | FORCEWIN32
   ret += ci(kLangCXX, "t1035.cxx", difffile);
   ret += mkci(kLangCXX, "t1035.cxx", "t1035.h", difffile);
   ret += ci(kLangCXX, "t1036.cxx", difffile);
   ret += mkci(kLangCXX, "t1040.cxx", "t1040.h", difffile); // gcc3.2 has problem
   ret += io("t1042.cxx", "t1042.ref", difffile);
#if !defined(G__WIN32) || defined(FORCEWIN32)
   ret += ci(kLangCXX,"t1046.cxx",difffile);
   ret += mkci(kLangCXX,"t1046.cxx","t1046.h",difffile);
#endif // !G__WIN32 | FORCEWIN32
   ret += ci(kLangCXX, "t1047.cxx", difffile);
   ret += mkci(kLangCXX, "t1047.cxx", "t1047.h", difffile);
   ret += ci(kLangCXX, "t1048.cxx", difffile);
   ret += mkci(kLangCXX, "t1048.cxx", "t1048.h", difffile, "-I.. -I../../inc -I../../src -I../../reflex/inc");
   ret += ci(kLangCXX, "t1049.cxx", difffile);
   ret += ci(kLangCXX, "t1054.cxx", difffile);
   ret += ci(kLangCXX, "t1055.cxx", difffile);
   ret += mkci(kLangCXX, "t1061.cxx", "t1061.h", difffile);
#if !defined(G__WIN32) || defined(FORCEWIN32)
   ret += mkci(kLangCXX, "t1062.cxx", "t1062.h", difffile);
#endif // !G__WIN32 | FORCEWIN32
   ret += ci(kLangCXX, "t1067.cxx", difffile);
   ret += mkci(kLangCXX, "t1067.cxx", "t1067.h", difffile);
   ret += ci(kLangCXX, "t1068.cxx", difffile);
   ret += mkci(kLangCXX, "t1068.cxx", "t1068.h", difffile);
   ret += ci(kLangCXX, "t1079.cxx", difffile);
   ret += mkci(kLangCXX, "t1079.cxx", "t1079.h", difffile);
   ret += ci(kLangCXX, "t1084.cxx", difffile);
   ret += ci(kLangCXX, "t1085.cxx", difffile);
   ret += ci(kLangCXX, "t1086.cxx", difffile);
   ret += ci(kLangCXX, "t1088.cxx", difffile);
   ret += ci(kLangCXX, "t1094.cxx", difffile);
   ret += ci(kLangCXX, "t1101.cxx", difffile);
   ret += mkci(kLangCXX, "t1115.cxx", "t1115.h", difffile);
   ret += ci(kLangCXX, "t1124.cxx", difffile);
   ret += ci(kLangCXX, "t1125.cxx", difffile);
   ret += ci(kLangCXX, "t1126.cxx", difffile);
#if !defined(G__APPLE)
   // This not work on macos and on 64bit linux because of var_arg
   if (sizeof(void*)<8 || (G__CINTVERSION > 70000000) ) {
      ret += ci(kLangCXX, "t1127.cxx", difffile);
      ret += mkci(kLangCXX, "t1127.cxx", "t1127.h", difffile);  //
   }
#endif // !G__APPLE
   ret += ci(kLangCXX, "t1128.cxx", difffile);  // looks to me gcc3.2 has a bug
   ret += ci(kLangCXX, "t1129.cxx", difffile);  // g++3.2 fails
   ret += ci(kLangCXX, "t1134.cxx", difffile);
   ret += ci(kLangCXX, "t1136.cxx", difffile);
   ret += ci(kLangCXX, "t1140.cxx", difffile);
   ret += ci(kLangCXX,"t1144.cxx",difffile);
   ret += ci(kLangCXX,"t1144.cxx",difffile,"","","-Y0");
   ret += ci(kLangCXX,"t1144.cxx",difffile,"","","-Y1");
   ret += ci(kLangCXX, "t1148.cxx", difffile);
   ret += ci(kLangCXX, "t1157.cxx", difffile);
   ret += ci(kLangCXX, "t1158.cxx", difffile);
   ret += ci(kLangCXX, "t1160.cxx", difffile);
   ret += ci(kLangCXX, "aryinit0.cxx", difffile);
   ret += ci(kLangCXX, "aryinit1.cxx", difffile);
   ret += ci(kLangCXX, "t1164.cxx", difffile);
   ret += ci(kLangCXX, "t1165.cxx", difffile);
   ret += ci(kLangCXX, "t1178.cxx", difffile);
   ret += mkci(kLangCXX, "t1187.cxx", "t1187.h", difffile);
   ret += ci(kLangCXX, "t1192.cxx", difffile);
   ret += mkci(kLangCXX, "t1193.cxx", "t1193.h", difffile);
   ret += ci(kLangCXX, "t1203.cxx", difffile);
   ret += ci(kLangCXX, "t1205.cxx", difffile);
   ret += mkci(kLangCXX, "t1205.cxx", "t1205.h", difffile);
   ret += ci(kLangCXX, "t1213.cxx", difffile);
   ret += ci(kLangCXX, "t1214.cxx", difffile);
   ret += ci(kLangCXX, "t1215.cxx", difffile);
   ret += mkci(kLangCXX, "t1215.cxx", "t1215.h", difffile);
   ret += ci(kLangCXX, "t1221.cxx", difffile);
   ret += ci(kLangCXX, "t1222.cxx", difffile);
   ret += ci(kLangCXX, "t1223.cxx", difffile);
   ret += ci(kLangCXX, "t1224.cxx", difffile);
   ret += io("t1228.cxx", "t1228.ref", difffile);
   ret += mkciN(kLangCXX, "t1247.cxx", "t1247.h", difffile, "", "t1247a.h");
   ret += mkci(kLangCXX, "t1276.cxx", "t1276.h", difffile);
   ret += mkci(kLangCXX, "t1277.cxx", "t1277.h", difffile); // works only with gcc2.96
   ret += ci(kLangCXX, "t1278.cxx", difffile);
   ret += ci(kLangCXX, "t1279.cxx", difffile);
   ret += ci(kLangCXX, "t1280.cxx", difffile);
   ret += ci(kLangCXX, "t1281.cxx", difffile);
   ret += ci(kLangCXX, "t1282.cxx", difffile);
   ret += ci(kLangCXX, "t1283.cxx", difffile);
   ret += ci(kLangCXX, "t1284.cxx", difffile);
#if G__CINTVERSION > 70000000
   ret += ci(kLangCXX, "t1286.cxx", difffile);
#endif
   ret += ci(kLangCXX, "postinc.cxx", difffile);
   ret += mkci(kLangCXX, "selfreference.cxx", "selfreference.h", difffile);
   ret += ci(kLangCXX, "abstract.cxx", difffile);
   ret += ci(kLangCXX, "TException.cxx", difffile);
   ret += mkci(kLangCXX, "enums.cxx", "enums.h", difffile);
   ret += io("classinfo.cxx", "classinfo.ref", difffile);
   ret += ci(kLangCXX, "iostream_state.cxx", difffile);

   printf("Summary==================================================\n");
   cat(stdout, difffile);
   printf("=========================================================\n");
#ifndef DEBUG
   rm("Makefile");
#endif

   return (ret > 0);
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
