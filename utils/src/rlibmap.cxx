// @(#)root/utils:$Name:  $:$Id:
// Author: Fons Rademakers   05/12/2003

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// This program generates a map between class name and shared library.
// Its output is in TEnv format.
// Usage: rlibmap [-f] [-o <libmapfile>] <sofile> <sofile> ...
// -f: output full library path name (not needed when ROOT library
//     search path is used)
// -o: output to write to specified file, otherwise to stdout


#include <stdio.h>
#include <string>
#include <map>

const char *usage = "Usage: %s [-f] [-o <libmapfile>] <sofile> <sofile> ...\n";

#if defined(__linux) || defined(__FreeBSD__)
#if defined(__INTEL_COMPILER) || (__GNUC__ >= 3)
const char *kNM = "nm --demangle=gnu-v3";
#else
const char *kNM = "nm -C";
#endif
const char  kDefined = 'T';
#elif defined (__sun)
const char *kNM = "nm -C -p";
const char  kDefined = 'T';
#elif defined(__alpha)
const char *kNM = "nm -B";
const char  kDefined = 'T';
#elif defined(__hpux)
namespace std { }
const char *kNM = "nm++ -p";
const char  kDefined = 'T';
#elif defined(__APPLE__)
const char *kNM = "nm";
const char  kDefined = 'T';
#elif defined(__sgi)
const char *kNM = "nm -C";
const char  kDefined = 'T';
#elif defined(_AIX)
const char *kNM = "nm -C";
const char  kDefined = 'T';
#elif defined (__CYGWIN__) && defined(__GNUC__)
const char *kNM = "nm --demangle=gnu-v3";
const char  kDefined = 'T';
#elif defined(_WIN32)
const char *kNM = "nm -C";
const char  kDefined = 'T';
#else
#warning Platform specific case missing
#endif

using namespace std;


int libmap(const char *lib, int fullpath, FILE *fp)
{
   // Write libmap. Returns -1 in case of error.

   char *nm = new char [strlen(lib)+50];

#if defined(__APPLE__)
   sprintf(nm, "%s %s | c++filt", kNM, lib);
#else
   sprintf(nm, "%s %s", kNM, lib);
#endif

   FILE *pf;
#ifndef _WIN32
   if ((pf = popen(nm, "r")) == 0) {
      fprintf(stderr, "cannot execute: %s\n", nm);
      return 1;
   }
#else
   // excute nm and write to tmp file, open tmp file on pf
   pf = 0;
   if (!pf) return -1;
#endif

   map<string,bool> unique;
   char line[4096];
   while ((fgets(line, 4096, pf)) != 0) {
      //printf("line: %s", line);
      unsigned long addr;
      char type[5], symbol[4096];
      addr = 0;
      if (line[0] == '0') {
          sscanf(line, "%lx %s %s", &addr, type, symbol);
          //printf("addr = %.8lx, type = %s, symbol = %s\n", addr, type, symbol);
      } else {
          sscanf(line, "%s %s", type, symbol);
          //printf("              type = %s, symbol = %s\n", type, symbol);
      }

      if (type[0] == kDefined) {
         char *r;
         if ((r = strrchr(symbol, ':'))) {
            r--;
            if (r && *r == ':') {
               *r = 0;
               string cls = symbol;
               r += 2;
               char *s;
               if ((s = strchr(r, '[')) || (s = strchr(r, '('))) {
                  *s = 0;
                  string meth = r;
                  if (cls == meth) {
                     //printf("class %s in library %s\n", cls.c_str(), lib);
                     unique[cls] = true;
                  }
               }
            }
         }
      }
   }

#ifndef _WIN32
   pclose(pf);
#else
   fclose(pf);
#endif

   const char *libbase = strrchr(lib, '/');
   if (libbase && !fullpath)
      libbase++;
   else
      libbase = lib;

   map<string,bool>::const_iterator it;
   for (it = unique.begin(); it != unique.end(); it++) {
      fprintf(fp, "Library.%-35s %s\n", ((*it).first+":").c_str(), libbase);
   }

   delete [] nm;

   return 0;
}

int main(int argc, char **argv)
{
   char **libs  = 0;
   int fullpath = 0;
   FILE *fp     = stdout;

   if (argc > 1) {
      int ic = 1;
      if (!strcmp(argv[ic], "-f")) {
         fullpath = 1;
         ic++;
      }
      if (!strcmp(argv[ic], "-o")) {
         ic++;
         fp = fopen(argv[ic], "w");
         if (!fp) {
            fprintf(stderr, "cannot open output file %s\n", argv[ic]);
            return 1;
         }
         ic++;
      }
      if (!strcmp(argv[ic], "-?") || !strcmp(argv[ic], "-h")) {
         fprintf(stderr, usage, argv[0]);
         return 1;
      }
      int args = argc - ic + 2;
      libs = new char* [args];
      libs[args-1] = 0;
      for (int i = ic, j = 0; i < argc; i++, j++) {
         libs[j] = argv[i];
      }
   } else {
      fprintf(stderr, usage, argv[0]);
      return 1;
   }

   int i = 0;
   while (libs[i]) {
      libmap(libs[i], fullpath, fp);
      i++;
   }

   if (fp != stdout)
      fclose(fp);

   return 0;
}
