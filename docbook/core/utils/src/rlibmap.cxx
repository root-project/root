// @(#)root/utils:$Id$
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
// Usage: rlibmap [-f] [-o <mapfile>] -l <sofile> -d <depsofiles>
//                 -c <linkdeffiles>
// -f: output full library path name (not needed when ROOT library
//     search path is used)
// -o: write output to specified file, otherwise to stdout
// -r: replace existing entries in the specified file
// -l: library containing the classes in the specified linkdef files
// -d: libraries on which the -l library depends
// -c: linkdef files containing the list of classes defined in the -l library


#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cctype>
#include <string>
#include <string.h>
#include <vector>
#ifndef WIN32
#   include <unistd.h>
#else
#   define ssize_t int
#   include <io.h>
#   include <sys/types.h>
#   include "cygpath.h"
#endif

#ifdef __APPLE__
#include <AvailabilityMacros.h>
#endif
#if (defined(__FreeBSD__) && (__FreeBSD__ < 4)) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && (!defined(MAC_OS_X_VERSION_10_3) || \
     (MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_10_3)))
#include <sys/file.h>
#define lockf(fd, op, sz)   flock((fd), (op))
#ifndef F_LOCK
#define F_LOCK             (LOCK_EX | LOCK_NB)
#endif
#ifndef F_ULOCK
#define F_ULOCK             LOCK_UN
#endif
#endif

#if defined(__CYGWIN__) && defined(__GNUC__)
#define F_LOCK F_WRLCK
#define F_ULOCK F_UNLCK
static int fcntl_lockf(int fd, int op, off_t off)
{
   flock fl;
   fl.l_whence = SEEK_SET;
   fl.l_start  = off;
   fl.l_len    = 0;       // whole file
   fl.l_pid    = getpid();
   fl.l_type   = op;
   return fcntl(fd, F_SETLK, &fl);
}
#define lockf fcntl_lockf
#endif

const char *usage = "Usage: %s [-f] [<-r|-o> <mapfile>] -l <sofile> -d <depsofiles> -c <linkdeffiles>\n";

namespace std {}
using namespace std;


#ifdef WIN32
#include <windows.h>
#include <errno.h>

#define ftruncate(fd, size)  win32_ftruncate(fd, size)

//______________________________________________________________________________
int win32_ftruncate(int fd, ssize_t size)
{
   HANDLE hfile;
   int curpos;

   if (fd < 0) return -1;

   hfile = (HANDLE) _get_osfhandle(fd);
   curpos = ::SetFilePointer(hfile, 0, 0, FILE_CURRENT);
   if (curpos == 0xFFFFFFFF ||
       ::SetFilePointer(hfile, size, 0, FILE_BEGIN) == 0xFFFFFFFF ||
       !::SetEndOfFile(hfile)) {
         int error = ::GetLastError();

      switch (error) {
         case ERROR_INVALID_HANDLE:
            errno = EBADF;
            break;
         default:
            errno = EIO;
         break;
      }
      return -1;
   }
   return 0;
}

#endif // WIN32

//______________________________________________________________________________
char *Compress(const char *str)
{
   // Remove all blanks from the string str. The returned string has to be
   // deleted by the user.

   static const char* composedTypes[] = {"const ", "signed ","unsigned "};
   static unsigned int composedTypesLen[] = {6, 7, 9};

   if (!str) return 0;

   const char *p = str;
   // allocate 20 extra characters in case of eg, vector<vector<T>>
   char *s, *s1 = new char[strlen(str)+20];
   s = s1;

   while (*p) {
      if (*p != ' ') {
         for (unsigned int i = 0; i < sizeof(composedTypes) / sizeof(char*); ++i) {
            if (!strncmp(p, composedTypes[i], composedTypesLen[i])
                && (p == str || !isalnum(p[-1]))) {
               // the last one will be copied after the for loop
               memcpy(s, composedTypes[i], composedTypesLen[i] - 1);
               p += composedTypesLen[i] - 1;
               s += composedTypesLen[i] - 1;
               continue;
            }
         }
         *s++ = *p;
      }
      p++;
   }
   *s = '\0';

   return s1;
}

//______________________________________________________________________________
void UnCompressTemplate(char*& str)
{
   // Replace ">>" by "> >" except for operator>>.
   // str might be changed; the old string gets deleted in here.

   // Even handles cases like "A<B<operator>>()>>::operator >>()".

   char* pos = strstr(str, ">>");
   char* fixed = 0;
   int countgtgt = 0;
   while (pos) {
      // first run: just count, so we can allocate space for fixed
      ++countgtgt;
      pos = strstr(pos+1, ">>");
   }
   if (!countgtgt)
      return;
   pos = strstr(str, ">>");
   while (pos && pos > str) {
      bool isop = false;
      // check that it's not op>>:
      if (pos - str >= 8) {
         char* posop = pos - 1;
         // remove spaces in front of ">>":
         while (posop >= str && *posop == ' ')
            --posop;
         if (!strncmp("operator", posop - 7, 8)) {
            // it is an operator!
            isop = true;
         }
      }
      if (!isop) {
         // not an operator; we need to add a space.
         if (!fixed) {
            fixed = new char[strlen(str) + countgtgt + 1];
            strcpy(fixed, str);
         }
         fixed[pos - str + 1] = ' ';
         strcpy(fixed + (pos - str) + 2, pos + 1);
      }
      pos = strstr(pos + 1, ">>");
   }
   delete [] str;
   str = fixed;
}

//______________________________________________________________________________
int RemoveLib(const string &solib, bool fullpath, FILE *fp)
{
   // Remove entries from the map file for the specified solib.

   fseek(fp, 0, SEEK_SET);

   // get file size
   struct stat sbuf;
   fstat(fileno(fp), &sbuf);
   size_t siz = sbuf.st_size;

   if (!siz) return 0;

   const char *libbase = solib.c_str();
   if (!fullpath) {
      if ((libbase = strrchr(libbase, '/')))
         libbase++;
   }

   // read file and remove lines matching specified libs
   char *fbuf = new char[siz+1];
   char *fptr = fbuf;

   while (fgets(fptr, 1+siz - size_t(fptr-fbuf), fp)) {

      char *line = new char[strlen(fptr)+1];
      strcpy(line, fptr);
      strtok(line, " ");
      char *lib = strtok(0, " \n");
      if (lib && strcmp(lib, libbase)) {
         fptr += strlen(fptr);
         if (*(fptr-1) != '\n') {
            *fptr = '\n';
            fptr++;
         }
      }
      delete [] line;

      // fgets() should return 0 in this case but doesn't
      if ( (siz - size_t(fptr - fbuf)) <= 0)
         break;
   }

   if (ftruncate(fileno(fp), 0)) {;}

   // write remaining lines back
   if (fptr != fbuf) {
      fseek(fp, 0, SEEK_SET);
      fwrite(fbuf, 1, size_t(fptr-fbuf), fp);
   }

   delete [] fbuf;

   fseek(fp, 0, SEEK_END);

   return 0;
}

//______________________________________________________________________________
int LibMap(const string &solib, const vector<string> &solibdeps,
           const vector<string> &linkdefs, bool fullpath, FILE *fp)
{
   // Write libmap. Returns -1 in case of error.

   vector<string> classes;

   vector<string>::const_iterator lk;
   for (lk = linkdefs.begin(); lk != linkdefs.end(); lk++) {
      const char *linkdef = lk->c_str();
      FILE *lfp;
      char pragma[1024];
      if ((lfp = fopen(linkdef, "r"))) {
         while (fgets(pragma, 1024, lfp)) {
            if (strcmp(strtok(pragma, " "), "#pragma")) continue;
            const char* linkOrCreate = strtok(0, " ");
            bool pragmaLink = (!strcmp(linkOrCreate, "link") &&
                               !strcmp(strtok(0, " "), "C++"));
            bool pragmaCreate = (!strcmp(linkOrCreate, "create") &&
                                 !strcmp(strtok(0," "), "TClass"));

            if (pragmaLink || pragmaCreate) {
               const char *type = pragmaLink ? strtok(0, " ") : "class";
               if (!strncmp(type, "option=", 7) || !strncmp(type, "options=", 8)) {
                  if (strstr(type, "nomap"))
                     continue;
                  type = strtok(0, " ");
               }
               // handles class, class+protected and class+private, typedef
               if (!strncmp(type, "class", 5) || !strcmp(type, "typedef")) {
                  char *cls = strtok(0, "-!+;");
                  // just in case remove trailing space and tab
                  while (*cls == ' ') cls++;
                  int len = strlen(cls) - 1;
                  while (cls[len] == ' ' || cls[len] == '\t')
                     cls[len--] = '\0';
                  //no space between tmpl arguments allowed
                  cls = Compress(cls);
                  // except for A<B<C> >!
                  UnCompressTemplate(cls);

                  // don't include "vector<string>" and "std::pair<" classes
                  if (!strncmp(cls, "vector<string>", 14) ||
                      !strncmp(cls, "std::pair<", 10))
                     continue;

                  // replace "::" by "@@" since TEnv uses ":" as delimeter
                  char *s = cls;
                  while (*s) {
                     if (*s == ':')
                        *s = '@';
                     else if (*s == ' ')
                        *s = '-';
                     s++;
                  }
                  classes.push_back(cls);
               }
            }
         }
         fclose(lfp);
      } else {
         fprintf(stderr, "cannot open linkdef file %s\n", linkdef);
      }
   }

   const char *libbase = solib.c_str();
   if (!fullpath) {
      if ((libbase = strrchr(libbase, '/')))
         libbase++;
      else
         libbase = solib.c_str();
   }

   vector<string>::const_iterator it;
   for (it = classes.begin(); it != classes.end(); it++) {
      fprintf(fp, "Library.%-35s %s", ((*it)+":").c_str(), libbase);

      if (solibdeps.size() > 0) {
         vector<string>::const_iterator depit;
         for (depit = solibdeps.begin(); depit != solibdeps.end(); depit++) {
#ifdef WIN32
            string::size_type i = depit->find(".lib");
            if (i != string::npos)
               continue;
#endif

            const char *deplib = depit->c_str();
            if (!fullpath) {
               if ((deplib = strrchr(deplib, '/')))
                  deplib++;
               else
                  deplib = depit->c_str();
            }
            fprintf(fp, " %s", deplib);
         }
      }
      fprintf(fp, "\n");
   }

   return 0;
}

//______________________________________________________________________________
int main(int argc, char **argv)
{
   string         solib;
   vector<string> solibdeps;
   vector<string> linkdefs;
   bool fullpath = false;
   bool replace  = false;
   FILE *fp      = stdout;

   if (argc > 1) {
      int ic = 1;
      if (!strcmp(argv[ic], "-?") || !strcmp(argv[ic], "-h")) {
         fprintf(stderr, usage, argv[0]);
         return 1;
      }
      if (!strcmp(argv[ic], "-f")) {
         fullpath = true;
         ic++;
      }
      if (!strcmp(argv[ic], "-o")) {
         ic++;
         std::string outfile(argv[ic]);
#ifdef WIN32
         FromCygToNativePath(outfile);
         fp = fopen(outfile.c_str(), "w");
#else
         fp = fopen(argv[ic], "w");
#endif
         if (!fp) {
            fprintf(stderr, "cannot open output file %s\n", outfile.c_str());
            return 1;
         }
         ic++;
      }
      if (!strcmp(argv[ic], "-r")) {
         replace = true;
         ic++;
         std::string outfile(argv[ic]);
#ifdef WIN32
         FromCygToNativePath(outfile);
         fp = fopen(outfile.c_str(), "a+");
#else
         fp = fopen(outfile.c_str(), "a+");
#endif
         if (!fp) {
            fprintf(stderr, "cannot open output file %s\n", outfile.c_str());
            return 1;
         }
         ic++;
      }
      if (!strcmp(argv[ic], "-l")) {
         ic++;
         solib = argv[ic];
#ifdef WIN32
         FromCygToNativePath(solib);
#endif
#ifdef __APPLE__
         string::size_type i = solib.find(".dylib");
         if (i != string::npos)
            solib.replace(i, 6, ".so");
#endif
         ic++;
      }
      if (!strcmp(argv[ic], "-d")) {
         ic++;
         for (int i = ic; i < argc && argv[i][0] != '-'; i++) {
            string dl = argv[i];
#ifdef WIN32
            FromCygToNativePath(dl);
#endif
#ifdef __APPLE__
            string::size_type j = dl.find(".dylib");
            if (j != string::npos)
               dl.replace(j, 6, ".so");
#endif
            solibdeps.push_back(dl);
            ic++;
         }
      }
      if (!strcmp(argv[ic], "-c")) {
         ic++;
         for (int i = ic; i < argc; i++) {
#ifdef WIN32
            std::string linkdef(argv[i]);
            FromCygToNativePath(linkdef);
            linkdefs.push_back(linkdef);
#else
            linkdefs.push_back(argv[i]);
#endif
            ic++;
         }
      }
   } else {
      fprintf(stderr, usage, argv[0]);
      return 1;
   }

   if (replace) {
#if !defined(WIN32) && !defined(__CYGWIN__) && !defined(__FreeBSD__) && \
    !defined(__OpenBSD__)
      // lock file
      if (lockf(fileno(fp), F_LOCK, (off_t)1) == -1) {
         fprintf(stderr, "rlibmap: locking failed, don't use gmake -j\n");
      }
#endif

      // remove entries for solib to be processed
      RemoveLib(solib, fullpath, fp);
   }

   LibMap(solib, solibdeps, linkdefs, fullpath, fp);

   if (replace) {
#if !defined(WIN32) && !defined(__CYGWIN__) && !defined(__FreeBSD__) && \
    !defined(__OpenBSD__)
      // remove lock
      lseek(fileno(fp), 0, SEEK_SET);
      if (lockf(fileno(fp), F_ULOCK, (off_t)1) == -1) {
         //fprintf(stderr, "rlibmap: error unlocking output file\n");
      }
#endif
   }

   if (fp != stdout)
      fclose(fp);

   return 0;
}
