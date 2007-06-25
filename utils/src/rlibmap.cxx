// @(#)root/utils:$Name:  $:$Id: rlibmap.cxx,v 1.22 2007/04/19 12:57:15 rdm Exp $
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
#include <string>
#include <vector>
#ifndef WIN32
#   include <unistd.h>
#else
#   define ssize_t int
#   include <io.h>
#   include <sys/types.h>
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

   if (!str) return 0;

   const char *p = str;
   // allocate 20 extra characters in case of eg, vector<vector<T>>
   char *s, *s1 = new char[strlen(str)+20];
   s = s1;

   while (*p) {
      if (*p != ' ')
         *s++ = *p;
      p++;
   }
   *s = '\0';

   return s1;
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

   ftruncate(fileno(fp), 0);

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
            if (!strcmp(strtok(pragma, " "), "#pragma") &&
                !strcmp(strtok(0,      " "), "link")    &&
                !strcmp(strtok(0,      " "), "C++")) {
               char *type = strtok(0, " ");
               if (!strncmp(type, "option=", 7) || !strncmp(type, "options=", 8)) {
                  if (strstr(type, "nomap"))
                     continue;
                  type = strtok(0, " ");
               }
               //the following statement had to be commented. Currently CINT
               //cannot autoload when executing a typedef (must be fixed!!!)
               //if (!strcmp(type, "class") || !strcmp(type, "typedef")) {
               if (!strcmp(type, "class")) {
                  char *cls = strtok(0, "-!+;");
                  // just in case remove trailing space and tab
                  while (*cls == ' ') cls++;
                  int len = strlen(cls) - 1;
                  while (cls[len] == ' ' || cls[len] == '\t')
                     cls[len--] = '\0';
                  //no space between tmpl arguments allowed
                  cls = Compress(cls);

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
   }

   vector<string>::const_iterator it;
   for (it = classes.begin(); it != classes.end(); it++) {
      fprintf(fp, "Library.%-35s %s", ((*it)+":").c_str(), libbase);

      if (solibdeps.size() > 0) {
         vector<string>::const_iterator depit;
         for (depit = solibdeps.begin(); depit != solibdeps.end(); depit++) {
            const char *deplib = depit->c_str();
            if (!fullpath) {
               if ((deplib = strrchr(deplib, '/')))
                  deplib++;
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
         fp = fopen(argv[ic], "w");
         if (!fp) {
            fprintf(stderr, "cannot open output file %s\n", argv[ic]);
            return 1;
         }
         ic++;
      }
      if (!strcmp(argv[ic], "-r")) {
         replace = true;
         ic++;
         fp = fopen(argv[ic], "a+");
         if (!fp) {
            fprintf(stderr, "cannot open output file %s\n", argv[ic]);
            return 1;
         }
         ic++;
      }
      if (!strcmp(argv[ic], "-l")) {
         ic++;
         solib = argv[ic];
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
            linkdefs.push_back(argv[i]);
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
