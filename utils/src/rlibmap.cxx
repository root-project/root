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
// -o: write output to specified file, otherwise to stdout
// -r: replace existing entries in the specified file


#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string>
#include <map>
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
#if (defined(__FreeBSD__) && (__FreeBSD__ < 4)) || \
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

const char *usage = "Usage: %s [-f] [<-r|-o> <libmapfile>] <sofile> <sofile> ...\n";

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


#ifdef WIN32
#include <windows.h>

#define ftruncate(fd, size)  win32_ftruncate(fd, size)

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

int removelibs(char **libs, bool fullpath, FILE *fp)
{
   // Remove entries from the map file for the specified libs.

   fseek(fp, 0, SEEK_SET);

   // get file size
   struct stat sbuf;
   fstat(fileno(fp), &sbuf);
   size_t siz = sbuf.st_size;

   // read file and remove lines matching specified libs
   char *fbuf = new char[siz+1];
   char *fptr = fbuf;

   while (fgets(fptr, siz - size_t(fptr-fbuf), fp)) {
      int i = 0;
      while (libs[i]) {
         const char *libbase = libs[i];
         if (!fullpath) {
            if ((libbase = strrchr(libs[i], '/')))
               libbase++;
            else
               libbase = libs[i];
         }

         if (!strstr(fptr, libbase)) {
            fptr += strlen(fptr);
            if (*(fptr-1) != '\n') {
               *fptr = '\n';
               fptr++;
            }
         }

         i++;
      }
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

int libmap(const char *lib, bool fullpath, FILE *fp)
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
               string cls;
               // check for nested class or class in namespace or both
               char *r1;
               if ((r1 = strrchr(symbol, ':')) && *(r1-1) == ':')
                  cls = r1+1;
               else
                  cls = symbol;
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

   const char *libbase = lib;
   if (!fullpath) {
      if ((libbase = strrchr(lib, '/')))
         libbase++;
      else
         libbase = lib;
   }

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
   bool fullpath = false;
   bool replace  = false;
   FILE *fp     = stdout;

   if (argc > 1) {
      int ic = 1;
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

   if (replace) {
#if !defined(WIN32) && !defined(__CYGWIN__)
      // lock file
      if (lockf(fileno(fp), F_LOCK, (off_t)1) == -1) {
         fprintf(stderr, "rlibmap: error locking output file\n");
         fclose(fp);
         return 1;
      }
#endif

      // remove entries for libs to be processed
      removelibs(libs, fullpath, fp);
   }

   int i = 0;
   while (libs[i]) {
      libmap(libs[i], fullpath, fp);
      i++;
   }

   if (replace) {
#if !defined(WIN32) && !defined(__CYGWIN__)
      // remove lock
      lseek(fileno(fp), 0, SEEK_SET);
      if (lockf(fileno(fp), F_ULOCK, (off_t)1) == -1) {
         fprintf(stderr, "rlibmap: error unlocking output file\n");
         fclose(fp);
         return 1;
      }
#endif
   }

   if (fp != stdout)
      fclose(fp);

   return 0;
}
