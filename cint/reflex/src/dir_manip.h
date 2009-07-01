// @(#)root/reflex:$Id$
// Author: Pere Mato 2006

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.


#ifndef Reflex_dir_manip
#define Reflex_dir_manip

#include <climits>
#include <sys/types.h>
#include <sys/stat.h>

#ifdef _WIN32     /* Windows  */
# include "io.h"
# include "direct.h"
# include "errno.h"
# define S_IRWXU _S_IREAD | _S_IWRITE
# define S_IRGRP 0
# define S_IROTH 0

typedef _finddata_t dirent;
inline const char*
directoryname(const dirent* e) { return e->name; }

inline int
mkdir(const char* d,
      int) { return mkdir(d); }

# define S_ISDIR(x) ((x & _S_IFDIR) == _S_IFDIR)
# define PATH_MAX _MAX_PATH

struct DIR {
   bool start;
   long handle;
   _finddata_t data;
};
typedef _finddata_t dirent;
inline DIR*
opendir(const char* specs) {
   struct stat buf;

   if (stat(specs, &buf) < 0) {
      return 0;
   }

   if (!S_ISDIR(buf.st_mode)) {
      return 0;
   }
   std::string path = specs;
   path += "/*";
   std::auto_ptr<DIR> dir(new DIR);
   dir->start = true;
   dir->handle = _findfirst(path.c_str(), &dir->data);

   if (dir->handle != -1) {
      return dir.release();
   }
   return 0;
} // opendir


inline dirent*
readdir(DIR* dir) {
   if (dir) {
      if (dir->start) {
         dir->start = false;
         return &dir->data;
      } else if (_findnext(dir->handle, &dir->data) == 0) {
         return &dir->data;
      }
   }
   return 0;
}


inline int
closedir(DIR* dir) {
   if (dir) {
      int sc = _findclose(dir->handle);
      delete dir;
      return sc;
   }
   return EBADF;
}


inline const char*
dirnameEx(const std::string& path) {
   static std::string p;
   std::string tmp = path;
   std::string::size_type idx = tmp.rfind("/");

   if (idx != std::string::npos) {
      p = tmp.substr(0, idx);
      return p.c_str();
   }

   if ((idx = tmp.rfind("\\")) != std::string::npos) {
      p = tmp.substr(0, idx);
      return p.c_str();
   }
   p = "";
   return p.c_str();
} // dirnameEx


inline const char*
basenameEx(const std::string& path) {
   static std::string p;
   std::string tmp = path;

   if (tmp.rfind("/") != std::string::npos) {
      p = tmp.substr(tmp.rfind("/") + 1);
      return p.c_str();
   }
   p = tmp;
   return p.c_str();
}


#else // _WIN32

# include <unistd.h>
# include <dirent.h>
# include <libgen.h>
# include <limits.h> /* open solaris */

inline const char*
directoryname(const dirent* e) { return e->d_name; }

inline const char*
dirnameEx(const std::string& path) {
   static std::string p, q;
   p = path.c_str();
   q = dirname(const_cast<char*>(p.c_str()));
   return q.c_str();
}


inline const char*
basenameEx(const std::string& path) {
   static std::string p, q;
   p = path.c_str();
   q = basename(const_cast<char*>(p.c_str()));
   return q.c_str();
}


#endif  // _WIN32

#endif // Reflex_dir_manip
