/* @(#)build/win:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2010, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_CygPath
#include <stdio.h>
#include <stdlib.h>
#include <direct.h>
#include <string>

static const char *GetCygwinRootDir() {
   // Get the install location of cygwin.
   static char buf[512] = {0};

   if (!buf[0]) {
      char pathbuffer[_MAX_PATH] = {0};
      // Search for cygpath in PATH environment variable
      _searchenv( "cygpath.exe", "PATH", pathbuffer );
      if( *pathbuffer == '\0' ) {
         sprintf(buf, "%c:", _getdrive());
         return buf;
      }
      FILE *pipe = _popen( "cygpath -m /", "rt" );

      if (!pipe) return 0;
      fgets(buf, sizeof(buf), pipe);
      int len = strlen(buf);
      while (buf[len - 1] == '\n' || buf[len - 1] == '\r') {
         buf[len - 1] = 0;
      }
      if (!feof(pipe)) _pclose(pipe);
      else fprintf(stderr, "GetCygwinRootDir() error: Failed to read the pipe to the end.\n");
   }
   return buf;
}

static bool FromCygToNativePath(std::string& path) {
   // Convert a cygwin path (/cygdrive/x/...,/home)
   // to a native Windows path. Return whether the path was changed.
   static std::string cygRoot;
   size_t posCygDrive = path.find("/cygdrive/");
   if (posCygDrive != std::string::npos) {
      path[posCygDrive] = path[posCygDrive + 10];
      path[posCygDrive + 1] = ':';
      path.erase(posCygDrive + 2, 9);
      return true;
   } else {
      size_t posHome = path.find("/home/");
      if (posHome != std::string::npos) {
         size_t posColumn = path.find(":");
         if (posColumn != std::string::npos && posColumn > 0) {
            // Don't convert C:/home or even C:/cygwin/home
            if (path[posColumn - 1] >= 'A' && path[posColumn - 1] <= 'Z')
               return false;
            if (path[posColumn - 1] >= 'a' && path[posColumn - 1] <= 'z')
               return false;
         }
         if (cygRoot.empty()) {
            cygRoot = GetCygwinRootDir();
            size_t len = cygRoot.length();
            if (cygRoot[len - 1] == '/') {
               cygRoot.erase(len - 1);
            }
         }
         path.insert(posHome, cygRoot);
         return true;
      }
   }
   return false;
}

#endif // ROOT_CygPath
