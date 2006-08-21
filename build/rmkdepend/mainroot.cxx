// @(#)root/build:$Name:  $:$Id: mainroot.cxx,v 1.4 2006/07/30 11:22:59 rdm Exp $
// Author: Axel Naumann   21/03/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <string>

extern "C" {
#if defined(__sun) && defined(__SUNPRO_CC)
#include <signal.h>
#endif
#include "def.h"
}

#ifndef WIN32
#include <unistd.h>
#else
extern "C" int unlink(const char *FILENAME);
#endif

extern "C" int main_orig(int argc, char **argv);


int rootBuild = 0;

int isDict = 0;
int newFile = 0;
std::string currentDependencies;
std::string currentFileBase;

extern "C"
void ROOT_newFile()
{
   newFile = 1;
}

void ROOT_flush()
{
   if (!currentFileBase.empty()) {
      fwrite(")\n", 2, 1, stdout); // closing "$(wildcard"
      bool haveOldNonDict = !isDict;
      if (haveOldNonDict) {
         currentFileBase += "o";
         fwrite(currentFileBase.c_str(), currentFileBase.length(), 1, stdout);
         currentDependencies += '\n';
         fwrite(currentDependencies.c_str(), currentDependencies.length(), 1, stdout);
      }
   }
   currentFileBase.clear();
   currentDependencies.clear();
}

extern "C"
void ROOT_adddep(char* buf, size_t len)
{
   char* posColon = 0;
   if (newFile)
      posColon = strstr(buf, ".o: ");

   if (!posColon) {
      fwrite(buf, len, 1, stdout);
      currentDependencies += buf;
      return;
   }

/* isDict:
   sed -e 's@^\(.*\)\.o[ :]*\(.*\)\@
             \1.d: $\(wildcard \2\)\@\1.cxx: $\(wildcard \2\)@'
       -e 's@^#.*$@@'
       -e '/^$/d'
   | tr '@' '\n'
else
   sed -e 's@^\(.*\)\.o[ :]*\(.*\)@
             \1.d: $\(wildcard \2\)\@\1.o: \2@'
       -e 's@^#.*$@@'
       -e '/^$/d' $1.tmp
   | tr '@' '\n'
*/
   // flush out the old dependencies
   ROOT_flush();

   newFile = 0;

   buf[0] = ' ';
   if (isDict) {
      posColon[1]=0;
      strcat(posColon, "cxx");
      fwrite(buf, (posColon - buf)+4, 1, stdout); // .cxx
   }

   posColon[1]='d';
   fwrite(buf, (posColon - buf)+2, 1, stdout); // .d

   if (!isDict) {
      posColon[1] = 0;
      currentFileBase = buf + 1;
      currentDependencies = posColon + 2;
   }
   fwrite(": $(wildcard ", 13, 1, stdout);
   fwrite(posColon + 4, len - (posColon + 4 - buf), 1, stdout);
}

int main(int argc, char **argv)
{
   if (argc<3 || strcmp(argv[1], "-R"))
      return main_orig(argc, argv);

   rootBuild = 1;
   int skip = 2;
   const char* outname = argv[2]+skip;
   while (outname[0] == ' ') outname = argv[2] + (++skip);
   if (outname)
      isDict = (strstr(outname, "/G__") != 0 && strstr(outname, ".cxx"));

   argv[1] = argv[0]; // keep program name
   int ret = main_orig(argc-1, &argv[1]);
   if (ret) {
      // delete output file
      unlink(outname);
   } else
      ROOT_flush();
   return ret;
}
