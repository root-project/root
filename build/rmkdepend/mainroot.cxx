// @(#)root/build:$Name:  $:$Id: mainroot.cxx,v 1.6 2006/08/21 12:31:37 rdm Exp $
// Author: Axel Naumann   21/03/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// ROOT wrapper around ROOT's mkdepend incarnation + wrapper script,
// known as depends.sh in earlier days.
// If the first(!) argument is '-R' it triggers a few special
// routines:
//  * dependencies for .d files use $(wildcard ...), so gmake doesn't
//    bail out if one of the dependencies doesn't exist.
//  * output files starting with '/G__' and ending on ".d" are assumed to
//    be dictionaries. rmkdepend generates rules for these dictionaries 
//    covering the .d file, and the .cxx file itself, 
//    so the dictionaries get re-egenerated when an included header 
//    file gets changed.
//  * the detection / wildcarding of a dictionary file can be changed 
//    by specifying -R=[tag]%[ext] as parameter to -R. The default is 
//    "-R=/G__%.d".
//  * remove output file if we encounter an error.

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

static int isDict = 0;
static int newFile = 0;
static int openWildcard = 0;
static std::string currentDependencies;
static std::string currentFileBase;

extern "C"
void ROOT_newFile()
{
   newFile = 1;
}

void ROOT_flush()
{
   if (openWildcard) {
      fwrite(")\n", 2, 1, stdout); // closing "$(wildcard"
      openWildcard = 0;
   }
   if (!currentFileBase.empty()) {
      currentFileBase += "o";
      fwrite(currentFileBase.c_str(), currentFileBase.length(), 1, stdout);
      currentDependencies += '\n';
      fwrite(currentDependencies.c_str(), currentDependencies.length(), 1, stdout);
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
      char s = posColon[4]; // sove char that will be overwritten by \0 of "cxx"
      strcat(posColon, "cxx");
      fwrite(buf, (posColon - buf)+4, 1, stdout); // .cxx
      posColon[4] = s;
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
   openWildcard = 1;
}

int main(int argc, char **argv)
{
   if (argc<3 || (strcmp(argv[1], "-R") && strncmp(argv[1], "-R=", 3)))
      return main_orig(argc, argv);

   rootBuild = 1;
   int skip = 2;
   const char* outname = argv[2]+skip;
   while (outname[0] == ' ') outname = argv[2] + (++skip);
   if (outname) {
      const char* dicttag = "/G__%.d";
      if (argv[1][2] == '=')
         dicttag = argv[1] + 3;
      std::string sDictTag(dicttag);
      std::string sDictExt;
      size_t posExt = sDictTag.find('%');
      if (posExt != std::string::npos) {
         sDictExt = sDictTag.substr(posExt + 1);
         sDictTag.erase(posExt);
      }
      isDict = (strstr(outname, sDictTag.c_str()) != 0 
         && (!sDictExt.length() || strstr(outname, sDictExt.c_str())));
   }

   argv[1] = argv[0]; // keep program name
   int ret = main_orig(argc-1, &argv[1]);
   if (ret) {
      // delete output file
      unlink(outname);
   } else
      ROOT_flush();
   return ret;
}
