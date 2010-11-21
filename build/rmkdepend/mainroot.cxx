// @(#)root/build:$Id$
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
#include "../../core/utils/src/cygpath.h"
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
      if (fwrite(")\n", 2, 1, stdout) != 1) // closing "$(wildcard"
         fprintf(stderr, "Warning: ROOT_flush: fwrite error\n");
      openWildcard = 0;
   }
   /* now done via "XYZ.d XYZ.o: $(wildcard dependencies)"
   if (!currentFileBase.empty()) {
      currentFileBase += "o";
      fwrite(currentFileBase.c_str(), currentFileBase.length(), 1, stdout);
      currentDependencies += '\n';
      fwrite(currentDependencies.c_str(), currentDependencies.length(), 1, stdout);
   }
   */
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
      if (fwrite(buf, len, 1, stdout) != 1)
         fprintf(stderr, "Warning: ROOT_adddep: fwrite error\n");
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
      char s = posColon[4]; // save char that will be overwritten by \0 of "cxx"
      strcat(posColon, "cxx");
      if (fwrite(buf, (posColon - buf)+4, 1, stdout) != 1) // .cxx
         fprintf(stderr, "Warning: ROOT_adddep: fwrite error\n");
      posColon[4] = s;
   }

   posColon[1]='o';
   if (fwrite(buf, (posColon - buf)+2, 1, stdout) != 1) // .o
      fprintf(stderr, "Warning: ROOT_adddep: fwrite error\n");
   posColon[1]='d';
   if (fwrite(buf, (posColon - buf)+2, 1, stdout) != 1) // .d
      fprintf(stderr, "Warning: ROOT_adddep: fwrite error\n");

   if (!isDict) {
      posColon[1] = 0;
      currentFileBase = buf + 1;
      currentDependencies = posColon + 2;
   }
   if (fwrite(": $(wildcard ", 13, 1, stdout) != 1)
      fprintf(stderr, "Warning: ROOT_adddep: fwrite error\n");
   if (fwrite(posColon + 4, len - (posColon + 4 - buf), 1, stdout) != 1)
      fprintf(stderr, "Warning: ROOT_adddep: fwrite error\n");
   openWildcard = 1;
}

int main(int argc, char **argv)
{
   isDict = false;
   if (argc<3 || (strcmp(argv[1], "-R") && strncmp(argv[1], "-R=", 3)))
      return main_orig(argc, argv);

   rootBuild = 1;
   const char* outname = argv[2]+2;
   while (*outname == ' ') ++outname;
   if (*outname) {
      if (argv[1][2] == '=') {
         // dictionary tag passed after -R=
         std::string sDictTag(argv[1] + 1);
         size_t posExt = sDictTag.find('%');
         if (posExt != std::string::npos && posExt < sDictTag.length() - 1) {
            std::string sDictExt = sDictTag.substr(posExt + 1);
            sDictTag.erase(posExt);
            isDict = (strstr(outname, sDictTag.c_str()))
               && !(strcmp(outname + strlen(outname) - sDictExt.length(),
                           sDictExt.c_str()));
         } else {
            isDict = (strstr(outname, sDictTag.c_str()) != 0);
         }
      } else {
         // no = after "-R", thus "/G__%.d";
         isDict = (strstr(outname, "/G__"))
            && (!strcmp(outname + strlen(outname) - 2, ".d"));
      }
   }

   argv[1] = argv[0]; // keep program name

#ifdef _WIN32
   for (int i = 2; i < argc; ++i) {
      std::string arg(argv[i]);
      if (FromCygToNativePath(arg)) {
         size_t len = arg.length();
         // yes, we leak.
         char* argvi = new char[len + 1];
         strncpy(argvi, arg.c_str(), len + 1);
         argv[i] = argvi;
      }
   }
#endif

   int ret = main_orig(argc-1, &argv[1]);
   if (ret) {
      // delete output file
      unlink(outname);
   } else
      ROOT_flush();
   return ret;
}
