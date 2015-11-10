// @(#)root/test:$name:  $:$id: filter.cxx,v 1.0 exp $
// Author: O.Couet

//
//    filters doc files.
//

#include <unistd.h>
#include <stdio.h>
#include <string>
#include <cstring>
#include <iostream>
#include <stdlib.h>
#include <stdarg.h>
#include <memory>

using namespace std;

// Auxiliary functions
void   GetClassName();
void   ExecuteMacro();
void   ExecuteCommand(string);
void   ReplaceAll(string&, const string&, const string&);
string StringFormat(const string fmt_str, ...);
bool   EndsWith(string const &, string const &);
bool   BeginsWith(const string&, const string&);

// Global variables.
char   gLine[255];   // Current line in the current input file
string gFileName;    // Input file name
string gLineString;  // Current line (as a string) in the current input file
string gClassName;   // Current class name
string gImageName;   // Current image name
string gMacroName;   // Current macro name
string gCwd;         // Current working directory
string gOutDir;      // Output directory
string gSourceDir;   // Source directory
bool   gHeader;      // True if the input file is a header
bool   gSource;      // True if the input file is a source file
bool   gImageSource; // True the source of the current macro should be shown
bool   gInClassDef;
bool   gClass;
int    gInMacro;
int    gImageID;
int    gMacroID;

////////////////////////////////////////////////////////////////////////////////
/// Filter ROOT files for Doxygen.

int main(int argc, char *argv[])
{
   // Initialisation

   gFileName    = argv[1];
   gHeader      = false;
   gSource      = false;
   gInClassDef  = false;
   gClass       = false;
   gImageSource = false;
   gInMacro     = 0;
   gImageID     = 0;
   gMacroID     = 0;
   if (EndsWith(gFileName,".cxx")) gSource = true;
   if (EndsWith(gFileName,".h"))   gHeader = true;
   GetClassName();

   // Retrieve the current working directory
   int last = gFileName.rfind("/");
   gCwd     = gFileName.substr(0,last);

   // Retrieve the output directory
   gOutDir = getenv("DOXYGEN_OUTPUT_DIRECTORY");
   ReplaceAll(gOutDir,"\"","");

   // Retrieve the source directory
   gSourceDir = getenv("DOXYGEN_SOURCE_DIRECTORY");
   ReplaceAll(gSourceDir,"\"","");

   // Open the input file name.
   FILE *f = fopen(gFileName.c_str(),"r");

   // File for inline macros.
   FILE *m = 0;

   // File header.
   if (gHeader) {
      while (fgets(gLine,255,f)) {
         gLineString = gLine;

         if (BeginsWith(gLineString,"class"))              gInClassDef = true;
         if (gLineString.find("ClassDef") != string::npos) gInClassDef = false;

         printf("%s",gLineString.c_str());
      }
      fclose(f);
      return 0;
   }

   // Source file.
   if (gSource) {
      while (fgets(gLine,255,f)) {
         gLineString = gLine;

         if (gLineString.find("/*! \\class")  != string::npos ||
             gLineString.find("/// \\class")  != string::npos ||
             gLineString.find("/** \\class")  != string::npos ||
             gLineString.find("///! \\class") != string::npos) gClass = true;

         if (gLineString.find("End_Macro") != string::npos) {
            ReplaceAll(gLineString,"End_Macro","");
            gImageSource = false;
            gInMacro = 0;
            if (m) {
               fclose(m);
               m = 0;
               ExecuteCommand(StringFormat("root -l -b -q \"makeimage.C(\\\"%s\\\",\\\"%s\\\",\\\"%s\\\")\""
                                              , StringFormat("%s_%3.3d.C", gClassName.c_str(), gMacroID).c_str()
                                              , StringFormat("%s_%3.3d.png", gClassName.c_str(), gImageID).c_str()
                                              , gOutDir.c_str()));
               ExecuteCommand(StringFormat("rm %s_%3.3d.C", gClassName.c_str(), gMacroID));
            }
         }

         if (gInMacro) {
            if (gInMacro == 1) {
               if (EndsWith(gLineString,".C\n")) {
                  ExecuteMacro();
                  gInMacro++;
               } else {
                  gMacroID++;
                  m = fopen(StringFormat("%s_%3.3d.C", gClassName.c_str(), gMacroID).c_str(), "w");
                  if (m) fprintf(m,"%s",gLineString.c_str());
                  if (BeginsWith(gLineString,"{")) {
                     if (gImageSource) {
                        ReplaceAll(gLineString,"{"
                                                , StringFormat("\\include %s_%3.3d.C"
                                                , gClassName.c_str()
                                                , gMacroID));
                     } else {
                        gLineString = "\n";
                     }
                  }
                  gInMacro++;
               }
            } else {
               if (m) fprintf(m,"%s",gLineString.c_str());
               if (BeginsWith(gLineString,"}")) {
                  ReplaceAll(gLineString,"}", StringFormat("\\image html %s_%3.3d.png", gClassName.c_str(), gImageID));
               } else {
                  gLineString = "\n";
               }
               gInMacro++;
            }
         }

         if (gLineString.find("Begin_Macro") != string::npos) {
            if (gLineString.find("source") != string::npos) gImageSource = true;
            gImageID++;
            gInMacro++;
            gLineString = "\n";
         }

         printf("%s",gLineString.c_str());
      }
      fclose(f);
      return 0;
   }

   // Output anything not header nor source
   while (fgets(gLine,255,f)) {
      gLineString = gLine;
      printf("%s",gLineString.c_str());
   }
   fclose(f);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve the class name.

void GetClassName()
{
   int i1 = 0;
   int i2 = 0;

   FILE *f = fopen(gFileName.c_str(),"r");

   // File header.
   if (gHeader) {
      while (fgets(gLine,255,f)) {
         gLineString = gLine;
         if (gLineString.find("ClassDef") != string::npos) {
            i1         = gLineString.find("(")+1;
            i2         = gLineString.find(",")-1;
            gClassName = gLineString.substr(i1,i2-i1+1);
            fclose(f);
            return;
         }
      }
   }

   // Source file.
   if (gSource) {
      while (fgets(gLine,255,f)) {
         gLineString = gLine;
         if (gLineString.find("ClassImp") != string::npos) {
            i1         = gLineString.find("(")+1;
            i2         = gLineString.find(")")-1;
            gClassName = gLineString.substr(i1,i2-i1+1);
            fclose(f);
            return;
         }
      }
   }

   fclose(f);

   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Execute the macro in gLineString and produce the corresponding picture

void ExecuteMacro()
{
   // Name of the next Image to be generated
   gImageName = StringFormat("%s_%3.3d.png", gClassName.c_str(), gImageID);

   // Retrieve the macro to be executed.
   if (gLineString.find("../../..") != string::npos) {
      ReplaceAll(gLineString,"../../..", gSourceDir.c_str());
   } else {
      gLineString.insert(0, StringFormat("%s/../doc/macros/",gCwd.c_str()));
   }
   int i1     = gLineString.rfind('/')+1;
   int i2     = gLineString.rfind('C');
   gMacroName = gLineString.substr(i1,i2-i1+1);

   // Build the ROOT command to be executed.
   gLineString.insert(0, StringFormat("root -l -b -q \"makeimage.C(\\\""));
   int l = gLineString.length();
   gLineString.replace(l-2,1,StringFormat("C\\\",\\\"%s\\\",\\\"%s\\\")\"", gImageName.c_str(), gOutDir.c_str()));

   ExecuteCommand(gLineString);

   if (gImageSource) {
      gLineString = StringFormat("\\include %s\n\\image html %s\n", gMacroName.c_str(), gImageName.c_str());
   } else {
      gLineString = StringFormat("\n\\image html %s\n", gImageName.c_str());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Execute a command making sure stdout will not go in the doxygen file.

void ExecuteCommand(string command)
{
   int o = dup(fileno(stdout));
   freopen("stdout.dat","a",stdout);
   system(command.c_str());
   dup2(o,fileno(stdout));
   close(o);
}

////////////////////////////////////////////////////////////////////////////////
/// Replace all instances of of a string with another string.

void ReplaceAll(string& str, const string& from, const string& to) {
   if (from.empty()) return;
   string wsRet;
   wsRet.reserve(str.length());
   size_t start_pos = 0, pos;
   while ((pos = str.find(from, start_pos)) != string::npos) {
      wsRet += str.substr(start_pos, pos - start_pos);
      wsRet += to;
      pos += from.length();
      start_pos = pos;
   }
   wsRet += str.substr(start_pos);
   str.swap(wsRet);
}

////////////////////////////////////////////////////////////////////////////////
/// std::string formatting like sprintf

string StringFormat(const string fmt_str, ...) {
   int final_n, n = ((int)fmt_str.size()) * 2; /* Reserve two times as much as the length of the fmt_str */
   string str;
   unique_ptr<char[]> formatted;
   va_list ap;
   while (1) {
      formatted.reset(new char[n]); /* Wrap the plain char array into the unique_ptr */
      strcpy(&formatted[0], fmt_str.c_str());
      va_start(ap, fmt_str);
      final_n = vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
      va_end(ap);
      if (final_n < 0 || final_n >= n) n += abs(final_n - n + 1);
      else break;
   }
   return string(formatted.get());
}

////////////////////////////////////////////////////////////////////////////////
/// find if a string ends with another string

bool EndsWith(string const &fullString, string const &ending) {
   if (fullString.length() >= ending.length()) {
      return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
   } else {
      return false;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// find if a string begins with another string

bool BeginsWith(const string& haystack, const string& needle) {
   return needle.length() <= haystack.length() && equal(needle.begin(), needle.end(), haystack.begin());
}
