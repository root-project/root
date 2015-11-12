// @(#)root/test:$name:  $:$id: filter.cxx,v 1.0 exp $
// Author: O.Couet

//
//    filters tutorials' files.
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
void   ExecuteCommand(string);
void   ReplaceAll(string&, const string&, const string&);
string StringFormat(const string fmt_str, ...);

// Global variables.
char   gLine[255];   // Current line in the current input file
string gFileName;    // Input file name
string gLineString;  // Current line (as a string) in the current input file
string gImageName;   // Current image name
string gMacroName;   // Current macro name
string gCwd;         // Current working directory
string gOutDir;      // Output directory
int    gShowSource;  // True if the source code should be shown

////////////////////////////////////////////////////////////////////////////////
/// Filter ROOT tutorials for Doxygen.

int main(int argc, char *argv[])
{
   // Initialisation

   gFileName   = argv[1];
   gShowSource = 0;

   // Retrieve the output directory
   gOutDir = getenv("TUTORIALS_OUTPUT_DIRECTORY");
   ReplaceAll(gOutDir,"\"","");

   // Open the input file name.
   FILE *f = fopen(gFileName.c_str(),"r");

   // File for inline macros.
   FILE *m = 0;

   // Extract the macro name
   int i1     = gFileName.rfind('/')+1;
   int i2     = gFileName.rfind('C');
   gMacroName = gFileName.substr(i1,i2-i1+1);

   // Parse the source and generate the image if needed
   while (fgets(gLine,255,f)) {
      gLineString = gLine;

      // \macro_image found
      if (gLineString.find("\\macro_image") != string::npos) {
         gImageName = StringFormat("%s.png", gMacroName.c_str()); // Image name
         ExecuteCommand(StringFormat("root -l -b -q \"makeimage.cxx(\\\"%s\\\",\\\"%s\\\",\\\"%s\\\")\"",
                                        gFileName.c_str(), gImageName.c_str(), gOutDir.c_str()));
         ReplaceAll(gLineString, "\\macro_image",
                                 StringFormat("\\image html %s",gImageName.c_str()));

      }

      // \macro_code found
      if (gLineString.find("\\macro_code") != string::npos) {
         gShowSource = 1;
         m = fopen(StringFormat("%s/html/%s",gOutDir.c_str(),gMacroName.c_str()).c_str(), "w");
         ReplaceAll(gLineString, "\\macro_code",
                                 StringFormat("\\include %s",gMacroName.c_str()));
      }

      // \author is the last comment line.
      if (gLineString.find("\\author")  != string::npos) {
          printf("%s",StringFormat("%s \n/// \\cond \n",gLineString.c_str()).c_str());
         if (gShowSource == 1) gShowSource = 2;
      } else {
         printf("%s",gLineString.c_str());
         if (m && gShowSource == 2) fprintf(m,"%s",gLineString.c_str());
      }
   }
   if (m) fclose(m);
   fclose(f);
   return 0;
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