// @(#)root/reflex:$Id: Class.cxx 20883 2007-11-19 11:52:08Z rdm $

// Include files----------------------------------------------------------------
#include "Reflex/Reflex.h"
#include "Reflex/PluginService.h"
#include "Reflex/SharedLibrary.h"
#include "../dir_manip.h"
#include <cstdlib>

#include <set>
#include <ctime>
#include <cerrno>
#include <fstream>
#include <iostream>
#include <exception>

using namespace std;
using namespace ROOT::Reflex;

class mapGenerator {
   fstream m_out;
   string m_lib;

public:
   mapGenerator(const string& file, const string& lib):
      m_out(file.c_str(), std::ios_base::out | std::ios_base::trunc),
      m_lib(lib) {}

   bool
   good() const { return m_out.good(); }

   void genHeader();
   void genFactory(const string& name,
                   const string& type);
   void genTrailer();
};

static string
currpath(string lib) {
   char buff[PATH_MAX];
   if (!::getcwd(buff, sizeof(buff))) {
      // coverity[secure_coding] - PATH_MAX is > 2
      strcpy(buff, ".");
   }
   string tmp = buff;
   tmp += "/" + lib;
   return tmp;
}


static int
help(const char* cmd) {
   cout << cmd << ":  Allowed options" << endl
        << "  -help            produce this help message" << endl
        << "  -debug           increase verbosity level" << endl
        << "  -input-library   library to extract the plugins" << endl
        << "  -output-file     output file. default <input-library>.rootmap" << endl;
   return 1;
}


//--- Command main program------------------------------------------------------
int
main(int argc,
     char** argv) {
   struct stat buf;
   bool deb = false;
   string rootmap, lib, err, out;
   string::size_type idx;

   for (int i = 1; i < argc; ++i) {
      const char* opt = argv[i], * val = 0;

      if (*opt == '/' || *opt == '-') {
         if (*++opt == '/' || *opt == '-') {
            ++opt;
         }
         val = (opt[1] != '=') ? ++i < argc ? argv[i] : 0 : opt + 2;

         switch (::toupper(opt[0])) {
         case 'D':
            deb = true;
            --i;
            break;
         case 'I':
            lib = val;
            break;
         case 'O':
            rootmap = val;
            break;
         default:
            return help(argv[0]);
         }
      }
   }

   if (lib.empty()) {
      cout << "ERROR occurred: input library required" << endl;
      return help(argv[0]);
   }

   if (rootmap.empty()) {
      string flib = lib;

      while ((idx = flib.find("\\")) != string::npos)
         flib.replace(idx, 1, "/");
#ifdef _WIN32

      if (flib[1] != ':' && flib[0] != '/') {
#else

      if (!(flib[0] == '/' || flib.find('/') != string::npos)) {
#endif
         flib = currpath(flib);
      }
      rootmap = ::dirnameEx(flib);
      rootmap += "/";
      string tmp = ::basenameEx(lib);

      if (tmp.rfind('.') == std::string::npos) {
         rootmap += tmp + ".rootmap";
      } else if (tmp.find("/") != std::string::npos && tmp.rfind(".") < tmp.rfind("/")) {
         rootmap += tmp + ".rootmap";
      } else {
         rootmap += tmp.substr(0, tmp.rfind('.')) + ".rootmap";
      }
   }

   if (deb) {
      cout << "Input Library: '" << lib << "'" << endl;
      cout << "ROOT Map file: '" << rootmap << "'" << endl;
   }

   if (::stat(rootmap.c_str(), &buf) == -1 && errno == ENOENT) {
      string dir = ::dirnameEx(rootmap);

      if (deb) {
         cout << "Output directory:" << dir << "'" << endl;
      }

      if (!dir.empty()) {
         if (::mkdir(dir.c_str(), S_IRWXU | S_IRGRP | S_IROTH) != 0 && errno != EEXIST) {
            cout << "ERR0R: error creating directory: '" << dir << "'" << endl;
            return 1;
         }
      }
   }
   out = rootmap;

   //--- Load the library -------------------------------------------------
   SharedLibrary sl(lib);

   if (!sl.Load()) {
      cout << "ERR0R: error loading library: '" << lib << "'" << endl
           << sl.Error() << endl;
      return 1;
   }

   mapGenerator map_gen(rootmap.c_str(), lib);

   if (!map_gen.good()) {
      cout << "ERR0R: cannot open output file: '" << rootmap << "'" << endl
           << sl.Error() << endl;
      return 1;
   }
   //--- Iterate over component factories ---------------------------------------
   Scope factories = Scope::ByName(PLUGINSVC_FACTORY_NS);

   if (factories) {
      map_gen.genHeader();
      set<string> used_names;

      try {
         for (Member_Iterator it = factories.FunctionMember_Begin();
              it != factories.FunctionMember_End(); ++it) {
            //string cname = it->Properties().PropertyAsString("name");
            string cname = it->Name();

            if (used_names.insert(cname).second) {
               map_gen.genFactory(cname, "");
            }
         }
         map_gen.genTrailer();
         return 0;
      }
      catch (std::exception& e) {
         cerr << "GENMAP: error creating map " << rootmap << ": "
              << e.what() << endl;
      }
      return 1;
   }
   cout << "library does not contain plugin factories" << endl;
   return 0;
} // main


//------------------------------------------------------------------------------
void
mapGenerator::genHeader() {
//------------------------------------------------------------------------------
   time_t rawtime;
   time(&rawtime);
   m_out << "# ROOT map file generated automatically by genmap on " << ctime(&rawtime) << endl;
}


//------------------------------------------------------------------------------
void
mapGenerator::genFactory(const string& name,
                         const string& /* type */) {
//------------------------------------------------------------------------------
   string mapname = string(PLUGINSVC_FACTORY_NS) + "@@" + PluginService::FactoryName(name);

   //  for ( string::const_iterator i = name.begin(); i != name.end(); i++) {
   //   switch(*i) {
   //     case ':': { newname += '@'; break; }
   //     case ' ': { break; }
   //     default:  { newname += *i; break; }
   //   }
   //  }
   m_out << "Library." << mapname << ": " << m_lib << endl;
}


//------------------------------------------------------------------------------
void
mapGenerator::genTrailer() {
//------------------------------------------------------------------------------
}
