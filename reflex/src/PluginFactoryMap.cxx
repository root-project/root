// @(#)root/reflex:$Name:  $:$Id: PluginFactoryMap.cxx,v 1.4 2007/04/03 08:25:40 axel Exp $
// Author: Pere Mato 2006

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
#define REFLEX_BUILD
#endif

#include "PluginFactoryMap.h"

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>

using namespace ROOT::Reflex;
using namespace std;

#include "dir_manip.h"

#if defined(_WIN32)      /* Windows  */
#define PATHENV "PATH"
#define PATHSEP ";"
#elif defined(__APPLE__) /* MacOS */
#define PATHENV "DYLD_LIBRARY_PATH"
#define PATHSEP ":"
#else                    /* Linux */
#define PATHENV "LD_LIBRARY_PATH"
#define PATHSEP ":"
#endif

typedef std::list<std::string> Directive_t;
typedef std::map<std::string, Directive_t> Map_t;

//-------------------------------------------------------------------------------
static Map_t & sMap() {
//-------------------------------------------------------------------------------
// Static wrapper for the map.
   static Map_t s_map;
   return s_map;
}

//-------------------------------------------------------------------------------
static void DumpFactoryDirective(std::ostream& out, 
                                 const Directive_t& directive) {
//-------------------------------------------------------------------------------
// Dump a directive to out.
   bool first = true;
   for (Directive_t::const_iterator iLib = directive.begin();
        iLib != directive.end(); ++iLib) {
      if (!first) 
         out << ", ";
      else first = false;
      out << *iLib;
   }
}

//-------------------------------------------------------------------------------
bool ConflictingDirective(const Directive_t& lhs, const Directive_t& rhs) {
//-------------------------------------------------------------------------------
// Check for inequality of directives, disregarding order of all but first.
   if (*lhs.begin() != *rhs.begin()) return true;
   if (lhs.size() < 2 || lhs.size() < 2) return false; // first entry equal, and it's all we have

   set<string> setLHS, setRHS;
   // can't use insert(iter, iter) because of solaris :-/
   for (Directive_t::const_iterator iLHS = ++lhs.begin(); iLHS != lhs.end(); ++iLHS)
      setLHS.insert(*iLHS);
   for (Directive_t::const_iterator iRHS = ++rhs.begin(); iRHS != rhs.end(); ++iRHS)
      setRHS.insert(*iRHS);
   if (setLHS.size() != setRHS.size()) return true;

   for (set<string>::const_iterator iSetLHS = setLHS.begin();
        iSetLHS != setLHS.end(); ++iSetLHS)
      if (setRHS.find(*iSetLHS) == setRHS.end()) return true;
   return false;
}


int ROOT::Reflex::PluginFactoryMap::fgDebugLevel = 0;

//-------------------------------------------------------------------------------
ROOT::Reflex::PluginFactoryMap::PluginFactoryMap(const std::string& pathenv ) {
//-------------------------------------------------------------------------------
// Constructor.
   vector<char*> tokens;
   struct stat buf;
   dirent* e = 0;
   DIR* dir = 0;
   string path = ::getenv(pathenv.empty() ? PATHENV : pathenv.c_str());
   for(char* t=strtok((char*)path.c_str(),PATHSEP); t; t=strtok(0,PATHSEP))  {
      if ( 0 == ::stat(t,&buf) && S_ISDIR(buf.st_mode) )
         tokens.push_back(t);
   }
   for(vector<char*>::iterator i=tokens.begin();i != tokens.end(); ++i) {
      if ( 0 != (dir=::opendir(*i)) )  {
         while ( 0 != (e=::readdir(dir)) )  {
            if ( strstr(::directoryname(e),"rootmap") != 0 )  {
               std::string fn = *i;
               fn += "/";
               fn += ::directoryname(e);
               FillMap(fn);
            }
         }
         ::closedir(dir);
      }
   }
}


//-------------------------------------------------------------------------------
ROOT::Reflex::PluginFactoryMap::~PluginFactoryMap() {
//-------------------------------------------------------------------------------
// Destructor.
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::PluginFactoryMap::FillMap(const std::string& filename) {
//-------------------------------------------------------------------------------
// Fill the map from the content of the map files.
   fstream file;
   string rawline;
   file.open(filename.c_str(),ios::in);
   if ( Debug() ) cout << "FactoryMap: Processing file " << filename << endl; 
   while( ! getline(file, rawline).eof() && file.good() ) {
      string::size_type p1 = rawline.find_first_not_of(' ');
      string::size_type p2 = rawline.find_last_not_of(' ');
      string line = rawline.substr(p1 == string::npos ? 0 : p1, 
                                   p2 == string::npos ? rawline.length() - 1 : p2 - p1 + 1);
      if ( line.size() == 0 || line[0] == '#' ) continue;
      if ( line.substr(0,8) == "Library." ) {
         string::size_type pc = line.find_first_of(':');
         string cname = line.substr(8,pc-8);
         string::size_type pv = line.substr(pc+1).find_first_not_of(' ');
         string vlibs = line.substr(pc+1+pv);
         Directive_t libs;
         for(char* t=strtok((char*)vlibs.c_str()," "); t; t = strtok(0," "))
            libs.push_back(t);

         // Check whether cname already has a directive,
         // warn and ignore this one if it's conflicting
         Map_t::const_iterator iPreviousDirective = sMap().find(cname);
         if (iPreviousDirective != sMap().end()) {
            if (ConflictingDirective(libs, iPreviousDirective->second)) {
               if (Debug()) {
                  cerr << "ROOT::Reflex::PluginFactoryMap::FillMap() - WARNING: "
                       << "conflicting directives for " << cname << endl
                       << "  Previous: \"";
                  DumpFactoryDirective(cerr, iPreviousDirective->second);
                  cerr << "\"" << endl
                       << "  Directive in " << filename << ": \"";
                  DumpFactoryDirective(cerr, libs);
                  cerr << "\"" << endl
                       << "  Previous takes precedence." << endl;
               }
            } else 
               if ( Debug() > 1 ) {
                  cout << "FactoryMap: copy of directive detected for Name " << cname << ": ";
                  DumpFactoryDirective(cout, libs);
                  cout << endl;
               }
         } else {
            // Inserting name in map 
            sMap()[cname] = libs;

            if ( Debug() > 1 ) {
               cout << "FactoryMap:    Name " << cname << ": ";
               DumpFactoryDirective(cout, libs);
               cout << endl;
            }
         }
      }
   }
   file.close();
}
        

//-------------------------------------------------------------------------------
std::list<std::string> ROOT::Reflex::PluginFactoryMap::GetLibraries(const std::string& name) const {
//-------------------------------------------------------------------------------
// Return all libs currently present.
   return sMap()[name];
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::PluginFactoryMap::SetDebug(int l) {
//-------------------------------------------------------------------------------
// Set debug level.
   fgDebugLevel = l;
}


//-------------------------------------------------------------------------------
int ROOT::Reflex::PluginFactoryMap::Debug() {
//-------------------------------------------------------------------------------
// Get debug level.
   return fgDebugLevel;
}
