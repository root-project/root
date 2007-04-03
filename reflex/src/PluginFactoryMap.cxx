// @(#)root/reflex:$Name:  $:$Id: PluginFactoryMap.cxx,v 1.3 2006/12/02 09:07:04 brun Exp $
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

typedef std::map<std::string,std::list<std::string> > Map_t;

//-------------------------------------------------------------------------------
static Map_t & sMap() {
//-------------------------------------------------------------------------------
// Static wrapper for the map.
   static Map_t s_map;
   return s_map;
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
         list<string> libs;
         for(char* t=strtok((char*)vlibs.c_str()," "); t; t = strtok(0," "))
            libs.push_back(t);
         // Inserting name in map 
         sMap()[cname] = libs;
         
         if ( Debug() > 1 ) {
            cout << "FactoryMap:    Name " << cname << ": ";
            for ( list<string>::iterator i = libs.begin(); i != libs.end(); i++ )
               cout << *i << " ";
            cout << endl;
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
