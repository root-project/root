#ifndef ROOT_ROOTCoreTeam
#define ROOT_ROOTCoreTeam

namespace ROOT {
namespace ROOTX {

//This string will be updated by external script, reading names from http://root.cern.ch/gitstats/authors.html.
//The names are sorted in alphabetical order.
//The string has an internal linkage (it has a definition here, not in rootxx.cxx or rootx-cocoa.mm files.
//So this header can be included in different places (as soon as you know what you're doing).
//Please, do not modify this file.

const char * gROOTCoreTeam =
//[STRINGTOREPLACE
"Andrei Gheata, Axel Naumann, Bertrand Bellenot, Cristina Cristescu,"
" Danilo Piparo, Fons Rademakers, Gerardo Ganis, Ilka Antcheva,"
" Lorenzo Moneta, Matevz Tadel, Olivier Couet, Paul Russo, Pere Mato,"
" Philippe Canal, Rene Brun, Timur Pocheptsov, Valeri Onuchin,"
" Vassil Vassilev, Wim Lavrijsen, Wouter Verkerke.\n\n";
//STRINGTOREPLACE]

}
}

#endif