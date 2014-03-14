#ifndef ROOT_ROOTCoreTeam
#define ROOT_ROOTCoreTeam

namespace ROOT {
namespace ROOTX {

//This array will be updated by external script, reading names from http://root.cern.ch/gitstats/authors.html.
//The names are sorted in an alphabetical order.
//Please note the structure: it should always like this - names as
//string literals in an array's initializer
//with a terminating 0 - that's what our rootxx.cxx and rootx-cocoa.mm expect!!!
//The array (or its elements actually) has an internal linkage
//(it has a definition here, not in rootxx.cxx or rootx-cocoa.mm files.
//So this header can be included in different places (as soon as you know what you're doing).
//Please, do not modify this file.

const char * gROOTCoreTeam[] = {
//[STRINGTOREPLACE
"Ilka Antcheva",
"Bertrand Bellenot",
"Rene Brun",
"Philippe Canal",
"Olivier Couet",
"Cristina Cristescu",
"Gerardo Ganis",
"Andrei Gheata",
"Wim Lavrijsen",
"Pere Mato",
"Lorenzo Moneta",
"Axel Naumann",
"Valeri Onuchin",
"Danilo Piparo",
"Timur Pocheptsov",
"Fons Rademakers",
"Paul Russo",
"Matevz Tadel",
"Vassil Vassilev",
"Wouter Verkerke",
0};
//STRINGTOREPLACE]
}
}

#endif
