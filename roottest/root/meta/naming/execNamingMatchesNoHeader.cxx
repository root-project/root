#include "execNamingMatches.cxx"

int execNamingMatchesNoHeader() {
   // Make the loading of the header file impossible,
   // even when autoparsing is triggered.
   gROOT->ProcessLine("#define namingMatches_cxx");

   return execNamingMatches();
}
