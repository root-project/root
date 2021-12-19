/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN  01/2022
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooFit/Common.h>

#include <TSystem.h>

namespace RooFit {

////////////////////////////////////////////////////////////////////////////////
/// Returns the path to the directory that should be used for temporary RooFit
/// files (e.g. for testing). The returned path includes the trailing
/// backslash. The first time this function is called, it will check if the
/// directory exists and create it if it doesn't.
std::string const &tmpPath()
{
   static const std::string dir{"/tmp/roofit/"};

   // The first time this funciton is used, we will attempt to create the
   // directory if it doesn't exist yet.
   static bool isFirstCall = true;
   if (isFirstCall) {
      gSystem->Exec((std::string("mkdir -p ") + dir).c_str());
      isFirstCall = false;
   }

   return dir;
}

} // namespace RooFit
