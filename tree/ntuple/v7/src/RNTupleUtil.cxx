/// \file RNTupleUtil.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch> & Max Orok <maxwellorok@gmail.com>
/// \date 2020-07-14
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RNTupleUtil.hxx"

#include "ROOT/RLogger.hxx"
#include "ROOT/RMiniFile.hxx"

#include <cstring>
#include <iostream>

ROOT::Experimental::RLogChannel &ROOT::Experimental::NTupleLog() {
   static RLogChannel sLog("ROOT.NTuple");
   return sLog;
}
