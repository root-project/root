// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov, 17/8/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_IOSFileScanner
#define ROOT_IOSFileScanner

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// ScanForVisibleObjects                                                //
//                                                                      //
// Function looks through file's contents, looking for TH1 and TGraph   //
// derived objects (objects, which can be visualized by the current     //
// ROOT code for iOS)                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#include <set>

#ifndef ROOT_TString
#include "TString.h"
#endif

class TObject;
class TFile;

namespace ROOT {
namespace iOS {
namespace FileUtils {

//Find objects of "visible" types in a root file.
void ScanFileForVisibleObjects(TFile *file, const std::set<TString> &visibleTypes, std::vector<TObject *> &objects, std::vector<TString> &options);

}//FileUtils
}//iOS
}//ROOT

#endif
