// $Id$
// Author: Sergey Linev   22/12/2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TRootSnifferFull.h"

#include "TH1.h"
#include "TGraph.h"
#include "TProfile.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TKey.h"
#include "TList.h"
#include "TMemFile.h"
#include "TStreamerInfo.h"
#include "TBufferFile.h"
#include "TBufferJSON.h"
#include "TBufferXML.h"
#include "TROOT.h"
#include "TTimer.h"
#include "TFolder.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TClass.h"
#include "TMethod.h"
#include "TFunction.h"
#include "TMethodArg.h"
#include "TMethodCall.h"
#include "TRealData.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TBaseClass.h"
#include "TObjString.h"
#include "TUrl.h"
#include "TImage.h"
#include "RZip.h"
#include "RVersion.h"
#include "TVirtualMutex.h"
#include "TRootSnifferStore.h"
#include "THttpCallArg.h"

#include <stdlib.h>
#include <vector>
#include <string.h>

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootSnifferFull                                                     //
//                                                                      //
// Subclass of TRootSniffer, which provides access to different         //
// ROOT collections and containers like TTree, TCanvas, ...             //
//////////////////////////////////////////////////////////////////////////

ClassImp(TRootSnifferFull);

////////////////////////////////////////////////////////////////////////////////
/// constructor

TRootSnifferFull::TRootSnifferFull(const char *name, const char *objpath)
   : TRootSniffer(name, objpath)
{
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TRootSnifferFull::~TRootSnifferFull()
{
}
