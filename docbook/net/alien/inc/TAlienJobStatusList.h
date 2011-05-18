// @(#)root/alien:$Id$
// Author: Andreas-Joachim Peters  10/12/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAlienJobStatusList
#define ROOT_TAlienJobStatusList

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienJobStatusList                                                  //
//                                                                      //
// Alien implementation of TGridJobStatusList                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGridJobStatusList
#include "TGridJobStatusList.h"
#endif

class TAlienJob;

class TAlienJobStatusList : public TGridJobStatusList {

protected:
   TString  fJobID;  // the job's ID

   virtual void PrintCollectionEntry(TObject* entry, Option_t* option, Int_t recurse) const;

public:
   TAlienJobStatusList() { gGridJobStatusList = this; }
   virtual ~TAlienJobStatusList() { if (gGridJobStatusList == this) gGridJobStatusList=0; }

   ClassDef(TAlienJobStatusList,1)  // ABC defining interface to a list of AliEn GRID jobs
};

#endif
