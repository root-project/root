// @(#)root/alien:$Name:  $:$Id: TAlienJobStatusList.h,v 1.2 2007/03/19 17:41:37 rdm Exp $
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

public:
   TAlienJobStatusList() { gGridJobStatusList = this; }
   virtual ~TAlienJobStatusList() { if (gGridJobStatusList == this); gGridJobStatusList=0;}
   virtual void Print(Option_t *options) const;
   virtual void Print(Option_t *wildcard, Option_t *option) const { TCollection::Print(wildcard, option); }

   ClassDef(TAlienJobStatusList,1)  // ABC defining interface to a list of AliEn GRID jobs
};

#endif
