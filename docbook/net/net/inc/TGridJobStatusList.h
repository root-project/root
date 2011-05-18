// @(#)root/net:$Id$
// Author: Andreas-Joachim Peters  10/12/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGridJobStatusList
#define ROOT_TGridJobStatusList

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGridJobStatusList                                                   //
//                                                                      //
// Abstract base class defining a list of GRID job status               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TList
#include "TList.h"
#endif

#ifndef ROOT_TGridJob
#include "TGridJob.h"
#endif


class TGridJob;

class TGridJobStatusList : public TList {

protected:
   TString  fJobID;  // the job's ID

public:
   TGridJobStatusList() : fJobID("") { }
   virtual ~TGridJobStatusList() { }

   ClassDef(TGridJobStatusList,1)  // ABC defining interface to a list of GRID jobs
};

R__EXTERN TGridJobStatusList *gGridJobStatusList;

#endif
