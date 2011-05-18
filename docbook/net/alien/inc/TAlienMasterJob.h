// @(#)root/alien:$Id$
// Author: Jan Fiete Grosse-Oetringhaus  27/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAlienMasterJob
#define ROOT_TAlienMasterJob

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienMasterJob                                                      //
//                                                                      //
// Special Grid job which contains a master job which controls          //
// underlying jobs resulting from job splitting.                        //
//                                                                      //
// Related classes are TAlienJobStatus.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGridJob
#include "TGridJob.h"
#endif


class TAlienMasterJob : public TGridJob {

public:
   TAlienMasterJob(TString jobID) : TGridJob(jobID) { }
   virtual ~TAlienMasterJob() { }

   virtual TGridJobStatus *GetJobStatus() const;

   void   Print(Option_t *) const;
   Bool_t Merge();
   Bool_t Merge(const char *inputname, const char *mergeoutput = 0);
   void   Browse(TBrowser* b);

   ClassDef(TAlienMasterJob,1) // Special Alien grid job controlling results of job splitting
};

#endif
