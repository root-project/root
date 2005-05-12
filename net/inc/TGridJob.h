// @(#)root/net:$Name:  $:$Id: TGridJob.h,v 1.3 2004/11/01 17:38:09 jgrosseo Exp $
// Author: Jan Fiete Grosse-Oetringhaus  06/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGridJob
#define ROOT_TGridJob

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGridJob                                                             //
//                                                                      //
// Abstract base class defining interface to a GRID job.                //
//                                                                      //
// Related classes are TGridJobStatus.                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TGridJobStatus;

typedef ULong64_t GridJobID_t;


class TGridJob : public TObject {

protected:
   GridJobID_t  fJobID;  // the job's ID

public:
   TGridJob(GridJobID_t jobID) : fJobID(jobID) { }
   virtual ~TGridJob() { }

   virtual GridJobID_t GetJobID() { return fJobID; }

   virtual TGridJobStatus *GetJobStatus() = 0;

   ClassDef(TGridJob,1)
};

#endif
