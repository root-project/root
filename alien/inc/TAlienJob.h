// @(#)root/alien:$Name:  $:$Id: TAlienJob.h,v 1.4 2004/10/28 08:58:54 jgrosseo Exp $
// Author: Jan Fiete Grosse-Oetringhaus  06/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAlienJob
#define ROOT_TAlienJob

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienJob                                                            //
//                                                                      //
// Alien implentation of TGridJob                                       //
//                                                                      //
// Related classes are TAlienJobStatus.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGridJob
#include "TGridJob.h"
#endif


class TAlienJob : public TGridJob {

public:
   TAlienJob(GridJobID_t jobID) : TGridJob(jobID) { }
   virtual ~TAlienJob() { }

   virtual TGridJobStatus *GetJobStatus() const;

   ClassDef(TAlienJob,1)  // Alien implementation of TGridJob
};

#endif
