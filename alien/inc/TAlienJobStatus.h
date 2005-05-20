// @(#)root/alien:$Name:  $:$Id: TAlienJobStatus.h,v 1.5 2004/11/01 17:38:08 jgrosseo Exp $
// Author: Jan Fiete Grosse-Oetringhaus   06/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAlienJobStatus
#define ROOT_TAlienJobStatus

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienJobStatus                                                      //
//                                                                      //
// Alien implementation of TGridJobStatus.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGridJobStatus
#include "TGridJobStatus.h"
#endif
#ifndef ROOT_TMap
#include "TMap.h"
#endif

class TAlienJob;
class TAlienMasterJob;


class TAlienJobStatus : public TGridJobStatus {

friend class TAlienJob;
friend class TAlienMasterJob;

private:
   TMap fStatus; // Contains the status information of the job.
                 // In the Alien implementation this is a string, string map.

   void ClearSetStatus(const char *status);

public:
   TAlienJobStatus() { }
   TAlienJobStatus(TMap *status);
   virtual ~TAlienJobStatus();

   virtual EGridJobStatus GetStatus() const;
   virtual void Print(Option_t *) const;
   void PrintJob(Bool_t full = kTRUE) const;

   Bool_t IsFolder() const { return kTRUE;}
   void Browse(TBrowser* b);

   ClassDef(TAlienJobStatus,1)  // Alien implementation of TGridJobStatus
};

#endif
