// @(#)root/alien:$Id$
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

#include "TGridJobStatus.h"
#include "TMap.h"

class TAlienJob;
class TAlienMasterJob;


class TAlienJobStatus : public TGridJobStatus {

friend class TAlienJob;
friend class TAlienMasterJob;

private:
   TMap fStatus;     // Contains the status information of the job.
                     // In the Alien implementation this is a string, string map.
   TString fJdlTag;  // JdlTag

   void ClearSetStatus(const char *status);

public:
   TAlienJobStatus() { }
   TAlienJobStatus(TMap *status);
   virtual ~TAlienJobStatus();

   const char *GetJdlKey(const char *key);
   const char *GetKey(const char *key);

   virtual EGridJobStatus GetStatus() const;
   virtual void Print(Option_t *) const;

   void PrintJob(Bool_t full = kTRUE) const;

   Bool_t IsFolder() const { return kTRUE;}
   void Browse(TBrowser *b);

   ClassDef(TAlienJobStatus,1)  // Alien implementation of TGridJobStatus
};

#endif
