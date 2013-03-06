// @(#)root/proofd:$Id$
// Author: Gerardo Ganis  Feb 2013

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XpdObject                                                            //
//                                                                      //
// Authors: G. Ganis, CERN, 2013                                        //
//                                                                      //
// Auxilliary class to stack protocols.                                 //
// Adapted version of XrdObject.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_XpdObject
#define ROOT_XpdObject

#include <string.h>
#include <strings.h>
#include <time.h>
#include <sys/types.h>
#include "Xrd/XrdJob.hh"
#include "XrdOuc/XrdOucTrace.hh"
#include "XrdSys/XrdSysPthread.hh"

class XpdObjectQ;
class XrdProofdProtocol;
class XrdScheduler;
  
class XpdObject {
public:
   friend class XpdObjectQ;

   // Item() supplies the item value associated with itself (used with Next()).
   XrdProofdProtocol  *objectItem() { return fItem; }

   // Next() supplies the next list node.
   XpdObject          *nextObject() { return fNext; }

   // Set the item pointer
   void                setItem(XrdProofdProtocol *ival) { fItem = ival; }

   XpdObject(XrdProofdProtocol *ival=0) { fNext = 0; fItem = ival; fQTime = 0; }
   ~XpdObject() {}

private:
   XpdObject         *fNext;
   XrdProofdProtocol *fItem;
   time_t             fQTime;  // Only used for time-managed objects
};

/******************************************************************************/
/*                           x r d _ O b j e c t Q                            */
/******************************************************************************/
  
// Note to properly cleanup this type of queue you must call Set() at least
// once to cause the time element to be sceduled.

class XrdOucTrace;
  
class XpdObjectQ : public XrdJob {
public:

   XrdProofdProtocol *Pop();
   void Push(XpdObject *Node);
   void Set(int inQMax, time_t agemax=1800);
   void Set(XrdScheduler *sp, XrdOucTrace *tp, int traceChk = 0)
            {fSched = sp; fTrace = tp; fTraceON = traceChk;}
   void DoIt();

   XpdObjectQ(const char *id, const char *desc) : XrdJob(desc) 
          {fCurage = fCount = 0; fMaxage = 0; fTraceID = id;
           fMaxinQ = 32; fMininQ = 16; fFirst = 0; }

   ~XpdObjectQ() {}

private:

   XrdSysMutex    fQMutex;
   XpdObject     *fFirst;
   int            fCount;
   int            fCurage;
   int            fMininQ;
   int            fMaxinQ;
   time_t         fMaxage;
   XrdOucTrace   *fTrace;
   XrdScheduler  *fSched;
   int            fTraceON;
   const char    *fTraceID;
};

#endif
