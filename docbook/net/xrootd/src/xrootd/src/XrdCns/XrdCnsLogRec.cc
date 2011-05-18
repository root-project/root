/******************************************************************************/
/*                                                                            */
/*                       X r d C n s L o g R e c . c c                        */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$

const char *XrdCnsLogRecCVSID = "$Id$";

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

#include "XrdCns/XrdCnsLogRec.hh"
#include "XrdSys/XrdSysTimer.hh"
 
/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
extern XrdSysError                  XrdLog;

XrdSysMutex            XrdCnsLogRec::fMutex;
XrdCnsLogRec          *XrdCnsLogRec::freeRec = 0;

XrdSysSemaphore        XrdCnsLogRec::qSem(0);
XrdSysMutex            XrdCnsLogRec::qMutex;
XrdCnsLogRec          *XrdCnsLogRec::frstRec = 0;
XrdCnsLogRec          *XrdCnsLogRec::lastRec = 0;

int                    XrdCnsLogRec::Running = 0;

const char            *XrdCnsLogRec::IArg = "I755          -1        ";
const char            *XrdCnsLogRec::iArg = "i644           0        ";

/******************************************************************************/
/*                                  A l l o c                                  */
/******************************************************************************/
  
XrdCnsLogRec *XrdCnsLogRec::Alloc()
{
   XrdCnsLogRec *rP;

// Allocate a request object. Develop a serial sequence if init wanted
//
   fMutex.Lock();
   if ((rP = freeRec)) freeRec = rP->Next;
      else rP = new XrdCnsLogRec();
   fMutex.UnLock();

// Pre-initialize the record
//
   rP->Next = 0;
   memset(&rP->Rec.Hdr, 0, sizeof(struct Ctl));
   memset(&rP->Rec.Data, ' ', FixDLen);
   rP->Rec.Data.Mode[2]  = '0';
   rP->Rec.Data.SorT[11] = '0';
   rP->Rec.Data.Type = '?';
   return rP;
}

/******************************************************************************/
/*                                   G e t                                    */
/******************************************************************************/
  
XrdCnsLogRec *XrdCnsLogRec::Get(char &lrType)
{
   XrdCnsLogRec *lrP;

// Find the request in the slot table
//
   qMutex.Lock();
   while(!(lrP = frstRec))
        {Running = 0;
         qMutex.UnLock();
         qSem.Wait();
         qMutex.Lock();
        }
   if (!(frstRec = lrP->Next)) lastRec = 0;
   qMutex.UnLock();

// Get the type and if its an eol marker recycle now
//
   if (!(lrType = lrP->Rec.Data.Type)) {lrP->Recycle(); lrP = 0;}
   return lrP;
}


/******************************************************************************/
/*                                 Q u e u e                                  */
/******************************************************************************/
  
void XrdCnsLogRec::Queue()
{

// Put request on the queue
//
   qMutex.Lock();
   if (frstRec) lastRec->Next = this;
      else      frstRec       = this;
   lastRec = this;

// Tell dequeue thread we have something if it's not already running
//
   if (!Running) {qSem.Post(); Running = 1;}
   qMutex.UnLock();
}

/******************************************************************************/
/*                               R e c y c l e                                */
/******************************************************************************/
  
void XrdCnsLogRec::Recycle()
{

// Put this object on the free queue
//
   fMutex.Lock();
   Next = freeRec;
   freeRec = this;
   fMutex.UnLock();
}

/******************************************************************************/
/*                               s e t D a t a                                */
/******************************************************************************/

int XrdCnsLogRec::setData(const char *dP1, const char *dP2)
{
  int n1 = strlen(dP1), n2 = strlen(dP2);
  char *dP;

// Make sure we have room "'lfn' + 'data1' + ' ' + data2"
//
   if (n1+n2+2 > MAXPATHLEN) return 0;

// Add the data in the fields
//
   setSize(static_cast<long long>(Rec.Hdr.lfn1Len));
   dP = Rec.Data.lfn + Rec.Hdr.lfn1Len+1;
   strcpy(dP, dP1);
   dP += n1; *dP++ = ' ';
   strcpy(dP, dP2);
   Rec.Hdr.lfn2Len = n1+n2+1;
   return Rec.Hdr.lfn2Len;
}
  
/******************************************************************************/
/*                               s e t T y p e                                */
/******************************************************************************/

int XrdCnsLogRec::setType(const char *lrName)
{
        if (!strcmp(lrName, "closew")) setType(lrClosew);
   else if (!strcmp(lrName, "create")) setType(lrCreate);
   else if (!strcmp(lrName, "mkdir"))  setType(lrMkdir);
   else if (!strcmp(lrName, "mv"))     setType(lrMv);
   else if (!strcmp(lrName, "rm"))     setType(lrRm);
   else if (!strcmp(lrName, "rmdir"))  setType(lrRmdir);
   else return 0;

   return 1;
}
