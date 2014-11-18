#ifndef __XRDOUCCALLBACK__HH_
#define __XRDOUCCALLBACK__HH_
/******************************************************************************/
/*                                                                            */
/*                     X r d O u c C a l l B a c k . h h                      */
/*                                                                            */
/* (c) 2011 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdSys/XrdSysPthread.hh"

/* The XrdOucCallBack object encapsulates the vagaries of handling callbacks
   in the xrootd framework; where callbacks are allowed. Once a callback is
   successfully established using Init() this object should not be deleted
   until Reply() of Cancel() is called. The destructor automatically calls
   Cancel() is a callback is outstanding. The object may be reused after
   Cancel() or Reply is called. See warnings on Init() and Cancel().

   This object is not MT-safe and must be used in a serial fashion.
*/
  
class XrdOucCallBack : public XrdOucEICB
{
public:

/* Allowed() tell you whether or not am XrdOucErrInfo object has been setup to
             allow callbacks. You should test this before assuming you can use
             the object to effect a callback.

   Returns:  True  - if a callback is allowed.
             False - otherwise.
*/
static int   Allowed(XrdOucErrInfo *eInfo) {return eInfo->getErrCB() != 0;}

/* Cancel()  cancels the callback. If no callback is oustanding, it does
             nothing. Otherwise, the associated endpoint is told to retry
             whatever operation caused the callback to be setup. Warning,
             calling Cancel() or deleting this object after calling Init()
             but not effecting a callback response will cause the calling
             thread to hang!
*/
       void  Cancel();

/* Init()    sets up a call back using the provided XrdOucErrInfo object.
             You must successfully call Init() before calling Reply()!
             Warning, once you cann Init() you *must* effect a callback
             response; otherwise, it is likely a subsequent thread using
             this object will hang!

   Returns:  True  - if a callback was set up.
             False - otherwise (i.e., object does not allow callbacks).
*/
       int   Init(XrdOucErrInfo *eInfo);

/* Reply()   sends the specified results to the endpoint associated with the
             callback esablished by Init(). The parameters are:
             retVal  - The value you would have synchrnously returned.
             eValue  - The numeric value that would have been returned in the
                       original XrdOucErrInfo object.
             eText   - The character string that would have been returned in the
                       original XrdOucErrInfo object.
             Path    - Optional path related to the reply. It is passed to the
                       callback effector and is used for tracing & monitoring.

   Returns:  True  - if a callback was initiated.
             False - callback failed; likely Init() was not successfully called.
*/
       int   Reply(int retVal, int eValue, const char *eText,
                                           const char *Path=0);

             XrdOucCallBack() : Next(0), cbSync(0), cbArg(0), cbObj(0) {}
            ~XrdOucCallBack() {if (cbObj) Cancel();}

// The following is a handy pointer to allow for linking these objects together
//
XrdOucCallBack *Next;

private:
void  Done(int &Result, XrdOucErrInfo *eInfo) {cbSync.Post();}
int   Same(unsigned long long arg1, unsigned long long arg2) {return 0;}

XrdSysSemaphore     cbSync;
unsigned long long  cbArg;
XrdOucEICB         *cbObj;
char                UserID[64];
};
#endif
