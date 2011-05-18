#ifndef __XRDCMSSTATE_H_
#define __XRDCMSSTATE_H_
/******************************************************************************/
/*                                                                            */
/*                        X r d C m s S t a t e . h h                         */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include "XrdSys/XrdSysPthread.hh"
#include "XrdCms/XrdCmsTypes.hh"

class XrdLink;

class XrdCmsState
{
public:

int   Suspended;
int   NoStaging;

void  Enable();

void *Monitor();

int   Port();

void  sendState(XrdLink *Link);

void  Set(int ncount);
void  Set(int ncount, int isman, const char *AdminPath);

enum  StateType {Active = 0, Counts, FrontEnd, Space, Stage};

void  Update(StateType StateT, int ActivVal, int StageVal=0);

      XrdCmsState();
     ~XrdCmsState() {}
  
static const char SRV_Suspend = 1;
static const char FES_Suspend = 2;
static const char All_Suspend = 3;
static const char All_NoStage = 4;

private:
unsigned char Status(int Changes, int theState);

XrdSysSemaphore mySemaphore;
XrdSysMutex     myMutex;

const char     *NoStageFile;
const char     *SuspendFile;

int             minNodeCnt;   // Minimum number of needed subscribers
int             numActive;    // Number of active subscribers
int             numStaging;   // Number of subscribers that can stage
int             dataPort;     // Current data port number

char            currState;    // Current  state
char            prevState;    // Previous state
char            feOK;         // Front end functioning
char            noSpace;      // We don't have enough space
char            adminSuspend; // Admin asked for suspension
char            adminNoStage; // Admin asked for no staging
char            isMan;        // We are a manager (i.e., have redirectors)
char            Enabled;      // We are now enabled for reporting
};

namespace XrdCms
{
extern    XrdCmsState CmsState;
}
#endif
