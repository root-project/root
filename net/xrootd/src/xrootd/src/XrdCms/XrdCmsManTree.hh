#ifndef __XRDCMSMANTREE_HH_
#define __XRDCMSMANTREE_HH_
/******************************************************************************/
/*                                                                            */
/*                      X r d C m s M a n T r e e . h h                       */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include "XrdCms/XrdCmsManager.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdCmsNode;
  
class XrdCmsManTree
{
public:

int  Connect(int nID, XrdCmsNode *nP);

void Disc(int nID);

int  Register();

void setMaxCon(int i);

int  Trying(int nID, int Lvl);

enum connStat {Active, Connected, None, Pending, Waiting};

     XrdCmsManTree() : maxTMI(0),   numConn(0), maxConn(0),    atRoot(0),
                       conLevel(0), conNID(-1), numWaiting(0),
                       myStatus(Active) {};
    ~XrdCmsManTree() {};

private:

void Redrive(int nID) {tmInfo[nID].Status = Active;
                       tmInfo[nID].theSem.Post();
                       numWaiting--;
                      }
void Pause(int nID)   {tmInfo[nID].Status = Waiting;
                       numWaiting++;
                       myMutex.UnLock();
                       tmInfo[nID].theSem.Wait();
                      }

XrdSysMutex     myMutex;


struct TreeInfo
       {XrdSysSemaphore theSem;
        XrdCmsNode     *nodeP;
        connStat        Status;
        int             Level;

        TreeInfo() : theSem(0), nodeP(0), Status(None), Level(0) {};
       ~TreeInfo() {};

       }         tmInfo[XrdCmsManager::MTMax];

char            buff[8];
int             maxTMI;
int             numConn;
int             maxConn;
int             atRoot;
int             conLevel;
int             conNID;
int             numWaiting;
connStat        myStatus;
};

namespace XrdCms
{
extern XrdCmsManTree ManTree;
}
#endif
