#ifndef __XRDOLBMANTREE_HH_
#define __XRDOLBMANTREE_HH_
/******************************************************************************/
/*                                                                            */
/*                      X r d O l b M a n T r e e . h h                       */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include "XrdOlb/XrdOlbManager.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdOlbServer;
  
class XrdOlbManTree
{
public:

int  Connect(int sID, XrdOlbServer *sp);

void Disc(int sID);

int  Register();

void setMaxCon(int i);

int  Trying(int sID, int Lvl);

enum connStat {Active, Connected, None, Pending, Waiting};

     XrdOlbManTree() : maxTMI(0),   numConn(0), maxConn(0),    atRoot(0),
                       conLevel(0), conSID(-1), numWaiting(0),
                       myStatus(Active) {};
    ~XrdOlbManTree() {};

private:

void Redrive(int sID) {tmInfo[sID].Status = Active;
                       tmInfo[sID].theSem.Post();
                       numWaiting--;
                      }
void Pause(int sID)   {tmInfo[sID].Status = Waiting;
                       numWaiting++;
                       myMutex.UnLock();
                       tmInfo[sID].theSem.Wait();
                      }

XrdSysMutex     myMutex;


struct TreeInfo
       {XrdSysSemaphore theSem;
        XrdOlbServer   *servP;
        connStat        Status;
        int             Level;

        TreeInfo() : theSem(0), servP(0), Status(None), Level(0) {};
       ~TreeInfo() {};

       }         tmInfo[XrdOlbManager::MTMax];

char            buff[8];
int             maxTMI;
int             numConn;
int             maxConn;
int             atRoot;
int             conLevel;
int             conSID;
int             numWaiting;
connStat        myStatus;
};

namespace XrdOlb
{
extern XrdOlbManTree ManTree;
}
#endif
