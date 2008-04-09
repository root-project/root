/******************************************************************************/
/*                                                                            */
/*                      X r d O l b M a n T r e e . c c                       */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <stdio.h>

#include "XrdOlb/XrdOlbManTree.hh"
#include "XrdOlb/XrdOlbServer.hh"
#include "XrdOlb/XrdOlbTrace.hh"
  
using namespace XrdOlb;

/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/
  
XrdOlbManTree XrdOlb::ManTree;

/******************************************************************************/
/*                               C o n n e c t                                */
/******************************************************************************/
  
int XrdOlbManTree::Connect(int sID, XrdOlbServer *sp)
{
   XrdSysMutexHelper Monitor(myMutex);
   char mybuff[8];
   int i;

// Rule 1: If we are already connected, thell the caller to disband the
//         connection as we must have a connection to an interior node.
//
   if (myStatus == Connected) return 0;
   numConn++;
   tmInfo[sID].servP = sp;

// Rule 2: If we connected to a root node then consider ourselves connected
//         only if all connections are to the root.
//
   if (tmInfo[sID].Level == 0)
      {if (numConn == maxConn)
          {myStatus = Connected; conLevel = 0; atRoot = 1;
           Say.Emsg("ManTree", "Now connected to", buff, "root node(s)");
          }
       tmInfo[sID].Status = Connected;
       return 1;
      }

// Rule 3: We connected to an interior node. Disband all other existing
//         connections (these should only be to root nodes) and consider 
//         ourselves connected.
//
   for (i = 0; i < maxTMI; i++)
       if (i != sID && tmInfo[i].Status == Connected)
          {tmInfo[i].servP->Send("1@0 disc\n", 9);
           tmInfo[i].Status = Pending;
          }
   myStatus = Connected;
   conLevel = tmInfo[sID].Level;
   conSID   = sID;
   atRoot   = 0;

// Document our connection configuration
//
   snprintf(mybuff, sizeof(mybuff), "%d", conLevel);
   Say.Emsg("ManTree", "Now connected to supervisor at level", mybuff);
   return 1;
}

/******************************************************************************/
/*                                  D i s c                                   */
/******************************************************************************/
  
void XrdOlbManTree::Disc(int sID)
{

// A connected caller has lost it's connection.
//
   myMutex.Lock();
   if (tmInfo[sID].Status == Connected || tmInfo[sID].Status == Pending) 
      numConn--;
   tmInfo[sID].Status = Active;
   if (atRoot || (conLevel && conSID == sID)) myStatus = Active;
   tmInfo[sID].servP = 0;
   myMutex.UnLock();
}

/******************************************************************************/
/*                              R e g i s t e r                               */
/******************************************************************************/
  
int XrdOlbManTree::Register()
{
   int sID;

// Add this server to the tree table. Register is called only once and there
// can be no more than MTMax connections to a manager. Hence, we dispense with
// error checking (how optimistic :-)
//
   myMutex.Lock();
   tmInfo[maxTMI].Status= Active;
   sID = maxTMI; maxTMI++;
   myMutex.UnLock();
   return sID;
}

/******************************************************************************/
/*                             s e t M a x C o n                              */
/******************************************************************************/
  
void XrdOlbManTree::setMaxCon(int n)
{
    maxConn = n;
    snprintf(buff, sizeof(buff), "%d", n);
}

/******************************************************************************/
/*                                T r y i n g                                 */
/******************************************************************************/
  
// This method arranges server connections to a manager to form a minimal B-Tree
// free of phantom arcs. The rule is simple, either all connections are to the
// root of tree or there is only one connection to an interior node. Because
// node discovery is non-determinstic, we must make sure that all root nodes
// are tried so as to discover the full set of supervisor nodes we can contact.
// This method returns True if the caller may continue at the indicated level
// and False if the caller should restart at the root node.
//
int XrdOlbManTree::Trying(int sID, int lvl)
{
   int i;

// Set the current status of the connection
//
   myMutex.Lock();
   tmInfo[sID].Level  = lvl;

// Rule 1: If we are already connected at level >0 then the caller must wait
//
   if (myStatus == Connected && conLevel > 0) 
      {Pause(sID);
       return (lvl == tmInfo[sID].Level);
      }

// Rule 2: If the caller is trying level 0 then any waiting threads must be
//         allowed to continue but forced to level 0. This allows us to discover 
//         all the supervisors connected to the root.
//
   if (!lvl)
      {if (numWaiting)
          for (i = 0; i < maxTMI; i++)
              if (i != sID && tmInfo[i].Status == Waiting)
                 {tmInfo[i].Level = 0; Redrive(i);}
          myMutex.UnLock();
          return 1;
      }

// Rule 3: If the caller is trying at a non-zero level (interior node) and
//         someone else is trying at a non-zero level, then the caller must
//         wait.
//
   for (i = 0; i < maxTMI; i++)
        if (i != sID && tmInfo[i].Status == Active && tmInfo[i].Level) break;
   if (i < maxTMI) Pause(sID);
      else myMutex.UnLock();

// The caller may continue. Indicate whether the caller must restart at the
// root node. If the caller may continue trying to connect to an interior
// node then it's the only thread trying to do so.
//
   return (lvl == tmInfo[sID].Level);
}
