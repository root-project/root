/******************************************************************************/
/*                                                                            */
/*                       X r d C m s R T a b l e . c c                        */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

// Original Version: 1.2 2006/04/05 02:28:05 abh

const char *XrdCmsRTableCVSID = "$Id$";

#include "XrdCms/XrdCmsRTable.hh"
#include "XrdCms/XrdCmsTrace.hh"

using namespace XrdCms;

/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
XrdCmsRTable XrdCms::RTable;

/******************************************************************************/
/*                                   A d d                                    */
/******************************************************************************/
  
short XrdCmsRTable::Add(XrdCmsNode *nP)
{
   int i;

// Find a free slot for this node.
//
   myMutex.Lock();
   for (i = 1; i < maxRD; i++) if (!Rtable[i]) break;

// Insert the node if found
//
   if (i >= maxRD) i = 0;
      else {Rtable[i] = nP;
            if (i > Hwm) Hwm = i;
           }

// All done
//
   myMutex.UnLock();
   return static_cast<short>(i);
}

/******************************************************************************/
/*                                   D e l                                    */
/******************************************************************************/
  
void XrdCmsRTable::Del(XrdCmsNode *nP)
{
   int i;

// Find the slot for this node.
//
   myMutex.Lock();
   for (i = 1; i <= Hwm; i++) if (Rtable[i] == nP) break;

// Remove the node if found
//
   if (i <= Hwm)
      {Rtable[i] = 0;
       if (i == Hwm) {while(--i) if (Rtable[i]) break; Hwm = i;}
      }

// All done
//
   myMutex.UnLock();
}

/******************************************************************************/
/*                                  F i n d                                   */
/******************************************************************************/

// Note that the caller *must* call Lock() prior to calling find. We do this
// because this is the only way we can interlock the use of the node object
// with deletion of that object as it must be removed prior to deletion.

XrdCmsNode *XrdCmsRTable::Find(short Num, int Inst)
{

// Find the instance of the node in the indicated slot
//
   if (Num <= Hwm && Rtable[Num] && Rtable[Num]->Inst() == Inst)
      return Rtable[Num];
   return (XrdCmsNode *)0;
}

/******************************************************************************/
/*                                  S e n d                                   */
/******************************************************************************/
  
void XrdCmsRTable::Send(const char *What, const char *data, int dlen)
{
   EPNAME("Send");
   int i;

// Send the data to all nodes in this table
//
   myMutex.Lock();
   for (i = 1; i <= Hwm; i++) 
       if (Rtable[i])
          {DEBUG(What <<" to " <<Rtable[i]->Ident);
           Rtable[i]->Send(data, dlen);
          }
   myMutex.UnLock();
}
