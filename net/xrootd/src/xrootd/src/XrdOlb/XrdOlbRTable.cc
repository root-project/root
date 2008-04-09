/******************************************************************************/
/*                                                                            */
/*                       X r d O l b R T a b l e . c c                        */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdOlbRTableCVSID = "$Id$";

#include "XrdOlb/XrdOlbRTable.hh"

/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
XrdOlbRTable XrdOlb::RTable;

/******************************************************************************/
/*                                   A d d                                    */
/******************************************************************************/
  
short XrdOlbRTable::Add(XrdOlbServer *sp)
{
   int i;

// Find a free slot for this server.
//
   myMutex.Lock();
   for (i = 1; i < maxRD; i++) if (!Rtable[i]) break;

// Insert the server if found
//
   if (i >= maxRD) i = 0;
      else {Rtable[i] = sp;
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
  
void XrdOlbRTable::Del(XrdOlbServer *sp)
{
   int i;

// Find the slot for this server.
//
   myMutex.Lock();
   for (i = 1; i <= Hwm; i++) if (Rtable[i] == sp) break;

// Remove the server if found
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
// because this is the only way we can interlock the use of the server object
// with deletion of that object as it must be removed prior to deletion.

XrdOlbServer *XrdOlbRTable::Find(short Num, int Inst)
{

// Find the instance of the server in the indicated slot
//
   if (Num <= Hwm && Rtable[Num] && Rtable[Num]->Inst() == Inst)
      return Rtable[Num];
   return (XrdOlbServer *)0;
}
