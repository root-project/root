/******************************************************************************/
/*                                                                            */
/*                       X r d X r o o t d P i o . c c                        */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//        $Id$

const char *XrdXrootdPioCVSID = "$Id$";
  
#include "XrdXrootd/XrdXrootdPio.hh"

/******************************************************************************/
/*                      S t a t i c   V a r i a b l e s                       */
/******************************************************************************/
  
XrdSysMutex        XrdXrootdPio::myMutex;
XrdXrootdPio      *XrdXrootdPio::Free = 0;
int                XrdXrootdPio::FreeNum = 0;

/******************************************************************************/
/*                                 A l l o c                                  */
/******************************************************************************/
  
XrdXrootdPio *XrdXrootdPio::Alloc(int Num)
{
   XrdXrootdPio *lqp, *qp=0;


// Allocate from the free stack
//
   myMutex.Lock();
   if ((qp = Free))
      {do {FreeNum--; Num--; lqp = Free;}
          while((Free = Free->Next) && Num);
       lqp->Next = 0;
      }
   myMutex.UnLock();

// Allocate additional if we have not allocated enough
//
   while(Num--) qp = new XrdXrootdPio(qp);

// All done
//
   return qp;
}

/******************************************************************************/
/*                               R e c y c l e                                */
/******************************************************************************/
  
void XrdXrootdPio::Recycle()
{

// Check if we can hold on to this or must delete it
//
   myMutex.Lock();
   if (FreeNum >= FreeMax) {myMutex.UnLock(); delete this; return;}

// Clean this up and push the element on the free stack
//
   Free = Clear(Free); FreeNum++;
   myMutex.UnLock();
}
