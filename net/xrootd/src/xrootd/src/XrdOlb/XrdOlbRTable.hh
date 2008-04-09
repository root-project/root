#ifndef __XRDOLBRTABLE_HH_
#define __XRDOLBRTABLE_HH_
/******************************************************************************/
/*                                                                            */
/*                       X r d O l b R T a b l e . h h                        */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <string.h>

#include "XrdOlb/XrdOlbServer.hh"
#include "XrdSys/XrdSysPthread.hh"
  
class XrdOlbRTable
{
public:

short         Add(XrdOlbServer *sp);

void          Del(XrdOlbServer *sp);

XrdOlbServer *Find(short Num, int Inst);

void          Lock() {myMutex.Lock();}

void          UnLock() {myMutex.UnLock();}

              XrdOlbRTable() {memset(Rtable, 0, sizeof(Rtable)); Hwm = -1;}

             ~XrdOlbRTable() {}

private:

static const int   maxRD = 65;  // slot 0 is never used.

XrdSysMutex   myMutex;
XrdOlbServer *Rtable[maxRD];
int           Hwm;
};

namespace XrdOlb
{
extern    XrdOlbRTable RTable;
}
#endif
