#ifndef __XRDCMSRTABLE_HH_
#define __XRDCMSRTABLE_HH_
/******************************************************************************/
/*                                                                            */
/*                       X r d C m s R T a b l e . h h                        */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <string.h>

#include "XrdCms/XrdCmsNode.hh"
#include "XrdCms/XrdCmsTypes.hh"
#include "XrdSys/XrdSysPthread.hh"
  
class XrdCmsRTable
{
public:

short         Add(XrdCmsNode *nP);

void          Del(XrdCmsNode *nP);

XrdCmsNode   *Find(short Num, int Inst);

void          Send(const char *What, const char *data, int dlen);

void          Lock() {myMutex.Lock();}

void          UnLock() {myMutex.UnLock();}

              XrdCmsRTable() {memset(Rtable, 0, sizeof(Rtable)); Hwm = -1;}

             ~XrdCmsRTable() {}

private:

XrdSysMutex   myMutex;
XrdCmsNode   *Rtable[maxRD];
int           Hwm;
};

namespace XrdCms
{
extern    XrdCmsRTable RTable;
}
#endif
