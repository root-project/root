#ifndef __XRDCMSNASH_HH__
#define __XRDCMSNASH_HH__
/******************************************************************************/
/*                                                                            */
/*                         X r d C m s N a s h . h h                          */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include "XrdCms/XrdCmsKey.hh"
  
class XrdCmsNash
{
public:
XrdCmsKeyItem *Add(XrdCmsKey &Key);

XrdCmsKeyItem *Find(XrdCmsKey &Key);

int            Recycle(XrdCmsKeyItem *rip);

// When allocateing a new nash, specify the required starting size. Make
// sure that the previous number is the correct Fibonocci antecedent. The
// series is simply n[j] = n[j-1] + n[j-2].
//
    XrdCmsNash(int psize = 17711, int size = 28657);
   ~XrdCmsNash() {} // Never gets deleted

private:

static const int LoadMax = 80;

void               Expand();

XrdCmsKeyItem  **nashtable;
int              prevtablesize;
int              nashtablesize;
int              nashnum;
int              Threshold;
};
#endif
