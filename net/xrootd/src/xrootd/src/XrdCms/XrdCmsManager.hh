#ifndef __CMS_MANAGER__H
#define __CMS_MANAGER__H
/******************************************************************************/
/*                                                                            */
/*                      X r d C m s M a n a g e r . h h                       */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <stdlib.h>
#include <string.h>
#include <strings.h>
  
#include "XProtocol/YProtocol.hh"

#include "XrdCms/XrdCmsManList.hh"
#include "XrdCms/XrdCmsTypes.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdLink;
class XrdCmsDrop;
class XrdCmsNode;
class XrdCmsServer;
  
/******************************************************************************/
/*                   C l a s s   X r d C m s M a n a g e r                    */
/******************************************************************************/
  
// This a single-instance global class
//
class XrdCmsManager
{
public:

static const int MTMax = 16;   // Maximum number of Managers

XrdCmsNode *Add(XrdLink *lp, int Lvl);

void        Inform(const char *What, const char *Data, int Dlen);
void        Inform(const char *What, struct iovec *vP, int vN, int vT=0);
void        Inform(XrdCms::CmsReqCode rCode, int rMod, const char *Arg=0, int Alen=0);
void        Inform(XrdCms::CmsRRHdr &Hdr, const char *Arg=0, int Alen=0);

int         Present() {return MTHi >= 0;};

void        Remove(XrdCmsNode *nP, const char *reason=0);

void        Reset();

            XrdCmsManager();
           ~XrdCmsManager() {} // This object should never be deleted

private:

XrdSysMutex   MTMutex;
XrdCmsNode   *MastTab[MTMax];

int  MTHi;
};

namespace XrdCms
{
extern    XrdCmsManager Manager;
}
#endif
