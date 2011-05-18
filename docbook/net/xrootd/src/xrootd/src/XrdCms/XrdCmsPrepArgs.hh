#ifndef __CMS_PREPARGS__H
#define __CMS_PREPARGS__H
/******************************************************************************/
/*                                                                            */
/*                     X r d C m s P r e p A r g s . h h                      */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$
  
#include "XProtocol/YProtocol.hh"

#include "Xrd/XrdJob.hh"
#include "XrdCms/XrdCmsNode.hh"
#include "XrdCms/XrdCmsRRData.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdCmsPrepArgs : public XrdJob
{
public:
static const int       iovNum = 2;

XrdCms::CmsRRHdr        Request;
        char           *Ident;
        char           *reqid;
        char           *notify;
        char           *prty;
        char           *mode;
        char           *path;
        char           *opaque;
        char           *clPath;   // ->coloc path, if any
        int             options;
        int             pathlen;  // Includes null byte

        struct iovec    ioV[iovNum];  // To forward the request

        void            DoIt() {if (!XrdCmsNode::do_SelPrep(*this)) delete this;}

static  void            Process();

        void            Queue();

static  XrdCmsPrepArgs *getRequest();

                        XrdCmsPrepArgs(XrdCmsRRData &Arg);

                       ~XrdCmsPrepArgs() {if (Data) free(Data);}

private:

static XrdSysMutex      PAQueue;
static XrdSysSemaphore  PAReady;
       XrdCmsPrepArgs  *Next;
static XrdCmsPrepArgs  *First;
static XrdCmsPrepArgs  *Last;
static int              isIdle;
       char            *Data;

};
#endif
