#ifndef __CMS_JOB_H__
#define __CMS_JOB_H__
/******************************************************************************/
/*                                                                            */
/*                          X r d C m s J o b . h h                           */
/*                                                                            */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

#include "XProtocol/YProtocol.hh"

#include "Xrd/XrdJob.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdCmsProtocol;
class XrdCmsRRData;

class XrdCmsJob : public XrdJob
{
public:

static XrdCmsJob      *Alloc(XrdCmsProtocol *, XrdCmsRRData *);

       void            DoIt();

       void            Recycle();

       XrdCmsJob() : XrdJob("cms request job"), JobLink(0) {}
      ~XrdCmsJob() {}

private:

static XrdSysMutex     JobMutex;
static XrdCmsJob      *JobStack;
       XrdCmsJob      *JobLink;

       XrdCmsProtocol *theProto;
       XrdCmsRRData   *theData;
};
#endif
