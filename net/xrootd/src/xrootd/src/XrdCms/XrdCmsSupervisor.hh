#ifndef __CMS_SUPERVISOR_H__
#define __CMS_SUPERVISOR_H__
/******************************************************************************/
/*                                                                            */
/*                   X r d C m s S u p e r v i s o r . h h                    */
/*                                                                            */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

class XrdInet;

class XrdCmsSupervisor
{
public:

static int superOK;

static int Init(const char *AdminPath, int AdminMode);

static void Start();

           XrdCmsSupervisor() {}
          ~XrdCmsSupervisor() {}

private:

static XrdInet *NetTCPr;
};
#endif
