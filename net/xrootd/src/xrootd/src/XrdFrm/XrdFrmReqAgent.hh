#ifndef __FRMREQAGENT_H__
#define __FRMREQAGENT_H__
/******************************************************************************/
/*                                                                            */
/*                     X r d F r m R e q A g e n t . h h                      */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include "XrdFrm/XrdFrmReqFile.hh"
#include "XrdFrm/XrdFrmRequest.hh"

class XrdFrmReqAgent
{
public:

void Add(XrdFrmRequest &Request);

void Del(XrdFrmRequest &Request);

int  List(XrdFrmRequest::Item *Items, int Num);
int  List(XrdFrmRequest::Item *Items, int Num, int Prty);

int  NextLFN(char *Buff, int Bsz, int Prty, int &Offs);

void Ping(const char *Msg=0);

int  Start(char *aPath, int aMode);

     XrdFrmReqAgent(const char *Me, int qVal);
    ~XrdFrmReqAgent() {}

private:

static char     *c2sFN;

XrdFrmReqFile   *rQueue[XrdFrmRequest::maxPQE];
const char      *Persona;
const char      *pingMsg;
const char      *myName;
int              theQ;
};
#endif
