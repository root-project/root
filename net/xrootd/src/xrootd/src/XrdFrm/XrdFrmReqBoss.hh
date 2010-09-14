#ifndef __FRMREQBOSS_H__
#define __FRMREQBOSS_H__
/******************************************************************************/
/*                                                                            */
/*                      X r d F r m R e q b o s s . h h                       */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include "XrdFrm/XrdFrmReqFile.hh"
#include "XrdFrm/XrdFrmRequest.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdFrmReqBoss
{
public:

void Add(XrdFrmRequest &Request);

void Del(XrdFrmRequest &Request);

void Process();

int  Server();

int  Start(char *aPath, int aMode);

void Wakeup(int PushIt=1);

     XrdFrmReqBoss(const char *Me, int qVal)
                  : rqReady(0),Persona(Me),theQ(qVal),isPosted(0) {}
    ~XrdFrmReqBoss() {}

private:
void Register(XrdFrmRequest &Req, int qNum);

XrdSysSemaphore  rqReady;
XrdFrmReqFile   *rQueue[XrdFrmRequest::maxPQE];
const char      *Persona;
int              theQ;
int              isPosted;
};
#endif
