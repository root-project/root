#ifndef __FRCREQAGENT_H__
#define __FRCREQAGENT_H__
/******************************************************************************/
/*                                                                            */
/*                     X r d F r c R e q A g e n t . h h                      */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include "XrdFrc/XrdFrcReqFile.hh"
#include "XrdFrc/XrdFrcRequest.hh"

class XrdFrcReqAgent
{
public:

void Add(XrdFrcRequest &Request);

void Del(XrdFrcRequest &Request);

int  List(XrdFrcRequest::Item *Items, int Num);
int  List(XrdFrcRequest::Item *Items, int Num, int Prty);

int  NextLFN(char *Buff, int Bsz, int Prty, int &Offs);

void Ping(const char *Msg=0);

int  Start(char *aPath, int aMode);

     XrdFrcReqAgent(const char *Me, int qVal);
    ~XrdFrcReqAgent() {}

private:

static char     *c2sFN;

XrdFrcReqFile   *rQueue[XrdFrcRequest::maxPQE];
const char      *Persona;
const char      *pingMsg;
const char      *myName;
int              theQ;
};
#endif
