#ifndef __FRMPSTG_H__
#define __FRMPSTG_H__
/******************************************************************************/
/*                                                                            */
/*                         X r d F r m P s t g . h h                          */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

class XrdOucStream;

class XrdFrmPstg
{
public:

int  Agent(char *c2sFN);

int  Server(int udpFD);

void Server_Driver(int PushIt);

void Server_Stage();

     XrdFrmPstg() {}
    ~XrdFrmPstg() {}

private:

void Agent_Add(XrdOucStream &Request, char *Tok);
void Agent_Del(XrdOucStream &Request, char *Tok);
void Agent_Lst(XrdOucStream &Request, char *Tok);
};
namespace XrdFrm
{
extern XrdFrmPstg PreStage;
}
#endif
