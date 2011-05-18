#ifndef __XRDCNS_LogServer__
#define __XRDCNS_LogServer__
/******************************************************************************/
/*                                                                            */
/*                    X r d C n s L o g S e r v e r . h h                     */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include <sys/param.h>

class XrdOucTList;
class XrdCnsLogClient;
class XrdCnsLogFile;
  
class XrdCnsLogServer
{
public:

int  Init(XrdOucTList *rList);

void Run();

     XrdCnsLogServer();
    ~XrdCnsLogServer() {}


private:
void Massage(XrdCnsLogRec *lrP);

XrdCnsLogClient *Client;
XrdCnsLogFile   *logFile;

char             logDir[MAXPATHLEN+1];
char            *logFN;
};
#endif
