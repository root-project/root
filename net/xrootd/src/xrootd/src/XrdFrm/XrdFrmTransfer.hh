#ifndef __FRMTRANSFER_H__
#define __FRMTRANSFER_H__
/******************************************************************************/
/*                                                                            */
/*                     X r d F r m T r a n s f e r . h h                      */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include "XrdOuc/XrdOucHash.hh"
#include "XrdSys/XrdSysPthread.hh"

class  XrdFrmReqFile;
class  XrdFrmRequest;
struct XrdFrmTranArg;
struct XrdFrmTranChk;
class  XrdFrmXfrJob;
class  XrdOucProg;

class XrdFrmTransfer
{
public:

static
const  char *checkFF(const char *Path);

static int   Init();

       void  Start();

             XrdFrmTransfer();
            ~XrdFrmTransfer() {}

private:
const char *Fetch();
const char *FetchDone(char *lfnpath, int &rc, time_t lktime);
const char *ffCheck();
      void  ffMake(int nofile=0);
      int   SetupCmd(XrdFrmTranArg *aP);
      int   TrackDC(char *Lfn, char *Mdp, char *Rfn);
      int   TrackDC(char *Rfn);
const char *Throw();
      void  Throwaway();
      void  ThrowDone(XrdFrmTranChk *cP, time_t endTime);
const char *ThrowOK(XrdFrmTranChk *cP);

static XrdSysMutex               pMutex;
static XrdOucHash<char>          pTab;

XrdOucProg    *xfrCmd[4];
XrdFrmXfrJob  *xfrP;
char           cmdBuff[4096];
};
#endif
