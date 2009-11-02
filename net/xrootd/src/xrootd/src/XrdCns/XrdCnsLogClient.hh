#ifndef __XRDCNSLogClient_h_
#define __XRDCNSLogClient_h_
/******************************************************************************/
/*                                                                            */
/*                    X r d C n s L o g C l i e n t . h h                     */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <sys/param.h>
  
#include "XrdSys/XrdSysPthread.hh"

class XrdClient;
class XrdClientAdmin;
class XrdCnsLogFile;
class XrdCnsLogRec;
class XrdCnsXref;
class XrdOucTList;

class XrdCnsLogClient
{
public:

int   Activate(XrdCnsLogFile *basefile);

int   Init();

int   Run(int Always=1);

int   Start();

      XrdCnsLogClient(XrdOucTList *rP, XrdCnsLogClient *pcP);
     ~XrdCnsLogClient() {}

private:
XrdClientAdmin *admConnect(XrdClientAdmin *adminP);

int  Archive(XrdCnsLogFile *lfP);
int  do_Create(XrdCnsLogRec *lrP, const char *lfn=0);
int  do_Mkdir(XrdCnsLogRec *lrP);
int  do_Mv(XrdCnsLogRec *lrP);
int  do_Rm(XrdCnsLogRec *lrP);
int  do_Rmdir(XrdCnsLogRec *lrP);
int  do_Trunc(XrdCnsLogRec *lrP, const char *lfn=0);
char getMount(char *Lfn, char *Pfn, XrdCnsXref &Mount);
int  Inventory(XrdCnsLogFile *lfp, const char *dPath);
int  Manifest();
int  mapError(int rc);
int  xrdEmsg(const char *Opname, const char *theFN, XrdClientAdmin *aP);
int  xrdEmsg(const char *Opname, const char *theFN);
int  xrdEmsg(const char *Opname, const char *theFN, XrdClient *fP);

XrdSysMutex      lfMutex;
XrdSysSemaphore  lfSem;
XrdCnsLogClient *Next;
XrdClientAdmin  *Admin;

XrdCnsLogFile   *logFirst;
XrdCnsLogFile   *logLast;

int              pfxNF;
int              sfxFN;
int              arkOnly;

char            *admURL;
char            *urlHost;

char             arkURL[MAXPATHLEN+512];
char            *arkPath;
char            *arkFN;
char             crtURL[MAXPATHLEN+512];
char            *crtFN;
char             logDir[MAXPATHLEN+1];
char            *logFN;
};
#endif
