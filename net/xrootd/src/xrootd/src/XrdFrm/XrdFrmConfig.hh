#ifndef _XRD_FRMCONFIG_H
#define _XRD_FRMCONFIG_H
/******************************************************************************/
/*                                                                            */
/*                       X r d F r m C o n f i g . h h                        */
/*                                                                            */
/* (C) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC02-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/

//          $Id$

class XrdOss;
class XrdOucMsubs;
class XrdOucName2Name;
class XrdOucStream;

class XrdFrmConfig
{
public:

const char         *myProg;
const char         *myName;
const char         *myInst;
const char         *myInsName;
const char         *myFrmid;
const char         *myFrmID;
char               *AdminPath;
char               *myInstance;
char               *StopFile;
char               *qPath;
char               *c2sFN;
char               *xfrCmd;
XrdOucMsubs        *xfrVec;
XrdOucName2Name    *the_N2N;   // -> File mapper object
XrdOss             *ossFS;
int                 AdminMode;
int                 isAgent;
int                 xfrMax;
int                 WaitTime;
int                 monStage;
int                 sSpec;

int   Configure(int argc, char **argv, int (*ppf)());

int   LocalPath (const char *oldp, char *newp, int newpsz);

int   RemotePath(const char *oldp, char *newp, int newpsz);

enum  SubSys {ssMigr, ssPstg, ssPurg};

      XrdFrmConfig(SubSys ss, const char *vopts, const char *uinfo);
     ~XrdFrmConfig() {}

private:
XrdOucMsubs *ConfigCmd(const char *cname, char *cdata);
int          ConfigN2N();
int          ConfigPaths();
int          ConfigProc();
int          ConfigXeq(char *var, int mbok);
int          Grab(const char *var, char **Dest, int nosubs);
void         Usage(int rc);
int          xapath();
int          xmaxx();
int          xnml();
int          xmon();
int          xwtm();

char               *ConfigFN;
char               *ossLib;
char               *LocalRoot;
char               *RemoteRoot;
XrdOucStream       *cFile;

int                 plnDTS;
const char         *pfxDTS;
const char         *vOpts;
const char         *uInfo;
char               *N2N_Lib;   // -> Name2Name Library Path
char               *N2N_Parms; // -> Name2Name Object Parameters
XrdOucName2Name    *lcl_N2N;   // -> File mapper for local  files
XrdOucName2Name    *rmt_N2N;   // -> File mapper for remote files
SubSys              ssID;
};
namespace XrdFrm
{
extern XrdFrmConfig Config;
}
#endif
