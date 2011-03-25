#ifndef __FRMADMIN__HH
#define __FRMADMIN__HH
/******************************************************************************/
/*                                                                            */
/*                        X r d F r m A d m i n . h h                         */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
#include <stdlib.h>
#include <sys/types.h>

#include "XrdOuc/XrdOucNSWalk.hh"

class  XrdFrmFileset;
class  XrdFrmProxy;
class  XrdOucArgs;
class  XrdOucTList;

class XrdFrmAdmin
{
public:

int  Audit();

int  Copy();

int  Create();

int  Find();

int  Help();

int  MakeLF();

int  Mark();

int  Mmap();

int  Pin();

int  Query();

int  Quit() {exit(finalRC); return 0;}

int  Reloc();

int  Remove();

int  Rename();

void setArgs(int argc, char **argv);

void setArgs(char *argv);

int  xeqArgs(char *Cmd);

     XrdFrmAdmin() : frmProxy(0), frmProxz(0), finalRC(0) {}
    ~XrdFrmAdmin() {}

private:
int  AuditNameNB(XrdFrmFileset *sP);
int  AuditNameNF(XrdFrmFileset *sP);
int  AuditNameNL(XrdFrmFileset *sP);
int  AuditNames();
int  AuditNameXA(XrdFrmFileset *sP);
int  AuditNameXB(XrdFrmFileset *sP);                // runOld
int  AuditNameXL(XrdFrmFileset *sP, int dorm);      // runOld
int  AuditRemove(XrdFrmFileset *sP);
int  AuditSpace();
int  AuditSpaceAX(const char *Path);
int  AuditSpaceAXDB(const char *Path);
int  AuditSpaceAXDC(const char *Path, XrdOucNSWalk::NSEnt *nP);
int  AuditSpaceAXDL(int dorm, const char *Path, const char *Dest);
int  AuditSpaceXA(const char *Space, const char *Path);
int  AuditSpaceXA(XrdFrmFileset *sP);
int  AuditSpaceXB(const char *Space, const char *Path); // runOld
int  AuditSpaceXANB(XrdFrmFileset *sP);                 // runOld
int  AuditUsage();
int  AuditUsage(char *Space);
int  AuditUsageAX(const char *Path);
int  AuditUsageXA(const char *Path, const char *Space);
int  isXA(XrdOucNSWalk::NSEnt *nP);

int  FindFail(XrdOucArgs &Spec);
int  FindMmap(XrdOucArgs &Spec);
int  FindNolk(XrdOucArgs &Spec);
int  FindPins(XrdOucArgs &Spec);
int  FindPins(XrdFrmFileset *sP);
int  FindUnmi(XrdOucArgs &Spec);

void ConfigProxy();

void Emsg(const char *tx1, const char *tx2=0, const char *tx3=0,
                           const char *tx4=0, const char *tx5=0);
void Emsg(int Enum,        const char *tx2=0, const char *tx3=0,
                           const char *tx4=0, const char *tx5=0);
void Msg(const char *tx1,  const char *tx2=0, const char *tx3=0,
                           const char *tx4=0, const char *tx5=0);

int          Parse(const char *What, XrdOucArgs &Spec, const char **Reqs);
int          ParseKeep(const char *What, const char *kTime);
int          ParseOwner(const char *What, char *Uname);
XrdOucTList *ParseSpace(char *Space, char **Path);

char ckAttr(int What, const char *Lfn, char *Pfn, int Pfnsz);
int  mkLock(const char *Lfn);
int  mkFile(int What, const char *Path, const char *Data=0, int Dlen=0);
int  mkMark(const char *Lfn);
int  mkMmap(const char *Lfn);
int  mkPin(const char *Lfn);
char mkStat(int What, const char *Lfn, char *Pfn, int Pfnsz);

// For mkFile and mkStat the following options may be passed via What
//
static const int isPFN= 0x0001; // Filename is actual physical name
static const int mkLF = 0x0002; // Make lock file or copy attribute
static const int mkMF = 0x0004; // Make mmap file or mmap attribute
static const int mkPF = 0x0008; // Make pin  file or pin  attribute

int  QueryPfn(XrdOucArgs &Spec);
int  QueryRfn(XrdOucArgs &Spec);
int  QuerySpace(XrdOucArgs &Spec);
int  QuerySpace(const char *Pfn, char *Lnk=0, int Lsz=0);
int  QueryUsage(XrdOucArgs &Spec);
int  QueryXfrQ(XrdOucArgs &Spec);

int  Reloc(char *srcLfn, char *Space);
int  RelocCP(const char *srcpfn, const char *trgpfn, off_t srcSz);
int  RelocWR(const char *outFn,  int oFD, char *Buff, size_t BLen, off_t Boff);

int  Unlink(const char *Path);
int  UnlinkDir(const char *Path, const char *lclPath);
int  UnlinkDir(XrdOucNSWalk::NSEnt *&nP, XrdOucNSWalk::NSEnt *&dP);
int  UnlinkFile(const char *lclPath);

int  VerifyAll(char *path);
char VerifyMP(const char *func, const char *path);

// The following are for runOld conversion purposes and will be removed
//
int          Convert();
int          ConvTest(int doNames, int doSpaces);
int          New2Old(int doNames, int doSpaces);
int          Old2New(int doNames, int doSpaces);
int          o2nFiles(XrdFrmFileset *sP, int &numOld);
int          o2nSpace(XrdFrmFileset *sP, const char *Space);
XrdOucTList *x2xPaths();
int          x2xRemove(const char *Type, const char *Path, int cvt=0);
XrdOucTList *x2xSpaces();

static const char *AuditHelp;
static const char *FindHelp;
static const char *HelpHelp;
static const char *MakeLFHelp;
static const char *MarkHelp;
static const char *MmapHelp;
static const char *PinHelp;
static const char *QueryHelp;
static const char *RelocHelp;
static const char *RemoveHelp;

// Frm agent/proxy control
//
XrdFrmProxy *frmProxy;
int          frmProxz;

// Command control
//
char    **ArgV;
char     *ArgS;
int       ArgC;

// The following are common variables for audit functions
//
long long numBytes;
long long numBLost;
int       numDirs;
int       numFiles;
int       numProb;
int       numFix;
int       finalRC;

// Options from the command
//
struct {char   All;
        char   Echo;
        char   Erase;
        char   Fix;
        char   Force;
        char   Keep;
        char   ktAlways;
        char   ktIdle;
        char   Local;
        char   MPType;
        char   Recurse;
        char  *Args[2];
        uid_t  Uid;
        gid_t  Gid;
        time_t KeepTime;
       } Opt;
};
namespace XrdFrm
{
extern XrdFrmAdmin Admin;
}
#endif
