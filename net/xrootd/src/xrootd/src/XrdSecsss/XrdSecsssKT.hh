#ifndef __SecsssKT__
#define __SecsssKT__
/******************************************************************************/
/*                                                                            */
/*                        X r d S e c s s s K T . h h                         */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//       $Id$

#include <string.h>
#include <time.h>
#include "XrdSys/XrdSysPthread.hh"

class XrdOucErrInfo;
class XrdOucStream;

class XrdSecsssKT
{
public:

class ktEnt
{
public:

static const int maxKLen = 128;
static const int NameSZ  = 192;
static const int UserSZ  = 128;
static const int GrupSZ  = 64;

struct ktData
       {long long ID;
        time_t    Crt;
        time_t    Exp;
        int       Opts;
        int       Len;
        char      Val[maxKLen];// Key strings are 1024 bits or less
        char      Name[NameSZ];// Key names are null terminated
        char      User[UserSZ];// Usr names are null terminated
        char      Grup[GrupSZ];// Grp names are null terminated
       }          Data;

static const int anyUSR = 2;
static const int anyGRP = 4;
static const int usrGRP = 8;

       void      NUG(ktEnt *ktP) {strcpy(Data.Name, ktP->Data.Name);
                                  strcpy(Data.User, ktP->Data.User);
                                  strcpy(Data.Grup, ktP->Data.Grup);
                                 }
       void      Set(ktEnt &rhs) {Data.ID=rhs.Data.ID; Data.Len = rhs.Data.Len;
                                  memcpy(Data.Val, rhs.Data.Val, Data.Len);
                                  Data.Crt=rhs.Data.Crt; Data.Exp=rhs.Data.Exp;
                                 }
       ktEnt    *Next;

       ktEnt() : Next(0) { Data.ID   = -1;   Data.Opts = 0;
                          *Data.Val = '\0'; *Data.Name = '\0';
                          *Data.User= '\0'; *Data.Grup = '\0';
                         }
      ~ktEnt() {}
};

void   addKey(ktEnt &ktNew);

int    delKey(ktEnt &ktDel);

static
char  *genFN();

static
void   genKey(char *Buff, int blen);

int    getKey(ktEnt &ktEql);

ktEnt *keyList() {return ktList;}

void   Refresh();

time_t RefrTime() {return ktRefT;}

int    Rewrite(int Keep, int &numKeys, int &numTot, int &numExp);

int    Same(const char *path) {return (ktPath && !strcmp(ktPath, path));}

void   setPath(const char *Path) 
              {if (ktPath) free(ktPath); ktPath = strdup(Path);}

enum   xMode {isAdmin = 0, isClient, isServer};

       XrdSecsssKT(XrdOucErrInfo *, const char *, xMode, int refr=60*60);
      ~XrdSecsssKT();

private:
int    eMsg(const char *epn, int rc, const char *txt1,
            const char *txt2=0, const char *txt3=0, const char *txt4=0);
ktEnt *getKeyTab(XrdOucErrInfo *eInfo, time_t Mtime, mode_t Amode);
mode_t fileMode(const char *Path);
int    isKey(ktEnt &ktRef, ktEnt *ktP, int Full=1);
void   keyB2X(ktEnt *theKT, char *buff);
void   keyX2B(ktEnt *theKT, char *xKey);
ktEnt *ktDecode0(XrdOucStream &kTab, XrdOucErrInfo *eInfo);

XrdSysMutex myMutex;
char       *ktPath;
ktEnt      *ktList;
time_t      ktMtime;
xMode       ktMode;
time_t      ktRefT;
int         kthiID;
static int  randFD;
};
#endif
