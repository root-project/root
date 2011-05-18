#ifndef _OSS_SPACE_H
#define _OSS_SPACE_H
/******************************************************************************/
/*                                                                            */
/*                        X r d O s s S p a c e . h h                         */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

class XrdSysError;

class XrdOssSpace
{
public:

       enum       sType {Serv = 0, Pstg = 1, Purg = 2, Admin = 3,
                         RsvA = 4, RsvB = 5, RsvC = 6, addT  = 7,
                         Totn = 8};

static const int  maxSNlen = 63;  // Maximum space name length (+1 for null)
static const int  minSNbsz = 64;

static void       Adjust(int Gent,          off_t Space, sType=Serv);

static void       Adjust(const char *GName, off_t Space, sType=Serv);

static int        Assign(const char *GName, long long &bytesUsed);

static const int  haveUsage = 1;
static const int  haveQuota = 2;

static int        Init(); // Return the "or" of havexxxx (above)

static int        Init(const char *aPath, const char *qFile, int isSOL);

static int        Quotas();

static int        Readjust();

static void       Refresh();

static int        Unassign(const char *GName);

static long long  Usage(int gent) {return (gent < 0 || gent >= maxEnt
                                   ? 0 : uData[gent].Bytes[Serv]);}

                  XrdOssSpace() {}  // Everything is static
                 ~XrdOssSpace() {}  // Never gets deleted

struct uEnt {char      gName[minSNbsz];
             long long Bytes[Totn]; // One of sType, above
            };

static long long  Usage(const char *GName, struct uEnt &uVal, int rrd=0);

private:
static int    findEnt(const char *GName);
static int    Readjust(int);
static int    UsageLock(int Dolock=1);

static const int ULen   = sizeof(long long);
static const int DataSz = 16384;
static const int maxEnt = DataSz/sizeof(uEnt);

static const char *qFname;
static const char *uFname;
static uEnt        uData[maxEnt];
static short       uDvec[maxEnt];
static time_t      lastMtime;
static int         fencEnt;
static int         freeEnt;
static int         aFD;
static int         Solitary;
};
#endif
