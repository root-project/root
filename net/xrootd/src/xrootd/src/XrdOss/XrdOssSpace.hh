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

void       Adjust(int Gent, off_t Space);

int        Assign(const char *GName, long long &bytesUsed);

int        Init();

int        Quotas();

long long  Usage(int gent)
                {return (gent < 0 || gent >= nextEnt ? 0 : uData[gent].Used);}

           XrdOssSpace(const char *aPath, const char *qFile);
          ~XrdOssSpace() {}  // Never gets deleted

private:

struct uEnt {char      gName[16];
             long long Used;
             long long Purged;
             long long Reserved[4];
            };

static const int DataSz = 16384;
static const int maxEnt = DataSz/sizeof(uEnt);

       const char *QFile;

       const char *aFname;
       uEnt        uData[maxEnt];
       int         nextEnt;
       int         aFD;
       time_t      lastMtime;
};
#endif
