#ifndef __XRDCnsSSI_H_
#define __XRDCnsSSI_H_
/******************************************************************************/
/*                                                                            */
/*                          X r d C n s S s i . h h                           */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

class  XrdCnsSsiDRec;
class  XrdCnsSsiFRec;
struct iovec;

class XrdCnsSsi
{
public:

static int List(const char *Host, const char *Path);

static int Updt(const char *Host, const char *Path);

static int Write(int xFD, struct iovec *iov, int n, int Bytes);

static int nErrs;
static int nDirs;
static int nFiles;

               XrdCnsSsi() {}
              ~XrdCnsSsi() {}

private:
static XrdCnsSsiDRec *AddDir(char *dP, char *lP);
static int            AddDel(char *pPo, char *lP);
static XrdCnsSsiFRec *AddFile(char *lfn,          char *lP);
static XrdCnsSsiFRec *AddFile(char *dP, char *fP, char *lP);
static void           AddSize(char *dP, char *fP, char *lP);
static int            ApplyLog(const char *Path);
static void           ApplyLogRec(char *Rec);
static void           FSize(char *oP, char *iP, int bsz);
static int            Write(int xFD, char *bP, int bL);
static int            Write(int xFD, int TOD, const char *Host);

};
#endif
