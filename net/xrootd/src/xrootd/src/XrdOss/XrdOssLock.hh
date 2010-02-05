#ifndef __XRDOSS_LOCK__
#define __XRDOSS_LOCK__
/******************************************************************************/
/*                                                                            */
/*                         X r d O s s L o c k . h h                          */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

class XrdOssLock
{
public:

int Serialize(const char *, int);
int Serialize(int lkFD, int Opt) {return XLock(lkFD, Opt);}
int NoSerialize(const char *, int);
int ReSerialize(const char *, const char *);
int UnSerialize(int opts=0);

    XrdOssLock() {lkfd = -1;}
   ~XrdOssLock() {if (lkfd >= 0) UnSerialize();}

private:

int lkfd;      // Lock file handle

int XLock(int, int);
int Build_LKFN(char *, int, const char *, int);
};

/******************************************************************************/
/*                       C a l l i n g   O p t i o n s                        */
/******************************************************************************/
  
// Defines when calling XrdOssSerialize() and XrdOssUnSerialize()
//
#define XrdOssLEAVE    0x00000001
#define XrdOssRETRY    0x00000002
#define XrdOssREGRADE  0x00000004
#define XrdOssNOWAIT   0x00000008
#define XrdOssEXC      0x00000010
#define XrdOssSHR      0x00000020
#define XrdOssDIR      0x00000040
#define XrdOssFILE     0x00000080
#define XrdOssRETIME   0x00000100

#endif
