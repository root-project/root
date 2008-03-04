#ifndef __XRDCS2DCMFILE__
#define __XRDCS2DCMFILE__
/******************************************************************************/
/*                                                                            */
/*                      X r d C S 2 D C M F i l e . h h                       */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

class XrdCS2DCMFile
{
public:

void               Close() {if (fd >= 0) {close(fd); fd = -1;}}

static int         Create(const char *fn, char Mode, const char *pfn);

       int         Init(const char *thePath, time_t UpTime=0);

       char        Mode() {return *fileData;}

       int         Modified() {return isMod;}

static void        Modify(const char *thePath);

char              *Pfn()  {return fileData+1;}

unsigned long long reqID();

                   XrdCS2DCMFile() {fd = -1; myPath = "?"; lp = 0; isMod = 0;}
                  ~XrdCS2DCMFile() {Close();}

private:

char               fileData[4096];
const char         *myPath;
char               *lp, *np;
int                fd;
int                isMod;
};
#endif
