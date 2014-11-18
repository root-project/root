#ifndef __XRDPOSIXXROOTDPATH_HH__
#define __XRDPOSIXXROOTDPATH_HH__
/******************************************************************************/
/*                                                                            */
/*                 X r d P o s i x X r o o t d P a t h . h h                  */
/*                                                                            */
/* (c) 2011 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
class XrdPosixXrootPath
{
public:

void  CWD(const char *path);

char *URL(const char *path, char *buff, int blen);

      XrdPosixXrootPath();
     ~XrdPosixXrootPath();

private:

struct xpath 
       {struct xpath *next;
        const  char  *server;
               int    servln;
        const  char  *path;
               int    plen;
        const  char  *nath;
               int    nlen;

        xpath(struct xpath *cur,
              const   char *pServ,
              const   char *pPath,
              const   char *pNath) : next(cur),
                                     server(pServ),
                                     servln(strlen(pServ)),
                                     path(pPath),
                                     plen(strlen(pPath)),
                                     nath(pNath),
                                     nlen(pNath ? strlen(pNath) : 0) {}
       ~xpath() {}
       };

struct xpath *xplist;
char         *pBase;
char         *cwdPath;
int           cwdPlen;
};
#endif
