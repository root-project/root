#ifndef _OSS_PATH_H
#define _OSS_PATH_H
/******************************************************************************/
/*                                                                            */
/*                         X r d O s s P a t h . h h                          */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

class XrdOssPath
{
public:

struct fnInfo
{
const char *Path;
const char *Sfx; // Of length sfxLen!
const char *User;
      char *Slash;
      int   Plen;
};

static const char xChar  = '%';

static const int  sfxLen = 4;

static int   Convert(char *dst, int dln, const char *oldP, const char *newP);

static char *genPath(const char *inPath, const char *cgrp, char *sfx);

static char *genPFN(fnInfo &Info, char *buff, int blen, const char *Path=0);

static void  getCname(const char *path, char *Cache);

static void  Trim2Base(char *eP);

             XrdOssPath() {}
            ~XrdOssPath() {}

private:
static char *bin2hex(char *inbuff, int dlen, char *buff);
static int   Init(char *pfnPfx);

static char h2c[16];
};
#endif
