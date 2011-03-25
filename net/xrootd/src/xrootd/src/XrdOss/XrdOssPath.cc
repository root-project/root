/******************************************************************************/
/*                                                                            */
/*                         X r d O s s P a t h . c c                          */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/param.h>

#include "XrdOss/XrdOssPath.hh"
#include "XrdOss/XrdOssSpace.hh"
#include "XrdSys/XrdSysPthread.hh"

/******************************************************************************/
/*                          L o c a l   M a c r o s                           */
/******************************************************************************/

#define XrdOssTAMP(dst, src) \
   while(*src) {*dst = (*src == '/' ? xChar : *src); src++; dst++;}; *dst='\0'
  
/******************************************************************************/
/*                        S t a t i c   O b j e c t s                         */
/******************************************************************************/
  
char   XrdOssPath::h2c[16] = {'0','1','2','3','4','5','6','7',
                              '8','9','A','B','C','D','E','F'};

const char XrdOssPath::xChar;

// The initialization must be in 1-to-1 order with theSfx enum!
//
const char *XrdOssPath::Sfx[XrdOssPath::sfxNum] =
                      {".anew", ".fail",  ".lock", ".pin",
                       ".mmap", ".mkeep", ".mlock",".pfn", 0};

/******************************************************************************/
/*                               C o n v e r t                                */
/******************************************************************************/
  
int XrdOssPath::Convert(char *dst, int dln, const char *oldP, const char *newP)
{
   int i;

// Copy all the char up to the first % into the new link buffer
//
   for (i = 0; oldP[i] && oldP[i] != xChar; i++) *dst++ = oldP[i];

// Now make sure the resulting name will fit
//
   if ((i + static_cast<int>(strlen(newP))) >= dln) return -ENAMETOOLONG;

// Complete the name
//
   XrdOssTAMP(dst, newP);
   return 0;
}
  
/******************************************************************************/
/*                               E x t r a c t                                */
/******************************************************************************/
  
const char *XrdOssPath::Extract(char *path, char *lbuf, int &lbsz)
{
   struct stat Stat;
   char *pP;
   int j, lnklen = 0;

// If path is 0, the caller already has read the link; else read it.
//
        if (!path) lnklen = lbsz;
   else if (!lstat(path, &Stat) && S_ISLNK(Stat.st_mode))
           {if ((lnklen = readlink(path, lbuf, lbsz-1)) < 0) *lbuf = 0;
               else *(lbuf+lnklen) = 0;
           }
   else {lnklen = strlen(path);
         if (lnklen >= lbsz) lnklen = lbsz-1;
         strncpy(lbuf, path, lnklen); *(lbuf+lnklen) = '\0';
        }

// Extract out the cache group name from "<path>/cgroup/nn/fn" if possible
//
   if (lnklen >= 4 && lbuf[lnklen-1] == xChar && (pP=posCname(lbuf,lnklen,j)))
      {*(pP+j) = '\0';
       if (pP != lbuf) *(pP-1) = '\0';
       return pP;
      }

// This is not a normal cache, so do something reasonable
//
   if ((pP = index(lbuf, xChar))) *pP = '\0';
      else *(lbuf+1) = '\0';
   return "public";
}

/******************************************************************************/
/*                               g e n P a t h                                */
/******************************************************************************/
  
char *XrdOssPath::genPath(const char *inPath, const char *cgrp, char *sfx)
{
   char *dirP, cgbuff[XrdOssSpace::minSNbsz], pbuff[MAXPATHLEN+64], *pP = pbuff;
   int n;

// Check if the group name is already in the path
//
   strcpy(pbuff, inPath); strcpy(cgbuff, cgrp); strcat(cgbuff, "/");
   do {if ((dirP = index(pP, '/')) && !strcmp(dirP+1, cgbuff)) break;
       pP = dirP+1;
      } while(dirP);

// If the group name is not in the path, add the group name to it.
//
   if (!dirP)
      {dirP = pbuff + strlen(inPath) - 1;
       strcpy(dirP+1, cgbuff);
      }

// Construct a suffix that will allow us to quickly find the group name
// We rely on the fact that group names are typically less than 16 characters 
// and cache group paths are less than 240 characters
//
   if ((n = strlen(cgrp)) > 15) sfx[2] = 'f';
      else sfx[2] = h2c[n];
   n = (dirP - pbuff + 1) & 0xff;
   sfx[1] = h2c[(n & 0x0f)]; n = n >> 4; sfx[0] = h2c[(n & 0x0f)];
   sfx[3] = xChar;

// Return the path
//
   return strdup(pbuff);
}

/******************************************************************************/
/*                                g e n P F N                                 */
/******************************************************************************/
  
char *XrdOssPath::genPFN(fnInfo &Info, char *buff, int blen, const char *Path)
{
    static XrdSysMutex myMutex;
    static char pfnPfx[8];
    static int mySeq = Init(pfnPfx);
    union {int  bin;
           char chr[4];} Seq;
    char   *bP = buff;

// Construct old style path if we have a path present (i.e., non-xa path)
//
   if (Path)
      {const char *pap = Path;
       if ((Info.Plen + static_cast<int>(strlen(Path))) >= blen) 
           {*buff = '\0'; return 0;}
       memcpy(bP, Info.Path,  Info.Plen);
       bP += Info.Plen;
       XrdOssTAMP(bP, pap);
       Info.Slash = buff;
       return 0;
      }

// Increment the sequence number
//
   myMutex.Lock();
   Seq.bin = mySeq++;
   myMutex.UnLock();

// Construct the path (buffer will be big enough)
//
   memcpy(bP, Info.Path, Info.Plen);
   bP += Info.Plen;
   *bP++ = h2c[((Seq.bin>4) & 0x0f)];
   *bP++ = h2c[( Seq.bin    & 0x0f)];
   Info.Slash= bP;
   *bP++ = '/';
   memcpy(bP, pfnPfx, sizeof(pfnPfx));
   bP = bin2hex(Seq.chr, sizeof(Seq.chr), bP+sizeof(pfnPfx));
   memcpy(bP, Info.Sfx, sfxLen);
   bP += sfxLen;
   *bP = '\0';

// All done
//
   return bP;
}

/******************************************************************************/

char *XrdOssPath::genPFN(char *dst, int dln, const char *src)
{
   char *pP;

   if (!(pP = (char *) index(src, xChar))|| dln <= (int)strlen(pP)) return 0;

   while(*pP) {*dst++ = (*pP == xChar ? '/' : *pP); pP++;}

   *dst = '\0';
   return dst;
}

/******************************************************************************/
/*                              g e t C n a m e                               */
/******************************************************************************/
  
int  XrdOssPath::getCname(const char *path, char *Cache,
                                char *lbuf, int   lbsz)
{
   struct stat lbuff;
   char *xP, lnkbuff[MAXPATHLEN+64];
   int j, lnklen = 0;

// Set up local buffer or remote buffer
//
   if (!lbuf) {lbuf = lnkbuff; lbsz = MAXPATHLEN;}

// If path is 0, the caller already has read the link; else read it.
//
   if (!path) lnklen = lbsz;
      else if (!lstat(path, &lbuff) && S_ISLNK(lbuff.st_mode))
              lnklen = readlink(path, lbuf, lbsz);

// Check if the symlink references a new cache. If not then the cache group is
// always deemed to be public.
//
   if (lnklen < 4 || lbuf[lnklen-1] != xChar)
      {strcpy(Cache, "public"); return (lnklen < 0 ? 0 : lnklen);}

// Extract out the cache group name from "<path>/cgroup/nn/fn"
//
   if (!(xP = posCname(lbuf, lnklen, j))) strcpy(Cache, "public");
      else {strncpy(Cache, xP, j); *(Cache+j)='\0';}

// All done
//
   return lnklen;
}

/******************************************************************************/
/*                              p a t h T y p e                               */
/******************************************************************************/

XrdOssPath::theSfx XrdOssPath::pathType(const char *Path, int chkWhat)
{
   static const int chkMM = chkMem | chkMig;

   char  *Dot;
   int    i, iBeg, iEnd;

// Compute ending test
//
        if ( chkWhat & chkAll)           {iBeg = 0; iEnd = int(sfxLast);}
   else if ((chkWhat & chkMM ) == chkMM) {iBeg = 1; iEnd = int(sfxMemL);}
   else if ( chkWhat & chkMig)           {iBeg = 1; iEnd = int(sfxMigL);}
   else if ( chkWhat & chkMem) {iBeg = int(sfxMemF);iEnd = int(sfxMigL);}
   else                                  {iBeg = 0; iEnd = 0;}

// Convert path to suffix number
//
	if ((Dot = (char *) rindex(Path, '.')))
      for (i = iBeg; i < iEnd; i++) if (!strcmp(Dot,Sfx[i])) return theSfx(i+1);
   return isBase;
}
  
/******************************************************************************/
/*                             T r i m 2 B a s e                              */
/******************************************************************************/
  
void XrdOssPath::Trim2Base(char *eP)
{
   int oneMore = (*eP == xChar);

// Trim to the cache group name in "<path>/cgroup/nn/fn" or "<path>/fn"
//
   do {eP--;} while(*eP != '/');
   if (oneMore) do {eP--;} while(*eP != '/');
   *(eP+1) = '\0';
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                               b i n 2 h e x                                */
/******************************************************************************/
  
char *XrdOssPath::bin2hex(char *inbuff, int dlen, char *buff)
{
    int i;

    for (i = 0; i < dlen; i++) {
        *buff++ = h2c[(inbuff[i] >> 4) & 0x0f];
        *buff++ = h2c[ inbuff[i]       & 0x0f];
        }
     return buff;
}

/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/
  
int XrdOssPath::Init(char *pfnPfx)
{
   time_t theTime = time(0);
   union {int  binT;
          char chrT[4];} xTime;

// Generate the pfn prefix
//
   xTime.binT = static_cast<int>(theTime);
   bin2hex(xTime.chrT, sizeof(xTime.binT), pfnPfx);
   return 0;
}

/******************************************************************************/
/*                              p o s C n a m e                               */
/******************************************************************************/

char *XrdOssPath::posCname(char *lbuf, int lbsz, int &cnsz)
{
   char *eP, *xP = lbuf + lbsz - 4;
   long xCode;
   int k;

// Extract out the cache group name from "<path>/cgroup/nn/fn"
//
   if ((xCode = strtol(xP, &eP, 16)) && *eP == xChar
   &&  (cnsz = xCode & 0x0f) && (k = xCode>>4) && k < (lbsz-cnsz))
      {xP = lbuf + k;
       if (cnsz == 15 && *(xP+cnsz) != '/' && (eP = index(xP+cnsz,'/'))
       &&  eP-xP <= XrdOssSpace::maxSNlen) cnsz = eP - xP;
       return xP;
      }
   return 0;
}
