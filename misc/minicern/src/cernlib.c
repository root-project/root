//------------------------------------------------------------------------------
//
// This file contains the kernlib's package subset needed to build h2root.
// It cannot be used by any kernlib application because many kernlib
// functionalities * are missing.
//
//------------------------------------------------------------------------------

#ifdef WIN32

#include <io.h>
typedef long off_t;

#define cfclos_   __stdcall CFCLOS
#define cfget_    __stdcall CFGET
#define cfseek_   __stdcall CFSEEK
#define ishftr_   __stdcall ISHFTR
#define lshift_   __stdcall LSHIFT
#define vxinvb_   __stdcall VXINVB
#define vxinvc_   __stdcall VXINVC
#define cfopei_   __stdcall CFOPEI
#define cfstati_  __stdcall CFSTATI
#define lnblnk_   __stdcall LNBLNK

#else
#include <unistd.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>


char *fchtak(char *ftext, int lgtext)
{
      char *ptalc, *ptuse;
      char *utext;
      int  nalc;
      int  ntx, jcol;

      nalc  = lgtext + 8;
      ptalc = malloc (nalc);
      if (ptalc == NULL)     goto exit;
      utext = ftext;

      ptuse = ptalc;
      ntx   = lgtext;
      for (jcol = 0; jcol < ntx; jcol++)  *ptuse++ = *utext++;

      *ptuse = '\0';
exit: return  ptalc;
}

//------------------------------------------------------------------------------

unsigned int ishftr_(unsigned int *arg, int *len)
{
   return(*arg >> *len);
}

//------------------------------------------------------------------------------

unsigned int lshift_(unsigned int *arg, int *len)
{
   return(*arg << *len);
}

//------------------------------------------------------------------------------

void vxinvb_(int *ixv, int *n)
{
   int limit, jloop;
   int in;
   limit = *n;
   for (jloop = 0; jloop < limit; jloop++) { 
      in = ixv[jloop];
      ixv[jloop] =
            ((in >> 24) & 0x000000ff) |
            ((in >> 8) & 0x0000ff00) |
            ((in << 8) & 0x00ff0000) |
            ((in << 24) & 0xff000000);
   }
   return;
}

//------------------------------------------------------------------------------

void vxinvc_ (int *iv, int *ixv, int *n)
{
   int limit, jloop;
   int in;
   limit = *n;
   for (jloop = 0; jloop < limit; jloop++) {
      in = iv[jloop];
      ixv[jloop] =
      ((in >> 24) & 0x000000ff) |
      ((in >> 8) & 0x0000ff00) |
      ((in << 8) & 0x00ff0000) | ((in << 24) & 0xff000000);
   }
   return;
}

//------------------------------------------------------------------------------

void cfget_(int *lundes, int *medium, int *nwrec, int *nwtak, char *mbuf, 
            int *astat)
{
   int fildes;   
   int nbdn, nbdo;   

   if (medium) { }

   *astat = 0;   
   if (*nwtak <= 0) return;   
		     
   fildes = *lundes;   
   nbdo = *nwrec * 4;   
   nbdn = read (fildes, mbuf, nbdo);   
   if (nbdn == 0) goto heof;   
   if (nbdn < 0) goto herror;   
   *nwtak = (nbdn - 1) / 4 + 1;
   return;   
   heof:
      *astat = -1;
      return;
   herror:
      *astat = 0;
      printf ("error in CFGET\n");
      return;
}

//------------------------------------------------------------------------------

void cfseek_(int *lundes, int *medium, int *nwrec, int *jcrec, int *astat)
{
   int fildes;
   int nbdo;
   int isw;

   if (medium) { }

   fildes = *lundes;
   nbdo = *jcrec * *nwrec * 4;
   isw = lseek (fildes, nbdo, 0); 
   if (isw < 0) goto trouble;
   *astat = 0;
   return;

   trouble: 
      *astat = -1;
      printf("error in CFSEEK\n");  
}

//------------------------------------------------------------------------------

void cfclos_(int *lundes, int *medium)
{
   int fildes;
   if (medium) { }
   fildes = *lundes;
   close (fildes);
   return;
}

//------------------------------------------------------------------------------
#ifdef WIN32
int cfstati_(char *fname, int lfname, int *info, int *lgname)
#else
int cfstati_(char *fname, int *info, int *lgname)
#endif
{
   struct stat buf;
   char *ptname, *fchtak();
   int istat=-1, stat();
   ptname = fchtak(fname,*lgname);
   if (ptname == ((void *)0)) return -1;
   istat = stat(ptname, &buf);
   if (!istat) {
      info[0] = (int) buf.st_dev;
      info[1] = (int) buf.st_ino;
      info[2] = (int) buf.st_mode;
      info[3] = (int) buf.st_nlink;
      info[4] = (int) buf.st_uid;
      info[5] = (int) buf.st_gid;
      info[6] = (int) buf.st_size;
#if defined(__APPLE__) || defined(__FreeBSD__)
      info[7] = (int) buf.st_atimespec.tv_sec;
      info[8] = (int) buf.st_mtimespec.tv_sec;
      info[9] = (int) buf.st_ctimespec.tv_sec;
      info[10] = (int) buf.st_blksize;
      info[11] = (int) buf.st_blocks;
#elif defined(_AIX)
      info[7] = (int) buf.st_atime;
      info[8] = (int) buf.st_mtime;
      info[9] = (int) buf.st_ctime;
      info[10] = (int) buf.st_blksize;
      info[11] = (int) buf.st_blocks;
#elif defined(WIN32)
      info[7] = 0;
      info[8] = 0;
      info[9] = 0;
      info[10] = 0;
      info[11] = 0;
#else
      info[7] = (int) buf.st_atim.tv_sec;
      info[8] = (int) buf.st_mtim.tv_sec;
      info[9] = (int) buf.st_ctim.tv_sec;
      info[10] = (int) buf.st_blksize;
      info[11] = (int) buf.st_blocks;
#endif
   };
   free(ptname);
   return istat;
}

//------------------------------------------------------------------------------

int cfopen_perm = 0;
#ifdef WIN32
void cfopei_(int *lundes, int *medium, int *nwrec, int *mode, int *nbuf,
             char *ftext, int lftext, int *astat, int *lgtx)
#else
void cfopei_(int *lundes, int *medium, int *nwrec, int *mode, int *nbuf,
             char *ftext, int *astat, int *lgtx)
#endif
{
   char *pttext, *fchtak();
   int flags = 0;
   int fildes;
   int perm;
   if (nwrec || nbuf) { }
   *lundes = 0;
   *astat = -1;
   perm = cfopen_perm;
   cfopen_perm = 0;
   if (*medium == 1) goto fltp;
   if (*medium == 3) goto fltp;
   if (mode[0] == 0)
   {if (mode[1] == 0)
   flags = 00;
   else
   flags = 02;}
   else if (mode[0] == 1)
   {if (mode[1] == 0)
   flags = 01 | 0100 | 01000;
   else
   flags = 02 | 0100 | 01000;}
   else if (mode[0] == 2)
   {if (mode[1] == 0)
   flags = 01 | 0100 | 02000;
   else
   flags = 02 | 0100 | 02000;}
   goto act;
fltp:
   if (mode[0] == 0)
   {if (mode[1] == 0)
   flags = 00;
   else
   flags = 02;}
   else if (mode[0] == 1)
   {if (mode[1] == 0)
   flags = 01;
   else
   flags = 02;}
   else if (mode[0] == 2) return;
act: 
   pttext = fchtak(ftext,*lgtx);
   if (pttext == 0) return;
   if (perm == 0) perm = 0644;
   fildes = open (pttext, flags, perm);
   if (fildes < 0) goto errm;
   *lundes = fildes;
   *astat = 0;
   goto done;
errm: 
   *astat = 0;
   printf("error in CFOPEN\n");
done: 
   free(pttext);
   return;
}

//------------------------------------------------------------------------------

int lnblnk_ (char *chline, int len)
{
   char  *chcur;
   chcur = chline + len;
   while (chcur > chline) { if (*--chcur != ' ') goto exit; }
   return 0;
   exit: return chcur+1 - chline;
}

//------------------------------------------------------------------------------

