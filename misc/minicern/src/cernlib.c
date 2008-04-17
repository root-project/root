//------------------------------------------------------------------------------
//
// This file contains the kernlib's package subset needed to build h2root.
// It cannot be used by any kernlib application because many kernlib
// functionnalities * are missing.
//
//------------------------------------------------------------------------------

#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>


char *fchtak(ftext,lgtext)
      char *ftext;
      int  lgtext;
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

unsigned int ishftr_(arg,len)
unsigned int *arg;
int *len;
{
   return(*arg >> *len);
}

//------------------------------------------------------------------------------

void vxinvb_(ixv, n)
   int *ixv, *n;
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

void vxinvc_ (iv, ixv, n)
int *iv, *ixv, *n;
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

void cfget_(int *lundes, int *medium, int *nwrec, int *nwtak, char *mbuf, int *stat)
{
   int fildes;   
   int nbdn, nbdo;   

   if (medium) { }

   *stat = 0;   
   if (*nwtak <= 0) return;   
		     
   fildes = *lundes;   
   nbdo = *nwrec * 4;   
   nbdn = read (fildes, mbuf, nbdo);   
   if (nbdn == 0) goto heof;   
   if (nbdn < 0) goto herror;   
   *nwtak = (nbdn - 1) / 4 + 1;
   return;   
   heof:
      *stat = -1;
      return;
   herror:
      *stat = 0;
      printf ("error in CFGET\n");
      return;
}

//------------------------------------------------------------------------------

void cfseek_(int *lundes, int *medium, int *nwrec, int *jcrec, int *stat)
{
   int fildes;
   int nbdo;
   int isw;

   if (medium) { }

   fildes = *lundes;
   nbdo = *jcrec * *nwrec * 4;
   isw = lseek (fildes, nbdo, 0); 
   if (isw < 0) goto trouble;
   *stat = 0;
   return;

   trouble: 
      *stat = -1;
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

int cfstati_(fname, info, lgname)
char *fname;
int *lgname;
int *info;
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
#ifdef __APPLE__
      info[7] = (int) buf.st_atimespec.tv_sec;
      info[8] = (int) buf.st_mtimespec.tv_sec;
      info[9] = (int) buf.st_ctimespec.tv_sec;
#else
      info[7] = (int) buf.st_atim.tv_sec;
      info[8] = (int) buf.st_mtim.tv_sec;
      info[9] = (int) buf.st_ctim.tv_sec;
#endif
      info[10] = (int) buf.st_blksize;
      info[11] = (int) buf.st_blocks;
   };
   free(ptname);
   return istat;
}

//------------------------------------------------------------------------------

int cfopen_perm = 0;
void cfopei_(int *lundes,int *medium,int *nwrec,int *mode,int *nbuf,char *ftext,int *stat,int *lgtx)
{
   char *pttext, *fchtak();
   int flags = 0;
   int fildes;
   int perm;
   if (nwrec || nbuf) { }
   *lundes = 0;
   *stat = -1;
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
   *stat = 0;
   goto done;
errm: 
   *stat = 0;
   printf("error in CFOPEN\n");
done: 
   free(pttext);
   return;
}

//------------------------------------------------------------------------------

