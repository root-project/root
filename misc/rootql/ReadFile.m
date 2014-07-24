/*
 *  ReadFile.m
 *  ROOTQL
 *
 *  Created by Fons Rademakers on 22/05/09.
 *  Copyright 2009 CERN. All rights reserved.
 *
 */

#include <stdio.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <stdlib.h>
#include <strings.h>
#include <errno.h>

#import <QuickLook/QuickLook.h>
#import <Cocoa/Cocoa.h>


#if defined(__i386__)
#   define R__BYTESWAP
#endif
#if defined(__x86_64__)
#   define R__BYTESWAP
#endif


struct FileHeader_t {
   long long   end;
   long long   seekFree;
   long long   seekInfo;
   long long   seekKeys;
   int         version;
   int         begin;
   int         compress;
   int         nbytesName;
   int         dateC;
   int         timeC;
   int         dateM;
   int         timeM;
   char        units;
   char       *uuid;
   char       *title;
   const char *name;
};


static void FromBufChar(char **buf, char *x)
{
   // Read a char from the buffer and advance the buffer.

   *x = **buf;
   *buf += 1;
}

static void FromBufShort(char **buf, short *x)
{
   // Read a short from the buffer and advance the buffer.

#ifdef R__BYTESWAP
   char *sw = (char *)x;
   sw[0] = (*buf)[1];
   sw[1] = (*buf)[0];
#else
   memcpy(x, *buf, sizeof(short));
#endif
   *buf += sizeof(short);
}

static void FromBufInt(char **buf, int *x)
{
   // Read an int from the buffer and advance the buffer.

#ifdef R__BYTESWAP
   char *sw = (char *)x;
   sw[0] = (*buf)[3];
   sw[1] = (*buf)[2];
   sw[2] = (*buf)[1];
   sw[3] = (*buf)[0];
#else
   memcpy(x, *buf, sizeof(int));
#endif
   *buf += sizeof(int);
}

static void FromBufLL(char **buf, long long *x)
{
   // Read a long long from the buffer and advance the buffer.

#ifdef R__BYTESWAP
   char *sw = (char *)x;
   sw[0] = (*buf)[7];
   sw[1] = (*buf)[6];
   sw[2] = (*buf)[5];
   sw[3] = (*buf)[4];
   sw[4] = (*buf)[3];
   sw[5] = (*buf)[2];
   sw[6] = (*buf)[1];
   sw[7] = (*buf)[0];
#else
   memcpy(x, *buf, sizeof(long long));
#endif
   *buf += sizeof(long long);
}

static void FromBufUUID(char **buf, char **uuid, int versiondir)
{
   // Read UUID from the buffer and return it as string in uuid.
   // Returned string must be freed by the caller.
   // We'll never come here if version < 2.

   unsigned int   timeLow;
   unsigned short version, timeMid, timeHiAndVersion;
   unsigned char  clockSeqHiAndReserved, clockSeqLow, node[6];

   if (versiondir > 2)
      FromBufShort(buf, (short*)&version);  //version
   FromBufInt(buf, (int*)&timeLow);
   FromBufShort(buf, (short*)&timeMid);
   FromBufShort(buf, (short*)&timeHiAndVersion);
   FromBufChar(buf, (char*)&clockSeqHiAndReserved);
   FromBufChar(buf, (char*)&clockSeqLow);
   int i;
   for (i = 0; i < 6; i++)
      FromBufChar(buf, (char*)&node[i]);

   *uuid = malloc(40);
   sprintf(*uuid, "%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x",
           timeLow, timeMid, timeHiAndVersion, clockSeqHiAndReserved,
           clockSeqLow, node[0], node[1], node[2], node[3], node[4],
           node[5]);
}

static void FromBufStr(char **buf, char **str)
{
   // Read string from the buffer and return it as string in str.
   // Returned string must be freed by the caller.

   unsigned char nwh;
   int           nchars;

   FromBufChar(buf, (char*)&nwh);
   if (nwh == 255)
      FromBufInt(buf, &nchars);
   else
      nchars = nwh;

   int i;
   *str = malloc(nchars+1);
   for (i = 0; i < nchars; i++)
      FromBufChar(buf, &(*str)[i]);
   (*str)[nchars] = '\0';
}

static void GetDateAndTime(unsigned int datetime, int *date, int *time)
{
   // Function that returns the date and time. The input is
   // in TDatime format (as obtained via TDatime::Get()).
   // Date is returned in the format 950223  February 23 1995.
   // Time is returned in the format 102459 10h 24m 59s.

   unsigned int year  = datetime>>26;
   unsigned int month = (datetime<<6)>>28;
   unsigned int day   = (datetime<<10)>>27;
   unsigned int hour  = (datetime<<15)>>27;
   unsigned int min   = (datetime<<20)>>26;
   unsigned int sec   = (datetime<<26)>>26;
   *date =  10000*(year+1995) + 100*month + day;
   *time =  10000*hour + 100*min + sec;
}

static int ReadBuffer(int fd, char *buffer, int len)
{
   ssize_t siz;
   while ((siz = read(fd, buffer, len)) < 0 && errno == EINTR)
      errno = 0;
   return (int)siz;
}

static int ReadHeader(int fd, struct FileHeader_t *fh, NSMutableString *html)
{
   // Read ROOT file header structure.
   // Returns -1 in case of error, 0 otherwise.

   const int len = 300;
   char *header = malloc(len);

   if (ReadBuffer(fd, header, len) != len) {
      free(header);
      return -1;
   }

   if (strncmp(header, "root", 4)) {
      free(header);
      return -1;
   }

   char *buffer = header + 4;  // skip the "root" file identifier

   FromBufInt(&buffer, &fh->version);
   FromBufInt(&buffer, &fh->begin);

   int dummy;
   long long dummyll;

   if (fh->version < 1000000) {
      // < 2GB file
      int seekfree, seekinfo;
      FromBufInt(&buffer, &dummy); fh->end = (long long) dummy;
      FromBufInt(&buffer, &seekfree); fh->seekFree = (long long) seekfree;
      FromBufInt(&buffer, &dummy); // nbytes free
      FromBufInt(&buffer, &dummy); // nfree
      FromBufInt(&buffer, &fh->nbytesName);
      FromBufChar(&buffer, &fh->units);
      FromBufInt(&buffer, &fh->compress);
      FromBufInt(&buffer, &seekinfo); fh->seekInfo = (long long) seekinfo;
      FromBufInt(&buffer, &dummy); // nbytes info
   } else {
      // > 2GB file
      FromBufLL(&buffer, &fh->end);
      FromBufLL(&buffer, &fh->seekFree);
      FromBufInt(&buffer, &dummy);  // nbytes free
      FromBufInt(&buffer, &dummy);  // nfree
      FromBufInt(&buffer, &fh->nbytesName);
      FromBufChar(&buffer, &fh->units);
      FromBufInt(&buffer, &fh->compress);
      FromBufLL(&buffer, &fh->seekInfo);
      FromBufInt(&buffer, &dummy);  // nbytes info
   }

   int nk = sizeof(int)+sizeof(short)+2*sizeof(int)+2*sizeof(short)+2*sizeof(int);
   int nbytes = fh->nbytesName + (22 + 2*sizeof(int) + 18);  //nbytesName + TDirectoryFile::Sizeof();
   if (fh->version >= 40000)
      nbytes += 12;

   if (nbytes + fh->begin > 300) {
      free(header);
      header = malloc(nbytes);
      lseek(fd, (off_t)fh->begin, SEEK_SET);
      if (ReadBuffer(fd, header, nbytes) != nbytes) {
         free(header);
         return -1;
      }
      buffer = header + fh->nbytesName;
   } else {
      buffer = header + fh->begin + fh->nbytesName;
      nk += fh->begin;
   }

   short dversion, versiondir;
   FromBufShort(&buffer, &dversion);
   versiondir = dversion%1000;
   FromBufInt(&buffer, &dummy);
   GetDateAndTime((unsigned int)dummy, &fh->dateC, &fh->timeC);
   FromBufInt(&buffer, &dummy);
   GetDateAndTime((unsigned int)dummy, &fh->dateM, &fh->timeM);
   FromBufInt(&buffer, &dummy);  // nbytes keys
   FromBufInt(&buffer, &dummy);  // nbytes name
   if (dversion > 1000) {
      FromBufLL(&buffer, &dummyll); // seek dir
      FromBufLL(&buffer, &dummyll); // seek parent
      FromBufLL(&buffer, &fh->seekKeys);
   } else {
      int skeys;
      FromBufInt(&buffer, &dummy); // seek dir
      FromBufInt(&buffer, &dummy); // seek parent
      FromBufInt(&buffer, &skeys); fh->seekKeys = (long long)skeys;
   }
   if (versiondir > 1)
      FromBufUUID(&buffer, &fh->uuid, versiondir);
   else
      fh->uuid = strdup("-");

   buffer = header + nk;
   char *str;
   FromBufStr(&buffer, &str); free(str);  // "TFile"
   FromBufStr(&buffer, &str); free(str); // orig filename
   FromBufStr(&buffer, &fh->title);

#ifdef DEBUG
   NSLog(@"ReadHeader: %s, version = %d, begin = %d, end = %lld, units = %hhd, compress = %d",
         fh->name, fh->version, fh->begin, fh->end, fh->units, fh->compress);
#endif

   [html appendFormat: @"<center><h3>%s</h3></center>\n", fh->name];
   [html appendString: @"<center>Header Summary</center><p>\n"];
   [html appendString: @"<table width=\"80%\" border=\"0\" cellpadding=\"2\" cellspacing=\"0\">\n"];
   [html appendFormat: @"<tr><td>Title:</td><td><b>%s</b></td></tr>\n", fh->title];
   [html appendFormat: @"<tr><td>Creation date:</td><td><b>%d/%d</b></td></tr>\n", fh->dateC, fh->timeC];
   [html appendFormat: @"<tr><td>Modification date:</td><td><b>%d/%d</b></td></tr>\n", fh->dateM, fh->timeM];
   [html appendFormat: @"<tr><td>File size (bytes):</td><td><b>%lld</b></td></tr>\n", fh->end];
   [html appendFormat: @"<tr><td>Compression level:</td><td><b>%d</b></td></tr>\n", fh->compress];
   [html appendFormat: @"<tr><td>UUID:</td><td><b>%s</b></td></tr>\n", fh->uuid];
   [html appendFormat: @"<tr><td>File version:</td><td><b>%d</b></td></tr>\n", fh->version];
   [html appendString: @"</table>\n"];

   free(header);

   return 0;
}

static int ReadKeys(int fd, struct FileHeader_t *fh, NSMutableString *html, QLPreviewRequestRef preview)
{
   // Loop over all keys and print information.
   // Returns -1 in case of error, 0 otherwise.

   int    nbytes;
   int    objlen;
   int    date;
   int    time;
   short  keylen;
   short  cycle;
   char  *classname;
   char  *name;
   char  *title;

   const int len = 256;
   int   nread, datime;
   short versionkey;
   char *header, *buffer;

   NSDate *startDate = [NSDate date];

   long long idcur = fh->begin;

   [html appendString: @"<br>\n"];
   [html appendString: @"<table width=\"100%\" border=\"1\" cellpadding=\"2\" cellspacing=\"0\">\n"];
   [html appendString: @"<tr>\n"];
   [html appendString: @"<th colspan=\"7\">List of Keys</th>\n"];
   [html appendString: @"</tr>\n"];
   [html appendString: @"<tr>\n"];
   [html appendString: @"<th>Name</th>\n"];
   [html appendString: @"<th>Title</th>\n"];
   [html appendString: @"<th>Class</th>\n"];
   [html appendString: @"<th>Date</th>\n"];
   [html appendString: @"<th>Offset</th>\n"];
   [html appendString: @"<th>Size</th>\n"];
   [html appendString: @"<th>CX</th>\n"];
   [html appendString: @"</tr>\n"];

   nread = len;
   while (idcur < fh->end) {
   again:
      header = malloc(nread);
      lseek(fd, (off_t)idcur, SEEK_SET);
      if (idcur+nread > fh->end) nread = fh->end-idcur-1;
      if (ReadBuffer(fd, header, nread) != nread) {
         free(header);
         return -1;
      }
      buffer = header;
      FromBufInt(&buffer, &nbytes);
      if (!nbytes) {
         [html appendString: @"<tr>\n"];
         [html appendString: @"<td colspan=\"7\" style=\"color:red\"><center><b>ERROR</b></center></td>\n"];
         [html appendString: @"</tr>\n"];
         free(header);
         break;
      }
      if (nbytes < 0) {
         [html appendString: @"<tr>\n"];
         [html appendString: @"<td colspan=\"4\" style=\"color:red\"><center><b>GAP</b></center></td>\n"];
         [html appendFormat: @"<td>%lld</td>\n", idcur];
         [html appendFormat: @"<td>%d</td>\n", -nbytes];
         [html appendFormat: @"<td>%5.2f</td>\n", 1.0];
         [html appendString: @"<tr>\n"];
         free(header);
         idcur -= nbytes;
         continue;
      }

      FromBufShort(&buffer, &versionkey);
      FromBufInt(&buffer, &objlen);
      FromBufInt(&buffer, &datime);
      GetDateAndTime((unsigned int)datime, &date, &time);
      FromBufShort(&buffer, &keylen);
      FromBufShort(&buffer, &cycle);
      if (versionkey > 1000) {
         long long dummyll;
         FromBufLL(&buffer, &dummyll); // seekkey
         FromBufLL(&buffer, &dummyll); // seekpdir
      } else {
         int dummy;
         FromBufInt(&buffer, &dummy); // seekkey
         FromBufInt(&buffer, &dummy); // seekpdir
      }
      if (keylen > nread) {
         free(header);
         nread = keylen;
         goto again;
      }
      FromBufStr(&buffer, &classname);
      FromBufStr(&buffer, &name);
      FromBufStr(&buffer, &title);
      if (idcur == fh->seekFree) {
         free(classname);
         classname = strdup("FreeSegments");
         name[0] = '\0';
         title[0] = '\0';
      }
      if (idcur == fh->seekInfo) {
         free(classname);
         classname = strdup("StreamerInfo");
         name[0] = '\0';
         title[0] = '\0';
      }
      if (idcur == fh->seekKeys) {
         free(classname);
         classname = strdup("KeysList");
         name[0] = '\0';
         title[0] = '\0';
      }
      float cx;
      if (objlen != nbytes-keylen)
         cx = (float)(objlen+keylen)/(float)nbytes;
      else
         cx = 1.0;

      [html appendString: @"<tr>\n"];
      [html appendFormat: @"<td>%s</td>\n", name];
      [html appendFormat: @"<td>%s</td>\n", title];
      [html appendFormat: @"<td>%s</td>\n", classname];
      [html appendFormat: @"<td>%d/%d</td>\n", date, time];
      [html appendFormat: @"<td>%lld</td>\n", idcur];
      [html appendFormat: @"<td>%d</td>\n", nbytes];
      [html appendFormat: @"<td>%5.2f</td>\n", cx];
      [html appendString: @"</tr>\n"];

      free(classname);
      free(name);
      free(title);
      free(header);

      nread = len;

      idcur += nbytes;

      if ([startDate timeIntervalSinceNow] < -0.1) {
         // Check for cancel once per second
#ifdef DEBUG
         NSLog(@"ReadKeys: checking for cancel %.3f", [startDate timeIntervalSinceNow]);
#endif
         if (QLPreviewRequestIsCancelled(preview)) {
#ifdef DEBUG
            NSLog(@"ReadKeys: cancelled");
#endif
            return -1;
         }
         startDate = [startDate addTimeInterval: 0.1];
      }
   }

   [html appendString: @"</table>\n"];

   return 0;
}

int ReadFile(NSString *fullPath, NSMutableString *html, QLPreviewRequestRef preview)
{
   // Read ROOT file structure for specified file.
   // Returns -1 in case of error, 0 otherwise.

   struct FileHeader_t fh;
   fh.name = [fullPath UTF8String];
   int fd = open(fh.name, O_RDONLY, 0644);
   if (fd == -1)
      return -1;

   if (ReadHeader(fd, &fh, html) == -1) {
      close(fd);
      return -1;
   }

   // Check for cancel
   if (QLPreviewRequestIsCancelled(preview)) {
      free(fh.uuid);
      free(fh.title);
      close(fd);
      return -1;
   }

   if (ReadKeys(fd, &fh, html, preview) == -1) {
      free(fh.uuid);
      free(fh.title);
      close(fd);
      return -1;
   }

   free(fh.uuid);
   free(fh.title);

   close(fd);

   return 0;
}
