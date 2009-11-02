// $Id$

const char *XrdSutRndmCVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*                        X r d S u t R n d m . c c                           */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <fcntl.h>
#include <errno.h>

#include <XrdOuc/XrdOucString.hh>
#include <XrdSut/XrdSutRndm.hh>
#include <XrdSut/XrdSutTrace.hh>

/******************************************************************************/
/*             M a s k s  f o r   A S C I I  c h a r a c t e r s              */
/******************************************************************************/
static kXR_int32 XrdSutCharMsk[4][4] =
   { {0x0, 0xffffff08, 0xafffffff, 0x2ffffffe}, // any printable char
     {0x0, 0x3ff0000, 0x7fffffe, 0x7fffffe},    // letters/numbers  (up/low case)
     {0x0, 0x3ff0000, 0x7e, 0x7e},              // hex characters   (up/low case)
     {0x0, 0x3ffc000, 0x7fffffe, 0x7fffffe} };  // crypt like [a-zA-Z0-9./]

/******************************************************************************/
/*                                                                            */
/*  Provider of random bunches of bits                                        */
/*                                                                            */
/******************************************************************************/

bool XrdSutRndm::fgInit = 0;

//______________________________________________________________________________
bool XrdSutRndm::Init(bool force)
{
   // Initialize the random machinery; try using /dev/urandom to avoid
   // hanging.
   // The bool 'force' can be used to force re-initialization.
   EPNAME("Rndm::Init");

   const char *randdev = "/dev/urandom";
   bool rc = 0;

   // We do not do it twice 
   if (fgInit && !force)
      return 1;

   int fd;
   unsigned int seed;
   if ((fd = open(randdev, O_RDONLY)) != -1) {
      DEBUG("taking seed from " <<randdev);
      if (read(fd, &seed, sizeof(seed)) == sizeof(seed)) rc = 1;
      close(fd);
   }
   if (rc == 0) {
      DEBUG(randdev  <<" not available: using time()");
      seed = time(0);   //better use times() + win32 equivalent
      rc = 1;
   }
   srand(seed);

   // Flag as initialized
   fgInit = 1;

   return rc;
}

//______________________________________________________________________________
int XrdSutRndm::GetString(const char *copt, int len, XrdOucString &str)
{
   // Static method to fill string str with len random characters.
   // Returns 0 if ok, -1 in case of error.
   // copt = "Any"      any printable char
   //        "LetNum"   letters and numbers  (upper and lower case)
   //        "Hex"      hex characters       (upper and lower case)
   //        "Crypt"    crypt like           [a-zA-Z0-9./]
   //
   // (opt is not case sensitive)

   int opt = 0;
   if (!strncasecmp(copt,"LetNum",6))
      opt = 1;
   else if (!strncasecmp(copt,"Hex",3))
      opt = 2;
   else if (!strncasecmp(copt,"Crypt",5))
      opt = 3;

   return XrdSutRndm::GetString(opt,len,str);
}

//______________________________________________________________________________
int XrdSutRndm::GetString(int opt, int len, XrdOucString &str)
{
   // Static method to fill string str with len random characters.
   // Returns 0 if ok, -1 in case of error.
   // opt = 0      any printable char
   //       1      letters and numbers  (upper and lower case)
   //       2      hex characters       (upper and lower case)
   //       3      crypt like           [a-zA-Z0-9./]
   EPNAME("Rndm::GetString");

   const char *cOpt[4] = { "Any", "LetNum", "Hex", "Crypt" };

   //  Default option 0
   if (opt < 0 || opt > 3) {
      opt = 0;
      DEBUG("unknown option: " <<opt <<": assume 0");
   }
   DEBUG("enter: len: " <<len <<" (type: " <<cOpt[opt] <<")");

   // Init Random machinery ... if needed
   if (!XrdSutRndm::fgInit)
      XrdSutRndm::fgInit = XrdSutRndm::Init();

   // randomize
   char *buf = new char[len+1];
   if (!buf) {
      errno = ENOSPC;
      return -1;
   }

   kXR_int32 k = 0;
   kXR_int32 i, j, l, m, frnd;
   while (k < len) {
      frnd = rand();
      for (m = 7; m < 32; m += 7) {
         i = 0x7F & (frnd >> m);
         j = i / 32;
         l = i - j * 32;
         if ((XrdSutCharMsk[opt][j] & (1 << l))) {
            buf[k] = i;
            k++;
         }
         if (k == len)
            break;
      }
   }

   // null terminated
   buf[len] = 0;
   DEBUG("got: " <<buf);

   // Fill output
   str = buf;
   delete[] buf;

   return 0;
}

//______________________________________________________________________________
char *XrdSutRndm::GetBuffer(int len, int opt)
{
   // Static method to fill randomly a buffer.
   // Returns the pointer to the buffer if ok, 0 in case of error.
   // If opt has one of the following values, the random bytes are
   // chosen between the corrsponding subset:
   // opt = 0      any printable char
   //       1      letters and numbers  (upper and lower case)
   //       2      hex characters       (upper and lower case)
   //       3      crypt like           [a-zA-Z0-9./]
   // The caller is responsible to destroy the buffer
   EPNAME("Rndm::GetBuffer");

   DEBUG("enter: len: " <<len);

   // Init Random machinery ... if needed
   if (!fgInit) {
      Init();
      fgInit = 1;
   }

   // randomize
   char *buf = new char[len];
   if (!buf) {
      errno = ENOSPC;
      return 0;
   }

   // Filtering ?
   bool filter = (opt >= 0 && opt <= 3);

   kXR_int32 k = 0;
   kXR_int32  i, m, frnd, j = 0, l = 0;
   while (k < len) {
      frnd = rand();
      for (m = 0; m < 32; m += 8) {
         i = 0xFF & (frnd >> m);
         bool keep = 1;
         if (filter) {
            j = i / 32;
            l = i - j * 32;
            keep = (XrdSutCharMsk[opt][j] & (1 << l));
         }
         if (keep) {
            buf[k] = i;
            k++;
         }
         if (k == len)
            break;
      }
   }

   return buf;
}

//______________________________________________________________________________
int XrdSutRndm::GetRndmTag(XrdOucString &rtag)
{
   // Static method generating a 64 bit random tag (8 chars in [a-zA-Z0-9./])
   // saved in rtag.
   // Return 0 in case of success; in case of error, -1 is returned
   // and errno set accordingly (see XrdSutRndm::GetString)

   return XrdSutRndm::GetString(3,8,rtag);
}


//______________________________________________________________________________
unsigned int XrdSutRndm::GetUInt()
{
   // Static method to return an unsigned int.

   // Init Random machinery ... if needed
   if (!fgInit) {
      Init();
      fgInit = 1;
   }

   // As simple as this
   return rand();
}
