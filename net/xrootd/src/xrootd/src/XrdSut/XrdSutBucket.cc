// $Id$
/******************************************************************************/
/*                                                                            */
/*                      X r d S u t B u c k e t . c c                         */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <stdio.h>
#include <string.h>

#include <XrdOuc/XrdOucString.hh>
#include <XrdSut/XrdSutBucket.hh>
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
/*  Unit for information exchange                                             */
/*                                                                            */
/******************************************************************************/
//______________________________________________________________________________
XrdSutBucket::XrdSutBucket(char *bp, int sz, int ty)
{
   // Default constructor

   buffer = membuf = bp;
   size=sz;
   type=ty;
}

//______________________________________________________________________________
XrdSutBucket::XrdSutBucket(XrdOucString &s, int ty) 
{
   // Constructor

   membuf = 0;
   size = 0;
   type = ty;

   if (s.length()) {
       membuf = new char [s.length()];
       if (membuf) {
          memcpy(membuf,s.c_str(),s.length());
          buffer = membuf;
          size = s.length();
       }
   }
}

//______________________________________________________________________________
XrdSutBucket::XrdSutBucket(XrdSutBucket &b)
{
   // Copy constructor

   membuf = new char[b.size]; 
   if (membuf) {
      memcpy(membuf,b.buffer,b.size);
      buffer = membuf;
      type = b.type;
      size = b.size;
   }
}

//______________________________________________________________________________
void XrdSutBucket::Update(char *nb, int ns, int ty)
{
   // Update content 

   if (membuf) 
      delete[] membuf;
   buffer = membuf = nb;
   size = ns;

   if (ty)
      type = ty;
}

//______________________________________________________________________________
int XrdSutBucket::Update(XrdOucString &s, int ty)
{
   // Update content 
   // Returns 0 if ok, -1 otherwise.

   if (membuf)
      delete[] membuf;
   membuf = buffer = 0;
   if (s.length()) {
      membuf = new char [s.length()];
      if (membuf) {
         memcpy(membuf,s.c_str(),s.length());
         buffer = membuf;
         size = s.length();
         if (ty)
            type = ty;
         return 0;
      }
   }
   return -1;
}

//______________________________________________________________________________
int XrdSutBucket::SetBuf(const char *nb, int ns)
{
   // Fill local buffer with ns bytes at nb.
   // Memory is properly allocated / deallocated
   // Returns 0 if ok, -1 otherwise.

   if (membuf)
      delete[] membuf;
   size = 0;
   membuf = buffer = 0;
   if (nb && ns) {
      membuf = new char [ns];
      if (membuf) {
         memcpy(membuf,nb,ns);
         buffer = membuf;
         size = ns;
         return 0;
      }
   }
   return -1;
}

//______________________________________________________________________________
void XrdSutBucket::ToString(XrdOucString &s)
{
   // Convert content into a null terminated string
   // (nb: the caller must be sure that the operation makes sense)

   s = "";
   char *b = new char[size+1];
   if (b) {
      memcpy(b,buffer,size);
      b[size] = 0;
      s = (const char *)b;
      delete[] b;
   }
}

//_____________________________________________________________________________
void XrdSutBucket::Dump(int opt)
{
   // Dump content of bucket
   // Options:
   //             1    print header and tail (default)
   //             0    dump only content
   EPNAME("Bucket::Dump");

   if (opt == 1) {
      PRINT("//-------------------------------------------------//");
      PRINT("//                                                 //");
      PRINT("//             XrdSutBucket DUMP                   //");
      PRINT("//                                                 //");
   }

   PRINT("//  addr: " <<this);
   PRINT("//  type: " <<type<<" ("<<XrdSutBuckStr(type)<<")");
   PRINT("//  size: " <<size <<" bytes");
   PRINT("//  content:");
   char btmp[XrdSutPRINTLEN] = {0};
   unsigned int nby = size;
   unsigned int k = 0, cur = 0;
   unsigned char i = 0, j = 0, l = 0;
   for (k = 0; k < nby && cur < sizeof(btmp) - 6; k++) {
      i = (int)buffer[k];
      j = i / 32;
      l = i - j * 32;
      if ((XrdSutCharMsk[3][j] & (1 << l)) || i == 0x20) {
         btmp[cur] = i;
         cur++;
      } else {
         char chex[8];
         sprintf(chex,"'0x%x'",(int)(i & 0x7F));
         sprintf(btmp,"%s%s",btmp,chex);
         cur += strlen(chex);
      }
      if (cur > XrdSutPRINTLEN - 10) {
         btmp[cur] = 0;
         PRINT("//    " <<btmp);
         memset(btmp,0,sizeof(btmp));
         cur = 0;
      }
   }
   PRINT("//    " <<btmp);

   if (opt == 1) {
      PRINT("//                                                 //");
      PRINT("//  NB: '0x..' is the hex of non-printable chars   //");
      PRINT("//                                                 //");
      PRINT("//-------------------------------------------------//");
   }
}

//______________________________________________________________________________
int XrdSutBucket::operator==(const XrdSutBucket &b)
{
   // Compare bucket b to local bucket: return 1 if matches, 0 if not

   if (b.size == size)
      if (!memcmp(buffer,b.buffer,size))
         return 1;
   return 0;
}
