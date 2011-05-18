// $Id$

const char *XrdSutBuckeCVSID = "$Id$";
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
   { {0x00000000, 0xffffffff, 0xffffffff, 0xfffffffe},   // any printable char
     {0x00000000, 0x0000ffc0, 0x7fffffe0, 0x7fffffe0},   // letters/numbers  (up/low case)
     {0x00000000, 0x0000ffc0, 0x7e000000, 0x7e000000},   // hex characters   (up/low case)
     {0x00000000, 0x03ffc000, 0x07fffffe, 0x07fffffe} }; // crypt like [a-zA-Z0-9./]

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
      PRINT("//-----------------------------------------------------//");
      PRINT("//                                                     //");
      PRINT("//             XrdSutBucket DUMP                       //");
      PRINT("//                                                     //");
   }

   PRINT("//  addr: " <<this);
   PRINT("//  type: " <<type<<" ("<<XrdSutBuckStr(type)<<")");
   PRINT("//  size: " <<size <<" bytes");
   PRINT("//  content:");
   char bhex[XrdSutPRINTLEN] = {0};
   char bpri[XrdSutPRINTLEN] = {0};
   unsigned int nby = size;
   unsigned int k = 0, curhex = 0, curpri = 0;
   unsigned char i = 0, j = 0, l = 0;
   for (k = 0; k < nby; k++) {
      i = (unsigned char)buffer[k];
      bool isascii = (i > 127) ? 0 : 1;
      if (isascii) {
         j = i / 32;
         l = i - j * 32;
      }
      char chex[8];
      sprintf(chex," 0x%02x",(int)(i & 0xFF));
      sprintf(bhex,"%s%s",bhex,chex);
      curhex += strlen(chex);
      if (isascii && ((XrdSutCharMsk[0][j] & (1 << (31-l+1))) || i == 0x20)) {
         bpri[curpri] = i;
      } else {
         bpri[curpri] = '.';
      }
      curpri++;
      if (curpri > 7) {
         bhex[curhex] = 0;
         bpri[curpri] = 0;
         PRINT("// " <<bhex<<"    "<<bpri);
         memset(bhex,0,sizeof(bhex));
         memset(bpri,0,sizeof(bpri));
         curhex = 0;
         curpri = 0;
      }
   }
   bpri[curpri] = 0;
   if (curpri > 0) { 
      while (curpri++ < 8) {
         sprintf(bhex,"%s     ",bhex);
         curhex += 5;
      }
   }
   bhex[curhex] = 0;
   PRINT("// " <<bhex<<"    "<<bpri);

   if (opt == 1) {
      PRINT("//                                                     //");
      PRINT("//-----------------------------------------------------//");
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
