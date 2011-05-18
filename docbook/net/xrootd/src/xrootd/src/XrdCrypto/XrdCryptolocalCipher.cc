// $Id$

const char *XrdCryptolocalCipherCVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*              X r d C r y p t o L o c a l C i p h e r . c c                 */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* Local implementation of XrdCryptoCipher based on PC1                       */
/*                                                                            */
/* ************************************************************************** */
#include <string.h>

#include <XrdSut/XrdSutRndm.hh>
#include <XrdCrypto/XrdCryptolocalCipher.hh>
#include <XrdCrypto/PC1.hh>
#include <XrdCrypto/PC3.hh>

// ---------------------------------------------------------------------------//
//
// Cipher interface
//
// ---------------------------------------------------------------------------//

//____________________________________________________________________________
XrdCryptolocalCipher::XrdCryptolocalCipher(const char *t, int l)
{
   // Main Constructor
   // Complete initialization of a cipher of type t and length l
   // The initialization vector is also created
   // Used to create ciphers

   valid = 0;
   bpub  = 0;
   bpriv = 0;

   // Check and set type
   int len = (l > 0 && l <= kPC1LENGTH) ? l : kPC1LENGTH ;

   // Generate and set a new key
   char *ktmp = XrdSutRndm::GetBuffer(len);
   if (ktmp) {
      // Store the key
      SetBuffer(len,ktmp);
      valid = 1;

      // Set also the key type (should be "PC1")
      if (!t || (t && !strcmp(t,"default")))
         SetType("PC1");
      else
         SetType(t);
   }
}

//____________________________________________________________________________
XrdCryptolocalCipher::XrdCryptolocalCipher(const char *t, int l, const char *k)
{
   // Constructor.
   // Initialize a cipher of type t and length l using the key at k
   // Used to import ciphers.

   valid = 0;
   bpub  = 0;
   bpriv = 0;

   // Check and set type
   int len = (l <= kPC1LENGTH) ? l : kPC1LENGTH ;

   if (k && len > 0) {
      // Set the key
      SetBuffer(len,k);

      valid = 1;

      // Set also the type
      if (!t || (t && !strcmp(t,"default")))
         SetType("PC1");
      else
         SetType(t);
   }
}

//____________________________________________________________________________
XrdCryptolocalCipher::XrdCryptolocalCipher(XrdSutBucket *bck)
{
   // Constructor from bucket.
   // Initialize a cipher of type t and length l using the key at k
   // Used to import ciphers.

   valid  = 0;
   bpub  = 0;
   bpriv = 0;

   if (bck && bck->size > 0) {
      valid = 1;
      char *pr = bck->buffer;
      kXR_int32 lbuf = 0;
      kXR_int32 ltyp = 0;
      kXR_int32 lpub = 0;
      kXR_int32 lpri = 0;
      memcpy(&lbuf,pr,sizeof(kXR_int32));
      pr += sizeof(kXR_int32);
      memcpy(&ltyp,pr,sizeof(kXR_int32));
      pr += sizeof(kXR_int32);
      memcpy(&lpub,pr,sizeof(kXR_int32));
      pr += sizeof(kXR_int32);
      memcpy(&lpri,pr,sizeof(kXR_int32));
      pr += sizeof(kXR_int32);
      // main buffer
      if (lbuf > 0) {
         char *buf = new char[lbuf];
         if (buf) {
            memcpy(buf,pr,lbuf);
            // Set the key
            SetBuffer(lbuf,buf);
            delete[] buf;
         } else
            valid = 0;
         pr += lbuf;
      }
      // type
      if (ltyp > 0) {
         char *buf = new char[ltyp+1];
         if (buf) {
            memcpy(buf,pr,ltyp);
            pr[ltyp] = 0;
            // Set the key
            SetType(buf);
            delete[] buf;
         } else
            valid = 0;
         pr += ltyp;
      }
      // bpub
      if (lpub > 0) {
         bpub = new uchar[lpub];
         if (bpub) {
            memcpy(bpub,pr,lpub);
         } else
            valid = 0;
         pr += lpub;
      }
      // bpriv
      if (lpri > 0) {
         bpriv = new uchar[lpri];
         if (bpriv) {
            memcpy(bpriv,pr,lpri);
         } else
            valid = 0;
         pr += lpri;
      }
   }
}

//____________________________________________________________________________
XrdCryptolocalCipher::XrdCryptolocalCipher(int bits, char *pub,
                                           int lpub, const char *t)
{
   // Constructor for key agreement.
   // Generates private + public parts. The public can be retrieved
   // using Public() to send to the counterpart.
   // The number of random bits to be used in 'bits'.
   // If pub is defined (with the public info of the counterpart)
   // finalizes the initialization by computing the cipher.
   // Sets also the name to 't', if defined (should be always 'PC1').
   // Used for key agreement.

   valid  = 0;
   bpub  = 0;
   bpriv = 0;
   lpub = kPC3SLEN;   

   //
   // Generate local info
   bpub = new uchar[kPC3SLEN];   
   if (bpub) {
      bpriv = new uchar[kPC3SLEN];   
      if (bpriv) {
         // at least 128 bits
         bits = (bits < kPC3MINBITS) ? kPC3MINBITS : bits; 
         // Generate the random passwd
         unsigned int lrpw = bits / 8 ;
         uchar *rpwd = (uchar *)XrdSutRndm::GetBuffer((int)lrpw);
         if (rpwd) {
            if (PC3InitDiPuk(rpwd, lrpw, bpub, bpriv) == 0)
               valid = 1;
            bpriv[kPC3SLEN-1] = 0;
            delete[] rpwd; rpwd = 0;
         } 
      }
   }
   if (!valid)
      Cleanup();
   //
   // If we are given already the counter part, we finalize
   // the operations
   if (valid && pub) {

      // Convert back from hex
      char *tpub = new char[strlen(pub)/2+2];
      int tlen = 0;
      if (tpub)
         XrdSutFromHex((const char *)pub, tpub, tlen);

      uchar *ktmp = new uchar[kPC3KEYLEN];   
      if (PC3DiPukExp((uchar *)tpub, bpriv, ktmp) == 0) {
         // Store the key
         SetBuffer(kPC3KEYLEN,(char *)ktmp);
         // Set also the key type (should be "PC1")
         if (!t || (t && !strcmp(t,"default")))
            SetType("PC1");
         else
            SetType(t);
      } else {
         valid = 0;
      }
   }   
}

//____________________________________________________________________________
XrdCryptolocalCipher::XrdCryptolocalCipher(const XrdCryptolocalCipher &c)
{
   // Copy Constructor

   valid = c.valid;
   // Copy buffer
   SetBuffer(c.Length(),c.Buffer());
   // Copy Type
   SetType(c.Type());
   // Copy Buffers for key agreement
   if (c.bpub) {
      bpub = new uchar[kPC3SLEN];   
      if (bpub)
         memcpy(bpub,c.bpub,kPC3SLEN);
      else
         valid = 0;
   }
   if (c.bpriv) {
      bpriv = new uchar[kPC3SLEN];   
      if (bpriv)
         memcpy(bpriv,c.bpriv,kPC3SLEN);
      else
         valid = 0;
   }
}

//____________________________________________________________________________
bool XrdCryptolocalCipher::Finalize(char *pub, int lpub, const char *t)
{
   // Final initialization for key agreement.
   // 'pub' is the buffer sent by teh counterpart.
   // The private part must be defined already.

   lpub = kPC3SLEN;   
   if (valid && bpriv && pub) {

      // Convert back from hex
      char *tpub = new char[strlen(pub)/2+2];
      int tlen = 0;
      if (tpub)
         XrdSutFromHex((const char *)pub, tpub, tlen);

      uchar *ktmp = new uchar[kPC3KEYLEN];   
      if (PC3DiPukExp((uchar *)tpub, bpriv, ktmp) == 0) {
         // Store the key
         SetBuffer(kPC3KEYLEN,(char *)ktmp);
         // Set also the key type (should be "PC1")
         if (!t || (t && !strcmp(t,"default")))
            SetType("PC1");
         else
            SetType(t);
         return 1;
      } else {
         valid = 0;
      }
   } else {
      valid = 0;
   }
   return 0;
}

//____________________________________________________________________________
void XrdCryptolocalCipher::Cleanup()
{
   // Cleanup temporary buffers used for key agreement

   if (bpub) delete[] bpub; bpub = 0;
   if (bpriv) delete[] bpriv; bpriv = 0;
}

//____________________________________________________________________________
char *XrdCryptolocalCipher::Public(int &lpub)
{
   // Return pointer to information to be sent to the 
   // counterpart during key agreement. The allocated buffer, of size
   // lpub, must be deleted by the caller.

   if (bpub) {
      char *pub = new char[2*(kPC3SLEN-1)+1];
      if (pub) {
         XrdSutToHex((const char *)bpub, kPC3SLEN-1, pub);;
         lpub = 2*(kPC3SLEN-1);
         return pub;
      }
   }

   // Not available
   lpub = 0;
   return (char *)0;
}

//_____________________________________________________________________________
XrdSutBucket *XrdCryptolocalCipher::AsBucket()
{
   // Return pointer to a bucket created using the internal information
   // serialized
   // The bucket is responsible for the allocated memory

   XrdSutBucket *buck = (XrdSutBucket *)0;

   if (valid) {

      // Serialize .. total length
      kXR_int32 lbuf = Length();
      kXR_int32 ltyp = Type() ? strlen(Type()) : 0;
      kXR_int32 lpub = bpub ? kPC3SLEN : 0;
      kXR_int32 lpri = bpriv ? kPC3SLEN : 0;
      int ltot = 4*sizeof(kXR_int32) + lpub + ltyp + lpub + lpri;
      char *newbuf = new char[ltot];
      if (newbuf) {
         int cur = 0;
         memcpy(newbuf+cur,&lbuf,sizeof(kXR_int32));
         cur += sizeof(kXR_int32);
         memcpy(newbuf+cur,&ltyp,sizeof(kXR_int32));
         cur += sizeof(kXR_int32);
         memcpy(newbuf+cur,&lpub,sizeof(kXR_int32));
         cur += sizeof(kXR_int32);
         memcpy(newbuf+cur,&lpri,sizeof(kXR_int32));
         cur += sizeof(kXR_int32);
         if (Buffer()) {
            memcpy(newbuf+cur,Buffer(),lbuf);
            cur += lbuf;
         }
         if (Type()) {
            memcpy(newbuf+cur,Type(),ltyp);
            cur += ltyp;
         }
         if (bpub) {
            memcpy(newbuf+cur,bpub,lpub);
            cur += lpub;
         }
         if (bpriv) {
            memcpy(newbuf+cur,bpriv,lpri);
            cur += lpri;
         }
         // The bucket now
         buck = new XrdSutBucket(newbuf,ltot,kXRS_cipher);
      }
   }

   return buck;
}

//____________________________________________________________________________
int XrdCryptolocalCipher::Encrypt(const char *in, int lin, char *out)
{
   // Encrypt lin bytes at in with local cipher.
   // The outbut buffer must be provided by the caller for at least
   // EncOutLength(lin) bytes.
   // Returns number of meaningful bytes in out, or 0 in case of problems

   return PC1Encrypt((const char *)in, lin,
                     (const char *)Buffer(), Length(), out);
}

//____________________________________________________________________________
int XrdCryptolocalCipher::Decrypt(const char *in, int lin, char *out)
{
   // Decrypt lin bytes at in with local cipher.
   // The outbut buffer must be provided by the caller for at least
   // DecOutLength(lin) bytes.
   // Returns number of meaningful bytes in out, or 0 in case of problems

   return PC1Decrypt((const char *)in, lin,
                     (const char *)Buffer(), Length(), out);
}

//____________________________________________________________________________
int XrdCryptolocalCipher::EncOutLength(int l)
{
   // Required buffer size for encrypting l bytes

   return (2*l);
}

//____________________________________________________________________________
int XrdCryptolocalCipher::DecOutLength(int l)
{
   // Required buffer size for decrypting l bytes

   return (l/2+1);
}


//____________________________________________________________________________
bool XrdCryptolocalCipher::IsDefaultLength() const
{
   // Returns true if cipher has the default length

   return Length() == kPC1LENGTH;
}
