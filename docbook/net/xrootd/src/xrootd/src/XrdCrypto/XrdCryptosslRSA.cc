// $Id$

const char *XrdCryptosslRSACVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*                   X r d C r y p t o S s l R S A . c c                      */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* OpenSSL implementation of XrdCryptoRSA                                     */
/*                                                                            */
/* ************************************************************************** */

#include <XrdSut/XrdSutRndm.hh>
#include <XrdCrypto/XrdCryptosslTrace.hh>
#include <XrdCrypto/XrdCryptosslRSA.hh>

#include <string.h>

#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/pem.h>

//_____________________________________________________________________________
XrdCryptosslRSA::XrdCryptosslRSA(int bits, int exp)
{
   // Constructor
   // Generate a RSA asymmetric key pair
   // Length will be 'bits' bits (min 512, default 1024), public
   // exponent `pubex` (default 65537).
   EPNAME("RSA::XrdCryptosslRSA");

   publen = -1;
   prilen = -1;

   // Create container, first
   if (!(fEVP = EVP_PKEY_new())) {
      DEBUG("cannot allocate new public key container");
      return;
   }

   // Minimum is XrdCryptoMinRSABits
   bits = (bits >= XrdCryptoMinRSABits) ? bits : XrdCryptoMinRSABits;

   // If pubex is not odd, use default
   if (!(exp & 1<<1))
      exp = XrdCryptoDefRSAExp;   // 65537 (0x10001)

   DEBUG("bits: "<<bits<<", exp:"<<exp);

   // Try Key Generation
   RSA *fRSA = RSA_generate_key(bits,exp,0,0);

   // Update status flag
   if (fRSA) {
      if (RSA_check_key(fRSA) != 0) {
         status = kComplete;
         DEBUG("basic length: "<<RSA_size(fRSA)<<" bytes");
         // Set the key
         EVP_PKEY_set1_RSA(fEVP, fRSA);
      } else {
         DEBUG("WARNING: generated key is invalid");
         // Generated an invalid key: cleanup
         RSA_free(fRSA);
      }
   }
}

//_____________________________________________________________________________
XrdCryptosslRSA::XrdCryptosslRSA(const char *pub, int lpub)
{
   // Constructor
   // Allocate a RSA key pair and fill the public part importing 
   // from string representation (pub) to internal representation.
   // If lpub>0 use the first lpub bytes; otherwise use strlen(pub)
   // bytes.

   fEVP = 0;
   publen = -1;
   prilen = -1;

   // Import key
   ImportPublic(pub,lpub);
}

//_____________________________________________________________________________
XrdCryptosslRSA::XrdCryptosslRSA(EVP_PKEY *key, bool check)
{
   // Constructor to import existing key
   EPNAME("RSA::XrdCryptosslRSA_key");

   fEVP = 0;
   publen = -1;
   prilen = -1;

   // Create container, first
   if (!key) {
      DEBUG("no input key");
      return;
   }

   if (check) {
      // Check consistency
      if (RSA_check_key(key->pkey.rsa) != 0) {
         fEVP = key;
         // Update status
         status = kComplete;
      } else {
         DEBUG("key contains inconsistent information");
      }
   } else {
      // Accept in any case (for incomplete keys)
      fEVP = key;
      // Update status
      status = kPublic;
   }
}


//____________________________________________________________________________
XrdCryptosslRSA::XrdCryptosslRSA(const XrdCryptosslRSA &r)
{
   // Copy Constructor
   EPNAME("RSA::XrdCryptosslRSA_copy");

   fEVP = 0;
   publen = -1;
   prilen = -1;

   if (!r.fEVP) {
      DEBUG("input key is empty");
      return;
   }

   // If the given key is set, copy it via a bio
   bool publiconly = (r.fEVP->pkey.rsa->d == 0);
   //
   // Bio for exporting the pub key
   BIO *bcpy = BIO_new(BIO_s_mem());
   if (bcpy) {
      // Write kref public key to BIO
      if (PEM_write_bio_PUBKEY(bcpy, r.fEVP)) {
         bool ok = (publiconly) ? 1 :
                   // Write kref private key to BIO
                   (PEM_write_bio_PrivateKey(bcpy,r.fEVP,0,0,0,0,0) != 0);
         if (ok) {
            // Read public key from BIO
            if ((fEVP = PEM_read_bio_PUBKEY(bcpy, 0, 0, 0))) {
               // Update status
               status = kPublic;
               ok = (publiconly) ? 1 :
                    // Read private key from BIO
                    (PEM_read_bio_PrivateKey(bcpy,&fEVP,0,0) != 0);
               if (ok) {
                  // Check consistency
                  if (!publiconly && RSA_check_key(fEVP->pkey.rsa) != 0) {
                     // Update status
                     status = kComplete;
                  }
               }
            }
         }
      }
      // Cleanup bio
      BIO_free(bcpy);
   }
}

//_____________________________________________________________________________
XrdCryptosslRSA::~XrdCryptosslRSA()
{
   // Destructor
   // Destroy the RSA asymmetric key pair

   if (fEVP)
      EVP_PKEY_free(fEVP);
   fEVP = 0;
}

//_____________________________________________________________________________
int XrdCryptosslRSA::GetOutlen(int lin)
{
   // Get minimal length of output buffer

   int lcmax = RSA_size(fEVP->pkey.rsa) - 42;

   return ((lin / lcmax) + 1) * RSA_size(fEVP->pkey.rsa);
}

//_____________________________________________________________________________
int XrdCryptosslRSA::ImportPublic(const char *pub, int lpub)
{
   // Import a public key
   // Allocate a RSA key pair and fill the public part importing 
   // from string representation (pub) to internal representation.
   // If lpub>0 use the first lpub bytes; otherwise use strlen(pub)
   // bytes.
   // Return 0 in case of success, -1 in case of failure

   if (fEVP)
      EVP_PKEY_free(fEVP);
   fEVP = 0;
   publen = -1;
   prilen = -1;

   // Temporary key
   EVP_PKEY *keytmp = 0;

   // Bio for exporting the pub key
   BIO *bpub = BIO_new(BIO_s_mem());

   // Check length
   lpub = (lpub <= 0) ? strlen(pub) : lpub;

   // Write key from pubexport to BIO
   BIO_write(bpub,(void *)pub,lpub);

   // Read pub key from BIO
   if ((keytmp = PEM_read_bio_PUBKEY(bpub, 0, 0, 0))) {
      fEVP = keytmp;
      // Update status
      status = kPublic;
      return 0;
   }
   return -1;
}

//_____________________________________________________________________________
int XrdCryptosslRSA::ImportPrivate(const char *pri, int lpri)
{
   // Import a private key
   // Fill the private part importing from string representation (pub) to
   // internal representation.
   // If lpub>0 use the first lpub bytes; otherwise use strlen(pub)
   // bytes.
   // Return 0 in case of success, -1 in case of failure

   if (!fEVP)
      return -1;
   prilen = -1;

   // Bio for exporting the pub key
   BIO *bpri = BIO_new(BIO_s_mem());

   // Check length
   lpri = (lpri <= 0) ? strlen(pri) : lpri;

   // Write key from private export to BIO
   BIO_write(bpri,(void *)pri,lpri);

   // Read private key from BIO
   if (PEM_read_bio_PrivateKey(bpri, &fEVP, 0, 0)) {
      // Update status
      status = kComplete;
      return 0;
   }
   return -1;
}

//_____________________________________________________________________________
void XrdCryptosslRSA::Dump()
{
   // Dump some info about the key
   EPNAME("RSA::Dump");

   DEBUG("---------------------------------------");
   DEBUG("address: "<<this);
   if (IsValid()) {
      char *btmp = new char[GetPublen()+1];
      if (btmp) {
         ExportPublic(btmp,GetPublen()+1);
         DEBUG("export pub key:"<<endl<<btmp);
         delete[] btmp;
      } else {
         DEBUG("cannot allocate memory for public key");
      }
   } else {
      DEBUG("key is invalid");
   }
   DEBUG("---------------------------------------");
}

//_____________________________________________________________________________
int XrdCryptosslRSA::GetPublen()
{
   // Minimu length of export format of public key 

   if (publen < 0) {
      // Bio for exporting the pub key
      BIO *bkey = BIO_new(BIO_s_mem());
      // Write public key to BIO
      PEM_write_bio_PUBKEY(bkey,fEVP);
      // data length
      char *cbio = 0;
      publen = (int) BIO_get_mem_data(bkey, &cbio);
      BIO_free(bkey);
   }
   return publen;
}
//_____________________________________________________________________________
int XrdCryptosslRSA::ExportPublic(char *out, int)
{
   // Export the public key into buffer out. The length of the buffer should be
   // at least GetPublen()+1 bytes. If out=0 the buffer is m-allocated internally
   // and should be freed by the caller.
   // Return 0 in case of success, -1 in case of failure
   EPNAME("RSA::ExportPublic");

   // Make sure we have a valid key
   if (!IsValid()) {
      DEBUG("key not valid");
      return -1;
   }

   // Bio for exporting the pub key
   BIO *bkey = BIO_new(BIO_s_mem());

   // Write public key to BIO
   PEM_write_bio_PUBKEY(bkey,fEVP);

   // data length
   char *cbio = 0;
   int lbio = (int) BIO_get_mem_data(bkey, &cbio);
   if (lbio <= 0 || !cbio) {
      DEBUG("problems attaching to BIO content");
      return -1;
   }

   // Check output buffer
   if (!out) {
      out = (char *) malloc(lbio+1);
      if (!out) {
         DEBUG("problems allocating output buffer");
         return -1;
      }
   }
   // Read key from BIO to buf
   memcpy(out, cbio, lbio);
   // Null terminate
   out[lbio] = 0;
   DEBUG("("<<lbio<<" bytes) "<< endl <<out);
   BIO_free(bkey);

   return 0;
}

//_____________________________________________________________________________
int XrdCryptosslRSA::GetPrilen()
{
   // Minimu length of export format of private key 

   if (prilen < 0) {
      // Bio for exporting the private key
      BIO *bkey = BIO_new(BIO_s_mem());
      // Write public key to BIO
      PEM_write_bio_PrivateKey(bkey,fEVP,0,0,0,0,0);
      // data length
      char *cbio = 0;
      prilen = (int) BIO_get_mem_data(bkey, &cbio);
      BIO_free(bkey);
   }
   return prilen;
}

//_____________________________________________________________________________
int XrdCryptosslRSA::ExportPrivate(char *out, int)
{
   // Export the private key into buffer out. The length of the buffer should be
   // at least GetPrilen()+1 bytes. If out=0 the buffer is m-allocated internally
   // and should be freed by the caller.
   // Return 0 in case of success, -1 in case of failure
   EPNAME("RSA::ExportPrivate");

   // Make sure we have a valid key
   if (!IsValid()) {
      DEBUG("key not valid");
      return -1;
   }

   // Bio for exporting the pub key
   BIO *bkey = BIO_new(BIO_s_mem());

   // Write public key to BIO
   PEM_write_bio_PrivateKey(bkey,fEVP,0,0,0,0,0);

   // data length
   char *cbio = 0;
   int lbio = (int) BIO_get_mem_data(bkey, &cbio);
   if (lbio <= 0 || !cbio) {
      DEBUG("problems attaching to BIO content");
      return -1;
   }

   // Check output buffer
   if (!out) {
      out = (char *) malloc(lbio+1);
      if (!out) {
         DEBUG("problems allocating output buffer");
         return -1;
      }
   }
   // Read key from BIO to buf
   memcpy(out, cbio, lbio);
   // Null terminate
   out[lbio] = 0;
   DEBUG("("<<lbio<<" bytes) "<< endl <<out);
   BIO_free(bkey);

   return 0;
}

//_____________________________________________________________________________
int XrdCryptosslRSA::EncryptPrivate(const char *in, int lin, char *out, int loutmax)
{
   // Encrypt lin bytes at 'in' using the internal private key.
   // The output buffer 'out' is allocated by the caller for max lout bytes.
   // The number of meaningful bytes in out is returned in case of success
   // (never larger that loutmax); -1 in case of error.
   EPNAME("RSA::EncryptPrivate");

   // Make sure we got something to encrypt
   if (!in || lin <= 0) {
      DEBUG("input buffer undefined");
      return -1;
   }

   // Make sure we got a buffer where to write
   if (!out || loutmax <= 0) {
      DEBUG("output buffer undefined");
      return -1;
   }

   //
   // Private encoding ...
   int lcmax = RSA_size(fEVP->pkey.rsa) - 11;  // Magic number (= 2*sha1_outlen + 2)
   int lout = 0;
   int len = lin;
   int kk = 0;
   int ke = 0;

   while (len > 0 && ke <= (loutmax - lout)) {
      int lc = (len > lcmax) ? lcmax : len ;
      if ((lout = RSA_private_encrypt(lc, (unsigned char *)&in[kk],
                                          (unsigned char *)&out[ke],
                                      fEVP->pkey.rsa, RSA_PKCS1_PADDING)) < 0) {
         char serr[120];
         ERR_error_string(ERR_get_error(), serr);
         DEBUG("error: " <<serr);
         return -1;
      }
      kk += lc;
      ke += lout;
      len -= lc;
   }
   if (len > 0 && ke > (loutmax - lout))
      DEBUG("buffer truncated");
   lout = ke;

   // Return   
   return lout;
}

//_____________________________________________________________________________
int XrdCryptosslRSA::EncryptPublic(const char *in, int lin, char *out, int loutmax)
{
   // Encrypt lin bytes at 'in' using the internal public key.
   // The output buffer 'out' is allocated by the caller for max lout bytes.
   // The number of meaningful bytes in out is returned in case of success
   // (never larger that loutmax); -1 in case of error.
   EPNAME("RSA::EncryptPublic");
   
   // Make sure we got something to encrypt
   if (!in || lin <= 0) {
      DEBUG("input buffer undefined");
      return -1;
   }

   // Make sure we got a buffer where to write
   if (!out || loutmax <= 0) {
      DEBUG("output buffer undefined");
      return -1;
   }

   //
   // Public encoding ...
   int lcmax = RSA_size(fEVP->pkey.rsa) - 42;  // Magic number (= 2*sha1_outlen + 2)
   int lout = 0;
   int len = lin;
   int kk = 0;
   int ke = 0;

   while (len > 0 && ke <= (loutmax - lout)) {
      int lc = (len > lcmax) ? lcmax : len ;
      if ((lout = RSA_public_encrypt(lc, (unsigned char *)&in[kk],
                                         (unsigned char *)&out[ke],
                                     fEVP->pkey.rsa, RSA_PKCS1_OAEP_PADDING)) < 0) {
         char serr[120];
         ERR_error_string(ERR_get_error(), serr);
         DEBUG("error: " <<serr);
         return -1;
      }
      kk += lc;
      ke += lout;
      len -= lc;
   }
   if (len > 0 && ke > (loutmax - lout))
      DEBUG("buffer truncated");
   lout = ke;

   // Return   
   return lout;
}

//_____________________________________________________________________________
int XrdCryptosslRSA::DecryptPrivate(const char *in, int lin, char *out, int loutmax)
{
   // Decrypt lin bytes at 'in' using the internal private key
   // The output buffer 'out' is allocated by the caller for max lout bytes.
   // The number of meaningful bytes in out is returned in case of success
   // (never larger that loutmax); -1 in case of error.
   EPNAME("RSA::DecryptPrivate");

   // Make sure we got something to decrypt
   if (!in || lin <= 0) {
      DEBUG("input buffer undefined");
      return -1;
   }

   // Make sure we got a buffer where to write
   if (!out || loutmax <= 0) {
      DEBUG("output buffer undefined");
      return -1;
   }

   int lout = 0;
   int len = lin;
   int lcmax = RSA_size(fEVP->pkey.rsa);
   int kk = 0;
   int ke = 0;

   //
   // Private decoding ...
   while (len > 0 && ke <= (loutmax - lout)) {
      if ((lout = RSA_private_decrypt(lcmax, (unsigned char *)&in[kk],
                                             (unsigned char *)&out[ke],
                                      fEVP->pkey.rsa, RSA_PKCS1_OAEP_PADDING)) < 0) {
         char serr[120];
         ERR_error_string(ERR_get_error(), serr);
         DEBUG("error: " <<serr);
         return -1;
      }
      kk += lcmax;
      len -= lcmax;
      ke += lout;
   }
   if (len > 0 && ke > (loutmax - lout))
      PRINT("buffer truncated");
   lout = ke;
   
   return lout;
}

//_____________________________________________________________________________
int XrdCryptosslRSA::DecryptPublic(const char *in, int lin, char *out, int loutmax)
{
   // Decrypt lin bytes at 'in' using the internal public key
   // The output buffer 'out' is allocated by the caller for max lout bytes.
   // The number of meaningful bytes in out is returned in case of success
   // (never larger that loutmax); -1 in case of error.
   EPNAME("RSA::DecryptPublic");

   // Make sure we got something to decrypt
   if (!in || lin <= 0) {
      DEBUG("input buffer undefined");
      return -1;
   }

   // Make sure we got a buffer where to write
   if (!out || loutmax <= 0) {
      DEBUG("output buffer undefined");
      return -1;
   }

   int lout = 0;
   int len = lin;
   int lcmax = RSA_size(fEVP->pkey.rsa);
   int kk = 0;
   int ke = 0;

   //
   // Private decoding ...
   while (len > 0 && ke <= (loutmax - lout)) {
      if ((lout = RSA_public_decrypt(lcmax, (unsigned char *)&in[kk],
                                            (unsigned char *)&out[ke],
                                     fEVP->pkey.rsa, RSA_PKCS1_PADDING)) < 0) {
         char serr[120];
         ERR_error_string(ERR_get_error(), serr);
         PRINT("error: " <<serr);
         return -1;
      }
      kk += lcmax;
      len -= lcmax;
      ke += lout;
   }
   if (len > 0 && ke > (loutmax - lout))
      PRINT("buffer truncated");
   lout = ke;
   
   return lout;
}
