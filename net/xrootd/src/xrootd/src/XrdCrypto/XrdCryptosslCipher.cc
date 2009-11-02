// $Id$

const char *XrdCryptosslCipherCVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*                  X r d C r y p t o S s l C i p h e r . c c                 */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* OpenSSL implementation of XrdCryptoCipher                                  */
/*                                                                            */
/* ************************************************************************** */
#include <string.h>

#include <XrdSut/XrdSutRndm.hh>
#include <XrdCrypto/XrdCryptosslTrace.hh>
#include <XrdCrypto/XrdCryptosslCipher.hh>

//#include <openssl/dsa.h>
#include <openssl/bio.h>
#include <openssl/pem.h>

// ---------------------------------------------------------------------------//
//
// Cipher interface
//
// ---------------------------------------------------------------------------//

//_____________________________________________________________________________
bool XrdCryptosslCipher::IsSupported(const char *cip)
{
   // Check if the specified cipher is supported

   return (EVP_get_cipherbyname(cip) != 0);
}

//____________________________________________________________________________
XrdCryptosslCipher::XrdCryptosslCipher(const char *t, int l)
{
   // Main Constructor
   // Complete initialization of a cipher of type t and length l
   // The initialization vector is also created
   // Used to create ciphers

   valid = 0;
   fIV = 0;
   lIV = 0;
   cipher = 0;
   fDH = 0;
   deflength = 1;

   // Check and set type
   char cipnam[64] = {"bf-cbc"};
   if (t && strcmp(t,"default")) {
      strcpy(cipnam,t); 
      cipnam[63] = 0;
   }
   cipher = EVP_get_cipherbyname(cipnam);

   if (cipher) {
      // Init context
      EVP_CIPHER_CTX_init(&ctx);
      // Determine key length
      l = (l > EVP_MAX_KEY_LENGTH) ? EVP_MAX_KEY_LENGTH : l;
      int ldef = EVP_CIPHER_key_length(cipher);
      int lgen = (l > ldef) ? l : ldef;
      // Generate and set a new key
      char *ktmp = XrdSutRndm::GetBuffer(lgen);
      if (ktmp) {
         valid = 1;
         // Try setting the key length
         if (l && l != ldef) {
            EVP_CipherInit(&ctx, cipher, 0, 0, 1);
            EVP_CIPHER_CTX_set_key_length(&ctx,l);
            EVP_CipherInit(&ctx, 0, (unsigned char *)ktmp, 0, 1);
            if (l == EVP_CIPHER_CTX_key_length(&ctx)) {
               // Use the l bytes at ktmp
               SetBuffer(l,ktmp);
               deflength = 0;
            }
         }
         if (!Length()) {
            EVP_CipherInit(&ctx, cipher, (unsigned char *)ktmp, 0, 1);
            SetBuffer(ldef,ktmp);
         }
         // Set also the type
         SetType(cipnam);
         // Cleanup
         delete[] ktmp;
      }
   }

   // Finally, generate and set a new IV
   if (valid)
      GenerateIV();
}

//____________________________________________________________________________
XrdCryptosslCipher::XrdCryptosslCipher(const char *t, int l,
                                       const char *k, int liv, const char *iv)
{
   // Constructor.
   // Initialize a cipher of type t and length l using the key at k and
   // the initialization vector at iv.
   // Used to import ciphers.
   valid = 0;
   fIV = 0;
   lIV = 0;
   fDH = 0;
   cipher = 0;
   deflength = 1;

   // Check and set type
   char cipnam[64] = {"bf-cbc"};
   if (t && strcmp(t,"default")) {
      strcpy(cipnam,t); 
      cipnam[63] = 0;
   }
   cipher = EVP_get_cipherbyname(cipnam);

   if (cipher) {
      // Init context
      EVP_CIPHER_CTX_init(&ctx);
      // Set the key
      SetBuffer(l,k);
      if (l != EVP_CIPHER_key_length(cipher))
         deflength = 0;
      // Set also the type
      SetType(cipnam);
      // Set validity flag
      valid = 1;
   }
   //
   // Init cipher
   if (valid) {
      //
      // Set the IV
      SetIV(liv,iv);

      if (deflength) {
         EVP_CipherInit(&ctx, cipher, (unsigned char *)Buffer(), 0, 1);
      } else {
         EVP_CipherInit(&ctx, cipher, 0, 0, 1);
         EVP_CIPHER_CTX_set_key_length(&ctx,Length());
         EVP_CipherInit(&ctx, 0, (unsigned char *)Buffer(), 0, 1);
      }
   }
}

//____________________________________________________________________________
XrdCryptosslCipher::XrdCryptosslCipher(XrdSutBucket *bck)
{
   // Constructor from bucket.
   // Initialize a cipher of type t and length l using the key at k
   // Used to import ciphers.
   valid = 0;
   fIV = 0;
   lIV = 0;
   fDH = 0;
   cipher = 0;
   deflength = 1;

   if (bck && bck->size > 0) {

      // Init context
      EVP_CIPHER_CTX_init(&ctx);

      valid = 1;

      kXR_int32 ltyp = 0;
      kXR_int32 livc = 0;
      kXR_int32 lbuf = 0;
      kXR_int32 lp = 0;
      kXR_int32 lg = 0;
      kXR_int32 lpub = 0;
      kXR_int32 lpri = 0;
      char *bp = bck->buffer;
      int cur = 0;
      memcpy(&ltyp,bp+cur,sizeof(kXR_int32));
      cur += sizeof(kXR_int32);
      memcpy(&livc,bp+cur,sizeof(kXR_int32));
      cur += sizeof(kXR_int32);
      memcpy(&lbuf,bp+cur,sizeof(kXR_int32));
      cur += sizeof(kXR_int32);
      memcpy(&lp,bp+cur,sizeof(kXR_int32));
      cur += sizeof(kXR_int32);
      memcpy(&lg,bp+cur,sizeof(kXR_int32));
      cur += sizeof(kXR_int32);
      memcpy(&lpub,bp+cur,sizeof(kXR_int32));
      cur += sizeof(kXR_int32);
      memcpy(&lpri,bp+cur,sizeof(kXR_int32));
      cur += sizeof(kXR_int32);
      // Type
      if (ltyp) {
         char *buf = new char[ltyp+1];
         if (buf) {
            memcpy(buf,bp+cur,ltyp);
            buf[ltyp] = 0;
            cipher = EVP_get_cipherbyname(buf);
            if (!cipher)
               cipher = EVP_get_cipherbyname("bf-cbc");
            if (cipher) {
               // Set the type
               SetType(buf);
            } else {
               valid = 0;
            } 
            delete[] buf;
         } else
            valid = 0;
         cur += ltyp;
      }
      // IV
      if (livc) {
         char *buf = new char[livc];
         if (buf) {
            memcpy(buf,bp+cur,livc);
            cur += livc;
            // Set the IV
            SetIV(livc,buf);
            delete[] buf;
         } else
            valid = 0;
         cur += livc;
      }
      // buffer
      if (lbuf) {
         char *buf = new char[lbuf];
         if (buf) {
            memcpy(buf,bp+cur,lbuf);
            // Set the buffer
            UseBuffer(lbuf,buf);
            if (cipher && lbuf != EVP_CIPHER_key_length(cipher))
               deflength = 0;
         } else
            valid = 0;
         cur += lbuf;
      }
      // DH, if any
      if (lp > 0 || lg > 0 || lpub > 0 || lpri > 0) {
         if ((fDH = DH_new())) {
            char *buf = 0;
            // p
            if (lp > 0) {
               buf = new char[lp+1];
               if (buf) {
                  memcpy(buf,bp+cur,lp);
                  buf[lp] = 0;
                  BN_hex2bn(&(fDH->p),buf);
                  delete[] buf;
               } else
                  valid = 0;
               cur += lp;
            }
            // g
            if (lg > 0) {
               buf = new char[lg+1];
               if (buf) {
                  memcpy(buf,bp+cur,lg);
                  buf[lg] = 0;
                  BN_hex2bn(&(fDH->g),buf);
                  delete[] buf;
               } else
                  valid = 0;
               cur += lg;
            }
            // pub_key
            if (lpub > 0) {
               buf = new char[lpub+1];
               if (buf) {
                  memcpy(buf,bp+cur,lpub);
                  buf[lpub] = 0;
                  BN_hex2bn(&(fDH->pub_key),buf);
                  delete[] buf;
               } else
                  valid = 0;
               cur += lpub;
            }
            // priv_key
            if (lpri > 0) {
               buf = new char[lpri+1];
               if (buf) {
                  memcpy(buf,bp+cur,lpri);
                  buf[lpri] = 0;
                  BN_hex2bn(&(fDH->priv_key),buf);
                  delete[] buf;
               } else
                  valid = 0;
               cur += lpri;
            }
            int dhrc = 0;
            DH_check(fDH,&dhrc);
            if (dhrc == 0)
               valid = 1;
         } else
            valid = 0;
      }
   }
   //
   // Init cipher
   if (valid) {
      if (deflength) {
         EVP_CipherInit(&ctx, cipher, (unsigned char *)Buffer(), 0, 1);
      } else {
         EVP_CipherInit(&ctx, cipher, 0, 0, 1);
         EVP_CIPHER_CTX_set_key_length(&ctx,Length());
         EVP_CipherInit(&ctx, 0, (unsigned char *)Buffer(), 0, 1);
      }
   }
}

//____________________________________________________________________________
XrdCryptosslCipher::XrdCryptosslCipher(int bits, char *pub,
                                       int lpub, const char *t)
{
   // Constructor for key agreement.
   // If pub is not defined, generates a DH full key,
   // the public part and parameters can be retrieved using Public().
   // The number of random bits to be used in 'bits'.
   // If pub is defined with the public part and parameters of the
   // counterpart fully initialize a cipher with that information.
   // Sets also the name to 't', if different from the default one.
   // Used for key agreement.
   EPNAME("sslCipher::XrdCryptosslCipher");

   valid = 0;
   fIV = 0;
   lIV = 0;
   fDH = 0;
   cipher = 0;
   deflength = 1;

   if (!pub) {
      DEBUG("generate DH full key");
      //
      // at least 128 bits
      bits = (bits < kDHMINBITS) ? kDHMINBITS : bits; 
      //
      // Generate params for DH object
      if ((fDH = DH_generate_parameters(bits,DH_GENERATOR_5,0,0))) {
         int prc = 0;
         DH_check(fDH,&prc);
         if (prc == 0) {
            //
            // Generate DH key
            if (DH_generate_key(fDH)) {
               valid = 1;
            } else {
               DH_free(fDH);
            }
         }
      }

   } else {
      DEBUG("initialize cipher from key-agreement buffer");
      //
      char *ktmp = 0;
      int ltmp = 0;
      // Extract string with bignumber
      BIGNUM *bnpub = 0;
      char *pb = strstr(pub,"---BPUB---");
      char *pe = strstr(pub,"---EPUB--"); // one less (pub not null-terminated)
      if (pb && pe) {
         lpub = (int)(pb-pub);
         pb += 10;
         *pe = 0;
         BN_hex2bn(&bnpub, pb);
         *pe = '-';
      }
      if (bnpub) {
         //
         // Prepare to decode the input buffer
         BIO *biop = BIO_new(BIO_s_mem());
         if (biop) {
            //
            // Write buffer into BIO
            BIO_write(biop,pub,lpub);
            //
            // Create a key object
            if ((fDH = DH_new())) {
               //
               // Read parms from BIO
               PEM_read_bio_DHparams(biop,&fDH,0,0);
               int prc = 0;
               DH_check(fDH,&prc);
               if (prc == 0) {
                  //
                  // generate DH key
                  if (DH_generate_key(fDH)) {
                     // Now we can compute the cipher
                     ktmp = new char[DH_size(fDH)];
                     memset(ktmp, 0, DH_size(fDH));
                     if (ktmp) {
                        if ((ltmp = DH_compute_key((unsigned char *)ktmp,
                                                    bnpub,fDH)) > 0)
                           valid = 1;
                     }
                  }
               }
            }
            BIO_free(biop);
         }
      }
      //
      // If a valid key has been computed, set the cipher
      if (valid) {

         // Check and set type
         char cipnam[64] = {"bf-cbc"};
         if (t && strcmp(t,"default")) {
            strcpy(cipnam,t); 
            cipnam[63] = 0;
         }
         if ((cipher = EVP_get_cipherbyname(cipnam))) {
            // Init context
            EVP_CIPHER_CTX_init(&ctx);
            // At most EVP_MAX_KEY_LENGTH bytes
            ltmp = (ltmp > EVP_MAX_KEY_LENGTH) ? EVP_MAX_KEY_LENGTH : ltmp;
            int ldef = EVP_CIPHER_key_length(cipher);
            // Try setting the key length
            if (ltmp != ldef) {
               EVP_CipherInit(&ctx, cipher, 0, 0, 1);
               EVP_CIPHER_CTX_set_key_length(&ctx,ltmp);
               EVP_CipherInit(&ctx, 0, (unsigned char *)ktmp, 0, 1);
               if (ltmp == EVP_CIPHER_CTX_key_length(&ctx)) {
                  // Use the ltmp bytes at ktmp
                  SetBuffer(ltmp,ktmp);
                  deflength = 0;
               }
            }
            if (!Length()) {
               EVP_CipherInit(&ctx, cipher, (unsigned char *)ktmp, 0, 1);
               SetBuffer(ldef,ktmp);
            }
            // Set also the type
            SetType(cipnam);
         }
      }     
      // Cleanup
      if (ktmp) delete[] ktmp; ktmp = 0;
   }

   // Cleanup, if invalid
   if (!valid)
      Cleanup();
}

//____________________________________________________________________________
XrdCryptosslCipher::XrdCryptosslCipher(const XrdCryptosslCipher &c)
{
   // Copy Constructor

   // Basics
   deflength = c.deflength;
   valid = c.valid;
   // IV
   lIV = 0;
   fIV = 0;
   SetIV(c.lIV,c.fIV);
   // Cipher
   cipher = c.cipher;
   // Init context
   EVP_CIPHER_CTX_init(&ctx);
   // Set the key
   SetBuffer(c.Length(),c.Buffer());
   // Set also the type
   SetType(c.Type());
   // DH
   fDH = 0;
   if (valid && c.fDH) {
      valid = 0;
      if ((fDH = DH_new())) {
         if (c.fDH->p) fDH->p = BN_dup(c.fDH->p);
         if (c.fDH->g) fDH->g = BN_dup(c.fDH->g);
         if (c.fDH->pub_key) fDH->pub_key = BN_dup(c.fDH->pub_key);
         if (c.fDH->priv_key) fDH->priv_key = BN_dup(c.fDH->priv_key);
         int dhrc = 0;
         DH_check(fDH,&dhrc);
         if (dhrc == 0)
            valid = 1;
      }
   }
}

//____________________________________________________________________________
XrdCryptosslCipher::~XrdCryptosslCipher()
{
   // Destructor.

   // Cleanup IV
   if (fIV)
      delete[] fIV;

   // Cleanups
   if (valid)
      EVP_CIPHER_CTX_cleanup(&ctx);
   Cleanup();
}

//____________________________________________________________________________
void XrdCryptosslCipher::Cleanup()
{
   // Cleanup temporary memory

   // Cleanup IV
   if (fDH) {
      DH_free(fDH);
      fDH = 0;
   }
}

//____________________________________________________________________________
bool XrdCryptosslCipher::Finalize(char *pub, int lpub, const char *t)
{
   // Finalize cipher during key agreement. Should be called
   // for a cipher build with special constructor defining member fDH.
   // The buffer pub should contain the public part of the counterpart.
   // Sets also the name to 't', if different from the default one.
   // Used for key agreement.
   EPNAME("sslCipher::Finalize");

   if (!fDH) {
      DEBUG("DH undefined: this cipher cannot be finalized"
            " by this method");
      return 0;
   }

   char *ktmp = 0;
   int ltmp = 0;
   if (pub) {
      //
      // Extract string with bignumber
      BIGNUM *bnpub = 0;
      char *pb = strstr(pub,"---BPUB---");
      char *pe = strstr(pub,"---EPUB--");
      if (pb && pe) {
         lpub = (int)(pb-pub);
         pb += 10;
         *pe = 0;
         BN_hex2bn(&bnpub, pb);
         *pe = '-';
      }
      if (bnpub) {
         // Now we can compute the cipher
         ktmp = new char[DH_size(fDH)];
         memset(ktmp, 0, DH_size(fDH));
         if (ktmp) {
            if ((ltmp =
                 DH_compute_key((unsigned char *)ktmp,bnpub,fDH)) > 0)
               valid = 1;
         }
      }
      //
      // If a valid key has been computed, set the cipher
      if (valid) {
         // Check and set type
         char cipnam[64] = {"bf-cbc"};
         if (t && strcmp(t,"default")) {
            strcpy(cipnam,t); 
            cipnam[63] = 0;
         }
         if ((cipher = EVP_get_cipherbyname(cipnam))) {
            // Init context
            EVP_CIPHER_CTX_init(&ctx);
            // At most EVP_MAX_KEY_LENGTH bytes
            ltmp = (ltmp > EVP_MAX_KEY_LENGTH) ? EVP_MAX_KEY_LENGTH : ltmp;
            int ldef = EVP_CIPHER_key_length(cipher);
            // Try setting the key length
            if (ltmp != ldef) {
               EVP_CipherInit(&ctx, cipher, 0, 0, 1);
               EVP_CIPHER_CTX_set_key_length(&ctx,ltmp);
               EVP_CipherInit(&ctx, 0, (unsigned char *)ktmp, 0, 1);
               if (ltmp == EVP_CIPHER_CTX_key_length(&ctx)) {
                  // Use the ltmp bytes at ktmp
                  SetBuffer(ltmp,ktmp);
                  deflength = 0;
               }
            }
            if (!Length()) {
               EVP_CipherInit(&ctx, cipher, (unsigned char *)ktmp, 0, 1);
               SetBuffer(ldef,ktmp);
            }
            // Set also the type
            SetType(cipnam);
         }
      }     
      // Cleanup
      if (ktmp) delete[] ktmp; ktmp = 0;
   }

   // Cleanup, if invalid
   if (!valid)
      Cleanup();

   // We are done
   return valid;
}

//_____________________________________________________________________________
int XrdCryptosslCipher::Publen()
{
   // Minimu length of export format of public key 
   static int lhdr = strlen("-----BEGIN DH PARAMETERS-----"
                            "-----END DH PARAMETERS-----") + 3;
   if (fDH) {
      // minimum length of the core is 22 bytes
      int l = 2*DH_size(fDH);
      if (l < 22) l = 22;
      // for headers
      l += lhdr;
      // some margin
      return (l+20);
   } else
      return 0;
}

//_____________________________________________________________________________
char *XrdCryptosslCipher::Public(int &lpub)
{
   // Return buffer with the public part of the DH key and the shared
   // parameters; lpub contains the length of the meaningful bytes.
   // Buffer should be deleted by the caller.
   static int lhend = strlen("-----END DH PARAMETERS-----");

   if (fDH) {
      //
      // Calculate and write public key hex
      char *phex = BN_bn2hex(fDH->pub_key);
      int lhex = strlen(phex);
      //
      // Prepare bio to export info buffer
      BIO *biop = BIO_new(BIO_s_mem());
      if (biop) {
         int ltmp = Publen() + lhex + 20;
         char *pub = new char[ltmp];
         if (pub) {
            // Write parms first
            PEM_write_bio_DHparams(biop,fDH);
            // Read key from BIO to buf
            BIO_read(biop,(void *)pub,ltmp);
            BIO_free(biop);
            // Add public key
            char *p = strstr(pub,"-----END DH PARAMETERS-----");
            // Buffer length up to now
            lpub = (int)(p - pub) + lhend + 1;
            if (phex && p) {
               // position at the end
               p += (lhend+1);
               // Begin of public key hex
               strncpy(p,"---BPUB---",10);
               p += 10;
               // Calculate and write public key hex
               strncpy(p,phex,lhex);
               OPENSSL_free(phex);
               // End of public key hex
               p += lhex;
               strncpy(p,"---EPUB---",10);
               // Calculate total length
               lpub += (20 + lhex);
            } else {
               if (phex) OPENSSL_free(phex);
            }
            // return
            return pub;
         }
      } else {
         if (phex) OPENSSL_free(phex);
      }
   }

   lpub = 0;
   return (char *)0;
}

//_____________________________________________________________________________
void XrdCryptosslCipher::PrintPublic(BIGNUM *pub)
{
   // Print public part

   //
   // Prepare bio to export info buffer
   BIO *biop = BIO_new(BIO_s_mem());
   if (biop) {
      // Use a DSA structure to export the public part
      DSA *dsa = DSA_new();
      if (dsa) {
         dsa->pub_key = BN_dup(pub);
         // Write public key to BIO
         PEM_write_bio_DSA_PUBKEY(biop,dsa);
         // Read key from BIO to buf
         int lpub = Publen();
         char *bpub = new char[lpub];
         if (bpub) {
            BIO_read(biop,(void *)bpub,lpub);
            cerr << bpub << endl;
            delete[] bpub;
         }
         DSA_free(dsa);
      }
      BIO_free(biop);
   }
}

//_____________________________________________________________________________
XrdSutBucket *XrdCryptosslCipher::AsBucket()
{
   // Return pointer to a bucket created using the internal information
   // serialized
   // The bucket is responsible for the allocated memory

   XrdSutBucket *buck = (XrdSutBucket *)0;

   if (valid) {

      // Serialize .. total length
      kXR_int32 lbuf = Length();
      kXR_int32 ltyp = Type() ? strlen(Type()) : 0;
      kXR_int32 livc = lIV;
      char *cp = (fDH && fDH->p) ? BN_bn2hex(fDH->p) : 0;
      char *cg = (fDH && fDH->g) ? BN_bn2hex(fDH->g) : 0;
      char *cpub = (fDH && fDH->pub_key) ? BN_bn2hex(fDH->pub_key) : 0;
      char *cpri = (fDH && fDH->priv_key) ? BN_bn2hex(fDH->priv_key) : 0;
      kXR_int32 lp = cp ? strlen(cp) : 0;
      kXR_int32 lg = cg ? strlen(cg) : 0;
      kXR_int32 lpub = cpub ? strlen(cpub) : 0;
      kXR_int32 lpri = cpri ? strlen(cpri) : 0;
      int ltot = 7*sizeof(kXR_int32) + ltyp + Length() + livc +
                 lp + lg + lpub + lpri;
      char *newbuf = new char[ltot];
      if (newbuf) {
         int cur = 0;
         memcpy(newbuf+cur,&ltyp,sizeof(kXR_int32));
         cur += sizeof(kXR_int32);
         memcpy(newbuf+cur,&livc,sizeof(kXR_int32));
         cur += sizeof(kXR_int32);
         memcpy(newbuf+cur,&lbuf,sizeof(kXR_int32));
         cur += sizeof(kXR_int32);
         memcpy(newbuf+cur,&lp,sizeof(kXR_int32));
         cur += sizeof(kXR_int32);
         memcpy(newbuf+cur,&lg,sizeof(kXR_int32));
         cur += sizeof(kXR_int32);
         memcpy(newbuf+cur,&lpub,sizeof(kXR_int32));
         cur += sizeof(kXR_int32);
         memcpy(newbuf+cur,&lpri,sizeof(kXR_int32));
         cur += sizeof(kXR_int32);
         if (Type()) {
            memcpy(newbuf+cur,Type(),ltyp);
            cur += ltyp;
         }
         if (fIV) {
            memcpy(newbuf+cur,fIV,livc);
            cur += livc;
         }
         if (Buffer()) {
            memcpy(newbuf+cur,Buffer(),lbuf);
            cur += lbuf;
         }
         if (cp) {
            memcpy(newbuf+cur,cp,lp);
            cur += lp;
            OPENSSL_free(cp);
         }
         if (cg) {
            memcpy(newbuf+cur,cg,lg);
            cur += lg;
            OPENSSL_free(cg);
         }
         if (cpub) {
            memcpy(newbuf+cur,cpub,lpub);
            cur += lpub;
            OPENSSL_free(cpub);
         }
         if (cpri) {
            memcpy(newbuf+cur,cpri,lpri);
            cur += lpri;
            OPENSSL_free(cpri);
         }
         // The bucket now
         buck = new XrdSutBucket(newbuf,ltot,kXRS_cipher);
      }
   }

   return buck;
}

//____________________________________________________________________________
void XrdCryptosslCipher::SetIV(int l, const char *iv)
{
   // Set IV from l bytes at iv

   if (fIV) {
      delete[] fIV;
      fIV = 0;
      lIV = 0;
   }

   if (iv && l > 0) {
      fIV = new char[l];
      if (fIV) {
         memcpy(fIV,iv,l);
         lIV = l;
      }
   }
}

//____________________________________________________________________________
char *XrdCryptosslCipher::RefreshIV(int &l)
{
   // Regenerate IV and return it

   // Generate a new IV
   GenerateIV();

   // Set output
   l = lIV;
   return fIV;
}

//____________________________________________________________________________
void XrdCryptosslCipher::GenerateIV()
{
   // Generate IV

   // Cleanup existing one, if any
   if (fIV) {
      delete[] fIV;
      fIV = 0;
      lIV = 0;
   }

   // Generate a new one
   fIV = XrdSutRndm::GetBuffer(EVP_MAX_IV_LENGTH);
   if (fIV)
      lIV = EVP_MAX_IV_LENGTH;
}

//____________________________________________________________________________
int XrdCryptosslCipher::Encrypt(const char *in, int lin, char *out)
{
   // Encrypt lin bytes at in with local cipher.
   // The outbut buffer must be provided by the caller for at least
   // EncOutLength(lin) bytes.
   // Returns number of meaningful bytes in out, or 0 in case of problems

   return EncDec(1, in, lin, out);
}

//____________________________________________________________________________
int XrdCryptosslCipher::Decrypt(const char *in, int lin, char *out)
{
   // Decrypt lin bytes at in with local cipher.
   // The outbut buffer must be provided by the caller for at least
   // DecOutLength(lin) bytes.
   // Returns number of meaningful bytes in out, or 0 in case of problems

   return EncDec(0, in, lin, out);
}

//____________________________________________________________________________
int XrdCryptosslCipher::EncDec(int enc, const char *in, int lin, char *out)
{
   // Encrypt (enc = 1)/ Decrypt (enc = 0) lin bytes at in with local cipher.
   // The outbut buffer must be provided by the caller for at least
   // EncOutLength(lin) or DecOutLength(lin) bytes.
   // Returns number of meaningful bytes in out, or 0 in case of problems
   EPNAME("Cipher::EncDec"); 

   int lout = 0;

   // Check inputs
   if (!in || lin <= 0 || !out) {
      DEBUG("wrong inputs arguments"); 
      if (!in) DEBUG("in: "<<in); 
      if (lin <= 0) DEBUG("lin: "<<lin); 
      if (!out) DEBUG("out: "<<out); 
      return 0;
   }

   // Set iv to the one in use
   unsigned char iv[EVP_MAX_IV_LENGTH];
   if (fIV) {
      memcpy((void *)iv,fIV,EVP_MAX_IV_LENGTH);
   } else {
      // We use 0's
      memset((void *)iv,0,EVP_MAX_IV_LENGTH);
   }

   // Action depend on the length of the key wrt default length
   if (deflength) {
      // Init ctx, set key (default length) and set IV
      if (!EVP_CipherInit(&ctx, cipher, (unsigned char *)Buffer(), iv, enc)) {
         DEBUG("error initializing"); 
         return 0;
      }
   } else {
      // Init ctx
      if (!EVP_CipherInit(&ctx, cipher, 0, 0, enc)) {
         DEBUG("error initializing - 1"); 
         return 0;
      }
      // Set key length
      EVP_CIPHER_CTX_set_key_length(&ctx,Length());
      // Set key and IV
      if (!EVP_CipherInit(&ctx, 0, (unsigned char *)Buffer(), iv, enc)) {
         DEBUG("error initializing - 2"); 
         return 0;
      }
   }

   // Encrypt / Decrypt
   int ltmp = 0;
   if (!EVP_CipherUpdate(&ctx, (unsigned char *)&out[0], &ltmp,
                               (unsigned char *)in, lin)) {
      DEBUG("error encrypting"); 
      return 0;
   }
   lout = ltmp;
   if (!EVP_CipherFinal(&ctx, (unsigned char *)&out[lout], &ltmp)) {
      DEBUG("error finalizing"); 
      return 0;
   }

   // Results
   lout += ltmp;
   return lout;
}

//____________________________________________________________________________
int XrdCryptosslCipher::EncOutLength(int l)
{
   // Required buffer size for encrypting l bytes

   return (l+EVP_CIPHER_CTX_block_size(&ctx));
}

//____________________________________________________________________________
int XrdCryptosslCipher::DecOutLength(int l)
{
   // Required buffer size for decrypting l bytes

   int lout = l+EVP_CIPHER_CTX_block_size(&ctx)+1;
   lout = (lout <= 0) ? l : lout;
   return lout;
}
