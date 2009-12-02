// $Id$

const char *XrdCryptosslX509ReqCVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*                 X r d C r y p t o s s l X 5 0 9 R e q. c c                 */
/*                                                                            */
/* (c) 2005 G. Ganis , CERN                                                   */
/*                                                                            */
/******************************************************************************/


/* ************************************************************************** */
/*                                                                            */
/* OpenSSL implementation of XrdCryptoX509Req                                 */
/*                                                                            */
/* ************************************************************************** */
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

#include <XrdCrypto/XrdCryptosslRSA.hh>
#include <XrdCrypto/XrdCryptosslX509Req.hh>
#include <XrdCrypto/XrdCryptosslAux.hh>
#include <XrdCrypto/XrdCryptosslTrace.hh>

#include <openssl/pem.h>

//_____________________________________________________________________________
XrdCryptosslX509Req::XrdCryptosslX509Req(XrdSutBucket *buck) : XrdCryptoX509Req()
{
   // Constructor certificate from BIO 'bcer'
   EPNAME("X509Req::XrdCryptosslX509Req_bio");

   // Init private members
   creq = 0;        // The certificate object
   subject = "";    // subject;
   subjecthash = ""; // hash of subject;
   bucket = 0;      // bucket for serialization
   pki = 0;         // PKI of the certificate

   // Make sure we got something;
   if (!buck) {
      DEBUG("got undefined opaque buffer");
      return;
   }

   //
   // Create a bio_mem to store the certificates
   BIO *bmem = BIO_new(BIO_s_mem());
   if (!bmem) {
      DEBUG("unable to create BIO for memory operations");
      return; 
   }

   // Write data to BIO
   int nw = BIO_write(bmem,(const void *)(buck->buffer),buck->size);
   if (nw != buck->size) {
      DEBUG("problems writing data to memory BIO (nw: "<<nw<<")");
      return; 
   }

   // Get certificate request from BIO
   if (!PEM_read_bio_X509_REQ(bmem,&creq,0,0)) {
      DEBUG("unable to read certificate request to memory BIO");
      return;
   }
   //
   // Free BIO
   BIO_free(bmem);
   //
   // Init some of the private members (the others upon need)
   Subject();
   //
   // Get the public key
   EVP_PKEY *evpp = X509_REQ_get_pubkey(creq);
   //
   if (evpp) {
      // init pki with the partial key
      if (!pki)
         pki = new XrdCryptosslRSA(evpp, 0);
   } else {
      DEBUG("could not access the public key");
   }
}

//_____________________________________________________________________________
XrdCryptosslX509Req::XrdCryptosslX509Req(X509_REQ *xc) : XrdCryptoX509Req()
{
   // Constructor: import X509_REQ object
   EPNAME("X509Req::XrdCryptosslX509Req_x509");

   // Init private members
   creq = 0;        // The certificate object
   subject = "";    // subject;
   subjecthash = ""; // hash of subject;
   bucket = 0;      // bucket for serialization
   pki = 0;         // PKI of the certificate

   // Make sure we got something;
   if (!xc) {
      DEBUG("got undefined X509 object");
      return;
   }

   // Set certificate
   creq = xc;
   //
   // Init some of the private members (the others upon need)
   Subject();
   //
   // Get the public key
   EVP_PKEY *evpp = X509_REQ_get_pubkey(creq);
   //
   if (evpp) {
      // init pki with the partial key
      if (!pki)
         pki = new XrdCryptosslRSA(evpp, 0);
   } else {
      DEBUG("could not access the public key");
   }
}

//_____________________________________________________________________________
XrdCryptosslX509Req::~XrdCryptosslX509Req()
{
   // Destructor

   // Cleanup certificate
   if (creq) X509_REQ_free(creq);
   // Cleanup key
   if (pki) delete pki;
}

//_____________________________________________________________________________
const char *XrdCryptosslX509Req::Subject()
{
   // Return subject name
   EPNAME("X509Req::Subject");

   // If we do not have it already, try extraction
   if (subject.length() <= 0) {

      // Make sure we have a certificate
      if (!creq) {
         DEBUG("WARNING: no certificate available - cannot extract subject name");
         return (const char *)0;
      }
      
      // Extract subject name
      subject = X509_NAME_oneline(X509_REQ_get_subject_name(creq), 0, 0);
   }

   // return what we have
   return (subject.length() > 0) ? subject.c_str() : (const char *)0;
}

//_____________________________________________________________________________
const char *XrdCryptosslX509Req::SubjectHash()
{
   // Return issuer name
   EPNAME("X509Req::SubjectHash");

   // If we do not have it already, try extraction
   if (subjecthash.length() <= 0) {

      // Make sure we have a certificate
      if (creq) {
         char chash[15];
#if OPENSSL_VERSION_NUMBER >= 0x10000000L
         sprintf(chash,"%08lx.0",X509_NAME_hash_old(creq->req_info->subject));
#else
         sprintf(chash,"%08lx.0",X509_NAME_hash(creq->req_info->subject));
#endif
         subjecthash = chash;
      } else {
         DEBUG("WARNING: no certificate available - cannot extract subject hash");
      }
   }

   // return what we have
   return (subjecthash.length() > 0) ? subjecthash.c_str() : (const char *)0;
}

//_____________________________________________________________________________
XrdCryptoX509Reqdata XrdCryptosslX509Req::GetExtension(const char *oid)
{
   // Return issuer name
   EPNAME("X509Req::GetExtension");
   XrdCryptoX509Reqdata ext = 0;

   // Make sure we got something to look for
   if (!oid) {
      DEBUG("OID string not defined");
      return ext;
   }
 
   // Make sure we got something to look for
   if (!creq) {
      DEBUG("certificate is not initialized");
      return ext;
   }

   // Are there any extension?
   STACK_OF(X509_EXTENSION) *esk = X509_REQ_get_extensions(creq);
   //
#if OPENSSL_VERSION_NUMBER >= 0x10000000L
   int numext = sk_X509_EXTENSION_num(esk);
#else /* OPENSSL */
   int numext = sk_num(esk);
#endif /* OPENSSL */
   if (numext <= 0) {
      DEBUG("certificate has got no extensions");
      return ext;
   }
   DEBUG("certificate request has "<<numext<<" extensions");

   // If the string is the Standard Name of a known extension check
   // searche the corresponding NID
   int nid = OBJ_sn2nid(oid);
   bool usenid = (nid > 0);

   // Loop to identify the one we would like
   int i = 0;
   X509_EXTENSION *wext = 0;
   for (i = 0; i< numext; i++) {
#if OPENSSL_VERSION_NUMBER >= 0x10000000L
      wext = sk_X509_EXTENSION_value(esk, i);
#else /* OPENSSL */
      wext = (X509_EXTENSION *)sk_value(esk, i);
#endif /* OPENSSL */
      if (usenid) {
         int enid = OBJ_obj2nid(X509_EXTENSION_get_object(wext));
         if (enid == nid)
            break;
      } else {
         // Try matching of the text
         char s[256];
         OBJ_obj2txt(s, sizeof(s), X509_EXTENSION_get_object(wext), 1);
         if (!strcmp(s, oid)) 
            break;
      }
      wext = 0;
   }

   // We are done if nothing was found
   if (!wext) {
      DEBUG("Extension "<<oid<<" not found"); 
      return ext;
   }

   // We are done
   return (XrdCryptoX509Reqdata)wext;
}

//_____________________________________________________________________________
XrdSutBucket *XrdCryptosslX509Req::Export()
{
   // Export in form of bucket
   EPNAME("X509Req::Export");

   // If we have already done it, return the previous result
   if (bucket) {
      DEBUG("serialization already performed:"
            " return previous result ("<<bucket->size<<" bytes)");
      return bucket;
   }

   // Make sure we got something to export
   if (!creq) {
      DEBUG("certificate is not initialized");
      return 0;
   }

   //
   // Now we create a bio_mem to serialize the certificate
   BIO *bmem = BIO_new(BIO_s_mem());
   if (!bmem) {
      DEBUG("unable to create BIO for memory operations");
      return 0;
   }
   
   // Write certificate to BIO
   if (!PEM_write_bio_X509_REQ(bmem, creq)) {
      DEBUG("unable to write certificate request to memory BIO");
      return 0;
   }

   // Extract pointer to BIO data and length of segment
   char *bdata = 0;  
   int blen = BIO_get_mem_data(bmem, &bdata);
   DEBUG("BIO data: "<<blen<<" bytes at 0x"<<(int *)bdata);

   // create the bucket now
   bucket = new XrdSutBucket(0,0,kXRS_x509_req);
   if (bucket) {
      // Fill bucket
      bucket->SetBuf(bdata, blen);
      DEBUG("result of serialization: "<<bucket->size<<" bytes");
   } else {
      DEBUG("unable to create bucket for serialized format");
      BIO_free(bmem);
      return 0;
   }
   //
   // Free BIO
   BIO_free(bmem);
   //
   // We are done
   return bucket;
}

//_____________________________________________________________________________
bool XrdCryptosslX509Req::Verify()
{
   // Verify signature of the request 
   EPNAME("X509Req::Verify");

   // We must have been initialized
   if (!creq)
      return 0;

   // Ok: we can verify
   int rc = X509_REQ_verify(creq,X509_REQ_get_pubkey(creq));
   if (rc <= 0) {
     // Failure
     if (rc == 0) {
       // Signatures are not OK
       DEBUG("signature not OK");
     } else {
       // General failure
       DEBUG("could not verify signature");
     }
     return 0;
   }
   // OK
   return 1;
}
