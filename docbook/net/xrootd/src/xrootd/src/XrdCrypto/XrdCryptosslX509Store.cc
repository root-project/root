// $Id$

const char *XrdCryptosslX509StoreCVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*               X r d C r y p t o s s l X 5 0 9 S t o r e . c c              */
/*                                                                            */
/* (c) 2005 G. Ganis , CERN                                                   */
/*                                                                            */
/******************************************************************************/


/* ************************************************************************** */
/*                                                                            */
/* OpenSSL implementation of XrdCryptoX509Store                               */
/*                                                                            */
/* ************************************************************************** */

#include <XrdCrypto/XrdCryptosslX509Store.hh>


//_____________________________________________________________________________
XrdCryptosslX509Store::XrdCryptosslX509Store(XrdCryptoX509 *xca) :
                       XrdCryptoX509Store()
{
   // Constructor

   chain = 0;
   store = X509_STORE_new();
   if (store) {
      // Init with CA certificate
      X509_STORE_set_verify_cb_func(store,0);
      // add CA certificate
      X509_STORE_add_cert(store,xca->cert);
      // Init chain
      if (!(chain = sk_X509_new_null())) {
         // Cleanup, if init failure
         X509_STORE_free(store);
         store = 0;
      }
   }
}

//_____________________________________________________________________________
bool XrdCryptosslX509Store::IsValid()
{
   // Test validity

   return (store && chain);
}

//_____________________________________________________________________________
void XrdCryptoX509Store::Dump()
{
   // Dump content
   ABSTRACTMETHOD("XrdCryptoX509Store::Dump");
}

//_____________________________________________________________________________
int XrdCryptoX509Store::Import(XrdSutBucket *bck)
{
   // Import certificates contained in bucket bck, if any

   ABSTRACTMETHOD("XrdCryptoX509Store::Add");
   return -1;
}

//_____________________________________________________________________________
bool XrdCryptoX509Store::Verify()
{
   // Verify certicate chain stored
   ABSTRACTMETHOD("XrdCryptoX509Store::Verify");
   return -1;
}
