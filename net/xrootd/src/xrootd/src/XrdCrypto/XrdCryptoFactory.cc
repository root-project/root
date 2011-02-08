// $Id$

const char *XrdCryptoFactoryCVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*                  X r d C r y p t o F a c t o r y . c c                     */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* Abstract interface for a crypto factory                                    */
/* Allows to plug-in modules based on different crypto implementation         */
/* (OpenSSL, Botan, ...)                                                      */
/*                                                                            */
/* ************************************************************************** */
#include <string.h>
#include <dlfcn.h>

#include <XrdCrypto/XrdCryptoAux.hh>
#include <XrdCrypto/XrdCryptoTrace.hh>
#include <XrdCrypto/XrdCryptoFactory.hh>
#include <XrdCrypto/XrdCryptolocalFactory.hh>

#ifndef LT_MODULE_EXT
#define LT_MODULE_EXT ".so"
#endif

// We have always an instance of the simple RSA implementation
static XrdCryptolocalFactory localCryptoFactory;

//____________________________________________________________________________
XrdCryptoFactory::XrdCryptoFactory(const char *n, int id)
{
   // Constructor (only called by derived classes).

   if (n) {
      int l = strlen(n);
      l = (l > (MAXFACTORYNAMELEN - 1)) ? (MAXFACTORYNAMELEN - 1) : l;  
      strncpy(name,n,l);
      name[l] = 0;  // null terminated
   }
   fID = id;
}

//______________________________________________________________________________
void XrdCryptoFactory::SetTrace(kXR_int32)
{
   // Set flags for tracing

   ABSTRACTMETHOD("XrdCryptoFactory::SetTrace");
}

//______________________________________________________________________________
bool XrdCryptoFactory::operator==(const XrdCryptoFactory factory)
{
   // Compare name of 'factory' to local name: return 1 if matches, 0 if not

   if (!strcmp(factory.Name(),Name()))
      return 1;
   return 0;
}

//______________________________________________________________________________
XrdCryptoKDFunLen_t XrdCryptoFactory::KDFunLen()
{
   // Return an instance of an implementation of a Key Der function length.

   ABSTRACTMETHOD("XrdCryptoFactory::KDFunLen");
   return 0;
}

//______________________________________________________________________________
XrdCryptoKDFun_t XrdCryptoFactory::KDFun()
{
   // Return an instance of an implementation of a Key Derivation function.

   ABSTRACTMETHOD("XrdCryptoFactory::KDFun");
   return 0;
}

//______________________________________________________________________________
bool XrdCryptoFactory::SupportedCipher(const char *)
{
   // Returns true id specified cipher is supported by the implementation

   ABSTRACTMETHOD("XrdCryptoFactory::SupportedCipher");
   return 0;
}

//______________________________________________________________________________
XrdCryptoCipher *XrdCryptoFactory::Cipher(const char *, int)
{
   // Return an instance of an implementation of XrdCryptoCipher.

   ABSTRACTMETHOD("XrdCryptoFactory::Cipher");
   return 0;
}

//______________________________________________________________________________
XrdCryptoCipher *XrdCryptoFactory::Cipher(const char *, int, const char *, 
                                          int, const char *)
{
   // Return an instance of an implementation of XrdCryptoCipher.

   ABSTRACTMETHOD("XrdCryptoFactory::Cipher");
   return 0;
}

//______________________________________________________________________________
XrdCryptoCipher *XrdCryptoFactory::Cipher(XrdSutBucket *)
{
   // Return an instance of an implementation of XrdCryptoCipher.

   ABSTRACTMETHOD("XrdCryptoFactory::Cipher");
   return 0;
}

//______________________________________________________________________________
XrdCryptoCipher *XrdCryptoFactory::Cipher(int, char *, int, const char *)
{
   // Return an instance of an implementation of XrdCryptoCipher.

   ABSTRACTMETHOD("XrdCryptoFactory::Cipher");
   return 0;
}

//______________________________________________________________________________
XrdCryptoCipher *XrdCryptoFactory::Cipher(const XrdCryptoCipher &)
{
   // Return an instance of an implementation of XrdCryptoCipher.

   ABSTRACTMETHOD("XrdCryptoFactory::Cipher");
   return 0;
}

//______________________________________________________________________________
bool XrdCryptoFactory::SupportedMsgDigest(const char *)
{
   // Returns true id specified digest is supported by the implementation

   ABSTRACTMETHOD("XrdCryptoFactory::SupportedMsgDigest");
   return 0;
}

//______________________________________________________________________________
XrdCryptoMsgDigest *XrdCryptoFactory::MsgDigest(const char *)
{
   // Return an instance of an implementation of XrdCryptoMsgDigest.

   ABSTRACTMETHOD("XrdCryptoFactory::MsgDigest");
   return 0;
}

//______________________________________________________________________________
XrdCryptoRSA *XrdCryptoFactory::RSA(int, int)
{
   // Return an instance of an implementation of XrdCryptoRSA.

   ABSTRACTMETHOD("XrdCryptoFactory::RSA");
   return 0;
}

//______________________________________________________________________________
XrdCryptoRSA *XrdCryptoFactory::RSA(const char *, int)
{
   // Return an instance of an implementation of XrdCryptoRSA.

   ABSTRACTMETHOD("XrdCryptoFactory::RSA");
   return 0;

}

//______________________________________________________________________________
XrdCryptoRSA *XrdCryptoFactory::RSA(const XrdCryptoRSA &)
{
   // Return an instance of an implementation of XrdCryptoRSA.

   ABSTRACTMETHOD("XrdCryptoFactory::RSA ("<<this<<")");
   return 0;
}

//______________________________________________________________________________
XrdCryptoX509 *XrdCryptoFactory::X509(const char *, const char *)
{
   // Return an instance of an implementation of XrdCryptoX509.

   ABSTRACTMETHOD("XrdCryptoFactory::X509");
   return 0;
}

//______________________________________________________________________________
XrdCryptoX509 *XrdCryptoFactory::X509(XrdSutBucket *)
{
   // Init XrdCryptoX509 from a bucket

   ABSTRACTMETHOD("XrdCryptoFactory::X509");
   return 0;
}

//______________________________________________________________________________
XrdCryptoX509Crl *XrdCryptoFactory::X509Crl(const char *, int)
{
   // Return an instance of an implementation of XrdCryptoX509Crl.

   ABSTRACTMETHOD("XrdCryptoFactory::X509Crl");
   return 0;
}

//______________________________________________________________________________
XrdCryptoX509Crl *XrdCryptoFactory::X509Crl(XrdCryptoX509 *)
{
   // Return an instance of an implementation of XrdCryptoX509Crl.

   ABSTRACTMETHOD("XrdCryptoFactory::X509Crl");
   return 0;
}

//______________________________________________________________________________
XrdCryptoX509Req *XrdCryptoFactory::X509Req(XrdSutBucket *)
{
   // Return an instance of an implementation of XrdCryptoX509Req.

   ABSTRACTMETHOD("XrdCryptoFactory::X509Req");
   return 0;
}

//______________________________________________________________________________
XrdCryptoX509VerifyCert_t XrdCryptoFactory::X509VerifyCert()
{
   // Return an instance of an implementation of a verification
   // function for X509 certificate.

   ABSTRACTMETHOD("XrdCryptoFactory::X509VerifyCert");
   return 0;
}

//______________________________________________________________________________
XrdCryptoX509VerifyChain_t XrdCryptoFactory::X509VerifyChain()
{
   // Return an instance of an implementation of a verification
   // function for X509 certificate chains.

   ABSTRACTMETHOD("XrdCryptoFactory::X509VerifyChain");
   return 0;
}

//______________________________________________________________________________
XrdCryptoX509ExportChain_t XrdCryptoFactory::X509ExportChain()
{
   // Return an instance of an implementation of a function
   // to export a X509 certificate chain.

   ABSTRACTMETHOD("XrdCryptoFactory::X509ExportChain");
   return 0;
}

//______________________________________________________________________________
XrdCryptoX509ChainToFile_t XrdCryptoFactory::X509ChainToFile()
{
   // Return an instance of an implementation of a function
   // to dump a X509 certificate chain to a file.

   ABSTRACTMETHOD("XrdCryptoFactory::X509ChainToFile");
   return 0;
}

//______________________________________________________________________________
XrdCryptoX509ParseFile_t XrdCryptoFactory::X509ParseFile()
{
   // Return an instance of an implementation of a function
   // to parse a file supposed to contain for X509 certificates.

   ABSTRACTMETHOD("XrdCryptoFactory::X509ParseFile");
   return 0;
}

//______________________________________________________________________________
XrdCryptoX509ParseBucket_t XrdCryptoFactory::X509ParseBucket()
{
   // Return an instance of an implementation of a function
   // to parse a bucket supposed to contain for X509 certificates.

   ABSTRACTMETHOD("XrdCryptoFactory::X509ParseBucket");
   return 0;
}

/* ************************************************************************** */
/*                                                                            */
/*                    G e t C r y p t o F a c t o r y                         */
/*                                                                            */
/* ************************************************************************** */

//
// Structure for the local record
typedef struct {
   XrdCryptoFactory *factory;
   char              factoryname[MAXFACTORYNAMELEN];
   bool              status; 
} FactoryEntry;

//____________________________________________________________________________
XrdCryptoFactory *XrdCryptoFactory::GetCryptoFactory(const char *factoryid)
{
   // Static method to load/locate the crypto factory named factoryid
 
   static FactoryEntry  *factorylist = 0;
   static int            factorynum = 0;
   XrdCryptoFactory     *(*efact)();
   void *libhandle;
   XrdCryptoFactory *factory;
   char factobjname[80], libfn[80], *libloc;
   EPNAME("Factory::GetCryptoFactory");

   //
   // The id must be defined
   if (!factoryid || !strlen(factoryid)) {
      DEBUG("crypto factory ID ("<<factoryid<<") undefined");
      return 0;
   }

   //
   // If the local simple implementation is required return the related pointer
   if (!strcmp(factoryid,"local")) {
      DEBUG("local crypto factory requested");
      return &localCryptoFactory;
   }

   // 
   // Check if already loaded
   if (factorynum) {
      int i = 0;
      for ( ; i < factorynum; i++ ) {
         if (!strcmp(factoryid,factorylist[i].factoryname)) {
            if (factorylist[i].status) {
               DEBUG(factoryid <<" crypto factory object already loaded ("
                               << factorylist[i].factory << ")");
               return factorylist[i].factory;
            } else {
               DEBUG("previous attempt to load crypto factory "
                     <<factoryid<<" failed - do nothing");
               return 0;
            }
         }
      }
   }

   //
   // Create new entry for this factory in the local record
   FactoryEntry *newfactorylist = new FactoryEntry[factorynum+1];
   if (newfactorylist) {
      int i = 0;
      for ( ; i < factorynum; i++ ) {
         newfactorylist[i].factory = factorylist[i].factory;
         newfactorylist[i].status = factorylist[i].status;
         strcpy(newfactorylist[i].factoryname,factorylist[i].factoryname);
      }
      newfactorylist[i].factory = 0;
      newfactorylist[i].status = 0;
      strcpy(newfactorylist[i].factoryname,factoryid);

      // Destroy previous vector
      if (factorylist) delete[] factorylist;

      // Update local list
      factorylist = newfactorylist;
      factorynum++;
   } else
      DEBUG("cannot create local record of loaded crypto factories");

   //
   // Try loading: name of routine to load
   sprintf(factobjname, "XrdCrypto%sFactoryObject", factoryid);

   //
   // Form library name
   snprintf(libfn, sizeof(libfn)-1, "libXrdCrypto%s", LT_MODULE_EXT);
   libfn[sizeof(libfn)-1] = '\0';

   //
   // Determine path
   libloc = libfn;
   DEBUG("loading " <<factoryid <<" crypto factory object from " <<libloc);

   //
   // Try opening the crypto module
   if (!(libhandle = dlopen(libloc, RTLD_NOW))) {
      DEBUG("problems opening shared library " << libloc
             << "(error: "<< dlerror() << ")");
      return 0;
   }

   //
   // Get the factory object creator
   if (!(efact = (XrdCryptoFactory *(*)())dlsym(libhandle, factobjname))) {

      //
      // Try also specific library name
      snprintf(libfn, sizeof(libfn)-1, "libXrdCrypto%s%s", factoryid, LT_MODULE_EXT);
      libfn[sizeof(libfn)-1] = '\0';
      
      //
      // Determine path
      libloc = libfn;
      DEBUG("loading " <<factoryid <<" crypto factory object from " <<libloc);
      
      //
      // Try opening the crypto module
      if (!(libhandle = dlopen(libloc, RTLD_NOW))) {
         DEBUG("problems opening shared library " << libloc
                << "(error: "<< dlerror() << ")");
         return 0;
      }


      //
      // Get the factory object creator
      if (!(efact = (XrdCryptoFactory *(*)())dlsym(libhandle, factobjname))) {
         DEBUG("problems finding crypto factory object creator " << factobjname);
         return 0;
      }
   }

   //
   // Get the factory object
   if (!(factory = (*efact)())) {
      DEBUG("problems creating crypto factory object");
      return 0;
   }

   //
   // Update local record
   factorylist[factorynum-1].factory = factory;
   factorylist[factorynum-1].status = 1;

   return factory;
}
