// $Id$

const char *XrdCryptoAuxCVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*                      X r d C r y p t o A u x . c c                         */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <XrdSys/XrdSysLogger.hh>
#include <XrdSys/XrdSysError.hh>

#include <XrdCrypto/XrdCryptoAux.hh>
#include <XrdCrypto/XrdCryptoTrace.hh>

//
// For error logging and tracing
static XrdSysLogger Logger;
static XrdSysError eDest(0,"crypto_");
XrdOucTrace *cryptoTrace = 0;

/******************************************************************************/
/*  X r d C r y p t o S e t T r a c e                                         */
/******************************************************************************/
//______________________________________________________________________________
void XrdCryptoSetTrace(kXR_int32 trace)
{
   // Set trace flags according to 'trace'

   //
   // Initiate error logging and tracing
   eDest.logger(&Logger);
   if (!cryptoTrace)
      cryptoTrace = new XrdOucTrace(&eDest);
   if (cryptoTrace) {
      // Set debug mask
      cryptoTrace->What = 0;
      // Low level only
      if ((trace & cryptoTRACE_Notify))
         cryptoTrace->What |= cryptoTRACE_Notify;
      // Medium level
      if ((trace & cryptoTRACE_Debug))
         cryptoTrace->What |= (cryptoTRACE_Notify | cryptoTRACE_Debug);
      // High level
      if ((trace & cryptoTRACE_Dump))
         cryptoTrace->What |= cryptoTRACE_ALL;
   }
}
