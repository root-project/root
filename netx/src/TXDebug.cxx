/***********************************************************/
/*                T X D e b u g . c c                      */
/*                        2003                             */
/*             Produced by Alvise Dorigo                   */
/*         & Fabrizio Furano for INFN padova               */
/***********************************************************/
//
//   $Id: TXDebug.cc,v 1.1 2004/05/10 13:58:56 dorigoa Exp $
//
// Author: Alvise Dorigo, Fabrizio Furano

#include "TXDebug.h"
#include "TError.h"
#include "TSystem.h"

TXDebug *TXDebug::fgInstance = 0;

//_____________________________________________________________________________
TXDebug* TXDebug::Instance()
{
   // Create unique instance

   if (!fgInstance) {
      fgInstance = new TXDebug;
      if (!fgInstance) {
         Error("TXDebug::Instance", "Fatal ERROR *** Object creation with new"
               " failed ! Probable system resources exhausted.");
         gSystem->Abort();
      }
   }
   return fgInstance;
}

//_____________________________________________________________________________
TXDebug::TXDebug()
{
   // Constructor

   fDbgLevel = gEnv->GetValue("XNet.Debug", 0);
}

//_____________________________________________________________________________
TXDebug::~TXDebug()
{
   // Destructor

   SafeDelete(fgInstance);
}
