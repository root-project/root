#include <Riostream.h>

#include "NdbDefs.h"
#include "NdbEndfIO.h"
#include "NdbMTReactionXS.h"

ClassImp(NdbMTReactionXS);

/* -------- LoadENDF -------- */
Bool_t
NdbMTReactionXS::LoadENDF( char *filename )
{
   NdbEndfIO   endf(filename,TENDF_READ);
   Bool_t      error;

   if (!endf.IsOpen()) return kTRUE;

   minxs = MAX_REAL;
   maxxs = -MAX_REAL;

   endf.FindMFMT(3,MT());      // Find total cross section
   endf.ReadReal(&error);   // ??
   endf.ReadReal(&error);
   endf.ReadLine();
   QM  = endf.ReadReal(&error);
   QI  = endf.ReadReal(&error);
   endf.ReadInt(&error);     // Skip number

   LR = endf.ReadInt(&error); if (error) return error;
   NR = endf.ReadInt(&error); if (error) return error;
   NP = endf.ReadInt(&error); if (error) return error;
   endf.ReadLine();          // Skip line

   IT = endf.ReadInt(&error,2);   // Interpolation type
   endf.ReadLine();          // Skip line

   ene.Set(NP);
   xs.Set(NP);

   for (int i=0; i<NP; i++) {
      Float_t   f;
      ene.AddAt(endf.ReadReal(&error), i);
      xs.AddAt( f = endf.ReadReal(&error), i);
      minxs = MIN(minxs,f);
      maxxs = MAX(maxxs,f);
   }
   return kFALSE;
} // loadENDF

/* -------- BinSearch -------- */
Int_t
NdbMTReactionXS::BinSearch( Float_t e )
{
   Int_t   low = 0;
   Int_t   high = NP-1;
   Int_t   mid;

   if (e < ene[low])  return NOTFOUND;
   if (e > ene[high]) return NOTFOUND;

   while (1) {
      mid = (low+high)/2;
      if (mid==low) return mid;
      if (e > ene[mid])
      low = mid;
      else
      if (e < ene[mid])
      high = mid;
      else
      return mid;
   }
} // BinSearch

/* -------- Interpolate -------- */
Float_t
NdbMTReactionXS::Interpolate( Float_t e )
{
   Int_t   p = BinSearch(e);

   if (p==NOTFOUND)
   return xs[ e<ene[0]? 0 : NP-1 ];

   if (p==NP-1)
   return xs[p];

   Float_t   el = ene[p];
   Float_t   xl = xs[p];
   p++;

   switch (IT) {
      case IT_LINLOG:
         std::cout << "Linear-Log interpolation" << std::endl;
         return 0.0;

      case IT_LOGLIN:
         std::cout << "Log-Linear interpolation" << std::endl;
         return 0.0;

      case IT_LOGLOG:
         std::cout << "Log-Log interpolation" << std::endl;
         return 0.0;

      case IT_GAMOW:
         std::cout << "GAMOW interpolation" << std::endl;
         return 0.0;

      case IT_LINLIN:
      return xs[p] + (e-el) * (xs[p]-xl) / (ene[p]-el);

      default:
      break;
   }
   return 0.0;
} // interpolate
