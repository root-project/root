////////////////////////////////////////////////////////////////////////////
// $Id$
//
// SEIdAltL
//
// SEIdAltL is a vector+iterator for SEIdAltLItems 
//
// Author:  R. Hatcher 2001.10.22
//
////////////////////////////////////////////////////////////////////////////

#ifndef SEIDALTL_H
#define SEIDALTL_H

// not inheriting from TObject so we need an explicit Rtypes
#include "Rtypes.h"

#include <iosfwd>

#include "SEIdAltLItem.h"

#include <vector>
class PlexCalib;

class SEIdAltL : public std::vector<SEIdAltLItem> {
  
   friend std::ostream& operator<<(std::ostream& os, 
                                   const SEIdAltL& alt);
 
public:

   SEIdAltL();                        // necessary for streamer io
   SEIdAltL(const SEIdAltL &rhs);             // need deep copy
   virtual ~SEIdAltL();
//   SEIdAltL& operator=(const SEIdAltL &rhs);  // need deep copy

   typedef SEIdAltL::iterator          SEIdAltLIter;
   typedef SEIdAltL::const_iterator    SEIdAltLConstIter;

   void           AddStripEndId(const Int_t& pseid, Float_t weight=0,
                                const PlexCalib* calib=0,
                                Int_t adc=0, Double_t time=0);
   void           ClearWeights();
   void           DropCurrent();
   void           DropZeroWeights();
   void           KeepTopWeights(UInt_t n=2, Bool_t keeporder=kFALSE);

   const SEIdAltLItem&   GetBestItem() const;
         SEIdAltLItem&   GetBestItem();
   Int_t                 GetBestSEId() const;
   Float_t               GetBestWeight() const;
   const SEIdAltLItem&   GetCurrentItem() const;
         SEIdAltLItem&   GetCurrentItem();
   Int_t                 GetCurrentSEId() const;
   Float_t               GetCurrentWeight() const; 

   inline Int_t   GetSize() const { return this->size(); } 

   Bool_t         IsValid() const;
   inline void    Next()     { fCurrent++; }
   inline void    Previous() { fCurrent--; }
   inline void    SetFirst() { fCurrent = 0; }
   inline void    SetLast()  { fCurrent = GetSize()-1; }

   void           SetCurrentWeight(Float_t weight);
   void           AddToCurrentWeight(Float_t wgtadd);
   void           NormalizeWeights(Float_t wgtsum = 1.0);

   void           Print(Option_t *option="") const;

   typedef enum EErrorMasks {
      kOkay         = 0x00,
      kBadDetector  = 0x01,
      kBadEnd       = 0x02,
      kBadPlane     = 0x04,
      kBadPlaneView = 0x08,
      kUnchecked    = 0x10
   } ErrorMask_t;

 protected:

 private:

   void           TestConsistency() const;

           UInt_t fCurrent;      // current position
   mutable Int_t  fError;        // -1=unchecked, 0=no consistency errors
                                 // 0x01=det, 0x02=end, 0x04=pln, 0x08=view

   ClassDef(SEIdAltL,3)
};

#endif // SEIDALTL_H
