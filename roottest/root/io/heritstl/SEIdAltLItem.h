////////////////////////////////////////////////////////////////////////////
// $Id$
//
// SEIdAltLItem
//
// SEIdAltLItem is a strip-end alternative list (vector) item
//
// Author:  R. Hatcher 2001.10.22
//
////////////////////////////////////////////////////////////////////////////

#ifndef SEIDALTLITEM_H
#define SEIDALTLITEM_H

// not inheriting from TObject so we need an explicit Rtypes
#include "Rtypes.h"

#include "Riostream.h"

class SEIdAltLItem {

   friend std::ostream& operator<<(std::ostream& os, const
                                   SEIdAltLItem& item);

public:

   SEIdAltLItem() :   // null ctor
      fWeight(0), fPE(0), fSigLin(0), fSigCorr(0), fTime(-1) { };

   SEIdAltLItem(Int_t seid, Float_t wgt=0,
                Float_t pe=0, Float_t siglin=0, Float_t sigcorr=0,
                Double_t time=0) : // basic ctor
      fStripEndId(seid), fWeight(wgt), fPE(pe), 
      fSigLin(siglin), fSigCorr(sigcorr),
      fTime(time) { };

   SEIdAltLItem(const SEIdAltLItem& that) // copy ctor
      { *this = that; }

   virtual ~SEIdAltLItem() { ; } // dtor

   Int_t          GetSEId() const { return fStripEndId; }
   Float_t        GetWeight() const { return fWeight; }
   Bool_t         IsZeroWeight() const { return fWeight == 0; }
   Float_t        GetPE() const { return fPE; }
   Float_t        GetSigLin() const { return fSigLin; }
   Float_t        GetSigCorr() const { return fSigCorr; }
   Double_t       GetTime() const { return fTime; }

   void           SetWeight(Float_t wgt) { fWeight = wgt; }
   void           AddToWeight(Float_t wgtadd) { fWeight += wgtadd; }
                    
   void           SetPE(Float_t pe) { fPE = pe; }
   void           SetSigLin(Float_t siglin) { fSigLin = siglin; }
   void           SetSigCorr(Float_t sigcorr) { fSigCorr = sigcorr; }
   void           SetTime(Double_t time) { fTime = time; }

   void           AddToTime(Double_t tadd) { fTime += tadd; }

   friend Bool_t  operator==(const SEIdAltLItem &lhs,
                             const SEIdAltLItem &rhs);

   // sort (by default) based on weight
   friend Bool_t  operator<(const SEIdAltLItem &lhs,
                            const SEIdAltLItem &rhs);

   virtual void   Print(Option_t *option="") const;

protected:

   Int_t             fStripEndId; // which strip
   Float_t           fWeight;     // stored demux weight
   Float_t           fPE;         // ADC value converted to photoelectrons
   Float_t           fSigLin;     // correct for time depend & linearity by LI
   Float_t           fSigCorr;    // correct for strip-to-strip variations
   Double_t          fTime;       // timestamp, corrected for offsets and scale differences. 

private:

   ClassDef(SEIdAltLItem,1)
};

inline Bool_t operator==(const SEIdAltLItem& lhs, 
                         const SEIdAltLItem& rhs)
{
   return lhs.fStripEndId == rhs.fStripEndId;
}

inline Bool_t operator<(const SEIdAltLItem& lhs, 
                         const SEIdAltLItem& rhs)
{
   return lhs.fWeight < rhs.fWeight;
}

#endif // SEIDALTLITEM_H
