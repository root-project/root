#ifndef ROO_MULTIPDF
#define ROO_MULTIPDF

#include "RooAbsPdf.h"
#include "RooCategory.h"
#include "RooCategoryProxy.h"
#include "RooListProxy.h"

class RooMultiPdf : public RooAbsPdf {
public:
   enum PenaltyScheme {
      PVAL,
      AIC
   };
   RooMultiPdf() {}
   RooMultiPdf(const char *name, const char *title, RooCategory &, const RooArgList &_c);
   RooMultiPdf(const RooMultiPdf &other, const char *name = nullptr);
   TObject *clone(const char *newname) const override { return new RooMultiPdf(*this, newname); }

   inline bool checkIndexDirty() const { return _oldIndex != x; }
   inline double getCorrection() const { return cFactor * static_cast<RooAbsReal *>(corr.at(x))->getVal(); }
   inline RooAbsPdf *getCurrentPdf() const { return getPdf(getCurrentIndex()); }
   int getNumPdfs() const { return c.size(); }
   void setCorrectionFactor(PenaltyScheme penal) { cFactor = penal == AIC ? 1.0 : 0.5; }
   void setCorrectionFactor(double penal) { cFactor = penal; }
   inline int getCurrentIndex() const { return static_cast<int>(x); }
   inline RooAbsPdf *getPdf(int index) const { return static_cast<RooAbsPdf *>(c.at(index)); }
   // Always normalized because each pdf is normalized
   bool selfNormalized() const override { return true; }

protected:
   RooListProxy c;
   RooListProxy corr;
   RooCategoryProxy x;

   int fIndex; // sigh, there should be a better way than this
   int nPdfs;  // not used, kept so we didn't have to change the class layout for IO
   mutable Int_t _oldIndex;

   Double_t evaluate() const override;
   Double_t getLogVal(const RooArgSet *set = nullptr) const override;
   double cFactor = 0.5; // correction to 2*NLL by default is -> 2*0.5 per param

private:
   ClassDefOverride(RooMultiPdf, 1) // Multi PDF
};
#endif
