#ifndef ROO_MULTIREAL
#define ROO_MULTIREAL

#include "RooAbsReal.h"
#include "RooCategory.h"
#include "RooCategoryProxy.h"
#include "RooListProxy.h"

class RooMultiReal : public RooAbsReal {
public:
   RooMultiReal() = default;
   RooMultiReal(const char *name, const char *title, RooCategory &indexCat, const RooArgList &models);
   RooMultiReal(const RooMultiReal &other, const char *name = nullptr);
   virtual ~RooMultiReal() {}

   TObject *clone(const char *newname) const override { return new RooMultiReal(*this, newname); }

   inline int getCurrentIndex() const { return static_cast<int>(_index); }
   inline RooAbsReal *getCurrentReal() const { return static_cast<RooAbsReal *>(_models.at(getCurrentIndex())); }
   inline int getNumModels() const { return _models.getSize(); }
   inline const RooCategoryProxy &indexCategory() const { return _index; }
   inline const RooListProxy &getModelList() const { return _models; }

   void getParametersHook(const RooArgSet *nset, RooArgSet *list, bool stripDisconnected) const override;

protected:
   RooListProxy _models;    // list of RooAbsReal models
   RooCategoryProxy _index; // index category proxy

   Double_t evaluate() const override;

private:
   ClassDefOverride(RooMultiReal, 1)
};

#endif