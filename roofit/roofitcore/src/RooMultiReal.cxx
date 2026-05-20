#include "RooMultiReal.h"
#include "RooConstVar.h"

// Constructor
RooMultiReal::RooMultiReal(const char *name, const char *title, RooCategory &indexCat, const RooArgList &models)
   : RooAbsReal(name, title),
     _models("_models", "List of RooAbsReal models", this), // name, title, owner
     _index("_index", "Index category", this, indexCat)     // name, title, owner, reference to RooCategory

{
   _models.add(models);

   RooCategory &cat = dynamic_cast<RooCategory &>(const_cast<RooAbsCategory &>(_index.arg()));

   for (int i = 0; i < _models.getSize(); ++i) {
      cat.defineType(("model" + std::to_string(i)).c_str(), i);
   }
}

// Copy constructor
RooMultiReal::RooMultiReal(const RooMultiReal &other, const char *name)
   : RooAbsReal(other, name),
     _models("_models", this, other._models), // name, owner, other list proxy
     _index("_index", this, other._index)     // name, owner, other category proxy

{
}

// Evaluate: returns the value of the currently selected RooAbsReal model
Double_t RooMultiReal::evaluate() const
{
   int currentIndex = static_cast<int>(_index);
   if (currentIndex < 0 || currentIndex >= _models.getSize()) {
      // Defensive fallback: invalid index
      return 0.0;
   }

   Double_t val = static_cast<RooAbsReal *>(_models.at(currentIndex))->getVal(_models.nset());

   return val;
}

// Propagate parameter fetching to the current model
void RooMultiReal::getParametersHook(const RooArgSet *nset, RooArgSet *list, bool stripDisconnected) const
{
   if (!stripDisconnected)
      return;

   list->removeAll();

   RooAbsReal *absReal = static_cast<RooAbsReal *>(_models.at(static_cast<int>(_index)));

   if (absReal->isFundamental()) {
      if (!nset || !absReal->dependsOn(*nset)) {
         list->add(*absReal);
      }
      return;
   }

   absReal->getParameters(nset, *list, stripDisconnected);
   list->add(*_index);
}