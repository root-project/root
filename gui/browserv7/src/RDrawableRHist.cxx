/// \file RDrawableRHist.cxx
/// \ingroup rbrowser
/// \author Bertrand Bellenot <bertrand.bellenot@cern.ch>
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2019-10-17
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include <ROOT/RDrawableProvider.hxx>
#include <ROOT/RHistDrawable.hxx>

#include <ROOT/RCanvas.hxx>

using namespace ROOT::Experimental;

class RV7HistDrawProvider : public RDrawableProvider {
protected:

   bool DoDrawV7(std::shared_ptr<RPadBase> &subpad, std::unique_ptr<Browsable::RObject> &obj, const std::string &) const override
   {
      auto hist = obj->get_shared<RH2D>();
      if (!hist) return false;

      if (subpad->NumPrimitives() > 0) {
         subpad->Wipe();
         subpad->GetCanvas()->Modified();
         subpad->GetCanvas()->Update(true);
      }

      subpad->Draw(hist);

      return true;
   }
};

struct RV7HistDrawProviderReg {
   std::shared_ptr<RV7HistDrawProvider> provider;
   RV7HistDrawProviderReg()
   {
      provider = std::make_shared<RV7HistDrawProvider>();
      // RDrawableProvider::RegisterV7(TObject::Class(), provider);
      RDrawableProvider::RegisterV7(TClass::GetClass<RH2D>(), provider);
   }
   ~RV7HistDrawProviderReg() { RDrawableProvider::Unregister(provider); }
} newRV7HistDrawProviderReg;

