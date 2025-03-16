#ifndef ROOT_Minuit2_MnFcnCaller
#define ROOT_Minuit2_MnFcnCaller

#include "Minuit2/MnFcn.h"

namespace ROOT {

namespace Minuit2 {

// Helper class to call the MnFcn, cashing the transformed parameters if necessary.
class MnFcnCaller {
public:
   MnFcnCaller(const MnFcn &mfcn) : fMfcn{mfcn}, fIsUserFcn{static_cast<bool>(mfcn.transform())}
   {
      if (!fIsUserFcn)
         return;

      MnUserTransformation const &transform = *fMfcn.transform();

      // get first initial values of parameter (in case some one is fixed)
      fVpar.assign(transform.InitialParValues().begin(), transform.InitialParValues().end());
   }

   double operator()(const MnAlgebraicVector &v)
   {
      if (!fIsUserFcn)
         return fMfcn(v);

      MnUserTransformation const &transform = *fMfcn.transform();

      bool firstCall = fLastInput.size() != v.size();

      fLastInput.resize(v.size());

      for (unsigned int i = 0; i < v.size(); i++) {
         if (firstCall || fLastInput[i] != v(i)) {
            fVpar[transform.ExtOfInt(i)] = transform.Int2ext(i, v(i));
            fLastInput[i] = v(i);
         }
      }

      return fMfcn.CallWithTransformedParams(fVpar);
   }

private:
   MnFcn const &fMfcn;
   bool fIsUserFcn = false;
   std::vector<double> fLastInput;
   std::vector<double> fVpar;
};

} // namespace Minuit2
} // namespace ROOT

#endif
