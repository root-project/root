#ifndef TMVA_RSTANDARDSCALER
#define TMVA_RSTANDARDSCALER

#include <TFile.h>

#include <TMVA/RTensor.hxx>
#include <ROOT/RStringView.hxx>

#include <cmath>
#include <vector>

namespace TMVA {
namespace Experimental {

template <typename T>
class RStandardScaler {
private:
   std::vector<T> fMeans;
   std::vector<T> fStds;

public:
   RStandardScaler() = default;
   RStandardScaler(std::string_view title, std::string_view filename);
   void Fit(const RTensor<T>& x);
   std::vector<T> Compute(const std::vector<T>& x);
   RTensor<T> Compute(const RTensor<T>& x);
   std::vector<T> GetMeans() const { return fMeans; }
   std::vector<T> GetStds() const { return fStds; }
   void Save(std::string_view title, std::string_view filename);
};

template <typename T>
inline RStandardScaler<T>::RStandardScaler(std::string_view title, std::string_view filename) {
    auto file = TFile::Open(filename.data(), "READ");
    RStandardScaler<T>* obj;
    file->GetObject(title.data(), obj);
    fMeans = obj->GetMeans();
    fStds = obj->GetStds();
    delete obj;
    file->Close();
}

template <typename T>
inline void RStandardScaler<T>::Save(std::string_view title, std::string_view filename) {
   auto file = TFile::Open(filename.data(), "UPDATE");
   file->WriteObject<RStandardScaler<T>>(this, title.data());
   file->Write();
   file->Close();
}

template <typename T>
inline void RStandardScaler<T>::Fit(const RTensor<T>& x) {
   const auto shape = x.GetShape();
   if (shape.size() != 2)
      throw std::runtime_error("Can only fit to input tensor of rank 2.");
   fMeans.clear();
   fMeans.resize(shape[1]);
   fStds.clear();
   fStds.resize(shape[1]);

   // Compute means
   for (std::size_t i = 0; i < shape[0]; i++) {
      for (std::size_t j = 0; j < shape[1]; j++) {
         fMeans[j] += x(i, j);
      }
   }
   for (std::size_t i = 0; i < shape[1]; i++) {
      fMeans[i] /= shape[0];
   }

   // Compute standard deviations using unbiased estimator
   for (std::size_t i = 0; i < shape[0]; i++) {
      for (std::size_t j = 0; j < shape[1]; j++) {
         fStds[j] += (x(i, j) - fMeans[j]) * (x(i, j) - fMeans[j]);
      }
   }
   for (std::size_t i = 0; i < shape[1]; i++) {
      fStds[i] = std::sqrt(fStds[i] / (shape[0] - 1));
   }
}

template <typename T>
inline std::vector<T> RStandardScaler<T>::Compute(const std::vector<T>& x) {
   const auto size = x.size();
   if (size != fMeans.size())
      throw std::runtime_error("Size of input vector is not equal to number of fitted variables.");

   std::vector<T> y(size);
   for (std::size_t i = 0; i < size; i++) {
      y[i] = (x[i] - fMeans[i]) / fStds[i];
   }

   return y;
}

template <typename T>
inline RTensor<T> RStandardScaler<T>::Compute(const RTensor<T>& x) {
   const auto shape = x.GetShape();
   if (shape.size() != 2)
      throw std::runtime_error("Can only compute output for input tensor of rank 2.");
   if (shape[1] != fMeans.size())
      throw std::runtime_error("Second dimension of input tensor is not equal to number of fitted variables.");

   RTensor<T> y(shape);
   for (std::size_t i = 0; i < shape[0]; i++) {
      for (std::size_t j = 0; j < shape[1]; j++) {
         y(i, j) = (x(i, j) - fMeans[j]) / fStds[j];
      }
   }

   return y;
}

} // namespace Experimental
} // namespace TMVA

#endif // TMVA_RSTANDARDSCALER
