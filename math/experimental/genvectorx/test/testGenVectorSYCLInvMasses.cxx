#include "MathX/LorentzRotation.h"
#include "MathX/PtEtaPhiM4D.h"
#include "MathX/LorentzVector.h"

#include "Math/LorentzRotation.h"
#include "Math/PtEtaPhiM4D.h"
#include "Math/LorentzVector.h"

#include "MathX/GenVectorX/AccHeaders.h"

#include <sycl/sycl.hpp>

#include <vector>

using namespace ROOT::ROOT_MATH_ARCH;

typedef LorentzVector<PtEtaPhiM4D<double>> vec4dSYCL;
typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>> vec4d;

template <class vec4d>
vec4d *GenVectors(int n)
{
   vec4d *vectors = new vec4d[n];

   // generate n -4 momentum quantities
   for (int i = 0; i < n; ++i) {
      // fill vectors
      vectors[i] = {1., 1., 1., 1.};
   }

   return vectors;
}

double *InvariantMassesSYCL(int N)
{
   sycl::default_selector device_selector;
   sycl::queue queue(device_selector);

   auto v1 = GenVectors<LorentzVector<PtEtaPhiM4D<double>>>(N);
   auto v2 = GenVectors<LorentzVector<PtEtaPhiM4D<double>>>(N);
   double *invMasses = new double[N];

   std::cout << "sycl::queue check - selected device:\n"
             << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

   {

      vec4dSYCL *d_v1 = sycl::malloc_device<vec4dSYCL>(N, queue);
      vec4dSYCL *d_v2 = sycl::malloc_device<vec4dSYCL>(N, queue);
      double *d_invMasses = sycl::malloc_device<double>(N, queue);

      queue.memcpy(d_v1, v1, N * sizeof(vec4dSYCL));
      queue.memcpy(d_v2, v2, N * sizeof(vec4dSYCL));
      queue.wait();

      queue.submit([&](sycl::handler &cgh) {
         cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> indx) {
            vec4dSYCL v = d_v2[indx] + d_v2[indx];
            d_invMasses[indx] = v.M();
         });
      });

      queue.wait();
      queue.memcpy(invMasses, d_invMasses, N * sizeof(double));
      queue.wait();
   }

   return invMasses;
}

double *InvariantMasses(int N)
{
   auto v1 = GenVectors<ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>>(N);
   auto v2 = GenVectors<ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>>(N);
   double *invMasses = new double[N];

   for (int i = 0; i < N; i++) {
      invMasses[i] = (v1[i] + v2[i]).M();
   }

   return invMasses;
}

int main()
{
   int n = 128;
   std::cout << "testing Invariant Masses Computation \t:\n";
   double *msycl = InvariantMassesSYCL(n);
   double *m = InvariantMasses(n);

   int iret = 0;

   for (int i = 0; i < n; i++) {
      iret += (std::abs(msycl[i] - m[i]) > 1e-5);
   }

   if (iret)
      std::cerr << "\t\t\t\t\t FAILED\n " << std::endl;
   else
      std::cout << "\t\t\t\t\t OK\n " << std::endl;
   return iret;
}
