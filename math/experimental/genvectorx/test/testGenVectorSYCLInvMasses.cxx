#include "MathX/LorentzRotation.h"
#include "MathX/PtEtaPhiM4D.h"
#include "MathX/LorentzVector.h"

#include "MathX/GenVectorX/AccHeaders.h"

#include <sycl/sycl.hpp>

#include <vector>

using namespace ROOT::ROOT_MATH_ARCH;

typedef LorentzVector<PtEtaPhiM4D<double>> vec4d;

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

int testInvariantMasses(int N)
{
   int iret_host = 0;
   std::cout << "testing Invariant Masses Computation \t:\n";

   sycl::default_selector device_selector;
   sycl::queue queue(device_selector);

   auto v1 = GenVectors(N);
   auto v2 = GenVectors(N);
   double *invMasses = new double[N];

   std::cout << "sycl::queue check - selected device:\n"
             << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

   {

      vec4d *d_v1 = sycl::malloc_device<vec4d>(N, queue);
      vec4d *d_v2 = sycl::malloc_device<vec4d>(N, queue);
      double *d_invMasses = sycl::malloc_device<double>(N, queue);

      queue.memcpy(d_v1, v1, N * sizeof(vec4d));
      queue.memcpy(d_v2, v2, N * sizeof(vec4d));
      queue.wait();

      queue.submit([&](sycl::handler &cgh) {
         cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> indx) {
            vec4d v = d_v2[indx] + d_v2[indx];
            d_invMasses[indx] = v.M();
         });
      });

      queue.wait();
      queue.memcpy(invMasses, d_invMasses, N * sizeof(double));
      queue.wait();
   }

   for (int i = 0; i < N; i++) {
      iret_host += (std::abs(invMasses[i] - 2.) > 1e-5);
   }

   if (iret_host == 0)
      std::cout << "\tOK\n";
   else
      std::cout << "\t FAILED\n";

   return iret_host;
}

int main()
{
   int n = 128;
   int ret = testInvariantMasses(n);
   if (ret)
      std::cerr << "test FAILED !!! " << std::endl;
   else
      std::cout << "test OK " << std::endl;
   return ret;
}
