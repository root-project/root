/// \file
/// \ingroup tutorial_heterogeneous
/// Introductory SYCL examples inside ROOT’s interpreter.
///
/// \note This tutorial requires ROOT to be built with **SYCL support** enabled.
/// Configure CMake with:
///   `-Dexperimental_adaptivecpp=ON`
/// to enable SYCL support in the interpreter.
///
/// This tutorial contains two examples:
///   - `sycladd()`: a minimal kernel computing a sum of two integers
///   - `syclvectoradd()`: adding two vectors
///
/// \macro_code
///
/// \author Devajith Valaparambil Sreeramaswamy (CERN)

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

void sycladd()
{
   sycl::queue q{sycl::cpu_selector_v}; // Use openMP CPU backend
   std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

   int a = 2, b = 3;
   int *sum = sycl::malloc_shared<int>(1, q);

   q.single_task([=] { *sum = a + b; }).wait();
   std::cout << "Sum = " << *sum << '\n';

   sycl::free(sum, q);
}

void syclvectoradd()
{
   sycl::queue q{sycl::default_selector_v}; // Portable backend choice
   const size_t N = 16;
   std::vector<int> A(N, 1), B(N, 2), C(N);

   int *a = sycl::malloc_shared<int>(N, q);
   int *b = sycl::malloc_shared<int>(N, q);
   int *c = sycl::malloc_shared<int>(N, q);

   std::copy(A.begin(), A.end(), a);
   std::copy(B.begin(), B.end(), b);

   q.parallel_for(N, [=](sycl::id<1> i) { c[i] = a[i] + b[i]; }).wait();

   std::copy(c, c + N, C.begin());

   std::cout << "C[0] = " << C[0] << ", C[N-1] = " << C[N - 1] << "\n";

   sycl::free(a, q);
   sycl::free(b, q);
   sycl::free(c, q);
}

void syclintro()
{
   sycladd();
   syclvectoradd();
}
