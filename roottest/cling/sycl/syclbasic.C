#include <sycl/sycl.hpp>
#include <iostream>

void syclbasic()
{
   sycl::queue q{sycl::cpu_selector_v}; // Only openMP CPU backend is supported right now
   std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

   int a = 2, b = 3;
   int *sum = sycl::malloc_shared<int>(1, q);

   q.single_task([=] { *sum = a + b; }).wait();
   std::cout << "Sum = " << *sum << '\n';

   sycl::free(sum, q);
}
