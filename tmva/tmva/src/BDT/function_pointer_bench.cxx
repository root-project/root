#include <functional>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

typedef unsigned long long counter_t;

struct Counter {
   volatile counter_t bare;
   volatile counter_t cxx;
   volatile counter_t cxo1;
   volatile counter_t virt;
   volatile counter_t lambda;

   Counter() : bare(0), cxx(0), cxo1(0), virt(0), lambda(0) {}
} counter;

void bare(Counter *counter)
{
   __sync_fetch_and_add(&counter->bare, 1);
}
void cxx(Counter *counter)
{
   __sync_fetch_and_add(&counter->cxx, 1);
}

struct CXO1 {
   void         cxo1(Counter *counter) { __sync_fetch_and_add(&counter->cxo1, 1); }
   virtual void virt(Counter *counter) { __sync_fetch_and_add(&counter->virt, 1); }
} cxo1;

void (*bare_cb)(Counter *) = nullptr;
std::function<void(Counter *)> cxx_cb;
std::function<void(Counter *)> cxo1_cb;
std::function<void(Counter *)> virt_cb;
std::function<void(Counter *)> lambda_cb;

void *bare_main(void *p)
{
   while (true) {
      bare_cb(&counter);
   }
}
void *cxx_main(void *p)
{
   while (true) {
      cxx_cb(&counter);
   }
}
void *cxo1_main(void *p)
{
   while (true) {
      cxo1_cb(&counter);
   }
}
void *virt_main(void *p)
{
   while (true) {
      virt_cb(&counter);
   }
}
void *lambda_main(void *p)
{
   while (true) {
      lambda_cb(&counter);
   }
}

int main()
{
   pthread_t bare_thread;
   pthread_t cxx_thread;
   pthread_t cxo1_thread;
   pthread_t virt_thread;
   pthread_t lambda_thread;

   bare_cb   = &bare;
   cxx_cb    = std::bind(&cxx, std::placeholders::_1);
   cxo1_cb   = std::bind(&CXO1::cxo1, &cxo1, std::placeholders::_1);
   virt_cb   = std::bind(&CXO1::virt, &cxo1, std::placeholders::_1);
   lambda_cb = [](Counter *counter) { __sync_fetch_and_add(&counter->lambda, 1); };

   pthread_create(&bare_thread, nullptr, &bare_main, nullptr);
   pthread_create(&cxx_thread, nullptr, &cxx_main, nullptr);
   pthread_create(&cxo1_thread, nullptr, &cxo1_main, nullptr);
   pthread_create(&virt_thread, nullptr, &virt_main, nullptr);
   pthread_create(&lambda_thread, nullptr, &lambda_main, nullptr);

   for (unsigned long long n = 1; true; ++n) {
      sleep(1);
      Counter c = counter;

      printf("%15llu bare function pointer\n"
             "%15llu C++11 function object to bare function\n"
             "%15llu C++11 function object to object method\n"
             "%15llu C++11 function object to object method (virtual)\n"
             "%15llu C++11 function object to lambda expression %30llu-th second.\n\n",
             c.bare, c.cxx, c.cxo1, c.virt, c.lambda, n);
   }
}
