#include <thread>
#include "TClassTable.h"
#include "TROOT.h"
#include <dlfcn.h>
#include <iostream>

void repeatopen(int n)
{
   for(int i = 0; i < n; ++i) {
     auto r = dlopen("./libUser.so", RTLD_LAZY);
     if (!r) {
        std::cerr << "Failed to open libUser.so at iteration " << i << "\n";
        return;
     }
     // Waste some time
     for(int j = 0; j < 10; ++j)
       auto d = TClassTable::GetDictNorm("UserClass");
     // std::cerr << "Seeing dict pointer: " << (void*)d << '\n';
     dlclose(r);
   }
}

void repeatlookup(int n)
{
   static DictFuncPtr_t value = nullptr;

   for(int i = 0; i < n; ++i) {
     auto r = TClassTable::GetDictNorm("UserClass");
     if (!value && r)
       value = r;
     // std::cerr << "Value of dictionary pointer has from " << (void*)value << " to " << (void*)r << '\n';
     if (r != nullptr && r != value)
       std::cerr << "Value of dictionary pointer changed from " << (void*)value << " to " << (void*)r << '\n';
   }
}


int exec_dlopenrace()
{
   ROOT::EnableThreadSafety();
   constexpr auto iter = 10000;
   std::thread openlib{repeatopen, iter};
   std::thread lookup{repeatlookup, iter};
   openlib.join();
   lookup.join();
   return 0;
}
