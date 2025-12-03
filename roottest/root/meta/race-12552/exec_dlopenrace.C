#include <thread>
#include "TClassTable.h"
#include "TROOT.h"
#include <dlfcn.h>
#include <iostream>
#include <atomic>

std::atomic<int> gOpenDone(0);
std::atomic<int> gLookupDone(0);

void repeatopen(int n)
{
   for(int i = 0; i < n; ++i) {
     if (i % 100 == 0) std::cerr << "Opening library #" << i << '\n';
     if (gLookupDone) {
        // No need to continue as the lookup has stopped
	std::cout << "Stopping after Opening library #" << i << '\n';
        break;
     }
     auto r = dlopen("./libUser.so", RTLD_LAZY);
     if (!r) {
        std::cerr << "Failed to open libUser.so at iteration " << i << "\n";
        return;
     }
     // Waste some time
     for(int j = 0; j < 10; ++j)
       auto d = TClassTable::GetDictNorm("UserClass");
     dlclose(r);
   }
   gOpenDone = 1;
   std::cout << "Done with opening the libraries\n";
}

void repeatlookup(Long64_t n)
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
   gLookupDone = 1;
   std::cout << "Done with doing lookups\n";
}


int exec_dlopenrace(int iter = 1000)
{
   // Preload some of the libraries to reduce but not eliminate
   // the number of libraries that unloaded and reloaded at each
   // iteration.
   // Note only some platform (eg. EL8 with clang) will actually
   // unload the dependent libraries as part of the library's dlclose.
   // When this happens (at least for now) glibc's cxa_atexit.c and cxa_finalize.c
   // accumulate atexit function/structure such that the cost of loop through
   // them increase (in this example) in O(N^2) of the number of dlopen/dlclose
   // and libraries actually (un)loaded.
   // We keep some to actually exercise the unloading and reloading of complex
   // dictionaries.
   gSystem->Load("libGpad");
   // If we load this then only libUser is loaded and unloaded.
   // gSystem->Load("libTree");

   ROOT::EnableThreadSafety();
   std::thread openlib{repeatopen, iter};
   std::thread lookup{repeatlookup, 10000*iter};
   openlib.join();
   lookup.join();
   return 0;
}
