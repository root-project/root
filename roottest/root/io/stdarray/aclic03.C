#include "infoDumper.h"
#include <array>
#include <memory>
namespace edm3 {

   class B{};

   class A{
      A(){
         for (int i=0;i<3;i++){
            a0[0].reset(new B());
            a1[i] = new B();
//             a2[i] = new B();
         }
         // no makeunique

      }
   private:
      std::array<unique_ptr<B>,3> a0 ;
      std::array<B*,3> a1 ;
      B a2[3];
   };

}

int aclic03() {

   auto className = "edm3::A";
   dumpInfo(className);

   return 0;
}

