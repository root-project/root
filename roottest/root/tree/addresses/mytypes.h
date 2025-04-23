#include <vector>

namespace mytypes {
   struct FirstBase {
      FirstBase() : fA(-1),fB(-1) {}
      double fA;
      double fB;
   };
   struct OtherBase {
      OtherBase() : fPx(-1),fPy(-1) {}
      double fPx;
      double fPy;
   };

   struct Trajectory : public FirstBase,std::vector<int>,OtherBase {};

   struct MyEntry {
      MyEntry()
      {
         for (int i=0; i < 8; i++) foo[i] = i;
      }
      unsigned char foo[8];
      Trajectory m_trajectory;
   };

   struct Collection : std::vector<mytypes::MyEntry> 
   {
      std::vector<mytypes::MyEntry> m_entries;
   };
}
