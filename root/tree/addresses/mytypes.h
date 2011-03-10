#include <vector>

namespace mytypes {
   struct Trajectory : public std::vector<int> {};

   struct MyEntry {
      MyEntry()
      {
         for (int i=0; i < 8; i++) foo[i] = i;
      }
      unsigned char foo[8];
      Trajectory m_trajectory;
   };

   struct Collection {
      std::vector<mytypes::MyEntry> m_entries;
   };
}
