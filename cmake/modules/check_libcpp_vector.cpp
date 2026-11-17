#include <vector>
#include <cstring>
#include <cstdint>

int main()
{
   std::vector<int> v{1};
   // The standard layout should use three pointers, so let's check:
   struct S {
      std::uintptr_t start, finish, capacity;
   } s;
   static_assert(sizeof(s) == sizeof(v), "Can't test vector's in-memory layout. Sizes don't match.");

   std::memcpy(&s, &v, sizeof(S));

   if (s.finish - s.start == sizeof(int)) {
      return 0;
   } else if (s.finish == 1) {
      return 1; // New ABI
   } else {
      return 2; // Unexpected
   }
}
