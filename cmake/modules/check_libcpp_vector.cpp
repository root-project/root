#include <cstddef>
#include <vector>

#if defined(_MSC_VER)
  #define NOINLINE __declspec(noinline)
#else
  #define NOINLINE __attribute__((noinline))
#endif

// Don't inline to prevent the optimizer from interfering with the test
NOINLINE static std::size_t GetTypeErasedSize(const void *vec)
{
   return reinterpret_cast<const std::vector<char> *>(vec)->size();
}

int main()
{
   std::vector<int> v(1);
   const auto size = GetTypeErasedSize(reinterpret_cast<const void *>(&v));
   if (size == sizeof(int)) {
      return 0;
   } else if (size == 1) {
      return 1; // New ABI
   } else {
      return 2; // Unexpected
   }
}
