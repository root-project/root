#define _STRINGIFY(x) #x
#define STRINGIFY(x) _STRINGIFY(x)

// `#pragma message` is supported in well-known compilers including gcc, clang, icc, and MSVC
#pragma message("__cplusplus=" STRINGIFY(__cplusplus))

int main(void)
{
   return 0;
}
