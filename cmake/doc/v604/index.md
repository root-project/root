## CMake infrastructure

### New functionalities

- Support ARM 64 bits architecture.

- Partial support for PPC 64 bits Little Endian architecture.

- Add "Optimized" CMAKE_BUILD_TYPE: allow highest level of optimisation of the GCC and Clang compilers (-Ofast).

- Support ccache activation with cmake configuration switch.

- Support link to jemalloc and tcmalloc allocators.

- Careful suppression of known and understood warnings, e.g. coming from external packages.
