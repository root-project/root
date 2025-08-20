# Install script for directory: /home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "clang-resource-headers" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/builtins.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/float.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/inttypes.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/iso646.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/limits.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/module.modulemap"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/stdalign.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/stdarg.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stdarg___gnuc_va_list.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stdarg___va_copy.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stdarg_va_arg.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stdarg_va_copy.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stdarg_va_list.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/stdatomic.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/stdbool.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/stdckdint.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/stddef.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stddef_max_align_t.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stddef_null.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stddef_nullptr_t.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stddef_offsetof.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stddef_ptrdiff_t.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stddef_rsize_t.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stddef_size_t.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stddef_unreachable.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stddef_wchar_t.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stddef_wint_t.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/stdint.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/stdnoreturn.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/tgmath.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/unwind.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/varargs.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/arm_acle.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/arm_cmse.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/armintr.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/arm64intr.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/arm_neon_sve_bridge.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_cuda_builtin_vars.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_cuda_math.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_cuda_cmath.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_cuda_complex_builtins.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_cuda_device_functions.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_cuda_intrinsics.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_cuda_texture_intrinsics.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_cuda_libdevice_declares.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_cuda_math_forward_declares.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_cuda_runtime_wrapper.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/hexagon_circ_brev_intrinsics.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/hexagon_protos.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/hexagon_types.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/hvx_hexagon_protos.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_hip_libdevice_declares.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_hip_cmath.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_hip_math.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_hip_stdlib.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_hip_runtime_wrapper.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/larchintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/lasxintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/lsxintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/msa.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/opencl-c.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/opencl-c-base.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/altivec.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/htmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/htmxlintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/riscv_bitmanip.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/riscv_crypto.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/riscv_ntlh.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/sifive_vector.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/s390intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/vecintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/velintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/velintrin_gen.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/velintrin_approx.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/adcintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/adxintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ammintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/amxcomplexintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/amxfp16intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/amxintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx2intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512bf16intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512bitalgintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512bwintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512cdintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512dqintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512erintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512fintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512fp16intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512ifmaintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512ifmavlintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512pfintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vbmi2intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vbmiintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vbmivlintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vlbf16intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vlbitalgintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vlbwintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vlcdintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vldqintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vlfp16intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vlintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vlvbmi2intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vlvnniintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vlvp2intersectintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vnniintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vp2intersectintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vpopcntdqintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vpopcntdqvlintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avxifmaintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avxintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avxneconvertintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avxvnniint16intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avxvnniint8intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avxvnniintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/bmi2intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/bmiintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cetintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cldemoteintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/clflushoptintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/clwbintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/clzerointrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cmpccxaddintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/crc32intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/emmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/enqcmdintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/f16cintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/fma4intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/fmaintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/fxsrintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/gfniintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/hresetintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ia32intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/immintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/invpcidintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/keylockerintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/lwpintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/lzcntintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/mm3dnow.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/mmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/movdirintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/mwaitxintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/nmmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/pconfigintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/pkuintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/pmmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/popcntintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/prfchiintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/prfchwintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ptwriteintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/raointintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/rdpruintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/rdseedintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/rtmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/serializeintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/sgxintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/sha512intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/shaintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/sm3intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/sm4intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/smmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/tbmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/tmmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/tsxldtrkintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/uintrintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/usermsrintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/vaesintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/vpclmulqdqintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/waitpkgintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/wbnoinvdintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__wmmintrin_aes.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/wmmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__wmmintrin_pclmul.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/x86gprintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/x86intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/xmmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/xopintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/xsavecintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/xsaveintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/xsaveoptintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/xsavesintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/xtestintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cet.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cpuid.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/wasm_simd128.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/vadefs.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/mm_malloc.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "clang-resource-headers" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include/cuda_wrappers" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cuda_wrappers/algorithm"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cuda_wrappers/cmath"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cuda_wrappers/complex"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cuda_wrappers/new"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "clang-resource-headers" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include/cuda_wrappers/bits" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cuda_wrappers/bits/shared_ptr_base.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cuda_wrappers/bits/basic_string.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cuda_wrappers/bits/basic_string.tcc"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "clang-resource-headers" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include/ppc_wrappers" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/mmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/xmmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/mm_malloc.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/emmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/pmmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/tmmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/smmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/nmmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/bmiintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/bmi2intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/immintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/x86intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/x86gprintrin.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "clang-resource-headers" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include/llvm_libc_wrappers" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/llvm_libc_wrappers/assert.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/llvm_libc_wrappers/stdio.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/llvm_libc_wrappers/stdlib.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/llvm_libc_wrappers/string.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/llvm_libc_wrappers/ctype.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/llvm_libc_wrappers/inttypes.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/llvm_libc_wrappers/time.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "clang-resource-headers" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include/openmp_wrappers" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/math.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/cmath"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/complex.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/complex"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/__clang_openmp_device_functions.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/complex_cmath.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/new"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "core-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/builtins.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/float.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/inttypes.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/iso646.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/limits.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/module.modulemap"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/stdalign.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/stdarg.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stdarg___gnuc_va_list.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stdarg___va_copy.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stdarg_va_arg.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stdarg_va_copy.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stdarg_va_list.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/stdatomic.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/stdbool.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/stdckdint.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/stddef.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stddef_max_align_t.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stddef_null.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stddef_nullptr_t.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stddef_offsetof.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stddef_ptrdiff_t.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stddef_rsize_t.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stddef_size_t.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stddef_unreachable.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stddef_wchar_t.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__stddef_wint_t.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/stdint.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/stdnoreturn.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/tgmath.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/unwind.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/varargs.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "arm-common-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/arm_acle.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "arm-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/arm_cmse.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/armintr.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "aarch64-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/arm64intr.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/arm_neon_sve_bridge.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cuda-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include/cuda_wrappers" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cuda_wrappers/algorithm"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cuda_wrappers/cmath"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cuda_wrappers/complex"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cuda_wrappers/new"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cuda-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include/cuda_wrappers/bits" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cuda_wrappers/bits/shared_ptr_base.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cuda_wrappers/bits/basic_string.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cuda_wrappers/bits/basic_string.tcc"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "cuda-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_cuda_builtin_vars.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_cuda_math.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_cuda_cmath.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_cuda_complex_builtins.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_cuda_device_functions.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_cuda_intrinsics.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_cuda_texture_intrinsics.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_cuda_libdevice_declares.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_cuda_math_forward_declares.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_cuda_runtime_wrapper.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "hexagon-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/hexagon_circ_brev_intrinsics.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/hexagon_protos.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/hexagon_types.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/hvx_hexagon_protos.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "hip-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_hip_libdevice_declares.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_hip_cmath.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_hip_math.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_hip_stdlib.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__clang_hip_runtime_wrapper.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "loongarch-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/larchintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/lasxintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/lsxintrin.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "mips-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/msa.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "ppc-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include/ppc_wrappers" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/mmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/xmmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/mm_malloc.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/emmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/pmmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/tmmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/smmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/nmmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/bmiintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/bmi2intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/immintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/x86intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ppc_wrappers/x86gprintrin.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "ppc-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/altivec.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "ppc-htm-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/htmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/htmxlintrin.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "riscv-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/riscv_bitmanip.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/riscv_crypto.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/riscv_ntlh.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/sifive_vector.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "systemz-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/s390intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/vecintrin.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "ve-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/velintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/velintrin_gen.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/velintrin_approx.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "webassembly-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/wasm_simd128.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "x86-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/adcintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/adxintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ammintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/amxcomplexintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/amxfp16intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/amxintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx2intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512bf16intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512bitalgintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512bwintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512cdintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512dqintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512erintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512fintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512fp16intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512ifmaintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512ifmavlintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512pfintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vbmi2intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vbmiintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vbmivlintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vlbf16intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vlbitalgintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vlbwintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vlcdintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vldqintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vlfp16intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vlintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vlvbmi2intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vlvnniintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vlvp2intersectintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vnniintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vp2intersectintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vpopcntdqintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avx512vpopcntdqvlintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avxifmaintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avxintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avxneconvertintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avxvnniint16intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avxvnniint8intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/avxvnniintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/bmi2intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/bmiintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cetintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cldemoteintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/clflushoptintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/clwbintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/clzerointrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cmpccxaddintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/crc32intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/emmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/enqcmdintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/f16cintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/fma4intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/fmaintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/fxsrintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/gfniintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/hresetintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ia32intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/immintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/invpcidintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/keylockerintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/lwpintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/lzcntintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/mm3dnow.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/mmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/movdirintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/mwaitxintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/nmmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/pconfigintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/pkuintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/pmmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/popcntintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/prfchiintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/prfchwintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/ptwriteintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/raointintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/rdpruintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/rdseedintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/rtmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/serializeintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/sgxintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/sha512intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/shaintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/sm3intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/sm4intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/smmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/tbmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/tmmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/tsxldtrkintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/uintrintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/usermsrintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/vaesintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/vpclmulqdqintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/waitpkgintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/wbnoinvdintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__wmmintrin_aes.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/wmmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/__wmmintrin_pclmul.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/x86gprintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/x86intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/xmmintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/xopintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/xsavecintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/xsaveintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/xsaveoptintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/xsavesintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/xtestintrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cet.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/cpuid.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "hlsl-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/hlsl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "hlsl-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include/hlsl" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/hlsl/hlsl_basic_types.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/hlsl/hlsl_intrinsics.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "opencl-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/opencl-c.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/opencl-c-base.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "openmp-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include/openmp_wrappers" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/math.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/cmath"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/complex.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/complex"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/__clang_openmp_device_functions.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/complex_cmath.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/new"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "openmp-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include/openmp_wrappers" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/math.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/cmath"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/complex.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/complex"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/__clang_openmp_device_functions.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/complex_cmath.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/openmp_wrappers/new"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "utility-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/mm_malloc.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "windows-resource-headers")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/clang/18/include" TYPE FILE FILES
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/intrin.h"
    "/home/runner/work/root/root/interpreter/llvm-project/clang/lib/Headers/vadefs.h"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/runner/work/root/root/build/interpreter/llvm-project/llvm/tools/clang/lib/Headers/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
