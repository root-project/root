/*===---- wmmintrin.h - AES intrinsics ------------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifdef __LLVM_LCCRT_INTRIN_wmmintrin_h__

#include __LLVM_LCCRT_INTRIN_wmmintrin_h__

#else /* !__LLVM_LCCRT_INTRIN_wmmintrin_h__ */

#ifndef __WMMINTRIN_H
#define __WMMINTRIN_H

#if !defined(__i386__) && !defined(__x86_64__)
#error "This header is only meant to be used on x86 and x64 architecture"
#endif

#include <emmintrin.h>

#include <__wmmintrin_aes.h>

#include <__wmmintrin_pclmul.h>

#endif /* __WMMINTRIN_H */

#endif /* __LLVM_LCCRT_INTRIN_adxintrin_h__ */
