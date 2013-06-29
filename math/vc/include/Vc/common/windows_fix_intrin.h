/*  This file is part of the Vc library.

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef VC_COMMON_WINDOWS_FIX_INTRIN_H
#define VC_COMMON_WINDOWS_FIX_INTRIN_H

#if defined(VC_MSVC) && !defined(__midl)
// MSVC sucks. If you include intrin.h you get all SSE and AVX intrinsics
// declared. This is a problem because we need to implement the intrinsics
// that are not supported in hardware ourselves.
// Something always includes intrin.h even if you don't
// do it explicitly. Therefore we try to be the first to include it
// but with __midl defined, in which case it is basically empty.
#ifdef __INTRIN_H_
#error "intrin.h was already included, polluting the namespace. Please fix your code to include the Vc headers before anything that includes intrin.h. (Vc will declare the relevant intrinsics as they are required by some system headers.)"
#endif
#define __midl
#include <intrin.h>
#undef __midl
#include <crtdefs.h>
#include <setjmp.h>
#include <stddef.h>
extern "C" {

#ifdef _WIN64
_CRTIMP double ceil(_In_ double);
__int64 _InterlockedDecrement64(__int64 volatile *);
__int64 _InterlockedExchange64(__int64 volatile *, __int64);
void * _InterlockedExchangePointer(void * volatile *, void *);
__int64 _InterlockedExchangeAdd64(__int64 volatile *, __int64);
void *_InterlockedCompareExchangePointer (void * volatile *, void *, void *);
__int64 _InterlockedIncrement64(__int64 volatile *);
int __cdecl _setjmpex(jmp_buf);
void __faststorefence(void);
__int64 __mulh(__int64,__int64);
unsigned __int64 __umulh(unsigned __int64,unsigned __int64);
unsigned __int64 __readcr0(void);
unsigned __int64 __readcr2(void);
unsigned __int64 __readcr3(void);
unsigned __int64 __readcr4(void);
unsigned __int64 __readcr8(void);
void __writecr0(unsigned __int64);
void __writecr3(unsigned __int64);
void __writecr4(unsigned __int64);
void __writecr8(unsigned __int64);
unsigned __int64 __readdr(unsigned int);
void __writedr(unsigned int, unsigned __int64);
unsigned __int64 __readeflags(void);
void __writeeflags(unsigned __int64);
void __movsq(unsigned long long *, unsigned long long const *, size_t);
unsigned char __readgsbyte(unsigned long Offset);
unsigned short __readgsword(unsigned long Offset);
unsigned long __readgsdword(unsigned long Offset);
unsigned __int64 __readgsqword(unsigned long Offset);
void __writegsbyte(unsigned long Offset, unsigned char Data);
void __writegsword(unsigned long Offset, unsigned short Data);
void __writegsdword(unsigned long Offset, unsigned long Data);
void __writegsqword(unsigned long Offset, unsigned __int64 Data);
void __addgsbyte(unsigned long Offset, unsigned char Data);
void __addgsword(unsigned long Offset, unsigned short Data);
void __addgsdword(unsigned long Offset, unsigned long Data);
void __addgsqword(unsigned long Offset, unsigned __int64 Data);
void __incgsbyte(unsigned long Offset);
void __incgsword(unsigned long Offset);
void __incgsdword(unsigned long Offset);
void __incgsqword(unsigned long Offset);
unsigned char __vmx_vmclear(unsigned __int64*);
unsigned char __vmx_vmlaunch(void);
unsigned char __vmx_vmptrld(unsigned __int64*);
unsigned char __vmx_vmread(size_t, size_t*);
unsigned char __vmx_vmresume(void);
unsigned char __vmx_vmwrite(size_t, size_t);
unsigned char __vmx_on(unsigned __int64*);
void __stosq(unsigned __int64 *,  unsigned __int64, size_t);
unsigned char _interlockedbittestandset64(__int64 volatile *a, __int64 b);
unsigned char _interlockedbittestandreset64(__int64 volatile *a, __int64 b);
short _InterlockedCompareExchange16_np(short volatile *Destination, short Exchange, short Comparand);
long _InterlockedCompareExchange_np (long volatile *, long, long);
__int64 _InterlockedCompareExchange64_np(__int64 volatile *, __int64, __int64);
void *_InterlockedCompareExchangePointer_np (void * volatile *, void *, void *);
unsigned char _InterlockedCompareExchange128(__int64 volatile *, __int64, __int64, __int64 *);
unsigned char _InterlockedCompareExchange128_np(__int64 volatile *, __int64, __int64, __int64 *);
long _InterlockedAnd_np(long volatile *, long);
char _InterlockedAnd8_np(char volatile *, char);
short _InterlockedAnd16_np(short volatile *, short);
__int64 _InterlockedAnd64_np(__int64 volatile *, __int64);
long _InterlockedOr_np(long volatile *, long);
char _InterlockedOr8_np(char volatile *, char);
short _InterlockedOr16_np(short volatile *, short);
__int64 _InterlockedOr64_np(__int64 volatile *, __int64);
long _InterlockedXor_np(long volatile *, long);
char _InterlockedXor8_np(char volatile *, char);
short _InterlockedXor16_np(short volatile *, short);
__int64 _InterlockedXor64_np(__int64 volatile *, __int64);
unsigned __int64 __lzcnt64(unsigned __int64);
unsigned __int64 __popcnt64(unsigned __int64);
__int64 _InterlockedOr64(__int64 volatile *, __int64);
__int64 _InterlockedXor64(__int64 volatile *, __int64);
__int64 _InterlockedAnd64(__int64 volatile *, __int64);
unsigned char _bittest64(__int64 const *a, __int64 b);
unsigned char _bittestandset64(__int64 *a, __int64 b);
unsigned char _bittestandreset64(__int64 *a, __int64 b);
unsigned char _bittestandcomplement64(__int64 *a, __int64 b);
unsigned char _BitScanForward64(unsigned long* Index, unsigned __int64 Mask);
unsigned char _BitScanReverse64(unsigned long* Index, unsigned __int64 Mask);
unsigned __int64 __shiftleft128(unsigned __int64 LowPart, unsigned __int64 HighPart, unsigned char Shift);
unsigned __int64 __shiftright128(unsigned __int64 LowPart, unsigned __int64 HighPart, unsigned char Shift);
unsigned __int64 _umul128(unsigned __int64 multiplier, unsigned __int64 multiplicand, unsigned __int64 *highproduct);
__int64 _mul128(__int64 multiplier, __int64 multiplicand, __int64 *highproduct);
#endif

long _InterlockedOr(long volatile *, long);
char _InterlockedOr8(char volatile *, char);
short _InterlockedOr16(short volatile *, short);
long _InterlockedXor(long volatile *, long);
char _InterlockedXor8(char volatile *, char);
short _InterlockedXor16(short volatile *, short);
long _InterlockedAnd(long volatile *, long);
char _InterlockedAnd8(char volatile *, char);
short _InterlockedAnd16(short volatile *, short);
unsigned char _bittest(long const *a, long b);
unsigned char _bittestandset(long *a, long b);
unsigned char _bittestandreset(long *a, long b);
unsigned char _bittestandcomplement(long *a, long b);
unsigned char _BitScanForward(unsigned long* Index, unsigned long Mask);
unsigned char _BitScanReverse(unsigned long* Index, unsigned long Mask);
_CRTIMP wchar_t * __cdecl wcscat( _Pre_cap_for_(_Source) _Prepost_z_ wchar_t *, _In_z_ const wchar_t * _Source);
_Check_return_ _CRTIMP int __cdecl wcscmp(_In_z_ const wchar_t *,_In_z_  const wchar_t *);
_CRTIMP wchar_t * __cdecl wcscpy(_Pre_cap_for_(_Source) _Post_z_ wchar_t *, _In_z_ const wchar_t * _Source);
_Check_return_ _CRTIMP size_t __cdecl wcslen(_In_z_ const wchar_t *);
#pragma warning(suppress: 4985)
_CRTIMP wchar_t * __cdecl _wcsset(_Inout_z_ wchar_t *, wchar_t);
void _ReadBarrier(void);
unsigned char _rotr8(unsigned char value, unsigned char shift);
unsigned short _rotr16(unsigned short value, unsigned char shift);
unsigned char _rotl8(unsigned char value, unsigned char shift);
unsigned short _rotl16(unsigned short value, unsigned char shift);
short _InterlockedIncrement16(short volatile *Addend);
short _InterlockedDecrement16(short volatile *Addend);
short _InterlockedCompareExchange16(short volatile *Destination, short Exchange, short Comparand);
void __nvreg_save_fence(void);
void __nvreg_restore_fence(void);

#ifdef _M_IX86
unsigned long __readcr0(void);
unsigned long __readcr2(void);
unsigned long __readcr3(void);
unsigned long __readcr4(void);
unsigned long __readcr8(void);
void __writecr0(unsigned);
void __writecr3(unsigned);
void __writecr4(unsigned);
void __writecr8(unsigned);
unsigned __readdr(unsigned int);
void __writedr(unsigned int, unsigned);
unsigned __readeflags(void);
void __writeeflags(unsigned);
void __addfsbyte(unsigned long Offset, unsigned char Data);
void __addfsword(unsigned long Offset, unsigned short Data);
void __addfsdword(unsigned long Offset, unsigned long Data);
void __incfsbyte(unsigned long Offset);
void __incfsword(unsigned long Offset);
void __incfsdword(unsigned long Offset);
unsigned char __readfsbyte(unsigned long Offset);
unsigned short __readfsword(unsigned long Offset);
unsigned long __readfsdword(unsigned long Offset);
unsigned __int64 __readfsqword(unsigned long Offset);
void __writefsbyte(unsigned long Offset, unsigned char Data);
void __writefsword(unsigned long Offset, unsigned short Data);
void __writefsdword(unsigned long Offset, unsigned long Data);
void __writefsqword(unsigned long Offset, unsigned __int64 Data);
long _InterlockedAddLargeStatistic(__int64 volatile *, long);
#endif

_Ret_bytecap_(_Size) void * __cdecl _alloca(size_t _Size);
int __cdecl abs(_In_ int);
_Check_return_ unsigned short __cdecl _byteswap_ushort(_In_ unsigned short value);
_Check_return_ unsigned long __cdecl _byteswap_ulong(_In_ unsigned long value);
_Check_return_ unsigned __int64 __cdecl _byteswap_uint64(_In_ unsigned __int64 value);
void __cdecl __debugbreak(void);
void __cdecl _disable(void);
__int64 __emul(int,int);
unsigned __int64 __emulu(unsigned int,unsigned int);
void __cdecl _enable(void);
long __cdecl _InterlockedDecrement(long volatile *);
long _InterlockedExchange(long volatile *, long);
short _InterlockedExchange16(short volatile *, short);
char _InterlockedExchange8(char volatile *, char);
long _InterlockedExchangeAdd(long volatile *, long);
short _InterlockedExchangeAdd16(short volatile *, short);
char _InterlockedExchangeAdd8(char volatile *, char);
long _InterlockedCompareExchange (long volatile *, long, long);
__int64 _InterlockedCompareExchange64(__int64 volatile *, __int64, __int64);
long __cdecl _InterlockedIncrement(long volatile *);
int __cdecl _inp(unsigned short);
int __cdecl inp(unsigned short);
unsigned long __cdecl _inpd(unsigned short);
unsigned long __cdecl inpd(unsigned short);
unsigned short __cdecl _inpw(unsigned short);
unsigned short __cdecl inpw(unsigned short);
long __cdecl labs(_In_ long);
_Check_return_ unsigned long __cdecl _lrotl(_In_ unsigned long,_In_ int);
_Check_return_ unsigned long __cdecl _lrotr(_In_ unsigned long,_In_ int);
unsigned __int64  __ll_lshift(unsigned __int64,int);
__int64  __ll_rshift(__int64,int);
_Check_return_ int __cdecl memcmp(_In_opt_bytecount_(_Size) const void *,_In_opt_bytecount_(_Size) const void *,_In_ size_t _Size);
void * __cdecl memcpy(_Out_opt_bytecapcount_(_Size) void *,_In_opt_bytecount_(_Size) const void *,_In_ size_t _Size);
void * __cdecl memset(_Out_opt_bytecapcount_(_Size) void *,_In_ int,_In_ size_t _Size);
int __cdecl _outp(unsigned short,int);
int __cdecl outp(unsigned short,int);
unsigned long __cdecl _outpd(unsigned short,unsigned long);
unsigned long __cdecl outpd(unsigned short,unsigned long);
unsigned short __cdecl _outpw(unsigned short,unsigned short);
unsigned short __cdecl outpw(unsigned short,unsigned short);
void * _ReturnAddress(void);
_Check_return_ unsigned int __cdecl _rotl(_In_ unsigned int,_In_ int);
_Check_return_ unsigned int __cdecl _rotr(_In_ unsigned int,_In_ int);
int __cdecl _setjmp(jmp_buf);
_Check_return_ int __cdecl strcmp(_In_z_ const char *,_In_z_ const char *);
_Check_return_ size_t __cdecl strlen(_In_z_ const char *);
char * __cdecl strset(_Inout_z_ char *,_In_ int);
unsigned __int64 __ull_rshift(unsigned __int64,int);
void * _AddressOfReturnAddress(void);

void _WriteBarrier(void);
void _ReadWriteBarrier(void);
void __wbinvd(void);
void __invlpg(void*);
unsigned __int64 __readmsr(unsigned long);
void __writemsr(unsigned long, unsigned __int64);
unsigned __int64 __rdtsc(void);
void __movsb(unsigned char *, unsigned char const *, size_t);
void __movsw(unsigned short *, unsigned short const *, size_t);
void __movsd(unsigned long *, unsigned long const *, size_t);
unsigned char __inbyte(unsigned short Port);
unsigned short __inword(unsigned short Port);
unsigned long __indword(unsigned short Port);
void __outbyte(unsigned short Port, unsigned char Data);
void __outword(unsigned short Port, unsigned short Data);
void __outdword(unsigned short Port, unsigned long Data);
void __inbytestring(unsigned short Port, unsigned char *Buffer, unsigned long Count);
void __inwordstring(unsigned short Port, unsigned short *Buffer, unsigned long Count);
void __indwordstring(unsigned short Port, unsigned long *Buffer, unsigned long Count);
void __outbytestring(unsigned short Port, unsigned char *Buffer, unsigned long Count);
void __outwordstring(unsigned short Port, unsigned short *Buffer, unsigned long Count);
void __outdwordstring(unsigned short Port, unsigned long *Buffer, unsigned long Count);
unsigned int __getcallerseflags();
void __vmx_vmptrst(unsigned __int64 *);
void __vmx_off(void);
void __svm_clgi(void);
void __svm_invlpga(void*, int);
void __svm_skinit(int);
void __svm_stgi(void);
void __svm_vmload(size_t);
void __svm_vmrun(size_t);
void __svm_vmsave(size_t);
void __halt(void);
void __sidt(void*);
void __lidt(void*);
void __ud2(void);
void __nop(void);
void __stosb(unsigned char *, unsigned char, size_t);
void __stosw(unsigned short *,  unsigned short, size_t);
void __stosd(unsigned long *,  unsigned long, size_t);
unsigned char _interlockedbittestandset(long volatile *a, long b);
unsigned char _interlockedbittestandreset(long volatile *a, long b);
void __cpuid(int a[4], int b);
void __cpuidex(int a[4], int b, int c);
unsigned __int64 __readpmc(unsigned long a);
unsigned long __segmentlimit(unsigned long a);
_Check_return_ unsigned __int64 __cdecl _rotl64(_In_ unsigned __int64,_In_ int);
_Check_return_ unsigned __int64 __cdecl _rotr64(_In_ unsigned __int64,_In_ int);
__int64 __cdecl _abs64(__int64);
void __int2c(void);
char _InterlockedCompareExchange8(char volatile *Destination, char Exchange, char Comparand);
unsigned short __lzcnt16(unsigned short);
unsigned int __lzcnt(unsigned int);
unsigned short __popcnt16(unsigned short);
unsigned int __popcnt(unsigned int);
unsigned __int64 __rdtscp(unsigned int*);
}
#endif

#endif // VC_COMMON_WINDOWS_FIX_INTRIN_H
