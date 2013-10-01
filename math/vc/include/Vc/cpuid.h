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

#ifndef CPUID_H
#define CPUID_H

namespace ROOT {
namespace Vc
{

/**
 * \ingroup Utilities
 * \headerfile cpuid.h <Vc/cpuid.h>
 *
 * This class is available for x86 / AMD64 systems to read and interpret information about the CPU's
 * capabilities.
 *
 * Before any of the getter functions may be called, the init() function must have been called. It
 * will be called automatically, but for any function executing before main, you better call
 * \c CpuId::init() first.
 *
 * %Vc users will most likely not need this class directly, but rely on the
 * isImplementationSupported, bestImplementationSupported, extraInstructionsSupported, and
 * currentImplementationSupported functions.
 */
class CpuId
{
    typedef unsigned char uchar;
    typedef unsigned short ushort;
    typedef unsigned int uint;

    public:
        enum ProcessorType {
            OriginalOemProcessor = 0,
            IntelOverDriveProcessor = 1,
            DualProcessor = 2,
            IntelReserved = 3
        };

        /**
         * Reads the CPU capabilities and stores them for faster subsequent access.
         *
         * Will be executed automatically before main, but not necessarily before other functions
         * executing before main.
         */
        static void init();

        //! Return the cache line size in bits.
        static inline ushort cacheLineSize() { return static_cast<ushort>(s_cacheLineSize) * 8u; }
        //! Return the ProcessorType.
        static inline ProcessorType processorType() { return s_processorType; }
        //! Return the family number of the processor (vendor dependent).
        static inline uint processorFamily() { return s_processorFamily; }
        //! Return the model number of the processor (vendor dependent).
        static inline uint processorModel() { return s_processorModel; }
        //! Return the number of logical processors.
        static inline uint logicalProcessors() { return s_logicalProcessors; }
        //! Return whether the CPU vendor is AMD.
        static inline bool isAmd   () { return s_ecx0 == 0x444D4163; }
        //! Return whether the CPU vendor is Intel.
        static inline bool isIntel () { return s_ecx0 == 0x6C65746E; }
        //! Return whether the CPU supports SSE3.
        static inline bool hasSse3 () { return s_processorFeaturesC & (1 << 0); }
        //! Return whether the CPU supports the PCLMULQDQ instruction.
        static inline bool hasPclmulqdq() { return (s_processorFeaturesC & (1 << 1)) != 0; }
        //! Return whether the CPU supports the MONITOR/MWAIT instructions.
        static inline bool hasMonitor() { return (s_processorFeaturesC & (1 << 3)) != 0; }
        //! Return whether the CPU supports the Virtual Machine Extensions.
        static inline bool hasVmx  () { return (s_processorFeaturesC & (1 << 5)) != 0; }
        //! Return whether the CPU supports the Safer Mode Extensions.
        static inline bool hasSmx  () { return (s_processorFeaturesC & (1 << 6)) != 0; }
        //! Return whether the CPU supports the Enhanced Intel SpeedStep technology.
        static inline bool hasEist () { return (s_processorFeaturesC & (1 << 7)) != 0; }
        //! Return whether the CPU supports Thermal Monitor 2.
        static inline bool hasTm2  () { return (s_processorFeaturesC & (1 << 8)) != 0; }
        //! Return whether the CPU supports SSSE3.
        static inline bool hasSsse3() { return (s_processorFeaturesC & (1 << 9)) != 0; }
        //! Return whether the CPU supports FMA extensions using YMM state.
        static inline bool hasFma  () { return (s_processorFeaturesC & (1 << 12)) != 0; }
        //! Return whether the CPU supports CMPXCHG16B.
        static inline bool hasCmpXchg16b() { return (s_processorFeaturesC & (1 << 13)) != 0; }
        //! Return whether the CPU supports the Perfmon and Debug Capability.
        static inline bool hasPdcm () { return (s_processorFeaturesC & (1 << 15)) != 0; }
        //! Return whether the CPU supports Direct Cache Access: prefetch data from a memory mapped device.
        static inline bool hasDca()   { return (s_processorFeaturesC & (1 << 18)) != 0; }
        //! Return whether the CPU supports SSE 4.1
        static inline bool hasSse41() { return (s_processorFeaturesC & (1 << 19)) != 0; }
        //! Return whether the CPU supports SSE 4.2
        static inline bool hasSse42() { return (s_processorFeaturesC & (1 << 20)) != 0; }
        //! Return whether the CPU supports the MOVBE instruction.
        static inline bool hasMovbe() { return (s_processorFeaturesC & (1 << 22)) != 0; }
        //! Return whether the CPU supports the POPCNT instruction.
        static inline bool hasPopcnt(){ return (s_processorFeaturesC & (1 << 23)) != 0; }
        //static inline bool hasTscDeadline() { return (s_processorFeaturesC & (1 << 24)) != 0; }
        //! Return whether the CPU supports the AESNI instructions.
        static inline bool hasAes  () { return (s_processorFeaturesC & (1 << 25)) != 0; }
        //static inline bool hasXsave() { return (s_processorFeaturesC & (1 << 26)) != 0; }
        //! Return whether the CPU and OS support the XSETBV/XGETBV instructions.
        static inline bool hasOsxsave() { return (s_processorFeaturesC & (1 << 27)) != 0; }
        //! Return whether the CPU supports AVX.
        static inline bool hasAvx  () { return (s_processorFeaturesC & (1 << 28)) != 0; }
        //! Return whether the CPU supports 16-bit floating-point conversion instructions.
        static inline bool hasF16c () { return (s_processorFeaturesC & (1 << 29)) != 0; }
        //! Return whether the CPU supports the RDRAND instruction.
        static inline bool hasRdrand(){ return (s_processorFeaturesC & (1 << 30)) != 0; }
        //! Return whether the CPU contains an x87 FPU.
        static inline bool hasFpu  () { return (s_processorFeaturesD & (1 << 0)) != 0; }
        static inline bool hasVme  () { return (s_processorFeaturesD & (1 << 1)) != 0; }
        //! Return whether the CPU contains Debugging Extensions.
        static inline bool hasDe   () { return (s_processorFeaturesD & (1 << 2)) != 0; }
        //! Return whether the CPU contains Page Size Extensions.
        static inline bool hasPse  () { return (s_processorFeaturesD & (1 << 3)) != 0; }
        //! Return whether the CPU supports the RDTSC instruction.
        static inline bool hasTsc  () { return (s_processorFeaturesD & (1 << 4)) != 0; }
        //! Return whether the CPU supports the Model Specific Registers instructions.
        static inline bool hasMsr  () { return (s_processorFeaturesD & (1 << 5)) != 0; }
        //! Return whether the CPU supports the Physical Address Extension.
        static inline bool hasPae  () { return (s_processorFeaturesD & (1 << 6)) != 0; }
        //! Return whether the CPU supports the CMPXCHG8B instruction.
        static inline bool hasCx8  () { return (s_processorFeaturesD & (1 << 8)) != 0; }
        //! Return whether the CPU supports Memory Type Range Registers.
        static inline bool hasMtrr () { return (s_processorFeaturesD & (1 << 12)) != 0; }
        //! Return whether the CPU supports CMOV instructions.
        static inline bool hasCmov () { return (s_processorFeaturesD & (1 << 15)) != 0; }
        //! Return whether the CPU supports the CLFLUSH instruction.
        static inline bool hasClfsh() { return (s_processorFeaturesD & (1 << 19)) != 0; }
        //! Return whether the CPU supports ACPI.
        static inline bool hasAcpi () { return (s_processorFeaturesD & (1 << 22)) != 0; }
        //! Return whether the CPU supports MMX.
        static inline bool hasMmx  () { return (s_processorFeaturesD & (1 << 23)) != 0; }
        //! Return whether the CPU supports SSE.
        static inline bool hasSse  () { return (s_processorFeaturesD & (1 << 25)) != 0; }
        //! Return whether the CPU supports SSE2.
        static inline bool hasSse2 () { return (s_processorFeaturesD & (1 << 26)) != 0; }
        static inline bool hasHtt  () { return (s_processorFeaturesD & (1 << 28)) != 0; }
        //! Return whether the CPU supports SSE4a.
        static inline bool hasSse4a() { return (s_processorFeatures8C & (1 << 6)) != 0; }
        //! Return whether the CPU supports misaligned SSE instructions.
        static inline bool hasMisAlignSse() { return (s_processorFeatures8C & (1 << 7)) != 0; }
        //! Return whether the CPU supports the AMD prefetchw instruction.
        static inline bool hasAmdPrefetch() { return (s_processorFeatures8C & (1 << 8)) != 0; }
        //! Return whether the CPU supports the XOP instructions.
        static inline bool hasXop ()        { return (s_processorFeatures8C & (1 << 11)) != 0; }
        //! Return whether the CPU supports the FMA4 instructions.
        static inline bool hasFma4 ()       { return (s_processorFeatures8C & (1 << 16)) != 0; }
        //! Return whether the CPU supports the RDTSCP instruction.
        static inline bool hasRdtscp()      { return (s_processorFeatures8D & (1 << 27)) != 0; }
        static inline bool has3DNow()       { return (s_processorFeatures8D & (1u << 31)) != 0; }
        static inline bool has3DNowExt()    { return (s_processorFeatures8D & (1 << 30)) != 0; }
        //! Return the size of the L1 instruction cache.
        static inline uint   L1Instruction() { return s_L1Instruction; }
        //! Return the size of the L1 data cache.
        static inline uint   L1Data() { return s_L1Data; }
        //! Return the size of the L2 cache.
        static inline uint   L2Data() { return s_L2Data; }
        //! Return the size of the L3 cache.
        static inline uint   L3Data() { return s_L3Data; }
        static inline ushort L1InstructionLineSize() { return s_L1InstructionLineSize; }
        static inline ushort L1DataLineSize() { return s_L1DataLineSize; }
        static inline ushort L2DataLineSize() { return s_L2DataLineSize; }
        static inline ushort L3DataLineSize() { return s_L3DataLineSize; }
        static inline uint   L1Associativity() { return s_L1Associativity; }
        static inline uint   L2Associativity() { return s_L2Associativity; }
        static inline uint   L3Associativity() { return s_L3Associativity; }
        static inline ushort prefetch() { return s_prefetch; }

    private:
        static void interpret(uchar byte, bool *checkLeaf4);

        static uint   s_ecx0;
        static uint   s_logicalProcessors;
        static uint   s_processorFeaturesC;
        static uint   s_processorFeaturesD;
        static uint   s_processorFeatures8C;
        static uint   s_processorFeatures8D;
        static uint   s_L1Instruction;
        static uint   s_L1Data;
        static uint   s_L2Data;
        static uint   s_L3Data;
        static ushort s_L1InstructionLineSize;
        static ushort s_L1DataLineSize;
        static ushort s_L2DataLineSize;
        static ushort s_L3DataLineSize;
        static uint   s_L1Associativity;
        static uint   s_L2Associativity;
        static uint   s_L3Associativity;
        static ushort s_prefetch;
        static uchar  s_brandIndex;
        static uchar  s_cacheLineSize;
        static uchar  s_processorModel;
        static uchar  s_processorFamily;
        static ProcessorType s_processorType;
        static bool   s_noL2orL3;
};
} // namespace Vc
} // namespace ROOT

#endif // CPUID_H
