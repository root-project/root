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

#ifndef VC_COMMON_MEMORY_H
#define VC_COMMON_MEMORY_H

#include "memorybase.h"
#include <assert.h>
#include <algorithm>
#include <cstring>
#include <cstddef>
#include "memoryfwd.h"
#include "macros.h"

namespace ROOT {
namespace Vc
{

/**
 * Allocates memory on the Heap with alignment and padding suitable for vectorized access.
 *
 * Memory that was allocated with this function must be released with Vc::free! Other methods might
 * work but are not portable.
 *
 * \param n Specifies the number of objects the allocated memory must be able to store.
 * \tparam T The type of the allocated memory. Note, that the constructor is not called.
 * \tparam A Determines the alignment of the memory. See \ref Vc::MallocAlignment.
 *
 * \return Pointer to memory of the requested type, or 0 on error. The allocated memory is padded at
 * the end to be a multiple of the requested alignment \p A. Thus if you request memory for 21
 * int objects, aligned via Vc::AlignOnCacheline, you can safely read a full cacheline until the
 * end of the array, without generating an out-of-bounds access. For a cacheline size of 64 Bytes
 * and an int size of 4 Bytes you would thus get an array of 128 Bytes to work with.
 *
 * \warning
 * \li The standard malloc function specifies the number of Bytes to allocate whereas this
 *     function specifies the number of values, thus differing in a factor of sizeof(T).
 * \li This function is mainly meant for use with builtin types. If you use a custom
 *     type with a sizeof that is not a multiple of 2 the results might not be what you expect.
 * \li The constructor of T is not called. You can make up for this:
 * \code
 * SomeType *array = new(Vc::malloc<SomeType, Vc::AlignOnCacheline>(N)) SomeType[N];
 * \endcode
 *
 * \see Vc::free
 *
 * \ingroup Utilities
 * \headerfile memory.h <Vc/Memory>
 */
template<typename T, Vc::MallocAlignment A>
Vc_ALWAYS_INLINE_L T *Vc_ALWAYS_INLINE_R malloc(size_t n)
{
    return static_cast<T *>(Internal::Helper::malloc<A>(n * sizeof(T)));
}

/**
 * Frees memory that was allocated with Vc::malloc.
 *
 * \param p The pointer to the memory to be freed.
 *
 * \tparam T The type of the allocated memory.
 *
 * \warning The destructor of T is not called. If needed, you can call the destructor before calling
 * free:
 * \code
 * for (int i = 0; i < N; ++i) {
 *   p[i].~T();
 * }
 * Vc::free(p);
 * \endcode
 *
 * \ingroup Utilities
 * \headerfile memory.h <Vc/Memory>
 *
 * \see Vc::malloc
 */
template<typename T>
Vc_ALWAYS_INLINE void free(T *p)
{
    Internal::Helper::free(p);
}

template<typename V, size_t Size> struct _MemorySizeCalculation
{
    enum AlignmentCalculations {
        Alignment = V::Size,
        AlignmentMask = Alignment - 1,
        MaskedSize = Size & AlignmentMask,
        Padding = Alignment - MaskedSize,
        PaddedSize = MaskedSize == 0 ? Size : Size + Padding
    };
};

/**
 * \ingroup Utilities
 * \headerfile memory.h <Vc/Memory>
 *
 * A helper class for fixed-size two-dimensional arrays.
 *
 * \param V The vector type you want to operate on. (e.g. float_v or uint_v)
 * \param Size1 Number of rows
 * \param Size2 Number of columns
 */
template<typename V, size_t Size1, size_t Size2> class Memory : public VectorAlignedBaseT<V>, public MemoryBase<V, Memory<V, Size1, Size2>, 2, Memory<V, Size2> >
{
    public:
        typedef typename V::EntryType EntryType;
    private:
        typedef MemoryBase<V, Memory<V, Size1, Size2>, 2, Memory<V, Size2> > Base;
            friend class MemoryBase<V, Memory<V, Size1, Size2>, 2, Memory<V, Size2> >;
            friend class MemoryDimensionBase<V, Memory<V, Size1, Size2>, 2, Memory<V, Size2> >;
            enum InternalConstants {
                PaddedSize2 = _MemorySizeCalculation<V, Size2>::PaddedSize
            };
#if defined(VC_ICC) && defined(_WIN32)
            __declspec(align(__alignof(VectorAlignedBaseT<V>)))
#elif defined(VC_CLANG)
            __attribute__((aligned(__alignof(VectorAlignedBaseT<V>))))
#elif defined(VC_MSVC)
        VectorAlignedBaseT<V> _force_alignment;
            // __declspec(align(#)) accepts only numbers not __alignof nor just VectorAlignment
            // by putting VectorAlignedBaseT<V> here _force_alignment is aligned correctly.
           // the downside is that there's a lot of padding before m_mem (32 Bytes with SSE) :(
#endif
            EntryType m_mem[Size1][PaddedSize2];
        public:
            using Base::vector;
            enum Constants {
                RowCount = Size1,
                VectorsCount = PaddedSize2 / V::Size
            };

            /**
             * \return the number of rows in the array.
             *
             * \note This function can be eliminated by an optimizing compiler.
             */
            _VC_CONSTEXPR size_t rowsCount() const { return RowCount; }
            /**
             * \return the number of scalar entries in the whole array.
             *
             * \warning Do not use this function for scalar iteration over the array since there will be
             * padding between rows if \c Size2 is not divisible by \c V::Size.
             *
             * \note This function can be optimized into a compile-time constant.
             */
            _VC_CONSTEXPR size_t entriesCount() const { return Size1 * Size2; }
            /**
             * \return the number of vectors in the whole array.
             *
             * \note This function can be optimized into a compile-time constant.
             */
            _VC_CONSTEXPR size_t vectorsCount() const { return VectorsCount * Size1; }

            /**
             * Copies the data from a different object.
             *
             * \param rhs The object to copy the data from.
             *
             * \return reference to the modified Memory object.
             *
             * \note Both objects must have the exact same vectorsCount().
             */
            template<typename Parent, typename RM>
            Vc_ALWAYS_INLINE Memory &operator=(const MemoryBase<V, Parent, 2, RM> &rhs) {
                assert(vectorsCount() == rhs.vectorsCount());
                Internal::copyVectors(*this, rhs);
                return *this;
            }

            Vc_ALWAYS_INLINE Memory &operator=(const Memory &rhs) {
                Internal::copyVectors(*this, rhs);
                return *this;
            }

            /**
             * Initialize all data with the given vector.
             *
             * \param v This vector will be used to initialize the memory.
             *
             * \return reference to the modified Memory object.
             */
            inline Memory &operator=(const V &v) {
                for (size_t i = 0; i < vectorsCount(); ++i) {
                    vector(i) = v;
                }
                return *this;
            }
    }
#if defined(VC_ICC) && VC_ICC < 20120212 && !defined(_WIN32)
    __attribute__((__aligned__(__alignof(VectorAlignedBaseT<V>))))
#endif
    ;

    /**
     * A helper class to simplify usage of correctly aligned and padded memory, allowing both vector and
     * scalar access.
     *
     * Example:
     * \code
        Vc::Memory<int_v, 11> array;

        // scalar access:
        for (size_t i = 0; i < array.entriesCount(); ++i) {
            int x = array[i]; // read
            array[i] = x;     // write
        }
        // more explicit alternative:
        for (size_t i = 0; i < array.entriesCount(); ++i) {
            int x = array.scalar(i); // read
            array.scalar(i) = x;     // write
        }

        // vector access:
        for (size_t i = 0; i < array.vectorsCount(); ++i) {
            int_v x = array.vector(i); // read
            array.vector(i) = x;       // write
        }
     * \endcode
     * This code allocates a small array and implements three equivalent loops (that do nothing useful).
     * The loops show how scalar and vector read/write access is best implemented.
     *
     * Since the size of 11 is not a multiple of int_v::Size (unless you use the
     * scalar Vc implementation) the last write access of the vector loop would normally be out of
     * bounds. But the Memory class automatically pads the memory such that the whole array can be
     * accessed with correctly aligned memory addresses.
     *
     * \param V The vector type you want to operate on. (e.g. float_v or uint_v)
     * \param Size The number of entries of the scalar base type the memory should hold. This
     * is thus the same number as you would use for a normal C array (e.g. float mem[11] becomes
     * Memory<float_v, 11> mem).
     *
     * \see Memory<V, 0u>
     *
     * \ingroup Utilities
     * \headerfile memory.h <Vc/Memory>
     */
    template<typename V, size_t Size> class Memory<V, Size, 0u> : public VectorAlignedBaseT<V>, public MemoryBase<V, Memory<V, Size, 0u>, 1, void>
    {
        public:
            typedef typename V::EntryType EntryType;
        private:
            typedef MemoryBase<V, Memory<V, Size, 0u>, 1, void> Base;
            friend class MemoryBase<V, Memory<V, Size, 0u>, 1, void>;
            friend class MemoryDimensionBase<V, Memory<V, Size, 0u>, 1, void>;
            enum InternalConstants {
                Alignment = V::Size,
                AlignmentMask = Alignment - 1,
                MaskedSize = Size & AlignmentMask,
                Padding = Alignment - MaskedSize,
                PaddedSize = MaskedSize == 0 ? Size : Size + Padding
            };
#if defined(VC_ICC) && defined(_WIN32)
            __declspec(align(__alignof(VectorAlignedBaseT<V>)))
#elif defined(VC_CLANG)
            __attribute__((aligned(__alignof(VectorAlignedBaseT<V>))))
#elif defined(VC_MSVC)
            VectorAlignedBaseT<V> _force_alignment;
            // __declspec(align(#)) accepts only numbers not __alignof nor just VectorAlignment
            // by putting VectorAlignedBaseT<V> here _force_alignment is aligned correctly.
            // the downside is that there's a lot of padding before m_mem (32 Bytes with SSE) :(
#endif
            EntryType m_mem[PaddedSize];
        public:
            using Base::vector;
            enum Constants {
                EntriesCount = Size,
                VectorsCount = PaddedSize / V::Size
            };

            /**
             * Wrap existing data with the Memory convenience class.
             *
             * This function returns a \em reference to a Memory<V, Size, 0> object that you must
             * capture to avoid a copy of the whole data:
             * \code
             * Memory<float_v, 16> &m = Memory<float_v, 16>::fromRawData(someAlignedPointerToFloat)
             * \endcode
             *
             * \param ptr An aligned pointer to memory of type \p V::EntryType (e.g. \c float for
             *            Vc::float_v).
             * \return A Memory object placed at the given location in memory.
             *
             * \warning The pointer \p ptr passed to this function must be aligned according to the
             * alignment restrictions of \p V.
             * \warning The size of the accessible memory must match \p Size. This includes the
             * required padding at the end to allow the last entries to be accessed via vectors. If
             * you know what you are doing you might violate this constraint.
             * \warning It is your responsibility to ensure that the memory is released correctly
             * (not too early/not leaked). This function simply adds convenience functions to \em
             * access the memory.
             */
            static Vc_ALWAYS_INLINE Vc_CONST Memory<V, Size, 0u> &fromRawData(EntryType *ptr)
            {
                // DANGER! This placement new has to use the right address. If the compiler decides
                // RowMemory requires padding before the actual data then the address has to be adjusted
                // accordingly
                char *addr = reinterpret_cast<char *>(ptr);
                typedef Memory<V, Size, 0u> MM;
                addr -= VC_OFFSETOF(MM, m_mem);
                return *new(addr) MM;
            }

            /**
             * \return the number of scalar entries in the whole array.
             *
             * \note This function can be optimized into a compile-time constant.
             */
            _VC_CONSTEXPR size_t entriesCount() const { return EntriesCount; }

            /**
             * \return the number of vectors in the whole array.
             *
             * \note This function can be optimized into a compile-time constant.
             */
            _VC_CONSTEXPR size_t vectorsCount() const { return VectorsCount; }

#ifdef VC_CXX11
            Vc_ALWAYS_INLINE Memory() = default;
#else
            Vc_ALWAYS_INLINE Memory() {}
#endif

            inline Memory(const Memory &rhs)
            {
                Internal::copyVectors(*this, rhs);
            }

            template <size_t S> inline Memory(const Memory<V, S> &rhs)
            {
                assert(vectorsCount() == rhs.vectorsCount());
                Internal::copyVectors(*this, rhs);
            }

            inline Memory &operator=(const Memory &rhs)
            {
                Internal::copyVectors(*this, rhs);
                return *this;
            }

            template <size_t S> inline Memory &operator=(const Memory<V, S> &rhs)
            {
                assert(vectorsCount() == rhs.vectorsCount());
                Internal::copyVectors(*this, rhs);
                return *this;
            }

            Vc_ALWAYS_INLINE Memory &operator=(const EntryType *rhs) {
                std::memcpy(m_mem, rhs, entriesCount() * sizeof(EntryType));
                return *this;
            }
            inline Memory &operator=(const V &v) {
                for (size_t i = 0; i < vectorsCount(); ++i) {
                    vector(i) = v;
                }
                return *this;
            }
    }
#if defined(VC_ICC) && VC_ICC < 20120212 && !defined(_WIN32)
    __attribute__((__aligned__(__alignof(VectorAlignedBaseT<V>)) ))
#endif
    ;

    /**
     * A helper class that is very similar to Memory<V, Size> but with dynamically allocated memory and
     * thus dynamic size.
     *
     * Example:
     * \code
        size_t size = 11;
        Vc::Memory<int_v> array(size);

        // scalar access:
        for (size_t i = 0; i < array.entriesCount(); ++i) {
            array[i] = i;
        }

        // vector access:
        for (size_t i = 0; i < array.vectorsCount(); ++i) {
            array.vector(i) = int_v::IndexesFromZero() + i * int_v::Size;
        }
     * \endcode
     * This code allocates a small array with 11 scalar entries
     * and implements two equivalent loops that initialize the memory.
     * The scalar loop writes each individual int. The vectorized loop writes int_v::Size values to
     * memory per iteration. Since the size of 11 is not a multiple of int_v::Size (unless you use the
     * scalar Vc implementation) the last write access of the vector loop would normally be out of
     * bounds. But the Memory class automatically pads the memory such that the whole array can be
     * accessed with correctly aligned memory addresses.
     * (Note: the scalar loop can be auto-vectorized, except for the last three assignments.)
     *
     * \note The internal data pointer is not declared with the \c __restrict__ keyword. Therefore
     * modifying memory of V::EntryType will require the compiler to assume aliasing. If you want to use
     * the \c __restrict__ keyword you need to use a standard pointer to memory and do the vector
     * address calculation and loads and stores manually.
     *
     * \param V The vector type you want to operate on. (e.g. float_v or uint_v)
     *
     * \see Memory<V, Size>
     *
     * \ingroup Utilities
     * \headerfile memory.h <Vc/Memory>
     */
    template<typename V> class Memory<V, 0u, 0u> : public MemoryBase<V, Memory<V, 0u, 0u>, 1, void>
    {
        public:
            typedef typename V::EntryType EntryType;
        private:
            typedef MemoryBase<V, Memory<V>, 1, void> Base;
            friend class MemoryBase<V, Memory<V>, 1, void>;
            friend class MemoryDimensionBase<V, Memory<V>, 1, void>;
        enum InternalConstants {
            Alignment = V::Size,
            AlignmentMask = Alignment - 1
        };
        size_t m_entriesCount;
        size_t m_vectorsCount;
        EntryType *m_mem;
        size_t calcPaddedEntriesCount(size_t x)
        {
            size_t masked = x & AlignmentMask;
            return (masked == 0 ? x : x + (Alignment - masked));
        }
    public:
        using Base::vector;

        /**
         * Allocate enough memory to access \p size values of type \p V::EntryType.
         *
         * The allocated memory is aligned and padded correctly for fully vectorized access.
         *
         * \param size Determines how many scalar values will fit into the allocated memory.
         */
        Vc_ALWAYS_INLINE Memory(size_t size)
            : m_entriesCount(size),
            m_vectorsCount(calcPaddedEntriesCount(m_entriesCount)),
            m_mem(Vc::malloc<EntryType, Vc::AlignOnVector>(m_vectorsCount))
        {
            m_vectorsCount /= V::Size;
        }

        /**
         * Copy the memory into a new memory area.
         *
         * The allocated memory is aligned and padded correctly for fully vectorized access.
         *
         * \param rhs The Memory object to copy from.
         */
        template<typename Parent, typename RM>
        Vc_ALWAYS_INLINE Memory(const MemoryBase<V, Parent, 1, RM> &rhs)
            : m_entriesCount(rhs.entriesCount()),
            m_vectorsCount(rhs.vectorsCount()),
            m_mem(Vc::malloc<EntryType, Vc::AlignOnVector>(m_vectorsCount * V::Size))
        {
            Internal::copyVectors(*this, rhs);
        }

        /**
         * Overload of the above function.
         *
         * (Because C++ would otherwise not use the templated cctor and use a default-constructed cctor instead.)
         *
         * \param rhs The Memory object to copy from.
         */
        Vc_ALWAYS_INLINE Memory(const Memory &rhs)
            : m_entriesCount(rhs.entriesCount()),
            m_vectorsCount(rhs.vectorsCount()),
            m_mem(Vc::malloc<EntryType, Vc::AlignOnVector>(m_vectorsCount * V::Size))
        {
            Internal::copyVectors(*this, rhs);
        }

        /**
         * Frees the memory which was allocated in the constructor.
         */
        Vc_ALWAYS_INLINE ~Memory()
        {
            Vc::free(m_mem);
        }

        /**
         * Swap the contents and size information of two Memory objects.
         *
         * \param rhs The other Memory object to swap.
         */
        inline void swap(Memory &rhs) {
            std::swap(m_mem, rhs.m_mem);
            std::swap(m_entriesCount, rhs.m_entriesCount);
            std::swap(m_vectorsCount, rhs.m_vectorsCount);
        }

        /**
         * \return the number of scalar entries in the whole array.
         */
        Vc_ALWAYS_INLINE Vc_PURE size_t entriesCount() const { return m_entriesCount; }

        /**
         * \return the number of vectors in the whole array.
         */
        Vc_ALWAYS_INLINE Vc_PURE size_t vectorsCount() const { return m_vectorsCount; }

        /**
         * Overwrite all entries with the values stored in \p rhs.
         *
         * \param rhs The object to copy the data from.
         *
         * \return reference to the modified Memory object.
         *
         * \note this function requires the vectorsCount() of both Memory objects to be equal.
         */
        template<typename Parent, typename RM>
        Vc_ALWAYS_INLINE Memory &operator=(const MemoryBase<V, Parent, 1, RM> &rhs) {
            assert(vectorsCount() == rhs.vectorsCount());
            Internal::copyVectors(*this, rhs);
            return *this;
        }

        Vc_ALWAYS_INLINE Memory &operator=(const Memory &rhs) {
            assert(vectorsCount() == rhs.vectorsCount());
            Internal::copyVectors(*this, rhs);
            return *this;
        }

        /**
         * Overwrite all entries with the values stored in the memory at \p rhs.
         *
         * \param rhs The array to copy the data from.
         *
         * \return reference to the modified Memory object.
         *
         * \note this function requires that there are entriesCount() many values accessible from \p rhs.
         */
        Vc_ALWAYS_INLINE Memory &operator=(const EntryType *rhs) {
            std::memcpy(m_mem, rhs, entriesCount() * sizeof(EntryType));
            return *this;
        }
};

/**
 * Prefetch the cacheline containing \p addr for a single read access.
 *
 * This prefetch completely bypasses the cache, not evicting any other data.
 *
 * \param addr The cacheline containing \p addr will be prefetched.
 *
 * \ingroup Utilities
 * \headerfile memory.h <Vc/Memory>
 */
Vc_ALWAYS_INLINE void prefetchForOneRead(const void *addr)
{
    Internal::Helper::prefetchForOneRead(addr);
}

/**
 * Prefetch the cacheline containing \p addr for modification.
 *
 * This prefetch evicts data from the cache. So use it only for data you really will use. When the
 * target system supports it the cacheline will be marked as modified while prefetching, saving work
 * later on.
 *
 * \param addr The cacheline containing \p addr will be prefetched.
 *
 * \ingroup Utilities
 * \headerfile memory.h <Vc/Memory>
 */
Vc_ALWAYS_INLINE void prefetchForModify(const void *addr)
{
    Internal::Helper::prefetchForModify(addr);
}

/**
 * Prefetch the cacheline containing \p addr to L1 cache.
 *
 * This prefetch evicts data from the cache. So use it only for data you really will use.
 *
 * \param addr The cacheline containing \p addr will be prefetched.
 *
 * \ingroup Utilities
 * \headerfile memory.h <Vc/Memory>
 */
Vc_ALWAYS_INLINE void prefetchClose(const void *addr)
{
    Internal::Helper::prefetchClose(addr);
}

/**
 * Prefetch the cacheline containing \p addr to L2 cache.
 *
 * This prefetch evicts data from the cache. So use it only for data you really will use.
 *
 * \param addr The cacheline containing \p addr will be prefetched.
 *
 * \ingroup Utilities
 * \headerfile memory.h <Vc/Memory>
 */
Vc_ALWAYS_INLINE void prefetchMid(const void *addr)
{
    Internal::Helper::prefetchMid(addr);
}

/**
 * Prefetch the cacheline containing \p addr to L3 cache.
 *
 * This prefetch evicts data from the cache. So use it only for data you really will use.
 *
 * \param addr The cacheline containing \p addr will be prefetched.
 *
 * \ingroup Utilities
 * \headerfile memory.h <Vc/Memory>
 */
Vc_ALWAYS_INLINE void prefetchFar(const void *addr)
{
    Internal::Helper::prefetchFar(addr);
}

} // namespace Vc
} // namespace ROOT

namespace std
{
    template<typename V> Vc_ALWAYS_INLINE void swap(Vc::Memory<V> &a, Vc::Memory<V> &b) { a.swap(b); }
} // namespace std

#include "undomacros.h"

#endif // VC_COMMON_MEMORY_H
