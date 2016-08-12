// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 20/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Definition of the CpuMatrix class which is a convenience class //
// for representing weight and bias matrices in neural nets.      //
////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_CPU_CPUMATRIX
#define TMVA_DNN_ARCHITECTURES_CPU_CPUMATRIX

#include <cstddef>
#include <vector>
#include "tbb/tbb.h"

#include "TMatrix.h"

#include <iostream>
#include "CpuBuffer.h"

namespace TMVA
{
namespace DNN
{

// TCpuMatrix
//______________________________________________________________________________
/** The TCpuMatrix class.
 *
 * Matrix class for multi-threaded CPU architectures. Holds matrix elements
 * in 1D std::vector stored in column-major format for compatibility with
 * BLAS. Also provides Map and MapFrom routines for the efficient mapping of
 * activation functions onto matrices.
 */
template<typename AReal>
class TCpuMatrix
{
private:

   static std::vector<AReal> fOnes;  ///< Vector filled with ones used for BLAS calls.

   TCpuBuffer<AReal> fBuffer; ///< The buffer holding the matrix elements
                              ///< in column-major format.
   size_t            fNCols;
   size_t            fNRows;

public:

   /** Returns pointer to a vector holding only ones with a guaranteed length
    *  of the number of columns of every instantiated CpuMatrix object. */
   static const AReal * GetOnePointer() {return fOnes.data();}

   /** Construct matrix and allocate space for its elements. */
   TCpuMatrix(size_t nRows, size_t nCols);
   TCpuMatrix(const TMatrixT<AReal> &);
   TCpuMatrix(const TCpuBuffer<AReal> &buffer, size_t m, size_t n);

   TCpuMatrix(const TCpuMatrix  &)             = default;
   TCpuMatrix(      TCpuMatrix &&)             = default;
   TCpuMatrix & operator=(const TCpuMatrix &)  = default;
   TCpuMatrix & operator=(TCpuMatrix &&)       = default;
   ~TCpuMatrix()                               = default;

   operator TMatrixT<AReal>() const;

   /** Map the given function over the matrix elements. Executed in parallel
    *  using tbb. */
   template <typename Function_t>
   void Map(Function_t &f);

   /** Same as maps but takes the input values from the matrix \p A and writes
    *  the results in this matrix. */
   template <typename Function_t>
   void MapFrom(Function_t &f, const TCpuMatrix & A);

   size_t GetNrows() const {return fNRows;}
   size_t GetNcols() const {return fNCols;}
   size_t GetNElements() const {return fNRows * fNCols;}

   /** Return matrix element in row \p i and column \p j. */
   AReal   operator()(size_t i, size_t j) const {return fBuffer[j * fNRows + i];}
   AReal & operator()(size_t i, size_t j)       {return fBuffer[j * fNRows + i];}

   /** Return raw pointer to the elements stored contiguously in column-major
    *  order. */
   AReal *       GetRawDataPointer()        {return fBuffer;}
   const AReal * GetRawDataPointer()  const {return fBuffer;}

private:

   void Initialize();

};

// Inline Functions.
//______________________________________________________________________________
template<typename AReal>
template<typename Function_t>
inline void TCpuMatrix<AReal>::Map(Function_t &f)
{
   AReal __restrict__ *data = GetRawDataPointer();

   auto fRange = [data, &f](const tbb::blocked_range<size_t> & range)
   {
      size_t rangeBegin = range.begin();
      size_t rangeEnd   = range.end();

      for (size_t i = rangeBegin; i != rangeEnd; ++i) {
         data[i] = f(data[i]);
      }
   };

   tbb::blocked_range<size_t> range(0, GetNElements());
   parallel_for(range, fRange);
}

template<typename AReal>
template<typename Function_t>
inline void TCpuMatrix<AReal>::MapFrom(Function_t &f, const TCpuMatrix &A)
{
         AReal __restrict__ *dataB = GetRawDataPointer();
   const AReal __restrict__ *dataA = A.GetRawDataPointer();

   auto fRange = [&dataB, &dataA, &f](const tbb::blocked_range<size_t> & range)
   {
      size_t rangeBegin = range.begin();
         size_t rangeEnd   = range.end();

         for (size_t i = rangeBegin; i != rangeEnd; ++i) {
            dataB[i] = f(dataA[i]);
         }
   };

   tbb::blocked_range<size_t> range(0, GetNElements());
   parallel_for(range, fRange);
}

} // namespace DNN
} // namespace TMVA

#endif
