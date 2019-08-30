// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 20/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////
// Definition of the CpuMatrix class used to represent  //
// weight and bias matrices in neural nets.             //
//////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_CPU_CPUMATRIX
#define TMVA_DNN_ARCHITECTURES_CPU_CPUMATRIX

#ifdef R__USE_IMT
#define DL_USE_MTE // use MT with tbb
#endif

#include <cstddef>
#include <vector>

#include "TMatrix.h"
#include "TMVA/Config.h"
#include "CpuBuffer.h"
#include <TMVA/Config.h>
#include <TMVA/RTensor.hxx>

// #define DEBUG_TMVA_TCPUMATRIX
#if defined(DEBUG_TMVA_TCPUMATRIX)
/*
 * Debug(!) function for printing matrices.
 *
 * Prints the input expression `mat` using preprocessor directives (with
 * `#mat`). E.g. `PrintMatrix(matA, "Test")` _could_ generate
 * "matA is null pointer".
 *
 * Note: This is a preprocessor macro. It does _not_ respect namespaces.
 *
 * @param mat  Matrix to print
 * @param text Name of matrix
 */
#define TMVA_DNN_PrintTCpuMatrix(mat, text)                                                                \
   {                                                                                                       \
      auto _dpointer = mat.GetRawDataPointer();                                                            \
      if (_dpointer == NULL) {                                                                             \
         std::cout << #mat << " is null pointer" << std::endl;                                             \
         exit(1);                                                                                          \
      }                                                                                                    \
      auto _nrows = mat.GetNrows();                                                                        \
      auto _ncols = mat.GetNcols();                                                                        \
      std::cout << "---------------------" << text << " " << #mat << "(" << _nrows << "," << _ncols << ")" \
                << "--------------------" << std::endl;                                                    \
      for (size_t _i = 0; _i < _nrows; _i++) {                                                             \
         for (size_t _j = 0; _j < _ncols; _j++) {                                                          \
            std::cout << mat(_i, _j);                                                                      \
            if (_j < _ncols - 1)                                                                           \
               std::cout << ",";                                                                           \
         }                                                                                                 \
         std::cout << std::endl;                                                                           \
      }                                                                                                    \
   }
#else
#define TMVA_DNN_PrintTCpuMatrix(mat, text)
#endif

namespace TMVA {
namespace DNN {

/** The TCpuMatrix class.
 *
 * Matrix class for multi-threaded CPU architectures. Uses the TCpuBuffer
 * class to store the matrices in column-major format for compatibility with
 * BLAS. Provides Map and MapFrom member functions to simplify the application of
 * activation functions and derivatives to matrices.
 *
 * Copying and assignment of TCpuMatrix objects only performs shallow copies, i.e.
 * copying is fast and the resulting objects share the element data.
 *
 * \tparam AFloat The floating point type used to represent the matrix elements.
 */
//______________________________________________________________________________
template <typename AFloat>
class TCpuMatrix {
private:
   static std::vector<AFloat> fOnes; ///< Vector filled with ones used for BLAS calls.

public:
   TCpuBuffer<AFloat> fBuffer; ///< The buffer holding the matrix elements
                               ///< in column-major format.
private:
   size_t fNCols;
   size_t fNRows;

public:
   // friend class TCpuTensor<AFloat>;

   /** Returns pointer to a vector holding only ones with a guaranteed length
    *  of the number of columns of every instantiated CpuMatrix object. */
   static const AFloat *GetOnePointer() { return fOnes.data(); }

   static size_t GetOnePointerSize() { return fOnes.size(); }

   static void InitializeOneVector(size_t n);

   /** Construct matrix and allocate space for its elements. */
   TCpuMatrix(size_t nRows, size_t nCols);
   /** Construct a TCpuMatrix object by (deeply) copying from a
    *  TMatrixT<Double_t> matrix. */
   TCpuMatrix(const TMatrixT<AFloat> &);
   /** Construct a m-times-n matrix from the given buffer. The size must of
    *  course match. */
   TCpuMatrix(const TCpuBuffer<AFloat> &buffer, size_t m, size_t n);

   // N.B the default copy constructor does a shallow copy (NOT a deep one) !
   TCpuMatrix(const TCpuMatrix &) = default;
   TCpuMatrix(TCpuMatrix &&) = default;
   TCpuMatrix &operator=(const TCpuMatrix &) = default;
   TCpuMatrix &operator=(TCpuMatrix &&) = default;
   ~TCpuMatrix() = default;

   /** Clear content of the matrix and initialize to zero elements
    */
   void Zero();

   /** Convert to a TMatrixT<AFloat_t> object. Performs a deep copy of the matrix
    *  elements. */
   operator TMatrixT<AFloat>() const;

   /** Map the given function over the matrix elements. Executed in parallel
    *  using TThreadExecutor. */
   template <typename Function_t>
   void Map(Function_t &f);

   /** Same as maps but takes the input values from the matrix \p A and writes
    *  the results in this matrix. */
   template <typename Function_t>
   void MapFrom(Function_t &f, const TCpuMatrix &A);

   size_t GetNrows() const { return fNRows; }
   size_t GetNcols() const { return fNCols; }
   size_t GetNoElements() const { return fNRows * fNCols; }
   size_t GetSize() const { return fNRows * fNCols; }

   /** Return matrix element in row \p i and column \p j. */
   AFloat operator()(size_t i, size_t j) const { return fBuffer[j * fNRows + i]; }
   AFloat &operator()(size_t i, size_t j) { return fBuffer[j * fNRows + i]; }

   /** Return raw pointer to the elements stored contiguously in column-major
    *  order. */
   AFloat *GetRawDataPointer() { return fBuffer; }
   const AFloat *GetRawDataPointer() const { return fBuffer; }

   static Executor &GetThreadExecutor() { return TMVA::Config::Instance().GetThreadExecutor(); }

   // static function to get the number of elements for task
   static size_t GetNWorkItems(size_t nelements);

   // print matrix
   void Print() const
   {
      TCpuMatrix cpuMatrix = *this;
      TMVA_DNN_PrintTCpuMatrix(cpuMatrix, "CpuMatrix");
   }

private:
   void Initialize();
};

template <typename AFloat>
std::vector<AFloat> TCpuMatrix<AFloat>::fOnes{};

// Inline Functions.
//______________________________________________________________________________
template <typename AFloat>
size_t TCpuMatrix<AFloat>::GetNWorkItems(size_t nElements)
{
   // nElements should have at least 100
   // const size_t nWorkers = TMVA::Config::Instance().GetNCpu();
   // return  (nElements > nWorkers) ?  (int) nElements/nWorkers : 1;
   const size_t minElements = 1000;
   const size_t nCpu = TMVA::Config::Instance().GetNCpu();
   if (nElements <= minElements)
      return nElements;
   if (nElements < nCpu * minElements) {
      size_t nt = nElements / minElements;
      return nElements / nt;
   }
   return nElements / nCpu;
   // if (nElements < nCpu*20) return nElements/nCpu;
   // return nElements/(nCpu*10);
}

//______________________________________________________________________________
template <typename AFloat>
template <typename Function_t>
inline void TCpuMatrix<AFloat>::Map(Function_t &f)
{
   AFloat *data = GetRawDataPointer();
   size_t nelements = GetNoElements();
   size_t nsteps = TCpuMatrix<AFloat>::GetNWorkItems(nelements);

   auto ff = [data, &nsteps, &nelements, &f](UInt_t workerID) {
      size_t jMax = std::min(workerID + nsteps, nelements);
      for (size_t j = workerID; j < jMax; ++j) {
         data[j] = f(data[j]);
      }
      return 0;
   };

   if (nsteps < nelements) {
      TMVA::Config::Instance().GetThreadExecutor().Foreach(ff, ROOT::TSeqI(0, nelements, nsteps));

      // for (size_t i = 0;  i < nelements; i+=nsteps)
      //    ff(i);

   } else {
      R__ASSERT(nelements == nsteps);
      ff(0);
   }
}

//______________________________________________________________________________
template <typename AFloat>
template <typename Function_t>
inline void TCpuMatrix<AFloat>::MapFrom(Function_t &f, const TCpuMatrix &A)
{
   AFloat *dataB = GetRawDataPointer();
   const AFloat *dataA = A.GetRawDataPointer();

   size_t nelements = GetNoElements();
   R__ASSERT(nelements == A.GetNoElements());
   size_t nsteps = TCpuMatrix<AFloat>::GetNWorkItems(nelements);

   auto ff = [&dataB, &dataA, &nsteps, &nelements, &f](UInt_t workerID) {
      size_t jMax = std::min(workerID + nsteps, nelements);
      for (size_t j = workerID; j < jMax; ++j) {
         dataB[j] = f(dataA[j]);
      }
      return 0;
   };
   if (nsteps < nelements) {
      TMVA::Config::Instance().GetThreadExecutor().Foreach(ff, ROOT::TSeqI(0, nelements, nsteps));
      // for (size_t i = 0;  i < nelements; i+=nsteps)
      //    ff(i);

   } else {
      R__ASSERT(nelements == nsteps);
      ff(0);
   }
}
//______________________________________________________________________________
template <typename AFloat>
void TCpuMatrix<AFloat>::Zero()
{
   for (size_t j = 0; j < fNCols; j++) {
      for (size_t i = 0; i < fNRows; i++) {
         (*this)(i, j) = 0;
      }
   }
}

// CPU Tensor Class
// It is a simple wrapper for TMVA RTensor based on
// memory owned by CPU Buffer
// We need to keep a pointer for CPUBuffer for fast conversion
// without copying to TCpuMatrix
// also provides compatibility with old interface

template <typename AFloat>
class TCpuTensor {

private:
   TCpuBuffer<AFloat> fBuffer;                  ///< The buffer holding the matrix elements
   TMVA::Experimental::RTensor<AFloat> fTensor; /// The tensor view

public:
   friend class TCpuMatrix<AFloat>;

   using Shape_t = typename TMVA::Experimental::RTensor<AFloat>::Shape_t;
   using MemoryLayout = TMVA::Experimental::MemoryLayout;
   using Matrix_t = TCpuMatrix<AFloat>; 

   // default constructor
   TCpuTensor() : fBuffer(0), fTensor(fBuffer, {0}) {}

   /** constructors from batch size, depth, height*width */
   TCpuTensor(size_t bsize, size_t depth, size_t hw, MemoryLayout memlayout = MemoryLayout::ColumnMajor)
      : fBuffer(bsize * depth * hw), fTensor(fBuffer, {depth, hw, bsize}, memlayout)
   {
      if (memlayout == MemoryLayout::RowMajor)
         fTensor = TMVA::Experimental::RTensor<AFloat>(fBuffer, {bsize, depth, hw}, memlayout);
   }

   /** constructors from batch size, depth, height, width */
   TCpuTensor(size_t bsize, size_t depth, size_t height, size_t width,
              MemoryLayout memlayout = MemoryLayout::ColumnMajor)
      : fBuffer(bsize * depth * height * width), fTensor(fBuffer, {depth, height, width, bsize}, memlayout)
   {
      if (memlayout == MemoryLayout::RowMajor)
         fTensor = TMVA::Experimental::RTensor<AFloat>(fBuffer, {bsize, depth, height, width}, memlayout);
   }

   /** constructors from a shape. Allocate the buffer in this case and initialize its elements to zero*/
   TCpuTensor(const Shape_t &shape, MemoryLayout memlayout = MemoryLayout::ColumnMajor)
      : fBuffer(TMVA::Experimental::Internal::GetSizeFromShape(shape)), fTensor(fBuffer, shape, memlayout)
   {
   }

   /** constructors from a TCpuBuffer and a rtensor */
   TCpuTensor(const TCpuBuffer<AFloat> &buffer, const TMVA::Experimental::RTensor<AFloat> &tensor)
      : fBuffer(buffer), fTensor(tensor)
   {
   }

   /** constructors from a TCpuBuffer and a shape */
   TCpuTensor(const TCpuBuffer<AFloat> &buffer, const Shape_t &shape,
              MemoryLayout memlayout = MemoryLayout::ColumnMajor)
      : fBuffer(buffer), fTensor(fBuffer, shape, memlayout)
   {
   }

   /** constructors from a TCpuMatrix. Memory layout is forced to be same as matrix (i.e. columnlayout) */
   TCpuTensor(const TCpuMatrix<AFloat> &matrix, size_t dim = 3, MemoryLayout memlayout = MemoryLayout::ColumnMajor)
      :  fBuffer(matrix.fBuffer), fTensor(fBuffer, {matrix.GetNrows(), matrix.GetNcols()}, memlayout) 
   {
      if (dim >  2) {
         Shape_t shape = fTensor.GetShape();

         if (memlayout == MemoryLayout::ColumnMajor) {
            shape.insert(shape.end(),dim-2, 1);
         } else {
            shape.insert(shape.begin(), dim - 2, 1);
         }
         // reshape does not add ones
         //fTensor.Reshape(shape);
         fTensor = TMVA::Experimental::RTensor<AFloat>( fBuffer, shape, memlayout);
      }
   }

   /** Return raw pointer to the elements stored contiguously in column-major
    *  order. */
   AFloat *GetRawDataPointer() { return fBuffer; }
   const AFloat *GetRawDataPointer() const { return fBuffer; }

   // for same API as CudaTensor (device buffer is the CpuBuffer)
   const TCpuBuffer<AFloat> & GetDeviceBuffer()     const {return fBuffer;}
   TCpuBuffer<AFloat>       & GetDeviceBuffer()           {return fBuffer;}

   AFloat *GetData() { return fTensor.GetData(); }
   const AFloat *GetData() const { return fTensor.GetData(); }

   const Shape_t & GetShape() const { return fTensor.GetShape(); }

   size_t GetNoElements() const { return fTensor.GetSize(); }
   size_t GetSize() const { return fTensor.GetSize(); }

   // return the size of the first dimension (if in row order) or last dimension if in column order
   // Tensor is  F x H x W x...for row order layout
   // or      H x W x ... x F  for column order layout
   size_t GetFirstSize() const
   {
      return (fTensor.GetMemoryLayout() == MemoryLayout::RowMajor) ? fTensor.GetShape().front()
                                                                   : fTensor.GetShape().back();
   }

   //
   size_t GetHSize() const
   {
      return (fTensor.GetMemoryLayout() == MemoryLayout::RowMajor) ? fTensor.GetShape()[1] : fTensor.GetShape()[0];
   }
   size_t GetWSize() const
   {
      return (fTensor.GetMemoryLayout() == MemoryLayout::RowMajor) ? fTensor.GetShape()[2] : fTensor.GetShape()[1];
   }

   MemoryLayout GetLayout() const { return fTensor.GetMemoryLayout(); }

   // Matrix conversion for tensors of shape 2
   TCpuMatrix<AFloat> GetMatrix() const
   {
      assert(fTensor.GetShape().size() == 2 || (fTensor.GetShape().size() == 3 && GetFirstSize() == 1));
      // t.b.d should squeeze the tensor
      return TCpuMatrix<AFloat>(fBuffer, GetHSize(), GetWSize());
   }
   // copy tensor content in a given matrix. No copy is done, just swapping the pointers
   static void TensorToMatrix(const TCpuTensor<AFloat> &tensor, TCpuMatrix<AFloat> &matrix)
   {
      assert(tensor.GetShape().size() == 2);
      assert(tensor.GetShape()[0] == matrix.GetNRows());
      assert(tensor.GetShape()[1] == matrix.GetNCols());
      matrix.fBuffer = tensor.fBuffer;
   }
   // copy matrix content in a given tensor. No copy is done, just swapping the pointers
   static void MatrixToTensor(const TCpuMatrix<AFloat> &matrix, TCpuTensor<AFloat> &tensor)
   {
      if (tensor.GetShape().size() != 2)
         tensor.fTensor.Squeeze();
      assert(tensor.GetShape().size() == 2);
      assert(tensor.GetShape()[0] == matrix.GetNRows());
      assert(tensor.GetShape()[1] == matrix.GetNCols());
      tensor.fBuffer = matrix.fBuffer;
   }

   // return slices in the first dimension (if row wise) or last dimension if colun wise
   // so single event slides
   TCpuTensor<AFloat> At(size_t i)
   {
   
      const Shape_t & shape = GetShape();
      Shape_t sliced_shape = (fTensor.GetMemoryLayout() == MemoryLayout::RowMajor)
                                ? Shape_t(shape.begin() + 1, shape.end())
                                : Shape_t(shape.begin(), shape.end() - 1);

      size_t buffsize = (fTensor.GetMemoryLayout() == MemoryLayout::RowMajor) ? fTensor.GetStrides().front()
                                                                              : fTensor.GetStrides().back();

      size_t offset = i * buffsize;

      return TCpuTensor<AFloat>(fBuffer.GetSubBuffer(offset, buffsize), sliced_shape, GetLayout());
   }
   TCpuTensor<AFloat> At(size_t i) const { return (const_cast<TCpuTensor<AFloat> &>(*this)).At(i); }

   // set all the tensor contents to zero
   void Zero()
   {
      AFloat *data = fBuffer;
      for (size_t i = 0; i < fBuffer.GetSize(); ++i)
         data[i] = 0;
   }

   // access single element - assume tensor dim is 2
   AFloat &operator()(size_t i, size_t j)
   {
      assert(fTensor.GetShape().size() == 2);
      return (fTensor.GetMemoryLayout() == MemoryLayout::RowMajor) ? fBuffer[i * fTensor.GetShape()[1] + j]
                                                                   : fBuffer[j * fTensor.GetShape()[0] + i];
   }
   // access single element - assume tensor dim is 3. First index i is always the major  indipendent of row-major or
   // column major row- major  I - J - K    . Column- major  is  J - K - I
   AFloat &operator()(size_t i, size_t j, size_t k)
   {
      assert(fTensor.GetShape().size() == 3);

      return (fTensor.GetMemoryLayout() == MemoryLayout::RowMajor)
                ? fBuffer[i * fTensor.GetShape()[1] * fTensor.GetShape()[2] + j * fTensor.GetShape()[2] + k]
                : fBuffer[i * fTensor.GetShape()[0] * fTensor.GetShape()[1] + k * fTensor.GetShape()[0] +
                          j]; // note that is J-K-I
   }

   // access single element - assume tensor dim is 2
   const AFloat & operator()(size_t i, size_t j) const
   {
      assert(fTensor.GetShape().size() == 2);
      return (fTensor.GetMemoryLayout() == MemoryLayout::RowMajor) ? GetData()[i * fTensor.GetShape()[1] + j]
                                                                   : GetData()[j * fTensor.GetShape()[0] + i];
   }
   const AFloat & operator()(size_t i, size_t j, size_t k) const
   {
      // note that i is condidered always the first dimension !!!
      assert(fTensor.GetShape().size() == 3);
      return (fTensor.GetMemoryLayout() == MemoryLayout::RowMajor)
                ? GetData()[i * fTensor.GetShape()[1] * fTensor.GetShape()[2] + j * fTensor.GetShape()[2] + k]
                : GetData()[i * fTensor.GetShape()[0] * fTensor.GetShape()[1] + k * fTensor.GetShape()[0] +
                            j]; // note that is J-K-I
   }

   /** Map the given function over the matrix elements. Executed in parallel
    *  using TThreadExecutor. */
   template <typename Function_t>
   void Map(Function_t &f);

   /** Same as maps but takes the input values from the tensor \p A and writes
    *  the results in this tensor. */
   template <typename Function_t>
   void MapFrom(Function_t &f, const TCpuTensor<AFloat> &A);

   size_t GetBufferUseCount() const { return fBuffer.GetUseCount(); }

   void Print() const {
      
      for (size_t i = 0; i < fTensor.GetSize(); i++) std::cout << GetData()[i] << "  ";
      std::cout << std::endl;
   }
};

//______________________________________________________________________________
template <typename AFloat>
template <typename Function_t>
inline void TCpuTensor<AFloat>::Map(Function_t &f)
{
   AFloat *data = GetRawDataPointer();
   size_t nelements = GetNoElements();
   size_t nsteps = TCpuMatrix<AFloat>::GetNWorkItems(nelements);

   auto ff = [data, &nsteps, &nelements, &f](UInt_t workerID) {
      size_t jMax = std::min(workerID + nsteps, nelements);
      for (size_t j = workerID; j < jMax; ++j) {
         data[j] = f(data[j]);
      }
      return 0;
   };

   if (nsteps < nelements) {
      TMVA::Config::Instance().GetThreadExecutor().Foreach(ff, ROOT::TSeqI(0, nelements, nsteps));

      // for (size_t i = 0;  i < nelements; i+=nsteps)
      //    ff(i);

   } else {
      R__ASSERT(nelements == nsteps);
      ff(0);
   }
}

//______________________________________________________________________________
template <typename AFloat>
template <typename Function_t>
inline void TCpuTensor<AFloat>::MapFrom(Function_t &f, const TCpuTensor<AFloat> &A)
{
   AFloat *dataB = GetRawDataPointer();
   const AFloat *dataA = A.GetRawDataPointer();

   size_t nelements = GetNoElements();
   R__ASSERT(nelements == A.GetNoElements());
   size_t nsteps = TCpuMatrix<AFloat>::GetNWorkItems(nelements);

   auto ff = [&dataB, &dataA, &nsteps, &nelements, &f](UInt_t workerID) {
      size_t jMax = std::min(workerID + nsteps, nelements);
      for (size_t j = workerID; j < jMax; ++j) {
         dataB[j] = f(dataA[j]);
      }
      return 0;
   };
   if (nsteps < nelements) {
      TMVA::Config::Instance().GetThreadExecutor().Foreach(ff, ROOT::TSeqI(0, nelements, nsteps));
      // for (size_t i = 0;  i < nelements; i+=nsteps)
      //    ff(i);

   } else {
      R__ASSERT(nelements == nsteps);
      ff(0);
   }
}


} // namespace DNN
} // namespace TMVA

#endif
