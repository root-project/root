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


   TCpuBuffer<AFloat>& GetBuffer() {return fBuffer;}
   const TCpuBuffer<AFloat>& GetBuffer() const {return fBuffer;}


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
class TCpuTensor : public TMVA::Experimental::RTensor<AFloat, TCpuBuffer<AFloat>> {

private:
   //TCpuTensor will have no extra private members than RTensor
public:
   friend class TCpuMatrix<AFloat>;

   using Shape_t = typename TMVA::Experimental::RTensor<AFloat>::Shape_t;
   using MemoryLayout = TMVA::Experimental::MemoryLayout;
   using Matrix_t = TCpuMatrix<AFloat>;
   using Scalar_t = AFloat;

   // default constructor
   TCpuTensor(): TMVA::Experimental::RTensor<AFloat, TCpuBuffer<AFloat>>(std::make_shared<TCpuBuffer<AFloat>>(0), {0})
   {}

   /** constructors from n m */
   TCpuTensor(size_t n, size_t m, MemoryLayout memlayout = MemoryLayout::ColumnMajor)
      : TMVA::Experimental::RTensor<AFloat, TCpuBuffer<AFloat>>(std::make_shared<TCpuBuffer<AFloat>>(n * m), {n, m}, memlayout)
   {}

   /** constructors from batch size, depth, height*width */
   TCpuTensor(size_t bsize, size_t depth, size_t hw, MemoryLayout memlayout = MemoryLayout::ColumnMajor)
      : TMVA::Experimental::RTensor<AFloat, TCpuBuffer<AFloat>>(std::make_shared<TCpuBuffer<AFloat>>(bsize * depth * hw), {depth, hw, bsize}, memlayout)
   {
      if (memlayout == MemoryLayout::RowMajor)
         this->ReshapeInplace({bsize, depth, hw});
   }

   /** constructors from batch size, depth, height, width */
   TCpuTensor(size_t bsize, size_t depth, size_t height, size_t width,
              MemoryLayout memlayout = MemoryLayout::ColumnMajor)
      : TMVA::Experimental::RTensor<AFloat, TCpuBuffer<AFloat>>(std::make_shared<TCpuBuffer<AFloat>>(bsize * depth * height * width),
      {depth, height, width, bsize}, memlayout)
   {
      if (memlayout == MemoryLayout::RowMajor)
         this->ReshapeInplace({bsize, depth, height, width});
   }

   /** constructors from a shape.*/
   TCpuTensor(Shape_t shape, MemoryLayout memlayout = MemoryLayout::ColumnMajor)
      : TMVA::Experimental::RTensor<AFloat, TCpuBuffer<AFloat>>(std::make_shared<TCpuBuffer<AFloat>>(TMVA::Experimental::Internal::GetSizeFromShape(shape)),
      shape, memlayout)
   {}

    /* constructors from a AFloat pointer  and a shape. This is a copy */

   TCpuTensor(AFloat *data, const Shape_t &shape,
              MemoryLayout memlayout = MemoryLayout::ColumnMajor)
      : TMVA::Experimental::RTensor<AFloat, TCpuBuffer<AFloat>>(std::make_shared<TCpuBuffer<AFloat>>(TMVA::Experimental::Internal::GetSizeFromShape(shape)), shape, memlayout)
   {
      auto& fContainer = *(this->GetContainer());
      for (size_t i = 0; i <  this->GetSize(); ++i) fContainer[i] = data[i];
   }
   


   /** constructors from a TCpuBuffer and a shape */
   //unsafe method for backwards compatibility, const not promised. A view.
   TCpuTensor(const TCpuBuffer<AFloat>& buffer, Shape_t shape, MemoryLayout memlayout = MemoryLayout::ColumnMajor)
      : TMVA::Experimental::RTensor<AFloat, TCpuBuffer<AFloat>>(std::make_shared<TCpuBuffer<AFloat>>(buffer), shape, memlayout) {
         R__ASSERT(this->GetSize() <= this->GetContainer()->GetSize());
      }



   /** constructors from a TCpuMatrix. Memory layout is forced to be same as matrix (i.e. columnlayout) */
   //unsafe method for backwards compatibility, const not promised. A view of underlying data.
   TCpuTensor(const TCpuMatrix<AFloat> &matrix, size_t dim = 3, MemoryLayout memlayout = MemoryLayout::ColumnMajor)
      : TMVA::Experimental::RTensor<AFloat, TCpuBuffer<AFloat>>(std::make_shared<TCpuBuffer<AFloat>>(matrix.GetBuffer()),{matrix.GetNrows(), matrix.GetNcols()}, memlayout)
   {

      if (dim >  2) {
         Shape_t shape = this->GetShape();

         if (this->GetLayout() == MemoryLayout::ColumnMajor) {
            shape.insert(shape.end(),dim-2, 1);
         } else {
            shape.insert(shape.begin(), dim - 2, 1);
         }
         this->ReshapeInplace(shape);
      }
   }


   /** Convert to a TMatrixT<AFloat_t> object. Performs a deep copy of the matrix
    *  elements. */

   operator TMatrixT<AFloat>() const {
      // this should work only for size 2 or 4 tensors
      if (this->GetShape().size() == 2 || (this->GetShape().size() == 3 && GetFirstSize() == 1)) {
         TCpuMatrix<AFloat> temp = GetMatrix();
         return temp;
      }
      // convert as a flat vector
      return TMatrixT<AFloat>(1, this->GetSize(), this->GetData());
   }


   /** Return raw pointer to the elements stored contiguously in column-major
    *  order. */
   AFloat *GetRawDataPointer() { return *(this->GetContainer()); }
   const AFloat *GetRawDataPointer() const { return *(this->GetContainer()); }

   // for same API as CudaTensor (device buffer is the CpuBuffer)
   const TCpuBuffer<AFloat> & GetDeviceBuffer()     const {return *(this->GetContainer());}
   TCpuBuffer<AFloat>       & GetDeviceBuffer()           {return *(this->GetContainer());}


   size_t GetNoElements() const { return this->GetSize(); }

   // return the size of the first dimension (if in row order) or last dimension if in column order
   // Tensor is  F x H x W x...for row order layout FHWC
   // or      H x W x ... x F  for column order layout CHWF
   // logic copied from TCudaTensor
   size_t GetFirstSize() const
   {
      auto& fShape = this->GetShape();
      return (this->GetMemoryLayout() == MemoryLayout::ColumnMajor) ? fShape.back() : fShape.front();
   }

   size_t GetCSize() const
   {
      auto& fShape = this->GetShape();
      if (fShape.size() == 2)  return 1;
      return (this->GetMemoryLayout() == MemoryLayout::ColumnMajor) ? fShape.front() : fShape[1]; // assume NHWC
   }
   //
   size_t GetHSize() const
   {
      auto& fShape = this->GetShape();
      if (fShape.size() == 2)  return fShape[0];
      if (fShape.size() == 3)  return (this->GetMemoryLayout() == MemoryLayout::ColumnMajor) ? fShape[0] : fShape[1] ;
      if (fShape.size() >= 4)  return fShape[2] ;
      return 0;

   }
   size_t GetWSize() const
   {
      auto& fShape = this->GetShape();
      if (fShape.size() == 2)  return fShape[1];
      if (fShape.size() == 3)  return (this->GetMemoryLayout() == MemoryLayout::ColumnMajor) ? fShape[1] : fShape[2] ;
      if (fShape.size() >= 4)  return fShape[3] ;
      return 0;

   }

   MemoryLayout GetLayout() const { return this->GetMemoryLayout(); }

   //this will be an unsafe view. Method exists for backwards compatibility only
   TCpuMatrix<AFloat> GetMatrix() const
   {
      Shape_t shape;
      auto& fShape = this->GetShape();
      //check if squeezable but do not actually squeeze
      for (auto& shape_i : fShape){
         if (shape_i != 1) {
            shape.emplace_back(shape_i);
         }
      }
      assert(shape.size() == 2);
      return TCpuMatrix<AFloat>(*(this->GetContainer()), GetHSize(), GetWSize());
   }


   // return a view of slices in the first dimension (if row wise) or last dimension if colun wise
   // so single event slices
   TCpuTensor<AFloat> At(size_t i)
   {
      auto& fShape = this->GetShape();
      auto fLayout = this->GetMemoryLayout();
      Shape_t sliced_shape = (fLayout == MemoryLayout::RowMajor)
                                ? Shape_t(fShape.begin() + 1, fShape.end())
                                : Shape_t(fShape.begin(), fShape.end() - 1);

      size_t buffsize = (fLayout == MemoryLayout::RowMajor) ? this->GetStrides().front() : this->GetStrides().back();
      size_t offset = i * buffsize;

      return TCpuTensor<AFloat>(this->GetContainer()->GetSubBuffer(offset, buffsize), sliced_shape, fLayout);
   }

   TCpuTensor<AFloat> At(size_t i) const { return (const_cast<TCpuTensor<AFloat> &>(*this)).At(i); }

   // set all the tensor contents to zero
   void Zero()
   {
      AFloat *data = *(this->GetContainer());
      for (size_t i = 0; i < this->GetSize(); ++i)
         data[i] = 0;
   }

   // access single element - assume tensor dim is 2
   AFloat &operator()(size_t i, size_t j)
   {
      auto& fShape = this->GetShape();
      assert(fShape.size() == 2);
      return (this->GetMemoryLayout() == MemoryLayout::RowMajor) ? (*(this->GetContainer()))[i * fShape[1] + j]
                                                                   : (*(this->GetContainer()))[j * fShape[0] + i];
   }

   // access single element - assume tensor dim is 3. First index i is always the major  indipendent of row-major or
   // column major row- major  I - J - K    . Column- major  is  J - K - I
   AFloat &operator()(size_t i, size_t j, size_t k)
   {
      auto& fShape = this->GetShape();
      assert(fShape.size() == 3);

      return (this->GetMemoryLayout() == MemoryLayout::RowMajor)
                ? (*(this->GetContainer()))[i * fShape[1] * fShape[2] + j * fShape[2] + k]
                : (*(this->GetContainer()))[i * fShape[0] * fShape[1] + k * fShape[0] + j]; // note that is J-K-I
   }

   // access single element - assume tensor dim is 2
   AFloat operator()(size_t i, size_t j) const
   {
      auto& fShape = this->GetShape();
      assert(fShape.size() == 2);
      return (this->GetMemoryLayout() == MemoryLayout::RowMajor) ? (this->GetData())[i * fShape[1] + j]
                                                                   : (this->GetData())[j * fShape[0] + i];
   }

   AFloat operator()(size_t i, size_t j, size_t k) const
   {
      auto& fShape = this->GetShape();
      assert(fShape.size() == 3);

      return (this->GetMemoryLayout() == MemoryLayout::RowMajor)
                ? (this->GetData())[i * fShape[1] * fShape[2] + j * fShape[2] + k]
                : (this->GetData())[i * fShape[0] * fShape[1] + k * fShape[0] + j]; // note that is J-K-I
   }

   /** Map the given function over the matrix elements. Executed in parallel
    *  using TThreadExecutor. */
   template <typename Function_t>
   void Map(Function_t &f);

   /** Same as maps but takes the input values from the tensor \p A and writes
    *  the results in this tensor. */
   template <typename Function_t>
   void MapFrom(Function_t &f, const TCpuTensor<AFloat> &A);

   size_t GetBufferUseCount() const { return this->GetContainer()->GetUseCount(); }

   void Print() const {

      auto& fData = this->GetData();
      auto& fSize = this->GetSize();
      for (size_t i = 0; i < fSize; i++) std::cout << fData[i] << "  ";
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
