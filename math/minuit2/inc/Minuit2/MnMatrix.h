// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnMatrix
#define ROOT_Minuit2_MnMatrix

#include <Minuit2/MnMatrixfwd.h> // For typedefs

#include <ROOT/RSpan.hxx>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <new>
#include <ostream>

// comment out this line and recompile if you want to gain additional
// performance (the gain is mainly for "simple" functions which are easy
// to calculate and vanishes quickly if going to cost-intensive functions)
// the library is no longer thread save however

#ifdef MN_USE_STACK_ALLOC
#define _MN_NO_THREAD_SAVE_
#endif

namespace ROOT {

namespace Minuit2 {

namespace MnMatrix {

// Whether to cut the maximum number of parameters shown for vector and matrices
// set maximum number of printed parameters and return previous value
// A negative value will mean all parameters are printed
int SetMaxNP(int value);

// retrieve maximum number of printed parameters
int MaxNP();

} // namespace MnMatrix

/// define stack allocator symbol

class StackOverflow {};
class StackError {};

/** StackAllocator controls the memory allocation/deallocation of Minuit. If
    _MN_NO_THREAD_SAVE_ is defined, memory is taken from a pre-allocated piece
    of heap memory which is then used like a stack, otherwise via standard
    malloc/free. Note that defining _MN_NO_THREAD_SAVE_ makes the code thread-
    unsave. The gain in performance is mainly for cost-cheap FCN functions.
 */

class StackAllocator {

public:
   //   enum {default_size = 1048576};
   enum {
      default_size = 524288
   };

   StackAllocator() : fStack(nullptr)
   {
#ifdef _MN_NO_THREAD_SAVE_
      // std::cout<<"StackAllocator Allocate "<<default_size<<std::endl;
      fStack = new unsigned char[default_size];
#endif
      fStackOffset = 0;
      fBlockCount = 0;
   }

   ~StackAllocator()
   {
#ifdef _MN_NO_THREAD_SAVE_
      // std::cout<<"StackAllocator destruct "<<fStackOffset<<std::endl;
      if (fStack)
         delete[] fStack;
#endif
   }

   void *Allocate(size_t nBytes)
   {
#ifdef _MN_NO_THREAD_SAVE_
      if (fStack == 0)
         fStack = new unsigned char[default_size];
      int nAlloc = AlignedSize(nBytes);
      CheckOverflow(nAlloc);

      //       std::cout << "Allocating " << nAlloc << " bytes, requested = " << nBytes << std::endl;

      // write the start position of the next block at the start of the block
      WriteInt(fStackOffset, fStackOffset + nAlloc);
      // write the start position of the new block at the end of the block
      WriteInt(fStackOffset + nAlloc - sizeof(int), fStackOffset);

      void *result = fStack + fStackOffset + sizeof(int);
      fStackOffset += nAlloc;
      fBlockCount++;

#ifdef DEBUG_ALLOCATOR
      CheckConsistency();
#endif

#else
      void *result = malloc(nBytes);
      if (!result)
         throw std::bad_alloc();
#endif

      return result;
   }

   void Deallocate(void *p)
   {
#ifdef _MN_NO_THREAD_SAVE_
      // int previousOffset = ReadInt( fStackOffset - sizeof(int));
      int delBlock = ToInt(p);
      int nextBlock = ReadInt(delBlock);
      int previousBlock = ReadInt(nextBlock - sizeof(int));
      if (nextBlock == fStackOffset) {
         // deallocating last allocated
         fStackOffset = previousBlock;
      } else {
         // overwrite previous adr of next block
         int nextNextBlock = ReadInt(nextBlock);
         WriteInt(nextNextBlock - sizeof(int), previousBlock);
         // overwrite head of deleted block
         WriteInt(previousBlock, nextNextBlock);
      }
      fBlockCount--;

#ifdef DEBUG_ALLOCATOR
      CheckConsistency();
#endif
#else
      free(p);
#endif
      // std::cout << "Block at " << delBlock
      //   << " deallocated, fStackOffset = " << fStackOffset << std::endl;
   }

   int ReadInt(int offset)
   {
      int *ip = (int *)(fStack + offset);

      // std::cout << "read " << *ip << " from offset " << offset << std::endl;

      return *ip;
   }

   void WriteInt(int offset, int Value)
   {

      // std::cout << "writing " << Value << " to offset " << offset << std::endl;

      int *ip = reinterpret_cast<int *>(fStack + offset);
      *ip = Value;
   }

   int ToInt(void *p)
   {
      unsigned char *pc = static_cast<unsigned char *>(p);

      // std::cout << "toInt: p = " << p << " fStack = " << (void*) fStack << std::endl;
      // VC 7.1 warning:conversion from __w64 int to int
      int userBlock = pc - fStack;
      return userBlock - sizeof(int); // correct for starting int
   }

   int AlignedSize(int nBytes)
   {
      const int fAlignment = 4;
      int needed = nBytes % fAlignment == 0 ? nBytes : (nBytes / fAlignment + 1) * fAlignment;
      return needed + 2 * sizeof(int);
   }

   void CheckOverflow(int n)
   {
      if (fStackOffset + n >= default_size) {
         // std::cout << " no more space on stack allocator" << std::endl;
         throw StackOverflow();
      }
   }

   bool CheckConsistency()
   {

      // std::cout << "checking consistency for " << fBlockCount << " blocks"<< std::endl;

      // loop over all blocks
      int beg = 0;
      int end = fStackOffset;
      int nblocks = 0;
      while (beg < fStackOffset) {
         end = ReadInt(beg);

         // std::cout << "beg = " << beg << " end = " << end
         //     << " fStackOffset = " << fStackOffset << std::endl;

         int beg2 = ReadInt(end - sizeof(int));
         if (beg != beg2) {
            // std::cout << "  beg != beg2 " << std::endl;
            return false;
         }
         nblocks++;
         beg = end;
      }
      if (end != fStackOffset) {
         // std::cout << " end != fStackOffset" << std::endl;
         return false;
      }
      if (nblocks != fBlockCount) {
         // std::cout << "nblocks != fBlockCount" << std::endl;
         return false;
      }
      // std::cout << "Allocator is in consistent state, nblocks = " << nblocks << std::endl;
      return true;
   }

private:
   unsigned char *fStack;
   //   unsigned char fStack[default_size];
   int fStackOffset;
   int fBlockCount;
};

class StackAllocatorHolder {

   // t.b.d need to use same trick as  Boost singleton.hpp to be sure that
   // StackAllocator is created before main()

public:
   static StackAllocator &Get()
   {
      static StackAllocator gStackAllocator;
      return gStackAllocator;
   }
};

class sym {};
class vec {};

// Helper base class to delete assignment.
//
// Note that base classes without any data members or virtual functions don't
// cause any runtime overhead.
//
// Assignment is often historically deleted (that was done probably to avoid
// mistakes from accidental re-assignment). Also define destructor and copy
// constructor in this case, according to the rule of five.
class DeleteAssignment {
public:
   DeleteAssignment() = default;
   ~DeleteAssignment() = default;
   DeleteAssignment(const DeleteAssignment &) = default;
   DeleteAssignment(DeleteAssignment &&) = default;
   DeleteAssignment &operator=(const DeleteAssignment &) = delete;
   DeleteAssignment &operator=(DeleteAssignment &&) = delete;
};

template <class Type, class M, class T = double>
class ABObj : public DeleteAssignment {

public:
   ABObj(const M &obj, T factor = 1.) : fObject(obj), fFactor(factor) {}

   const M &Obj() const { return fObject; }

   T f() const { return fFactor; }

private:
   M fObject;
   T fFactor;
};

// templated scaling operator *
template <class mt, class M, class T>
ABObj<mt, M, T> operator*(T f, const M &obj)
{
   return {obj, f};
}

// templated operator /
template <class mt, class M, class T>
ABObj<mt, M, T> operator/(const M &obj, T f)
{
   return {obj, T(1.) / f};
}

// templated unary operator -
template <class mt, class M, class T>
ABObj<mt, M, T> operator-(const M &obj)
{
   return {obj, -1.};
}

// factor * ABObj
template <class mt, class M, class T>
ABObj<mt, M, T> operator*(T f, const ABObj<mt, M, T> &obj)
{
   return {obj.Obj(), obj.f() * f};
}

// ABObj / factor
template <class mt, class M, class T>
ABObj<mt, M, T> operator/(const ABObj<mt, M, T> &obj, T f)
{
   return {obj.Obj(), obj.f() / f};
}

// -ABObj
template <class mt, class M, class T>
ABObj<mt, M, T> operator-(const ABObj<mt, M, T> &obj)
{
   return {obj.Obj(), T(-1.) * obj.f()};
}

template <class M1, class M2>
class ABSum : public DeleteAssignment {
public:
   ABSum(const M1 &a, const M2 &b) : fA(a), fB(b) {}

   const M1 &A() const { return fA; }
   const M2 &B() const { return fB; }

private:
   M1 fA;
   M2 fB;
};

// ABObj + ABObj
template <class atype, class A, class B, class T>
ABObj<atype, ABSum<ABObj<atype, A, T>, ABObj<atype, B, T>>, T>
operator+(const ABObj<atype, A, T> &a, const ABObj<atype, B, T> &b)
{

   return {ABSum<ABObj<atype, A, T>, ABObj<atype, B, T>>(a, b)};
}

// ABObj - ABObj
template <class atype, class A, class B, class T>
ABObj<atype, ABSum<ABObj<atype, A, T>, ABObj<atype, B, T>>, T>
operator-(const ABObj<atype, A, T> &a, const ABObj<atype, B, T> &b)
{

   return {ABSum<ABObj<atype, A, T>, ABObj<atype, B, T>>(a, ABObj<atype, B, T>(b.Obj(), T(-1.) * b.f()))};
}

template <class M1, class M2>
class ABProd : public DeleteAssignment {
public:
   ABProd(const M1 &a, const M2 &b) : fA(a), fB(b) {}

   const M1 &A() const { return fA; }
   const M2 &B() const { return fB; }

private:
   M1 fA;
   M2 fB;
};

// ABObj * ABObj (only supported for sym * vec)
template <class A, class B, class T>
ABObj<vec, ABProd<ABObj<sym, A, T>, ABObj<vec, B, T>>, T>
operator*(const ABObj<sym, A, T> &a, const ABObj<vec, B, T> &b)
{

   return {ABProd<ABObj<sym, A, T>, ABObj<vec, B, T>>(a, b)};
}

template <class M, class T>
class VectorOuterProduct {

public:
   VectorOuterProduct(const M &obj) : fObject(obj) {}

   const M &Obj() const { return fObject; }

private:
   M fObject;
};

template <class M, class T>
ABObj<sym, VectorOuterProduct<ABObj<vec, M, T>, T>, T> Outer_product(const ABObj<vec, M, T> &obj)
{
   return {VectorOuterProduct<ABObj<vec, M, T>, T>(obj)};
}

template <class mtype, class M, class T>
class MatrixInverse {

public:
   MatrixInverse(const M &obj) : fObject(obj) {}

   const M &Obj() const { return fObject; }

private:
   M fObject;
};

// Matrix inverse of a vector is not possible.
template <class M, class T>
class MatrixInverse<vec, M, T> {
   MatrixInverse() = delete;
   MatrixInverse(const MatrixInverse &) = delete;
   MatrixInverse(MatrixInverse &&) = delete;
};

template <class mt, class M, class T>
inline ABObj<mt, MatrixInverse<mt, ABObj<mt, M, T>, T>, T> Inverse(const ABObj<mt, M, T> &obj)
{
   return {MatrixInverse<mt, ABObj<mt, M, T>, T>(obj)};
}

void Mndaxpy(unsigned int, double, const double *, double *);
void Mndscal(unsigned int, double, double *);

class LASymMatrix;
class LAVector;

int Invert(LASymMatrix &);

/**
   Class describing a symmetric matrix of size n.
   The size is specified as a run-time argument passed in the
   constructor.
   The class uses expression templates for the operations and functions.
   Only the independent data are kept in the fdata array of size n*(n+1)/2
   containing the lower triangular data
 */

class LASymMatrix {

public:
   typedef sym Type;

   LASymMatrix(unsigned int n)
      : fSize(n * (n + 1) / 2),
        fNRow(n),
        fData((n > 0) ? (double *)StackAllocatorHolder::Get().Allocate(sizeof(double) * n * (n + 1) / 2) : nullptr)
   {
      //     assert(fSize>0);
      if (fData)
         std::memset(fData, 0, fSize * sizeof(double));
   }

   ~LASymMatrix()
   {
      if (fData)
         StackAllocatorHolder::Get().Deallocate(fData);
   }

   LASymMatrix(const LASymMatrix &v)
      : fSize(v.size()),
        fNRow(v.Nrow()),
        fData((double *)StackAllocatorHolder::Get().Allocate(sizeof(double) * v.size()))
   {
      std::memcpy(fData, v.Data(), fSize * sizeof(double));
   }

   LASymMatrix(LASymMatrix &&v)
      : fSize(v.size()),
        fNRow(v.Nrow()),
        fData(v.fData)
   {
      v.fData = nullptr;
   }


   LASymMatrix &operator=(const LASymMatrix &v)
   {
      if (fSize < v.size()) {
         if (fData)
            StackAllocatorHolder::Get().Deallocate(fData);
         fSize = v.size();
         fNRow = v.Nrow();
         fData = (double *)StackAllocatorHolder::Get().Allocate(sizeof(double) * fSize);
      } else if (fSize > v.size()) {
         throw std::runtime_error("Can't assign smaller LASymMatrix to larger LASymMatrix");
      }
      std::memcpy(fData, v.Data(), fSize * sizeof(double));
      return *this;
   }

   LASymMatrix &operator=(LASymMatrix &&v)
   {
      fSize = v.size();
      fNRow = v.Nrow();
      fData = v.Data();
      v.fData = nullptr;
      return *this;
   }

   template <class T>
   LASymMatrix(const ABObj<sym, LASymMatrix, T> &v)
      : fSize(v.Obj().size()),
        fNRow(v.Obj().Nrow()),
        fData((double *)StackAllocatorHolder::Get().Allocate(sizeof(double) * v.Obj().size()))
   {
      //     std::cout<<"LASymMatrix(const ABObj<sym, LASymMatrix, T>& v)"<<std::endl;
      // std::cout<<"allocate "<<fSize<<std::endl;
      std::memcpy(fData, v.Obj().Data(), fSize * sizeof(double));
      Mndscal(fSize, double(v.f()), fData);
      // std::cout<<"fData= "<<fData[0]<<" "<<fData[1]<<std::endl;
   }

   template <class A, class B, class T>
   LASymMatrix(const ABObj<sym, ABSum<ABObj<sym, A, T>, ABObj<sym, B, T>>, T> &sum)
   {
      //     std::cout<<"template<class A, class B, class T> LASymMatrix(const ABObj<sym, ABSum<ABObj<sym, A, T>,
      //     ABObj<sym, B, T> > >& sum)"<<std::endl; recursive construction
      (*this) = sum.Obj().A();
      (*this) += sum.Obj().B();
      // std::cout<<"leaving template<class A, class B, class T> LASymMatrix(const ABObj..."<<std::endl;
   }

   template <class A, class T>
   LASymMatrix(const ABObj<sym, ABSum<ABObj<sym, LASymMatrix, T>, ABObj<sym, A, T>>, T> &sum)
   {
      //     std::cout<<"template<class A, class T> LASymMatrix(const ABObj<sym, ABSum<ABObj<sym, LASymMatrix, T>,
      //     ABObj<sym, A, T> >,T>& sum)"<<std::endl;

      // recursive construction
      // std::cout<<"(*this)=sum.Obj().B();"<<std::endl;
      (*this) = sum.Obj().B();
      // std::cout<<"(*this)+=sum.Obj().A();"<<std::endl;
      (*this) += sum.Obj().A();
      // std::cout<<"leaving template<class A, class T> LASymMatrix(const ABObj<sym, ABSum<ABObj<sym,
      // LASymMatrix,.."<<std::endl;
   }

   template <class A, class T>
   LASymMatrix(const ABObj<sym, ABObj<sym, A, T>, T> &something)
   {
      //     std::cout<<"template<class A, class T> LASymMatrix(const ABObj<sym, ABObj<sym, A, T>, T>&
      //     something)"<<std::endl;
      (*this) = something.Obj();
      (*this) *= something.f();
      // std::cout<<"leaving template<class A, class T> LASymMatrix(const ABObj<sym, ABObj<sym, A, T>, T>&
      // something)"<<std::endl;
   }

   template <class T>
   LASymMatrix(const ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, T>, T>, T> &inv)
      : fSize(inv.Obj().Obj().Obj().size()),
        fNRow(inv.Obj().Obj().Obj().Nrow()),
        fData((double *)StackAllocatorHolder::Get().Allocate(sizeof(double) * inv.Obj().Obj().Obj().size()))
   {
      std::memcpy(fData, inv.Obj().Obj().Obj().Data(), fSize * sizeof(double));
      Mndscal(fSize, double(inv.Obj().Obj().f()), fData);
      Invert(*this);
      Mndscal(fSize, double(inv.f()), fData);
   }

   template <class A, class T>
   LASymMatrix(
      const ABObj<sym, ABSum<ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, T>, T>, T>, ABObj<sym, A, T>>, T>
         &sum)
   {
      //     std::cout<<"template<class A, class T> LASymMatrix(const ABObj<sym, ABSum<ABObj<sym, MatrixInverse<sym,
      //     ABObj<sym, LASymMatrix, T>, T>, T>, ABObj<sym, A, T> >, T>& sum)"<<std::endl;

      // recursive construction
      (*this) = sum.Obj().B();
      (*this) += sum.Obj().A();
      // std::cout<<"leaving template<class A, class T> LASymMatrix(const ABObj<sym, ABSum<ABObj<sym,
      // LASymMatrix,.."<<std::endl;
   }

   LASymMatrix(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, double>, double>, double> &);

   template <class A, class T>
   LASymMatrix(
      const ABObj<sym, ABSum<ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T>, ABObj<sym, A, T>>, T> &sum)
   {
      //     std::cout<<"template<class A, class T> LASymMatrix(const ABObj<sym, ABSum<ABObj<sym,
      //     VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T> ABObj<sym, A, T> >,T>& sum)"<<std::endl;

      // recursive construction
      (*this) = sum.Obj().B();
      (*this) += sum.Obj().A();
      // std::cout<<"leaving template<class A, class T> LASymMatrix(const ABObj<sym, ABSum<ABObj<sym,
      // LASymMatrix,.."<<std::endl;
   }

   LASymMatrix &operator+=(const LASymMatrix &m)
   {
      //     std::cout<<"LASymMatrix& operator+=(const LASymMatrix& m)"<<std::endl;
      assert(fSize == m.size());
      Mndaxpy(fSize, 1., m.Data(), fData);
      return *this;
   }

   LASymMatrix &operator-=(const LASymMatrix &m)
   {
      //     std::cout<<"LASymMatrix& operator-=(const LASymMatrix& m)"<<std::endl;
      assert(fSize == m.size());
      Mndaxpy(fSize, -1., m.Data(), fData);
      return *this;
   }

   template <class T>
   LASymMatrix &operator+=(const ABObj<sym, LASymMatrix, T> &m)
   {
      //     std::cout<<"template<class T> LASymMatrix& operator+=(const ABObj<sym, LASymMatrix, T>& m)"<<std::endl;
      assert(fSize == m.Obj().size());
      if (m.Obj().Data() == fData) {
         Mndscal(fSize, 1. + double(m.f()), fData);
      } else {
         Mndaxpy(fSize, double(m.f()), m.Obj().Data(), fData);
      }
      // std::cout<<"fData= "<<fData[0]<<" "<<fData[1]<<std::endl;
      return *this;
   }

   template <class A, class T>
   LASymMatrix &operator+=(const ABObj<sym, A, T> &m)
   {
      //     std::cout<<"template<class A, class T> LASymMatrix& operator+=(const ABObj<sym, A,T>& m)"<<std::endl;
      (*this) += LASymMatrix(m);
      return *this;
   }

   template <class T>
   LASymMatrix &operator+=(const ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, T>, T>, T> &m)
   {
      //     std::cout<<"template<class T> LASymMatrix& operator+=(const ABObj<sym, MatrixInverse<sym, ABObj<sym,
      //     LASymMatrix, T>, T>, T>& m)"<<std::endl;
      assert(fNRow > 0);
      LASymMatrix tmp(m.Obj().Obj());
      Invert(tmp);
      tmp *= double(m.f());
      (*this) += tmp;
      return *this;
   }

   template <class T>
   LASymMatrix &operator+=(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, T>, T>, T> &m)
   {
      //     std::cout<<"template<class T> LASymMatrix& operator+=(const ABObj<sym, VectorOuterProduct<ABObj<vec,
      //     LAVector, T>, T>, T>&"<<std::endl;
      assert(fNRow > 0);
      Outer_prod(*this, m.Obj().Obj().Obj(), m.f() * m.Obj().Obj().f() * m.Obj().Obj().f());
      return *this;
   }

   LASymMatrix &operator*=(double scal)
   {
      Mndscal(fSize, scal, fData);
      return *this;
   }

   double operator()(unsigned int row, unsigned int col) const
   {
      assert(row < fNRow && col < fNRow);
      if (row > col)
         return fData[col + row * (row + 1) / 2];
      else
         return fData[row + col * (col + 1) / 2];
   }

   double &operator()(unsigned int row, unsigned int col)
   {
      assert(row < fNRow && col < fNRow);
      if (row > col)
         return fData[col + row * (row + 1) / 2];
      else
         return fData[row + col * (col + 1) / 2];
   }

   const double *Data() const { return fData; }

   double *Data() { return fData; }

   unsigned int size() const { return fSize; }

   unsigned int Nrow() const { return fNRow; }

   unsigned int Ncol() const { return Nrow(); }

private:
   unsigned int fSize = 0;
   unsigned int fNRow = 0;
   double *fData = nullptr;

public:
   template <class T>
   LASymMatrix &operator=(const ABObj<sym, LASymMatrix, T> &v)
   {
      // std::cout<<"template<class T> LASymMatrix& operator=(ABObj<sym, LASymMatrix, T>& v)"<<std::endl;
      if (fSize == 0 && !fData) {
         fSize = v.Obj().size();
         fNRow = v.Obj().Nrow();
         fData = (double *)StackAllocatorHolder::Get().Allocate(sizeof(double) * fSize);
      } else {
         assert(fSize == v.Obj().size());
      }
      // std::cout<<"fData= "<<fData[0]<<" "<<fData[1]<<std::endl;
      std::memcpy(fData, v.Obj().Data(), fSize * sizeof(double));
      (*this) *= v.f();
      return *this;
   }

   template <class A, class T>
   LASymMatrix &operator=(const ABObj<sym, ABObj<sym, A, T>, T> &something)
   {
      // std::cout<<"template<class A, class T> LASymMatrix& operator=(const ABObj<sym, ABObj<sym, A, T>, T>&
      // something)"<<std::endl;
      if (fSize == 0 && fData == nullptr) {
         (*this) = something.Obj();
         (*this) *= something.f();
      } else {
         LASymMatrix tmp(something.Obj());
         tmp *= something.f();
         assert(fSize == tmp.size());
         std::memcpy(fData, tmp.Data(), fSize * sizeof(double));
      }
      // std::cout<<"template<class A, class T> LASymMatrix& operator=(const ABObj<sym, ABObj<sym, A, T>, T>&
      // something)"<<std::endl;
      return *this;
   }

   template <class A, class B, class T>
   LASymMatrix &operator=(const ABObj<sym, ABSum<ABObj<sym, A, T>, ABObj<sym, B, T>>, T> &sum)
   {
      // std::cout<<"template<class A, class B, class T> LASymMatrix& operator=(const ABObj<sym, ABSum<ABObj<sym, A, T>,
      // ABObj<sym, B, T> >,T>& sum)"<<std::endl;
      // recursive construction
      if (fSize == 0 && fData == nullptr) {
         (*this) = sum.Obj().A();
         (*this) += sum.Obj().B();
         (*this) *= sum.f();
      } else {
         LASymMatrix tmp(sum.Obj().A());
         tmp += sum.Obj().B();
         tmp *= sum.f();
         assert(fSize == tmp.size());
         std::memcpy(fData, tmp.Data(), fSize * sizeof(double));
      }
      return *this;
   }

   template <class A, class T>
   LASymMatrix &operator=(const ABObj<sym, ABSum<ABObj<sym, LASymMatrix, T>, ABObj<sym, A, T>>, T> &sum)
   {
      // std::cout<<"template<class A, class T> LASymMatrix& operator=(const ABObj<sym, ABSum<ABObj<sym, LASymMatrix,
      // T>, ABObj<sym, A, T> >,T>& sum)"<<std::endl;

      if (fSize == 0 && fData == nullptr) {
         // std::cout<<"fSize == 0 && fData == 0"<<std::endl;
         (*this) = sum.Obj().B();
         (*this) += sum.Obj().A();
         (*this) *= sum.f();
      } else {
         // std::cout<<"creating tmp variable"<<std::endl;
         LASymMatrix tmp(sum.Obj().B());
         tmp += sum.Obj().A();
         tmp *= sum.f();
         assert(fSize == tmp.size());
         std::memcpy(fData, tmp.Data(), fSize * sizeof(double));
      }
      // std::cout<<"leaving LASymMatrix& operator=(const ABObj<sym, ABSum<ABObj<sym, LASymMatrix..."<<std::endl;
      return *this;
   }

   template <class T>
   LASymMatrix &operator=(const ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix, T>, T>, T> &inv)
   {
      if (fSize == 0 && fData == nullptr) {
         fSize = inv.Obj().Obj().Obj().size();
         fNRow = inv.Obj().Obj().Obj().Nrow();
         fData = (double *)StackAllocatorHolder::Get().Allocate(sizeof(double) * fSize);
         std::memcpy(fData, inv.Obj().Obj().Obj().Data(), fSize * sizeof(double));
         (*this) *= inv.Obj().Obj().f();
         Invert(*this);
         (*this) *= inv.f();
      } else {
         LASymMatrix tmp(inv.Obj().Obj());
         Invert(tmp);
         tmp *= double(inv.f());
         assert(fSize == tmp.size());
         std::memcpy(fData, tmp.Data(), fSize * sizeof(double));
      }
      return *this;
   }

   LASymMatrix &operator=(const ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector, double>, double>, double> &);
};

inline ABObj<sym, ABSum<ABObj<sym, LASymMatrix>, ABObj<sym, LASymMatrix>>>
operator+(const ABObj<sym, LASymMatrix> &a, const ABObj<sym, LASymMatrix> &b)
{
   return {ABSum<ABObj<sym, LASymMatrix>, ABObj<sym, LASymMatrix>>(a, b)};
}

inline ABObj<sym, ABSum<ABObj<sym, LASymMatrix>, ABObj<sym, LASymMatrix>>>
operator-(const ABObj<sym, LASymMatrix> &a, const ABObj<sym, LASymMatrix> &b)
{
   return {ABSum<ABObj<sym, LASymMatrix>, ABObj<sym, LASymMatrix>>(a, ABObj<sym, LASymMatrix>(b.Obj(), -1. * b.f()))};
}

inline ABObj<sym, LASymMatrix> operator*(double f, const LASymMatrix &obj)
{
   return {obj, f};
}

inline ABObj<sym, LASymMatrix> operator/(const LASymMatrix &obj, double f)
{
   return {obj, 1. / f};
}

inline ABObj<sym, LASymMatrix> operator-(const LASymMatrix &obj)
{
   return {obj, -1.};
}

void Mndaxpy(unsigned int, double, const double *, double *);
void Mndscal(unsigned int, double, double *);
void Mndspmv(unsigned int, double, const double *, const double *, double, double *);

class LAVector {

public:
   typedef vec Type;

   LAVector(unsigned int n)
      : fSize(n), fData((n > 0) ? (double *)StackAllocatorHolder::Get().Allocate(sizeof(double) * n) : nullptr)
   {
      if (fData)
         std::memset(fData, 0, size() * sizeof(double));
   }

   ~LAVector()
   {
      if (fData)
         StackAllocatorHolder::Get().Deallocate(fData);
   }

   LAVector(LAVector &&v)
      : fSize(v.size()), fData(v.Data())
   {
      v.fData = nullptr;
   }

   LAVector(const LAVector &v) : LAVector{std::span<const double>{v.Data(), v.size()}} {}

   explicit LAVector(std::span<const double> v)
      : fSize(v.size()), fData((double *)StackAllocatorHolder::Get().Allocate(sizeof(double) * v.size()))
   {
      std::memcpy(fData, v.data(), fSize * sizeof(double));
   }

   LAVector &operator=(const LAVector &v)
   {
      // implement proper copy constructor in case size is different
      if (v.size() > fSize) {
         if (fData)
            StackAllocatorHolder::Get().Deallocate(fData);
         fSize = v.size();
         fData = (double *)StackAllocatorHolder::Get().Allocate(sizeof(double) * fSize);
      } else if (fSize > v.size()) {
         throw std::runtime_error("Can't assign smaller LAVector to larger LAVector");
      }
      std::memcpy(fData, v.Data(), fSize * sizeof(double));
      return *this;
   }

   LAVector &operator=(LAVector &&v)
   {
      fSize = v.fSize;
      fData = v.fData;
      v.fData = nullptr;
      return *this;
   }

   template <class T>
   LAVector(const ABObj<vec, LAVector, T> &v)
      : fSize(v.Obj().size()), fData((double *)StackAllocatorHolder::Get().Allocate(sizeof(double) * v.Obj().size()))
   {
      //     std::cout<<"LAVector(const ABObj<LAVector, T>& v)"<<std::endl;
      //     std::cout<<"allocate "<<fSize<<std::endl;
      std::memcpy(fData, v.Obj().Data(), fSize * sizeof(T));
      (*this) *= T(v.f());
      //     std::cout<<"fData= "<<fData[0]<<" "<<fData[1]<<std::endl;
   }

   template <class A, class B, class T>
   LAVector(const ABObj<vec, ABSum<ABObj<vec, A, T>, ABObj<vec, B, T>>, T> &sum)
   {
      //     std::cout<<"template<class A, class B, class T> LAVector(const ABObj<ABSum<ABObj<A, T>, ABObj<B, T> > >&
      //     sum)"<<std::endl;
      (*this) = sum.Obj().A();
      (*this) += sum.Obj().B();
      (*this) *= double(sum.f());
   }

   template <class A, class T>
   LAVector(const ABObj<vec, ABSum<ABObj<vec, LAVector, T>, ABObj<vec, A, T>>, T> &sum)
   {
      //     std::cout<<"template<class A, class T> LAVector(const ABObj<ABSum<ABObj<LAVector, T>, ABObj<A, T> >,T>&
      //     sum)"<<std::endl;

      // recursive construction
      //     std::cout<<"(*this)=sum.Obj().B();"<<std::endl;
      (*this) = sum.Obj().B();
      //     std::cout<<"(*this)+=sum.Obj().A();"<<std::endl;
      (*this) += sum.Obj().A();
      (*this) *= double(sum.f());
      //     std::cout<<"leaving template<class A, class T> LAVector(const ABObj<ABSum<ABObj<LAVector,.."<<std::endl;
   }

   template <class A, class T>
   LAVector(const ABObj<vec, ABObj<vec, A, T>, T> &something)
   {
      //     std::cout<<"template<class A, class T> LAVector(const ABObj<ABObj<A, T>, T>& something)"<<std::endl;
      (*this) = something.Obj();
      (*this) *= something.f();
   }

   //
   template <class T>
   LAVector(const ABObj<vec, ABProd<ABObj<sym, LASymMatrix, T>, ABObj<vec, LAVector, T>>, T> &prod)
      : fSize(prod.Obj().B().Obj().size()),
        fData((double *)StackAllocatorHolder::Get().Allocate(sizeof(double) * prod.Obj().B().Obj().size()))
   {
      //     std::cout<<"template<class T> LAVector(const ABObj<vec, ABProd<ABObj<sym, LASymMatrix, T>, ABObj<vec,
      //     LAVector, T> >, T>& prod)"<<std::endl;

      Mndspmv(fSize, prod.f() * prod.Obj().A().f() * prod.Obj().B().f(), prod.Obj().A().Obj().Data(),
              prod.Obj().B().Obj().Data(), 0., fData);
   }

   //
   template <class T>
   LAVector(const ABObj<
            vec,
            ABSum<ABObj<vec, ABProd<ABObj<sym, LASymMatrix, T>, ABObj<vec, LAVector, T>>, T>, ABObj<vec, LAVector, T>>,
            T> &prod)
   {
      (*this) = prod.Obj().B();
      (*this) += prod.Obj().A();
      (*this) *= double(prod.f());
   }

   //
   LAVector &operator+=(const LAVector &m)
   {
      //     std::cout<<"LAVector& operator+=(const LAVector& m)"<<std::endl;
      assert(fSize == m.size());
      Mndaxpy(fSize, 1., m.Data(), fData);
      return *this;
   }

   LAVector &operator-=(const LAVector &m)
   {
      //     std::cout<<"LAVector& operator-=(const LAVector& m)"<<std::endl;
      assert(fSize == m.size());
      Mndaxpy(fSize, -1., m.Data(), fData);
      return *this;
   }

   template <class T>
   LAVector &operator+=(const ABObj<vec, LAVector, T> &m)
   {
      //     std::cout<<"template<class T> LAVector& operator+=(const ABObj<LAVector, T>& m)"<<std::endl;
      assert(fSize == m.Obj().size());
      if (m.Obj().Data() == fData) {
         Mndscal(fSize, 1. + double(m.f()), fData);
      } else {
         Mndaxpy(fSize, double(m.f()), m.Obj().Data(), fData);
      }
      //     std::cout<<"fData= "<<fData[0]<<" "<<fData[1]<<std::endl;
      return *this;
   }

   template <class A, class T>
   LAVector &operator+=(const ABObj<vec, A, T> &m)
   {
      //     std::cout<<"template<class A, class T> LAVector& operator+=(const ABObj<A,T>& m)"<<std::endl;
      (*this) += LAVector(m);
      return *this;
   }

   template <class T>
   LAVector &operator+=(const ABObj<vec, ABProd<ABObj<sym, LASymMatrix, T>, ABObj<vec, LAVector, T>>, T> &prod)
   {
      Mndspmv(fSize, prod.f() * prod.Obj().A().f() * prod.Obj().B().f(), prod.Obj().A().Obj().Data(),
              prod.Obj().B().Data(), 1., fData);
      return *this;
   }

   LAVector &operator*=(double scal)
   {
      Mndscal(fSize, scal, fData);
      return *this;
   }

   double operator()(unsigned int i) const
   {
      assert(i < fSize);
      return fData[i];
   }

   double &operator()(unsigned int i)
   {
      assert(i < fSize);
      return fData[i];
   }

   double operator[](unsigned int i) const
   {
      assert(i < fSize);
      return fData[i];
   }

   double &operator[](unsigned int i)
   {
      assert(i < fSize);
      return fData[i];
   }

   const double *Data() const { return fData; }

   double *Data() { return fData; }

   unsigned int size() const { return fSize; }

private:
   unsigned int fSize = 0;
   double *fData = nullptr;

public:
   template <class T>
   LAVector &operator=(const ABObj<vec, LAVector, T> &v)
   {
      //     std::cout<<"template<class T> LAVector& operator=(ABObj<LAVector, T>& v)"<<std::endl;
      if (fSize == 0 && !fData) {
         fSize = v.Obj().size();
         fData = (double *)StackAllocatorHolder::Get().Allocate(sizeof(double) * fSize);
      } else {
         assert(fSize == v.Obj().size());
      }
      std::memcpy(fData, v.Obj().Data(), fSize * sizeof(double));
      (*this) *= T(v.f());
      return *this;
   }

   template <class A, class T>
   LAVector &operator=(const ABObj<vec, ABObj<vec, A, T>, T> &something)
   {
      //     std::cout<<"template<class A, class T> LAVector& operator=(const ABObj<ABObj<A, T>, T>&
      //     something)"<<std::endl;
      if (fSize == 0 && !fData) {
         (*this) = something.Obj();
      } else {
         LAVector tmp(something.Obj());
         assert(fSize == tmp.size());
         std::memcpy(fData, tmp.Data(), fSize * sizeof(double));
      }
      (*this) *= something.f();
      return *this;
   }

   template <class A, class B, class T>
   LAVector &operator=(const ABObj<vec, ABSum<ABObj<vec, A, T>, ABObj<vec, B, T>>, T> &sum)
   {
      if (fSize == 0 && !fData) {
         (*this) = sum.Obj().A();
         (*this) += sum.Obj().B();
      } else {
         LAVector tmp(sum.Obj().A());
         tmp += sum.Obj().B();
         assert(fSize == tmp.size());
         std::memcpy(fData, tmp.Data(), fSize * sizeof(double));
      }
      (*this) *= sum.f();
      return *this;
   }

   template <class A, class T>
   LAVector &operator=(const ABObj<vec, ABSum<ABObj<vec, LAVector, T>, ABObj<vec, A, T>>, T> &sum)
   {
      if (fSize == 0 && !fData) {
         (*this) = sum.Obj().B();
         (*this) += sum.Obj().A();
      } else {
         LAVector tmp(sum.Obj().A());
         tmp += sum.Obj().B();
         assert(fSize == tmp.size());
         std::memcpy(fData, tmp.Data(), fSize * sizeof(double));
      }
      (*this) *= sum.f();
      return *this;
   }

   //
   template <class T>
   LAVector &operator=(const ABObj<vec, ABProd<ABObj<sym, LASymMatrix, T>, ABObj<vec, LAVector, T>>, T> &prod)
   {
      if (fSize == 0 && !fData) {
         fSize = prod.Obj().B().Obj().size();
         fData = (double *)StackAllocatorHolder::Get().Allocate(sizeof(double) * fSize);
         Mndspmv(fSize, double(prod.f() * prod.Obj().A().f() * prod.Obj().B().f()), prod.Obj().A().Obj().Data(),
                 prod.Obj().B().Obj().Data(), 0., fData);
      } else {
         LAVector tmp(prod.Obj().B());
         assert(fSize == tmp.size());
         Mndspmv(fSize, double(prod.f() * prod.Obj().A().f()), prod.Obj().A().Obj().Data(), tmp.Data(), 0., fData);
      }
      return *this;
   }

   //
   template <class T>
   LAVector &
   operator=(const ABObj<
             vec,
             ABSum<ABObj<vec, ABProd<ABObj<sym, LASymMatrix, T>, ABObj<vec, LAVector, T>>, T>, ABObj<vec, LAVector, T>>,
             T> &prod)
   {
      if (fSize == 0 && !fData) {
         (*this) = prod.Obj().B();
         (*this) += prod.Obj().A();
      } else {
         //       std::cout<<"creating tmp variable"<<std::endl;
         LAVector tmp(prod.Obj().B());
         tmp += prod.Obj().A();
         assert(fSize == tmp.size());
         std::memcpy(fData, tmp.Data(), fSize * sizeof(double));
      }
      (*this) *= prod.f();
      return *this;
   }
};

inline ABObj<vec, ABSum<ABObj<vec, LAVector>, ABObj<vec, LAVector>>>
operator+(const ABObj<vec, LAVector> &a, const ABObj<vec, LAVector> &b)
{
   return {ABSum<ABObj<vec, LAVector>, ABObj<vec, LAVector>>(a, b)};
}

template <class V>
ABObj<vec, ABSum<ABObj<vec, LAVector>, ABObj<vec, V>>> operator+(const ABObj<vec, LAVector> &a, const ABObj<vec, V> &b)
{
   return {ABSum<ABObj<vec, LAVector>, ABObj<vec, V>>(a, b)};
}

inline ABObj<vec, ABSum<ABObj<vec, LAVector>, ABObj<vec, LAVector>>>
operator-(const ABObj<vec, LAVector> &a, const ABObj<vec, LAVector> &b)
{
   return {ABSum<ABObj<vec, LAVector>, ABObj<vec, LAVector>>(a, ABObj<vec, LAVector>(b.Obj(), -1. * b.f()))};
}

inline ABObj<vec, LAVector> operator*(double f, const LAVector &obj)
{
   return {obj, f};
}
inline ABObj<vec, LAVector> operator/(const LAVector &obj, double f)
{
   return {obj, 1. / f};
}
inline ABObj<vec, LAVector> operator-(const LAVector &obj)
{
   return {obj, -1.};
}

// Matrix-vector product
inline ABObj<vec, ABProd<ABObj<sym, LASymMatrix>, ABObj<vec, LAVector>>>
operator*(const ABObj<sym, LASymMatrix> &a, const ABObj<vec, LAVector> &b)
{
   return {ABProd<ABObj<sym, LASymMatrix>, ABObj<vec, LAVector>>(a, b)};
}

///    LAPACK Algebra functions
///    specialize the Invert function for LASymMatrix

inline ABObj<sym, MatrixInverse<sym, ABObj<sym, LASymMatrix>, double>> Inverse(const ABObj<sym, LASymMatrix> &obj)
{
   return {MatrixInverse<sym, ABObj<sym, LASymMatrix>, double>{obj}};
}

int Invert(LASymMatrix &);

int Invert_undef_sym(LASymMatrix &);

///    LAPACK Algebra function
///    specialize the Outer_product function for LAVector;

inline ABObj<sym, VectorOuterProduct<ABObj<vec, LAVector>, double>> Outer_product(const ABObj<vec, LAVector> &obj)
{
   return {VectorOuterProduct<ABObj<vec, LAVector>, double>{obj}};
}

void Outer_prod(LASymMatrix &, const LAVector &, double f = 1.);

double inner_product(const LAVector &, const LAVector &);
double similarity(const LAVector &, const LASymMatrix &);
double sum_of_elements(const LASymMatrix &);
LAVector eigenvalues(const LASymMatrix &);

std::ostream &operator<<(std::ostream &, const LAVector &);

std::ostream &operator<<(std::ostream &, const LASymMatrix &);

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_MnMatrix
