// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TOp
#define ROOT_Mpi_TOp
namespace ROOT {
   namespace Mpi {
      template<typename T> class Op {
         T(*fOp)(const T &, const T &);
      public:
         Op(T(*op)(const T &, const T &)): fOp(op) {}
         Op(const Op<T> &op): fOp(op.fOp) {}

         Op<T> &operator=(Op<T> const &obj)
         {
            fOp = obj.fOp;
            return *this;
         }

         T Call(const T &a, const T &b) const
         {
            return fOp(a, b);
         }

         T operator()(const T &a, const T &b) const
         {
            return fOp(a, b);
         }
      };

#if (__cplusplus >= 201402L)  //C++14
      template<class T> Op<T> SUM()
      {
         return Op<T>([](auto a, auto b) {
            return a + b;
         });
      }
      template<class T> Op<T> PROD()
      {
         return Op<T>([](auto a, auto b) {
            return a * b;
         });
      }

      template<class T> Op<T> MIN()
      {
         return Op<T>([](auto a, auto b) {
            return a < b ? a : b ;
         });
      }

      template<class T> Op<T> MAX()
      {
         return Op<T>([](auto a, auto b) {
            return a > b ? a : b ;
         });
      }

#else
      template<class T> T SUM_OP(const T &a, const T &b)
      {
         return a + b;
      }
      template<class T> Op<T> SUM()
      {
         return Op<T>(SUM_OP<T>);
      }

      template<class T> T PROD_OP(const T &a, const T &b)
      {
         return a * b;
      }
      template<class T> Op<T> PROD()
      {
         return Op<T>(PROD_OP<T>);
      }

      template<class T> T MIN_OP(const T &a, const T &b)
      {
         return a < b ? a : b;
      }

      template<class T> Op<T> MIN()
      {
         return Op<T>(MIN_OP<T>);
      }

      template<class T> T MAX_OP(const T &a, const T &b)
      {
         return a > b ? a : b;
      }

      template<class T> Op<T> MAX()
      {
         return Op<T>(MAX_OP<T>);
      }

#endif

   }//end namespace Mpi
}//end namespace ROOT


#endif
