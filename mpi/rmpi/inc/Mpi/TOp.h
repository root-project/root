// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TOp
#define ROOT_Mpi_TOp
namespace ROOT {
   namespace Mpi {
   /**
    * \class TOp
    * template class with helper functions to perform operations in a reduce
    * schema.
    * \see also ROOT::Mpi::SUM ROOT::Mpi::PROD ROOT::Mpi::MIN ROOT::Mpi::MAX
    * \ingroup Mpi
    */

   template <typename T> class TOp {
     T (*fOp)(const T &, const T &);

   public:
     TOp(T (*op)(const T &, const T &)) : fOp(op) {}
     TOp(const TOp<T> &op) : fOp(op.fOp) {}

     TOp<T> &operator=(TOp<T> const &obj) {
       fOp = obj.fOp;
       return *this;
     }

     //______________________________________________________________________________
     /**
      * Method to call the encapsulate function with the operation.
      * \param a object to perform the binary operation
      * \param b object to perform the binary operation
      * \return object with the result of the operation.
      */
     T Call(const T &a, const T &b) const { return fOp(a, b); }

     T operator()(const T &a, const T &b) const { return fOp(a, b); }
      };

      //______________________________________________________________________________
      /**
       * template utility function to perform SUM operation,
       * if a and b are objects then a and b must be overloaded the operator +
       * \return object of TOp with the operation + saved like a function.
       */
      template <class T> TOp<T> SUM() {
        return TOp<T>([](const T &a, const T &b) { return a + b; });
      }

      //______________________________________________________________________________
      /**
       * template utility function to perform PROD operation,
       * if a and b are objects then a and b must be overloaded the operator *
       * \return object of TOp with the operation * saved like a function.
       */
      template <class T> TOp<T> PROD() {
        return TOp<T>([](const T &a, const T &b) { return a * b; });
      }

      //______________________________________________________________________________
      /**
       * template utility function to perform MIN operation,
       * if a and b are objects then a and b must be overloaded the operator <
       * \return object of TOp with operation function that compute the min
       * between two values.
       */
      template <class T> TOp<T> MIN() {
        return TOp<T>([](const T &a, const T &b) { return a < b ? a : b; });
      }

      //______________________________________________________________________________
      /**
       * template utility function to perform MAX operation,
       * if a and b are objects then a and b must be overloaded the operator >
       * \return object of TOp with operation function that compute the max
       * between two values.
       */
      template <class T> TOp<T> MAX() {
        return TOp<T>([](const T &a, const T &b) { return a > b ? a : b; });
      }


   }//end namespace Mpi
}//end namespace ROOT


#endif
