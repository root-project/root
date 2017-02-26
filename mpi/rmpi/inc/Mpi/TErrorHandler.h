// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TErrorHandler
#define ROOT_Mpi_TErrorHandler

#include<Mpi/Globals.h>

namespace ROOT {
   namespace Mpi {
      class TErrorHandler: public TObject {
         MPI_Errhandler fErrorHandler;
      public:
         TErrorHandler();
         TErrorHandler(const TErrorHandler &err);

         virtual ~TErrorHandler() { }

         static Int_t GetErrorClass(Int_t errcode);
         static TString GetErrorString(Int_t errcode);
         static void  SetErrorString(Int_t errcode, const TString msg);
         static Int_t CreateErrorClass();
         static Int_t CreateErrorCode(Int_t errclass);

         inline TErrorHandler &operator=(const TErrorHandler &errhanlder)
         {
            fErrorHandler = errhanlder.fErrorHandler;
            return *this;
         }

         // comparison
         inline Bool_t operator==(const TErrorHandler &errhanlder)
         {
            return (Bool_t)(fErrorHandler == errhanlder.fErrorHandler);
         }

         inline Bool_t operator!=(const TErrorHandler &errhanlder)
         {
            return (Bool_t)!(*this == errhanlder);
         }
         virtual void Free();
      protected:
         // inter-language operability
         inline TErrorHandler &operator= (const MPI_Errhandler &errhanlder)
         {
            fErrorHandler = errhanlder;
            return *this;
         }

         inline operator MPI_Errhandler() const
         {
            return fErrorHandler;
         }

         ClassDef(TErrorHandler, 0)
      };
   }
}
#endif
