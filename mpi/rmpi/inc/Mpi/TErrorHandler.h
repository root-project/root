// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TErrorHandler
#define ROOT_Mpi_TErrorHandler

#include<Mpi/Globals.h>
#include<Mpi/TEnvironment.h>

namespace ROOT {
   namespace Mpi {
      class TErrorHandler: public TObject {
         friend class TEnvironment;
         MPI_Errhandler fErrorHandler;
         static Bool_t fVerbose;
         TErrorHandler();
      public:
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

         static void SetVerbose(Bool_t status = kTRUE)
         {
            fVerbose = status;
         }

         static Bool_t IsVerbose()
         {
            return fVerbose;
         }

         template<class T> static void TraceBack(const T *comm, const Char_t *function, const Char_t *file, Int_t line, Int_t errcode, const Char_t *msg);

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

      template<class T> void TErrorHandler::TraceBack(const T *comm, const Char_t *function, const Char_t *file, Int_t line, Int_t errcode, const Char_t *_msg)
      {
         TString msg;
         if (TErrorHandler::IsVerbose()) {
            msg += Form("\nRank = %d", comm->GetRank());
            msg += Form("\nSize = %d", comm->GetSize());
            msg += Form("\nComm = %s", comm->ClassName());
            msg += Form("\nHost = %s", TEnvironment::GetProcessorName().Data());
         }

         msg += Form("\nCode = %d", errcode);
         msg += Form("\nName = %s", GetErrorString(errcode).Data());
         msg += Form("\nMessage = %s", _msg);
         msg += "\nAborting, finishing the remaining processes.";
         msg += "\n--------------------------------------------------------------------------\n";

         comm->Error(Form("%s(...) %s[%d]", function, file, line), "%s", msg.Data());
         comm->Abort(errcode, kTRUE);
      }

      template<> void TErrorHandler::TraceBack(const Char_t *class_name, const Char_t *function, const Char_t *file, Int_t line, Int_t errcode, const Char_t *msg);

      template<> void TErrorHandler::TraceBack(const TGroup *group, const Char_t *function, const Char_t *file, Int_t line, Int_t errcode, const Char_t *msg);

   }
}
#endif
