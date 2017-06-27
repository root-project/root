// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TStatus
#define ROOT_Mpi_TStatus

#include<Mpi/Globals.h>
#include<Mpi/TErrorHandler.h>

namespace ROOT {
   namespace Mpi {
   /**
   \class TStatus
      Class to handle status in the ROOTMpi calls.
      \ingroup Mpi
    */

   class TStatus : public TObject {
     friend class TCommunicator;
     friend class TRequest;

   protected:
     MPI_Status fStatus; // internal MP_Status object
     Int_t fMsgSize;     // size of the message in the serialization
   public:
     /**
     Default constructor
          */
     TStatus();
     /**
     Copy constructor
      \param obj other TStatus object
      */
     TStatus(const TStatus &obj);

     virtual ~TStatus() {}

     TStatus &operator=(const TStatus &obj) {
       fStatus = obj.fStatus;
       return *this;
     }

     TStatus &operator=(const MPI_Status &obj) {
       fStatus = obj;
       return *this;
     }

     operator MPI_Status() const { return fStatus; }

     virtual Bool_t IsCancelled() const;

     virtual Int_t GetSource() const;

     virtual void SetSource(Int_t source);

     virtual Int_t GetTag() const;

     virtual void SetTag(Int_t tag);

     virtual Int_t GetError() const;

     virtual void SetError(Int_t error);

     virtual void SetCancelled(Bool_t flag);

     Int_t GetMsgSize() const;

     void SetMsgSize(Int_t size);

     ClassDef(TStatus, 1)
      };


   }
}
#endif
