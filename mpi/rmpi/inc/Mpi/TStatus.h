// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TStatus
#define ROOT_Mpi_TStatus

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#include<TObject.h>
#include<mpi.h>

namespace ROOT {
   namespace Mpi {
      /**
      \class TComm
         Class to handle status in the ROOTMpi calls.
         \ingroup Mpi
       */

      class TStatus: public TObject {
         friend class TCommunicator;
         friend class TRequest;

      protected:
         MPI::Status fStatus;       //internal MPI::Status object
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

         TStatus &operator=(const TStatus &obj)
         {
            fStatus = obj.fStatus;
            return *this;
         }

         TStatus &operator= (const MPI::Status &obj)
         {
            fStatus = obj;
            return *this;
         }

         operator MPI_Status() const
         {
            return fStatus;
         }

         operator MPI::Status() const
         {
            return fStatus;
         }

         /**
         Method to get if the current process was cancelled
              \return integer with the status value
              */
         virtual Bool_t IsCancelled() const;

         /**
         Method to get the source of the process
              \return integer with the rank or process id value
              */
         virtual Int_t GetSource() const;

         /**
         Method to set the source to the status message
              \param source integer with the rank or process id value
              */
         virtual void SetSource(Int_t source);

         /**
         Method to get the tag id of the process
              \return integer with the tag id
              */
         virtual Int_t GetTag() const;

         /**
         Method to set the tag id to the status of the process
              \param tag integer with the tag id
              */
         virtual void SetTag(Int_t tag);

         /**
         Method to get the error id of the process
              \return integer with the error id
              */
         virtual Int_t GetError() const;

         /**
         Method to set the error id to the status message
              \param error integer with the error id
              */
         virtual void SetError(Int_t error);

         /**
         Method to set the cancelled flag to the status message
              \param flag boolean to set the cancelled flag to the message
              */
         virtual void SetCancelled(Bool_t flag);


         ClassDef(TStatus, 1)
      };
   }
}
#endif
