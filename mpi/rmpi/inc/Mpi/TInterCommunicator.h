// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TInterCommunicator
#define ROOT_Mpi_TInterCommunicator

#ifndef ROOT_Mpi_TCommunicator
#include<Mpi/TCommunicator.h>
#endif

namespace ROOT {

   namespace Mpi {
      class TInterCommunicator;
      class TInterCommunicator: public TCommunicator {
      public:

         TInterCommunicator();

         TInterCommunicator(const TInterCommunicator &data);

         TInterCommunicator(const MPI_Comm &comm);

         TInterCommunicator &operator=(const TInterCommunicator &comm)
         {
            fComm = comm.fComm;
            return *this;
         }

         TInterCommunicator &operator=(const MPI_Comm &comm)
         {
            fComm = comm;
            return *this;
         }

         virtual Int_t GetRemoteSize() const;

         virtual TGroup GetRemoteGroup() const;

         TInterCommunicator Dup() const;

         virtual TInterCommunicator &Clone() const;

         virtual TIntraCommunicator Merge(Bool_t high);

         virtual TInterCommunicator Create(const TGroup &group) const;

         virtual TInterCommunicator Split(Int_t color, Int_t key) const;

         ClassDef(TInterCommunicator, 3)
      };
   }
}

#endif
