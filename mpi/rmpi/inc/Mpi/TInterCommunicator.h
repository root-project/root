// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TInterCommunicator
#define ROOT_Mpi_TInterCommunicator

#include<Mpi/TCommunicator.h>

namespace ROOT {

   namespace Mpi {
   /**
    * \class TInterCommunicator
    * The processes communicate each other through communicators, the
    * ROOT::Mpi::TIntraCommunicator is for proccesses that have communication in
    the sigle group and The class ROOT::Mpi::TInterCommunicator is communication
    between two groups of processes.
    * Both classes are derived from an abstract base ROOT::Mpi::TCommunicator.
    * This class allows to do communication in a multiple different contexts
    ROOT::Mpi::TIntraCommunicator
    \see TGroup TIntraCommunicator TCommunicator
    \ingroup Mpi
 */

   class TInterCommunicator : public TCommunicator {
   public:
     TInterCommunicator();

     TInterCommunicator(const TInterCommunicator &data);

     TInterCommunicator(const MPI_Comm &comm);

     TInterCommunicator &operator=(const TInterCommunicator &comm) {
       fComm = comm.fComm;
       return *this;
     }

     TInterCommunicator &operator=(const MPI_Comm &comm) {
       fComm = comm;
       return *this;
     }

     inline operator MPI_Comm() const { return fComm; }

     virtual Int_t GetRemoteSize() const;

     virtual TGroup GetRemoteGroup() const;

     TInterCommunicator Dup() const;

     virtual TInterCommunicator &Clone() const;

     virtual TIntraCommunicator Merge(Int_t high);

     virtual TInterCommunicator Create(const TGroup &group) const;

     virtual TInterCommunicator Split(Int_t color, Int_t key) const;

     ClassDef(TInterCommunicator, 3)
      };
   }
}

#endif
