// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TIntraCommunicator
#define ROOT_Mpi_TIntraCommunicator

#ifndef ROOT_Mpi_TCommunicator
#include<Mpi/TCommunicator.h>
#endif

namespace ROOT {

   namespace Mpi {
      class TInfo;

      /**
       * \class TIntraCommunicator
       * The processes communicate each other through communicators, the \class TIntraCommunicator is for
       * proccesses that have communication in the sigle group and The class \class TInterCommunicator is communication between two groups of processes. Both classes are derived from an abstract base \class TCommunicator.
       * \see TGroup
       * \ingroup Mpi
       */

      class TIntraCommunicator: public TCommunicator {
      public:

         TIntraCommunicator():TCommunicator(){}
          
         TIntraCommunicator(const TIntraCommunicator &comm);

         TIntraCommunicator(const MPI_Comm &comm);

         // assignment
         TIntraCommunicator &operator=(const TIntraCommunicator &comm)
         {
            fComm = comm.fComm;
            return *this;
         }


         TIntraCommunicator &operator=(const TNullCommunicator &comm)
         {
            fComm = comm;
            return *this;
         }

         TIntraCommunicator &operator=(const MPI_Comm &comm)
         {
            fComm = comm;
            return *this;
         }

         TIntraCommunicator Dup() const;

         virtual TIntraCommunicator &Clone() const;

         virtual TIntraCommunicator Create(const TGroup &group) const;

         virtual TIntraCommunicator Split(int color, int key) const;

         ClassDef(TIntraCommunicator, 3) //
      };
   }
}


#endif
