// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TIntraCommunicator
#define ROOT_Mpi_TIntraCommunicator

#include<Mpi/TCommunicator.h>

namespace ROOT {

   namespace Mpi {
      class TInfo;
      class TInterCommunicator;

      /**
       * \class TIntraCommunicator
       * The processes communicate each other through communicators, the \class TIntraCommunicator is for
       * proccesses that have communication in the sigle group and The class \class TInterCommunicator is communication between two groups of processes. Both classes are derived from an abstract base \class TCommunicator.
       * \see TGroup
       * \ingroup Mpi
       */

      class TIntraCommunicator: public TCommunicator {
      public:

         TIntraCommunicator(): TCommunicator() {}

         TIntraCommunicator(const TIntraCommunicator &comm);

         TIntraCommunicator(const MPI_Comm &comm);

         // assignment
         TIntraCommunicator &operator=(const TIntraCommunicator &comm)
         {
            fComm = comm.fComm;
            return *this;
         }

         TIntraCommunicator &operator=(const MPI_Comm &comm)
         {
            fComm = comm;
            return *this;
         }

         inline operator MPI_Comm() const
         {
            return fComm;
         }

         TIntraCommunicator Dup() const;

         virtual TIntraCommunicator &Clone() const;

         virtual TIntraCommunicator Create(const TGroup &group) const;

         virtual TIntraCommunicator Split(Int_t color, Int_t key) const;

         virtual TInterCommunicator CreateIntercomm(Int_t local_leader, const TIntraCommunicator &peer_comm, Int_t remote_leader, Int_t tag) const;

         //
         // Process Creation and Management
         //

         virtual TInterCommunicator Accept(const Char_t *port_name, const TInfo &info, Int_t root)const;

         virtual TInterCommunicator Connect(const Char_t *port_name, const TInfo &info, Int_t root)const;

         virtual TInterCommunicator Spawn(const Char_t *command, const Char_t *argv[], Int_t maxprocs, const TInfo &info, Int_t root) const;

         virtual TInterCommunicator Spawn(const Char_t *command, const Char_t *argv[], Int_t maxprocs, const TInfo &info,
                                          Int_t root, Int_t array_of_errcodes[]) const;

         virtual TInterCommunicator SpawnMultiple(Int_t count, const Char_t *array_of_commands[], const Char_t **array_of_argv[],
               const Int_t array_of_maxprocs[], const TInfo array_of_info[], Int_t root);

         virtual TInterCommunicator SpawnMultiple(Int_t count, const Char_t *array_of_commands[], const Char_t **array_of_argv[],
               const Int_t array_of_maxprocs[], const TInfo array_of_info[], Int_t root,
               Int_t array_of_errcodes[]);

         ClassDef(TIntraCommunicator, 3) //
      };
   }
}


#endif
