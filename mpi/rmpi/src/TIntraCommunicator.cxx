#include<Mpi/TIntraCommunicator.h>
#include<Mpi/TInterCommunicator.h>
#include<Mpi/TInfo.h>
#include<Mpi/TPort.h>

using namespace ROOT::Mpi;

//______________________________________________________________________________
TIntraCommunicator::TIntraCommunicator(const MPI_Comm &comm): TCommunicator(comm)
{
}

//______________________________________________________________________________
TIntraCommunicator::TIntraCommunicator(const TIntraCommunicator &comm): TCommunicator(comm.fComm)
{
}

//______________________________________________________________________________
TIntraCommunicator TIntraCommunicator::Dup() const
{
   MPI_Comm dupcomm;
   MPI_Comm_dup(fComm, &dupcomm);
   return  dupcomm;
}

//______________________________________________________________________________
TIntraCommunicator &TIntraCommunicator::Clone() const
{
   MPI_Comm dupcomm;
   MPI_Comm_dup(fComm, &dupcomm);
   TIntraCommunicator *icomm = new TIntraCommunicator(dupcomm);
   return  *icomm;
}

//______________________________________________________________________________
TIntraCommunicator TIntraCommunicator::Create(const TGroup &group) const
{
   MPI_Comm ncomm;
   ROOT_MPI_CHECK_GROUP(group, this);
   MPI_Comm_create(fComm, group, &ncomm);
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return  ncomm;
}

//______________________________________________________________________________
TIntraCommunicator TIntraCommunicator::Split(Int_t color, Int_t key) const
{
   MPI_Comm ncomm;
   MPI_Comm_split(fComm, color, key, &ncomm);
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return ncomm;
}

//______________________________________________________________________________
TInterCommunicator TIntraCommunicator::CreateIntercomm(Int_t local_leader, const TIntraCommunicator &peer_comm, Int_t remote_leader, Int_t tag) const
{
   MPI_Comm ncomm;
   MPI_Intercomm_create(fComm, local_leader, peer_comm.fComm, remote_leader, tag, &ncomm);
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return ncomm;
}

//______________________________________________________________________________
TInterCommunicator TIntraCommunicator::Accept(const TPort &port, Int_t root) const
{
   MPI_Comm ncomm;
   MPI_Comm_accept(port.GetPortName(), port.GetInfo(), root, fComm, &ncomm);
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return ncomm;
}

//______________________________________________________________________________
TInterCommunicator TIntraCommunicator::Connect(const TPort &port, Int_t root) const
{
   MPI_Comm ncomm;
   MPI_Comm_connect(port.GetPortName(), port.GetInfo(), root, fComm, &ncomm);
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return ncomm;
}

//______________________________________________________________________________
TInterCommunicator TIntraCommunicator::Spawn(const Char_t *command, const Char_t *argv[], Int_t maxprocs, const TInfo &info, Int_t root) const
{
   MPI_Comm ncomm;
   MPI_Comm_spawn(command, const_cast<Char_t **>(argv), maxprocs, info, root, fComm, &ncomm, (Int_t *)MPI_ERRCODES_IGNORE);
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return ncomm;
}

//______________________________________________________________________________
TInterCommunicator TIntraCommunicator::Spawn(const Char_t *command, const Char_t *argv[], Int_t maxprocs, const TInfo &info,
      Int_t root, Int_t array_of_errcodes[]) const
{
   MPI_Comm ncomm;
   MPI_Comm_spawn(command, const_cast<Char_t **>(argv), maxprocs, info, root, fComm, &ncomm, array_of_errcodes);
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return ncomm;
}

//______________________________________________________________________________
TInterCommunicator TIntraCommunicator::SpawnMultiple(Int_t count, const Char_t *array_of_commands[], const Char_t **array_of_argv[],
      const Int_t array_of_maxprocs[], const TInfo array_of_info[], Int_t root)
{
   MPI_Comm ncomm;

   MPI_Info *array_of_mpi_info = new MPI_Info[count];
   for (Int_t i = 0; i < count; i++) {
      array_of_mpi_info[i] = array_of_info[i];
   }

   MPI_Comm_spawn_multiple(count, const_cast<Char_t **>(array_of_commands),
                           const_cast<Char_t ** *>(array_of_argv),
                           const_cast<Int_t *>(array_of_maxprocs),
                           array_of_mpi_info, root,
                           fComm, &ncomm, (Int_t *)MPI_ERRCODES_IGNORE);
   delete[] array_of_mpi_info;
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return ncomm;
}

//______________________________________________________________________________
TInterCommunicator TIntraCommunicator::SpawnMultiple(Int_t count, const Char_t *array_of_commands[], const Char_t **array_of_argv[],
      const Int_t array_of_maxprocs[], const TInfo array_of_info[], Int_t root,
      Int_t array_of_errcodes[])
{
   MPI_Comm ncomm;

   MPI_Info *array_of_mpi_info = new MPI_Info[count];
   for (Int_t i = 0; i < count; i++) {
      array_of_mpi_info[i] = array_of_info[i];
   }

   MPI_Comm_spawn_multiple(count, const_cast<Char_t **>(array_of_commands),
                           const_cast<Char_t ** *>(array_of_argv),
                           const_cast<Int_t *>(array_of_maxprocs),
                           array_of_mpi_info, root,
                           fComm, &ncomm, array_of_errcodes);
   delete[] array_of_mpi_info;
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return ncomm;
}

