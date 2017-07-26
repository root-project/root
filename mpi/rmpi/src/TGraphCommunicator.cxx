#include <Mpi/TGraphCommunicator.h>
using namespace ROOT::Mpi;

//______________________________________________________________________________
/**
 * Default constructor for graph-communicator that is a null communicator
 */

TGraphCommunicator::TGraphCommunicator() : TIntraCommunicator()
{
}

//______________________________________________________________________________
/**
 * Copy constructor for inter-communicator
 * \param comm other TGraphCommunicator object
 */

TGraphCommunicator::TGraphCommunicator(const TGraphCommunicator &comm) : TIntraCommunicator(comm)
{
}

//______________________________________________________________________________
TGraphCommunicator::TGraphCommunicator(const MPI_Comm &comm) : TIntraCommunicator(comm)
{
   Int_t status = 0;
   if (TEnvironment::IsInitialized() && (comm != MPI_COMM_NULL)) {
      MPI_Topo_test(comm, &status);
      if (status == MPI_GRAPH)
         fComm = comm;
      else
         fComm = MPI_COMM_NULL;
   } else {
      fComm = comm;
   }
}

//______________________________________________________________________________
/**
 * Duplicates an existing communicator with all its cached information.
 * Duplicates  the  existing communicator comm with associated key values.
 * For each key value, the respective copy callback function determines the
 * attribute value associated with this key in the new communicator; one
 * particular action that a copy callback may take is to delete the attribute
 * from the  new communicator.
 * \return Returns a new communicator with the same group, any copied cached
 * information, but a new context (see Section 5.7.1 of the MPI-1 Standard,
 * "Functionality").
 */

TGraphCommunicator *TGraphCommunicator::Dup() const
{
   MPI_Comm dupcomm;
   ROOT_MPI_CHECK_CALL(MPI_Comm_dup, (fComm, &dupcomm), this);
   ROOT_MPI_CHECK_COMM(dupcomm, this);
   auto fDupComm = new TGraphCommunicator(dupcomm);
   return fDupComm;
}
