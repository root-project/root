#include <Mpi/TIntraCommunicator.h>
#include <Mpi/TInterCommunicator.h>
using namespace ROOT::Mpi;

//______________________________________________________________________________
/**
 * Default constructor for inter-communicator that is a null communicator
 */

TInterCommunicator::TInterCommunicator() : TCommunicator()
{
}

//______________________________________________________________________________
/**
 * Copy constructor for inter-communicator
 * \param comm other TInterCommunicator object
 */

TInterCommunicator::TInterCommunicator(const TInterCommunicator &comm) : TCommunicator(comm)
{
}

//______________________________________________________________________________
TInterCommunicator::TInterCommunicator(const MPI_Comm &comm) : TCommunicator(comm)
{
}

//______________________________________________________________________________
/**
 * This  function  creates  an  intracommunicator from the union of the two
 * groups that are associated with intercomm.
 * All processes should provide the same high value within each of the two
 * groups. If processes in one group provide the value high = false and
 * processes in the other group provide the value high =  true,then the union
 * orders the "low" group before the "high" group. If all processes provide the
 * same high argument, then the order of the union is arbitrary. This call is
 * blocking and collective within the union of the two groups.
 * \param high Used to order the groups of the two intracommunicators within
 * comm when creating the new communicator (type indicator).
 * \return Created intracommunicator (type indicator).
 */
TIntraCommunicator TInterCommunicator::Merge(Int_t high)
{
   MPI_Comm ncomm;
   ROOT_MPI_CHECK_CALL(MPI_Intercomm_merge, (fComm, high, &ncomm), this);
   return ncomm;
}

//______________________________________________________________________________
/**
 * Determines the size of the remote group associated with an intercommunicator.
 * The  intercommunicator accessors (ROOT::Mpi::TCommunicator::IsInter,
 * ROOT::Mpi::TInterCommunicator::GetRemoteSize,ROOT::Mpi::TInterCommunicator::GetRemoteGroup)
 * are all local operations.
 * \return Number of processes in the remote group of comm (integer).
 */
Int_t TInterCommunicator::GetRemoteSize() const
{
   Int_t size;
   ROOT_MPI_CHECK_CALL(MPI_Comm_remote_size, (fComm, &size), this);
   return size;
}

//______________________________________________________________________________
/**
 * Accesses the remote group associated with an intercommunicator.
 * The  intercommunicator accessors (ROOT::Mpi::TCommunicator::IsInter,
 * ROOT::Mpi::TInterCommunicator::GetRemoteSize,ROOT::Mpi::TInterCommunicator::GetRemoteGroup)
 * are all local operations.
 * \return  Remote group of communicator.
 */
TGroup TInterCommunicator::GetRemoteGroup() const
{
   MPI_Group group;
   ROOT_MPI_CHECK_CALL(MPI_Comm_remote_group, (fComm, &group), this);
   return group;
}

//______________________________________________________________________________
/*
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

TInterCommunicator *TInterCommunicator::Dup() const
{
   MPI_Comm dupcomm;
   ROOT_MPI_CHECK_CALL(MPI_Comm_dup, (fComm, &dupcomm), this);
   ROOT_MPI_CHECK_COMM(dupcomm, this);
   auto fDupComm = new TInterCommunicator(dupcomm);
   return fDupComm;
}

//______________________________________________________________________________
/**
 * This  method creates a new communicator with communication group defined by
 * group and a new context.
 * The function sets the returned communicator to a new communicator that spans
 * all the processes that are in the group.  It sets the returned communicator
 * to ROOT::Mpi::COMM_NULL for processes that are not in the group.
 * Each process must call with a group argument that is a subgroup of the group
 * associated with comm; this could be ROOT::Mpi::GROUP_EMPTY. The  processes
 * may  specify different  values  for  the  group  argument.
 * If a process calls with a non-empty group, then all processes in that group
 * must call the function with the same group as argument, that is: the same
 * processes in the same order. Otherwise the call is erroneous.
 * \param group Group, which is a subset of the group of comm (handle).
 * \return New communicator (handle).
 */
TInterCommunicator TInterCommunicator::Create(const TGroup &group) const
{
   MPI_Comm ncomm;
   ROOT_MPI_CHECK_GROUP(group, this);
   ROOT_MPI_CHECK_CALL(MPI_Comm_create, (fComm, group, &ncomm), this);
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return ncomm;
}

//______________________________________________________________________________
/**
 * This function partitions the group associated with comm into disjoint
 * subgroups, one for each value of color.
 * Each subgroup contains all processes of the same color.
 * Within each subgroup, the processes are ranked in the order defined by the
 * value of the argument key, with ties broken according to their rank  in  the
 * old  group.
 * A new communicator is created for each subgroup and returned in newcomm.
 * A process may supply the color value ROOT::Mpi::UNDEFINED, in which case
 * newcomm returns ROOT::Mpi::COMM_NULL. This is a collective call, but each
 * process is permitted to provide different values for color and key.
 *
 * When you call ROOT::Mpi::TInterCommunicator::Split on an inter-communicator,
 * the processes on the left with the same color as those on the right combine
 * to create a new  inter-communicator.
 * The  key argument describes the relative rank of processes on each side of
 * the inter-communicator.
 * The function returns ROOT::Mpi::COMM_NULL for  those colors that are
 * specified on only one side of the inter-communicator, or for those that
 * specify ROOT::Mpi::UNDEFINED as the color.
 *
 * A call to ROOT::Mpi::TInterCommunicator::Create is equivalent to a call to
 * ROOT::Mpi::TInterCommunicator::Split( color, key), where all members of group
 * provide color =  0  and  key = rank in group, and all processes that are not
 * members of group provide color = ROOT::Mpi::UNDEFINED.
 * The function ROOT::Mpi::TInterCommunicator::Split allows more general
 * partitioning of a group into one or more subgroups with optional reordering.
 * The value of color must be nonnegative or ROOT::Mpi::UNDEFINED.
 * \param color Control of subset assignment (nonnegative integer).
 * \param key Control of rank assignment (integer).
 * \return New communicator (handle).
 */

TInterCommunicator TInterCommunicator::Split(Int_t color, Int_t key) const
{
   MPI_Comm ncomm;
   ROOT_MPI_CHECK_CALL(MPI_Comm_split, (fComm, color, key, &ncomm), this);
   ROOT_MPI_CHECK_COMM(ncomm, this);
   return ncomm;
}
