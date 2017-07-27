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

//______________________________________________________________________________
/**
 * Retrieves graph topology information associated with a communicator.
 *
 * Functions ROOT::Mpi::TGraphCommunicator::GetDims and ROOT::Mpi::TGraphCommunicator::GetTopo retrieve the
 * graph-topology information that was associated with a communicator by ROOT::Mpi::TIntraCommunicator::CreateGraphcomm.
 *
 * The information provided by ROOT::Mpi::TGraphCommunicator::GetDims can be used to dimension the vectors index and
 * edges correctly for a call to ROOT::Mpi::TGraphCommunicator::GetTopo.
 *
 * \param nnodes Number of nodes in graph (output integer).
 * \param nedges Number of edges in graph (output integer).
 */
void TGraphCommunicator::GetDims(Int_t nnodes[], Int_t nedges[]) const
{
   ROOT_MPI_CHECK_CALL(MPI_Graphdims_get, (fComm, nnodes, nedges), this);
}

//______________________________________________________________________________
/**
 * Retrieves graph topology information associated with a communicator.
 *
 * Functions ROOT::Mpi::TGraphCommunicator::GetDims and ROOT::Mpi::TGraphCommunicator::GetTopo retrieve the
 * graph-topology information that was associated with a communicator by ROOT::Mpi::TIntraCommunicator::CreateGraphcomm.
 *
 * The information provided by ROOT::Mpi::TGraphCommunicator::GetDims can be used to dimension the vectors index and
 * edges correctly for a call to ROOT::Mpi::TGraphCommunicator::GetTopo.
 *
 * \param maxindex Length of vector index in the calling program (integer).
 * \param maxedges Length of vector edges in the calling program (integer).
 * \param index Array of integers containing the graph structure (for details see the definition of
 * ROOT::Mpi::TIntraCommunicator::CreateGraphcomm)(output).
 * \param edges Array of integers containing the graph structure.(output)
 */
void TGraphCommunicator::GetTopo(Int_t maxindex, Int_t maxedges, Int_t index[], Int_t edges[]) const
{
   ROOT_MPI_CHECK_CALL(MPI_Graph_get, (fComm, maxindex, maxedges, index, edges), this);
}

//______________________________________________________________________________
/**
 * Returns the number of neighbors of a node associated with a graph topology.
 *
 * ROOT::Mpi::TGraphCommunicator::GetNeighborsCount and ROOT::Mpi::TGraphCommunicator::GetNeighbors provide adjacency
 * information for a general, graph topology. ROOT::Mpi::TGraphCommunicator::GetNeighborsCount returns the number of
 * neighbors for the process signified by rank.
 *
 * \param rank Rank of process in group of comm (integer).
 * \return Number of neighbors of specified process (integer).
 */
Int_t TGraphCommunicator::GetNeighborsCount(Int_t rank) const
{
   Int_t nneighbors;
   ROOT_MPI_CHECK_CALL(MPI_Graph_neighbors_count, (fComm, rank, &nneighbors), this);
   return nneighbors;
}

//______________________________________________________________________________
/**
 * Returns the neighbors of a node associated with a graph topology.
 *
 * Example:  Suppose that comm is a communicator with a shuffle-exchange topology.
 * The group has 2n members. Each process is labeled by a(1), ..., a(n) with a(i)
 * E{0,1}, and has three neighbors: exchange (a(1), ..., a(n) = a(1), ..., a(n-1), a(n) (a = 1 - a), shuffle (a(1), ...,
 * a(n))  =  a(2), ...,  a(n),  a(1),  and unshuffle (a(1), ..., a(n)) = a(n), a(1), ..., a(n-1). The graph adjacency
 * list is illustrated below for n=3.
 *
 *                      exchange       shuffle        unshuffle
 *          node       neighbors(1)   neighbors(2)   neighbors(3)
 *          0(000)         1              0              0
 *          1(001)         0              2              4
 *          2(010)         3              4              1
 *          3(011)         2              6              5
 *          4(100)         5              1              2
 *          5(101)         4              3              6
 *          6(110)         7              5              3
 *          7(111)         6              7              7
 *
 * \param rank Rank of process in group of comm (input integer).
 * \param maxneighbors Size of array neighbors (input integer).
 * \param neighbors Ranks of processes that are neighbors to specified process (output array of integers).
 */
void TGraphCommunicator::GetNeighbors(Int_t rank, Int_t maxneighbors, Int_t neighbors[]) const
{
   ROOT_MPI_CHECK_CALL(MPI_Graph_neighbors, (fComm, rank, maxneighbors, neighbors), this);
}

//______________________________________________________________________________
/**
 * Maps process to graph topology information.
 *
 * ROOT::Mpi::TCartCommunicator::Map  and  ROOT::Mpi::TGraphCommunicator::Map can be used to implement all other
 * topology functions. In general they will not be called by the user directly, unless he or she is creating additional
 * virtual topology capability other than that provided by MPI.
 *
 * \param nnodes Number of graph nodes (integer).
 * \param index Integer array specifying the graph structure, see  ROOT::Mpi::TIntraCommunicator::CreateGraphcomm.
 * \param edges Integer array specifying the graph structure.
 * \return Reordered rank of the calling process; ROOT::Mpi::UNDEFINED if the calling process does not belong to graph
 * (integer).
 */
Int_t TGraphCommunicator::Map(Int_t nnodes, const Int_t index[], const Int_t edges[]) const
{
   Int_t nrank;
   ROOT_MPI_CHECK_CALL(MPI_Graph_map, (fComm, nnodes, const_cast<Int_t *>(index), const_cast<Int_t *>(edges), &nrank),
                       this);
   return nrank;
}
