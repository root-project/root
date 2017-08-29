#include <Mpi/TCartCommunicator.h>
using namespace ROOT::Mpi;

//______________________________________________________________________________
/**
 * Default constructor for cartesian-communicator that is a null communicator
 */

TCartCommunicator::TCartCommunicator() : TIntraCommunicator()
{
}

//______________________________________________________________________________
/**
 * Copy constructor for inter-communicator
 * \param comm other TCartCommunicator object
 */

TCartCommunicator::TCartCommunicator(const TCartCommunicator &comm) : TIntraCommunicator(comm)
{
}

//______________________________________________________________________________
TCartCommunicator::TCartCommunicator(const MPI_Comm &comm) : TIntraCommunicator(comm)
{
   Int_t status = 0;
   if (TEnvironment::IsInitialized() && (comm != MPI_COMM_NULL)) {
      MPI_Topo_test(comm, &status);
      if (status == MPI_CART)
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

TCartCommunicator *TCartCommunicator::Dup() const
{
   MPI_Comm dupcomm;
   ROOT_MPI_CHECK_CALL(MPI_Comm_dup, (fComm, &dupcomm), this);
   ROOT_MPI_CHECK_COMM(dupcomm, this);
   auto fDupComm = new TCartCommunicator(dupcomm);
   return fDupComm;
}

//______________________________________________________________________________
/**
 * Retrieves Cartesian topology information associated with a communicator.
 * \return returns the number of dimensions of the Cartesian structure.
 */
Int_t TCartCommunicator::GetDim() const
{
   Int_t ndims;
   ROOT_MPI_CHECK_CALL(MPI_Cartdim_get, (fComm, &ndims), this);
   return ndims;
}

//______________________________________________________________________________
/**
 * Retrieves Cartesian topology information associated with a communicator.
 * The functions ROOT::Mpi::TCartCommunicator::GetDim and ROOT::Mpi::TCartCommunicator::GetTop return the Cartesian
 * topology information that was associated with a communicator.
 * \param maxdims  Length of vectors dims, periods, and coords in the calling program (input integer).
 * \param dims Number of processes for each Cartesian dimension (output array of integers).
 * \param periods Periodicity (true/false) for each Cartesian dimension (output array of logicals).
 * \param coords Coordinates of calling process in Cartesian structure (output array of integers).
 */
void TCartCommunicator::GetTopo(Int_t maxdims, Int_t dims[], Bool_t periods[], Int_t coords[]) const
{
   Int_t *int_periods = new Int_t[maxdims];
   Int_t i;
   for (i = 0; i < maxdims; i++) {
      int_periods[i] = (Int_t)periods[i];
   }
   ROOT_MPI_CHECK_CALL(MPI_Cart_get, (fComm, maxdims, dims, int_periods, coords), this);
   for (i = 0; i < maxdims; i++) {
      periods[i] = Bool_t(int_periods[i]);
   }
   delete[] int_periods;
}

//______________________________________________________________________________
/**
 * Determines process rank in communicator given Cartesian location.
 * For  a  process group with Cartesian structure, the function ROOT::Mpi::TCartCommunicator::GetCartRank translates the
 * logical process coordinates to process ranks as they are used by the point-to-point routines.  For dimension i with
 * periods(i) = true, if the coordinate, coords(i), is out of range, that is,  coords(i)  <  0  or   coords(i)  >=
 * dims(i), it is shifted back to the interval  0 =< coords(i) < dims(i) automatically. Out-of-range coordinates are
 * erroneous for nonperiodic dimensions.
 * \param coords Integer array (of size ndims, which was defined by MPI_Cart_create call) specifying the Cartesian
 * coordinates of a process.
 * \return Rank of specified process (integer).
 */
Int_t TCartCommunicator::GetCartRank(const Int_t coords[]) const
{
   Int_t rank;
   ROOT_MPI_CHECK_CALL(MPI_Cart_rank, (fComm, const_cast<Int_t *>(coords), &rank), this);
   return rank;
}

//______________________________________________________________________________
/**
 * Determines process rank in communicator given Cartesian location.
 * For  a  process group with Cartesian structure, the function ROOT::Mpi::TCartCommunicator::GetCartRank translates the
 * logical process coordinates to process ranks as they are used by the point-to-point routines.  For dimension i with
 * periods(i) = true, if the coordinate, coords(i), is out of range, that is,  coords(i)  <  0  or   coords(i)  >=
 * dims(i), it is shifted back to the interval  0 =< coords(i) < dims(i) automatically. Out-of-range coordinates are
 * erroneous for nonperiodic dimensions.
 * \param coords Integer array (of size ndims, which was defined by MPI_Cart_create call) specifying the Cartesian
 * coordinates of a process.
 * \return Rank of specified process (integer).
 */
Int_t TCartCommunicator::GetCartRank(const std::vector<Int_t> coords) const
{
   Int_t rank;
   ROOT_MPI_CHECK_CALL(MPI_Cart_rank, (fComm, const_cast<Int_t *>(&coords[0]), &rank), this);
   return rank;
}

//______________________________________________________________________________
/**
 * Determines process coords in Cartesian topology given rank in group.
 * provies a mapping of ranks to Cartesian coordinates.
 * \param rank Rank of a process within group of comm (input integer).
 * \param maxdims Length of vector coords in the calling program (input integer).
 * \param coords integer array (of size ndims,which was defined by MPI_Cart_create call) containing the Cartesian
 * coordinates of specified process (output integers).
 */
void TCartCommunicator::GetCoords(Int_t rank, Int_t maxdims, Int_t coords[]) const
{
   ROOT_MPI_CHECK_CALL(MPI_Cart_coords, (fComm, rank, maxdims, coords), this);
}

//______________________________________________________________________________
/**
 * Returns the shifted source and destination ranks, given a shift direction and amount.
 *
 * The direction argument indicates the dimension of the shift, i.e., the coordinate whose value is modified by the
 * shift. The coordinates are numbered from 0 to ndims-1, where ndims is the number of dimensions.
 *
 * Note:  The direction argument is in the range [0, n-1] for an n-dimensional Cartesian mesh.
 * Depending  on  the periodicity of the Cartesian group in the specified coordinate direction,
 * ROOT::Mpi::TCartCommunicator::Shift provides the identifiers for a circular or an end-off shift. In the case of an
 * end-off shift, the value ROOT::Mpi::PROC_NULL may be returned in rank_source or rank_dest, indicating that the source
 * or the destination for the shift is out of range.
 * \param direction Coordinate dimension of shift (integer).
 * \param disp Displacement ( > 0: upward shift, < 0: downward shift) (integer).
 * \param rank_source Rank of source process (output integer).
 * \param rank_dest Rank of destination process (output integer).
 */
void TCartCommunicator::Shift(Int_t direction, Int_t disp, Int_t &rank_source, Int_t &rank_dest) const
{
   ROOT_MPI_CHECK_CALL(MPI_Cart_shift, (fComm, direction, disp, &rank_source, &rank_dest), this);
}

//______________________________________________________________________________
/**
 * Partitions a communicator into subgroups, which form lower-dimensional Cartesian subgrids.
 *
 * If  a Cartesian topology has been created with ROOT::Mpi::TIntraCommunicator::CreateCartcomm, the function
 * ROOT::Mpi::TCartCommunicator::Sub can be used to partition the communicator group into subgroups that form
 * lower-dimensional Cartesian subgrids, and to build for each subgroup a communicator with the associated subgrid
 * Cartesian  topology.  (This  function  is closely related to ROOT::Mpi::TIntraCommunicator::Split.)
 * \param remain_dims The ith entry of remain_dims specifies whether the ith dimension is kept in the subgrid (true) or
 * is dropped (false) (logical vector).
 * \return Communicator containing the subgrid that includes the calling process (handle).
 */
TCartCommunicator TCartCommunicator::Sub(const Bool_t remain_dims[]) const
{
   return Sub(std::vector<Bool_t>(GetDim(), remain_dims));
}

//______________________________________________________________________________
/**
 * Partitions a communicator into subgroups, which form lower-dimensional Cartesian subgrids.
 *
 * If  a Cartesian topology has been created with ROOT::Mpi::TIntraCommunicator::CreateCartcomm, the function
 * ROOT::Mpi::TCartCommunicator::Sub can be used to partition the communicator group into subgroups that form
 * lower-dimensional Cartesian subgrids, and to build for each subgroup a communicator with the associated subgrid
 * Cartesian  topology.  (This  function  is closely related to ROOT::Mpi::TIntraCommunicator::Split.)
 * \param remain_dims The ith entry of remain_dims specifies whether the ith dimension is kept in the subgrid (true) or
 * is dropped (false) (logical vector).
 * \return Communicator containing the subgrid that includes the calling process (handle).
 */
TCartCommunicator TCartCommunicator::Sub(const std::vector<Bool_t> remain_dims) const
{
   Int_t ndims = GetDim();
   Int_t *int_remain_dims = new Int_t[ndims];
   for (Int_t i = 0; i < ndims; i++) {
      int_remain_dims[i] = (Int_t)remain_dims[i];
   }
   MPI_Comm newcomm;
   ROOT_MPI_CHECK_CALL(MPI_Cart_sub, (fComm, int_remain_dims, &newcomm), this);
   delete[] int_remain_dims;
   return TCartCommunicator(newcomm);
}

//______________________________________________________________________________
/**
 * Maps process to Cartesian topology information.
 * ROOT::Mpi::TCartCommunicator::Map can be used to implement all other topology functions. In general they will not be
 * called by the user directly, unless he or she is creating additional virtual topology capability other than that
 * provided by MPI.
 *
 * ROOT::Mpi::TCartCommunicator::Map computes an "optimal" placement for the calling process on the physical machine. A
 * possible implementation of this function is to  always  return the rank of the calling process, that is, not to
 * perform any reordering.
 * \param ndims Number of dimensions of Cartesian structure (integer).
 * \param dims Integer array of size ndims specifying the number of processes in each coordinate direction.
 * \param periods Logical array of size ndims specifying the periodicity specification in each coordinate direction.
 * \return Reordered rank of the calling process; ROOT::Mpi::UNDEFINED if calling process does not belong to grid
 * (integer).
 */
Int_t TCartCommunicator::Map(Int_t ndims, const Int_t dims[], const Bool_t periods[]) const
{
   Int_t *int_periods = new Int_t[ndims];
   for (Int_t i = 0; i < ndims; i++) {
      int_periods[i] = (Int_t)periods[i];
   }
   Int_t newrank;
   ROOT_MPI_CHECK_CALL(MPI_Cart_map, (fComm, ndims, const_cast<Int_t *>(dims), int_periods, &newrank), this);
   delete[] int_periods;
   return newrank;
}
