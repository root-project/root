#include<Mpi/TGroup.h>
#include<Mpi/TErrorHandler.h>

using namespace ROOT::Mpi;

//______________________________________________________________________________
/**
 * Returns the size of a group.
 * \return Number of processes in the group (integer).
 */
Int_t TGroup::GetSize() const
{
   Int_t size;
   ROOT_MPI_CHECK_CALL(MPI_Group_size, (fGroup, &size), this);
   return size;
}

//______________________________________________________________________________
/**
 * Returns the rank of the calling process in the given group.
 * \return ank of the calling process in group, or ROOT::Mpi::UNDEFINED if the
 * process is not a member (integer).
 */
Int_t TGroup::GetRank() const
{
   Int_t rank;
   ROOT_MPI_CHECK_CALL(MPI_Group_rank, (fGroup, &rank), this);
   return rank;
}

//______________________________________________________________________________
/**
 * Translates the ranks of processes in one group to those in another group.
 *\param group1 First group (handle).
 *\param n  Number of ranks in ranks1 and ranks2 arrays (integer).
 *\param ranks1 Array of zero or more valid ranks in group1.
 *\param group2 Second group (handle).
 *\param ranks2 Output array of corresponding ranks in group2,
 *ROOT::Mpi::UNDEFINED
 *when no correspondence exists.
 */
void TGroup::TranslateRanks(const TGroup &group1, Int_t n, const Int_t ranks1[], const TGroup &group2, Int_t ranks2[])
{
   ROOT_MPI_CHECK_CALL(MPI_Group_translate_ranks, (group1.fGroup, n, const_cast<Int_t *>(ranks1), group2.fGroup, const_cast<Int_t *>(ranks2)), TGroup::Class_Name());
}

//______________________________________________________________________________
/**
 * Compares two groups.
 * ROOT::Mpi::IDENT results if the group members and group order is exactly the
 * same in
 * both groups. This happens for instance if group1 and group2 are the same
 * handle.
 * ROOT::Mpi::SIMILAR results if the group members are the same but the order is
 * different. ROOT::Mpi::UNEQUAL results otherwise.
 * \param group1 First group (handle).
 * \param group2 Second group (handle).
 * \return Integer  which  is  MPI_IDENT if the order and members of the two
 * groups are the same, ROOT::Mpi::SIMILAR if only the members are the same, and
 * ROOT::Mpi::UNEQUAL otherwise.
 */
Int_t TGroup::Compare(const TGroup &group1, const TGroup &group2)
{
   Int_t result;
   ROOT_MPI_CHECK_CALL(MPI_Group_compare, (group1.fGroup, group2.fGroup, &result), TGroup::Class_Name());
   return result;

}

//______________________________________________________________________________
/**
 * Compares two groups.(Current group respect other)
 * ROOT::Mpi::IDENT results if the group members and group order is exactly the
 * same in
 * both groups. This happens for instance if group1 and group2 are the same
 * handle.
 * ROOT::Mpi::SIMILAR results if the group members are the same but the order is
 * different. ROOT::Mpi::UNEQUAL results otherwise.
 * \param group2 Second group (handle).
 * \return Integer  which  is  ROOT::Mpi::IDENT if the order and members of the
 * two
 * groups are the same, ROOT::Mpi::SIMILAR if only the members are the same, and
 * ROOT::Mpi::UNEQUAL otherwise.
 */
Int_t TGroup::Compare(const TGroup &group2)
{
   Int_t result;
   ROOT_MPI_CHECK_CALL(MPI_Group_compare, (fGroup, group2.fGroup, &result), this);
   return result;

}

//______________________________________________________________________________
/**
 * Produces a group by combining two groups.
 * The set-like operations are defined as follows:
 * - union -- All elements of the first group (group1), followed by all
 * elements of second group (group2) not in first.
 * - intersect -- all elements of the first group that are also in the
 * second group, ordered as in first group.
 * - difference -- all elements of the first group that are not in the
 * second group, ordered as in the first group.
 * Note  that for these operations the order of processes in the output
 * group is determined primarily by order in the first group (if possible) and
 * then, if necessary, by order in the second group. Neither union nor
 * intersection are commutative, but both are associative.
 *       The new group can be empty, that is, equal to ROOT::Mpi::GROUP_EMPTY.
 * \param group1 First group (handle).
 * \param group2 Second group (handle).
 * \return TGroup object with union group (handle).
 */
TGroup TGroup::Union(const TGroup &group1, const TGroup &group2)
{
   MPI_Group newgroup;
   ROOT_MPI_CHECK_CALL(MPI_Group_union, (group1.fGroup, group2.fGroup, &newgroup), TGroup::Class_Name());
   return newgroup;
}

//______________________________________________________________________________
/**
 * Produces a group at the intersection of two existing groups.
 * The set-like operations are defined as follows:
 * - union -- All elements of the first group (group1), followed by all
 * elements of second group (group2) not in first.
 * - intersect -- all elements of the first group that are also in the
 * second group, ordered as in first group.
 * - difference -- all elements of the first group that are not in the
 * second group, ordered as in the first group.
 * Note  that for these operations the order of processes in the output
 * group is determined primarily by order in the first group (if possible) and
 * then, if necessary, by order in the second group. Neither union nor
 * intersection are commutative, but both are associative.
 *       The new group can be empty, that is, equal to ROOT::Mpi::GROUP_EMPTY.
 * \param group1 First group (handle).
 * \param group2 Second group (handle).
 * \return TGroup object with intersection group (handle).
 */
TGroup TGroup::Intersect(const TGroup &group1, const TGroup &group2)
{
   MPI_Group newgroup;
   ROOT_MPI_CHECK_CALL(MPI_Group_intersection,
                       (group1.fGroup, group2.fGroup, &newgroup),
                       TGroup::Class_Name());
   return newgroup;
}

//______________________________________________________________________________
/**
 * Makes a group from the difference of two groups.
 * The set-like operations are defined as follows:
 * - union -- All elements of the first group (group1), followed by all
 * elements of second group (group2) not in first.
 * - intersect -- all elements of the first group that are also in the
 * second group, ordered as in first group.
 * - difference -- all elements of the first group that are not in the
 * second group, ordered as in the first group.
 * Note  that for these operations the order of processes in the output
 * group is determined primarily by order in the first group (if possible) and
 * then, if necessary, by order in the second group. Neither union nor
 * intersection are commutative, but both are associative.
 * The new group can be empty, that is, equal to ROOT::Mpi::GROUP_EMPTY.
 * \param group1 First group (handle).
 * \param group2 Second group (handle).
 * \return TGroup object with difference group (handle).
 */
TGroup TGroup::Difference(const TGroup &group1, const TGroup &group2)
{
   MPI_Group newgroup;
   ROOT_MPI_CHECK_CALL(MPI_Group_difference, (group1.fGroup, group2.fGroup, &newgroup), TGroup::Class_Name());
   return newgroup;
}

//______________________________________________________________________________
/**
 * Produces a group by reordering an existing group and taking only listed
 * members.
 * The Method ROOT::Mpi::TGroup::Include creates a group that consists of the n
 * processes in group with ranks rank[0], ..., rank[n-1];
 * the process with rank i in output group is the process with rank ranks[i] in
 * group. Each of the n elements of ranks must be a valid rank in group and all
 * elements must be distinct, or else the program is erroneous. If n = 0, then
 * group_out is ROOT::Mpi::GROUP_EMPTY. This function can, for instance, be used
 * to reorder the elements of a group.
 * \param n Number of elements in array ranks (and size of newgroup)(integer).
 * \param ranks Ranks of processes in group to appear in new group (array of
 * integers).
 * \return New group derived from above, in the order defined by ranks (handle).
 */
TGroup TGroup::Include(Int_t n, const Int_t ranks[]) const
{
   MPI_Group newgroup;
   ROOT_MPI_CHECK_CALL(MPI_Group_incl, (fGroup, n, const_cast<Int_t *>(ranks), &newgroup), this);
   return newgroup;

}

//______________________________________________________________________________
/**
 * Produces a group by reordering an existing group and taking only unlisted
 * members.
 * The function ROOT::Mpi::TGroup::Exclude creates a group of processes that is
 * obtained by deleting from group those processes with ranks ranks[0], ...
 * ranks[n-1].
 * The ordering of processes in newgroup is identical to the ordering in group.
 * Each of the n elements of ranks must be a valid rank in group  and  all
 * elements must be distinct; otherwise, the call is erroneous. If n = 0, then
 * new group is identical to group.
 * \param n Number of elements in array ranks (integer).
 * \param ranks Array of integer ranks in group not to appear in new group.
 * \return New group derived from above, preserving the order defined by group
 * (handle).
 */
TGroup TGroup::Exclude(Int_t n, const Int_t ranks[]) const
{
   MPI_Group newgroup;
   ROOT_MPI_CHECK_CALL(MPI_Group_excl, (fGroup, n, const_cast<Int_t *>(ranks), &newgroup), this);
   return newgroup;
}

//______________________________________________________________________________
/**
 * Creates a new group from ranges of ranks in an existing group.
 * If ranges consist of the triplets
 *            (first1, last1, stride1), ..., (firstn, lastn, striden)
 * then new group consists of the sequence of processes in group with ranks\n
 * \f$
 * first(1), first(1) + stride(1),..., first(1)
 * +\frac{last(1)-first(1)}{stride(1)} stride(1),...\\
 *
 * first(n), first(n) + stride(n),..., first(n)
 * +\frac{last(n)-first(n)}{stride(n)} stride(n). \\
 * \f$
 * \n
 * Each computed rank must be a valid rank in group and all computed ranks must
 * be distinct, or else the program is erroneous. Note that we may have
 * first(i)  > last(i), and stride(i) may be negative, but cannot be zero.
 *
 * The  functionality  of this routine is specified to be equivalent to
 * expanding the array of ranges to an array of the included ranks and passing
 * the resulting array of ranks and other arguments to
 * ROOT::Mpi::TGroup::Include. A call to ROOT::Mpi::TGroup::Include is
 * equivalent to a call to ROOT::Mpi::TGroup::RangInclude  with  each  rank  i
 * in  ranks  replaced by the triplet (i,i,1) in the argument ranges.
 * \param n Number of triplets in array ranges (integer).
 * \param ranges A  one-dimensional  array  of integer triplets, of the form
 * (first rank, last rank, stride) indicating ranks in group or processes to be
 * included in new group.
 * \return New group derived from above, in the order defined by ranges
 * (handle).
 */
TGroup TGroup::RangeInclude(Int_t n, const Int_t ranges[][3]) const
{
   MPI_Group newgroup;
   ROOT_MPI_CHECK_CALL(MPI_Group_range_incl, (fGroup, n, const_cast<Int_t(*)[3]>(ranges), &newgroup), this);
   return newgroup;
}

//______________________________________________________________________________
/**
 * Produces a group by excluding ranges of processes from an existing group.
 * Each computed rank must be a valid rank in group and all computed ranks must
 *be distinct, or else the program is erroneous.
 * The functionality of this routine is specified to be equivalent to expanding
 *the array of ranges to an array of the excluded ranks and passing  the
 *resulting array  of  ranks  and  other  arguments  to
 *ROOT::Mpi::TGroup::Exclude. A call to ROOT::Mpi::TGroup::Exclude is equivalent
 *to a call to ROOT::Mpi::TGroup::RangeExclude with each rank i in ranks
 *replaced by the triplet (i,i,1) in the argument ranges.
 *\param n Number of triplets in array ranges (integer).
 * \param ranges A  one-dimensional  array of integer triplets of the form
 *(first rank, last rank, stride), indicating the ranks in group of processes to
 *be excluded from the output group.
 * \return New group derived from above, preserving the order in group (handle).
 */
TGroup TGroup::RangeExclude(Int_t n, const Int_t ranges[][3]) const
{
   MPI_Group newgroup;
   ROOT_MPI_CHECK_CALL(MPI_Group_range_excl, (fGroup, n, const_cast<int(*)[3]>(ranges), &newgroup), this);
   return newgroup;
}

//______________________________________________________________________________
/**
 * Frees a group.
 * This  operation marks a group object for deallocation. The handle group is
 * set to ROOT::Mpi::GROUP_NULL by the call. Any ongoing operation using this
 * group will complete normally.
 */
void TGroup::Free()
{
   ROOT_MPI_CHECK_CALL(MPI_Group_free, (&fGroup), this);
}
