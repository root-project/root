#include<Mpi/TGroup.h>
#include<Mpi/TErrorHandler.h>

using namespace ROOT::Mpi;

//______________________________________________________________________________
Int_t TGroup::GetSize() const
{
   Int_t size;
   ROOT_MPI_CHECK_CALL(MPI_Group_size, (fGroup, &size), this);
   return size;
}

//______________________________________________________________________________
Int_t TGroup::GetRank() const
{
   Int_t rank;
   ROOT_MPI_CHECK_CALL(MPI_Group_rank, (fGroup, &rank), this);
   return rank;
}

//______________________________________________________________________________
void TGroup::TranslateRanks(const TGroup &group1, Int_t n, const Int_t ranks1[], const TGroup &group2, Int_t ranks2[])
{
   ROOT_MPI_CHECK_CALL(MPI_Group_translate_ranks, (group1.fGroup, n, const_cast<Int_t *>(ranks1), group2.fGroup, const_cast<Int_t *>(ranks2)), TGroup::Class_Name());
}

//______________________________________________________________________________
Int_t TGroup::Compare(const TGroup &group1, const TGroup &group2)
{
   Int_t result;
   ROOT_MPI_CHECK_CALL(MPI_Group_compare, (group1.fGroup, group2.fGroup, &result), TGroup::Class_Name());
   return result;

}

//______________________________________________________________________________
Int_t TGroup::Compare(const TGroup &group2)
{
   Int_t result;
   ROOT_MPI_CHECK_CALL(MPI_Group_compare, (fGroup, group2.fGroup, &result), this);
   return result;

}

//______________________________________________________________________________
TGroup TGroup::Union(const TGroup &group1, const TGroup &group2)
{
   MPI_Group newgroup;
   ROOT_MPI_CHECK_CALL(MPI_Group_union, (group1.fGroup, group2.fGroup, &newgroup), TGroup::Class_Name());
   return newgroup;
}

//______________________________________________________________________________
TGroup TGroup::Intersect(const TGroup &group1, const TGroup &group2)
{
   MPI_Group newgroup;
   MPI_Group_intersection(group1.fGroup,  group2.fGroup, &newgroup);
   return newgroup;
}

//______________________________________________________________________________
TGroup TGroup::Difference(const TGroup &group1, const TGroup &group2)
{
   MPI_Group newgroup;
   ROOT_MPI_CHECK_CALL(MPI_Group_difference, (group1.fGroup, group2.fGroup, &newgroup), TGroup::Class_Name());
   return newgroup;
}

//______________________________________________________________________________
TGroup TGroup::Include(Int_t n, const Int_t ranks[]) const
{
   MPI_Group newgroup;
   ROOT_MPI_CHECK_CALL(MPI_Group_incl, (fGroup, n, const_cast<Int_t *>(ranks), &newgroup), this);
   return newgroup;

}

//______________________________________________________________________________
TGroup TGroup::Exclude(Int_t n, const Int_t ranks[]) const
{
   MPI_Group newgroup;
   ROOT_MPI_CHECK_CALL(MPI_Group_excl, (fGroup, n, const_cast<Int_t *>(ranks), &newgroup), this);
   return newgroup;
}

//______________________________________________________________________________
TGroup TGroup::RangeInclude(Int_t n, const Int_t ranges[][3]) const
{
   MPI_Group newgroup;
   ROOT_MPI_CHECK_CALL(MPI_Group_range_incl, (fGroup, n, const_cast<Int_t(*)[3]>(ranges), &newgroup), this);
   return newgroup;
}

//______________________________________________________________________________
TGroup TGroup::RangeExclude(Int_t n, const Int_t ranges[][3]) const
{
   MPI_Group newgroup;
   ROOT_MPI_CHECK_CALL(MPI_Group_range_excl, (fGroup, n, const_cast<int(*)[3]>(ranges), &newgroup), this);
   return newgroup;
}

//______________________________________________________________________________
void TGroup::Free()
{
   ROOT_MPI_CHECK_CALL(MPI_Group_free, (&fGroup), this);
}
