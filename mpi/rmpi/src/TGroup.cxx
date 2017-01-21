#include<Mpi/TGroup.h>


using namespace ROOT::Mpi;

//______________________________________________________________________________
Int_t TGroup::GetSize() const
{
   Int_t size;
   MPI_Group_size(fGroup, &size);
   return size;
}

//______________________________________________________________________________
Int_t TGroup::GetRank() const
{
   Int_t rank;
   MPI_Group_rank(fGroup, &rank);
   return rank;
}

//______________________________________________________________________________
void TGroup::TranslateRanks(const TGroup &group1, Int_t n, const Int_t ranks1[], const TGroup &group2, Int_t ranks2[])
{
   MPI_Group_translate_ranks(group1.fGroup, n, const_cast<Int_t *>(ranks1), group2.fGroup, const_cast<Int_t *>(ranks2));
}

//______________________________________________________________________________
Int_t TGroup::Compare(const TGroup &group1, const TGroup &group2)
{
   Int_t result;
   MPI_Group_compare(group1.fGroup, group2.fGroup, &result);
   return result;

}

//______________________________________________________________________________
TGroup TGroup::Union(const TGroup &group1, const TGroup &group2)
{
   MPI_Group newgroup;
   MPI_Group_union(group1.fGroup, group2.fGroup, &newgroup);
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
   MPI_Group_difference(group1.fGroup, group2.fGroup, &newgroup);
   return newgroup;
}

//______________________________________________________________________________
TGroup TGroup::Include(Int_t n, const Int_t ranks[]) const
{
   MPI_Group newgroup;
   MPI_Group_incl(fGroup, n, const_cast<Int_t *>(ranks), &newgroup);
   return newgroup;

}

//______________________________________________________________________________
TGroup TGroup::Exclude(Int_t n, const Int_t ranks[]) const
{
   MPI_Group newgroup;
   MPI_Group_excl(fGroup, n, const_cast<Int_t *>(ranks), &newgroup);
   return newgroup;
}

//______________________________________________________________________________
TGroup TGroup::RangeInclude(Int_t n, const Int_t ranges[][3]) const
{
   MPI_Group newgroup;
   MPI_Group_range_incl(fGroup, n, const_cast<Int_t(*)[3]>(ranges), &newgroup);
   return newgroup;
}

//______________________________________________________________________________
TGroup TGroup::RangeExclude(Int_t n, const Int_t ranges[][3]) const
{
   MPI_Group newgroup;
   MPI_Group_range_excl(fGroup, n, const_cast<int(*)[3]>(ranges), &newgroup);
   return newgroup;
}

//______________________________________________________________________________
void TGroup::Free()
{
   MPI_Group_free(&fGroup);
}