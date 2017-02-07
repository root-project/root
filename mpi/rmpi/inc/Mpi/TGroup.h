// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TGroup
#define ROOT_Mpi_TGroup

#ifndef ROOT_Mpi_Globals
#include<Mpi/Globals.h>
#endif

namespace ROOT {

   namespace Mpi {
      class TCommunicator;
      class TGroup: public TObject {
         friend class TCommunicator;
      protected:
         MPI_Group fGroup; //!
      public:
         using TObject::Compare;
         // construction
         TGroup(): TObject(), fGroup(GROUP_EMPTY) {}
         TGroup(const MPI_Group &group): fGroup(group) {}
         TGroup(const TGroup &group): TObject(group), fGroup(group.fGroup) {}

         inline virtual ~TGroup() {}

         inline TGroup &operator=(const TGroup &group)
         {
            fGroup = group.fGroup;
            return *this;
         }

         inline Bool_t operator== (const TGroup &group)
         {
            return (Bool_t)(fGroup == group.fGroup);
         }
         inline Bool_t operator!= (const TGroup &group)
         {
            return (Bool_t)!(fGroup == group.fGroup);
         }

         inline TGroup &operator= (const MPI_Group &group)
         {
            fGroup = group;
            return *this;
         }
         inline operator MPI_Group() const
         {
            return (const MPI_Group)fGroup;
         }


         //
         // Groups, Contexts, and Communicators
         //

         virtual Int_t GetSize() const;

         virtual Int_t GetRank() const;

         static void TranslateRanks(const TGroup &group1, Int_t n, const Int_t ranks1[],
                                    const TGroup &group2, Int_t ranks2[]);

         static Int_t Compare(const TGroup &group1, const TGroup &group2);

         Int_t Compare(const TGroup &group2);

         static TGroup Union(const TGroup &group1, const TGroup &group2);

         static TGroup Intersect(const TGroup &group1, const TGroup &group2);

         static TGroup Difference(const TGroup &group1, const TGroup &group2);

         virtual TGroup Include(Int_t n, const Int_t ranks[]) const;

         virtual TGroup Exclude(Int_t n, const Int_t ranks[]) const;

         virtual TGroup RangeInclude(Int_t n, const Int_t ranges[][3]) const;

         virtual TGroup RangeExclude(Int_t n, const Int_t ranges[][3]) const;

         virtual void Free();

         ClassDef(TGroup, 1)
      };
   }

}

#endif
