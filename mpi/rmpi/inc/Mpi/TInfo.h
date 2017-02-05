// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TInfo
#define ROOT_Mpi_TInfo

#ifndef ROOT_Mpi_Globals
#include<Mpi/Globals.h>
#endif


namespace ROOT {

   namespace Mpi {
      class TCommunicator;
      class TInfo : public TObject {
         friend class TCommunicator;
      protected:
         MPI_Info fInfo;
         virtual Bool_t GetValueLength(const TString key, Int_t &valuelen) const;
      public:
         TInfo() {};
         TInfo(const TInfo &info): TObject(info)
         {
            fInfo = info.fInfo;
         }
         TInfo(const MPI_Info &info)
         {
            fInfo = info;
         }

         virtual ~TInfo() {}

         TInfo &operator=(const TInfo &data)
         {
            fInfo = data.fInfo;
            return *this;
         }

         static TInfo Create();

         virtual void Delete(const TString key);

         TInfo Dup() const;

         virtual void Free();

         virtual void Set(const TString key, const TString value);

         virtual Int_t GetNKeys() const;

         virtual void GetNthKey(Int_t n, TString key) const;

         virtual Bool_t Get(const TString key, TString &value) const;

         operator MPI_Info()
         {
            return fInfo;
         }
         ClassDef(TInfo, 1)

      };
   }
}
#endif
