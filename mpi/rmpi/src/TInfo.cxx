#include<Mpi/TInfo.h>


using namespace ROOT::Mpi;

//______________________________________________________________________________
TInfo TInfo::Create()
{
   MPI_Info ninfo;
   MPI_Info_create(&ninfo);
   return TInfo(ninfo);
}

//______________________________________________________________________________
TInfo  TInfo::Dup() const
{
   MPI_Info ninfo;
   MPI_Info_dup(fInfo, &ninfo);
   return TInfo(ninfo);
}

//______________________________________________________________________________
void TInfo::Delete(const TString key)
{
   MPI_Info_delete(fInfo, const_cast<Char_t *>(key.Data()));
}


//______________________________________________________________________________
void TInfo::Free()
{
   MPI_Info_free(&fInfo);
}

//______________________________________________________________________________
Bool_t TInfo::Get(const TString key, TString &value) const
{
   Int_t flag;
   Int_t len;
   if (!GetValueLength(key, len)) {
      //TODO: error handling here
   }

   Char_t *rvalue = new Char_t[len];
   MPI_Info_get(fInfo, const_cast<Char_t *>(key.Data()), len, rvalue, &flag);
   value.Resize(len);
   value = rvalue;
   return flag;
}

//______________________________________________________________________________
Int_t TInfo::GetNKeys()const
{
   Int_t nkeys;
   MPI_Info_get_nkeys(fInfo, &nkeys);
   return nkeys;
}

//______________________________________________________________________________
void TInfo::GetNthKey(Int_t n, TString key) const
{
   MPI_Info_get_nthkey(fInfo, n, const_cast<Char_t *>(key.Data()));
}

//______________________________________________________________________________
Bool_t TInfo::GetValueLength(const TString key, Int_t &valuelen) const
{
   Int_t flag;
   MPI_Info_get_valuelen(fInfo, const_cast<Char_t *>(key.Data()), &valuelen, &flag);
   return (Bool_t)flag;
}

//______________________________________________________________________________
void TInfo::Set(const TString key, const TString value)
{
   MPI_Info_set(fInfo, const_cast<Char_t *>(key.Data()), const_cast<Char_t *>(value.Data()));
}
