#include<Mpi/TInfo.h>
#include<iostream>

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
Int_t TInfo::GetNKeys() const
{
   Int_t nkeys;
   MPI_Info_get_nkeys(fInfo, &nkeys);
   return nkeys;
}

//______________________________________________________________________________
TString TInfo::GetNthKey(Int_t n) const
{
   Char_t *key = new Char_t[MPI_MAX_INFO_KEY];
   MPI_Info_get_nthkey(fInfo, n, const_cast<Char_t *>(key));
   return TString(key);
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

//______________________________________________________________________________
TString TInfo::GetValue(const TString key) const
{
   TString value;
   if (!Get(key, value)) {
      //TODO: added error handling here
   }
   return value;
}

//______________________________________________________________________________
Bool_t TInfo::IsEmpty() const
{
   if (IsNull()) return kTRUE;
   else return GetNKeys() == 0 ? kTRUE : kFALSE;
}

//______________________________________________________________________________
Bool_t TInfo::IsNull() const
{
   return fInfo == MPI_INFO_NULL;
}

//______________________________________________________________________________
Bool_t TInfo::HasKey(TString key)
{
   if (IsNull()) return kFALSE;
   if (IsEmpty()) return kFALSE;
   Bool_t status = kFALSE;
   auto i = 0;
   while (i < GetNKeys()) {
      if (key == GetNthKey(i)) {
         status = kTRUE;
         break;
      }
      i++;
   }
   return status;
}

//______________________________________________________________________________
void TInfo::Print() const
{
//TODO: if is null print something
   for (auto i = 0; i < GetNKeys(); i++) {
      std::cout << std::setw(MPI_MAX_INFO_KEY) << std::left << Form("[\"%s\"]", GetNthKey(i).Data()) << " = " << GetValue(GetNthKey(i)) << std::endl;
      std::cout.flush();
   }
}

//______________________________________________________________________________
Bool_t TInfo::operator==(const TInfo &info) const
{
   if (IsNull() && info.IsNull()) return kTRUE;
   if (IsNull() != info.IsNull()) return kFALSE;

   if (GetNKeys() != info.GetNKeys()) return kFALSE;
   if (IsEmpty() == info.IsEmpty()) return kTRUE;
   if (IsEmpty() != info.IsEmpty()) return kFALSE;

   Bool_t status = kTRUE;
   Int_t i = 0;
   while (i < GetNKeys()) {
      if (GetNthKey(i) != info.GetNthKey(i)) {
         status = kFALSE;
         break;
      }
      i++;
   }
   return status;
}

//______________________________________________________________________________
Bool_t TInfo::operator!=(const TInfo &info) const
{
   return *this == info ? kFALSE : kTRUE ;
}

