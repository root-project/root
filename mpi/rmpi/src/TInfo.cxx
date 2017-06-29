#include <Mpi/TInfo.h>
#include <iostream>

using namespace ROOT::Mpi;

//______________________________________________________________________________
/**
 * Creates a new info object. The newly created object contains no key/value
 * pairs.
 * \return Info object created (handle).
 */
TInfo TInfo::Create()
{
   MPI_Info ninfo;
   ROOT_MPI_CHECK_CALL(MPI_Info_create, (&ninfo), TInfo::Class_Name());
   return TInfo(ninfo);
}

//______________________________________________________________________________
/**
 * Duplicates an existing info object, creating a new object, with the same
 * (key,value) pairs and the same ordering of keys.
 * \return Info object (handle).
 */
TInfo TInfo::Dup() const
{
   MPI_Info ninfo;
   MPI_Info_dup(fInfo, &ninfo);
   return TInfo(ninfo);
}

//______________________________________________________________________________
/**
 * Deletes a (key,value) pair from info. If key is not defined in info, the call
 * raises an error of class ROOT::Mpi::ERR_INFO_NOKEY.
 * \param key Key (string).
 */
void TInfo::Delete(const TString key)
{
   ROOT_MPI_CHECK_CALL(MPI_Info_delete, (fInfo, const_cast<Char_t *>(key.Data())), TInfo::Class_Name());
}

//______________________________________________________________________________
/**
 * Frees info and sets it to ROOT::Mpi::INFO_NULL.
 */
void TInfo::Free()
{
   ROOT_MPI_CHECK_CALL(MPI_Info_free, (&fInfo), TInfo::Class_Name());
}

//______________________________________________________________________________
/**
 * Retrieves  the value associated with key in a previous call to
 * ROOT::Mpi::TInfo::Set.
 * If such a key exists, it sets flag to true and returns the value in value;
 * otherwise it sets flag to false and leaves value unchanged.
 * valuelen is the number of characters available in value. If it is less than
 * the actual size of the value, the returned value is truncated.
 * In C, valuelen should be one less than the amount of allocated space to allow
 * for the null terminator.
 * If key is larger than ROOT::Mpi::MAX_INFO_KEY, the call is erroneous.
 * \param key Key (string).
 * \param value Reference object to TString with the output value.
 */
Bool_t TInfo::Get(const TString key, TString &value) const
{
   Int_t flag;
   Int_t len;
   if (!GetValueLength(key, len)) {
      // TODO: error handling here
   }

   Char_t *rvalue = new Char_t[len];
   ROOT_MPI_CHECK_CALL(MPI_Info_get, (fInfo, const_cast<Char_t *>(key.Data()), len, rvalue, &flag),
                       TInfo::Class_Name());
   value.Resize(len);
   value = rvalue;
   return flag;
}

//______________________________________________________________________________
/**
 * Gets the number of keys currently defined in an info object.
 * \return Number of defined keys (integer).
 */
Int_t TInfo::GetNKeys() const
{
   Int_t nkeys;
   ROOT_MPI_CHECK_CALL(MPI_Info_get_nkeys, (fInfo, &nkeys), TInfo::Class_Name());
   return nkeys;
}

//______________________________________________________________________________
/**
 * Returns  the nth defined key in info. Keys are numbered 0...N - 1 where N is
 * the value returned by ROOT::Mpi::GetNKeys.
 * All keys between 0 and N - 1 are guaranteed to be defined.
 * The number of a given key does not change as long as info is not modified
 * with ROOT::Mpi::TInfo::Set or ROOT::Mpi::TInfo::Delete.
 * \param n  Key number (integer).
 * \return TString object with the key.
 */
TString TInfo::GetNthKey(Int_t n) const
{
   Char_t *key = new Char_t[MPI_MAX_INFO_KEY];
   ROOT_MPI_CHECK_CALL(MPI_Info_get_nthkey, (fInfo, n, const_cast<Char_t *>(key)), TInfo::Class_Name());
   return TString(key);
}

//______________________________________________________________________________
/**
 * Retrieves the length of the key value associated with an info object.
 * \param key Key (string).
 * \param valuelen reference to integer with length of value arg (integer).
 * \return Returns true if key defined, false if not (boolean).
 */
Bool_t TInfo::GetValueLength(const TString key, Int_t &valuelen) const
{
   Int_t flag;
   ROOT_MPI_CHECK_CALL(MPI_Info_get_valuelen, (fInfo, const_cast<Char_t *>(key.Data()), &valuelen, &flag),
                       TInfo::Class_Name());
   return (Bool_t)flag;
}

//______________________________________________________________________________
/**
 * Adds the (key,value) pair to info and overrides the value if a value for the
 * same key was previously set.
 * The key and value parameters are null-terminated strings in C.
 * If either key or value is larger than  the  allowed  maximums,the error
 * ROOT::Mpi::ERR_INFO_KEY or ROOT::Mpi::ERR_INFO_VALUE is raised, respectively.
 * \param key Key (string).
 * \param value Value (string).
 */
void TInfo::Set(const TString key, const TString value)
{
   ROOT_MPI_CHECK_CALL(MPI_Info_set, (fInfo, const_cast<Char_t *>(key.Data()), const_cast<Char_t *>(value.Data())),
                       TInfo::Class_Name());
}

//______________________________________________________________________________
/**
 * Adds the (key,value) pair to info and overrides the value if a value for the
 * same key was previously set.
 * The key and value parameters are null-terminated strings in C.
 * If either key or value is larger than  the  allowed  maximums,the error
 * ROOT::Mpi::ERR_INFO_KEY or ROOT::Mpi::ERR_INFO_VALUE is raised, respectively.
 * \param key Key (string).
 * \return Value (string).
 */
TString TInfo::GetValue(const TString key) const
{
   TString value;
   if (!Get(key, value)) {
      // TODO: added error handling here
   }
   return value;
}

//______________________________________________________________________________
/**
 * Method to check is info object is empty
 * \return kFALSE if there is not keys (boolean).
 */
Bool_t TInfo::IsEmpty() const
{
   if (IsNull())
      return kTRUE;
   else
      return GetNKeys() == 0 ? kTRUE : kFALSE;
}

//______________________________________________________________________________
/**
 * Method to check is info object is null
 * \return kTRUE if there if internal MPI_Info object is null(boolean).
 */
Bool_t TInfo::IsNull() const
{
   return fInfo == MPI_INFO_NULL;
}

//______________________________________________________________________________
/**
 * Method to check if key exists.
 * \return kTRUE if key exists (boolean).
 */
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
/**
 * Method to print all pairs key/vale store in info.
 */
void TInfo::Print() const
{
   // TODO: if is null print something
   for (auto i = 0; i < GetNKeys(); i++) {
      std::cout << std::setw(MPI_MAX_INFO_KEY) << std::left << Form("[\"%s\"]", GetNthKey(i).Data()) << " = "
                << GetValue(GetNthKey(i)) << std::endl;
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
   return *this == info ? kFALSE : kTRUE;
}
