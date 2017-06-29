// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TInfo
#define ROOT_Mpi_TInfo

#include <Mpi/Globals.h>
#include <Mpi/TErrorHandler.h>

#include <sstream>

namespace ROOT {

namespace Mpi {
/**
\class TInfo
 Many of the routines in MPI take an argument info. info is an opaque object
with a handle of type MPI_Info in C and ROOT::Mpi::TInfo in C++. It stores an
unordered set of ( key, value) pairs (both key and value are strings). A key
can have only one value. MPI reserves several keys and requires that if an
implementation uses a reserved key, it must provide the specified
functionality. An implementation is not required to support these keys and
may support any others not reserved by MPI.
 With this class you can to manipulte MPI_Info object with facilities in C++,
that allow to use it like a map with overloaded
 operators and template methods.
 \ingroup Mpi
 */
class TCommunicator;
class TInfo : public TObject {
   friend class TCommunicator;

   class Binding {
   private:
      MPI_Info fInternalInfo;
      TString fInternalKey;

   public:
      Binding(MPI_Info &finf, TString key) : fInternalInfo(finf), fInternalKey(key) {}
      Binding(const Binding &obj) : fInternalInfo(obj.fInternalInfo), fInternalKey(obj.fInternalKey) {}
      ~Binding() {}
      void SetKey(TString key) { fInternalKey = key; }
      TString GetKey() { return fInternalKey; }
      Binding &operator=(const Binding &obj)
      {
         fInternalInfo = obj.fInternalInfo;
         fInternalKey = obj.fInternalKey;
         return *this;
      }

      template <class T>
      Binding &operator=(const T &value)
      {
         TString ivalue;
         ParseValue(ivalue, *const_cast<T *>(&value)); // parsing to string
         ROOT_MPI_CHECK_CALL(MPI_Info_set, (fInternalInfo, const_cast<Char_t *>(fInternalKey.Data()),
                                            const_cast<Char_t *>(ivalue.Data())),
                             TInfo::Class_Name());
         return *this;
      }

      template <class T>
      operator T()
      {
         return GetValue<T>();
      }
      template <class T>
      T GetValue()
      {
         T result;
         Int_t flag;
         Int_t valuelen;
         ROOT_MPI_CHECK_CALL(MPI_Info_get_valuelen,
                             (fInternalInfo, const_cast<Char_t *>(fInternalKey.Data()), &valuelen, &flag),
                             TInfo::Class_Name());
         if (!flag) {
            // TODO:added error handling here
         }
         Char_t *value = new Char_t[valuelen];
         ROOT_MPI_CHECK_CALL(MPI_Info_get,
                             (fInternalInfo, const_cast<Char_t *>(fInternalKey.Data()), valuelen, value, &flag),
                             TInfo::Class_Name());
         if (!flag) {
            // TODO:added error handling here
         }
         TString ovalue(value, valuelen);
         ParseValue(ovalue, result, kFALSE);
         return result;
      }

      template <class T>
      void ParseValue(TString &str, T &value, Bool_t input = kTRUE)
      {
         std::stringstream fStringStream;
         if (input) {
            fStringStream << value;
            str = fStringStream.str();
         } else {
            fStringStream << str.Data();
            fStringStream >> value;
         }
      }
   };

protected:
   MPI_Info fInfo;  //!
   Binding fBinder; //!
public:
   TInfo() : fInfo(Create()), fBinder(fInfo, "") {}
   TInfo(const TInfo &info) : TObject(info), fInfo(info.fInfo), fBinder(fInfo, "") {}
   TInfo(const MPI_Info &info) : fInfo(info), fBinder(fInfo, "") {}

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

   virtual TString GetNthKey(Int_t n) const;

   virtual Bool_t Get(const TString key, TString &value) const;

   virtual TString GetValue(const TString key) const;

   virtual Bool_t GetValueLength(const TString key, Int_t &valuelen) const;

   Bool_t IsEmpty() const;

   Bool_t IsNull() const;

   Bool_t HasKey(TString key);

   Binding &operator[](const Char_t *key)
   {
      fBinder.SetKey(key);
      return fBinder;
   }

   Bool_t operator==(const TInfo &info) const;
   Bool_t operator!=(const TInfo &info) const;

   virtual void Print() const;

   operator MPI_Info() const { return fInfo; }

   ClassDef(TInfo, 1)
};
}
}
#endif
