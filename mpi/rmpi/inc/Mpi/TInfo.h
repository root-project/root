// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TInfo
#define ROOT_Mpi_TInfo

#include<Mpi/Globals.h>

#include <sstream>

namespace ROOT {

   namespace Mpi {
      class TCommunicator;
      class TInfo : public TObject {
         friend class TCommunicator;

         class Binding {
         private:
            MPI_Info fInternalInfo;
            TString fInternalKey;
         public:
            Binding(MPI_Info  &finf, TString key): fInternalInfo(finf), fInternalKey(key) {}
            Binding(const Binding &obj): fInternalInfo(obj.fInternalInfo), fInternalKey(obj.fInternalKey) {}
            ~Binding() {}
            void SetKey(TString key)
            {
               fInternalKey = key;
            }
            TString GetKey()
            {
               return fInternalKey;
            }
            Binding &operator=(const Binding &obj)
            {
               fInternalInfo  = obj.fInternalInfo;
               fInternalKey  = obj.fInternalKey;
               return *this;
            }

            template<class T> Binding &operator=(const T &value)
            {
               TString ivalue;
               ParseValue(ivalue, *const_cast<T *>(&value)); //parsing to string
               MPI_Info_set(fInternalInfo, const_cast<Char_t *>(fInternalKey.Data()), const_cast<Char_t *>(ivalue.Data()));
               return *this;
            }

            template<class T> operator T()
            {
               return GetValue<T>();
            }
            template<class T> T GetValue()
            {
               T result;
               Int_t flag;
               Int_t valuelen;
               MPI_Info_get_valuelen(fInternalInfo, const_cast<Char_t *>(fInternalKey.Data()), &valuelen, &flag);
               if (!flag) {
                  //TODO:added error handling here
               }
               Char_t *value = new Char_t[valuelen];
               MPI_Info_get(fInternalInfo, const_cast<Char_t *>(fInternalKey.Data()), valuelen, value, &flag);
               if (!flag) {
                  //TODO:added error handling here
               }
               TString ovalue(value, valuelen);
               ParseValue(ovalue, result, kFALSE);
               return result;
            }

            template<class T> void  ParseValue(TString &str, T &value, Bool_t input = kTRUE)
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
         TInfo(): fInfo(Create()), fBinder(fInfo, "") {}
         TInfo(const TInfo &info): TObject(info), fInfo(info.fInfo), fBinder(fInfo, "") {}
         TInfo(const MPI_Info &info): fInfo(info), fBinder(fInfo, "") {}

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

         Bool_t IsEmpty();

         Bool_t HasKey(TString key);

         Binding &operator[](const Char_t *key)
         {
            fBinder.SetKey(key);
            return fBinder;
         }

         virtual void Print();

         operator MPI_Info() const
         {
            return fInfo;
         }

         ClassDef(TInfo, 1)

      };
   }
}
#endif
