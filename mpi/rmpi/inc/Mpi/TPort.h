// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TPort
#define ROOT_Mpi_TPort

#include<Mpi/Globals.h>
#include<Mpi/TInfo.h>

namespace ROOT {

   namespace Mpi {

      class TPort : public TObject {
      protected:
         TString fPort;
         TString fPublishName;
         TInfo   fInfo;
         TPort(const TInfo &info, TString port, TString pname);
      public:
         TPort(const TInfo &info = INFO_NULL);
         TPort(const TPort &port);
         virtual ~TPort();

         const TString GetPortName() const;
         const TInfo   GetInfo() const;
         const TString GetPublishName() const;

         void Open(const TInfo &info = INFO_NULL);
         void Close();
         Bool_t IsOpen();

         static TPort LookupName(TString service_name, const TInfo &info = INFO_NULL);
         void PublishName(TString service_name);
         void UnpublishName(TString service_name);

         TPort &operator=(const TPort &data)
         {
            fPort = data.fPort;
            fPublishName = data.fPublishName;
            fInfo = data.fInfo;
            return *this;
         }

         const Bool_t &operator==(const TPort &data) const
         {
            if (fPort == data.fPort && fPublishName == data.fPublishName) return kTRUE;
            else return kFALSE;
//             fInfo = data.fInfo;//TODO:comparison operator for TInfo?
         }
         void Print();


         ClassDef(TPort, 1)

      };
   }
}
#endif
