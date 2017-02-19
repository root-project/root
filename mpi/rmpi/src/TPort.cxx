#include<Mpi/TPort.h>
#include<iostream>

using namespace ROOT::Mpi;

//______________________________________________________________________________
TPort::TPort(const TInfo &info): fPort(""), fPublishName(""), fInfo(info)
{
   Open(info);
}

//______________________________________________________________________________
TPort::TPort(const TInfo &info, TString port, TString pname): fPort(port), fPublishName(pname), fInfo(info) {}

//______________________________________________________________________________
TPort::TPort(const TPort &port): TObject(port), fPort(port.fPort), fPublishName(port.fPublishName), fInfo(port.fInfo)
{
}

//______________________________________________________________________________
TPort::~TPort()
{
   if (!IsOpen()) Close();
}

//______________________________________________________________________________
const TString TPort::GetPortName() const
{
   return fPort;
}

//______________________________________________________________________________
const TInfo TPort::GetInfo() const
{
   return fInfo;
}

//______________________________________________________________________________
const TString TPort::GetPublishName() const
{
   return fPublishName;
}

//______________________________________________________________________________
void TPort::Open(const TInfo &info)
{
   fInfo = info;
   Char_t *port = new Char_t[MAX_PORT_NAME];
   MPI_Open_port(info, port);
   fPort = port;
}

//______________________________________________________________________________
void TPort::Close()
{
   Char_t *port = const_cast<Char_t *>(fPort.Data());
   MPI_Close_port(port);
   fPort = "";
}

//______________________________________________________________________________
Bool_t TPort::IsOpen()
{
   return fPort.IsNull() != kTRUE;
}

//______________________________________________________________________________
TPort TPort::LookupName(const TString service_name, const TInfo &info)
{
   //TODO: error handling here
   Char_t *port = new Char_t[MAX_PORT_NAME];
   MPI_Lookup_name(service_name, info, port);
   return TPort(info, port, service_name);
}

//______________________________________________________________________________
void TPort::PublishName(const TString service_name)
{
   //TODO: error handling here
   MPI_Publish_name(const_cast<Char_t *>(service_name.Data()), fInfo, const_cast<Char_t *>(fPort.Data()));
   fPublishName = service_name;
}

//______________________________________________________________________________
void TPort::UnpublishName(const TString service_name)
{
   //TODO: error handling here
   MPI_Unpublish_name(const_cast<Char_t *>(service_name.Data()), fInfo, const_cast<Char_t *>(fPort.Data()));
   fPublishName = "";
}

//______________________________________________________________________________
void TPort::Print()
{
   std::cout << std::flush << fPort << std::endl << std::flush;
}
