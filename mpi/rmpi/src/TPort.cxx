#include <Mpi/TPort.h>
#include <iostream>

using namespace ROOT::Mpi;

//______________________________________________________________________________
/**
 * Create a port and opens it with the info provided.
 * \see ROOT::Mpi::TIntraCommunicator::Connect
 * ROOT::Mpi::TIntraCommunicator::Aceept
 * \param info TInfo object with the port information
 */
TPort::TPort(const TInfo &info) : fPort(""), fPublishName(""), fInfo(info)
{
   Open(info);
}

//______________________________________________________________________________
TPort::TPort(const TInfo &info, TString port, TString pname) : fPort(port), fPublishName(pname), fInfo(info)
{
}

//______________________________________________________________________________
TPort::TPort(const TPort &port) : TObject(port), fPort(port.fPort), fPublishName(port.fPublishName), fInfo(port.fInfo)
{
}

//______________________________________________________________________________
TPort::~TPort()
{
   if (!IsOpen()) Close();
}

//______________________________________________________________________________
/**
 * Returns the port name.
 * \return port name (string).
 */
const TString TPort::GetPortName() const
{
   return fPort;
}

//______________________________________________________________________________
/**
 * Returns the port's information
 * \return ROOT::Mpi::TInfo object with the port's information.
 */
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
/**
 * Establishes  a network address, encoded in the port_name string, at which the
 * server will be able to accept connections from clients. port_name is supplied
 * by the system.
 * MPI copies a system-supplied port name into port_name. port_name identifies
 * the newly opened port and can be used by a client to contact the server. The
 * maximum size string that may be supplied by the system is
 * ROOT::Mpi::MAX_PORT_NAME.
 * \param info TInfo object with the port information.
 */
void TPort::Open(const TInfo &info)
{
   fInfo = info;
   Char_t *port = new Char_t[MAX_PORT_NAME];
   ROOT_MPI_CHECK_CALL(MPI_Open_port, (info, port), TPort::Class_Name());
   fPort = port;
}

//______________________________________________________________________________
/**
 * Releases the specified network address.
 */
void TPort::Close()
{
   Char_t *port = const_cast<Char_t *>(fPort.Data());
   ROOT_MPI_CHECK_CALL(MPI_Close_port, (port), TPort::Class_Name());
   fPort = "";
}

//______________________________________________________________________________
Bool_t TPort::IsOpen()
{
   return fPort.IsNull() != kTRUE;
}

//______________________________________________________________________________
/**
 * Finds port associated with a service name.
 * \param service_name A service name (string)
 * \param info TInfo object with extra options.
 * \return TPort object with the port information.
 */
TPort TPort::LookupName(const TString service_name, const TInfo &info)
{
   // TODO: error handling here
   Char_t *port = new Char_t[MAX_PORT_NAME];
   ROOT_MPI_CHECK_CALL(MPI_Lookup_name, (service_name, info, port), TPort::Class_Name());
   return TPort(info, port, service_name);
}

//______________________________________________________________________________
/**
 * This  routine  publishes  the  pair (service_name, port_name) so that an
 * application may retrieve port_name by calling ROOT::Mpi::TPort::LookupName
 * with service_name as an argument. It is an error to publish the same
 * service_name twice, or to use a port_name argument that was not previously
 * opened by the calling  process  via  a call to ROOT::Mpi::TPort::Open.
 * \param service_name A service name (string).
 */
void TPort::PublishName(const TString service_name)
{
   // TODO: error handling here
   ROOT_MPI_CHECK_CALL(MPI_Publish_name,
                       (const_cast<Char_t *>(service_name.Data()), fInfo, const_cast<Char_t *>(fPort.Data())),
                       TPort::Class_Name());
   fPublishName = service_name;
}

//______________________________________________________________________________
/**
 * This  routine  removes  the pair (service_name, port_name) so that
 * applications may no longer retrieve port_name by calling
 * ROOT::Mpi::TPort::LookupName. It is an error to unpublish a service_name that
 * was not published via ROOT::Mpi::TPort::PublishName. Both the service_name
 * and port_name arguments to ROOT::Mpi::TPort::UnPublishName must be identical
 * to the arguments to the previous call to ROOT::Mpi::TPort::PublishName.
 * \param service_name A service name (string).
 */
void TPort::UnPublishName(const TString service_name)
{
   // TODO: error handling here
   ROOT_MPI_CHECK_CALL(MPI_Unpublish_name,
                       (const_cast<Char_t *>(service_name.Data()), fInfo, const_cast<Char_t *>(fPort.Data())),
                       TPort::Class_Name());
   fPublishName = "";
}

//______________________________________________________________________________
void TPort::Print()
{
   std::cout << std::flush << fPort << std::endl << std::flush;
}
