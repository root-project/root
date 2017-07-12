// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2016 http://oproject.org
#ifndef ROOT_Mpi_TMpiMessage
#define ROOT_Mpi_TMpiMessage

#include <Mpi/Globals.h>
#include <TMessage.h>

namespace ROOT {
namespace Mpi {
/**
 * \class TMpiMessageInfo
 * Class to encapsulate message information, associated with
 * the communication and the serialization.
 * \ingroup Mpi
 */

class TMpiMessageInfo : public TObject {
protected:
   TString fBuffer;       // Message buffer
   TString fDataTypeName; // Datatype name encapsulate in this message
   Int_t fSource;         // Rank(Process ID) of origin of this message
   Int_t fDestination;    // Rank(Process ID) of destination of this message
   Int_t fTag;            // Id of the message
   Int_t fRoot;           // Id for collective message
public:
   TMpiMessageInfo(const TMpiMessageInfo &msgi);

   TMpiMessageInfo(const Char_t *buffer = 0, UInt_t size = 0);

   inline void SetDataTypeName(TString name) { fDataTypeName = name; }

   inline void SetSource(Int_t src) { fSource = src; }

   inline void SetDestination(Int_t dest) { fDestination = dest; }

   inline void SetTag(Int_t tag) { fTag = tag; }

   inline void SetRoot(Int_t root) { fRoot = root; }

   inline TString GetDataTypeName() { return fDataTypeName; }

   inline const Char_t *GetBuffer() { return fBuffer.Data(); }

   inline Int_t GetBufferSize() { return fBuffer.Length(); }

   inline Int_t GetSource() { return fSource; }
   inline Int_t GetDestination() { return fDestination; }

   inline Int_t GetTag() { return fTag; }

   inline Int_t GetRoot() { return fRoot; }

   ClassDef(TMpiMessageInfo, 1)
};

/**
 * \class TMpiMessage
 * Class to encapsulate messages using serialization.
 * \ingroup Mpi
 */
class TMpiMessage : public TMessage {
protected:
   TString fDataTypeName; // Datatype name encapsulate in this message
public:
   using TMessage::WriteObject;
   TMpiMessage(Char_t *buffer, Int_t size);
   TMpiMessage(UInt_t what = kMESS_ANY, Int_t bufsiz = TBuffer::kInitialSize);

   virtual ~TMpiMessage() {}

   //______________________________________________________________________________
   /**
    * Utility method to get the data type in a string.
    * \return string with the name of the data type.
    */
   inline TString GetDataTypeName() const { return fDataTypeName; }

   template <class ClassType>
   void WriteObject(ClassType *obj);
   template <class ClassType>
   void WriteObject(ClassType &obj);

#if PYTHON_FOUND
   TMpiMessage(PyObject *obj);
   PyObject *ReadPyObject();
#endif

   ClassDef(TMpiMessage, 1);
};

//______________________________________________________________________________
/**
 * template method to write objects in the message using serialization.
 * \param obj any serialzable object pointer supported by ROOT.
 */
template <class ClassType>
void TMpiMessage::WriteObject(ClassType *obj)
{
   const std::type_info &type = typeid(*obj);
   fDataTypeName = type.name();
   TClass *cl = gROOT->GetClass(type);
   WriteObjectAny(obj, cl);
}

//______________________________________________________________________________
/**
 * template method to write objects in the message using serialization.
 * \param obj any serialzable object reference supported by ROOT.
 */
template <class ClassType>
void TMpiMessage::WriteObject(ClassType &obj)
{
   WriteObject<ClassType>(&obj);
}

#if PYTHON_FOUND
template <>
void TMpiMessage::WriteObject<PyObject>(PyObject *obj);
#endif
}
}

#endif
