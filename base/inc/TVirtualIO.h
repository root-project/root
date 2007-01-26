// @(#)root/base:$Name:  $:$Id: TVirtualIO.h,v 1.1 2007/01/25 11:46:20 brun Exp $
// Author: Rene Brun 24/01/2007

#ifndef ROOT_TVirtualIO
#define ROOT_TVirtualIO

//////////////////////////////////////////////////////////////////////////
//                                                                      
// TVirtualIO                                                       
//                                                                     
// TVirtualIO is an interface class for File I/O operations that cannot
// be performed via TDirectory.
// The class is used to decouple the base classes from the I/O packages.
// The concrete I/O sub-system is dynamically linked by the PluginManager.
// The default implementation TFileIO can be changed in system.rootrc.
//                                     
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif

class TProcessID;
class TRefTable;

class TVirtualIO: public TObject {

 protected:
   static TVirtualIO *fgIO;      //Pointer to concrete I/O class
   static TString     fgDefault; //Name of the default I/O system
   
   TVirtualIO(const TVirtualIO& io);
   TVirtualIO& operator=(const TVirtualIO& io);
   
 public:

   TVirtualIO();
   virtual ~TVirtualIO();

   virtual TObject    *CloneObject(const TObject *obj) = 0;
   virtual TObject    *FindObjectAny(const char *name) const = 0;
   virtual TProcessID *GetLastProcessID(TBuffer &b, TRefTable *reftable) const = 0;
   virtual UInt_t      GetTRefExecId() = 0;
   virtual TObject    *Open(const char *name, Option_t *option = "",
                            const char *ftitle = "", Int_t compress = 1,
                            Int_t netopt = 0) = 0;
   virtual UShort_t    ReadProcessID (TBuffer &b, TProcessID *pid) = 0;
   virtual void        ReadRefUniqueID(TBuffer &b, TObject *obj) = 0;    
   virtual Int_t       SaveObjectAs(const TObject *obj, const char *filename="", Option_t *option="") = 0;
   virtual void        SetRefAction(TObject *ref, TObject *parent) = 0;
   virtual UShort_t    WriteProcessID(TBuffer &b, TProcessID *pid) = 0;
   virtual void        WriteRefUniqueID(TBuffer &b, TObject *obj) = 0;
   
   static TVirtualIO  *GetIO();
   static const char  *GetDefaultIO();
   static void         SetDefaultIO(const char *name ="");

   ClassDef(TVirtualIO, 0); //abstract interface for File I/O operations
};

#endif
