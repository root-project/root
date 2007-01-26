// @(#)root/base:$Name:  $:$Id: TFileIO.h,v 1.1 2007/01/25 11:47:06 brun Exp $
// Author: Rene Brun 24/01/2007

#ifndef ROOT_TFileIO
#define ROOT_TFileIO

//////////////////////////////////////////////////////////////////////////
//                                                                      
// TFileIO                                                       
//                                                                     
// TFileIO is the concrete implementation of TVirtualIO.
//                                     
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualIO
#include "TVirtualIO.h"
#endif

class TProcessID;
class TRefTable;

class TFileIO: public TVirtualIO {

protected:
   
   TFileIO(const TFileIO& io);
   TFileIO& operator=(const TFileIO& io);
   
public:

   TFileIO();
   virtual ~TFileIO();

   virtual TObject    *CloneObject(const TObject *obj);
   virtual TObject    *FindObjectAny(const char *name) const;
   virtual TProcessID *GetLastProcessID(TBuffer &b, TRefTable *reftable) const;
   virtual UInt_t      GetTRefExecId();
   virtual TObject    *Open(const char *name, Option_t *option = "",
                            const char *ftitle = "", Int_t compress = 1,
                            Int_t netopt = 0);
   virtual UShort_t    ReadProcessID (TBuffer &b, TProcessID *pid);
   virtual void        ReadRefUniqueID(TBuffer &b, TObject *obj);    
   virtual Int_t       SaveObjectAs(const TObject *obj, const char *filename="", Option_t *option="");
   virtual void        SetRefAction(TObject *ref, TObject *parent);
   virtual UShort_t    WriteProcessID(TBuffer &b, TProcessID *pid);
   virtual void        WriteRefUniqueID(TBuffer &b, TObject *obj);
   
   ClassDef(TFileIO, 0); //Concrete implementation of TVirtualIO
};

#endif
