// @(#)root/base:$Name:  $:$Id: TKey.h,v 1.6 2002/02/02 11:54:34 brun Exp $
// Author: Rene Brun   28/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TKey
#define ROOT_TKey


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TKey                                                                 //
//                                                                      //
// Header description of a logical record on file.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TDatime
#include "TDatime.h"
#endif
#ifndef ROOT_TBuffer
#include "TBuffer.h"
#endif

class TClass;
class TBrowser;

class TKey : public TNamed {

protected:
    Int_t       fVersion;        //Key version identifier
    Int_t       fNbytes;         //Number of bytes for the object on file
    Int_t       fObjlen;         //Length of uncompressed object in bytes
    TDatime     fDatime;         //Date/Time of insertion in file
    Short_t     fKeylen;         //Number of bytes for the key itself
    Short_t     fCycle;          //Cycle number
    Seek_t      fSeekKey;        //Location of object on file
    Seek_t      fSeekPdir;       //Location of parent directory on file
    TString     fClassName;      //Object Class name
    Int_t       fLeft;           //Number of bytes left in current segment
    char        *fBuffer;        //Object buffer
    TBuffer     *fBufferRef;     //Pointer to the TBuffer object

    virtual void     Create(Int_t nbytes);
    virtual Int_t    Read(const char *name) { return TObject::Read(name); }

public:
    TKey();
    TKey(const char *name, const char *title, TClass *cl, Int_t nbytes);
    TKey(const TString &name, const TString &title, TClass *cl, Int_t nbytes);
    TKey(TObject *obj, const char *name, Int_t bufsize);
    TKey(Seek_t pointer, Int_t nbytes);
    virtual ~TKey();
    virtual void      Browse(TBrowser *b);
    virtual void      Delete(Option_t *option="");
    virtual void      DeleteBuffer();
    virtual void      FillBuffer(char *&buffer);
    virtual const char *GetClassName() const {return fClassName.Data();}
    virtual char     *GetBuffer() const {return fBuffer+fKeylen;}
         TBuffer     *GetBufferRef() const {return fBufferRef;}
         Short_t      GetCycle() const ;
         Short_t      GetKeep() const;
           Int_t      GetKeylen() const  {return fKeylen;}
           Int_t      GetNbytes() const  {return fNbytes;}
           Int_t      GetObjlen() const  {return fObjlen;}
           Int_t      GetVersion() const {return fVersion;}
    virtual Seek_t    GetSeekKey() const  {return fSeekKey;}
    virtual Seek_t    GetSeekPdir() const {return fSeekPdir;}
    virtual ULong_t   Hash() const;
    Bool_t            IsFolder() const;
    virtual void      Keep();
    virtual void      ls(Option_t *option="") const;
    virtual void      Print(Option_t *option="") const;
    virtual Int_t     Read(TObject *obj);
    virtual TObject  *ReadObj();
    virtual void      ReadBuffer(char *&buffer);
    virtual void      ReadFile();
    virtual void      SetBuffer() { fBuffer = new char[fNbytes];}
    virtual void      SetParent(TObject *parent);
    virtual Int_t     Sizeof() const;
    virtual Int_t     WriteFile(Int_t cycle=1);

    ClassDef(TKey,2)  //Header description of a logical record on file
};

#endif
