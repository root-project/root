// @(#)root/treeviewer:$Id$
//Author : Andrei Gheata   21/02/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTVSession
#define ROOT_TTVSession

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// TTVSession and TTVRecord - I/O classes for TreeViewer session handling    //
//     TTreeViewer                                                           //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "TObject.h"
#include "TString.h"

class TTreeViewer;
class TClonesArray;
class TGVButtonGroup;

class TTVRecord : public TObject {

public:
   TString              fName;                  // name of this record
   TString              fX, fXAlias;            // X expression and alias
   TString              fY, fYAlias;            // Y expression and alias
   TString              fZ, fZAlias;            // Z expression and alias
   TString              fCut, fCutAlias;        // cut expression and alias
   TString              fOption;                // graphic option
   Bool_t               fScanRedirected;        // redirect switch
   Bool_t               fCutEnabled;            // true if current cut is active
   TString              fUserCode;              // command executed when record is conected
   Bool_t               fAutoexec;              // autoexecute user code command
public:
   TTVRecord();                                 // default constructor
   ~TTVRecord() {}                              // destructor

   void           ExecuteUserCode();
   void           FormFrom(TTreeViewer *tv);
   void           PlugIn(TTreeViewer *tv);
   const char    *GetX() const {return fX;}
   const char    *GetY() const {return fY;}
   const char    *GetZ() const {return fZ;}
   virtual const char *GetName() const {return fName;}
   const char    *GetUserCode() const {return fUserCode;}
   Bool_t         HasUserCode() const {return fUserCode.Length() != 0 ? kTRUE : kFALSE;}
   Bool_t         MustExecuteCode() const {return fAutoexec;}
   void           SetAutoexec(Bool_t autoexec=kTRUE) {fAutoexec=autoexec;} // *TOGGLE* *GETTER=MustExecuteCode
   void           SetName(const char* name = "") {fName = name;}
   void           SetX(const char *x = "", const char *xal = "-empty-") {fX = x; fXAlias = xal;}
   void           SetY(const char *y = "", const char *yal = "-empty-") {fY = y; fYAlias = yal;}
   void           SetZ(const char *z = "", const char *zal = "-empty-") {fZ = z; fZAlias = zal;}
   void           SetCut(const char *cut = "", const char *cal = "-empty-") {fCut = cut; fCutAlias = cal;}
   void           SetOption(const char *option = "")             {fOption = option;}
   void           SetRC(Bool_t redirect = kFALSE, Bool_t cut = kTRUE) {fScanRedirected = redirect; fCutEnabled = cut;}
   void           SetUserCode(const char *code, Bool_t autoexec=kTRUE) {fUserCode = code; fAutoexec=autoexec;} // *MENU*
   void           SaveSource(std::ofstream &out);

   ClassDef(TTVRecord, 0)    // A draw record for TTreeViewer
};

class TTVSession : public TObject {

private:
   TClonesArray  *fList;                        // list of TV records
   TString        fName;                        // name of this session
   TTreeViewer   *fViewer;                      // associated tree viewer
   Int_t          fCurrent;                     // index of current record
   Int_t          fRecords;                     // number of records

public:
   TTVSession(TTreeViewer *tv);
   ~TTVSession();
   virtual const char *GetName() const      {return fName;}
   void           SetName(const char *name) {fName = name;}
   void           SetRecordName(const char* name);
   TTVRecord     *AddRecord(Bool_t fromFile = kFALSE);
   Int_t          GetEntries() {return fRecords;}
   TTVRecord     *GetCurrent() {return GetRecord(fCurrent);}
   TTVRecord     *GetRecord(Int_t i);
   TTVRecord     *First()    {return GetRecord(0);}
   TTVRecord     *Last()     {return GetRecord(fRecords-1);}
   TTVRecord     *Next()     {return GetRecord(fCurrent+1);}
   TTVRecord     *Previous() {return GetRecord(fCurrent-1);}

   void           RemoveLastRecord();
   void           Show(TTVRecord *rec);
   void           SaveSource(std::ofstream &out);
   void           UpdateRecord(const char *name);

   ClassDef(TTVSession, 0)   // A tree viewer session
};

#endif
