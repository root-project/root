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

/// @file
/// @brief TTVSession and TTVRecord - I/O classes for TreeViewer session handling

#include "TObject.h"
#include "TString.h"

class TTreeViewer;
class TClonesArray;
class TGVButtonGroup;

class TTVRecord : public TObject {

public:
   TString              fName;                  ///< Name of this record
   TString              fX;                     ///< X expression
   TString              fXAlias;                ///< X alias
   TString              fY;                     ///< Y expression
   TString              fYAlias;                ///< Y alias
   TString              fZ;                     ///< Z expression
   TString              fZAlias;                ///< Z alias
   TString              fCut;                   ///< Cut expression
   TString              fCutAlias;              ///< Cut alias
   TString              fOption;                ///< Graphic option
   bool                 fScanRedirected;        ///< Redirect switch
   bool                 fCutEnabled;            ///< True if current cut is active
   TString              fUserCode;              ///< Command executed when record is connected
   bool                 fAutoexec;              ///< Autoexecute user code command

public:
   TTVRecord();                                 ///< Default constructor
   ~TTVRecord() override {}                              ///< Destructor

   void           ExecuteUserCode();
   void           FormFrom(TTreeViewer *tv);
   void           PlugIn(TTreeViewer *tv);
   const char    *GetX() const {return fX;}
   const char    *GetY() const {return fY;}
   const char    *GetZ() const {return fZ;}
   const char *GetName() const override {return fName;}
   const char    *GetUserCode() const {return fUserCode;}
   bool           HasUserCode() const {return fUserCode.Length() != 0 ? true : false;}
   bool           MustExecuteCode() const {return fAutoexec;}
   void           SetAutoexec(bool autoexec=true) {fAutoexec=autoexec;} // *TOGGLE* *GETTER=MustExecuteCode
   void           SetName(const char* name = "") {fName = name;}
   void           SetX(const char *x = "", const char *xal = "-empty-") {fX = x; fXAlias = xal;}
   void           SetY(const char *y = "", const char *yal = "-empty-") {fY = y; fYAlias = yal;}
   void           SetZ(const char *z = "", const char *zal = "-empty-") {fZ = z; fZAlias = zal;}
   void           SetCut(const char *cut = "", const char *cal = "-empty-") {fCut = cut; fCutAlias = cal;}
   void           SetOption(const char *option = "")             {fOption = option;}
   void           SetRC(bool redirect = false, bool cut = true) {fScanRedirected = redirect; fCutEnabled = cut;}
   void           SetUserCode(const char *code, bool autoexec=true) {fUserCode = code; fAutoexec=autoexec;} // *MENU*
   void           SaveSource(std::ofstream &out);

   ClassDefOverride(TTVRecord, 0)    // A draw record for TTreeViewer
};

class TTVSession : public TObject {

private:
   TClonesArray  *fList;                        ///< List of TV records
   TString        fName;                        ///< Name of this session
   TTreeViewer   *fViewer;                      ///< Associated tree viewer
   Int_t          fCurrent;                     ///< Index of current record
   Int_t          fRecords;                     ///< Number of records

public:
   TTVSession(TTreeViewer *tv);
   ~TTVSession() override;
   const char *GetName() const override      {return fName;}
   void           SetName(const char *name) {fName = name;}
   void           SetRecordName(const char* name);
   TTVRecord     *AddRecord(bool fromFile = false);
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

   ClassDefOverride(TTVSession, 0)   // A tree viewer session
};

#endif
