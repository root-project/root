// @(#)root/tmva $Id: TMVA_AsciiConverter.h,v 1.6 2006/04/29 23:55:41 andreas.hoecker Exp $ 
// Author: unknown

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_AsciiConverter                                                   *
 *                                                                                *
 * Description:                                                                   *
 *      Convert ascii file into TTree object                                      *
 *      (TMVA_AsciiConverter does not own this TTree object)                      *
 *                                                                                *
 *      Supported data types are:                                                 *
 *      Float_t  Double_t  Int_t TString                                          *
 *                                                                                *
 *      Ascii file format:                                                        *
 *      Var1/D:Var2/F:Var3/I ...                                                  *
 *      1.3    3.55 3.2 5    ...                                                  *
 *      ...                                                                       *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: TMVA_AsciiConverter.h,v 1.6 2006/04/29 23:55:41 andreas.hoecker Exp $       
 **********************************************************************************/

#ifndef ROOT_TMVA_AsciiConverter
#define ROOT_TMVA_AsciiConverter

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_AsciiConverter                                                  // 
//                                                                      //
// Converts ascii file into TTree object                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include <fstream>
#include "TObjString.h"

class TString;
class TArrayF;
class TArrayD;
class TArrayI;
class TTree;

class TMVA_AsciiConverter : public TObject {
  
 public:

  TMVA_AsciiConverter ( void );
  TMVA_AsciiConverter ( TString infile, TTree* theTree );
  virtual ~TMVA_AsciiConverter( void );
  
  void        ParseFormatLine( void );
  void        CreateBranches ( void );
  void        FillFromFile   ( void );
  
  TTree*      GetTree        ( void ) const { return fTree;       }
  Int_t       GetNBranch     ( void ) const { return fNbranch;    }
  TList*      GetFormatList  ( void ) const { return fFormatList; }
  TList*      GetLabelList   ( void ) const { return fLabelList;  }
  Int_t       GetNumInt      ( void ) const { return fInt_cnt;    }
  Int_t       GetNumDbl      ( void ) const { return fDbl_cnt;    }
  Int_t       GetNumFloat    ( void ) const { return fFloat_cnt;  }
  Bool_t      GetFileStatus  ( void ) const { return fFileStatus; }

 protected:

  void        SetInputFile( TString infile );

  Bool_t      IsInteger( TString* branchformat );
  Bool_t      IsDouble ( TString* branchformat );
  Bool_t      IsFloat  ( TString* branchformat );
  Bool_t      IsString ( TString* branchformat );


  Double_t*   fData_dbl; 
  Float_t*    fData_float;
  Int_t*      fData_int; 
  TObjString* fData_str[1000]; // pointer to TObjString branch
  
  Int_t       fDbl_cnt; 
  Int_t       fFloat_cnt;
  Int_t       fInt_cnt; 
  Int_t       fChar_cnt; 
  Int_t       fDbl_array_cnt;
  Int_t       fFloat_array_cnt;
  Int_t       fInt_array_cnt; 

  TList*      fFormatList; 
  TList*      fLabelList;  

  ifstream    fInfile; 
  Int_t       fNbranch; 
  Bool_t      fFileStatus; 
  TTree*      fTree;
  
  Bool_t      fParseFormatLineDone;
  Bool_t      fCreateBranchesDone; 
  Bool_t      fFillFromFileDone; 
 
  ClassDef(TMVA_AsciiConverter,0) // Converts ascii file into TTree object
};

#endif
