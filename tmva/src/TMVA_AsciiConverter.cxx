// @(#)root/tmva $Id: TMVA_AsciiConverter.cxx,v 1.3 2006/05/08 17:56:50 brun Exp $
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
 **********************************************************************************/

//_______________________________________________________________________
//
// Converts ascii file into TTree object
//
//_______________________________________________________________________

#include "TMVA_AsciiConverter.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TBrowser.h"
#include "Riostream.h"
#include "TString.h"
#include <stdlib.h>

#define DEBUG_TMVA_AsciiConverter kFALSE

ClassImp(TMVA_AsciiConverter)

//_______________________________________________________________________
TMVA_AsciiConverter::TMVA_AsciiConverter( void )
  : fData_dbl(0),
    fData_float(0),
    fData_int(0),
    fFormatList(0),
    fLabelList(0),
    fFileStatus(0),
    fTree(0)
{
  // initialize counters and boolean tags
  fParseFormatLineDone = fCreateBranchesDone = fFillFromFileDone = kFALSE;
  fDbl_cnt = fInt_cnt = fFloat_cnt = fChar_cnt = 0;
}

//_______________________________________________________________________
TMVA_AsciiConverter::TMVA_AsciiConverter( TString infile, TTree* theTree )
  : fData_dbl(0),
    fData_float(0),
    fData_int(0),
    fFormatList(0),
    fLabelList(0),
    fFileStatus(0)
{
  // get the TTree object (that TMVA_AsciiConverter shall not own)
  fTree = theTree;

  SetInputFile( infile );

  // initialize counters and boolean tags
  fParseFormatLineDone = fCreateBranchesDone = fFillFromFileDone = kFALSE;
  fDbl_cnt = fInt_cnt = fFloat_cnt = fChar_cnt = 0;

  // create the tree
  ParseFormatLine();
  CreateBranches();
  FillFromFile();
}

//_______________________________________________________________________
TMVA_AsciiConverter::~TMVA_AsciiConverter( void )
{
  // Default destructor for TMVA_AsciiConverter
  // delete all objects created with the tree

  delete [] fData_dbl;
  delete [] fData_int;
  for (Int_t i=0; i<fChar_cnt; i++) fData_str[i]->Delete();
  fFormatList->Delete();
  fLabelList ->Delete();
}

//_______________________________________________________________________
void TMVA_AsciiConverter::SetInputFile( TString infile )
{
  // Set input ASCII file.  This only has to be called if the default
  // constructor was used (input file not specified).

  fInfile.open(infile);
  fFileStatus = fInfile.is_open();
}

//_______________________________________________________________________
void TMVA_AsciiConverter::CreateBranches( void )
{
  // Create tree branches; ParseFormatLine() must have been run first
  if (!fParseFormatLineDone) {
    cout << "--- " << GetName() << ": "
         << "You have not read the file header with ParseFormatLine(). Exiting."
         << endl;
    exit(1);
  }
  else {
    fData_dbl   = new Double_t[fNbranch];
    fData_int   = new Int_t[fNbranch];
    fData_float = new Float_t[fNbranch];

    TString *branchlabel  = new TString();
    TString *branchformat = new TString();

    for (Int_t i=0; i<fNbranch; i++) {

      branchlabel ->Append(((TObjString*)fLabelList ->At(i))->String());
      branchformat->Append(((TObjString*)fFormatList->At(i))->String());

      if (IsDouble(branchformat)) {
        fTree->Branch(branchlabel->Data(),&fData_dbl[fDbl_cnt],
                            branchformat->Data());
        fDbl_cnt++;
      }
      else if (IsFloat(branchformat)) {
        fTree->Branch(branchlabel->Data(),&fData_float[fFloat_cnt],
                            branchformat->Data());
        fFloat_cnt++;
      }
      else if (IsInteger(branchformat)) {
        fTree->Branch(branchlabel->Data(),&fData_int[fInt_cnt],
                            branchformat->Data());
        fInt_cnt++;
      }
      else if (IsString(branchformat)) {
        fData_str[fChar_cnt] = new TObjString();
        fTree->Branch(branchlabel->Data(),"TObjString",
                            &fData_str[fChar_cnt],32000,99);
        fChar_cnt++;
      }
      else {
        cout << "--- " << GetName() << ": "
             << "Error in format string: " << branchformat->Data()
             << " ... Exiting from TMVA_AsciiConverter::CreateBranches()" << endl;
        return;
      }

      branchlabel->Resize(0); branchformat->Resize(0);
    }
    delete branchlabel;
    delete branchformat;
    fCreateBranchesDone = kTRUE; // Branch creation complete!
  }
}

//_______________________________________________________________________
void TMVA_AsciiConverter::FillFromFile( void )
{
  // Fill the tree from data in file.
  // This is called after ParseFormat() and CreateBranches().

  // Check to see if the header has been parsed and the branches created.
  if (!fParseFormatLineDone || !fCreateBranchesDone) {
    cout << "--- " << GetName() << ": "
        << "Either you haven't read the file header with ParseFormatLine() "
        << "or you haven't created the branches with CreateBranches() or both!  Exiting." << endl;
    exit(1);
  }
  else {
    //TString* branchFormats[fNbranch];
    TString** branchFormats = new TString*[fNbranch];

    Int_t    char_cnt=0, int_cnt=0, dbl_cnt=0, float_cnt=0;
    char     buffer[10000];
    Double_t buf_dbl;
    Float_t  buf_float;
    Int_t    buf_int,i,j;

    for (i=0; i<fNbranch; i++)
      branchFormats[i] = new TString(((TObjString*) fFormatList->At(i))->String());

    while (!fInfile.eof()) {

      for (i=0; i<fNbranch; i++) {
       for (j=0; j<100; j++)
         buffer[j] = ' ';
       if (IsString(branchFormats[i])) {
         fInfile >> buffer;
         fData_str[char_cnt]->SetString(buffer);
         char_cnt++;
       }
       else if (IsInteger(branchFormats[i])) {
         fInfile >> buf_int;
         fData_int[int_cnt] = buf_int;
         int_cnt++;
       }
       else if (IsFloat(branchFormats[i])) {
         fInfile >> buf_float;
         fData_float[float_cnt] = buf_float;
         float_cnt++;
       }
       else if (IsDouble(branchFormats[i])) {
         fInfile >> buf_dbl;
         fData_dbl[dbl_cnt] = buf_dbl;
         dbl_cnt++;
       }
       else {
         cout << "--- " << GetName() << ": "
              << "Invalid format string! exit(1)" << endl;
         return;
       }
      }

      fTree->Fill();

      // reset counters
      char_cnt = int_cnt = dbl_cnt = float_cnt = 0;

    }
    fInfile.close();

    // delete temporary objects
    for (Int_t i=0; i<fNbranch; i++) delete branchFormats[i];

    fFillFromFileDone = kTRUE; // Filling from file complete!
  }
}

//_______________________________________________________________________
Bool_t TMVA_AsciiConverter::IsInteger( TString* branchformat )
{
  // Test to see if branchformat string contains a "/I".
  return (branchformat->Contains("/B") || branchformat->Contains("/b") ||
         branchformat->Contains("/I") || branchformat->Contains("/i")) ? kTRUE : kFALSE;
}

//_______________________________________________________________________
Bool_t TMVA_AsciiConverter::IsDouble( TString* branchformat )
{
  // Test to see if branchformat string contains a "/D".
  return (branchformat->Contains("/D")) ? kTRUE : kFALSE;
}

//_______________________________________________________________________
Bool_t TMVA_AsciiConverter::IsFloat( TString* branchformat )
{
  // Test to see if branchformat string contains a "/F".
  return (branchformat->Contains("/F")) ? kTRUE : kFALSE;
}

//_______________________________________________________________________
Bool_t TMVA_AsciiConverter::IsString( TString* branchformat )
{
  // Test to see if branchformat string contains a "/S".
  return (branchformat->Contains("/S")) ? kTRUE : kFALSE;
}

//_______________________________________________________________________
void TMVA_AsciiConverter::ParseFormatLine( void )
{
  // Parse the file header and place branch information into fFormatList
  // and fLabelList.  I'm sure this can be done in two lines, but I'm
  // not that good.

  fFormatList          = new TList();
  fLabelList           = new TList();

  TString* formatstring = new TString();
  TString* format       = new TString();
  TString* label        = new TString();
  Int_t    j = 0;

  formatstring->ReadLine(fInfile); // read format line from file

  Int_t n = formatstring->Length();
  //TObjString* format_obj[n]; // array of formats (e.g. 'blah/D')
  //TObjString* label_obj[n];  // array of labels (e.g. 'blah')
  TObjString* format_obj[1000]; // please check
  TObjString* label_obj[1000];  // please check

  for (Int_t i=0; i<n; i++) {
    format->Append((*formatstring)(i));
    label->Append((*formatstring)(i));
    if ((*formatstring)(i)==':') {
      format->Chop();
      label->Chop();
      label->Chop();
      label->Chop();
      format_obj[j] = new TObjString(format->Data());
      label_obj[j]  = new TObjString(label->Data());
      fFormatList->Add(format_obj[j]);
      fLabelList->Add(label_obj[j]);
      format->Resize(0); label->Resize(0);
      j++;
    }
    if (i==(n-1)) {
      label->Chop();label->Chop();
      format_obj[j] = new TObjString(format->Data());
      label_obj[j]  = new TObjString(label->Data());
      fFormatList->Add(format_obj[j]);
      fLabelList->Add(label_obj[j]);
      format->Resize(0); label->Resize(0);
      j++;
    }
  }
  if (DEBUG_TMVA_AsciiConverter) fLabelList->Print();
  fNbranch = j; // Set number of branches

  // delete temporary objects
  delete formatstring;
  delete format;
  delete label;

  fParseFormatLineDone = kTRUE; // Header parsing completed!
}

