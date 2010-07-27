// @(#)root/graf2d:$Id$
// Author: Claudi Martinez, July 19th 2010

/*************************************************************************
 * Copyright (C) 1995-2010, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFITS
#define ROOT_TFITS

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFITS                                                                //
//                                                                      //
// Interface to FITS astronomical files.                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TMatrixDfwd
#include "TMatrixDfwd.h"
#endif
#ifndef ROOT_TVectorDfwd
#include "TVectorDfwd.h"
#endif

class TArrayI;
class TArrayD;
class TH1;
class TASImage;
class TImagePalette;


class TFITSHDU : public TNamed {

private:
   void _release_resources();
   void _initialize_me();

public:
   enum EHDUTypes {         // HDU types
      kImageHDU,
      kTableHDU
   };

   struct HDURecord {       // FITS HDU record
      TString fKeyword;
      TString fValue;
      TString fComment;
   };

protected:
   TString           fCleanFilePath;    // Path to HDU's file (without filter)
   struct HDURecord *fRecords;          // HDU metadata records
   Int_t             fNRecords;         // Number of records
   enum EHDUTypes    fType;             // HDU type
   TString           fExtensionName;    // Extension Name
   Int_t             fNumber;           // HDU number (1=PRIMARY)
   TArrayI          *fSizes;            // Image sizes in each dimension (when fType == kImageHDU)
   TArrayD          *fPixels;           // Image pixels (when fType == kImageHDU)

   Bool_t            LoadHDU(TString& filepath_filter);
   static void       CleanFilePath(const char *filepath_with_filter, TString &dst);
   void              PrintHDUMetadata(const Option_t *opt="") const;
   void              PrintFileMetadata(const Option_t *opt="") const;

public:
   TFITSHDU(const char *filepath_with_filter);
   TFITSHDU(const char *filepath, Int_t extension_number);
   TFITSHDU(const char *filepath, const char *extension_name);
   ~TFITSHDU();

   Int_t              GetRecordNumber() const { return fNRecords; }
   struct HDURecord  *GetRecord(const char *keyword);
   TString&           GetKeywordValue(const char *keyword);
   void               Print(const Option_t *opt="") const;

   //Image readers
   TH1               *ReadAsHistogram();
   TASImage          *ReadAsImage(Int_t layer = 0, TImagePalette *pal = 0);
   TMatrixD          *ReadAsMatrix(Int_t layer = 0);
   TVectorD          *GetArrayRow(UInt_t row);
   TVectorD          *GetArrayColumn(UInt_t col);

   //Table readers
   //TODO

   ClassDef(TFITSHDU,1)  // Class interfacing FITS HDUs
};


#endif
