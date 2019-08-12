/// \file ROOT/RNTupleBrowser.hxx
/// \ingroup NTupleBrowse ROOT7
/// \author Simon Leisibach <simon.satoshi.rene.leisibach@cern.ch>
/// \date 2019-07-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleBrowser
#define ROOT7_RNTupleBrowser

#include <ROOT/RBrowseVisitor.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RNTuple.hxx>

#include <TBrowser.h>
#include <TDirectory.h>
#include <TH1.h>
#include <TNamed.h>

#include <memory>
#include <vector>

/* Description of how the entire TBrowser-support for ntuple works: The consequence of each click on a TBrowser is
 * explained here.
 *
 * Opening the TBrowser:
 * Nothing happens in this class.
 *
 * Double-clicking a .root file containing a ntuple:
 * Still nothing happens in this class. A TDirectoryFile with the name of the ntuple is displayed on the TBrowser.
 *
 * Double-clicking a TDirectoryFile emerged from the situation just above:
 * In TGFileBrowser an instance of RNTupleBrowser is created (if there isn't already one) and calls the functions
 * RNTupleBrower::SetDirectory() and RNTupleBrowser::Browse(). SetDirectory() makes sure, that RNTupleBrowser creates
 * the correct instance of RNTupleReader to read its contents. Browse() uses the RNTupleReader to traverse through the
 * fields and instantiates a RNTupleFieldElement (Field has no child) or RNTupleFieldElementFolder (Field has child)
 * for the top level fields in RBrowseVisitor.cxx. RNTupleFieldElement and RNTupleFieldElementFolder then call
 * AddBrowse(), which adds them on the TBrowser with an icon and name.
 *
 * Clicking on an RNTupleFieldElement in TBrowser (field without child):
 * This calls the Browse() method of RNTupleFieldElement. This function draws a histogram of type TH1F for suited fields
 * and does nothing for unsuited fields. If a field is suited for drawing is checked by fType, which is set when the
 * RNTupleFieldElement is created. To check in which cases a histogram is drawn, see RNTupleFieldElement::Browse().
 *
 * Clicking on RNTupleFieldElementFolder in TBrowser (field with child, currently displayed as folder):
 * This calls the TraverseVisitor() function (defined in RField.cxx) starting from the current field and displays it's
 * direct child fields on the TBrowser via the AddBrowse() function in RNTupleBrowser.cxx via RBrowseVisitor.cxx.
 * Again if the child fields contain child fields themselves, they are displayed as RNTupleFieldElementFolder and if not as
 * RNTupleFieldElement. (Left-click on an element in TBrowser allows to see its class.)
 */

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RNTupleBrowser
\ingroup NTupleBrowser
\brief Coordinates the communication between TBrowser and RNTupleReader
    
.root files created by ntuples have a TDirectory. RNTupleBrowser reads this TDirectory and allows browsing through the contents of the .root file. To browse the contents, an instance of RNTupleReader is created, which traverses through the fields stored in the .root file. It also stores various objects like the displayed histogram which shouldn't be deleted until the TBrowser is closed.
*/
// clang-format on
   
class RNTupleBrowser: public TNamed {
private:
   
   /// Holds the Tdirectory of the .root file, in which RNTuple is stored.
   TDirectory*                                  fDirectory;
   /// Keeps a list of previous directories. So when a previously used directory appears again, the necessary fReaderPtr can be directly assigned from fReaderPtrVec instead of doing costly operations and creating a new one.
   std::vector<TDirectory*>                     fDirectoryVec;
   /// Holds the currently used RNTupleReader. It is changed when the folder directly beneath the .root file is doubleclicked.
   std::shared_ptr<RNTupleReader>               fReaderPtr;
   // holds previously used RNTupleReader pointers. This allows the destructor for RNTupleReader to be called when the destructor for RNTupleBrowser is called. The problems when deleting a RNTupleReader earlier are:
   // 1. deleting RNTupleReader causes a chain reaction of destrors being called, which even deletes the currently displayed TH1F Histo. Instead of modifying code on other places which could influence other parts of the program, the problem was fixed here.
   // 2. Coming back to a field in rntuple after a field from a different ntuple and not calling SetDirectory(TDirectory*) results in segmentation faults in any possible way, if the associated RNTupleReader is deleted.
   /// Stores pointers of created RNTupleReaders so they can be deleted in ~RNTupleBrowser.
   std::vector<std::shared_ptr<RNTupleReader>>  fReaderPtrVec;
   // For normal use the value is always set to 0. In case of a unit test, the value is >0 and can give feedback to the unittest if something worked or not. For details, see test/ntuplebrowse.cxx
   /// Used for testing. Should always be 0 during normal use.
   int                                          fUnitTest = 0;
   
public:
   // Allows to keep created TH1F histo outside created function. Else user gets to see the histogram for less than a second. Only points to histograms created through RNTupleElementField.
   /// Points to currently displayed histogram.
   TH1F*                                        fCurrentTH1F;
   // Keeps hold of pointers of dynamically allocated objects representing rntuple-fields in. They are all deleted in one step in the destructor. Their job is to prevent memory leak. Deleting these object before calling the destructor would make them uncklickable in TBrowser.
   /// Stores instances of RNTupleFieldElement and RNTupleFieldElementFolder to prevent memory leak.
   std::vector<TNamed*>                         fNTupleBrowsePtrVec;
   
   
   RNTupleBrowser(TDirectory* directory=nullptr, int fUnitTest = 0);
   ~RNTupleBrowser();
   
   void SetDirectory(TDirectory* directory);
   // Called when double-clicking the folder which appears afer double-clicking the .root file.
   void Browse(TBrowser *b);
   RNTupleReader* GetReaderPtr() const { return fReaderPtr.get();}
   int GetfUnitTest() const { return fUnitTest; }
   void IncreasefUnitTest() { ++fUnitTest; }
   Bool_t IsFolder() const { return kTRUE; }
   
   ClassDef(RNTupleBrowser,2)
};
   
// This class makes sure, that fields with children will be shown with folder icons. By using a folder icon for certain fields, the user knows that more fields are located under the field.
class RNTupleFieldElementFolder: public TNamed {
private:
   const Detail::RFieldBase*  fFieldPtr;
   std::string                fFieldName;
   /// Pointer to the instance of RNTupleBrowser which created it through RBrowseVisitor::VisitField().
   RNTupleBrowser*            fRNTupleBrowserPtr;
   RNTupleReader*             fNtupleReaderPtr;
public:
   RNTupleFieldElementFolder(std::string name = "TestField") { fName = name; fFieldPtr = nullptr;}
   
   void SetRNTupleBrowser(RNTupleBrowser* ntplbPtr) { fRNTupleBrowserPtr = ntplbPtr; fNtupleReaderPtr = ntplbPtr->GetReaderPtr(); }
   void SetField(const Detail::RFieldBase &field) { fFieldName = field.GetName(); fFieldPtr = &field;}
   std::string GetFieldName() const { return fFieldName; }
   // Called when double-clicking the field in TBrowser
   /// why templated MockTBrowser should support Add method.
   void Browse(TBrowser *b);
   void AddBrowse(TBrowser *b);
   
   Bool_t IsFolder() const { return true; }
   ClassDef(RNTupleFieldElementFolder, 2)
};

   
// Used to display RNTuple Fields in the TBrowser which don't have children. Unlike the class above they don't show a folder icon.
class RNTupleFieldElement: public TNamed {
public:
   RNTupleReader* fReaderPtr;
   RNTupleBrowser* fRNTupleBrowserPtr;
   numericDatatype fType;
   
   RNTupleFieldElement(std::string name = "TestField", RNTupleBrowser* ntplb = nullptr, numericDatatype numType = numericDatatype_nonNumeric): fRNTupleBrowserPtr{ntplb}, fType{numType} { fName = TString(name); fReaderPtr = fRNTupleBrowserPtr->GetReaderPtr();}
   
   void Browse(TBrowser *b);
   void AddBrowse(TBrowser *b);
   template <typename T>
   void TemplatedBrowse(bool isIntegraltype);
   
   Bool_t IsFolder() const { return false; }
   ClassDef(RNTupleFieldElement, 2)
};
   
} // namespace Experimental
} // namespace ROOT



#endif /* ROOT7_RNTupleBrowser */


