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

#include <memory>
#include <vector>

#include <TBrowser.h>
#include <TDirectory.h>
#include <TH1.h>
#include <TNamed.h>

#include <ROOT/RBrowseVisitor.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RNTuple.hxx>


/* Here is a description of how the entire TBrowser-support for ntuple works by explaining the consequence of each
 * action on a TBrowser.
 *
 * Opening the TBrowser:
 * Nothing happens in this class.
 *
 * Double-clicking a .root file containing a ntuple:
 * Still nothing happens in this class. A TDirectoryFile with the name of the ntuple is displayed on the TBrowser.
 *
 * Double-clicking a TDirectoryFile emerged from the situation above:
 * In TGFileBrowser an instance of RNTupleBrowser is created (if there isn't already one) and calls the functions
 * RNTupleBrower::SetDirectory() and RNTupleBrowser::Browse(). SetDirectory() ensures, that RNTupleBrowser creates
 * the correct instance of RNTupleReader to read its contents. Browse() uses the RNTupleReader to traverse through
 * the fields and instantiates a RNTupleFieldElement (field has no subfield) or RNTupleFieldElementFolder
 * (field has subfields) for the top level fields. Each RNTupleFieldElement and RNTupleFieldElementFolder then call
 * AddBrowse(), which adds them on the TBrowser with an icon and name.
 *
 * Double-clicking on a RNTupleFieldElement in TBrowser (field without child):
 * This calls the Browse() method of RNTupleFieldElement. This function draws a histogram of type TH1F for
 * non-vector fields containing numerical data and does nothing for other fields. If a field is suited for
 * drawing a TH1F is checked by fType, which is set when the RNTupleFieldElement is created.
 *
 * Double-clicking on a RNTupleFieldElementFolder in TBrowser (field with child, currently displayed as folder):
 * This calls the Browse() method of RNTupleFieldElementFolder, which then calls RFieldBase::TraverseVisitor()
 * (defined in RField.cxx) starting from the current field. This displays it's direct subfields on the TBrowser
 * with RNTupleFieldElementFolder::AddBrowse() via RBrowseVisitor::VisitField(). Again if the subfields contain
 * subfields themselves, they are displayed as RNTupleFieldElementFolder and if not as RNTupleFieldElement.
 * (Left-click on an element in TBrowser allows to see the class of a displayed object.)
 */

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RNTupleBrowser
\ingroup NTupleBrowse
\brief Coordinates the communication between TBrowser and RNTupleReader
    
.root files created by ntuples have a TDirectory. RNTupleBrowser reads this TDirectory and allows browsing through the contents of the .root file. To browse the contents, an instance of RNTupleReader is created, which traverses through the fields stored in the .root file. It also stores various objects which shouldn't be deleted until the TBrowser is closed, like the displayed histogram.
*/
// clang-format on
   
class RNTupleBrowser: public TNamed {
private:
   
   /// Holds the Tdirectory of the .root file, in which RNTuple is stored.
   TDirectory*                                  fDirectory;
   /// Keeps a list of previous directories. When a previously used directory appears again, the necessary fReaderPtr can be directly assigned from fReaderPtrVec instead of doing costly operations and creating a new one.
   std::vector<TDirectory*>                     fDirectoryVec;
   /// Holds the currently used RNTupleReader. It is changed when the folder directly beneath the .root file is double-clicked.
   std::shared_ptr<RNTupleReader>               fReaderPtr;
   // holds previously used RNTupleReader pointers. This allows the destructor for RNTupleReader to be called when the destructor for RNTupleBrowser is called. The problems when deleting a RNTupleReader earlier are:
   // 1. deleting RNTupleReader causes a chain reaction of destructors being called, which even deletes the currently displayed TH1F Histo. Instead of modifying code on other places which could influence other parts of the program, the problem was fixed here.
   // 2. Coming back to a field in rntuple after a field from a different ntuple and not calling SetDirectory(TDirectory*) results in segmentation fault, if the associated RNTupleReader is deleted.
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
   
   /// Instantiates the RNTupleReader associated to the TDirectory.
   void SetDirectory(TDirectory* directory);
   /// Called when double-clicking the folder which appears afer double-clicking the .root file.
   void Browse(TBrowser *b);
   RNTupleReader* GetReaderPtr() const { return fReaderPtr.get(); }
   int GetfUnitTest() const { return fUnitTest; }
   void IncreasefUnitTest() { ++fUnitTest; }
   Bool_t IsFolder() const { return kTRUE; }
   
   ClassDef(RNTupleBrowser,2)
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleFieldElementFolder
\ingroup NTupleBrowse
\brief Is displayed in the TBrowser as a field with children.
 
It represents an ntuple-field which has children. It is displayed in the TBrowser with a folder symbol. The subfields can be displayed by double-clicking this object, which calls void Browse(TBrowser *b)
*/
// clang-format on

class RNTupleFieldElementFolder: public TNamed {
private:
   /// Pointer to the field it represents.
   const Detail::RFieldBase*  fFieldPtr;
   /// Pointer to the instance of RNTupleBrowser which created it through RBrowseVisitor::VisitField().
   RNTupleBrowser*            fRNTupleBrowserPtr;
   /// RNTupleReader used for reading fFieldPtr's subfields.
   RNTupleReader*             fNtupleReaderPtr;
public:
   // ClassDef requires a constructor which can be called without any arguments. The constructor is never called with default arguments.
   RNTupleFieldElementFolder(std::string name = "DummyName", const Detail::RFieldBase* FieldPtr = nullptr, RNTupleBrowser* ntplbPtr = nullptr): fFieldPtr{FieldPtr}, fRNTupleBrowserPtr{ntplbPtr}, fNtupleReaderPtr{ntplbPtr->GetReaderPtr()} {
      fName = name;
   }
   
   std::string GetFieldName() const { return static_cast<std::string>((const char*)fName); }
   
   /// Displays the field it represents in TBrowser. Called in RBrowseVisitor::VisitField()
   void AddBrowse(TBrowser *b);
   /// Displays subfields in TBrowser. Called when double-clicked.
   void Browse(TBrowser *b);
   
   
   Bool_t IsFolder() const { return true; }
   ClassDef(RNTupleFieldElementFolder, 2)
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleFieldElement
\ingroup NTupleBrowse
\brief Is displayed in the TBrowser as a field without children.
    
It represents an ntuple-field without children. If it represents a field with numerical data, double-clicking creates a TH1F-histogram.
*/
// clang-format on
   
class RNTupleFieldElement: public TNamed {
private:
   /// Data type of the field it represents.
   fieldDatatype     fType;
   /// Pointer to the instance of RNTupleBrowser which created it through RBrowseVisitor::VisitField().
   RNTupleBrowser*   fRNTupleBrowserPtr;
   /// RNTupleReader used for reading fFieldPtr's subfields.
   RNTupleReader*    fReaderPtr;
   
   /// Called by Browse(TBrowser *b). It is responsible for drawing the TH1F histogram.
   template <typename T>
   void TemplatedBrowse(bool isIntegraltype);
   
public:
   // ClassDef requires a constructor which can be called without any arguments.
   RNTupleFieldElement(std::string name = "DummyField", RNTupleBrowser* ntplb = nullptr, fieldDatatype numType = fieldDatatype_nonNumeric): fType{numType}, fRNTupleBrowserPtr{ntplb} {
      fName = TString(name);
      fReaderPtr = fRNTupleBrowserPtr->GetReaderPtr();
   }
   
   fieldDatatype GetType() const { return fType; }
   
   /// Displays the field it represents in TBrowser. Called in RBrowseVisitor::VisitField()
   void AddBrowse(TBrowser *b);
   /// Displays a TH1F for certain values of fType in TBrowser. Called when double-clicked.
   void Browse(TBrowser *b);
   
   Bool_t IsFolder() const { return false; }
   ClassDef(RNTupleFieldElement, 2)
};
   
} // namespace Experimental
} // namespace ROOT



#endif /* ROOT7_RNTupleBrowser */


