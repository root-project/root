/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooWorkspace.h,v 1.3 2007/07/16 21:04:28 wouter Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_WORKSPACE
#define ROO_WORKSPACE

#include "RooPrintable.h"
#include "RooArgSet.h"
#include "RooLinkedList.h"
#include "RooCmdArg.h"
#include "RooExpensiveObjectCache.h"
#include "TUUID.h"
#include <map>
#include <list>
#include <memory>
#include <string>

class TClass ;
class RooAbsPdf ;
class RooAbsData ;
class RooRealVar ;
class RooCategory ;
class RooAbsReal ;
class RooAbsCategory ;
class RooFactoryWSTool ;
class RooAbsStudy ;

#include "TNamed.h"
#include "TDirectoryFile.h"

class RooWorkspace : public TNamed {
public:

  RooWorkspace() ;
  RooWorkspace(const char* name, bool doCINTExport)
  R__SUGGEST_ALTERNATIVE("The \"doCINTExprot\" argument has no effect anymore since ROOT 6."
          "Consider using RooWorkspace(const char* name, const char* title=nullptr).");
  RooWorkspace(const char* name, const char* title=nullptr) ;
  RooWorkspace(const RooWorkspace& other) ;
  ~RooWorkspace() override ;

  bool importClassCode(const char* pat="*", bool doReplace=false) ;
  bool importClassCode(TClass* theClass, bool doReplace=false) ;

  // Import functions for dataset, functions, generic objects
  bool import(const RooAbsArg& arg,
      const RooCmdArg& arg1=RooCmdArg(),const RooCmdArg& arg2=RooCmdArg(),const RooCmdArg& arg3=RooCmdArg(),
      const RooCmdArg& arg4=RooCmdArg(),const RooCmdArg& arg5=RooCmdArg(),const RooCmdArg& arg6=RooCmdArg(),
      const RooCmdArg& arg7=RooCmdArg(),const RooCmdArg& arg8=RooCmdArg(),const RooCmdArg& arg9=RooCmdArg()) ;
  bool import(const RooArgSet& args,
      const RooCmdArg& arg1=RooCmdArg(),const RooCmdArg& arg2=RooCmdArg(),const RooCmdArg& arg3=RooCmdArg(),
      const RooCmdArg& arg4=RooCmdArg(),const RooCmdArg& arg5=RooCmdArg(),const RooCmdArg& arg6=RooCmdArg(),
      const RooCmdArg& arg7=RooCmdArg(),const RooCmdArg& arg8=RooCmdArg(),const RooCmdArg& arg9=RooCmdArg()) ;
  bool import(RooAbsData& data,
      const RooCmdArg& arg1=RooCmdArg(),const RooCmdArg& arg2=RooCmdArg(),const RooCmdArg& arg3=RooCmdArg(),
      const RooCmdArg& arg4=RooCmdArg(),const RooCmdArg& arg5=RooCmdArg(),const RooCmdArg& arg6=RooCmdArg(),
      const RooCmdArg& arg7=RooCmdArg(),const RooCmdArg& arg8=RooCmdArg(),const RooCmdArg& arg9=RooCmdArg()) ;
  bool import(const char *fileSpec,
      const RooCmdArg& arg1=RooCmdArg(),const RooCmdArg& arg2=RooCmdArg(),const RooCmdArg& arg3=RooCmdArg(),
      const RooCmdArg& arg4=RooCmdArg(),const RooCmdArg& arg5=RooCmdArg(),const RooCmdArg& arg6=RooCmdArg(),
      const RooCmdArg& arg7=RooCmdArg(),const RooCmdArg& arg8=RooCmdArg(),const RooCmdArg& arg9=RooCmdArg()) ;
  bool import(TObject& object, bool replaceExisting=false) ;
  bool import(TObject& object, const char* aliasName, bool replaceExisting=false) ;

  // Transaction management interface for multi-step import operations
  bool startTransaction() ;
  bool cancelTransaction() ;
  bool commitTransaction() ;

  // Named set management
  bool defineSet(const char* name, const RooArgSet& aset, bool importMissing=false) ;
  bool defineSet(const char* name, const char* contentList) ;
  bool extendSet(const char* name, const char* newContents) ;
  bool renameSet(const char* name, const char* newName) ;
  bool removeSet(const char* name) ;
  const RooArgSet* set(const char* name) ;

  // Import, load and save parameter value snapshots
  bool saveSnapshot(const char* name, const char* paramNames) ;
  bool saveSnapshot(const char* name, const RooArgSet& params, bool importValues=false) ;
  bool loadSnapshot(const char* name) ;
  const RooArgSet* getSnapshot(const char* name) const ;

  // Retrieve list of parameter snapshots
  RooLinkedList getSnapshots(){ return _snapshots; }

  void merge(const RooWorkspace& /*other*/) {} ;

  // Accessor functions
  RooAbsPdf* pdf(RooStringView name) const ;
  RooAbsReal* function(RooStringView name) const ;
  RooRealVar* var(RooStringView name) const ;
  RooCategory* cat(RooStringView name) const ;
  RooAbsCategory* catfunc(RooStringView name) const ;
  RooAbsData* data(RooStringView name) const ;
  RooAbsData* embeddedData(RooStringView name) const ;
  RooAbsArg* arg(RooStringView name) const ;
  RooAbsArg* fundArg(RooStringView name) const ;
  RooArgSet argSet(RooStringView nameList) const ;
  TIterator* componentIterator() const
  R__SUGGEST_ALTERNATIVE("Better iterate over RooWorkspace::components() with range-based loop instead of using RooWorkspace::componentIterator().")
  { return _allOwnedNodes.createIterator() ; }
  const RooArgSet& components() const { return _allOwnedNodes ; }
  TObject* genobj(RooStringView name) const ;
  TObject* obj(RooStringView name) const ;

  // Group accessors
  RooArgSet allVars() const;
  RooArgSet allCats() const ;
  RooArgSet allFunctions() const ;
  RooArgSet allCatFunctions() const ;
  RooArgSet allPdfs() const ;
  RooArgSet allResolutionModels() const ;
  std::list<RooAbsData*> allData() const ;
  std::list<RooAbsData*> allEmbeddedData() const ;
  std::list<TObject*> allGenericObjects() const ;

  bool makeDir() ;
  bool cd(const char* path = 0) ;

  bool writeToFile(const char* fileName, bool recreate=true) ;

  /// Make internal collection use an unordered_map for
  /// faster searching. Important when large trees are
  /// imported / or modified in the workspace.
  /// Note that RooAbsCollection may eventually switch
  /// this on by itself.
  void useFindsWithHashLookup(bool flag) {
    _allOwnedNodes.useHashMapForFind(flag);
  }

  void RecursiveRemove(TObject *obj) override;

  // Tools management
  RooFactoryWSTool& factory() ;
  RooAbsArg* factory(RooStringView expr) ;

  // RooStudyManager modules
  bool addStudy(RooAbsStudy& study) ;
  TIterator* studyIterator() { return _studyMods.MakeIterator() ; }
  void clearStudies() ;

  // Print function
  void Print(Option_t* opts=nullptr) const override ;

  static void autoImportClassCode(bool flag) ;

  static void addClassDeclImportDir(const char* dir) ;
  static void addClassImplImportDir(const char* dir) ;
  static void setClassFileExportDir(const char* dir=nullptr) ;

  const TUUID& uuid() const { return _uuid ; }

  RooExpensiveObjectCache& expensiveObjectCache() { return _eocache ; }

  class CodeRepo : public TObject {
  public:
    CodeRepo(RooWorkspace* wspace=nullptr) : _wspace(wspace), _compiledOK(true) {} ;

    CodeRepo(const CodeRepo& other, RooWorkspace* wspace=nullptr) : TObject(other) ,
          _wspace(wspace?wspace:other._wspace),
          _c2fmap(other._c2fmap),
          _fmap(other._fmap),
          _ehmap(other._ehmap),
          _compiledOK(other._compiledOK) {} ;

    ~CodeRepo() override {} ;

    bool autoImportClass(TClass* tc, bool doReplace=false) ;
    bool compileClasses() ;

    bool compiledOK() const { return _compiledOK ; }

    std::string listOfClassNames() const ;



    class ClassRelInfo {
    public:
      TString _baseName;
      TString _fileBase ;
    } ;

    class ClassFiles {
    public:
      ClassFiles() : _extracted(false) {}
      TString _hext ;
      TString _hfile ;
      TString _cxxfile ;
      bool _extracted ;
    } ;


    class ExtraHeader {
    public:
      TString _hname ;
      TString _hfile ;
   } ;

  protected:
    RooWorkspace* _wspace ; // owning workspace
    std::map<TString,ClassRelInfo> _c2fmap ; // List of contained classes
    std::map<TString,ClassFiles> _fmap ; // List of contained files
    std::map<TString,ExtraHeader> _ehmap ; // List of extra header files
    bool _compiledOK ; //! Flag indicating that classes compiled OK

    ClassDefOverride(CodeRepo,2) ; // Code repository for RooWorkspace
  } ;


  class WSDir : public TDirectoryFile {
  public:
    WSDir(const char* name, const char* title, RooWorkspace* wspace) :
      TDirectoryFile(name,title,"RooWorkspace::WSDir",0),
      _wspace(wspace)
      {
      }

    ~WSDir() override { Clear("nodelete") ; } ;


    void Add(TObject*,bool) override ;
    void Append(TObject*,bool) override ;

  protected:
    friend class RooWorkspace ;
    void InternalAppend(TObject* obj) ;
    RooWorkspace* _wspace ; //! do not persist

    ClassDefOverride(WSDir,1) ; // TDirectory representation of RooWorkspace
  } ;


 private:
    friend class RooAbsArg;
    friend class RooAbsPdf;
    friend class RooConstraintSum;
    bool defineSetInternal(const char *name, const RooArgSet &aset);

    friend class CodeRepo;
    static std::list<std::string> _classDeclDirList;
    static std::list<std::string> _classImplDirList;
    static std::string _classFileExportDir;

    TUUID _uuid; // Unique workspace ID

    static bool _autoClass; // Automatic import of non-distribution class code

    CodeRepo _classes; // Repository of embedded class code. This data member _must_ be first

    RooArgSet _allOwnedNodes;                    ///< List of owned pdfs and components
    RooLinkedList _dataList;                     ///< List of owned datasets
    RooLinkedList _embeddedDataList;             ///< List of owned datasets that are embedded in pdfs
    RooLinkedList _views;                        ///< List of model views
    RooLinkedList _snapshots;                    ///< List of parameter snapshots
    RooLinkedList _genObjects;                   ///< List of generic objects
    RooLinkedList _studyMods;                    ///< List if StudyManager modules
    std::map<std::string, RooArgSet> _namedSets; ///< Map of named RooArgSets

    WSDir *_dir = nullptr; ///<! Transient ROOT directory representation of workspace

    RooExpensiveObjectCache _eocache; ///< Cache for expensive objects

    std::unique_ptr<RooFactoryWSTool> _factory; ///<! Factory tool associated with workspace

    bool _doExport;          ///<! Export contents of workspace to CINT?
    std::string _exportNSName; ///<! Name of CINT namespace to which contents are exported

    bool _openTrans = false; ///<! Is there a transaction open?
    RooArgSet _sandboxNodes; ///<! Sandbox for incoming objects in a transaction

    ClassDefOverride(RooWorkspace, 8) // Persistable project container for (composite) pdfs, functions, variables and datasets
} ;

#endif
