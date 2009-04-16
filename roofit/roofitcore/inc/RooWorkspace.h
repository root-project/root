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
#include <string>

class TClass ;
class RooAbsPdf ;
class RooAbsData ;
class RooRealVar ;
class RooCategory ;
class RooAbsReal ;
class RooAbsCategory ;
class RooFactoryWSTool ;
//class RooModelView ;

#include "TNamed.h"
#include "TDirectoryFile.h"

class RooWorkspace : public TNamed {
public:

  RooWorkspace() ;
  RooWorkspace(const char* name, Bool_t doCINTExport) ;
  RooWorkspace(const char* name, const char* title=0) ;
  RooWorkspace(const RooWorkspace& other) ;
  ~RooWorkspace() ;

  void exportToCint(const char* namespaceName=0) ;

  Bool_t importClassCode(const char* pat="*", Bool_t doReplace=kFALSE) ;
  Bool_t importClassCode(TClass* theClass, Bool_t doReplace=kFALSE) ;

  // Import functions for dataset, functions 
  Bool_t import(const RooAbsArg& arg, const RooCmdArg& arg1=RooCmdArg(),const RooCmdArg& arg2=RooCmdArg(),const RooCmdArg& arg3=RooCmdArg()) ;
  Bool_t import(const RooArgSet& args, const RooCmdArg& arg1=RooCmdArg(),const RooCmdArg& arg2=RooCmdArg(),const RooCmdArg& arg3=RooCmdArg()) ;
  Bool_t import(RooAbsData& data, const RooCmdArg& arg1=RooCmdArg(),const RooCmdArg& arg2=RooCmdArg(),const RooCmdArg& arg3=RooCmdArg()) ;

  // Transaction management interface for multi-step import operations
  Bool_t startTransaction() ;
  Bool_t cancelTransaction() ;
  Bool_t commitTransaction() ;

  // Named set management
  Bool_t defineSet(const char* name, const RooArgSet& aset, Bool_t importMissing=kFALSE) ;
  const RooArgSet* set(const char* name) ;

  // Import, load and save parameter value snapshots
  Bool_t saveSnapshot(const char* name, const RooArgSet& params, Bool_t importValues=kFALSE) ;
  Bool_t loadSnapshot(const char* name) ;  

  // Import other workspaces
  Bool_t merge(const RooWorkspace& other) ;
  Bool_t join(const RooWorkspace& other) ;

  // Accessor functions 
  RooAbsPdf* pdf(const char* name) ;
  RooAbsReal* function(const char* name) ;
  RooRealVar* var(const char* name) ;
  RooCategory* cat(const char* name) ;
  RooAbsCategory* catfunc(const char* name) ;
  RooAbsData* data(const char* name) ;
  RooAbsArg* arg(const char* name) ;
  RooAbsArg* fundArg(const char* name) ;
  TIterator* componentIterator() { return _allOwnedNodes.createIterator() ; }
  const RooArgSet& components() const { return _allOwnedNodes ; }

  Bool_t makeDir() ; 
  Bool_t cd(const char* path = 0) ;

  Bool_t writeToFile(const char* fileName, Bool_t recreate=kTRUE) ;

  // Tools management
  RooFactoryWSTool& factory() ;
  RooAbsArg* factory(const char* expr) ;

  // View management
//RooModelView* addView(const char* name, const RooArgSet& observables) ;
//RooModelView* view(const char* name) ;
//void removeView(const char* name) ;

  // Print function
  void Print(Option_t* opts=0) const ;

  static void autoImportClassCode(Bool_t flag) ;
 
  static void addClassDeclImportDir(const char* dir) ;
  static void addClassImplImportDir(const char* dir) ;
  static void setClassFileExportDir(const char* dir=0) ; 

  const TUUID& uuid() const { return _uuid ; }

  RooExpensiveObjectCache& expensiveObjectCache() { return _eocache ; }

  class CodeRepo : public TObject {
  public:
    CodeRepo(RooWorkspace* wspace=0) : _wspace(wspace), _compiledOK(kTRUE) {} ;
    virtual ~CodeRepo() {} ;

    Bool_t autoImportClass(TClass* tc, Bool_t doReplace=kFALSE) ;
    Bool_t compileClasses() ;

    Bool_t compiledOK() const { return _compiledOK ; }

    std::string listOfClassNames() const ;

    class ClassRelInfo {
    public:
      TString _baseName;      
      TString _fileBase ;
    } ;

    class ClassFiles {
    public:
      ClassFiles() : _extracted(kFALSE) {}
      TString _hext ;
      TString _hfile ;
      TString _cxxfile ;
      Bool_t _extracted ; 
    } ;    
    
  protected:
    RooWorkspace* _wspace ; // owning workspace
    std::map<TString,ClassRelInfo> _c2fmap ; // List of contained classes
    std::map<TString,ClassFiles> _fmap ; // List of contained files
    Bool_t _compiledOK ; //! Flag indicating that classes compiled OK

    ClassDef(CodeRepo,1) ; // Code repository for RooWorkspace
  } ;


  class WSDir : public TDirectoryFile {    
  public:
    WSDir(const char* name, const char* title, RooWorkspace* wspace) : 
      TDirectoryFile(name,title,"RooWorkspace::WSDir",0), 
      _wspace(wspace) 
      {
      }

    virtual ~WSDir() { Clear("nodelete") ; } ; 


#if ROOT_VERSION_CODE <= 332546
    virtual void Add(TObject*) ;
    virtual void Append(TObject*) ;
#else 
    virtual void Add(TObject*,Bool_t) ; 
    virtual void Append(TObject*,Bool_t) ; 
#endif 

  protected:
    friend class RooWorkspace ;
    void InternalAppend(TObject* obj) ;
    RooWorkspace* _wspace ; //! do not persist

    ClassDef(WSDir,1) ; // TDirectory representation of RooWorkspace
  } ;


 private:

  Bool_t isValidCPPID(const char* name) ;
  void exportObj(TObject* obj) ;
  void unExport() ;

  friend class CodeRepo ;
  static std::list<std::string> _classDeclDirList ;
  static std::list<std::string> _classImplDirList ;
  static std::string            _classFileExportDir ;

  TUUID       _uuid ;  // Unique workspace ID

  static Bool_t _autoClass ; // Automatic import of non-distribution class code
  
  CodeRepo _classes ; // Repository of embedded class code. This data member _must_ be first

  RooArgSet _allOwnedNodes ; // List of owned pdfs and components
  RooLinkedList _dataList ; // List of owned datasets
  RooLinkedList _views ; // List of model views  
  RooLinkedList _snapshots ; // List of parameter snapshots
  std::map<std::string,RooArgSet> _namedSets ; // Map of named RooArgSets

  WSDir* _dir ; //! Transient ROOT directory representation of workspace

  RooExpensiveObjectCache _eocache ; // Cache for expensive objects  

  RooFactoryWSTool* _factory ; //! Factory tool associated with workspace

  Bool_t      _doExport ;     //! Export contents of workspace to CINT?
  std::string _exportNSName ; //! Name of CINT namespace to which contents are exported

  Bool_t      _openTrans ;    //! Is there a transaction open?
  RooArgSet   _sandboxNodes ; //! Sandbox for incoming objects in a transaction

  ClassDef(RooWorkspace,5)  // Persistable project container for (composite) pdfs, functions, variables and datasets
  
} ;

#endif
