/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// The RooWorkspace is a persistable container for RooFit projects. A workspace
// can contain and own variables, p.d.f.s, functions and datasets. All objects
// that live in the workspace are owned by the workspace. The import() method
// enforces consistency of objects upon insertion into the workspace (e.g. no
// duplicate object with the same name are allowed) and makes sure all objects
// in the workspace are connected to each other. Easy accessor methods like
// pdf(), var() and data() allow to refer to the contents of the workspace by
// object name. The entire RooWorkspace can be saved into a ROOT TFile and organises
// the consistent streaming of its contents without duplication.
// <p>
// If a RooWorkspace contains custom classes, i.e. classes not in the 
// ROOT distribution, portability of workspaces can be enhanced by
// storing the source code of those classes in the workspace as well.
// This process is also organized by the workspace through the
// importClassCode() method.
// END_HTML
//

#include "RooFit.h"
#include "RooWorkspace.h"
#include "RooAbsPdf.h"
#include "RooRealVar.h"
#include "RooCategory.h"
#include "RooAbsData.h"
#include "RooCmdConfig.h"
#include "RooMsgService.h"
#include "TInterpreter.h"
#include "TClassTable.h"
#include "TBaseClass.h"
#include "TSystem.h"
#include "TRegexp.h"
#include <map>
#include <string>
#include <list>

using namespace std ;


#if ROOT_VERSION_CODE <= ROOT_VERSION(5,19,02)
#include "Api.h"
#endif


#include "TClass.h"
#include "Riostream.h"
#include <string.h>
#include <assert.h>

ClassImp(RooWorkspace)
;

//_____________________________________________________________________________
ClassImp(RooWorkspace::CodeRepo)
;

//_____________________________________________________________________________
ClassImp(RooWorkspace::WSDir)
;

list<string> RooWorkspace::_classDeclDirList ;
list<string> RooWorkspace::_classImplDirList ;
string RooWorkspace::_classFileExportDir = ".wscode.%s" ;
Bool_t RooWorkspace::_autoClass = kFALSE ;


//_____________________________________________________________________________
void RooWorkspace::addClassDeclImportDir(const char* dir) 
{
  // Add 'dir' to search path for class declaration (header) files, when
  // attempting to import class code with importClassClode()

  _classDeclDirList.push_back(dir) ;
}


//_____________________________________________________________________________
void RooWorkspace::addClassImplImportDir(const char* dir) 
{
  // Add 'dir' to search path for class implementation (.cxx) files, when
  // attempting to import class code with importClassClode()

  _classImplDirList.push_back(dir) ;
}


//_____________________________________________________________________________
void RooWorkspace::setClassFileExportDir(const char* dir) 
{
  // Specify the name of the directory in which embedded source
  // code is unpacked and compiled. The specified string may contain
  // one '%s' token which will be substituted by the workspace name

  if (dir) {
    _classFileExportDir = dir ;
  } else {
    _classFileExportDir = ".wscode.%s" ;
  }
}


//_____________________________________________________________________________
void RooWorkspace::autoImportClassCode(Bool_t flag) 
{
  // If flag is true, source code of classes not the the ROOT distribution
  // is automatically imported if on object of such a class is imported
  // in the workspace
  _autoClass = flag ; 
}



//_____________________________________________________________________________
RooWorkspace::RooWorkspace() : _classes(this), _dir(0)
{
  // Default constructor
}



//_____________________________________________________________________________
RooWorkspace::RooWorkspace(const char* name, const char* title) : TNamed(name,title?title:name), _classes(this), _dir(0)
{
  // Construct empty workspace with given name and title
}



//_____________________________________________________________________________
RooWorkspace::RooWorkspace(const RooWorkspace& other) : TNamed(other), _classes(this), _dir(0)
{
  // Workspace copy constructor

  other._allOwnedNodes.snapshot(_allOwnedNodes,kTRUE) ;

  TIterator* iter = other._dataList.MakeIterator() ;
  TObject* data2 ;
  while((data2=iter->Next())) {
    _dataList.Add(data2->Clone()) ;
  }
  delete iter ;

  TIterator* iter2 = other._snapshots.MakeIterator() ;
  RooArgSet* snap ;
  while((snap=(RooArgSet*)iter2->Next())) {
    RooArgSet* snapClone = (RooArgSet*) snap->snapshot() ;
    snapClone->setName(snap->GetName()) ;
    _snapshots.Add(snapClone) ;
  }
  delete iter2 ;
}



//_____________________________________________________________________________
RooWorkspace::~RooWorkspace() 
{
  // Workspace destructor

  _dataList.Delete() ;
  if (_dir) {
    delete _dir ;
  }
  _snapshots.Delete() ;
}


//_____________________________________________________________________________
Bool_t RooWorkspace::import(const RooArgSet& args, const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3) 
{
  // Import multiple RooAbsArg objects into workspace. For details on arguments see documentation
  // of import() method for single RooAbsArg

  TIterator* iter = args.createIterator() ;
  RooAbsArg* oneArg ;
  Bool_t ret(kFALSE) ;
  while((oneArg=(RooAbsArg*)iter->Next())) {
    ret |= import(*oneArg,arg1,arg2,arg3) ;
  }
  return ret ;
}



//_____________________________________________________________________________
Bool_t RooWorkspace::import(const RooAbsArg& inArg, const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3) 
{
  //  Import a RooAbsArg object, e.g. function, p.d.f or variable into the workspace. This import function clones the input argument and will
  //  own the clone. If a composite object is offered for import, e.g. a p.d.f with parameters and observables, the
  //  complete tree of objects is imported. If any of the _variables_ of a composite object (parameters/observables) are already 
  //  in the workspace the imported p.d.f. is connected to the already existing variables. If any of the _function_ objects (p.d.f, formulas) 
  //  to be imported already exists in the workspace an error message is printed and the import of the entire tree of objects is cancelled. 
  //  Several optional arguments can be provided to modify the import procedure.
  //
  //  Accepted arguments
  //  -------------------------------
  //  RenameConflictNodes(const char* suffix) -- Add suffix to branch node name if name conflicts with existing node in workspace
  //  RenameAllNodes(const char* suffix) -- Add suffix to all branch node names including top level node
  //  RenameAllVariables(const char* suffix) -- Add suffix to all variables names
  //  RenameVariable(const char* inputName, const char* outputName) -- Rename variable as specified upon import.
  //  RecycleConflictNodes() -- If any of the function objects to be imported already exist in the name space, connect the
  //                            imported expression to the already existing nodes. WARNING: use with care! If function definitions
  //                            do not match, this alters the definition of your function upon import
  //
  //  The RenameConflictNodes, RenameNodes and RecycleConflictNodes arguments are mutually exclusive. The RenameVariable argument can be repeated
  //  as often as necessary to rename multiple variables. Alternatively, a single RenameVariable argument can be given with
  //  two comma separated lists.

  RooLinkedList args ;
  args.Add((TObject*)&arg1) ;
  args.Add((TObject*)&arg2) ;
  args.Add((TObject*)&arg3) ;

  // Select the pdf-specific commands 
  RooCmdConfig pc(Form("RooWorkspace::import(%s)",GetName())) ;

  pc.defineString("conflictSuffix","RenameConflictNodes",0) ;
  pc.defineString("allSuffix","RenameAllNodes",0) ;
  pc.defineString("allVarsSuffix","RenameAllVariables",0) ;
  pc.defineString("varChangeIn","RenameVar",0,"",kTRUE) ;
  pc.defineString("varChangeOut","RenameVar",1,"",kTRUE) ;
  pc.defineInt("useExistingNodes","RecycleConflictNodes",0,0) ;
  pc.defineMutex("RenameConflictNodes","RenameAllNodes") ;
  pc.defineMutex("RenameConflictNodes","RecycleConflictNodes") ;
  pc.defineMutex("RenameAllNodes","RecycleConflictNodes") ;
  pc.defineMutex("RenameVariable","RenameAllVariables") ;

  // Process and check varargs 
  pc.process(args) ;
  if (!pc.ok(kTRUE)) {
    return kTRUE ;
  }

  // Decode renaming logic into suffix string and boolean for conflictOnly mode
  const char* suffixC = pc.getString("conflictSuffix") ;
  const char* suffixA = pc.getString("allSuffix") ;
  const char* suffixV = pc.getString("allVarsSuffix") ;
  const char* varChangeIn = pc.getString("varChangeIn") ;
  const char* varChangeOut = pc.getString("varChangeOut") ;
  Int_t useExistingNodes = pc.getInt("useExistingNodes") ;

  // Turn zero length strings into null pointers 
  if (suffixC && strlen(suffixC)==0) suffixC = 0 ;
  if (suffixA && strlen(suffixA)==0) suffixA = 0 ;

  Bool_t conflictOnly = suffixA ? kFALSE : kTRUE ;
  const char* suffix = suffixA ? suffixA : suffixC ;

  // Process any change in variable names 
  map<string,string> varMap ;
  if (strlen(varChangeIn)>0) {
    
    // Parse comma separated lists into map<string,string>
    char tmp[1024] ;
    strcpy(tmp,varChangeIn) ;
    list<string> tmpIn,tmpOut ;
    char* ptr = strtok(tmp,",") ;
    while (ptr) {
      tmpIn.push_back(ptr) ;
      ptr = strtok(0,",") ;
    }
    strcpy(tmp,varChangeOut) ;
    ptr = strtok(tmp,",") ;
    while (ptr) {
      tmpOut.push_back(ptr) ;
      ptr = strtok(0,",") ;
    }    
    list<string>::iterator iin = tmpIn.begin() ;
    list<string>::iterator iout = tmpOut.begin() ;
    for (;iin!=tmpIn.end() ; ++iin,++iout) {
      varMap[*iin]=*iout ;
    }       
  }

  // Process RenameAllVariables argument if specified
  if (suffixV != 0 && strlen(suffixV)>0) {
    RooArgSet* vars = inArg.getVariables() ;
    TIterator* iter = vars->createIterator() ;
    RooAbsArg* v ;
    while((v=(RooAbsArg*)iter->Next())) {
      varMap[v->GetName()] = Form("%s_%s",v->GetName(),suffixV) ;
    }
    delete iter ;
    delete vars ;
  }
  
  // Scan for overlaps with current contents
  RooAbsArg* wsarg = _allOwnedNodes.find(inArg.GetName()) ;
  if (!suffix && wsarg && !useExistingNodes && !(inArg.isFundamental() && varMap[inArg.GetName()]!="")) {
    if (wsarg!=&inArg) {
      coutE(ObjectHandling) << "RooWorkSpace::import(" << GetName() << ") ERROR importing object named " << inArg.GetName() 
			    << ": another instance with same name already in the workspace and no conflict resolution protocol specified" << endl ;
      return kTRUE ;    
    } else {
      coutI(ObjectHandling) << "RooWorkSpace::import(" << GetName() << ") Object " << inArg.GetName() << " is already in workspace!" << endl ;
      return kTRUE ;    
    }
  }

  // Make list of conflicting nodes
  RooArgSet conflictNodes ;
  RooArgSet branchSet ;
  inArg.branchNodeServerList(&branchSet) ;
  TIterator* iter = branchSet.createIterator() ;
  RooAbsArg* branch ;
  while ((branch=(RooAbsArg*)iter->Next())) {
    RooAbsArg* wsbranch = _allOwnedNodes.find(branch->GetName()) ;
    if (wsbranch && wsbranch!=branch) {
      conflictNodes.add(*branch) ;
    }
  }
  delete iter ;
  
  // Terminate here if there are conflicts and no resolution protocol
  if (conflictNodes.getSize()>0 && !suffix && !useExistingNodes) {
      coutE(ObjectHandling) << "RooWorkSpace::import(" << GetName() << ") ERROR object named " << inArg.GetName() << ": component(s) " 
	   << conflictNodes << " already in the workspace and no conflict resolution protocol specified" << endl ;      
      return kTRUE ;
  }
    
  // Now create a working copy of the incoming object tree
  RooArgSet* cloneSet = (RooArgSet*) RooArgSet(inArg).snapshot(kTRUE) ;
  RooAbsArg* cloneTop = cloneSet->find(inArg.GetName()) ;

  // Mark all nodes for renaming if we are not in conflictOnly mode
  if (!conflictOnly) {
    conflictNodes.removeAll() ;
    conflictNodes.add(branchSet) ;
  }

  // Mark nodes that are to be renamed with special attribute
  TIterator* citer = conflictNodes.createIterator() ;
  string topName2 = cloneTop->GetName() ;
  RooAbsArg* cnode ;
  while ((cnode=(RooAbsArg*)citer->Next())) {
    RooAbsArg* cnode2 = cloneSet->find(cnode->GetName()) ;
    string origName = cnode2->GetName() ;
    cnode2->SetName(Form("%s_%s",cnode2->GetName(),suffix)) ;
    cnode2->SetTitle(Form("%s (%s)",cnode2->GetTitle(),suffix)) ;
    string tag = Form("ORIGNAME:%s",origName.c_str()) ;
    cnode2->setAttribute(tag.c_str()) ;

    // Save name of new top level node for later use
    if (cnode2==cloneTop) {
      topName2 = cnode2->GetName() ;      
    }

    coutI(ObjectHandling) << "RooWorkspace::import(" << GetName() 
		       << ") Resolving name conflict in workspace by changing name of imported node  " 
		       << origName << " to " << cnode2->GetName() << endl ;
  }  
  delete citer ;

  // Process any change in variable names 
  if (strlen(varChangeIn)>0 || (suffixV && strlen(suffixV)>0)) {
    
    // Process all changes in variable names
    TIterator* cliter = cloneSet->createIterator() ;
    while ((cnode=(RooAbsArg*)cliter->Next())) {
      
      if (varMap.find(cnode->GetName())!=varMap.end()) { 	
	string origName = cnode->GetName() ;
	cnode->SetName(varMap[cnode->GetName()].c_str()) ;
	string tag = Form("ORIGNAME:%s",origName.c_str()) ;
	cnode->setAttribute(tag.c_str()) ;
	coutI(ObjectHandling) << "RooWorkspace::import(" << GetName() << ") Changing name of variable " 
			   << origName << " to " << cnode->GetName() << " on request" << endl ;

	if (cnode==cloneTop) {
	  topName2 = cnode->GetName() ;
	}

      }    
    }
    delete cliter ;
  }
  
  // Now clone again with renaming effective
  RooArgSet* cloneSet2 = (RooArgSet*) RooArgSet(*cloneTop).snapshot(kTRUE) ;
  RooAbsArg* cloneTop2 = cloneSet2->find(topName2.c_str()) ;

  // Make final check list of conflicting nodes
  RooArgSet conflictNodes2 ;
  RooArgSet branchSet2 ;
  inArg.branchNodeServerList(&branchSet) ;
  TIterator* iter2 = branchSet2.createIterator() ;
  RooAbsArg* branch2 ;
  while ((branch2=(RooAbsArg*)iter2->Next())) {
    if (_allOwnedNodes.find(branch2->GetName())) {
      conflictNodes2.add(*branch2) ;
    }
  }
  delete iter2 ;

  // Terminate here if there are conflicts and no resolution protocol
  if (conflictNodes2.getSize()) {
    coutE(ObjectHandling) << "RooWorkSpace::import(" << GetName() << ") ERROR object named " << inArg.GetName() << ": component(s) " 
			  << conflictNodes2 << " cause naming conflict after conflict resolution protocol was executed" << endl ;      
    return kTRUE ;
  }
    
  // Print a message for each imported node
  iter = cloneSet2->createIterator() ;
  RooAbsArg* node ;
  RooArgSet recycledNodes ;
  while((node=(RooAbsArg*)iter->Next())) {

    if (_autoClass) {
      if (!_classes.autoImportClass(node->IsA())) {
	coutW(ObjectHandling) << "RooWorkspace::import(" << GetName() << ") WARNING: problems import class code of object " 
			      << node->IsA()->GetName() << "::" << node->GetName() << ", reading of workspace will require external definition of class" << endl ;
      }
    }

    // Point expensiveObjectCache to copy in this workspace
    RooExpensiveObjectCache& oldCache = node->expensiveObjectCache() ;
    node->setExpensiveObjectCache(_eocache) ;    
    _eocache.importCacheObjects(oldCache,node->GetName(),kTRUE) ;

    // Check if node is already in workspace (can only happen for variables or identical instances, unless RecycleConflictNodes is specified)
    if (_allOwnedNodes.find(node->GetName())) {
      // Do not import node, add not to list of nodes that require reconnection
      coutI(ObjectHandling) << "RooWorkspace::import(" << GetName() << ") using existing copy of " << node->IsA()->GetName() 
			 << "::" << node->GetName() << " for import of " << cloneTop2->IsA()->GetName() << "::" 
			 << cloneTop2->GetName() << endl ;      
      recycledNodes.add(*_allOwnedNodes.find(node->GetName())) ;

    } else {
      // Import node
      coutI(ObjectHandling) << "RooWorkspace::import(" << GetName() << ") importing " << node->IsA()->GetName() << "::" 
			 << node->GetName() << endl ;
      _allOwnedNodes.addOwned(*node) ;
      if (_dir) {
	_dir->InternalAppend(node) ;
      }
    }
  }

  // Release working copy
  delete cloneSet ;


  // Reconnect any nodes that need to be
  if (recycledNodes.getSize()>0) {
    iter->Reset() ;
    while((node=(RooAbsArg*)iter->Next())) {
      node->redirectServers(recycledNodes) ;
    }
  }

  delete iter ;
  

  return kFALSE ;
}



//_____________________________________________________________________________
Bool_t RooWorkspace::import(RooAbsData& inData, const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3) 
{
  //  Import a dataset (RooDataSet or RooDataHist) into the work space. The workspace will contain a copy of the data
  //  The dataset and its variables can be renamed upon insertion with the options below
  //
  //  Accepted arguments
  //  -------------------------------
  //  RenameDataset(const char* suffix) -- Rename dataset upon insertion
  //  RenameVariable(const char* inputName, const char* outputName) -- Change names of observables in dataset upon insertion

  coutI(ObjectHandling) << "RooWorkspace::import(" << GetName() << ") importing dataset " << inData.GetName() << endl ;

  RooLinkedList args ;
  args.Add((TObject*)&arg1) ;
  args.Add((TObject*)&arg2) ;
  args.Add((TObject*)&arg3) ;

  // Select the pdf-specific commands 
  RooCmdConfig pc(Form("RooWorkspace::import(%s)",GetName())) ;

  pc.defineString("dsetName","RenameDataset",0,"") ;
  pc.defineString("varChangeIn","RenameVar",0,"",kTRUE) ;
  pc.defineString("varChangeOut","RenameVar",1,"",kTRUE) ;

  // Process and check varargs 
  pc.process(args) ;
  if (!pc.ok(kTRUE)) {
    return kTRUE ;
  }

  // Decode renaming logic into suffix string and boolean for conflictOnly mode
  const char* dsetName = pc.getString("dsetName") ;
  const char* varChangeIn = pc.getString("varChangeIn") ;
  const char* varChangeOut = pc.getString("varChangeOut") ;

  // Transform emtpy string into null pointer
  if (dsetName && strlen(dsetName)==0) {
    dsetName=0 ;
  }

  // Rename dataset if required
  RooAbsData* clone ;
  if (dsetName) {
    coutI(ObjectHandling) << "RooWorkSpace::import(" << GetName() << ") changing name of dataset from  " << inData.GetName() << " to " << dsetName << endl ;
    clone = (RooAbsData*) inData.Clone(dsetName) ;
  } else {
    clone = (RooAbsData*) inData.Clone(inData.GetName()) ;
  }


  // Process any change in variable names 
  if (strlen(varChangeIn)>0) {
    
    // Parse comma separated lists of variable name changes
    char tmp[1024] ;
    strcpy(tmp,varChangeIn) ;
    list<string> tmpIn,tmpOut ;
    char* ptr = strtok(tmp,",") ;
    while (ptr) {
      tmpIn.push_back(ptr) ;
      ptr = strtok(0,",") ;
    }
    strcpy(tmp,varChangeOut) ;
    ptr = strtok(tmp,",") ;
    while (ptr) {
      tmpOut.push_back(ptr) ;
      ptr = strtok(0,",") ;
    }    
    list<string>::iterator iin = tmpIn.begin() ;
    list<string>::iterator iout = tmpOut.begin() ;

    for (; iin!=tmpIn.end() ; ++iin,++iout) {
      coutI(ObjectHandling) << "RooWorkSpace::import(" << GetName() << ") changing name of dataset observable " << *iin << " to " << *iout << endl ;
      clone->changeObservableName(iin->c_str(),iout->c_str()) ;
    }
  }

  // Now import the dataset observables
  TIterator* iter = clone->get()->createIterator() ;
  RooAbsArg* carg ;
  while((carg=(RooAbsArg*)iter->Next())) {
    import(*carg) ;
  }
  delete iter ;
    
  _dataList.Add(clone) ;
  return kFALSE ;
}



//_____________________________________________________________________________
Bool_t RooWorkspace::importClassCode(TClass* theClass, Bool_t doReplace) 
{
  return _classes.autoImportClass(theClass,doReplace) ;
}



//_____________________________________________________________________________
Bool_t RooWorkspace::importClassCode(const char* pat, Bool_t doReplace)  
{
  // Inport code of all classes in the workspace that have a class name
  // that matches pattern 'pat' and which are not found to be part of
  // the standard ROOT distribution. If doReplace is true any existing
  // class code saved in the workspace is replaced

  Bool_t ret(kTRUE) ;

  TRegexp re(pat,kTRUE) ;
  TIterator* iter = componentIterator() ;
  RooAbsArg* carg ;
  while((carg=(RooAbsArg*)iter->Next())) {
    TString className = carg->IsA()->GetName() ;
    if (className.Index(re)>=0 && !_classes.autoImportClass(carg->IsA(),doReplace)) {
      coutW(ObjectHandling) << "RooWorkspace::import(" << GetName() << ") WARNING: problems import class code of object " 
			    << carg->IsA()->GetName() << "::" << carg->GetName() << ", reading of workspace will require external definition of class" << endl ;
      ret = kFALSE ;
    }
  }  
  delete iter ;
  return ret ;
}



//_____________________________________________________________________________
Bool_t RooWorkspace::saveSnapshot(const char* name, const RooArgSet& params, Bool_t importValues) 
{
  // Save snapshot of values and attributes (including "Constant") of parameters 'params'
  // If importValues is FALSE, the present values from the object in the workspace are
  // saved. If importValues is TRUE, the values of the objects passed in the 'params'
  // argument are saved

  RooArgSet* actualParams = (RooArgSet*) _allOwnedNodes.selectCommon(params) ;
  RooArgSet* snapshot = (RooArgSet*) actualParams->snapshot() ;
  delete actualParams ;

  snapshot->setName(name) ;

  if (importValues) {
    *snapshot = params ;
  }

  RooArgSet* oldSnap = (RooArgSet*) _snapshots.FindObject(name) ;
  if (oldSnap) {
    coutI(ObjectHandling) << "RooWorkspace::saveSnaphot(" << GetName() << ") replacing previous snapshot with name " << name << endl ;
    _snapshots.Remove(oldSnap) ;
    delete oldSnap ;
  }

  _snapshots.Add(snapshot) ;

  return kTRUE ;
}




//_____________________________________________________________________________
Bool_t RooWorkspace::loadSnapshot(const char* name) 
{
  // Load the values and attributes of the parameters in the snapshot saved with
  // the given name

  RooArgSet* snap = (RooArgSet*) _snapshots.find(name) ;
  if (!snap) {
    coutE(ObjectHandling) << "RooWorkspace::loadSnapshot(" << GetName() << ") no snapshot with name " << name << " is available" << endl ;
    return kFALSE ;
  }

  RooArgSet* actualParams = (RooArgSet*) _allOwnedNodes.selectCommon(*snap) ;
  *actualParams = *snap ;
  delete actualParams ;

  return kTRUE ;
}



//_____________________________________________________________________________
Bool_t RooWorkspace::merge(const RooWorkspace& /*other*/) 
{
  // Stub for merge function with another workspace (not implemented yet)
  return kFALSE ;
}



//_____________________________________________________________________________
Bool_t RooWorkspace::join(const RooWorkspace& /*other*/) 
{
  // Stub for join function with another workspace (not implemented yet)
  return kFALSE ;
}


//_____________________________________________________________________________
RooAbsPdf* RooWorkspace::pdf(const char* name) 
{ 
  // Retrieve p.d.f (RooAbsPdf) with given name. A null pointer is returned if not found

  return dynamic_cast<RooAbsPdf*>(_allOwnedNodes.find(name)) ; 
}


//_____________________________________________________________________________
RooAbsReal* RooWorkspace::function(const char* name) 
{ 
  // Retrieve function (RooAbsReal) with given name. Note that all RooAbsPdfs are also RooAbsReals. A null pointer is returned if not found.

  return dynamic_cast<RooAbsReal*>(_allOwnedNodes.find(name)) ; 
}


//_____________________________________________________________________________
RooRealVar* RooWorkspace::var(const char* name) 
{ 
  // Retrieve real-valued variable (RooRealVar) with given name. A null pointer is returned if not found

  return dynamic_cast<RooRealVar*>(_allOwnedNodes.find(name)) ; 
}


//_____________________________________________________________________________
RooCategory* RooWorkspace::cat(const char* name) 
{ 
  // Retrieve discrete variable (RooCategory) with given name. A null pointer is returned if not found

  return dynamic_cast<RooCategory*>(_allOwnedNodes.find(name)) ; 
}


//_____________________________________________________________________________
RooAbsCategory* RooWorkspace::catfunc(const char* name)
{
  // Retrieve discrete function (RooAbsCategory) with given name. A null pointer is returned if not found
  return dynamic_cast<RooAbsCategory*>(_allOwnedNodes.find(name)) ; 
}



//_____________________________________________________________________________
RooAbsArg* RooWorkspace::arg(const char* name) 
{
  // Return RooAbsArg with given name. A null pointer is returned if none is found.
  return _allOwnedNodes.find(name) ;
}


//_____________________________________________________________________________
RooAbsArg* RooWorkspace::fundArg(const char* name) 
{
  // Return fundamental (i.e. non-derived) RooAbsArg with given name. Fundamental types
  // are e.g. RooRealVar, RooCategory. A null pointer is returned if none is found.
  RooAbsArg* tmp = arg(name) ;
  if (!tmp) {
    return 0 ;
  }
  return tmp->isFundamental() ? tmp : 0 ;
}



//_____________________________________________________________________________
RooAbsData* RooWorkspace::data(const char* name) 
{
  // Retrieve dataset (binned or unbinned) with given name. A null pointer is returned if not found

  return (RooAbsData*)_dataList.FindObject(name) ;
}



//_____________________________________________________________________________
Bool_t RooWorkspace::CodeRepo::autoImportClass(TClass* tc, Bool_t doReplace) 
{
  // Import code of class 'tc' into the repository. If code is already in repository it is only imported
  // again if doReplace is false. The names and location of the source files is determined from the information
  // in TClass. If no location is found in the TClass information, the files are searched in the workspace
  // search path, defined by addClassDeclImportDir() and addClassImplImportDir() for declaration and implementation
  // files respectively. If files cannot be found, abort with error status, otherwise update the internal
  // class-to-file map and import the contents of the files, if they are not imported yet.


  oocxcoutD(_wspace,ObjectHandling) << "RooWorkspace::CodeRepo(" << _wspace->GetName() << ") request to import code of class " << tc->GetName() << endl ;

  // *** PHASE 1 *** Check if file needs to be imported, or is in ROOT distribution, and check if it can be persisted

  // Check if we already have the class (i.e. it is in the classToFile map)
  if (!doReplace && _c2fmap.find(tc->GetName())!=_c2fmap.end()) {
    oocxcoutD(_wspace,ObjectHandling) << "RooWorkspace::CodeRepo(" << _wspace->GetName() << ") code of class " << tc->GetName() << " already imported, skipping" << endl ;
    return kTRUE ;
  }

  // Retrieve file names through ROOT TClass interface
  string implfile = tc->GetImplFileName() ;
  string declfile = tc->GetDeclFileName() ;

  // Check that file names are not empty
  if (implfile.empty() || declfile.empty()) {
    oocoutE(_wspace,ObjectHandling) << "RooWorkspace::CodeRepo(" << _wspace->GetName() << ") ERROR: cannot retrieve code file names for class " 
				   << tc->GetName() << " through ROOT TClass interface, unable to import code" << endl ;
    return kFALSE ;
  }

  // Check if header filename is found in ROOT distribution, if so, do not import class
  TString rootsys = gSystem->Getenv("ROOTSYS") ;
  char* implpath = gSystem->ConcatFileName(rootsys.Data(),implfile.c_str()) ;
  if (!gSystem->AccessPathName(implpath)) {
    oocxcoutD(_wspace,ObjectHandling) << "RooWorkspace::CodeRepo(" << _wspace->GetName() << ") code of class " << tc->GetName() << " is in ROOT distribution, skipping " << endl ;
    delete[] implpath ;
    return kTRUE ;
  }
  delete[] implpath ;
  implpath=0 ;

  // Require that class meets technical criteria to be persistable (i.e it has a default ctor)
  // (We also need a default ctor of abstract classes, but cannot check that through is interface
  //  as TClass::HasDefaultCtor only returns true for callable default ctors)
#if ROOT_VERSION_CODE <= ROOT_VERSION(5,19,02)
  if (!(tc->GetClassInfo()->Property()&G__BIT_ISABSTRACT) && !tc->HasDefaultConstructor()) {
#else
  if (!(gCint->ClassInfo_Property(tc->GetClassInfo())&G__BIT_ISABSTRACT) && !tc->HasDefaultConstructor()) {
#endif
    oocoutW(_wspace,ObjectHandling) << "RooWorkspace::autoImportClass(" << _wspace->GetName() << ") WARNING cannot import class " 
				    << tc->GetName() << " : it cannot be persisted because it doesn't have a default constructor. Please fix " << endl ;
    return kFALSE ;      
  }


  // *** PHASE 2 *** Check if declaration and implementation files can be located 

  char* declpath = 0 ;

  // Check if header file can be found in specified location
  // If not, scan through list of 'class declaration' paths in RooWorkspace
  if (gSystem->AccessPathName(declfile.c_str())) {

    // Check list of additional declaration paths
    list<string>::iterator diter = RooWorkspace::_classDeclDirList.begin() ;

    while(diter!= RooWorkspace::_classDeclDirList.end()) {
      
      declpath = gSystem->ConcatFileName(diter->c_str(),declfile.c_str()) ;      
      if (!gSystem->AccessPathName(declpath)) {
	// found declaration file
	break ;
      }
      // cleanup and continue ;
      delete[] declpath ;
      declpath=0 ;

      ++diter ;
    }
    
    // Header file cannot be found anywhere, warn user and abort operation
    if (!declpath) {
      oocoutW(_wspace,ObjectHandling) << "RooWorkspace::autoImportClass(" << _wspace->GetName() << ") WARNING Cannot access code of class " 
				      << tc->GetName() << " because header file " << declfile << " is not found in current directory nor in $ROOTSYS" ;
      if (_classDeclDirList.size()>0) {
	ooccoutW(_wspace,ObjectHandling) << ", nor in the search path " ;
	diter = RooWorkspace::_classDeclDirList.begin() ;

	while(diter!= RooWorkspace::_classDeclDirList.end()) {

	  if (diter!=RooWorkspace::_classDeclDirList.begin()) {
	    ooccoutW(_wspace,ObjectHandling) << "," ;
	  }
	  ooccoutW(_wspace,ObjectHandling) << diter->c_str() ;
	  ++diter ;
	}
      }
      ooccoutW(_wspace,ObjectHandling) << ". To fix this problem add the required directory to the search "
				       << "path using RooWorkspace::addClassDeclDir(const char* dir)" << endl ;
      
      return kFALSE ;
    }
  }

  
  // Check if implementation file can be found in specified location
  // If not, scan through list of 'class implementation' paths in RooWorkspace
  if (gSystem->AccessPathName(implfile.c_str())) {

    // Check list of additional declaration paths
    list<string>::iterator iiter = RooWorkspace::_classImplDirList.begin() ;

    while(iiter!= RooWorkspace::_classImplDirList.end()) {
      
      implpath = gSystem->ConcatFileName(iiter->c_str(),implfile.c_str()) ;      
      if (!gSystem->AccessPathName(implpath)) {
	// found implementation file
	break ;
      }
      // cleanup and continue ;
      delete[] implpath ;
      implpath=0 ;

      ++iiter ;
    }
     
    // Implementation file cannot be found anywhere, warn user and abort operation
    if (!implpath) {
      oocoutW(_wspace,ObjectHandling) << "RooWorkspace::autoImportClass(" << _wspace->GetName() << ") WARNING Cannot access code of class " 
				      << tc->GetName() << " because implementation file " << implfile << " is not found in current directory nor in $ROOTSYS" ;
      if (_classDeclDirList.size()>0) {
	ooccoutW(_wspace,ObjectHandling) << ", nor in the search path " ;
	iiter = RooWorkspace::_classImplDirList.begin() ;

	while(iiter!= RooWorkspace::_classImplDirList.end()) {

	  if (iiter!=RooWorkspace::_classImplDirList.begin()) {
	    ooccoutW(_wspace,ObjectHandling) << "," ;
	  }
	  ooccoutW(_wspace,ObjectHandling) << iiter->c_str() ;
	  ++iiter ;
	}
      }
      ooccoutW(_wspace,ObjectHandling) << ". To fix this problem add the required directory to the search "
				       << "path using RooWorkspace::addClassImplDir(const char* dir)" << endl ;    
      return kFALSE ;
    }
  }
  
  char buf[1024] ;

  // *** Phase 3 *** Prepare to import code from files into STL string buffer
  //
  // Code storage is organized in two linked maps
  //
  // _fmap contains stl strings with code, indexed on declaration file name
  //
  // _c2fmap contains list of declaration file names and list of base classes
  //                  and is indexed on class name
  //
  // Phase 3 is skipped if fmap already contains an entry with given filebasename

  string declfilename = declpath?gSystem->BaseName(declpath):gSystem->BaseName(declfile.c_str()) ;

  // Split in base and extension
  int dotpos2 = strrchr(declfilename.c_str(),'.') - declfilename.c_str() ;
  string declfilebase = declfilename.substr(0,dotpos2) ;
  string declfileext = declfilename.substr(dotpos2+1) ;

  // If file has not beed stored yet, enter stl strings with implementation and declaration in file map
  if (_fmap.find(declfilebase) == _fmap.end()) {

    // Open declaration file
    fstream fdecl(declpath?declpath:declfile.c_str()) ;
    
    // Abort import if declaration file cannot be opened
    if (!fdecl) {
      oocoutE(_wspace,ObjectHandling) << "RooWorkspace::autoImportClass(" << _wspace->GetName() 
				      << ") ERROR opening declaration file " <<  declfile << endl ;
      return kFALSE ;      
    }
    
    oocoutI(_wspace,ObjectHandling) << "RooWorkspace::autoImportClass(" << _wspace->GetName() 
				    << ") importing code of class " << tc->GetName() 
				    << " from " << (implpath?implpath:implfile.c_str()) 
				    << " and " << (declpath?declpath:declfile.c_str()) << endl ;
    
    
    // Read entire file into an stl string
    string decl ;
    while(fdecl.getline(buf,1023)) {
      decl += buf ;
      decl += '\n' ;
    }
    
    // Open implementation file
    fstream fimpl(implpath?implpath:implfile.c_str()) ;
    
    // Abort import if implementation file cannot be opened
    if (!fimpl) {
      oocoutE(_wspace,ObjectHandling) << "RooWorkspace::autoImportClass(" << _wspace->GetName() 
				      << ") ERROR opening implementation file " <<  implfile << endl ;
      return kFALSE ;      
    }
    
    
    // Import entire implentation file into stl string
    string impl ;
    while(fimpl.getline(buf,1023)) {
      // Process #include statements here
      
      // Look for include state of self
      Bool_t foundSelfInclude=kFALSE ;
      
      // Look for include of declaration file corresponding to this implementation file
      if (strstr(buf,"#include")) {
	// Process #include statements here
	char tmp[1024] ;
	strcpy(tmp,buf) ;
	strtok(tmp," <\"") ;
	char* incfile = strtok(0," <\"") ;
	
	if (strstr(incfile,declfilename.c_str())) {
	  foundSelfInclude=kTRUE ;
	}
      } 
      
      // Explicitly rewrite include of own declaration file to string
      // any directory prefixes, copy all other lines verbatim in stl string
      if (foundSelfInclude) {
	// If include of self is found, substitute original include 
	// which may have directory structure with a plain include
	impl += Form("#include \"%s.%s\"\n",declfilebase.c_str(),declfileext.c_str()) ;
      } else {
	impl += buf ;
	impl += '\n' ;
      }
    }
    
        
    // Create entry in file map
    _fmap[declfilebase]._hfile = decl ;
    _fmap[declfilebase]._cxxfile = impl ;   
    _fmap[declfilebase]._hext = declfileext ;

  } else {

    // Inform that existing file entry is being recycled because it already contained class code
    oocoutI(_wspace,ObjectHandling) << "RooWorkspace::autoImportClass(" << _wspace->GetName() 
				    << ") code of class " << tc->GetName() 
				    << " was already imported from " << (implpath?implpath:implfile.c_str()) 
				    << " and " << (declpath?declpath:declfile.c_str()) << endl ;
    
  }


  // *** PHASE 4 *** Import stl strings with code into workspace 
  //
  // If multiple classes are declared in a single code unit, there will be
  // multiple _c2fmap entries all pointing to the same _fmap entry.  
  
  // Make list of all immediate base classes of this class
  TString baseNameList ;
  TList* bl = tc->GetListOfBases() ;
  TIterator* iter = bl->MakeIterator() ;
  TBaseClass* base ;
  list<TClass*> bases ;
  while((base=(TBaseClass*)iter->Next())) {
    if (baseNameList.Length()>0) {
      baseNameList += "," ;
    }
    baseNameList += base->GetClassPointer()->GetName() ;
    bases.push_back(base->GetClassPointer()) ;
  }
  
  // Map class name to above _fmap entries, along with list of base classes
  // in _c2fmap
  _c2fmap[tc->GetName()]._baseName = baseNameList ;
  _c2fmap[tc->GetName()]._fileBase = declfilebase ;
  
  // Recursive store all base classes.
  list<TClass*>::iterator biter = bases.begin() ;
  while(biter!=bases.end()) {
    autoImportClass(*biter,doReplace) ;
    ++biter ;
  }

  // Cleanup 
  if (implpath) {
    delete[] implpath ;
  }
  if (declpath) {
    delete[] declpath ;
  }

  return kTRUE ;
}


//_____________________________________________________________________________
Bool_t RooWorkspace::makeDir() 
{
  // Create transient TDirectory representation of this workspace. This directory
  // will appear as a subdirectory of the directory that contains the workspace
  // and will have the name of the workspace suffixed with "Dir". The TDirectory
  // interface is read-only. Any attempt to insert objects into the workspace
  // directory representation will result in an error message. Note that some
  // ROOT object like TH1 automatically insert themselves into the current directory
  // when constructed. This will give error messages when done in a workspace
  // directory.

  if (_dir) return kTRUE ;

  TString name = Form("%sDir",GetName()) ;
  TString title= Form("TDirectory representation of RooWorkspace %s",GetName()) ;
  _dir = new WSDir(name.Data(),title.Data(),this) ;

  TIterator* iter = componentIterator() ;
  RooAbsArg* darg ;
  while((darg=(RooAbsArg*)iter->Next())) {
    _dir->InternalAppend(darg) ;
  }

  return kTRUE ;
}



//_____________________________________________________________________________
void RooWorkspace::Print(Option_t* /*opts*/) const 
{
  // Print contents of the workspace 

  cout << endl << "RooWorkspace(" << GetName() << ") " << GetTitle() << " contents" << endl << endl  ;

  RooAbsArg* parg ;

  RooArgSet pdfSet ;
  RooArgSet funcSet ;
  RooArgSet varSet ;
  RooArgSet catfuncSet ;

  // Split list of components in pdfs, functions and variables
  TIterator* iter = _allOwnedNodes.createIterator() ;
  while((parg=(RooAbsArg*)iter->Next())) {

    if (parg->IsA()->InheritsFrom(RooAbsPdf::Class())) {
      pdfSet.add(*parg) ;
    }

    if (parg->IsA()->InheritsFrom(RooAbsReal::Class()) && 
	!parg->IsA()->InheritsFrom(RooAbsPdf::Class()) && 
	!parg->IsA()->InheritsFrom(RooRealVar::Class())) {
      funcSet.add(*parg) ;
    }

    if (parg->IsA()->InheritsFrom(RooRealVar::Class())) {
      varSet.add(*parg) ;
    }

    if (parg->IsA()->InheritsFrom(RooAbsCategory::Class()) && 
	!parg->IsA()->InheritsFrom(RooCategory::Class())) {
      catfuncSet.add(*parg) ;
    }

    if (parg->IsA()->InheritsFrom(RooCategory::Class())) {
      varSet.add(*parg) ;
    }

  }
  delete iter ;


  if (varSet.getSize()>0) {
    cout << "variables" << endl ;
    cout << "---------" << endl ;
    cout << varSet << endl ;
    cout << endl ;
  }

  if (pdfSet.getSize()>0) {
    cout << "p.d.f.s" << endl ;
    cout << "-------" << endl ;
    iter = pdfSet.createIterator() ;
    while((parg=(RooAbsArg*)iter->Next())) {
      parg->Print() ;
    }
    delete iter ;
    cout << endl ;
  }

  if (funcSet.getSize()>0) {
    cout << "functions" << endl ;
    cout << "--------" << endl ;
    iter = funcSet.createIterator() ;
    while((parg=(RooAbsArg*)iter->Next())) {
      parg->Print() ;
    }
    delete iter ;
    cout << endl ;
  }

  if (catfuncSet.getSize()>0) {
    cout << "category functions" << endl ;
    cout << "------------------" << endl ;
    iter = catfuncSet.createIterator() ;
    while((parg=(RooAbsArg*)iter->Next())) {
      parg->Print() ;
    }
    delete iter ;
    cout << endl ;
  }

  if (_dataList.GetSize()>0) {
    cout << "datasets" << endl ;
    cout << "--------" << endl ;
    iter = _dataList.MakeIterator() ;
    RooAbsData* data2 ;
    while((data2=(RooAbsData*)iter->Next())) {
      cout << data2->IsA()->GetName() << "::" << data2->GetName() << *data2->get() << endl ;
    }
    delete iter ;
    cout << endl ;
  }

  if (_snapshots.GetSize()>0) {
    cout << "parameter snapshots" << endl ;
    cout << "-------------------" << endl ;
    iter = _snapshots.MakeIterator() ;
    RooArgSet* snap ;
    while((snap=(RooArgSet*)iter->Next())) {
      cout << snap->GetName() << " = (" ;
      TIterator* aiter = snap->createIterator() ;
      RooAbsArg* a ;
      Bool_t first(kTRUE) ;
      while((a=(RooAbsArg*)aiter->Next())) {
	if (first) { first=kFALSE ; } else { cout << "," ; }
	cout << a->GetName() << "=" ; 
	a->printValue(cout) ;
	if (a->isConstant()) {
	  cout << "[C]" ;
	}
      }
      cout << ")" << endl ;
      delete aiter ;
    }
    delete iter ;
    cout << endl ;
  }

  if (_classes.listOfClassNames().size()>0) {
    cout << "embedded class code" << endl ;
    cout << "-------------------" << endl ;
    cout << _classes.listOfClassNames() << endl ;
    cout << endl ;
  }

  if (_eocache.size()>0) {
    cout << "embedded precalculated expensive components" << endl ;
    cout << "-------------------------------------------" << endl ;
    _eocache.print() ;
  }


//   if (_views.GetSize()>0) {
//     cout << "views" << endl ;
//     cout << "-----" << endl ;
//     iter = _views.MakeIterator() ;
//     RooModelView* view ;
//     while((view=(RooModelView*)iter->Next())) {
//       view->Print() ;
//     }
//     delete iter ;
//   }

  return ;
}


//_____________________________________________________________________________
void RooWorkspace::CodeRepo::Streamer(TBuffer &R__b)
{
  // Custom streamer for the workspace. Stream contents of workspace
  // and code repository. When reading, read code repository first
  // and compile missing classes before proceeding with streaming
  // of workspace contents to avoid errors.

  typedef ::RooWorkspace::CodeRepo thisClass;

   // Stream an object of class RooWorkspace::CodeRepo.
   if (R__b.IsReading()) {

     UInt_t R__s, R__c;
     R__b.ReadVersion(&R__s, &R__c); 
     
     // Stream contents of ClassFiles map
     Int_t count(0) ;
     R__b >> count ;
     while(count--) {
       TString name ;
       name.Streamer(R__b) ;       
       _fmap[name]._hext.Streamer(R__b) ;
       _fmap[name]._hfile.Streamer(R__b) ;
       _fmap[name]._cxxfile.Streamer(R__b) ;    
     }     
 
     // Stream contents of ClassRelInfo map
     count=0 ;
     R__b >> count ;
     while(count--) {
       TString name,bname,fbase ;
       name.Streamer(R__b) ;       
       _c2fmap[name]._baseName.Streamer(R__b) ;
       _c2fmap[name]._fileBase.Streamer(R__b) ;
     }     
     R__b.CheckByteCount(R__s, R__c, thisClass::IsA());

     // Instantiate any classes that are not defined in current session
     _compiledOK = !compileClasses() ;

   } else {
     
     UInt_t R__c;
     R__c = R__b.WriteVersion(thisClass::IsA(), kTRUE);
     
     // Stream contents of ClassFiles map
     UInt_t count = _fmap.size() ;
     R__b << count ;
     map<TString,ClassFiles>::iterator iter = _fmap.begin() ;
     while(iter!=_fmap.end()) {       
       TString key_copy(iter->first) ;
       key_copy.Streamer(R__b) ;
       iter->second._hext.Streamer(R__b) ;
       iter->second._hfile.Streamer(R__b);
       iter->second._cxxfile.Streamer(R__b);

       ++iter ;
     }
     
     // Stream contents of ClassRelInfo map
     count = _c2fmap.size() ;
     R__b << count ;
     map<TString,ClassRelInfo>::iterator iter2 = _c2fmap.begin() ;
     while(iter2!=_c2fmap.end()) {
       TString key_copy(iter2->first) ;
       key_copy.Streamer(R__b) ;
       iter2->second._baseName.Streamer(R__b) ;
       iter2->second._fileBase.Streamer(R__b);
       ++iter2 ;
     }

     R__b.SetByteCount(R__c, kTRUE);
     
   }
}



//_____________________________________________________________________________
std::string RooWorkspace::CodeRepo::listOfClassNames() const 
{
  // Return STL string with last of class names contained in the code repository

  string ret ;
  map<TString,ClassRelInfo>::const_iterator iter = _c2fmap.begin() ;
  while(iter!=_c2fmap.end()) {
    if (ret.size()>0) {
      ret += ", " ;
    }
    ret += iter->first ;    
    ++iter ;
  }  
  
  return ret ;
}



//_____________________________________________________________________________
Bool_t RooWorkspace::CodeRepo::compileClasses() 
{
  // For all classes in the workspace for which no class definition is
  // found in the ROOT class table extract source code stored in code
  // repository into temporary directory set by
  // setClassFileExportDir(), compile classes and link them with
  // current ROOT session. If a compilation error occurs print
  // instructions for user how to fix errors and recover workspace and
  // abort import procedure.

  Bool_t haveDir=kFALSE ;

  // Retrieve name of directory in which to export code files
  string dirName = Form(_classFileExportDir.c_str(),_wspace->GetName()) ;

  // Process all class entries in repository
  map<TString,ClassRelInfo>::iterator iter = _c2fmap.begin() ;
  while(iter!=_c2fmap.end()) {

    oocxcoutD(_wspace,ObjectHandling) << "RooWorkspace::CodeRepo::compileClasses() now processing class " << iter->first.Data() << endl ;

    // If class is already known, don't load
    if (gClassTable->GetDict(iter->first.Data())) {
      oocoutI(_wspace,ObjectHandling) << "RooWorkspace::CodeRepo::compileClasses() Embedded class " 
				      << iter->first << " already in ROOT class table, skipping" << endl ;
      ++iter ;
      continue ;
    }

    // Check that export directory exists
    if (!haveDir) {

      // If not, make local directory to extract files 
      if (!gSystem->AccessPathName(dirName.c_str())) {
	oocoutI(_wspace,ObjectHandling) << "RooWorkspace::CodeRepo::compileClasses() reusing code export directory " << dirName.c_str() 
					<< " to extract coded embedded in workspace" << endl ;
      } else {
	if (gSystem->MakeDirectory(dirName.c_str())==0) { 
	  oocoutI(_wspace,ObjectHandling) << "RooWorkspace::CodeRepo::compileClasses() creating code export directory " << dirName.c_str() 
					  << " to extract coded embedded in workspace" << endl ;
	} else {
	  oocoutE(_wspace,ObjectHandling) << "RooWorkspace::CodeRepo::compileClasses() ERROR creating code export directory " << dirName.c_str() 
					  << " to extract coded embedded in workspace" << endl ;
	  return kFALSE ;
	}
      }
      haveDir=kTRUE ;
    }

    // Navigate from class to file
    ClassFiles& cfinfo = _fmap[iter->second._fileBase] ;

    oocxcoutD(_wspace,ObjectHandling) << "RooWorkspace::CodeRepo::compileClasses() now processing file with base " << iter->second._fileBase << endl ;
    
    // If file is already processed, skip to next class
    if (cfinfo._extracted) {
      oocxcoutD(_wspace,ObjectHandling) << "RooWorkspace::CodeRepo::compileClasses() file with base name " << iter->second._fileBase 
					 << " has already been extracted, skipping to next class" << endl ;
      continue ;
    }

    // Check if identical declaration file (header) is already written
    Bool_t needDeclWrite=kTRUE ;
    string fdname = Form("%s/%s.%s",dirName.c_str(),iter->second._fileBase.Data(),cfinfo._hext.Data()) ;
    ifstream ifdecl(fdname.c_str()) ;
    if (ifdecl) {
      TString contents ;
      char buf[1024] ;
      while(ifdecl.getline(buf,1024)) {
	contents += buf ;
	contents += "\n" ;
      }      
      UInt_t crcFile = RooAbsArg::crc32(contents.Data()) ;
      UInt_t crcWS   = RooAbsArg::crc32(cfinfo._hfile.Data()) ;
      needDeclWrite = (crcFile!=crcWS) ;
    }

    // Write declaration file if required 
    if (needDeclWrite) {
      oocoutI(_wspace,ObjectHandling) << "RooWorkspace::CodeRepo::compileClasses() Extracting declaration code of class " << iter->first << ", file " << fdname << endl ;
      ofstream fdecl(fdname.c_str()) ;
      if (!fdecl) {
	oocoutE(_wspace,ObjectHandling) << "RooWorkspace::CodeRepo::compileClasses() ERROR opening file" 
					<< fdname << " for writing" << endl ;
	return kFALSE ;
      }
      fdecl << cfinfo._hfile ;
      fdecl.close() ;
    }

    // Check if identical implementation file is already written
    Bool_t needImplWrite=kTRUE ;
    string finame = Form("%s/%s.cxx",dirName.c_str(),iter->second._fileBase.Data()) ;
    ifstream ifimpl(finame.c_str()) ;
    if (ifimpl) {
      TString contents ;
      char buf[1024] ;
      while(ifimpl.getline(buf,1024)) {
	contents += buf ;
	contents += "\n" ;
      }      
      UInt_t crcFile = RooAbsArg::crc32(contents.Data()) ;
      UInt_t crcWS   = RooAbsArg::crc32(cfinfo._cxxfile.Data()) ;
      needImplWrite = (crcFile!=crcWS) ;
    }

    // Write implementation file if required
    if (needImplWrite) {
      oocoutI(_wspace,ObjectHandling) << "RooWorkspace::CodeRepo::compileClasses() Extracting implementation code of class " << iter->first << ", file " << finame << endl ;
      ofstream fimpl(finame.c_str()) ;
      if (!fimpl) {
	oocoutE(_wspace,ObjectHandling) << "RooWorkspace::CodeRepo::compileClasses() ERROR opening file" 
					<< finame << " for writing" << endl ;
	return kFALSE ;
      }
      fimpl << cfinfo._cxxfile ;
      fimpl.close() ;
    }

    // Mark this file as extracted
    cfinfo._extracted = kTRUE ;
    oocxcoutD(_wspace,ObjectHandling) << "RooWorkspace::CodeRepo::compileClasses() marking code unit  " << iter->second._fileBase << " as extracted" << endl ;

    // Compile class
    oocoutI(_wspace,ObjectHandling) << "RooWorkspace::CodeRepo::compileClasses() Compiling code unit " << iter->second._fileBase.Data() << " to define class " << iter->first << endl ;
    Bool_t ok = gSystem->CompileMacro(finame.c_str(),"k") ;
    
    if (!ok) {
      oocoutE(_wspace,ObjectHandling) << "RooWorkspace::CodeRepo::compileClasses() ERROR compiling class " << iter->first.Data() << ", to fix this you can do the following: " << endl 
				      << "  1) Fix extracted source code files in directory " << dirName.c_str() << "/" << endl 
				      << "  2) In clean ROOT session compiled fixed classes by hand using '.x " << dirName.c_str() << "/ClassName.cxx+'" << endl
				      << "  3) Reopen file with RooWorkspace with broken source code in UPDATE mode. Access RooWorkspace to force loading of class" << endl
				      << "     Broken instances in workspace will _not_ be compiled, instead precompiled fixed instances will be used." << endl
				      << "  4) Reimport fixed code in workspace using 'RooWorkspace::importClassCode(\"*\",kTRUE)' method, Write() updated workspace to file and close file" << endl
				      << "  5) Reopen file in clean ROOT session to confirm that problems are fixed" << endl ;
	return kFALSE ;
    }
    
    ++iter ;
  }

  return kTRUE ;
}



//_____________________________________________________________________________
void RooWorkspace::WSDir::InternalAppend(TObject* obj) 
{
  // Internal access to TDirectory append method

#if ROOT_VERSION_CODE <= ROOT_VERSION(5,19,02)
  TDirectory::Append(obj) ;
#else
  TDirectory::Append(obj,kFALSE) ;
#endif

}


//_____________________________________________________________________________
#if ROOT_VERSION_CODE <= ROOT_VERSION(5,19,02)
void RooWorkspace::WSDir::Add(TObject*) 
#else
void RooWorkspace::WSDir::Add(TObject*,Bool_t) 
#endif
{
  // Overload TDirectory interface method to prohibit insertion of objects in read-only directory workspace representation
  coutE(ObjectHandling) << "RooWorkspace::WSDir::Add(" << GetName() << ") ERROR: Directory is read-only representation of a RooWorkspace, use RooWorkspace::import() to add objects" << endl ;
} 


//_____________________________________________________________________________
#if ROOT_VERSION_CODE <= ROOT_VERSION(5,19,02)
void RooWorkspace::WSDir::Append(TObject*) 
#else
void RooWorkspace::WSDir::Append(TObject*,Bool_t) 
#endif
{
  // Overload TDirectory interface method to prohibit insertion of objects in read-only directory workspace representation
  coutE(ObjectHandling) << "RooWorkspace::WSDir::Add(" << GetName() << ") ERROR: Directory is read-only representation of a RooWorkspace, use RooWorkspace::import() to add objects" << endl ;
}
