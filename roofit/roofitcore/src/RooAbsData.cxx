/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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

//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// RooAbsData is the common abstract base class for binned and unbinned
// datasets. The abstract interface defines plotting and tabulating entry
// points for its contents and provides an iterator over its elements
// (bins for binned data sets, data points for unbinned datasets).
// END_HTML
//
//

#include "RooFit.h"
#include "Riostream.h"

#include "TClass.h"
#include "TMath.h"

#include "RooAbsData.h"
#include "RooAbsData.h"
#include "RooFormulaVar.h"
#include "RooCmdConfig.h"
#include "RooAbsRealLValue.h"
#include "RooMsgService.h"
#include "RooMultiCategory.h"
#include "Roo1DTable.h"
#include "RooAbsDataStore.h"

#include "RooRealVar.h"
#include "RooGlobalFunc.h"
#include "RooPlot.h"
#include "RooCurve.h"
#include "RooHist.h"

#include "TMatrixDSym.h"
#include "TPaveText.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"


ClassImp(RooAbsData)
;

static std::set<RooAbsData*> _dlist ;

//_____________________________________________________________________________
Bool_t RooAbsData::isAlive(RooAbsData* data) 
{
  return _dlist.find(data)!=_dlist.end() ;
}



//_____________________________________________________________________________
RooAbsData::RooAbsData() 
{
  // Default constructor
  _dlist.insert(this) ;
  _dstore = 0 ;
  _iterator = _vars.createIterator() ;
  _cacheIter = _cachedVars.createIterator() ;
}



//_____________________________________________________________________________
RooAbsData::RooAbsData(const char *name, const char *title, const RooArgSet& vars, RooAbsDataStore* dstore) :
  TNamed(name,title), _vars("Dataset Variables"), _cachedVars("Cached Variables"), _dstore(dstore)
{
  // Constructor from a set of variables. Only fundamental elements of vars
  // (RooRealVar,RooCategory etc) are stored as part of the dataset

  _dlist.insert(this) ;

  // clone the fundamentals of the given data set into internal buffer
  TIterator* iter = vars.createIterator() ;
  RooAbsArg *var;
  while((0 != (var= (RooAbsArg*)iter->Next()))) {
    if (!var->isFundamental()) {
      coutE(InputArguments) << "RooAbsDataStore::initialize(" << GetName() 
			    << "): Data set cannot contain non-fundamental types, ignoring " 
			    << var->GetName() << endl ;
    } else {
      _vars.addClone(*var);
    }
  }
  delete iter ;

  // reconnect any paramaterized ranges to internal dataset observables
  iter = _vars.createIterator() ;
  while((0 != (var= (RooAbsArg*)iter->Next()))) {
    var->attachDataSet(*this) ;
  } 
  delete iter ;

  _iterator= _vars.createIterator();
  _cacheIter = _cachedVars.createIterator() ;
}



//_____________________________________________________________________________
RooAbsData::RooAbsData(const RooAbsData& other, const char* newname) : 
  TNamed(newname?newname:other.GetName(),other.GetTitle()), 
  RooPrintable(other), _vars(),
  _cachedVars("Cached Variables")
{
  // Copy constructor

  _dlist.insert(this) ;
  _vars.addClone(other._vars) ;

  // reconnect any paramaterized ranges to internal dataset observables
  TIterator* iter = _vars.createIterator() ;
  RooAbsArg* var ;
  while((0 != (var= (RooAbsArg*)iter->Next()))) {
    var->attachDataSet(*this) ;
  } 
  delete iter ;


  _iterator= _vars.createIterator();
  _cacheIter = _cachedVars.createIterator() ;

  _dstore = other._dstore->clone(_vars,newname?newname:other.GetName()) ;
  
}



//_____________________________________________________________________________
RooAbsData::~RooAbsData() 
{
  // Destructor
  _dlist.erase(this) ;
  
  // delete owned contents.
  delete _dstore ;
  delete _iterator ;
  delete _cacheIter ;
}



//_____________________________________________________________________________
Bool_t RooAbsData::changeObservableName(const char* from, const char* to)
{
  return _dstore->changeObservableName(from,to) ;
}





//_____________________________________________________________________________
void RooAbsData::fill()
{
  _dstore->fill() ;
}





//_____________________________________________________________________________
Int_t RooAbsData::numEntries() const
{
  return _dstore->numEntries() ;
}





//_____________________________________________________________________________
void RooAbsData::reset()
{
  _dstore->reset() ;
}




//_____________________________________________________________________________
const RooArgSet* RooAbsData::get(Int_t index) const 
{
  checkInit() ;
  return _dstore->get(index) ;
}




//_____________________________________________________________________________
void RooAbsData::cacheArgs(const RooAbsArg* cacheOwner, RooArgSet& varSet, const RooArgSet* nset) 
{
  // Internal method -- Cache given set of functions with data
  _dstore->cacheArgs(cacheOwner,varSet,nset) ;
}





//_____________________________________________________________________________
void RooAbsData::resetCache() 
{
  // Internal method -- Remove cached function values
  _dstore->resetCache() ;
  _cachedVars.removeAll() ;
}



//_____________________________________________________________________________
void RooAbsData::attachCache(const RooAbsArg* newOwner, const RooArgSet& cachedVars) 
{
  // Internal method -- Attach dataset copied with cache contents to copied instances of functions
  _dstore->attachCache(newOwner, cachedVars) ;
}








//_____________________________________________________________________________
void RooAbsData::setArgStatus(const RooArgSet& set, Bool_t active) 
{
  _dstore->setArgStatus(set,active) ;
}




//_____________________________________________________________________________
void RooAbsData::setDirtyProp(Bool_t flag) 
{ 
  // Control propagation of dirty flags from observables in dataset
  _dstore->setDirtyProp(flag) ; 
}

  




//_____________________________________________________________________________
RooAbsData* RooAbsData::reduce(const RooCmdArg& arg1,const RooCmdArg& arg2,const RooCmdArg& arg3,const RooCmdArg& arg4,
			       const RooCmdArg& arg5,const RooCmdArg& arg6,const RooCmdArg& arg7,const RooCmdArg& arg8) 
{
  // Create a reduced copy of this dataset. The caller takes ownership of the returned dataset
  //
  // The following optional named arguments are accepted
  //
  //   SelectVars(const RooArgSet& vars) -- Only retain the listed observables in the output dataset
  //   Cut(const char* expression)       -- Only retain event surviving the given cut expression
  //   Cut(const RooFormulaVar& expr)    -- Only retain event surviving the given cut formula
  //   CutRange(const char* name)        -- Only retain events inside range with given name. Multiple CutRange
  //                                        arguments may be given to select multiple ranges
  //   EventRange(int lo, int hi)        -- Only retain events with given sequential event numbers
  //   Name(const char* name)            -- Give specified name to output dataset
  //   Title(const char* name)           -- Give specified title to output dataset
  //

  // Define configuration for this method
  RooCmdConfig pc(Form("RooAbsData::reduce(%s)",GetName())) ;
  pc.defineString("name","Name",0,"") ;
  pc.defineString("title","Title",0,"") ;
  pc.defineString("cutRange","CutRange",0,"") ;
  pc.defineString("cutSpec","CutSpec",0,"") ;
  pc.defineObject("cutVar","CutVar",0,0) ;
  pc.defineInt("evtStart","EventRange",0,0) ;
  pc.defineInt("evtStop","EventRange",1,2000000000) ;
  pc.defineObject("varSel","SelectVars",0,0) ;
  pc.defineMutex("CutVar","CutSpec") ;

  // Process & check varargs 
  pc.process(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
  if (!pc.ok(kTRUE)) {
    return 0 ;
  }

  // Extract values from named arguments
  const char* cutRange = pc.getString("cutRange",0,kTRUE) ;
  const char* cutSpec = pc.getString("cutSpec",0,kTRUE) ;
  RooFormulaVar* cutVar = static_cast<RooFormulaVar*>(pc.getObject("cutVar",0)) ;
  Int_t nStart = pc.getInt("evtStart",0) ;
  Int_t nStop = pc.getInt("evtStop",2000000000) ;
  RooArgSet* varSet = static_cast<RooArgSet*>(pc.getObject("varSel")) ;
  const char* name = pc.getString("name",0,kTRUE) ;
  const char* title = pc.getString("title",0,kTRUE) ;

  // Make sure varSubset doesn't contain any variable not in this dataset
  RooArgSet varSubset ;
  if (varSet) {
    varSubset.add(*varSet) ;
    TIterator* iter = varSubset.createIterator() ;
    RooAbsArg* arg ;
    while((arg=(RooAbsArg*)iter->Next())) {
      if (!_vars.find(arg->GetName())) {
	coutW(InputArguments) << "RooAbsData::reduce(" << GetName() << ") WARNING: variable " 
			      << arg->GetName() << " not in dataset, ignored" << endl ;
	varSubset.remove(*arg) ;
      }
    }
    delete iter ;    
  } else {
    varSubset.add(*get()) ;
  }

  RooAbsData* ret = 0 ;
  if (cutSpec) {

    RooFormulaVar cutVarTmp(cutSpec,cutSpec,*get()) ;
    ret =  reduceEng(varSubset,&cutVarTmp,cutRange,nStart,nStop,kFALSE) ;      

  } else if (cutVar) {

    ret = reduceEng(varSubset,cutVar,cutRange,nStart,nStop,kFALSE) ;

  } else {

    ret = reduceEng(varSubset,0,cutRange,nStart,nStop,kFALSE) ;

  }
  
  if (!ret) return 0 ;

  if (name) {
    ret->SetName(name) ;
  }
  if (title) {
    ret->SetTitle(title) ;
  }

  return ret ;
}



//_____________________________________________________________________________
RooAbsData* RooAbsData::reduce(const char* cut) 
{ 
  // Create a subset of the data set by applying the given cut on the data points.
  // The cut expression can refer to any variable in the data set. For cuts involving 
  // other variables, such as intermediate formula objects, use the equivalent 
  // reduce method specifying the as a RooFormulVar reference.

  RooFormulaVar cutVar(cut,cut,*get()) ;
  return reduceEng(*get(),&cutVar,0,0,2000000000,kFALSE) ;
}



//_____________________________________________________________________________
RooAbsData* RooAbsData::reduce(const RooFormulaVar& cutVar) 
{
  // Create a subset of the data set by applying the given cut on the data points.
  // The 'cutVar' formula variable is used to select the subset of data points to be 
  // retained in the reduced data collection.

  return reduceEng(*get(),&cutVar,0,0,2000000000,kFALSE) ;
}



//_____________________________________________________________________________
RooAbsData* RooAbsData::reduce(const RooArgSet& varSubset, const char* cut) 
{
  // Create a subset of the data set by applying the given cut on the data points
  // and reducing the dimensions to the specified set.
  // 
  // The cut expression can refer to any variable in the data set. For cuts involving 
  // other variables, such as intermediate formula objects, use the equivalent 
  // reduce method specifying the as a RooFormulVar reference.

  // Make sure varSubset doesn't contain any variable not in this dataset
  RooArgSet varSubset2(varSubset) ;
  TIterator* iter = varSubset.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (!_vars.find(arg->GetName())) {
      coutW(InputArguments) << "RooAbsData::reduce(" << GetName() << ") WARNING: variable " 
			    << arg->GetName() << " not in dataset, ignored" << endl ;
      varSubset2.remove(*arg) ;
    }
  }
  delete iter ;

  if (cut && strlen(cut)>0) {
    RooFormulaVar cutVar(cut,cut,*get()) ;
    return reduceEng(varSubset2,&cutVar,0,0,2000000000,kFALSE) ;      
  } 
  return reduceEng(varSubset2,0,0,0,2000000000,kFALSE) ;
}



//_____________________________________________________________________________
RooAbsData* RooAbsData::reduce(const RooArgSet& varSubset, const RooFormulaVar& cutVar) 
{
  // Create a subset of the data set by applying the given cut on the data points
  // and reducing the dimensions to the specified set.
  // 
  // The 'cutVar' formula variable is used to select the subset of data points to be 
  // retained in the reduced data collection.

  // Make sure varSubset doesn't contain any variable not in this dataset
  RooArgSet varSubset2(varSubset) ;
  TIterator* iter = varSubset.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (!_vars.find(arg->GetName())) {
      coutW(InputArguments) << "RooAbsData::reduce(" << GetName() << ") WARNING: variable " 
			    << arg->GetName() << " not in dataset, ignored" << endl ;
      varSubset2.remove(*arg) ;
    }
  }
  delete iter ;

  return reduceEng(varSubset2,&cutVar,0,0,2000000000,kFALSE) ;
}



//_____________________________________________________________________________
Double_t RooAbsData::weightError(ErrorType) const 
{ 
  // Return error on current weight (dummy implementation returning zero)
  return 0 ; 
} 



//_____________________________________________________________________________
void RooAbsData::weightError(Double_t& lo, Double_t& hi, ErrorType) const 
{ 
  // Return asymmetric error on weight. (Dummy implementation returning zero)
  lo=0 ; hi=0 ; 
} 



//_____________________________________________________________________________
RooPlot* RooAbsData::plotOn(RooPlot* frame, const RooCmdArg& arg1, const RooCmdArg& arg2,
			    const RooCmdArg& arg3, const RooCmdArg& arg4, const RooCmdArg& arg5, 
			    const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8) const
{
  // Plot dataset on specified frame. By default an unbinned dataset will use the default binning of
  // the target frame. A binned dataset will by default retain its intrinsic binning.
  //
  // The following optional named arguments can be used to modify the default behavior
  //
  // Data representation options
  // ---------------------------
  // Asymmetry(const RooCategory& c) -- Show the asymmetry of the daya in given two-state category [F(+)-F(-)] / [F(+)+F(-)]. 
  //                                    Category must have two states with indices -1 and +1 or three states with indeces -1,0 and +1.
  // DataError(RooAbsData::EType)    -- Select the type of error drawn: Poisson (default) draws asymmetric Poisson
  //                                    confidence intervals. SumW2 draws symmetric sum-of-weights error
  // Binning(double xlo, double xhi, -- Use specified binning to draw dataset
  //                      int nbins)
  // Binning(const RooAbsBinning&)   -- Use specified binning to draw dataset
  // Binning(const char* name)       -- Use binning with specified name to draw dataset
  // RefreshNorm(Bool_t flag)        -- Force refreshing for PDF normalization information in frame.
  //                                    If set, any subsequent PDF will normalize to this dataset, even if it is
  //                                    not the first one added to the frame. By default only the 1st dataset
  //                                    added to a frame will update the normalization information
  // Rescale(Double_t factor)        -- Apply global rescaling factor to histogram
  //
  // Histogram drawing options
  // -------------------------
  // DrawOption(const char* opt)     -- Select ROOT draw option for resulting TGraph object
  // LineStyle(Int_t style)          -- Select line style by ROOT line style code, default is solid
  // LineColor(Int_t color)          -- Select line color by ROOT color code, default is black
  // LineWidth(Int_t width)          -- Select line with in pixels, default is 3
  // MarkerStyle(Int_t style)        -- Select the ROOT marker style, default is 21
  // MarkerColor(Int_t color)        -- Select the ROOT marker color, default is black
  // MarkerSize(Double_t size)       -- Select the ROOT marker size
  // XErrorSize(Double_t frac)       -- Select size of X error bar as fraction of the bin width, default is 1
  //
  //
  // Misc. other options
  // -------------------
  // Name(const chat* name)          -- Give curve specified name in frame. Useful if curve is to be referenced later
  // Invisble(Bool_t flag)           -- Add curve to frame, but do not display. Useful in combination AddTo()
  // AddTo(const char* name,         -- Add constructed histogram to already existing histogram with given name and relative weight factors
  // double_t wgtSelf, double_t wgtOther)
  // 
  //                                    
  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;  
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;  
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;
  return plotOn(frame,l) ;  
}




//_____________________________________________________________________________
TH1 *RooAbsData::createHistogram(const char* varNameList, Int_t xbins, Int_t ybins, Int_t zbins) const
{
  // Create and fill a ROOT histogram TH1,TH2 or TH3 with the values of this dataset for the variables with given names
  // The range of each observable that is histogrammed is always automatically calculated from the distribution in
  // the dataset. The number of bins can be controlled using the [xyz]bins parameters. For a greater degree of control
  // use the createHistogram() method below with named arguments
  //
  // The caller takes ownership of the returned histogram

  // Parse list of variable names
  char buf[1024] ;
  strlcpy(buf,varNameList,1024) ;
  char* varName = strtok(buf,",:") ;
  
  RooRealVar* xvar = (RooRealVar*) get()->find(varName) ;
  if (!xvar) {
    coutE(InputArguments) << "RooAbsData::createHistogram(" << GetName() << ") ERROR: dataset does not contain an observable named " << varName << endl ;
    return 0 ;
  }
  varName = strtok(0,",") ; 
  RooRealVar* yvar = varName ? (RooRealVar*) get()->find(varName) : 0 ;
  if (varName && !yvar) {
    coutE(InputArguments) << "RooAbsData::createHistogram(" << GetName() << ") ERROR: dataset does not contain an observable named " << varName << endl ;
    return 0 ;
  }
  varName = strtok(0,",") ;  
  RooRealVar* zvar = varName ? (RooRealVar*) get()->find(varName) : 0 ;
  if (varName && !zvar) {
    coutE(InputArguments) << "RooAbsData::createHistogram(" << GetName() << ") ERROR: dataset does not contain an observable named " << varName << endl ;
    return 0 ;
  }

  // Construct list of named arguments to pass to the implementation version of createHistogram()

  RooLinkedList argList ; 
  if (xbins<=0  || !xvar->hasMax() || !xvar->hasMin() ) {
    argList.Add(RooFit::AutoBinning(xbins==0?xvar->numBins():abs(xbins)).Clone()) ;
  } else {
    argList.Add(RooFit::Binning(xbins).Clone()) ;
  }
 
  if (yvar) {        
    if (ybins<=0 || !yvar->hasMax() || !yvar->hasMin() ) {
      argList.Add(RooFit::YVar(*yvar,RooFit::AutoBinning(ybins==0?yvar->numBins():abs(ybins))).Clone()) ;
    } else {
      argList.Add(RooFit::YVar(*yvar,RooFit::Binning(ybins)).Clone()) ;
    }
  }

  if (zvar) {    
    if (zbins<=0 || !zvar->hasMax() || !zvar->hasMin() ) {
      argList.Add(RooFit::ZVar(*zvar,RooFit::AutoBinning(zbins==0?zvar->numBins():abs(zbins))).Clone()) ;
    } else {
      argList.Add(RooFit::ZVar(*zvar,RooFit::Binning(zbins)).Clone()) ;
    }
  }



  // Call implementation function
  TH1* result = createHistogram(GetName(),*xvar,argList) ;

  // Delete temporary list of RooCmdArgs 
  argList.Delete() ;

  return result ;
}



//_____________________________________________________________________________
TH1 *RooAbsData::createHistogram(const char *name, const RooAbsRealLValue& xvar,
				 const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3, const RooCmdArg& arg4, 
				 const RooCmdArg& arg5, const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8) const 
{
  // Create and fill a ROOT histogram TH1,TH2 or TH3 with the values of this dataset. 
  //
  // This function accepts the following arguments
  //
  // name -- Name of the ROOT histogram
  // xvar -- Observable to be mapped on x axis of ROOT histogram
  //
  // AutoBinning(Int_t nbins, Double_y margin)    -- Automatically calculate range with given added fractional margin, set binning to nbins
  // AutoSymBinning(Int_t nbins, Double_y margin) -- Automatically calculate range with given added fractional margin, 
  //                                                 with additional constraint that mean of data is in center of range, set binning to nbins
  // Binning(const char* name)                    -- Apply binning with given name to x axis of histogram
  // Binning(RooAbsBinning& binning)              -- Apply specified binning to x axis of histogram
  // Binning(int nbins, double lo, double hi)     -- Apply specified binning to x axis of histogram
  //
  // YVar(const RooAbsRealLValue& var,...)    -- Observable to be mapped on y axis of ROOT histogram
  // ZVar(const RooAbsRealLValue& var,...)    -- Observable to be mapped on z axis of ROOT histogram
  //
  // The YVar() and ZVar() arguments can be supplied with optional Binning() Auto(Sym)Range() arguments to control the binning of the Y and Z axes, e.g.
  // createHistogram("histo",x,Binning(-1,1,20), YVar(y,Binning(-1,1,30)), ZVar(z,Binning("zbinning")))
  //
  // The caller takes ownership of the returned histogram

  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;  
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;  
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;

  return createHistogram(name,xvar,l) ;
}


//_____________________________________________________________________________
TH1 *RooAbsData::createHistogram(const char *name, const RooAbsRealLValue& xvar, const RooLinkedList& argListIn) const
{
  // Internal method that implements histogram filling
  RooLinkedList argList(argListIn) ;
  
  // Define configuration for this method
  RooCmdConfig pc(Form("RooAbsData::createHistogram(%s)",GetName())) ;
  pc.defineString("cutRange","CutRange",0,"",kTRUE) ;
  pc.defineString("cutString","CutSpec",0,"") ;
  pc.defineObject("yvar","YVar",0,0) ;
  pc.defineObject("zvar","ZVar",0,0) ;
  pc.allowUndefined() ;
  
  // Process & check varargs 
  pc.process(argList) ;
  if (!pc.ok(kTRUE)) {
    return 0 ;
  }

  const char* cutSpec = pc.getString("cutString",0,kTRUE) ;
  const char* cutRange = pc.getString("cutRange",0,kTRUE) ;

  RooArgList vars(xvar) ;
  RooAbsArg* yvar = static_cast<RooAbsArg*>(pc.getObject("yvar")) ;
  if (yvar) {
    vars.add(*yvar) ;
  }
  RooAbsArg* zvar = static_cast<RooAbsArg*>(pc.getObject("zvar")) ;
  if (zvar) {
    vars.add(*zvar) ;
  }

  pc.stripCmdList(argList,"CutRange,CutSpec") ;

  // Swap Auto(Sym)RangeData with a Binning command
  RooLinkedList ownedCmds ;
  RooCmdArg* autoRD = (RooCmdArg*) argList.find("AutoRangeData") ;
  if (autoRD) {
    Double_t xmin,xmax ;
    getRange((RooRealVar&)xvar,xmin,xmax,autoRD->getDouble(0),autoRD->getInt(0)) ;
    RooCmdArg* bincmd = (RooCmdArg*) RooFit::Binning(autoRD->getInt(1),xmin,xmax).Clone() ;
    ownedCmds.Add(bincmd) ;
    argList.Replace(autoRD,bincmd) ;
  }

  if (yvar) {
    RooCmdArg* autoRDY = (RooCmdArg*) ((RooCmdArg*)argList.find("YVar"))->subArgs().find("AutoRangeData") ;
    if (autoRDY) {
      Double_t ymin,ymax ;
      getRange((RooRealVar&)(*yvar),ymin,ymax,autoRDY->getDouble(0),autoRDY->getInt(0)) ;
      RooCmdArg* bincmd = (RooCmdArg*) RooFit::Binning(autoRDY->getInt(1),ymin,ymax).Clone() ;
      //ownedCmds.Add(bincmd) ;
      ((RooCmdArg*)argList.find("YVar"))->subArgs().Replace(autoRDY,bincmd) ;
      delete autoRDY ;
    }
  }

  if (zvar) {
    RooCmdArg* autoRDZ = (RooCmdArg*) ((RooCmdArg*)argList.find("ZVar"))->subArgs().find("AutoRangeData") ;
    if (autoRDZ) {
      Double_t zmin,zmax ;
      getRange((RooRealVar&)(*zvar),zmin,zmax,autoRDZ->getDouble(0),autoRDZ->getInt(0)) ;
      RooCmdArg* bincmd = (RooCmdArg*) RooFit::Binning(autoRDZ->getInt(1),zmin,zmax).Clone() ;
      //ownedCmds.Add(bincmd) ;
      ((RooCmdArg*)argList.find("ZVar"))->subArgs().Replace(autoRDZ,bincmd) ;
      delete autoRDZ ;
    }
  }


  TH1* histo = xvar.createHistogram(name,argList) ;
  fillHistogram(histo,vars,cutSpec,cutRange) ;

  ownedCmds.Delete() ;

  return histo ;
}




//_____________________________________________________________________________
Roo1DTable* RooAbsData::table(const RooArgSet& catSet, const char* cuts, const char* opts) const 
{
  // Construct table for product of categories in catSet
  RooArgSet catSet2 ;

  string prodName("(") ;
  TIterator* iter = catSet.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (dynamic_cast<RooAbsCategory*>(arg)) {
      catSet2.add(*arg) ;
      if (prodName.length()>1) {
	prodName += " x " ;
      }
      prodName += arg->GetName() ;
    } else {
      coutW(InputArguments) << "RooAbsData::table(" << GetName() << ") non-RooAbsCategory input argument " << arg->GetName() << " ignored" << endl ;
    }
  }
  prodName += ")" ;
  delete iter ;

  RooMultiCategory tmp(prodName.c_str(),prodName.c_str(),catSet2) ;
  return table(tmp,cuts,opts) ;
}




//_____________________________________________________________________________
void RooAbsData::printName(ostream& os) const 
{
  // Print name of dataset

  os << GetName() ;
}



//_____________________________________________________________________________
void RooAbsData::printTitle(ostream& os) const 
{
  // Print title of dataset
  os << GetTitle() ;
}



//_____________________________________________________________________________
void RooAbsData::printClassName(ostream& os) const 
{
  // Print class name of dataset
  os << IsA()->GetName() ;
}



//_____________________________________________________________________________
void RooAbsData::printMultiline(ostream& os, Int_t contents, Bool_t verbose, TString indent) const
{
  _dstore->printMultiline(os,contents,verbose,indent) ;
}




//_____________________________________________________________________________
Int_t RooAbsData::defaultPrintContents(Option_t* /*opt*/) const 
{
  // Define default print options, for a given print style

  return kName|kClassName|kArgs|kValue ;
}



//_____________________________________________________________________________
Double_t RooAbsData::standMoment(RooRealVar &var, Double_t order, const char* cutSpec, const char* cutRange) const 
{
  // Calculate standardized moment < (X - <X>)^n > / sigma^n,  where n = order.
  // 
  // If cutSpec and/or cutRange are specified
  // the moment is calculated on the subset of the data which pass the C++ cut specification expression 'cutSpec'
  // and/or are inside the range named 'cutRange'

  // Hardwire invariant answer for first and second moment
  if (order==1) return 0 ;
  if (order==2) return 1 ;

  return moment(var,order,cutSpec,cutRange) / TMath::Power(sigma(var,cutSpec,cutRange),order) ; 
}



//_____________________________________________________________________________
Double_t RooAbsData::moment(RooRealVar &var, Double_t order, const char* cutSpec, const char* cutRange) const 
{
  // Calculate moment < (X - <X>)^n > where n = order.
  // 
  // If cutSpec and/or cutRange are specified
  // the moment is calculated on the subset of the data which pass the C++ cut specification expression 'cutSpec'
  // and/or are inside the range named 'cutRange'

  Double_t offset = order>1 ? moment(var,1,cutSpec,cutRange) : 0 ;
  return moment(var,order,offset,cutSpec,cutRange) ;

}





//_____________________________________________________________________________
Double_t RooAbsData::moment(RooRealVar &var, Double_t order, Double_t offset, const char* cutSpec, const char* cutRange) const
{
  // Return the 'order'-ed moment of observable 'var' in this dataset. If offset is non-zero it is subtracted
  // from the values of 'var' prior to the moment calculation. If cutSpec and/or cutRange are specified
  // the moment is calculated on the subset of the data which pass the C++ cut specification expression 'cutSpec'
  // and/or are inside the range named 'cutRange'

  // Lookup variable in dataset
  RooRealVar *varPtr= (RooRealVar*) _vars.find(var.GetName());
  if(0 == varPtr) {
    coutE(InputArguments) << "RooDataSet::moment(" << GetName() << ") ERROR: unknown variable: " << var.GetName() << endl ;
    return 0;
  }

  // Check if found variable is of type RooRealVar
  if (!dynamic_cast<RooRealVar*>(varPtr)) {
    coutE(InputArguments) << "RooDataSet::moment(" << GetName() << ") ERROR: variable " << var.GetName() << " is not of type RooRealVar" << endl ;
    return 0;
  }

  // Check if dataset is not empty
  if(sumEntries() == 0.) {
    coutE(InputArguments) << "RooDataSet::moment(" << GetName() << ") WARNING: empty dataset" << endl ;
    return 0;
  }

  // Setup RooFormulaVar for cutSpec if it is present
  RooFormula* select = 0 ;
  if (cutSpec) {
    select = new RooFormula("select",cutSpec,*get()) ;
  }


  // Calculate requested moment
  Double_t sum(0);
  const RooArgSet* vars ;
  for(Int_t index= 0; index < numEntries(); index++) {
    vars = get(index) ;
    if (select && select->eval()==0) continue ;
    if (cutRange && vars->allInRange(cutRange)) continue ;
    
    sum+= weight() * TMath::Power(varPtr->getVal() - offset,order);
  }
  return sum/sumEntries();
}




//_____________________________________________________________________________
RooRealVar* RooAbsData::dataRealVar(const char* methodname, RooRealVar& extVar) const 
{
  // Internal method to check if given RooRealVar maps to a RooRealVar in this dataset

  // Lookup variable in dataset
  RooRealVar *xdata = (RooRealVar*) _vars.find(extVar.GetName());
  if(!xdata) {
    coutE(InputArguments) << "RooDataSet::" << methodname << "(" << GetName() << ") ERROR: variable : " << extVar.GetName() << " is not in data" << endl ;
    return 0;
  }
  // Check if found variable is of type RooRealVar
  if (!dynamic_cast<RooRealVar*>(xdata)) {
    coutE(InputArguments) << "RooDataSet::" << methodname << "(" << GetName() << ") ERROR: variable : " << extVar.GetName() << " is not of type RooRealVar in data" << endl ;
    return 0;
  }
  return xdata;
}


//_____________________________________________________________________________
Double_t RooAbsData::corrcov(RooRealVar &x,RooRealVar &y, const char* cutSpec, const char* cutRange, Bool_t corr) const 
{
  // Internal method to calculate single correlation and covariance elements

  // Lookup variable in dataset
  RooRealVar *xdata = dataRealVar(corr?"correlation":"covariance",x) ;
  RooRealVar *ydata = dataRealVar(corr?"correlation":"covariance",y) ;
  if (!xdata||!ydata) return 0 ;

  // Check if dataset is not empty
  if(sumEntries() == 0.) {
    coutW(InputArguments) << "RooDataSet::" << (corr?"correlation":"covariance") << "(" << GetName() << ") WARNING: empty dataset, returning zero" << endl ;
    return 0;
  }

  // Setup RooFormulaVar for cutSpec if it is present
  RooFormula* select = cutSpec ? new RooFormula("select",cutSpec,*get()) : 0 ;

  // Calculate requested moment
  Double_t xysum(0),xsum(0),ysum(0),x2sum(0),y2sum(0);
  const RooArgSet* vars ;
  for(Int_t index= 0; index < numEntries(); index++) {
    vars = get(index) ;
    if (select && select->eval()==0) continue ;
    if (cutRange && vars->allInRange(cutRange)) continue ;

    xysum += weight()*xdata->getVal()*ydata->getVal() ;
    xsum += weight()*xdata->getVal() ;
    ysum += weight()*ydata->getVal() ;
    if (corr) {
      x2sum += weight()*xdata->getVal()*xdata->getVal() ;
      y2sum += weight()*ydata->getVal()*ydata->getVal() ;
    }
  }

  // Normalize entries
  xysum/=sumEntries() ;
  xsum/=sumEntries() ;
  ysum/=sumEntries() ;
  if (corr) {
    x2sum/=sumEntries() ;
    y2sum/=sumEntries() ;
  }

  // Cleanup
  if (select) delete select ;

  // Return covariance or correlation as requested
  if (corr) {
    return (xysum-xsum*ysum)/(sqrt(x2sum-(xsum*xsum))*sqrt(y2sum-(ysum*ysum))) ;
  } else {
    return (xysum-xsum*ysum);
  }
}



//_____________________________________________________________________________
TMatrixDSym* RooAbsData::corrcovMatrix(const RooArgList& vars, const char* cutSpec, const char* cutRange, Bool_t corr) const 
{
  // Return covariance matrix from data for given list of observables

  RooArgList varList ;
  TIterator* iter = vars.createIterator() ;
  RooRealVar* var ;
  while((var=(RooRealVar*)iter->Next())) {
    RooRealVar* datavar = dataRealVar("covarianceMatrix",*var) ;
    if (!datavar) {
      delete iter ;
      return 0 ;
    } 
    varList.add(*datavar) ;
  }
  delete iter ;


  // Check if dataset is not empty
  if(sumEntries() == 0.) {
    coutW(InputArguments) << "RooDataSet::covariance(" << GetName() << ") WARNING: empty dataset, returning zero" << endl ;
    return 0;
  }

  // Setup RooFormulaVar for cutSpec if it is present
  RooFormula* select = cutSpec ? new RooFormula("select",cutSpec,*get()) : 0 ;

  iter = varList.createIterator() ;
  TIterator* iter2 = varList.createIterator() ;

  TMatrixDSym xysum(varList.getSize()) ;
  vector<double> xsum(varList.getSize()) ;
  vector<double> x2sum(varList.getSize()) ;

  // Calculate <x_i> and <x_i y_j>
  for(Int_t index= 0; index < numEntries(); index++) {
    const RooArgSet* dvars = get(index) ;
    if (select && select->eval()==0) continue ;
    if (cutRange && dvars->allInRange(cutRange)) continue ;

    RooRealVar* varx, *vary ;
    iter->Reset() ;
    Int_t ix=0,iy=0 ;
    while((varx=(RooRealVar*)iter->Next())) {
      xsum[ix] += weight()*varx->getVal() ;
      if (corr) {
	x2sum[ix] += weight()*varx->getVal()*varx->getVal() ;
      }	

      *iter2=*iter ; iy=ix ;
      vary=varx ;
      while(vary) {
	xysum(ix,iy) += weight()*varx->getVal()*vary->getVal() ;
	xysum(iy,ix) = xysum(ix,iy) ;
	iy++ ;
	vary=(RooRealVar*)iter2->Next() ;
      }
      ix++ ;
    }
    
  }

  // Normalize sums 
  for (Int_t ix=0 ; ix<varList.getSize() ; ix++) {
    xsum[ix] /= sumEntries() ;
    if (corr) {
      x2sum[ix] /= sumEntries() ;
    }
    for (Int_t iy=0 ; iy<varList.getSize() ; iy++) {      
      xysum(ix,iy) /= sumEntries() ;
    }
  }    

  // Calculate covariance matrix
  TMatrixDSym* C = new TMatrixDSym(varList.getSize()) ;
  for (Int_t ix=0 ; ix<varList.getSize() ; ix++) {
    for (Int_t iy=0 ; iy<varList.getSize() ; iy++) {      
      (*C)(ix,iy) = xysum(ix,iy)-xsum[ix]*xsum[iy] ;
      if (corr) {
	(*C)(ix,iy) /= sqrt((x2sum[ix]-(xsum[ix]*xsum[ix]))*(x2sum[iy]-(xsum[iy]*xsum[iy]))) ;
      }
    }    
  }

  if (select) delete select ;
  delete iter ;
  delete iter2 ;

  return C ;
}



//_____________________________________________________________________________
RooRealVar* RooAbsData::meanVar(RooRealVar &var, const char* cutSpec, const char* cutRange) const
{
  // Create a RooRealVar containing the mean of observable 'var' in
  // this dataset.  If cutSpec and/or cutRange are specified the
  // moment is calculated on the subset of the data which pass the C++
  // cut specification expression 'cutSpec' and/or are inside the
  // range named 'cutRange'
  
  // Create a new variable with appropriate strings. The error is calculated as
  // RMS/Sqrt(N) which is generally valid.

  // Create holder variable for mean
  TString name(var.GetName()),title("Mean of ") ;
  name.Append("Mean");
  title.Append(var.GetTitle());
  RooRealVar *meanv= new RooRealVar(name,title,0) ;
  meanv->setConstant(kFALSE) ;

  // Adjust plot label
  TString label("<") ;
  label.Append(var.getPlotLabel());
  label.Append(">");
  meanv->setPlotLabel(label.Data());

  // fill in this variable's value and error
  Double_t meanVal=moment(var,1,0,cutSpec,cutRange) ;
  Double_t N(sumEntries(cutSpec,cutRange)) ;

  Double_t rmsVal= sqrt(moment(var,2,meanVal,cutSpec,cutRange)*N/(N-1));
  meanv->setVal(meanVal) ;
  meanv->setError(N > 0 ? rmsVal/sqrt(N) : 0);

  return meanv;
}



//_____________________________________________________________________________
RooRealVar* RooAbsData::rmsVar(RooRealVar &var, const char* cutSpec, const char* cutRange) const
{
  // Create a RooRealVar containing the RMS of observable 'var' in
  // this dataset.  If cutSpec and/or cutRange are specified the
  // moment is calculated on the subset of the data which pass the C++
  // cut specification expression 'cutSpec' and/or are inside the
  // range named 'cutRange'

  // Create a new variable with appropriate strings. The error is calculated as
  // RMS/(2*Sqrt(N)) which is only valid if the variable has a Gaussian distribution.

  // Create RMS value holder
  TString name(var.GetName()),title("RMS of ") ;
  name.Append("RMS");
  title.Append(var.GetTitle());
  RooRealVar *rms= new RooRealVar(name,title,0) ;
  rms->setConstant(kFALSE) ;

  // Adjust plot label
  TString label(var.getPlotLabel());
  label.Append("_{RMS}");
  rms->setPlotLabel(label);

  // Fill in this variable's value and error
  Double_t meanVal(moment(var,1,0,cutSpec,cutRange)) ;
  Double_t N(sumEntries());
  Double_t rmsVal= sqrt(moment(var,2,meanVal,cutSpec,cutRange)*N/(N-1));
  rms->setVal(rmsVal) ;
  rms->setError(rmsVal/sqrt(2*N));

  return rms;
}


//_____________________________________________________________________________
RooPlot* RooAbsData::statOn(RooPlot* frame, const RooCmdArg& arg1, const RooCmdArg& arg2, 
			    const RooCmdArg& arg3, const RooCmdArg& arg4, const RooCmdArg& arg5, 
			    const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8)
{
  // Add a box with statistics information to the specified frame. By default a box with the
  // event count, mean and rms of the plotted variable is added.
  //
  // The following optional named arguments are accepted
  //
  //   What(const char* whatstr)          -- Controls what is printed: "N" = count, "M" is mean, "R" is RMS.
  //   Format(const char* optStr)         -- Classing [arameter formatting options, provided for backward compatibility
  //   Format(const char* what,...)       -- Parameter formatting options, details given below
  //   Label(const chat* label)           -- Add header label to parameter box
  //   Layout(Double_t xmin,              -- Specify relative position of left,right side of box and top of box. Position of 
  //       Double_t xmax, Double_t ymax)     bottom of box is calculated automatically from number lines in box
  //   Cut(const char* expression)        -- Apply given cut expression to data when calculating statistics
  //   CutRange(const char* rangeName)    -- Only consider events within given range when calculating statistics. Multiple
  //                                         CutRange() argument may be specified to combine ranges
  //
  // The Format(const char* what,...) has the following structure
  //
  //   const char* what          -- Controls what is shown. "N" adds name, "E" adds error, 
  //                                "A" shows asymmetric error, "U" shows unit, "H" hides the value
  //   FixedPrecision(int n)     -- Controls precision, set fixed number of digits
  //   AutoPrecision(int n)      -- Controls precision. Number of shown digits is calculated from error 
  //                                + n specified additional digits (1 is sensible default)
  //   VerbatimName(Bool_t flag) -- Put variable name in a \verb+   + clause.
  //

  // Stuff all arguments in a list
  RooLinkedList cmdList;
  cmdList.Add(const_cast<RooCmdArg*>(&arg1)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg2)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg3)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg4)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg5)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg6)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg7)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg8)) ;

  // Select the pdf-specific commands 
  RooCmdConfig pc(Form("RooTreeData::statOn(%s)",GetName())) ;
  pc.defineString("what","What",0,"MNR") ;
  pc.defineString("label","Label",0,"") ;
  pc.defineDouble("xmin","Layout",0,0.65) ;
  pc.defineDouble("xmax","Layout",1,0.99) ;
  pc.defineInt("ymaxi","Layout",0,Int_t(0.95*10000)) ;
  pc.defineString("formatStr","Format",0,"NELU") ;
  pc.defineInt("sigDigit","Format",0,2) ;
  pc.defineInt("dummy","FormatArgs",0,0) ;
  pc.defineString("cutRange","CutRange",0,"",kTRUE) ;
  pc.defineString("cutString","CutSpec",0,"") ;
  pc.defineMutex("Format","FormatArgs") ;

  // Process and check varargs 
  pc.process(cmdList) ;
  if (!pc.ok(kTRUE)) {
    return frame ;
  }

  const char* label = pc.getString("label") ;
  Double_t xmin = pc.getDouble("xmin") ;
  Double_t xmax = pc.getDouble("xmax") ;
  Double_t ymax = pc.getInt("ymaxi") / 10000. ;
  const char* formatStr = pc.getString("formatStr") ;
  Int_t sigDigit = pc.getInt("sigDigit") ;  
  const char* what = pc.getString("what") ;

  const char* cutSpec = pc.getString("cutString",0,kTRUE) ;
  const char* cutRange = pc.getString("cutRange",0,kTRUE) ;

  if (pc.hasProcessed("FormatArgs")) {
    RooCmdArg* formatCmd = static_cast<RooCmdArg*>(cmdList.FindObject("FormatArgs")) ;
    return statOn(frame,what,label,0,0,xmin,xmax,ymax,cutSpec,cutRange,formatCmd) ;
  } else {
    return statOn(frame,what,label,sigDigit,formatStr,xmin,xmax,ymax,cutSpec,cutRange) ;
  }
}



//_____________________________________________________________________________
RooPlot* RooAbsData::statOn(RooPlot* frame, const char* what, const char *label, Int_t sigDigits,
			     Option_t *options, Double_t xmin, Double_t xmax, Double_t ymax, 
			     const char* cutSpec, const char* cutRange, const RooCmdArg* formatCmd) 
{
  // Implementation back-end of statOn() mehtod with named arguments

  Bool_t showLabel= (label != 0 && strlen(label) > 0);

  TString whatStr(what) ;
  whatStr.ToUpper() ;
  Bool_t showN = whatStr.Contains("N") ;
  Bool_t showR = whatStr.Contains("R") ;
  Bool_t showM = whatStr.Contains("M") ;
  Int_t nPar= 0;
  if (showN) nPar++ ;
  if (showR) nPar++ ;
  if (showM) nPar++ ;

  // calculate the box's size
  Double_t dy(0.06), ymin(ymax-nPar*dy);
  if(showLabel) ymin-= dy;

  // create the box and set its options
  TPaveText *box= new TPaveText(xmin,ymax,xmax,ymin,"BRNDC");
  if(!box) return 0;
  box->SetName(Form("%s_statBox",GetName())) ;
  box->SetFillColor(0);
  box->SetBorderSize(1);
  box->SetTextAlign(12);
  box->SetTextSize(0.04F);
  box->SetFillStyle(1001);

  // add formatted text for each statistic
  RooRealVar N("N","Number of Events",sumEntries(cutSpec,cutRange));
  N.setPlotLabel("Entries") ;
  RooRealVar *meanv= meanVar(*(RooRealVar*)frame->getPlotVar(),cutSpec,cutRange);
  meanv->setPlotLabel("Mean") ;
  RooRealVar *rms= rmsVar(*(RooRealVar*)frame->getPlotVar(),cutSpec,cutRange);
  rms->setPlotLabel("RMS") ;
  TString *rmsText, *meanText, *NText ;
  if (options) {
    rmsText= rms->format(sigDigits,options);
    meanText= meanv->format(sigDigits,options);
    NText= N.format(sigDigits,options);
  } else {
    rmsText= rms->format(*formatCmd);
    meanText= meanv->format(*formatCmd);
    NText= N.format(*formatCmd);
  }
  if (showR) box->AddText(rmsText->Data());
  if (showM) box->AddText(meanText->Data());
  if (showN) box->AddText(NText->Data());

  // cleanup heap memory
  delete NText;
  delete meanText;
  delete rmsText;
  delete meanv;
  delete rms;

  // add the optional label if specified
  if(showLabel) box->AddText(label);

  frame->addObject(box) ;
  return frame ;
}




//_____________________________________________________________________________
TH1 *RooAbsData::fillHistogram(TH1 *hist, const RooArgList &plotVars, const char *cuts, const char* cutRange) const
{
  // Loop over columns of our tree data and fill the input histogram. Returns a pointer to the
  // input histogram, or zero in case of an error. The input histogram can be any TH1 subclass, and
  // therefore of arbitrary dimension. Variables are matched with the (x,y,...) dimensions of the input
  // histogram according to the order in which they appear in the input plotVars list.

  // Do we have a valid histogram to use?
  if(0 == hist) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillHistogram: no valid histogram to fill" << endl;
    return 0;
  }

  // Check that the number of plotVars matches the input histogram's dimension
  Int_t hdim= hist->GetDimension();
  if(hdim != plotVars.getSize()) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillHistogram: plotVars has the wrong dimension" << endl;
    return 0;
  }

  // Check that the plot variables are all actually RooAbsReal's and print a warning if we do not
  // explicitly depend on one of them. Clone any variables that we do not contain directly and
  // redirect them to use our event data.
  RooArgSet plotClones,localVars;
  for(Int_t index= 0; index < plotVars.getSize(); index++) {
    const RooAbsArg *var= plotVars.at(index);
    const RooAbsReal *realVar= dynamic_cast<const RooAbsReal*>(var);
    if(0 == realVar) {
      coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillHistogram: cannot plot variable \"" << var->GetName()
	   << "\" of type " << var->ClassName() << endl;
      return 0;
    }
    RooAbsArg *found= _vars.find(realVar->GetName());
    if(!found) {
      RooAbsArg *clone= plotClones.addClone(*realVar,kTRUE); // do not complain about duplicates
      assert(0 != clone);
      if(!clone->dependsOn(_vars)) {
	coutW(InputArguments) << ClassName() << "::" << GetName()
	     << ":fillHistogram: WARNING: data does not contain variable: " << realVar->GetName() << endl;
      }
      else {
	clone->recursiveRedirectServers(_vars);
      }
      localVars.add(*clone);
    }
    else {
      localVars.add(*found);
    }
  }

  // Create selection formula if selection cuts are specified
  RooFormula* select = 0;
  if(0 != cuts && strlen(cuts)) {
    select=new RooFormula(cuts,cuts,_vars);
    if (!select || !select->ok()) {
      coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillHistogram: invalid cuts \"" << cuts << "\"" << endl;
      delete select;
      return 0 ;
    }
  }
  
  // Lookup each of the variables we are binning in our tree variables
  const RooAbsReal *xvar = 0;
  const RooAbsReal *yvar = 0;
  const RooAbsReal *zvar = 0;
  switch(hdim) {
  case 3:
    zvar= dynamic_cast<RooAbsReal*>(localVars.find(plotVars.at(2)->GetName()));
    assert(0 != zvar);
    // fall through to next case...
  case 2:
    yvar= dynamic_cast<RooAbsReal*>(localVars.find(plotVars.at(1)->GetName()));
    assert(0 != yvar);
    // fall through to next case...
  case 1:
    xvar= dynamic_cast<RooAbsReal*>(localVars.find(plotVars.at(0)->GetName()));
    assert(0 != xvar);
    break;
  default:
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillHistogram: cannot fill histogram with "
	 << hdim << " dimensions" << endl;
    break;
  }

  // Parse cutRange specification
  vector<string> cutVec ;
  if (cutRange && strlen(cutRange)>0) {
    if (strchr(cutRange,',')==0) {
      cutVec.push_back(cutRange) ;
    } else {
      char* buf = new char[strlen(cutRange)+1] ;
      strlcpy(buf,cutRange,strlen(cutRange)+1) ;
      const char* oneRange = strtok(buf,",") ;
      while(oneRange) {
	cutVec.push_back(oneRange) ;
	oneRange = strtok(0,",") ;
      }
      delete[] buf ;
    }
  }

  // Loop over events and fill the histogram  
  Int_t nevent= numEntries() ; //(Int_t)_tree->GetEntries();
  for(Int_t i=0; i < nevent; ++i) {

    //Int_t entryNumber= _tree->GetEntryNumber(i);
    //if (entryNumber<0) break;
    get(i);

    // Apply expression based selection criteria
    if (select && select->eval()==0) {
      continue ;
    }


    // Apply range based selection criteria
    Bool_t selectByRange = kTRUE ;
    if (cutRange) {
      _iterator->Reset() ;
      RooAbsArg* arg ;
      while((arg=(RooAbsArg*)_iterator->Next())) {
	Bool_t selectThisArg = kFALSE ;
	UInt_t icut ;
	for (icut=0 ; icut<cutVec.size() ; icut++) {
	  if (arg->inRange(cutVec[icut].c_str())) {
	    selectThisArg = kTRUE ;
	    break ;
	  }
	}
	if (!selectThisArg) {
	  selectByRange = kFALSE ;
	  break ;
	}
      }
    }

    if (!selectByRange) {
      // Go to next event in loop over events
      continue ;
    }

    Int_t bin(0);
    switch(hdim) {
    case 1:
      bin= hist->FindBin(xvar->getVal());
      hist->Fill(xvar->getVal(),weight()) ;
      break;
    case 2:
      bin= hist->FindBin(xvar->getVal(),yvar->getVal());
      static_cast<TH2*>(hist)->Fill(xvar->getVal(),yvar->getVal(),weight()) ;
      break;
    case 3:
      bin= hist->FindBin(xvar->getVal(),yvar->getVal(),zvar->getVal());
      static_cast<TH3*>(hist)->Fill(xvar->getVal(),yvar->getVal(),zvar->getVal(),weight()) ;
      break;
    default:
      assert(hdim < 3);
      break;
    }

    Double_t error2 = TMath::Power(hist->GetBinError(bin),2)-TMath::Power(weight(),2)  ;
    Double_t we = weightError(RooAbsData::SumW2) ;


    if (we==0) we = weight() ;
    error2 += TMath::Power(we,2) ;
    //hist->AddBinContent(bin,weight());
    hist->SetBinError(bin,sqrt(error2)) ;

    //cout << "RooTreeData::fillHistogram() bin = " << bin << " weight() = " << weight() << " we = " << we << endl ;

  }

  if(0 != select) delete select;

  return hist;
}



//_____________________________________________________________________________
TList* RooAbsData::split(const RooAbsCategory& splitCat, Bool_t createEmptyDataSets) const
{
  // Split dataset into subsets based on states of given splitCat in this dataset.
  // A TList of RooDataSets is returned in which each RooDataSet is named
  // after the state name of splitCat of which it contains the dataset subset.
  // The observables splitCat itself is no longer present in the sub datasets.
  // If createEmptyDataSets is kFALSE (default) this method only creates datasets for states 
  // which have at least one entry The caller takes ownership of the returned list and its contents

  // Sanity check
  if (!splitCat.dependsOn(*get())) {
    coutE(InputArguments) << "RooTreeData::split(" << GetName() << ") ERROR category " << splitCat.GetName() 
	 << " doesn't depend on any variable in this dataset" << endl ;
    return 0 ;
  }

  // Clone splitting category and attach to self
  RooAbsCategory* cloneCat =0;
  RooArgSet* cloneSet = 0;
  if (splitCat.isDerived()) {
    cloneSet = (RooArgSet*) RooArgSet(splitCat).snapshot(kTRUE) ;
    if (!cloneSet) {
      coutE(InputArguments) << "RooTreeData::split(" << GetName() << ") Couldn't deep-clone splitting category, abort." << endl ;
      return 0 ;
    }
    cloneCat = (RooAbsCategory*) cloneSet->find(splitCat.GetName()) ;
    cloneCat->attachDataSet(*this) ;
  } else {
    cloneCat = dynamic_cast<RooAbsCategory*>(get()->find(splitCat.GetName())) ;
    if (!cloneCat) {
      coutE(InputArguments) << "RooTreeData::split(" << GetName() << ") ERROR category " << splitCat.GetName() 
	   << " is fundamental and does not appear in this dataset" << endl ;
      return 0 ;      
    }
  }

  // Split a dataset in a series of subsets, each corresponding
  // to a state of splitCat
  TList* dsetList = new TList ;

  // Construct set of variables to be included in split sets = full set - split category
  RooArgSet subsetVars(*get()) ;
  if (splitCat.isDerived()) {
    RooArgSet* vars = splitCat.getVariables() ;
    subsetVars.remove(*vars,kTRUE,kTRUE) ;
    delete vars ;
  } else {
    subsetVars.remove(splitCat,kTRUE,kTRUE) ;
  }

  // If createEmptyDataSets is true, prepopulate with empty sets corresponding to all states
  if (createEmptyDataSets) {
    TIterator* stateIter = cloneCat->typeIterator() ;
    RooCatType* state ;
    while ((state=(RooCatType*)stateIter->Next())) {
      RooAbsData* subset = emptyClone(state->GetName(),state->GetName(),&subsetVars) ;
      dsetList->Add((RooAbsArg*)subset) ;    
    }
  }

  
  // Loop over dataset and copy event to matching subset
  Int_t i ;
  for (i=0 ; i<numEntries() ; i++) {
    const RooArgSet* row =  get(i) ;
    RooAbsData* subset = (RooAbsData*) dsetList->FindObject(cloneCat->getLabel()) ;
    if (!subset) {
      subset = emptyClone(cloneCat->getLabel(),cloneCat->getLabel(),&subsetVars) ;
      dsetList->Add((RooAbsArg*)subset) ;
    }
    subset->add(*row,weight()) ;
  }

  delete cloneSet ;
  return dsetList ;
}



//_____________________________________________________________________________
RooPlot* RooAbsData::plotOn(RooPlot* frame, const RooLinkedList& argList) const
{
  // Plot dataset on specified frame. By default an unbinned dataset will use the default binning of
  // the target frame. A binned dataset will by default retain its intrinsic binning.
  //
  // The following optional named arguments can be used to modify the default behavior
  //
  // Data representation options
  // ---------------------------
  // Asymmetry(const RooCategory& c) -- Show the asymmetry of the data in given two-state category [F(+)-F(-)] / [F(+)+F(-)]. 
  //                                    Category must have two states with indices -1 and +1 or three states with indeces -1,0 and +1.
  // Efficiency(const RooCategory& c)-- Show the efficiency F(acc)/[F(acc)+F(rej)]. Category must have two states with indices 0 and 1
  // DataError(RooAbsData::EType)    -- Select the type of error drawn: 
  //                                     - Auto(default) results in Poisson for unweighted data and SumW2 for weighted data
  //                                     - Poisson draws asymmetric Poisson confidence intervals. 
  //                                     - SumW2 draws symmetric sum-of-weights error ( sum(w)^2/sum(w^2) )
  //                                     - None draws no error bars
  // Binning(double xlo, double xhi, -- Use specified binning to draw dataset
  //                      int nbins)
  // Binning(const RooAbsBinning&)   -- Use specified binning to draw dataset
  // Binning(const char* name)       -- Use binning with specified name to draw dataset
  // RefreshNorm(Bool_t flag)        -- Force refreshing for PDF normalization information in frame.
  //                                    If set, any subsequent PDF will normalize to this dataset, even if it is
  //                                    not the first one added to the frame. By default only the 1st dataset
  //                                    added to a frame will update the normalization information
  // Rescale(Double_t f)             -- Rescale drawn histogram by given factor
  //
  // Histogram drawing options
  // -------------------------
  // DrawOption(const char* opt)     -- Select ROOT draw option for resulting TGraph object
  // LineStyle(Int_t style)          -- Select line style by ROOT line style code, default is solid
  // LineColor(Int_t color)          -- Select line color by ROOT color code, default is black
  // LineWidth(Int_t width)          -- Select line with in pixels, default is 3
  // MarkerStyle(Int_t style)        -- Select the ROOT marker style, default is 21
  // MarkerColor(Int_t color)        -- Select the ROOT marker color, default is black
  // MarkerSize(Double_t size)       -- Select the ROOT marker size
  // FillStyle(Int_t style)          -- Select fill style, default is filled. 
  // FillColor(Int_t color)          -- Select fill color by ROOT color code
  // XErrorSize(Double_t frac)       -- Select size of X error bar as fraction of the bin width, default is 1
  //
  //
  // Misc. other options
  // -------------------
  // Name(const chat* name)          -- Give curve specified name in frame. Useful if curve is to be referenced later
  // Invisble()                      -- Add curve to frame, but do not display. Useful in combination AddTo()
  // AddTo(const char* name,         -- Add constructed histogram to already existing histogram with given name and relative weight factors
  // double_t wgtSelf, double_t wgtOther)
  // 
  //                                    
  //

  // New experimental plotOn() with varargs...

  // Define configuration for this method
  RooCmdConfig pc(Form("RooTreeData::plotOn(%s)",GetName())) ;
  pc.defineString("drawOption","DrawOption",0,"P") ;
  pc.defineString("cutRange","CutRange",0,"",kTRUE) ;
  pc.defineString("cutString","CutSpec",0,"") ;
  pc.defineString("histName","Name",0,"") ;
  pc.defineObject("cutVar","CutVar",0) ;
  pc.defineObject("binning","Binning",0) ;
  pc.defineString("binningName","BinningName",0,"") ;
  pc.defineInt("nbins","BinningSpec",0,100) ;
  pc.defineDouble("xlo","BinningSpec",0,0) ;
  pc.defineDouble("xhi","BinningSpec",1,1) ;
  pc.defineObject("asymCat","Asymmetry",0) ;
  pc.defineObject("effCat","Efficiency",0) ;
  pc.defineInt("lineColor","LineColor",0,-999) ;
  pc.defineInt("lineStyle","LineStyle",0,-999) ;
  pc.defineInt("lineWidth","LineWidth",0,-999) ;
  pc.defineInt("markerColor","MarkerColor",0,-999) ;
  pc.defineInt("markerStyle","MarkerStyle",0,-999) ;
  pc.defineDouble("markerSize","MarkerSize",0,-999) ;
  pc.defineInt("fillColor","FillColor",0,-999) ;
  pc.defineInt("fillStyle","FillStyle",0,-999) ;
  pc.defineInt("errorType","DataError",0,(Int_t)RooAbsData::Auto) ;
  pc.defineInt("histInvisible","Invisible",0,0) ;
  pc.defineInt("refreshFrameNorm","RefreshNorm",0,1) ;
  pc.defineString("addToHistName","AddTo",0,"") ;
  pc.defineDouble("addToWgtSelf","AddTo",0,1.) ;
  pc.defineDouble("addToWgtOther","AddTo",1,1.) ;
  pc.defineDouble("xErrorSize","XErrorSize",0,1.) ;
  pc.defineDouble("scaleFactor","Rescale",0,1.) ;
  pc.defineMutex("DataError","Asymmetry","Efficiency") ;
  pc.defineMutex("Binning","BinningName","BinningSpec") ;

  // Process & check varargs 
  pc.process(argList) ;
  if (!pc.ok(kTRUE)) {
    return frame ;
  }

  PlotOpt o ;

  // Extract values from named arguments
  o.drawOptions = pc.getString("drawOption") ;
  o.cuts = pc.getString("cutString") ;
  if (pc.hasProcessed("Binning")) {
    o.bins = (RooAbsBinning*) pc.getObject("binning") ;
  } else if (pc.hasProcessed("BinningName")) {
    o.bins = &frame->getPlotVar()->getBinning(pc.getString("binningName")) ;
  } else if (pc.hasProcessed("BinningSpec")) {
    Double_t xlo = pc.getDouble("xlo") ;
    Double_t xhi = pc.getDouble("xhi") ;
    o.bins = new RooUniformBinning((xlo==xhi)?frame->getPlotVar()->getMin():xlo,
				   (xlo==xhi)?frame->getPlotVar()->getMax():xhi,pc.getInt("nbins")) ;
  }
  const RooAbsCategoryLValue* asymCat = (const RooAbsCategoryLValue*) pc.getObject("asymCat") ;
  const RooAbsCategoryLValue* effCat = (const RooAbsCategoryLValue*) pc.getObject("effCat") ;
  o.etype = (RooAbsData::ErrorType) pc.getInt("errorType") ;
  o.histInvisible = pc.getInt("histInvisible") ;
  o.xErrorSize = pc.getDouble("xErrorSize") ;
  o.cutRange = pc.getString("cutRange",0,kTRUE) ;
  o.histName = pc.getString("histName",0,kTRUE) ;
  o.addToHistName = pc.getString("addToHistName",0,kTRUE) ;
  o.addToWgtSelf = pc.getDouble("addToWgtSelf") ;
  o.addToWgtOther = pc.getDouble("addToWgtOther") ;
  o.refreshFrameNorm = pc.getInt("refreshFrameNorm") ;
  o.scaleFactor = pc.getDouble("scaleFactor") ;

  // Map auto error type to actual type
  if (o.etype == Auto) {
    o.etype = isNonPoissonWeighted() ? SumW2 : Poisson ;    
    if (o.etype == SumW2) {
      coutI(InputArguments) << "RooAbsData::plotOn(" << GetName() 
			    << ") INFO: dataset has non-integer weights, auto-selecting SumW2 errors instead of Poisson errors" << endl ;
    }
  }
  
  if (o.addToHistName && !frame->findObject(o.addToHistName,RooHist::Class())) {
    coutE(InputArguments) << "RooAbsData::plotOn(" << GetName() << ") cannot find existing histogram " << o.addToHistName 
			  << " to add to in RooPlot" << endl ;
    return frame ;
  }

  RooPlot* ret ;
  if (!asymCat && !effCat) {
    ret = plotOn(frame,o) ;
  } else if (asymCat) {
    ret = plotAsymOn(frame,*asymCat,o) ;    
  } else {
    ret = plotEffOn(frame,*effCat,o) ;    
  }

  Int_t lineColor   = pc.getInt("lineColor") ;
  Int_t lineStyle   = pc.getInt("lineStyle") ;
  Int_t lineWidth   = pc.getInt("lineWidth") ;
  Int_t markerColor = pc.getInt("markerColor") ;
  Int_t markerStyle = pc.getInt("markerStyle") ;
  Size_t markerSize  = pc.getDouble("markerSize") ;
  Int_t fillColor = pc.getInt("fillColor") ;
  Int_t fillStyle = pc.getInt("fillStyle") ;
  if (lineColor!=-999) ret->getAttLine()->SetLineColor(lineColor) ;
  if (lineStyle!=-999) ret->getAttLine()->SetLineStyle(lineStyle) ;
  if (lineWidth!=-999) ret->getAttLine()->SetLineWidth(lineWidth) ;
  if (markerColor!=-999) ret->getAttMarker()->SetMarkerColor(markerColor) ;
  if (markerStyle!=-999) ret->getAttMarker()->SetMarkerStyle(markerStyle) ;
  if (markerSize!=-999) ret->getAttMarker()->SetMarkerSize(markerSize) ;
  if (fillColor!=-999) ret->getAttFill()->SetFillColor(fillColor) ;
  if (fillStyle!=-999) ret->getAttFill()->SetFillStyle(fillStyle) ;

  if (pc.hasProcessed("BinningSpec")) {
    delete o.bins ;
  }

  return ret ;
}



//_____________________________________________________________________________
RooPlot *RooAbsData::plotOn(RooPlot *frame, PlotOpt o) const 
{
  // Create and fill a histogram of the frame's variable and append it to the frame.
  // The frame variable must be one of the data sets dimensions.
  //
  // The plot range and the number of plot bins is determined by the parameters
  // of the plot variable of the frame (RooAbsReal::setPlotRange(), RooAbsReal::setPlotBins())
  // 
  // The optional cut string expression can be used to select the events to be plotted.
  // The cut specification may refer to any variable contained in the data set
  //
  // The drawOptions are passed to the TH1::Draw() method

  if(0 == frame) {
    coutE(Plotting) << ClassName() << "::" << GetName() << ":plotOn: frame is null" << endl;
    return 0;
  }
  RooAbsRealLValue *var= (RooAbsRealLValue*) frame->getPlotVar();
  if(0 == var) {
    coutE(Plotting) << ClassName() << "::" << GetName()
	 << ":plotOn: frame does not specify a plot variable" << endl;
    return 0;
  }

  // create and fill a temporary histogram of this variable
  TString histName(GetName());
  histName.Append("_plot");
  TH1F *hist ;
    if (o.bins) {
    hist= static_cast<TH1F*>(var->createHistogram(histName.Data(), RooFit::AxisLabel("Events"), RooFit::Binning(*o.bins))) ;
  } else {
    hist= var->createHistogram(histName.Data(), "Events", 
			       frame->GetXaxis()->GetXmin(), frame->GetXaxis()->GetXmax(), frame->GetNbinsX());
  }

  // Keep track of sum-of-weights error
  hist->Sumw2() ;

  if(0 == fillHistogram(hist,RooArgList(*var),o.cuts,o.cutRange)) {
    coutE(Plotting) << ClassName() << "::" << GetName()
	 << ":plotOn: fillHistogram() failed" << endl;
    return 0;
  }

  // If frame has no predefined bin width (event density) it will be adjusted to 
  // our histograms bin width so we should force that bin width here
  Double_t nomBinWidth ;
  if (frame->getFitRangeNEvt()==0 && o.bins) {
    nomBinWidth = o.bins->averageBinWidth() ;
  } else {
    nomBinWidth = o.bins ? frame->getFitRangeBinW() : 0 ;
  }

  // convert this histogram to a RooHist object on the heap
  RooHist *graph= new RooHist(*hist,nomBinWidth,1,o.etype,o.xErrorSize,o.correctForBinWidth,o.scaleFactor); 
  if(0 == graph) {
    coutE(Plotting) << ClassName() << "::" << GetName()
	 << ":plotOn: unable to create a RooHist object" << endl;
    delete hist;
    return 0;
  }  

  // If the dataset variable has a wide range than the plot variable,
  // calculate the number of entries in the dataset in the plot variable fit range
  RooAbsRealLValue* dataVar = (RooAbsRealLValue*) _vars.find(var->GetName()) ;
  Double_t nEnt(sumEntries()) ;
  if (dataVar->getMin()<var->getMin() || dataVar->getMax()>var->getMax()) {
    RooAbsData* tmp = ((RooAbsData*)this)->reduce(*var) ;
    nEnt = tmp->sumEntries() ;
    delete tmp ;
  }

  // Store the number of entries before the cut, if any was made
  if ((o.cuts && strlen(o.cuts)) || o.cutRange) {
    coutI(Plotting) << "RooTreeData::plotOn: plotting " << hist->GetSum() << " events out of " << nEnt << " total events" << endl ;
    graph->setRawEntries(nEnt) ;
  }

  // Add self to other hist if requested
  if (o.addToHistName) {
    RooHist* otherGraph = static_cast<RooHist*>(frame->findObject(o.addToHistName,RooHist::Class())) ;

    if (!graph->hasIdenticalBinning(*otherGraph)) {
      coutE(Plotting) << "RooTreeData::plotOn: ERROR Histogram to be added to, '" << o.addToHistName << "',has different binning" << endl ;
      delete graph ;
      return frame ;
    }

    RooHist* sumGraph = new RooHist(*graph,*otherGraph,o.addToWgtSelf,o.addToWgtOther,o.etype) ;
    delete graph ;
    graph = sumGraph ;
  }  

  // Rename graph if requested
  if (o.histName) {
    graph->SetName(o.histName) ;
  } else {
    TString hname(Form("h_%s",GetName())) ;
    if (o.cutRange && strlen(o.cutRange)>0) {
      hname.Append(Form("_CutRange[%s]",o.cutRange)) ;
    } 
    if (o.cuts && strlen(o.cuts)>0) {
      hname.Append(Form("_Cut[%s]",o.cuts)) ;
    } 
    graph->SetName(hname.Data()) ;
  }
  
  // initialize the frame's normalization setup, if necessary
  frame->updateNormVars(_vars);


  // add the RooHist to the specified plot
  frame->addPlotable(graph,o.drawOptions,o.histInvisible,o.refreshFrameNorm);



  // cleanup
  delete hist;

  return frame;  
}




//_____________________________________________________________________________
RooPlot* RooAbsData::plotAsymOn(RooPlot* frame, const RooAbsCategoryLValue& asymCat, PlotOpt o) const 
{
  // Create and fill a histogram with the asymmetry N[+] - N[-] / ( N[+] + N[-] ),
  // where N(+/-) is the number of data points with asymCat=+1 and asymCat=-1 
  // as function of the frames variable. The asymmetry category 'asymCat' must
  // have exactly 2 (or 3) states defined with index values +1,-1 (and 0)
  // 
  // The plot range and the number of plot bins is determined by the parameters
  // of the plot variable of the frame (RooAbsReal::setPlotRange(), RooAbsReal::setPlotBins())
  // 
  // The optional cut string expression can be used to select the events to be plotted.
  // The cut specification may refer to any variable contained in the data set
  //
  // The drawOptions are passed to the TH1::Draw() method

  if(0 == frame) {
    coutE(Plotting) << ClassName() << "::" << GetName() << ":plotAsymOn: frame is null" << endl;
    return 0;
  }
  RooAbsRealLValue *var= (RooAbsRealLValue*) frame->getPlotVar();
  if(0 == var) {
    coutE(Plotting) << ClassName() << "::" << GetName()
	 << ":plotAsymOn: frame does not specify a plot variable" << endl;
    return 0;
  }

  // create and fill temporary histograms of this variable for each state
  TString hist1Name(GetName()),hist2Name(GetName());
  hist1Name.Append("_plot1");
  TH1F *hist1, *hist2 ;
  hist2Name.Append("_plot2");

  if (o.bins) {
    hist1= var->createHistogram(hist1Name.Data(), "Events", *o.bins) ;
    hist2= var->createHistogram(hist2Name.Data(), "Events", *o.bins) ;
  } else {
    hist1= var->createHistogram(hist1Name.Data(), "Events", 
				frame->GetXaxis()->GetXmin(), frame->GetXaxis()->GetXmax(),
				frame->GetNbinsX());
    hist2= var->createHistogram(hist2Name.Data(), "Events", 
				frame->GetXaxis()->GetXmin(), frame->GetXaxis()->GetXmax(),
				frame->GetNbinsX());
  }

  assert(0 != hist1 && 0 != hist2);

  TString cuts1,cuts2 ;
  if (o.cuts && strlen(o.cuts)) {
    cuts1 = Form("(%s)&&(%s>0)",o.cuts,asymCat.GetName());
    cuts2 = Form("(%s)&&(%s<0)",o.cuts,asymCat.GetName());
  } else {
    cuts1 = Form("(%s>0)",asymCat.GetName());
    cuts2 = Form("(%s<0)",asymCat.GetName());
  }

  if(0 == fillHistogram(hist1,RooArgList(*var),cuts1.Data(),o.cutRange) ||
     0 == fillHistogram(hist2,RooArgList(*var),cuts2.Data(),o.cutRange)) {
    coutE(Plotting) << ClassName() << "::" << GetName()
	 << ":plotAsymOn: createHistogram() failed" << endl;
    return 0;
  }

  // convert this histogram to a RooHist object on the heap
  RooHist *graph= new RooHist(*hist1,*hist2,0,1,o.etype,o.xErrorSize,kFALSE,o.scaleFactor);
  graph->setYAxisLabel(Form("Asymmetry in %s",asymCat.GetName())) ;

  // initialize the frame's normalization setup, if necessary
  frame->updateNormVars(_vars);

  // Rename graph if requested
  if (o.histName) {
    graph->SetName(o.histName) ;
  } else {
    TString hname(Form("h_%s_Asym[%s]",GetName(),asymCat.GetName())) ;
    if (o.cutRange && strlen(o.cutRange)>0) {
      hname.Append(Form("_CutRange[%s]",o.cutRange)) ;
    } 
    if (o.cuts && strlen(o.cuts)>0) {
      hname.Append(Form("_Cut[%s]",o.cuts)) ;
    } 
    graph->SetName(hname.Data()) ;
  }

  // add the RooHist to the specified plot
  frame->addPlotable(graph,o.drawOptions,o.histInvisible,o.refreshFrameNorm);

  // cleanup
  delete hist1;
  delete hist2;

  return frame;  
}



//_____________________________________________________________________________
RooPlot* RooAbsData::plotEffOn(RooPlot* frame, const RooAbsCategoryLValue& effCat, PlotOpt o) const 
{
  // Create and fill a histogram with the effiency N[1] / ( N[1] + N[0] ),
  // where N(1/0) is the number of data points with effCat=1 and effCat=0
  // as function of the frames variable. The efficiency category 'effCat' must
  // have exactly 2 +1 and 0.
  // 
  // The plot range and the number of plot bins is determined by the parameters
  // of the plot variable of the frame (RooAbsReal::setPlotRange(), RooAbsReal::setPlotBins())
  // 
  // The optional cut string expression can be used to select the events to be plotted.
  // The cut specification may refer to any variable contained in the data set
  //
  // The drawOptions are passed to the TH1::Draw() method

  if(0 == frame) {
    coutE(Plotting) << ClassName() << "::" << GetName() << ":plotEffOn: frame is null" << endl;
    return 0;
  }
  RooAbsRealLValue *var= (RooAbsRealLValue*) frame->getPlotVar();
  if(0 == var) {
    coutE(Plotting) << ClassName() << "::" << GetName()
	 << ":plotEffOn: frame does not specify a plot variable" << endl;
    return 0;
  }

  // create and fill temporary histograms of this variable for each state
  TString hist1Name(GetName()),hist2Name(GetName());
  hist1Name.Append("_plot1");
  TH1F *hist1, *hist2 ;
  hist2Name.Append("_plot2");

  if (o.bins) {
    hist1= var->createHistogram(hist1Name.Data(), "Events", *o.bins) ;
    hist2= var->createHistogram(hist2Name.Data(), "Events", *o.bins) ;
  } else {
    hist1= var->createHistogram(hist1Name.Data(), "Events", 
				frame->GetXaxis()->GetXmin(), frame->GetXaxis()->GetXmax(),
				frame->GetNbinsX());
    hist2= var->createHistogram(hist2Name.Data(), "Events", 
				frame->GetXaxis()->GetXmin(), frame->GetXaxis()->GetXmax(),
				frame->GetNbinsX());
  }

  assert(0 != hist1 && 0 != hist2);

  TString cuts1,cuts2 ;
  if (o.cuts && strlen(o.cuts)) {
    cuts1 = Form("(%s)&&(%s==1)",o.cuts,effCat.GetName());
    cuts2 = Form("(%s)&&(%s==0)",o.cuts,effCat.GetName());
  } else {
    cuts1 = Form("(%s==1)",effCat.GetName());
    cuts2 = Form("(%s==0)",effCat.GetName());
  }

  if(0 == fillHistogram(hist1,RooArgList(*var),cuts1.Data(),o.cutRange) ||
     0 == fillHistogram(hist2,RooArgList(*var),cuts2.Data(),o.cutRange)) {
    coutE(Plotting) << ClassName() << "::" << GetName()
	 << ":plotEffOn: createHistogram() failed" << endl;
    return 0;
  }

  // convert this histogram to a RooHist object on the heap
  RooHist *graph= new RooHist(*hist1,*hist2,0,1,o.etype,o.xErrorSize,kTRUE);
  graph->setYAxisLabel(Form("Efficiency of %s=%s",effCat.GetName(),effCat.lookupType(1)->GetName())) ;

  // initialize the frame's normalization setup, if necessary
  frame->updateNormVars(_vars);

  // Rename graph if requested
  if (o.histName) {
    graph->SetName(o.histName) ;
  } else {
    TString hname(Form("h_%s_Eff[%s]",GetName(),effCat.GetName())) ;
    if (o.cutRange && strlen(o.cutRange)>0) {
      hname.Append(Form("_CutRange[%s]",o.cutRange)) ;
    } 
    if (o.cuts && strlen(o.cuts)>0) {
      hname.Append(Form("_Cut[%s]",o.cuts)) ;
    } 
    graph->SetName(hname.Data()) ;
  }

  // add the RooHist to the specified plot
  frame->addPlotable(graph,o.drawOptions,o.histInvisible,o.refreshFrameNorm);

  // cleanup
  delete hist1;
  delete hist2;

  return frame;  
}


//_____________________________________________________________________________
Roo1DTable* RooAbsData::table(const RooAbsCategory& cat, const char* cuts, const char* /*opts*/) const
{
  // Create and fill a 1-dimensional table for given category column
  // This functions is the equivalent of plotOn() for category dimensions. 
  //
  // The optional cut string expression can be used to select the events to be tabulated
  // The cut specification may refer to any variable contained in the data set
  //
  // The option string is currently not used

  // First see if var is in data set 
  RooAbsCategory* tableVar = (RooAbsCategory*) _vars.find(cat.GetName()) ;
  RooArgSet *tableSet = 0;
  Bool_t ownPlotVar(kFALSE) ;
  if (!tableVar) {
    if (!cat.dependsOn(_vars)) {
      coutE(Plotting) << "RooTreeData::Table(" << GetName() << "): Argument " << cat.GetName() 
	   << " is not in dataset and is also not dependent on data set" << endl ;
      return 0 ; 
    }

    // Clone derived variable 
    tableSet = (RooArgSet*) RooArgSet(cat).snapshot(kTRUE) ;
    if (!tableSet) {
      coutE(Plotting) << "RooTreeData::table(" << GetName() << ") Couldn't deep-clone table category, abort." << endl ;
      return 0 ;
    }
    tableVar = (RooAbsCategory*) tableSet->find(cat.GetName()) ;
    ownPlotVar = kTRUE ;    

    //Redirect servers of derived clone to internal ArgSet representing the data in this set
    tableVar->recursiveRedirectServers(_vars) ;
  }

  TString tableName(GetName()) ;
  if (cuts && strlen(cuts)) {
    tableName.Append("(") ;
    tableName.Append(cuts) ;
    tableName.Append(")") ;    
  }
  Roo1DTable* table2 = tableVar->createTable(tableName) ;

  // Make cut selector if cut is specified
  RooFormulaVar* cutVar = 0;
  if (cuts && strlen(cuts)) {
    cutVar = new RooFormulaVar("cutVar",cuts,_vars) ;
  }

  // Dump contents   
  Int_t nevent= numEntries() ;
  for(Int_t i=0; i < nevent; ++i) {
    get(i);

    if (cutVar && cutVar->getVal()==0) continue ;
    
    table2->fill(*tableVar,weight()) ;
  }

  if (ownPlotVar) delete tableSet ;
  if (cutVar) delete cutVar ;

  return table2 ;
}


//_____________________________________________________________________________
Bool_t RooAbsData::getRange(RooRealVar& var, Double_t& lowest, Double_t& highest, Double_t marginFrac, Bool_t symMode) const 
{
  // Fill Doubles 'lowest' and 'highest' with the lowest and highest value of
  // observable 'var' in this dataset. If the return value is kTRUE and error
  // occurred

  // Lookup variable in dataset
  RooRealVar *varPtr= (RooRealVar*) _vars.find(var.GetName());
  if(0 == varPtr) {
    coutE(InputArguments) << "RooDataSet::getRange(" << GetName() << ") ERROR: unknown variable: " << var.GetName() << endl ;
    return kTRUE;
  }

  // Check if found variable is of type RooRealVar
  if (!dynamic_cast<RooRealVar*>(varPtr)) {
    coutE(InputArguments) << "RooDataSet::getRange(" << GetName() << ") ERROR: variable " << var.GetName() << " is not of type RooRealVar" << endl ;
    return kTRUE;
  }

  // Check if dataset is not empty
  if(sumEntries() == 0.) {
    coutE(InputArguments) << "RooDataSet::getRange(" << GetName() << ") WARNING: empty dataset" << endl ;
    return kTRUE;
  }

  // Look for highest and lowest value 
  lowest = RooNumber::infinity() ;
  highest = -RooNumber::infinity() ;
  for (Int_t i=0 ; i<numEntries() ; i++) {
    get(i) ;
    if (varPtr->getVal()<lowest) {
      lowest = varPtr->getVal() ;
    }
    if (varPtr->getVal()>highest) {
      highest = varPtr->getVal() ;
    }
  }  

  if (marginFrac>0) {
    if (symMode==kFALSE) {

      Double_t margin = marginFrac*(highest-lowest) ;    
      lowest -= margin ;
      highest += margin ; 
      if (lowest<var.getMin()) lowest = var.getMin() ;
      if (highest>var.getMax()) highest = var.getMax() ;

    } else {

      Double_t mom1 = moment(var,1) ;
      Double_t delta = ((highest-mom1)>(mom1-lowest)?(highest-mom1):(mom1-lowest))*(1+marginFrac) ;
      lowest = mom1-delta ;
      highest = mom1+delta ;
      if (lowest<var.getMin()) lowest = var.getMin() ;
      if (highest>var.getMax()) highest = var.getMax() ;

    }
  }
  
  return kFALSE ;
}




//_____________________________________________________________________________
void RooAbsData::optimizeReadingWithCaching(RooAbsArg& arg, const RooArgSet& cacheList, const RooArgSet& keepObsList)
{
  // Prepare dataset for use with cached constant terms listed in
  // 'cacheList' of expression 'arg'. Deactivate tree branches
  // for any dataset observable that is either not used at all,
  // or is used exclusively by cached branch nodes.

  RooArgSet pruneSet ;

  // Add unused observables in this dataset to pruneSet
  pruneSet.add(*get()) ;
  RooArgSet* usedObs = arg.getObservables(*this) ;
  pruneSet.remove(*usedObs,kTRUE,kTRUE) ;

  // Add observables exclusively used to calculate cached observables to pruneSet
  TIterator* vIter = get()->createIterator() ;
  RooAbsArg *var ;
  while ((var=(RooAbsArg*) vIter->Next())) {
    if (allClientsCached(var,cacheList)) {
      pruneSet.add(*var) ;
    }
  }
  delete vIter ;


  if (pruneSet.getSize()!=0) {

    // Go over all used observables and check if any of them have parameterized
    // ranges in terms of pruned observables. If so, remove those observable
    // from the pruning list
    TIterator* uIter = usedObs->createIterator() ;
    RooAbsArg* obs ;
    while((obs=(RooAbsArg*)uIter->Next())) {
      RooRealVar* rrv = dynamic_cast<RooRealVar*>(obs) ;
      if (rrv && !rrv->getBinning().isShareable()) {
	RooArgSet depObs ;
	RooAbsReal* loFunc = rrv->getBinning().lowBoundFunc() ;
	RooAbsReal* hiFunc = rrv->getBinning().highBoundFunc() ;
	if (loFunc) {
	  loFunc->leafNodeServerList(&depObs,0,kTRUE) ;
	}
	if (hiFunc) {
	  hiFunc->leafNodeServerList(&depObs,0,kTRUE) ;
	}
	if (depObs.getSize()>0) {
	  pruneSet.remove(depObs,kTRUE,kTRUE) ;
	}
      }
    }
    delete uIter ;
  }


  // Remove all observables in keep list from prune list
  pruneSet.remove(keepObsList,kTRUE,kTRUE) ;

  if (pruneSet.getSize()!=0) {
    
    // Deactivate tree branches here
    cxcoutI(Optimization) << "RooTreeData::optimizeReadingForTestStatistic(" << GetName() << "): Observables " << pruneSet
			    << " in dataset are either not used at all, orserving exclusively p.d.f nodes that are now cached, disabling reading of these observables for TTree" << endl ;
    setArgStatus(pruneSet,kFALSE) ;
  }

  delete usedObs ;
  
}


//_____________________________________________________________________________
Bool_t RooAbsData::allClientsCached(RooAbsArg* var, const RooArgSet& cacheList)
{
  // Utility function that determines if all clients of object 'var'
  // appear in given list of cached nodes.

  Bool_t ret(kTRUE), anyClient(kFALSE) ;

  TIterator* cIter = var->valueClientIterator() ;    
  RooAbsArg* client ;
  while ((client=(RooAbsArg*) cIter->Next())) {
    anyClient = kTRUE ;
    if (!cacheList.find(client->GetName())) {
      // If client is not cached recurse
      ret &= allClientsCached(client,cacheList) ;
    }
  }
  delete cIter ;
  
  return anyClient?ret:kFALSE ;
}




//_____________________________________________________________________________
void RooAbsData::checkInit() const
{ 
  _dstore->checkInit() ; 
}


//_____________________________________________________________________________
void RooAbsData::Draw(Option_t* option) 
{ 
  // Forward draw command to data store
  if (_dstore) _dstore->Draw(option) ; 
}



//_____________________________________________________________________________
Bool_t RooAbsData::hasFilledCache() const 
{ 
  return _dstore->hasFilledCache() ; 
}


//_____________________________________________________________________________
const TTree* RooAbsData::tree() const 
{ 
  return _dstore->tree() ; 
}
