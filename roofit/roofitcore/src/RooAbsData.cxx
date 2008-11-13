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

#include "RooAbsData.h"
#include "RooAbsData.h"
#include "RooFormulaVar.h"
#include "RooCmdConfig.h"
#include "RooAbsRealLValue.h"
#include "RooMsgService.h"
#include "RooMultiCategory.h"

#include "RooRealVar.h"
#include "RooGlobalFunc.h"



ClassImp(RooAbsData)
;


//_____________________________________________________________________________
RooAbsData::RooAbsData() 
{
  // Default constructor

  _iterator = _vars.createIterator() ;
  _cacheIter = _cachedVars.createIterator() ;
}



//_____________________________________________________________________________
RooAbsData::RooAbsData(const char *name, const char *title, const RooArgSet& vars) :
  TNamed(name,title), _vars("Dataset Variables"), _cachedVars("Cached Variables"), 
  _doDirtyProp(kTRUE) 
{
  // Constructor from a set of variables. Only fundamental elements of vars
  // (RooRealVar,RooCategory etc) are stored as part of the dataset

  // clone the fundamentals of the given data set into internal buffer
  TIterator* iter = vars.createIterator() ;
  RooAbsArg *var;
  while((0 != (var= (RooAbsArg*)iter->Next()))) {
    if (!var->isFundamental()) {
      coutE(InputArguments) << "RooAbsSet::initialize(" << GetName() 
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
  _cachedVars("Cached Variables"), _doDirtyProp(kTRUE)
{
  // Copy constructor

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
}



//_____________________________________________________________________________
RooAbsData::~RooAbsData() 
{
  // Destructor
  
  // delete owned contents.
  delete _iterator ;
  delete _cacheIter ;
}



//_____________________________________________________________________________
RooAbsData* RooAbsData::reduce(RooCmdArg arg1,RooCmdArg arg2,RooCmdArg arg3,RooCmdArg arg4,
			       RooCmdArg arg5,RooCmdArg arg6,RooCmdArg arg7,RooCmdArg arg8) 
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
  strcpy(buf,varNameList) ;
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
Int_t RooAbsData::defaultPrintContents(Option_t* /*opt*/) const 
{
  // Define default print options, for a given print style

  return kName|kClassName|kArgs|kValue ;
}
