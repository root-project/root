/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
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
#ifndef ROO_VECTOR_DATA_STORE
#define ROO_VECTOR_DATA_STORE

#include <list>
#include <vector>
#include <string>
#include "RooAbsDataStore.h" 
#include "TString.h"
#include "RooCatType.h"
#include "RooAbsCategory.h"
#include "RooAbsReal.h"

class RooAbsArg ;
class RooArgList ;
class TTree ;
class RooFormulaVar ;
class RooArgSet ;



class RooVectorDataStore : public RooAbsDataStore {
public:

  RooVectorDataStore() ; 

  // Empty ctor
  RooVectorDataStore(const char* name, const char* title, const RooArgSet& vars, const char* wgtVarName=0) ;
  virtual RooAbsDataStore* clone(const RooArgSet& vars, const char* newname=0) const { return new RooVectorDataStore(*this,vars,newname) ; }

  RooVectorDataStore(const RooVectorDataStore& other, const char* newname=0) ;
  RooVectorDataStore(const RooTreeDataStore& other, const RooArgSet& vars, const char* newname=0) ;
  RooVectorDataStore(const RooVectorDataStore& other, const RooArgSet& vars, const char* newname=0) ;


  RooVectorDataStore(const char *name, const char *title, RooAbsDataStore& tds, 
		     const RooArgSet& vars, const RooFormulaVar* cutVar, const char* cutRange,
		     Int_t nStart, Int_t nStop, Bool_t /*copyCache*/, const char* wgtVarName=0) ;

  virtual ~RooVectorDataStore() ;

  RooArgSet varsNoWeight(const RooArgSet& allVars, const char* wgtName) ;
  RooRealVar* weightVar(const RooArgSet& allVars, const char* wgtName) ;

  // Write current row
  virtual Int_t fill() ;

  // Retrieve a row
  using RooAbsDataStore::get ;
  virtual const RooArgSet* get(Int_t index) const ;
  virtual Double_t weight() const ;
  virtual Double_t weightError(RooAbsData::ErrorType etype=RooAbsData::Poisson) const ;
  virtual void weightError(Double_t& lo, Double_t& hi, RooAbsData::ErrorType etype=RooAbsData::Poisson) const ; 
  virtual Double_t weight(Int_t index) const ;
  virtual Bool_t isWeighted() const { return (_wgtVar!=0||_extWgtArray!=0) ; }

  // Change observable name
  virtual Bool_t changeObservableName(const char* from, const char* to) ;
  
  // Add one or more columns
  virtual RooAbsArg* addColumn(RooAbsArg& var, Bool_t adjustRange=kTRUE) ;
  virtual RooArgSet* addColumns(const RooArgList& varList) ;

  // Merge column-wise
  RooAbsDataStore* merge(const RooArgSet& allvars, std::list<RooAbsDataStore*> dstoreList) ;

  // Add rows 
  virtual void append(RooAbsDataStore& other) ;

  // General & bookkeeping methods
  virtual Bool_t valid() const ;
  virtual Int_t numEntries() const ;
  virtual Double_t sumEntries() const { return _sumWeight ; }
  virtual void reset() ;
  
  // Constant term  optimizer interface
  virtual const RooAbsArg* cacheOwner() { return _cacheOwner ; }
  virtual void cacheArgs(const RooAbsArg* owner, RooArgSet& varSet, const RooArgSet* nset=0) ;
  virtual void attachCache(const RooAbsArg* newOwner, const RooArgSet& cachedVars) ;
  virtual void resetCache() ;

  virtual void setArgStatus(const RooArgSet& set, Bool_t active) ;

  const RooVectorDataStore* cache() const { return _cache ; }

  void loadValues(const RooAbsDataStore *tds, const RooFormulaVar* select=0, const char* rangeName=0, Int_t nStart=0, Int_t nStop=2000000000) ;
  
  void dump()
  {
    cout << "_varsww = " << endl ; _varsww.Print("v") ;
    cout << "realVector list is" << endl ;
    std::vector<RealVector*>::iterator iter = _realStoreList.begin() ;
    for (; iter!=_realStoreList.end() ; ++iter) {
      cout << "RealVector " << *iter << " _real = " << (*iter)->_real << " = " << (*iter)->_real->GetName() << " bufptr = " << (*iter)->_buf  << endl ;
    }    
  }


  void setExternalWeightArray(Double_t* arrayWgt, Double_t* arrayWgtErrLo, Double_t* arrayWgtErrHi, Double_t* arraySumW2) { 
    _extWgtArray = arrayWgt ; 
    _extWgtErrLoArray = arrayWgtErrLo ;
    _extWgtErrHiArray = arrayWgtErrHi ;
    _extSumW2Array = arraySumW2 ;
  }

  virtual void setDirtyProp(Bool_t flag) { 
    _doDirtyProp = flag ; 
    if (_cache) {
      _cache->setDirtyProp(flag) ;
    }
}

  //virtual void checkInit() const;

  const RooArgSet& row() { return _varsww ; }

  class RealVector {
  public:
    RealVector(UInt_t initialCapacity=100) : 
      _real(0), _buf(0), _vec0(0) { 
      _vec.reserve(initialCapacity) ; 
    }

    RealVector(RooAbsReal* arg, UInt_t initialCapacity=100) : 
      _real(arg), _buf(0), _vec0(0) { 
      _vec.reserve(initialCapacity) ; 
    }

    ~RealVector() {
    }

    RealVector(const RealVector& other, RooAbsReal* real=0) : 
      _real(real?real:other._real), _buf(other._buf), _vec(other._vec)  {
      _vec0 = _vec.size()>0 ? &_vec.front() : 0 ;
    }

    RealVector& operator=(const RealVector& other) {
      if (&other==this) return *this ;
      _real = other._real ;
      _buf = other._buf ;
      _vec = other._vec ;
      _vec0 = _vec.size()>0 ? &_vec.front() : 0 ;
      return *this ;
    }

    void setBufArg(RooAbsReal* arg) { _real = arg ; }
    const RooAbsReal* bufArg() const { return _real ; }

    void setBuffer(Double_t* newBuf) { 
      _buf = newBuf ; 
    }

    void fill() { 
      _vec.push_back(*_buf) ; 
      _vec0 = &_vec.front() ;
    } ;

    void write(Int_t i) {
      _vec[i] = *_buf ;
    }
    
    void reset() { 
      _vec.clear() ; 
      _vec0 = &_vec.front() ;
    }

    inline void get(Int_t idx) const { 
      *_buf = *(_vec0+idx) ; 
    }
    Int_t size() const { return _vec.size() ; }

    void resize(Int_t siz) {
      _vec.resize(siz) ;
      _vec0 = &_vec.front() ;
    }

  private:
    friend class RooVectorDataStore ;
    RooAbsReal* _real ;
    Double_t* _buf ; 
    std::vector<Double_t> _vec ;
    Double_t* _vec0 ;
  } ;
  

  class RealFullVector {
  public:
    RealFullVector(UInt_t initialCapacity=100) : 
      _real(0), _buf(0), _bufE(0), _bufEL(0), 
      _bufEH(0), _vecE(0), _vecEL(0), _vecEH(0), _vec0(0) { 
      _vec.reserve(initialCapacity) ; 
    }

    RealFullVector(RooAbsReal* arg, UInt_t initialCapacity=100) : 
      _real(arg), _buf(0), _bufE(0), _bufEL(0), 
      _bufEH(0), _vecE(0), _vecEL(0), _vecEH(0), _vec0(0) { 
      _vec.reserve(initialCapacity) ; 
    }

    ~RealFullVector() {
      if (_vecE) delete _vecE ;
      if (_vecEL) delete _vecEL ;
      if (_vecEH) delete _vecEH ;
    }

    RealFullVector(const RealFullVector& other, RooAbsReal* real=0) : 
      _real(real?real:other._real), _buf(other._buf), _vec(other._vec), 
      _bufE(other._bufE), _bufEL(other._bufEL), _bufEH(other._bufEH) {
      _vecE = (other._vecE) ? new vector<Double_t>(*other._vecE) : 0 ;
      _vecEL = (other._vecEL) ? new vector<Double_t>(*other._vecEL) : 0 ;
      _vecEH = (other._vecEH) ? new vector<Double_t>(*other._vecEH) : 0 ;
      _vec0 = _vec.size()>0 ? &_vec.front() : 0 ;
    }

    RealFullVector(const RealVector& other, RooAbsReal* real=0) : 
      _real(real?real:other._real), _buf(other._buf), _vec(other._vec), 
      _bufE(0), _bufEL(0), _bufEH(0) {
      _vecE = 0 ;
      _vecEL = 0 ; 
      _vecEH = 0 ; 
      _vec0 = _vec.size()>0 ? &_vec.front() : 0 ;
    }

    RealFullVector& operator=(const RealFullVector& other) {
      if (&other==this) return *this ;
      _real = other._real ;
      _buf = other._buf ;
      _vec = other._vec ;
      _bufE = other._bufE ;
      _bufEL = other._bufEL ;
      _bufEH = other._bufEH ;
      _vecE = other._vecE ? new vector<Double_t>(*other._vecE) : 0 ;
      _vecEL = other._vecEL ? new vector<Double_t>(*other._vecEL) : 0 ;
      _vecEH = other._vecEH ? new vector<Double_t>(*other._vecEH) : 0 ;
      _vec0 = _vec.size()>0 ? &_vec.front() : 0 ;
      return *this ;
    }

    void setBufArg(RooAbsReal* arg) { _real = arg ; }
    const RooAbsReal* bufArg() const { return _real ; }

    void setBuffer(Double_t* newBuf) { 
      _buf = newBuf ; 
    }
    void setErrorBuffer(Double_t* newBuf) { 
      _bufE = newBuf ; 
      if (!_vecE) _vecE = new vector<Double_t> ;
      _vecE->reserve(_vec.capacity()) ;
    }
    void setAsymErrorBuffer(Double_t* newBufL, Double_t* newBufH) { 
      _bufEL = newBufL ; _bufEH = newBufH ; 
      if (!_vecEL) {
	_vecEL = new vector<Double_t> ;
	_vecEH = new vector<Double_t> ;
	_vecEL->reserve(_vec.capacity()) ;
	_vecEH->reserve(_vec.capacity()) ;
      }
    }
    
    void fill() { 
      _vec.push_back(*_buf) ; 
      _vec0 = &_vec.front() ;
      if (_vecE) _vecE->push_back(*_bufE) ;
      if (_vecEL) _vecEL->push_back(*_bufEL) ;
      if (_vecEH) _vecEH->push_back(*_bufEH) ;
    } ;

    void write(Int_t i) {
      _vec[i] = *_buf ;
      if (_vecE) (*_vecE)[i] = *_bufE ;
      if (_vecEL) (*_vecEL)[i] = *_bufEL ;
      if (_vecEH) (*_vecEH)[i] = *_bufEH ;
    }
    
    void reset() { 
      _vec.clear() ; 
      _vec0 = &_vec.front() ;
      if (_vecE) _vecE->clear() ;
      if (_vecEL) _vecEL->clear() ;
      if (_vecEH) _vecEH->clear() ;
    }
    inline void get(Int_t idx) const { 
      *_buf = *(_vec0+idx) ; 
      if (_vecE) *_bufE = (*_vecE)[idx];
      if (_vecEL) *_bufEL = (*_vecEL)[idx] ;
      if (_vecEH) *_bufEH = (*_vecEH)[idx] ;
    }
    Int_t size() const { return _vec.size() ; }

    void resize(Int_t siz) {
      _vec.resize(siz) ;
      _vec0 = &_vec.front() ;
      if (_vecE) _vecE->resize(siz) ;
      if (_vecEL) _vecEL->resize(siz) ;
      if (_vecEH) _vecEH->resize(siz) ;
    }

  private:
    friend class RooVectorDataStore ;
    RooAbsReal* _real ;
    Double_t* _buf ; 
    std::vector<Double_t> _vec ;
    Double_t *_bufE, *_bufEL, *_bufEH ;
    std::vector<Double_t> *_vecE, *_vecEL, *_vecEH ;
    Double_t* _vec0 ;
  } ;
  

  class CatVector {
  public:
    CatVector(UInt_t initialCapacity=100) : 
      _cat(0), _buf(0), _vec0(0)
    {
      _vec.reserve(initialCapacity) ;
    }

    CatVector(RooAbsCategory* cat, UInt_t initialCapacity=100) : 
      _cat(cat), _buf(0), _vec0(0)
    {
      _vec.reserve(initialCapacity) ;
    }

    ~CatVector() {
    }

    CatVector(const CatVector& other, RooAbsCategory* cat=0) : 
      _cat(cat?cat:other._cat), _buf(other._buf), _vec(other._vec) 
      {
	_vec0 = _vec.size()>0 ? &_vec.front() : 0 ;
      }

    CatVector& operator=(const CatVector& other) {
      if (&other==this) return *this ;
      _cat = other._cat ;
      _buf = other._buf ;
      _vec = other._vec ;
      _vec0 = _vec.size()>0 ? &_vec.front() : 0 ;
      return *this ;
    }

    void setBuffer(RooCatType* newBuf) { 
      _buf = newBuf ; 
    }
    void fill() { 
      _vec.push_back(*_buf) ; 
      _vec0 = &_vec.front() ;
    } ;
    void write(Int_t i) { 
      _vec[i]=*_buf ; 
    } ;
    void reset() { 
      _vec.clear() ; 
      _vec0 = &_vec.front() ;
    }
    inline void get(Int_t idx) const { 
      *_buf = *(_vec0+idx) ; 
    }
    Int_t size() const { return _vec.size() ; }

    void resize(Int_t siz) {
      _vec.resize(siz) ;
      _vec0 = &_vec.front() ;
    }

    void setBufArg(RooAbsCategory* arg) { _cat = arg ; }
    const RooAbsCategory* bufArg() const { return _cat ; }


  private:
    friend class RooVectorDataStore ;
    RooAbsCategory* _cat ;
    RooCatType* _buf ; 
    std::vector<RooCatType> _vec ;
    RooCatType* _vec0 ;
  } ;
  

 protected:

  friend class RooAbsReal ;
  friend class RooAbsCategory ;
  friend class RooRealVar ;
  std::vector<RealVector*>& realStoreList() { return _realStoreList ; }
  std::vector<RealFullVector*>& realfStoreList() { return _realfStoreList ; }
  std::vector<CatVector*>& catStoreList() { return _catStoreList ; }

  CatVector* addCategory(RooAbsCategory* cat) {

    CatVector* cv(0) ;

    // First try a match by name
    std::vector<CatVector*>::iterator iter = _catStoreList.begin() ;
    for (; iter!=_catStoreList.end() ; ++iter) {
      if (std::string((*iter)->bufArg()->GetName())==cat->GetName()) {
	cv = (*iter)  ;
	return cv ;
      }
    }

    // If nothing found this will make an entry
    _catStoreList.push_back(new CatVector(cat)) ;
    _nCat++ ;

    // Update cached ptr to first element as push_back may have reallocated
    _firstCat = &_catStoreList.front() ;

    return _catStoreList.back() ;
  }

  RealVector* addReal(RooAbsReal* real) {
    
    RealVector* rv(0) ;
    
    // First try a match by name
    std::vector<RealVector*>::iterator iter = _realStoreList.begin() ;
    for (; iter!=_realStoreList.end() ; ++iter) {
      if (std::string((*iter)->bufArg()->GetName())==real->GetName()) {
	rv = (*iter) ;
	return rv ;
      }
    }    

    // Then check if an entry already exists for a full real    
    std::vector<RealFullVector*>::iterator iter2 = _realfStoreList.begin() ;
    for (; iter2!=_realfStoreList.end() ; ++iter2) {
      if (std::string((*iter2)->bufArg()->GetName())==real->GetName()) {
	cout << "NEED TO REFER TO FULL SLOT HERE" << endl ;
      }
    }    

    // If nothing found this will make an entry
    _realStoreList.push_back(new RealVector(real)) ;
    _nReal++ ;

    // Update cached ptr to first element as push_back may have reallocated
    _firstReal = &_realStoreList.front() ;


    return _realStoreList.back() ;
  }

  RealFullVector* addRealFull(RooAbsReal* real) {
    
    RealFullVector* rv(0) ;
    
    // First try a match by name
    std::vector<RealFullVector*>::iterator iter = _realfStoreList.begin() ;
    for (; iter!=_realfStoreList.end() ; ++iter) {
      if (std::string((*iter)->bufArg()->GetName())==real->GetName()) {
	rv = (*iter) ;
	return rv ;
      }
    }    

    // Then check if an entry already exists for a bare real    
    std::vector<RealVector*>::iterator iter2 = _realStoreList.begin() ;
    for (; iter2!=_realStoreList.end() ; ++iter2) {
      if (std::string((*iter2)->bufArg()->GetName())==real->GetName()) {

	// Convert element to full and add to full list
	_realfStoreList.push_back(new RealFullVector(*(*iter2),real)) ;
	_nRealF++ ;
	_firstRealF = &_realfStoreList.front() ;
	
	// Delete bare element
	RealVector* tmp = *iter2 ;
	_realStoreList.erase(iter2) ;
	delete tmp ;
	_firstReal = &_realStoreList.front() ;
	_nReal-- ;

	return _realfStoreList.back() ;
      }
    }    

    // If nothing found this will make an entry
    _realfStoreList.push_back(new RealFullVector(real)) ;
    _nRealF++ ;

    // Update cached ptr to first element as push_back may have reallocated
    _firstRealF = &_realfStoreList.front() ;


    return _realfStoreList.back() ;
  }

 private:
  RooArgSet _varsww ;
  RooRealVar* _wgtVar ;     // Pointer to weight variable (if set)

  std::vector<RealVector*> _realStoreList ;
  std::vector<RealFullVector*> _realfStoreList ;
  std::vector<CatVector*> _catStoreList ;

  Int_t _nReal ;
  Int_t _nRealF ;
  Int_t _nCat ;
  Int_t _nEntries ;
  RealVector** _firstReal ;
  RealFullVector** _firstRealF ;
  CatVector** _firstCat ;
  Double_t _sumWeight ; 

  Double_t* _extWgtArray ;         //! External weight array
  Double_t* _extWgtErrLoArray ;    //! External weight array - low error
  Double_t* _extWgtErrHiArray ;    //! External weight array - high error
  Double_t* _extSumW2Array ;       //! External sum of weights array

  mutable Double_t  _curWgt ;      // Weight of current event
  mutable Double_t  _curWgtErrLo ; // Weight of current event
  mutable Double_t  _curWgtErrHi ; // Weight of current event
  mutable Double_t  _curWgtErr ;   // Weight of current event

  RooVectorDataStore* _cache ; //! Optimization cache
  RooAbsArg* _cacheOwner ; //! Cache owner

  ClassDef(RooVectorDataStore,1) // STL-vector-based Data Storage class
};


#endif
