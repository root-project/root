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

#include "RooAbsDataStore.h"
#include "RooAbsCategory.h"
#include "RooAbsReal.h"
#include "RooChangeTracker.h"
#include "RooRealVar.h"

#include "ROOT/RStringView.hxx"
#include "Rtypes.h"

#include <list>
#include <vector>
#include <algorithm>

class RooAbsArg ;
class RooArgList ;
class TTree ;
class RooFormulaVar ;
class RooArgSet ;
class RooTreeDataStore ;

#define VECTOR_BUFFER_SIZE 1024

class RooVectorDataStore : public RooAbsDataStore {
public:

  RooVectorDataStore() ;

  // Empty ctor
  RooVectorDataStore(RooStringView name, RooStringView title, const RooArgSet& vars, const char* wgtVarName=nullptr) ;

  RooAbsDataStore* clone(const char* newname=nullptr) const override { return new RooVectorDataStore(*this,newname) ; }
  RooAbsDataStore* clone(const RooArgSet& vars, const char* newname=nullptr) const override { return new RooVectorDataStore(*this,vars,newname) ; }

  std::unique_ptr<RooAbsDataStore> reduce(RooStringView name, RooStringView title,
                          const RooArgSet& vars, const RooFormulaVar* cutVar, const char* cutRange,
                          std::size_t nStart, std::size_t nStop) override;

  RooVectorDataStore(const RooVectorDataStore& other, const char* newname=nullptr) ;
  RooVectorDataStore(const RooTreeDataStore& other, const RooArgSet& vars, const char* newname=nullptr) ;
  RooVectorDataStore(const RooVectorDataStore& other, const RooArgSet& vars, const char* newname=nullptr) ;


  RooVectorDataStore(RooStringView name, RooStringView title, RooAbsDataStore& tds,
                     const RooArgSet& vars, const RooFormulaVar* cutVar, const char* cutRange,
                     std::size_t nStart, std::size_t nStop, const char* wgtVarName=nullptr) ;

  ~RooVectorDataStore() override ;

  /// \class ArraysStruct
  /// Output struct for the RooVectorDataStore::getArrays() helper function.
  /// Meant to be used for RooFit internal use and might change without warning.
  struct ArraysStruct {

    template<class T>
    struct ArrayInfo {
        ArrayInfo(RooStringView n, T const* d) : name{n}, data{d} {}
        std::string name;
        T const* data;
    };

    std::vector<ArrayInfo<double>> reals;
    std::vector<ArrayInfo<RooAbsCategory::value_type>> cats;

    std::size_t size;
  };

  /// \name Internal RooFit interface.
  /// The classes and functions in the internal RooFit interface are
  /// implementation details and not part of the public user interface.
  /// Everything in this group might change without warning.
  /// @{
  ArraysStruct getArrays() const;
  void recomputeSumWeight();
  /// @}

private:
  RooArgSet varsNoWeight(const RooArgSet& allVars, const char* wgtName);
  RooRealVar* weightVar(const RooArgSet& allVars, const char* wgtName);

  // reserve storage for nEvt entries
  void reserve(Int_t nEvt);

public:
  // Write current row
  Int_t fill() override;

  // Retrieve a row
  using RooAbsDataStore::get;
  const RooArgSet* get(Int_t index) const override;

  using RooAbsDataStore::weight ;
  /// Return the weight of the last-retrieved data point.
  double weight() const override
  {
    if (_extWgtArray)
      return _extWgtArray[_currentWeightIndex];
    if (_wgtVar)
      return _wgtVar->getVal();

    return 1.0;
  }
  double weightError(RooAbsData::ErrorType etype=RooAbsData::Poisson) const override;
  void weightError(double& lo, double& hi, RooAbsData::ErrorType etype=RooAbsData::Poisson) const override;
  bool isWeighted() const override { return _wgtVar || _extWgtArray; }

  RooAbsData::RealSpans getBatches(std::size_t first, std::size_t len) const override;
  RooAbsData::CategorySpans getCategoryBatches(std::size_t /*first*/, std::size_t len) const override;
  RooSpan<const double> getWeightBatch(std::size_t first, std::size_t len) const override;

  // Change observable name
  bool changeObservableName(const char* from, const char* to) override;

  // Add one column
  RooAbsArg* addColumn(RooAbsArg& var, bool adjustRange=true) override;

  // Merge column-wise
  RooAbsDataStore* merge(const RooArgSet& allvars, std::list<RooAbsDataStore*> dstoreList) override;

  // Add rows
  void append(RooAbsDataStore& other) override;

  // General & bookkeeping methods
  Int_t numEntries() const override { return static_cast<int>(size()); }
  double sumEntries() const override { return _sumWeight ; }
  /// Get size of stored dataset.
  std::size_t size() const {
    if (!_realStoreList.empty()) {
      return _realStoreList.front()->size();
    } else if (!_realfStoreList.empty()) {
      return _realfStoreList.front()->size();
    } else if (!_catStoreList.empty()) {
      return _catStoreList.front()->size();
    }

    return 0;
  }
  void reset() override;

  // Buffer redirection routines used in inside RooAbsOptTestStatistics
  void attachBuffers(const RooArgSet& extObs) override;
  void resetBuffers() override;


  // Constant term  optimizer interface
  const RooAbsArg* cacheOwner() override { return _cacheOwner ; }
  void cacheArgs(const RooAbsArg* owner, RooArgSet& varSet, const RooArgSet* nset=nullptr, bool skipZeroWeights=true) override;
  void attachCache(const RooAbsArg* newOwner, const RooArgSet& cachedVars) override;
  void resetCache() override;
  void recalculateCache(const RooArgSet* /*proj*/, Int_t firstEvent, Int_t lastEvent, Int_t stepSize, bool skipZeroWeights) override;

  void setArgStatus(const RooArgSet& set, bool active) override;

  const RooVectorDataStore* cache() const { return _cache ; }

  void loadValues(const RooAbsDataStore *tds, const RooFormulaVar* select=nullptr, const char* rangeName=nullptr, std::size_t nStart=0, std::size_t nStop = std::numeric_limits<std::size_t>::max()) override;

  void dump() override;

  void setExternalWeightArray(const double* arrayWgt, const double* arrayWgtErrLo,
      const double* arrayWgtErrHi, const double* arraySumW2) override {
    _extWgtArray = arrayWgt ;
    _extWgtErrLoArray = arrayWgtErrLo ;
    _extWgtErrHiArray = arrayWgtErrHi ;
    _extSumW2Array = arraySumW2 ;
  }

  void setDirtyProp(bool flag) override {
    _doDirtyProp = flag ;
    if (_cache) {
      _cache->setDirtyProp(flag) ;
    }
  }

  const RooArgSet& row() { return _varsww ; }

  class RealVector {
  public:

    RealVector(UInt_t initialCapacity=(VECTOR_BUFFER_SIZE / sizeof(double))) :
      _nativeReal(nullptr), _real(nullptr), _buf(nullptr), _nativeBuf(nullptr), _tracker(nullptr), _nset(nullptr) {
      _vec.reserve(initialCapacity);
    }

    RealVector(RooAbsReal* arg, UInt_t initialCapacity=(VECTOR_BUFFER_SIZE / sizeof(double))) :
      _nativeReal(arg), _real(nullptr), _buf(nullptr), _nativeBuf(nullptr), _tracker(nullptr), _nset(nullptr) {
      _vec.reserve(initialCapacity);
    }

    virtual ~RealVector() {
      delete _tracker;
      if (_nset) delete _nset ;
    }

    RealVector(const RealVector& other, RooAbsReal* real=nullptr) :
      _vec(other._vec), _nativeReal(real?real:other._nativeReal), _real(real?real:other._real), _buf(other._buf), _nativeBuf(other._nativeBuf), _nset(nullptr) {
      if (other._tracker) {
        _tracker = new RooChangeTracker(Form("track_%s",_nativeReal->GetName()),"tracker",other._tracker->parameters()) ;
      } else {
        _tracker = nullptr ;
      }
      if (other._nset) {
        _nset = new RooArgSet(*other._nset) ;
      }
    }

    RealVector& operator=(const RealVector& other) {
      if (&other==this) return *this;
      _nativeReal = other._nativeReal;
      _real = other._real;
      _buf = other._buf;
      _nativeBuf = other._nativeBuf;
      if (other._vec.size() <= _vec.capacity() / 2 && _vec.capacity() > (VECTOR_BUFFER_SIZE / sizeof(double))) {
        std::vector<double> tmp;
        tmp.reserve(std::max(other._vec.size(), VECTOR_BUFFER_SIZE / sizeof(double)));
        tmp.assign(other._vec.begin(), other._vec.end());
        _vec.swap(tmp);
      } else {
        _vec = other._vec;
      }

      return *this;
    }

    void setNset(RooArgSet* newNset) { _nset = newNset ? new RooArgSet(*newNset) : nullptr ; }

    RooArgSet* nset() const { return _nset ; }

    void setBufArg(RooAbsReal* arg) { _nativeReal = arg ; }
    const RooAbsReal* bufArg() const { return _nativeReal ; }

    void setBuffer(RooAbsReal* real, double* newBuf) {
      _real = real ;
      _buf = newBuf ;
      if (_nativeBuf==nullptr) {
        _nativeBuf=newBuf ;
      }
    }

    void setNativeBuffer(double* newBuf=nullptr) {
      _nativeBuf = newBuf ? newBuf : _buf ;
    }

    void setDependents(const RooArgSet& deps) {
      if (_tracker) {
        delete _tracker ;
      }
      _tracker = new RooChangeTracker(Form("track_%s",_nativeReal->GetName()),"tracker",deps) ;
    }

    bool needRecalc() {
      if (!_tracker) return false ;
      return _tracker->hasChanged(true) ;
    }

    void fill() {
      _vec.push_back(*_buf);
    }

    void write(Int_t i) {
      assert(static_cast<std::size_t>(i) < _vec.size());
      _vec[i] = *_buf ;
    }

    void reset() {
      _vec.clear();
    }

    inline void load(std::size_t idx) const {
      assert(idx < _vec.size());
      *_buf = *(_vec.begin() + idx) ;
      *_nativeBuf = *_buf ;
    }

    RooSpan<const double> getRange(std::size_t first, std::size_t last) const {
      auto beg = std::min(_vec.cbegin() + first, _vec.cend());
      auto end = std::min(_vec.cbegin() + last,  _vec.cend());

      return RooSpan<const double>(beg, end);
    }

    std::size_t size() const { return _vec.size() ; }

    void resize(Int_t siz) {
      if (siz < Int_t(_vec.capacity()) / 2 && _vec.capacity() > (VECTOR_BUFFER_SIZE / sizeof(double))) {
        // do an expensive copy, if we save at least a factor 2 in size
        std::vector<double> tmp;
        tmp.reserve(std::max(siz, Int_t(VECTOR_BUFFER_SIZE / sizeof(double))));
        if (!_vec.empty())
          tmp.assign(_vec.begin(), std::min(_vec.end(), _vec.begin() + siz));
        if (Int_t(tmp.size()) != siz)
          tmp.resize(siz);
        _vec.swap(tmp);
      } else {
        _vec.resize(siz);
      }
    }

    void reserve(Int_t siz) {
      _vec.reserve(siz);
    }

    const std::vector<double>& data() const {
      return _vec;
    }

    std::vector<double>& data() { return _vec; }

  protected:
    std::vector<double> _vec;

  private:
    friend class RooVectorDataStore ;
    RooAbsReal* _nativeReal ; ///< Instance which our data belongs to. This is the variable in the dataset.
    RooAbsReal* _real ; ///< Instance where we should write data into when load() is called.
    double* _buf ; ///<!
    double* _nativeBuf ; ///<!
    RooChangeTracker* _tracker ;
    RooArgSet* _nset ; ///<!
    ClassDef(RealVector,1) // STL-vector-based Data Storage class
  } ;


  class RealFullVector : public RealVector {
  public:
    RealFullVector(UInt_t initialCapacity=(VECTOR_BUFFER_SIZE / sizeof(double))) : RealVector(initialCapacity) {}

    RealFullVector(RooAbsReal* arg, UInt_t initialCapacity=(VECTOR_BUFFER_SIZE / sizeof(double))) :
      RealVector(arg,initialCapacity) {}

    RealFullVector(const RealFullVector& other, RooAbsReal* real=nullptr) : RealVector(other,real),
      _bufE(other._bufE), _bufEL(other._bufEL), _bufEH(other._bufEH),
      _vecE{other._vecE}, _vecEL{other._vecEL}, _vecEH{other._vecEH}
    {
    }

    RealFullVector(const RealVector& other, RooAbsReal* real=nullptr) : RealVector(other,real) {}

    RealFullVector& operator=(RealFullVector const& other) = delete;

    void setErrorBuffer(double* newBuf) {
      _bufE = newBuf ;
      _vecE.reserve(_vec.capacity()) ;
    }
    void setAsymErrorBuffer(double* newBufL, double* newBufH) {
      _bufEL = newBufL;
      _bufEH = newBufH;
      _vecEL.reserve(_vec.capacity()) ;
      _vecEH.reserve(_vec.capacity()) ;
    }

    void fill() {
      RealVector::fill() ;
      if (_bufE) _vecE.push_back(*_bufE) ;
      if (_bufEL) _vecEL.push_back(*_bufEL) ;
      if (_bufEH) _vecEH.push_back(*_bufEH) ;
    } ;

    void write(Int_t i) {
      RealVector::write(i) ;
      if (_bufE) _vecE[i] = *_bufE ;
      if (_bufEL) _vecEL[i] = *_bufEL ;
      if (_bufEH) _vecEH[i] = *_bufEH ;
    }

    void reset() {
      RealVector::reset();
      _vecE.clear();
      _vecEL.clear();
      _vecEH.clear();
    }

    inline void load(Int_t idx) const {
      RealVector::load(idx) ;
      if (_bufE) *_bufE = _vecE[idx];
      if (_bufEL) *_bufEL = _vecEL[idx];
      if (_bufEH) *_bufEH = _vecEH[idx];
    }

    void resize(Int_t siz) {
      RealVector::resize(siz);
      if(_bufE) _vecE.resize(siz);
      if(_bufEL) _vecEL.resize(siz);
      if(_bufEH) _vecEH.resize(siz);
    }

    void reserve(Int_t siz) {
      RealVector::reserve(siz);
      if(_bufE) _vecE.reserve(siz);
      if(_bufEL) _vecEL.reserve(siz);
      if(_bufEH) _vecEH.reserve(siz);
    }

    double* bufE() const { return _bufE; }
    double* bufEL() const { return _bufEL; }
    double* bufEH() const { return _bufEH; }

    std::vector<double> const& dataE() const { return _vecE; }
    std::vector<double> const& dataEL() const { return _vecEL; }
    std::vector<double> const& dataEH() const { return _vecEH; }

  private:

    double *_bufE = nullptr; ///<!
    double *_bufEL = nullptr; ///<!
    double *_bufEH = nullptr; ///<!
    std::vector<double> _vecE;
    std::vector<double> _vecEL;
    std::vector<double> _vecEH;
    ClassDefOverride(RealFullVector,2); // STL-vector-based Data Storage class
  };


  class CatVector {
  public:
    CatVector(UInt_t initialCapacity = VECTOR_BUFFER_SIZE) :
      _cat(nullptr), _buf(nullptr), _nativeBuf(nullptr)
    {
      _vec.reserve(initialCapacity);
    }

    CatVector(RooAbsCategory* cat, UInt_t initialCapacity = VECTOR_BUFFER_SIZE) :
      _cat(cat), _buf(nullptr), _nativeBuf(nullptr)
    {
      _vec.reserve(initialCapacity);
    }

    virtual ~CatVector() {
    }

    CatVector(const CatVector& other, RooAbsCategory* cat = nullptr) :
      _cat(cat?cat:other._cat), _buf(other._buf), _nativeBuf(other._nativeBuf), _vec(other._vec)
    {

    }

    CatVector& operator=(const CatVector& other) {
      if (&other==this) return *this;
      _cat = other._cat;
      _buf = other._buf;
      _nativeBuf = other._nativeBuf;
      if (other._vec.size() <= _vec.capacity() / 2 && _vec.capacity() > VECTOR_BUFFER_SIZE) {
        std::vector<RooAbsCategory::value_type> tmp;
        tmp.reserve(std::max(other._vec.size(), std::size_t(VECTOR_BUFFER_SIZE)));
        tmp.assign(other._vec.begin(), other._vec.end());
        _vec.swap(tmp);
      } else {
        _vec = other._vec;
      }

      return *this;
    }

    void setBuffer(RooAbsCategory::value_type* newBuf) {
      _buf = newBuf ;
      if (!_nativeBuf) _nativeBuf = newBuf;
    }

    void setNativeBuffer(RooAbsCategory::value_type* newBuf = nullptr) {
      _nativeBuf = newBuf ? newBuf : _buf;
    }

    void fill() {
      _vec.push_back(*_buf) ;
    }

    void write(std::size_t i) {
      _vec[i] = *_buf;
    }

    void reset() {
      // make sure the vector releases the underlying memory
      std::vector<RooAbsCategory::value_type> tmp;
      _vec.swap(tmp);
    }

    inline void load(std::size_t idx) const {
      *_buf = _vec[idx];
      *_nativeBuf = *_buf;
    }

    RooSpan<const RooAbsCategory::value_type> getRange(std::size_t first, std::size_t last) const {
      auto beg = std::min(_vec.cbegin() + first, _vec.cend());
      auto end = std::min(_vec.cbegin() + last,  _vec.cend());

      return RooSpan<const RooAbsCategory::value_type>(beg, end);
    }


    std::size_t size() const { return _vec.size() ; }

    void resize(Int_t siz) {
      if (siz < Int_t(_vec.capacity()) / 2 && _vec.capacity() > VECTOR_BUFFER_SIZE) {
        // do an expensive copy, if we save at least a factor 2 in size
        std::vector<RooAbsCategory::value_type> tmp;
        tmp.reserve(std::max(siz, VECTOR_BUFFER_SIZE));
        if (!_vec.empty())
          tmp.assign(_vec.begin(), std::min(_vec.end(), _vec.begin() + siz));
        if (Int_t(tmp.size()) != siz)
          tmp.resize(siz);
        _vec.swap(tmp);
      } else {
        _vec.resize(siz);
      }
    }

    void reserve(Int_t siz) {
      _vec.reserve(siz);
    }

    void setBufArg(RooAbsCategory* arg) { _cat = arg; }
    const RooAbsCategory* bufArg() const { return _cat; }

    std::vector<RooAbsCategory::value_type>& data() { return _vec; }

  private:
    friend class RooVectorDataStore ;
    RooAbsCategory* _cat;
    RooAbsCategory::value_type* _buf;  ///<!
    RooAbsCategory::value_type* _nativeBuf;  ///<!
    std::vector<RooAbsCategory::value_type> _vec;
    ClassDef(CatVector,2) // STL-vector-based Data Storage class
  } ;

  std::vector<RealVector*>& realStoreList() { return _realStoreList ; }
  std::vector<RealFullVector*>& realfStoreList() { return _realfStoreList ; }
  std::vector<CatVector*>& catStoreList() { return _catStoreList ; }

 protected:

  friend class RooAbsReal ;
  friend class RooAbsCategory ;
  friend class RooRealVar ;

  CatVector* addCategory(RooAbsCategory* cat);

  RealVector* addReal(RooAbsReal* real);

  bool isFullReal(RooAbsReal* real);

  bool hasError(RooAbsReal* real);

  bool hasAsymError(RooAbsReal* real);

  RealFullVector* addRealFull(RooAbsReal* real);

  bool hasFilledCache() const override { return _cache ? true : false ; }

  void forceCacheUpdate() override;

 private:
  RooArgSet _varsww ;
  RooRealVar* _wgtVar = nullptr; ///< Pointer to weight variable (if set)

  std::vector<RealVector*> _realStoreList ;
  std::vector<RealFullVector*> _realfStoreList ;
  std::vector<CatVector*> _catStoreList ;

  void setAllBuffersNative() ;

  double _sumWeight = 0.0;
  double _sumWeightCarry = 0.0;

  const double* _extWgtArray = nullptr;      ///<! External weight array
  const double* _extWgtErrLoArray = nullptr; ///<! External weight array - low error
  const double* _extWgtErrHiArray = nullptr; ///<! External weight array - high error
  const double* _extSumW2Array = nullptr;    ///<! External sum of weights array

  mutable ULong64_t _currentWeightIndex{0}; ///<

  RooVectorDataStore* _cache = nullptr; ///<! Optimization cache
  RooAbsArg* _cacheOwner = nullptr; ///<! Cache owner

  bool _forcedUpdate = false; ///<! Request for forced cache update

  ClassDefOverride(RooVectorDataStore, 7) // STL-vector-based Data Storage class
};


#endif
