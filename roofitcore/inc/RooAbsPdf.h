#ifdef NO

class RooAbsPdf : public RooNormFunction {
public:

  // PDF-specific plotting & display

  /** Plot the negative log-likelihood of this PDF for the specified
   * dataset, as a function of the specified parameter. */
  TH1F *Scan(RooDataSet* data, RooRealVar &param, Int_t bins= 0);
  TH1F *Scan(RooDataSet& data, RooRealVar &param, Int_t bins= 0);
  /** Create a graph of the 2D contours at n-sigma with respect to
   * the specified variables, based on the results of the last fit. */
  TH2F *PlotContours(RooRealVar& var1, RooRealVar& var2,
		     Double_t n1= 1, Double_t n2= 2, Double_t n3= 0);
  /** Create a text box with our current parameter values and return
   * a pointer to it. */
  TPaveText *Parameters(const char *label= "", Int_t sigDigits = 2,
			Option_t *options = "NELU", Double_t xmin=0.65,
                        Double_t xmax= 0.99,Double_t ymax=0.95);

  // Interactions with a dataset

  /** Evaluate the negative log-likelihood of the specified data set using
   * our probability density function with the current parameter values.
   * Include extended maximum likelihood factors if the option is set. */
  Double_t nLogLikelihood(RooDataSet& data, Bool_t extended= kFALSE);
  /** Fit this object's parameters to the specified data set. */
  Int_t fitTo(RooDataSet& data, Option_t *options = "", Double_t *minValue= 0);
  Int_t fitTo(TH1F* hist, Option_t *options = "", Double_t *minValue= 0);

  // Interpret our clients as params / dependents --> RooNormFunction ?

  /** Get the number of parameters for this PDF (including constants) */
  inline Int_t getNPar() const { return _params.GetSize(); }
  /** Get a list of dependents for this PDF (including constants) */
  inline const RooVarList* getDependents() const { return &_depends; }
  /** Get a list of parameters for this PDF (including constants) */
  inline const RooVarList* getParameters() const { return &_params; }
  /** Call the setLimits() method on all of our parameters. */
  void setParamLimits(Bool_t value= kTRUE);
  /** Call the setConstant() method on all of our parameters. */
  void fixParameters(Bool_t value= kTRUE);

  // Access parameters via index
protected:
  /** Get a parameter for this PDF (including constants) */
  Double_t getParameter(Int_t index);
  /** Set the value of the indexed parameter. */
  void setParameter(Int_t index, Double_t value);
  /** Set the value of all the parameters. */
  void setParameters(const RooVarList* params);
  /** Set the parabolic error of the indexed parameter. */
  void setError(Int_t index, Double_t value);
  /** Get the parabolic error of the indexed parameter. */
  Double_t getError(Int_t index);	
public:

  /** custom PDF printToStream(...) */

  /** Support for extended maximum likelihood. */
  virtual Bool_t canBeExtended();
  virtual Double_t expectedEvents();
protected:  
  virtual Double_t extendedTerm(UInt_t observedEvents, Bool_t trace);
public:

  // friend void RooFitGlue(Int_t&,Double_t*,Double_t&,Double_t*,Int_t);
private:
  static void fitGlue(Int_t&,Double_t*,Double_t&,Double_t*,Int_t);

  ClassDef(RooAbsPdf,1) // a PDF abstract base class
};

#endif
