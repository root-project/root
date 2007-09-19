// @(#)root/minuit:$Id$
// Author: Rene Brun, Frederick James   12/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// ---------------------------------- minuit.h



//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMinuit                                                              //
//                                                                      //
// The MINUIT minimisation package (base class)                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMinuit
#define ROOT_TMinuit

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROOT_TMethodCall
#include "TMethodCall.h"
#endif

class TMinuit : public TNamed {

private:
   TMinuit(const TMinuit &m);
   TMinuit& operator=(const TMinuit &m); // Not implemented

// should become private....
public:
        enum{kMAXWARN=100};
          
        Int_t        fNpfix;            //Number of fixed parameters
        Int_t        fEmpty;            //Initialization flag (1 = Minuit initialized)
        Int_t        fMaxpar;           //Maximum number of parameters
        Int_t        fMaxint;           //Maximum number of internal parameters
        Int_t        fNpar;             //Number of free parameters (total number of pars = fNpar + fNfix)
        Int_t        fMaxext;           //Maximum number of external parameters
        Int_t        fMaxIterations;    //Maximum number of iterations
        Int_t        fMaxpar5;          // fMaxpar*(fMaxpar+1)/2
        Int_t        fMaxcpt;
        Int_t        fMaxpar2;          // fMaxpar*fMaxpar
        Int_t        fMaxpar1;          // fMaxpar*(fMaxpar+1)
        
        Double_t     fAmin;             //Minimum value found for FCN
        Double_t     fUp;               //FCN+-UP defines errors (for chisquare fits UP=1)
        Double_t     fEDM;              //Estimated vertical distance to the minimum
        Double_t     fFval3;            //
        Double_t     fEpsi;             //
        Double_t     fApsi;             //
        Double_t     fDcovar;           //Relative change in covariance matrix
        Double_t     fEpsmac;           //machine precision for floating points:
        Double_t     fEpsma2;           //sqrt(fEpsmac)
        Double_t     fVlimlo;           //
        Double_t     fVlimhi;           //
        Double_t     fUndefi;           //Undefined number = -54321
        Double_t     fBigedm;           //Big EDM = 123456
        Double_t     fUpdflt;           //
        Double_t     fXmidcr;           //
        Double_t     fYmidcr;           //
        Double_t     fXdircr;           //
        Double_t     fYdircr;           //
        
        Double_t     *fU;               //[fMaxpar2] External (visible to user in FCN) value of parameters
        Double_t     *fAlim;            //[fMaxpar2] Lower limits for parameters. If zero no limits
        Double_t     *fBlim;            //[fMaxpar2] Upper limits for parameters
        Double_t     *fErp;             //[fMaxpar] Positive Minos errors if calculated
        Double_t     *fErn;             //[fMaxpar] Negative Minos errors if calculated
        Double_t     *fWerr;            //[fMaxpar] External parameters error (standard deviation, defined by UP)
        Double_t     *fGlobcc;          //[fMaxpar] Global Correlation Coefficients
        Double_t     *fX;               //[fMaxpar] Internal parameters values
        Double_t     *fXt;              //[fMaxpar] Internal parameters values X saved as Xt
        Double_t     *fDirin;           //[fMaxpar] (Internal) step sizes for current step
        Double_t     *fXs;              //[fMaxpar] Internal parameters values saved for fixed params
        Double_t     *fXts;             //[fMaxpar] Internal parameters values X saved as Xt for fixed params
        Double_t     *fDirins;          //[fMaxpar] (Internal) step sizes for current step for fixed params
        Double_t     *fGrd;             //[fMaxpar] First derivatives
        Double_t     *fG2;              //[fMaxpar] 
        Double_t     *fGstep;           //[fMaxpar] Step sizes
        Double_t     *fGin;             //[fMaxpar2] 
        Double_t     *fDgrd;            //[fMaxpar] Uncertainties
        Double_t     *fGrds;            //[fMaxpar] 
        Double_t     *fG2s;             //[fMaxpar] 
        Double_t     *fGsteps;          //[fMaxpar] 
        Double_t     *fVhmat;           //[fMaxpar5] (Internal) error matrix stored as Half MATrix, since it is symmetric
        Double_t     *fVthmat;          //[fMaxpar5] VHMAT is sometimes saved in VTHMAT, especially in MNMNOT
        Double_t     *fP;               //[fMaxpar1] 
        Double_t     *fPstar;           //[fMaxpar2] 
        Double_t     *fPstst;           //[fMaxpar] 
        Double_t     *fPbar;            //[fMaxpar] 
        Double_t     *fPrho;            //[fMaxpar] Minimum point of parabola
        Double_t     *fWord7;           //[fMaxpar] 
        Double_t     *fXpt;             //[fMaxcpt] X array of points for contours
        Double_t     *fYpt;             //[fMaxcpt] Y array of points for contours
        
        Double_t     *fCONTgcc;         //[fMaxpar] array used in mncont
        Double_t     *fCONTw;           //[fMaxpar] array used in mncont
        Double_t     *fFIXPyy;          //[fMaxpar] array used in mnfixp
        Double_t     *fGRADgf;          //[fMaxpar] array used in mngrad
        Double_t     *fHESSyy;          //[fMaxpar] array used in mnhess
        Double_t     *fIMPRdsav;        //[fMaxpar] array used in mnimpr
        Double_t     *fIMPRy;           //[fMaxpar] array used in mnimpr
        Double_t     *fMATUvline;       //[fMaxpar] array used in mnmatu
        Double_t     *fMIGRflnu;        //[fMaxpar] array used in mnmigr
        Double_t     *fMIGRstep;        //[fMaxpar] array used in mnmigr
        Double_t     *fMIGRgs;          //[fMaxpar] array used in mnmigr
        Double_t     *fMIGRvg;          //[fMaxpar] array used in mnmigr
        Double_t     *fMIGRxxs;         //[fMaxpar] array used in mnmigr
        Double_t     *fMNOTxdev;        //[fMaxpar] array used in mnmnot
        Double_t     *fMNOTw;           //[fMaxpar] array used in mnmnot
        Double_t     *fMNOTgcc;         //[fMaxpar] array used in mnmnot
        Double_t     *fPSDFs;           //[fMaxpar] array used in mnpsdf
        Double_t     *fSEEKxmid;        //[fMaxpar] array used in mnseek
        Double_t     *fSEEKxbest;       //[fMaxpar] array used in mnseek
        Double_t     *fSIMPy;           //[fMaxpar] array used in mnsimp
        Double_t     *fVERTq;           //[fMaxpar] array used in mnvert
        Double_t     *fVERTs;           //[fMaxpar] array used in mnvert
        Double_t     *fVERTpp;          //[fMaxpar] array used in mnvert
        Double_t     *fCOMDplist;       //[fMaxpar] array used in mncomd
        Double_t     *fPARSplist;       //[fMaxpar] array used in mnpars
        
        Int_t        *fNvarl;           //[fMaxpar2] parameters flag (-1=undefined, 0=constant..)
        Int_t        *fNiofex;          //[fMaxpar2] Internal parameters number, or zero if not currently variable
        Int_t        *fNexofi;          //[fMaxpar] External parameters number for currently variable parameters
        Int_t        *fIpfix;           //[fMaxpar] List of fixed parameters
        Int_t        fNu;               //
        Int_t        fIsysrd;           //standardInput unit
        Int_t        fIsyswr;           //standard output unit
        Int_t        fIsyssa;           //
        Int_t        fNpagwd;           //Page width
        Int_t        fNpagln;           //Number of lines per page
        Int_t        fNewpag;           //
        Int_t        fIstkrd[10];       //
        Int_t        fNstkrd;           //
        Int_t        fIstkwr[10];       //
        Int_t        fNstkwr;           //
        Int_t        fISW[7];           //Array of switches
        Int_t        fIdbg[11];         //Array of internal debug switches
        Int_t        fNblock;           //Number of Minuit data blocks
        Int_t        fIcomnd;           //Number of commands
        Int_t        fNfcn;             //Number of calls to FCN
        Int_t        fNfcnmx;           //Maximum number of calls to FCN
        Int_t        fNfcnlc;           //
        Int_t        fNfcnfr;           //
        Int_t        fItaur;            //
        Int_t        fIstrat;           //
        Int_t        fNwrmes[2];        //
        Int_t        fNfcwar[20];       //
        Int_t        fIcirc[2];         //
        Int_t        fStatus;           //Status flag for the last called Minuit function
        Int_t        fKe1cr;            //
        Int_t        fKe2cr;            //
        Bool_t       fLwarn;            //true if warning messges are to be put out (default=true)
        Bool_t       fLrepor;           //true if exceptional conditions are put out (default=false)
        Bool_t       fLimset;           //true if a parameter is up against limits (for MINOS)
        Bool_t       fLnolim;           //true if there are no limits on any parameters (not yet used)
        Bool_t       fLnewmn;           //true if the previous process has unexpectedly improved FCN
        Bool_t       fLphead;           //true if a heading should be put out for the next parameter definition
        Bool_t       fGraphicsMode;     //true if graphics mode on (default)
        char         *fChpt;            //!Character to be plotted at the X,Y contour positions
        TString      *fCpnam;           //[fMaxpar2] Array of parameters names
        TString      fCfrom;            //
        TString      fCstatu;           //
        TString      fCtitl;            //
        TString      fCword;            //
        TString      fCundef;           //
        TString      fCvrsn;            //
        TString      fCovmes[4];        //
        TString      fOrigin[kMAXWARN]; //
        TString      fWarmes[kMAXWARN]; //
        TObject      *fObjectFit;       //Pointer to object being fitted
        TObject      *fPlot;            //Pointer to TGraph object created by mncont
        TMethodCall  *fMethodCall;      //Pointer to MethodCall in case of interpreted function
        void         (*fFCN)(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag); //!

// methods performed on TMinuit class
public:
   TMinuit();
   TMinuit(Int_t maxpar);
   virtual       ~TMinuit();
   virtual void   BuildArrays(Int_t maxpar=15);
   virtual TObject *Clone(const char *newname="") const;   //Clone-Method to copy the function-pointer fFCN
   virtual Int_t  Command(const char *command);
   virtual TObject *Contour(Int_t npoints=10, Int_t pa1=0, Int_t pa2=1);
   virtual Int_t  DefineParameter( Int_t parNo, const char *name, Double_t initVal, Double_t initErr, Double_t lowerLimit, Double_t upperLimit );
   virtual void   DeleteArrays();
   virtual Int_t  Eval(Int_t npar, Double_t *grad, Double_t &fval, Double_t *par, Int_t flag);
   virtual Int_t  FixParameter( Int_t parNo );
   TMethodCall   *GetMethodCall() const {return fMethodCall;}
   TObject       *GetObjectFit() const {return fObjectFit;}
   Int_t          GetMaxIterations() const {return fMaxIterations;}
   virtual Int_t  GetNumFixedPars() const;
   virtual Int_t  GetNumFreePars() const;
   virtual Int_t  GetNumPars() const;
   virtual Int_t  GetParameter( Int_t parNo, Double_t &currentValue, Double_t &currentError ) const;
   virtual TObject *GetPlot() const {return fPlot;}
   Int_t          GetStatus() const {return fStatus;}
   virtual Int_t  Migrad();
   virtual void   mnamin();
   virtual void   mnbins(Double_t a1, Double_t a2, Int_t naa, Double_t &bl, Double_t &bh, Int_t &nb, Double_t &bwid);
   virtual void   mncalf(Double_t *pvec, Double_t &ycalf);
   virtual void   mncler();
   virtual void   mncntr(Int_t ke1, Int_t ke2, Int_t &ierrf);
   virtual void   mncomd(const char *crdbin, Int_t &icondn);
   virtual void   mncont(Int_t ke1, Int_t ke2, Int_t nptu, Double_t *xptu, Double_t *yptu, Int_t &ierrf);
   virtual void   mncrck(TString crdbuf, Int_t maxcwd, TString &comand, Int_t &lnc
                    ,  Int_t mxp, Double_t *plist, Int_t &llist, Int_t &ierr, Int_t isyswr);
   virtual void   mncros(Double_t &aopt, Int_t &iercr);
   virtual void   mncuve();
   virtual void   mnderi();
   virtual void   mndxdi(Double_t pint, Int_t ipar, Double_t &dxdi);
   virtual void   mneig(Double_t *a, Int_t ndima, Int_t n, Int_t mits, Double_t *work, Double_t precis, Int_t &ifault);
   virtual void   mnemat(Double_t *emat, Int_t ndim);
   virtual void   mnerrs(Int_t number, Double_t &eplus, Double_t &eminus, Double_t &eparab, Double_t &gcc);
   virtual void   mneval(Double_t anext, Double_t &fnext, Int_t &ierev);
   virtual void   mnexcm(const char *comand, Double_t *plist, Int_t llist, Int_t &ierflg) ;
   virtual void   mnexin(Double_t *pint);
   virtual void   mnfixp(Int_t iint, Int_t &ierr);
   virtual void   mnfree(Int_t k);
   virtual void   mngrad();
   virtual void   mnhelp(TString comd);
   virtual void   mnhelp(const char *command="");
   virtual void   mnhess();
   virtual void   mnhes1();
   virtual void   mnimpr();
   virtual void   mninex(Double_t *pint);
   virtual void   mninit(Int_t i1, Int_t i2, Int_t i3);
   virtual void   mnlims();
   virtual void   mnline(Double_t *start, Double_t fstart, Double_t *step, Double_t slope, Double_t toler);
   virtual void   mnmatu(Int_t kode);
   virtual void   mnmigr();
   virtual void   mnmnos();
   virtual void   mnmnot(Int_t ilax, Int_t ilax2, Double_t &val2pl, Double_t &val2mi);
   virtual void   mnparm(Int_t k, TString cnamj, Double_t uk, Double_t wk, Double_t a, Double_t b, Int_t &ierflg);
   virtual void   mnpars(TString &crdbuf, Int_t &icondn);
   virtual void   mnpfit(Double_t *parx2p, Double_t *pary2p, Int_t npar2p, Double_t *coef2p, Double_t &sdev2p);
   virtual void   mnpint(Double_t &pexti, Int_t i, Double_t &pinti);
   virtual void   mnplot(Double_t *xpt, Double_t *ypt, char *chpt, Int_t nxypt, Int_t npagwd, Int_t npagln);
   virtual void   mnpout(Int_t iuext, TString &chnam, Double_t &val, Double_t &err, Double_t &xlolim, Double_t &xuplim, Int_t &iuint) const;
   virtual void   mnprin(Int_t inkode, Double_t fval);
   virtual void   mnpsdf();
   virtual void   mnrazz(Double_t ynew, Double_t *pnew, Double_t *y, Int_t &jh, Int_t &jl);
   virtual void   mnrn15(Double_t &val, Int_t &inseed);
   virtual void   mnrset(Int_t iopt);
   virtual void   mnsave();
   virtual void   mnscan();
   virtual void   mnseek();
   virtual void   mnset();
   virtual void   mnsimp();
   virtual void   mnstat(Double_t &fmin, Double_t &fedm, Double_t &errdef, Int_t &npari, Int_t &nparx, Int_t &istat);
   virtual void   mntiny(Double_t epsp1, Double_t &epsbak);
   Bool_t         mnunpt(TString &cfname);
   virtual void   mnvert(Double_t *a, Int_t l, Int_t m, Int_t n, Int_t &ifail);
   virtual void   mnwarn(const char *copt, const char *corg, const char *cmes);
   virtual void   mnwerr();
   virtual Int_t  Release( Int_t parNo );
   virtual Int_t  SetErrorDef( Double_t up );
   virtual void   SetFCN(void *fcn);
   virtual void   SetFCN(void (*fcn)(Int_t &, Double_t *, Double_t &f, Double_t *, Int_t));
   virtual void   SetGraphicsMode(Bool_t mode=kTRUE) {fGraphicsMode = mode;}
   virtual void   SetMaxIterations(Int_t maxiter=500) {fMaxIterations = maxiter;}
   virtual void   SetObjectFit(TObject *obj) {fObjectFit=obj;}
   virtual Int_t  SetPrintLevel( Int_t printLevel=0 );

   ClassDef(TMinuit,1)  //The MINUIT minimisation package
};

R__EXTERN TMinuit  *gMinuit;

#endif

