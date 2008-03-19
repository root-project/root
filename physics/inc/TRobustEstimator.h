// @(#)root/physics:$Id$
// Author: Anna Kreshuk  08/10/2004


//////////////////////////////////////////////////////////////////////////////
//
//  TRobustEstimator
//
// Minimum Covariance Determinant Estimator - a Fast Algorithm
// invented by Peter J.Rousseeuw and Katrien Van Dreissen
// "A Fast Algorithm for the Minimum covariance Determinant Estimator"
// Technometrics, August 1999, Vol.41, NO.3
//
//////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TRobustEstimator
#define ROOT_TRobustEstimator

#include "TArrayI.h"
#include "TMatrixDSym.h"
#include "TMatrixDSymEigen.h"

class TRobustEstimator : public TObject {

protected:

   Int_t        fNvar;          //number of variables
   Int_t        fH;             //algorithm parameter, determining the subsample size
   Int_t        fN;             //number of observations

   Int_t        fVarTemp;       //number of variables already added to the data matrix
   Int_t        fVecTemp;       //number of observations already added to the data matrix

   Int_t        fExact;         //if there was an exact fit, stores the number of points on a hyperplane 

   TVectorD     fMean;          //location estimate (mean values)
   TMatrixDSym  fCovariance;    //covariance matrix estimate
   TMatrixDSym  fInvcovariance; //inverse of the covariance matrix
   TMatrixDSym  fCorrelation;   //correlation matrix
   TVectorD     fRd;            //array of robust distances, size n
   TVectorD     fSd;            //array of standard deviations
   TArrayI      fOut;           //array of indexes of ouliers, size <0.5*n
   TVectorD     fHyperplane;    //in case more than fH observations lie on a hyperplane
                               //the equation of this hyperplane is stored here
 
   TMatrixD fData;              //the original data

   //functions needed for evaluation

   void     AddToSscp(TMatrixD &sscp, TVectorD &vec);
   void     ClearSscp(TMatrixD &sscp); 

   void     Classic();
   void     Covar(TMatrixD &sscp, TVectorD &m, TMatrixDSym &cov, TVectorD &sd, Int_t nvec); 
   void     Correl();

   void     CreateSubset(Int_t ntotal, Int_t htotal, Int_t p, Int_t *index, TMatrixD &data, 
                    TMatrixD &sscp, Double_t *ndist);
   void     CreateOrtSubset(TMatrixD &dat, Int_t *index, Int_t hmerged, Int_t nmerged, TMatrixD &sscp, Double_t *ndist);

   Double_t CStep(Int_t ntotal, Int_t htotal, Int_t *index, TMatrixD &data, TMatrixD &sscp, Double_t *ndist);

   Int_t    Exact(Double_t *ndist); 
   Int_t    Exact2(TMatrixD &mstockbig, TMatrixD &cstockbig, TMatrixD &hyperplane,
               Double_t *deti, Int_t nbest,Int_t kgroup, 
               TMatrixD &sscp, Double_t *ndist);

   Int_t    Partition(Int_t nmini, Int_t *indsubdat); 
   Int_t    RDist(TMatrixD &sscp);
   void     RDraw(Int_t *subdat, Int_t ngroup, Int_t *indsubdat);

   Double_t KOrdStat(Int_t ntotal, Double_t *arr, Int_t k, Int_t *work);

public:

   TRobustEstimator();
   TRobustEstimator(Int_t nvectors, Int_t nvariables, Int_t hh=0);
   virtual ~TRobustEstimator(){;}

   void    AddColumn(Double_t *col);         //adds a column to the data matrix
   void    AddRow(Double_t *row);            //adds a row to the data matrix

   void    Evaluate();
   void    EvaluateUni(Int_t nvectors, Double_t *data, Double_t &mean, Double_t &sigma, Int_t hh=0);

   Int_t   GetBDPoint();                     //returns the breakdown point of the algorithm

   void    GetCovariance(TMatrixDSym &matr); //returns robust covariance matrix estimate
   const   TMatrixDSym* GetCovariance() const{return &fCovariance;}
   void    GetCorrelation(TMatrixDSym &matr); //returns robust correlation matrix estimate
   const   TMatrixDSym* GetCorrelation() const{return &fCorrelation;}
   void    GetHyperplane(TVectorD &vec);      //if the data lies on a hyperplane, returns this hyperplane
   const   TVectorD* GetHyperplane() const;   //if the data lies on a hyperplane, returns this hyperplane
   Int_t   GetNHyp() {return fExact;}         //returns the number of points on a hyperplane
   void    GetMean(TVectorD &means);                        //returns robust mean vector estimate
   const   TVectorD* GetMean() const {return &fMean;}       //returns robust mean vector estimate
   void    GetRDistances(TVectorD &rdist);                  //returns robust distances of all observations
   const   TVectorD* GetRDistances() const {return &fRd;}   //returns robust distances of all observations
   Int_t   GetNumberObservations() const {return fN;}
   Int_t   GetNvar() const {return fNvar;}
   const   TArrayI* GetOuliers() const{return &fOut;}       //returns an array of outlier indexes
   Int_t   GetNOut(); //returns the number of points outside the tolerance ellipsoid.
                      //ONLY those with robust distances significantly larger than the
                      //cutoff value, should be considered outliers!
   Double_t GetChiQuant(Int_t i) const;
   
   ClassDef(TRobustEstimator,1)  //Minimum Covariance Determinant Estimator
 
};


#endif

