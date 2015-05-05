// @(#)root/physics:$Id$
// Author: Anna Kreshuk  08/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////////
//
//  TRobustEstimator
//
// Minimum Covariance Determinant Estimator - a Fast Algorithm
// invented by Peter J.Rousseeuw and Katrien Van Dreissen
// "A Fast Algorithm for the Minimum covariance Determinant Estimator"
// Technometrics, August 1999, Vol.41, NO.3
//
// What are robust estimators?
// "An important property of an estimator is its robustness. An estimator
// is called robust if it is insensitive to measurements that deviate
// from the expected behaviour. There are 2 ways to treat such deviating
// measurements: one may either try to recongize them and then remove
// them from the data sample; or one may leave them in the sample, taking
// care that they do not influence the estimate unduly. In both cases robust
// estimators are needed...Robust procedures compensate for systematic errors
// as much as possible, and indicate any situation in which a danger of not being
// able to operate reliably is detected."
// R.Fruhwirth, M.Regler, R.K.Bock, H.Grote, D.Notz
// "Data Analysis Techniques for High-Energy Physics", 2nd edition
//
// What does this algorithm do?
// It computes a highly robust estimator of multivariate location and scatter.
// Then, it takes those estimates to compute robust distances of all the
// data vectors. Those with large robust distances are considered outliers.
// Robust distances can then be plotted for better visualization of the data.
//
// How does this algorithm do it?
// The MCD objective is to find h observations(out of n) whose classical
// covariance matrix has the lowest determinant. The MCD estimator of location
// is then the average of those h points and the MCD estimate of scatter
// is their covariance matrix. The minimum(and default) h = (n+nvariables+1)/2
// so the algorithm is effective when less than (n+nvar+1)/2 variables are outliers.
// The algorithm also allows for exact fit situations - that is, when h or more
// observations lie on a hyperplane. Then the algorithm still yields the MCD location T
// and scatter matrix S, the latter being singular as it should be. From (T,S) the
// program then computes the equation of the hyperplane.
//
// How can this algorithm be used?
// In any case, when contamination of data is suspected, that might influence
// the classical estimates.
// Also, robust estimation of location and scatter is a tool to robustify
// other multivariate techniques such as, for example, principal-component analysis
// and discriminant analysis.
//
//
//
//
// Technical details of the algorithm:
// 0.The default h = (n+nvariables+1)/2, but the user may choose any interger h with
//   (n+nvariables+1)/2<=h<=n. The program then reports the MCD's breakdown value
//   (n-h+1)/n. If you are sure that the dataset contains less than 25% contamination
//   which is usually the case, a good compromise between breakdown value and
//  efficiency is obtained by putting h=[.75*n].
// 1.If h=n,the MCD location estimate is the average of the whole dataset, and
//   the MCD scatter estimate is its covariance matrix. Report this and stop
// 2.If nvariables=1 (univariate data), compute the MCD estimate by the exact
//   algorithm of Rousseeuw and Leroy (1987, pp.171-172) in O(nlogn)time and stop
// 3.From here on, h<n and nvariables>=2.
//   3a.If n is small:
//    - repeat (say) 500 times:
//    -- construct an initial h-subset, starting from a random (nvar+1)-subset
//    -- carry out 2 C-steps (described in the comments of CStep function)
//    - for the 10 results with lowest det(S):
//    -- carry out C-steps until convergence
//    - report the solution (T, S) with the lowest det(S)
//   3b.If n is larger (say, n>600), then
//    - construct up to 5 disjoint random subsets of size nsub (say, nsub=300)
//    - inside each subset repeat 500/5 times:
//    -- construct an initial subset of size hsub=[nsub*h/n]
//    -- carry out 2 C-steps
//    -- keep the best 10 results (Tsub, Ssub)
//    - pool the subsets, yielding the merged set (say, of size nmerged=1500)
//    - in the merged set, repeat for each of the 50 solutions (Tsub, Ssub)
//    -- carry out 2 C-steps
//    -- keep the 10 best results
//    - in the full dataset, repeat for those best results:
//    -- take several C-steps, using n and h
//    -- report the best final result (T, S)
// 4.To obtain consistency when the data comes from a multivariate normal
//   distribution, covariance matrix is multiplied by a correction factor
// 5.Robust distances for all elements, using the final (T, S) are calculated
//   Then the very final mean and covariance estimates are calculated only for
//   values, whose robust distances are less than a cutoff value (0.975 quantile
//   of chi2 distribution with nvariables degrees of freedom)
//
//////////////////////////////////////////////////////////////////////////////

#include "TRobustEstimator.h"
#include "TRandom.h"
#include "TMath.h"
#include "TH1D.h"
#include "TPaveLabel.h"
#include "TDecompChol.h"

ClassImp(TRobustEstimator)

const Double_t kChiMedian[50]= {
         0.454937, 1.38629, 2.36597, 3.35670, 4.35146, 5.34812, 6.34581, 7.34412, 8.34283,
         9.34182, 10.34, 11.34, 12.34, 13.34, 14.34, 15.34, 16.34, 17.34, 18.34, 19.34,
        20.34, 21.34, 22.34, 23.34, 24.34, 25.34, 26.34, 27.34, 28.34, 29.34, 30.34,
        31.34, 32.34, 33.34, 34.34, 35.34, 36.34, 37.34, 38.34, 39.34, 40.34,
        41.34, 42.34, 43.34, 44.34, 45.34, 46.34, 47.34, 48.34, 49.33};

const Double_t kChiQuant[50]={
         5.02389, 7.3776,9.34840,11.1433,12.8325,
        14.4494,16.0128,17.5346,19.0228,20.4831,21.920,23.337,
        24.736,26.119,27.488,28.845,30.191,31.526,32.852,34.170,
        35.479,36.781,38.076,39.364,40.646,41.923,43.194,44.461,
        45.722,46.979,48.232,49.481,50.725,51.966,53.203,54.437,
        55.668,56.896,58.120,59.342,60.561,61.777,62.990,64.201,
        65.410,66.617,67.821,69.022,70.222,71.420};

//_____________________________________________________________________________
TRobustEstimator::TRobustEstimator(){
  //this constructor should be used in a univariate case:
  //first call this constructor, then - the EvaluateUni(..) function

}

//______________________________________________________________________________
TRobustEstimator::TRobustEstimator(Int_t nvectors, Int_t nvariables, Int_t hh)
   :fMean(nvariables),
    fCovariance(nvariables),
    fInvcovariance(nvariables),
    fCorrelation(nvariables),
    fRd(nvectors),
    fSd(nvectors),
    fOut(1),
    fHyperplane(nvariables),
    fData(nvectors, nvariables)
{
   //constructor

   if ((nvectors<=1)||(nvariables<=0)){
      Error("TRobustEstimator","Not enough vectors or variables");
      return;
   }
   if (nvariables==1){
      Error("TRobustEstimator","For the univariate case, use the default constructor and EvaluateUni() function");
      return;
   }
   
   fN=nvectors;
   fNvar=nvariables;
   if (hh<(fN+fNvar+1)/2){
      if (hh>0)
         Warning("TRobustEstimator","chosen h is too small, default h is taken instead");
      fH=(fN+fNvar+1)/2;
   } else
      fH=hh;
   
   fVarTemp=0;
   fVecTemp=0;
   fExact=0;
}

//_____________________________________________________________________________
void TRobustEstimator::AddColumn(Double_t *col)
{
   //adds a column to the data matrix
   //it is assumed that the column has size fN
   //variable fVarTemp keeps the number of columns l
   //already added
   
   if (fVarTemp==fNvar) {
      fNvar++;
      fCovariance.ResizeTo(fNvar, fNvar);
      fInvcovariance.ResizeTo(fNvar, fNvar);
      fCorrelation.ResizeTo(fNvar, fNvar);
      fMean.ResizeTo(fNvar);
      fHyperplane.ResizeTo(fNvar);
      fData.ResizeTo(fN, fNvar);
   }
   for (Int_t i=0; i<fN; i++) {
      fData(i, fVarTemp)=col[i];
   }
   fVarTemp++;
}

//_______________________________________________________________________________
void TRobustEstimator::AddRow(Double_t *row)
{
   //adds a vector to the data matrix
   //it is supposed that the vector is of size fNvar

   if(fVecTemp==fN) {
      fN++;
      fRd.ResizeTo(fN);
      fSd.ResizeTo(fN);
      fData.ResizeTo(fN, fNvar);
   }
   for (Int_t i=0; i<fNvar; i++)
      fData(fVecTemp, i)=row[i];
   
   fVecTemp++;
}

//_______________________________________________________________________________
void TRobustEstimator::Evaluate()
{
//Finds the estimate of multivariate mean and variance

   Double_t kEps=1e-14;
   
   if (fH==fN){
      Warning("Evaluate","Chosen h = #observations, so classic estimates of location and scatter will be calculated");
      Classic();
      return;
   }
   
   Int_t i, j, k;
   Int_t ii, jj;
   Int_t nmini = 300;
   Int_t k1=500;
   Int_t nbest=10;
   TMatrixD sscp(fNvar+1, fNvar+1);
   TVectorD vec(fNvar);
   
   Int_t *index = new Int_t[fN];
   Double_t *ndist = new Double_t[fN];
   Double_t det;
   Double_t *deti=new Double_t[nbest];
   for (i=0; i<nbest; i++)
      deti[i]=1e16;
   
   for (i=0; i<fN; i++)
      fRd(i)=0;
   ////////////////////////////
   //for small n
   ////////////////////////////
   if (fN<nmini*2) {
      //for storing the best fMeans and covariances
      
      TMatrixD mstock(nbest, fNvar);
      TMatrixD cstock(fNvar, fNvar*nbest);
      
      for (k=0; k<k1; k++) {
         CreateSubset(fN, fH, fNvar, index, fData, sscp, ndist);
         //calculate the mean and covariance of the created subset
         ClearSscp(sscp);
         for (i=0; i<fH; i++) {
            for(j=0; j<fNvar; j++)
               vec(j)=fData[index[i]][j];
            AddToSscp(sscp, vec);
         }
         Covar(sscp, fMean, fCovariance, fSd, fH);
         det = fCovariance.Determinant();
         if (det < kEps) {
            fExact = Exact(ndist);
            delete [] index;
            delete [] ndist;
            delete [] deti;
            return;
         }
         //make 2 CSteps
         det = CStep(fN, fH, index, fData, sscp, ndist);
         if (det < kEps) {
            fExact = Exact(ndist);
            delete [] index;
            delete [] ndist;
            delete [] deti;
            return;
         }
         det = CStep(fN, fH, index, fData, sscp, ndist);
         if (det < kEps) {
            fExact = Exact(ndist);
            delete [] index;
            delete [] ndist;
            delete [] deti;
            return;
         } else {
            Int_t maxind=TMath::LocMax(nbest, deti);
            if(det<deti[maxind]) {
               deti[maxind]=det;
               for(ii=0; ii<fNvar; ii++) {
                  mstock(maxind, ii)=fMean(ii);
                  for(jj=0; jj<fNvar; jj++)
                     cstock(ii, jj+maxind*fNvar)=fCovariance(ii, jj);
               }
            }
         }
      }
      
      //now for nbest best results perform CSteps until convergence
      
      for (i=0; i<nbest; i++) {
         for(ii=0; ii<fNvar; ii++) {
            fMean(ii)=mstock(i, ii);
            for (jj=0; jj<fNvar; jj++)
               fCovariance(ii, jj)=cstock(ii, jj+i*fNvar);
         }
         
         det=1;
         while (det>kEps) {
            det=CStep(fN, fH, index, fData, sscp, ndist);
            if(TMath::Abs(det-deti[i])<kEps)
               break;
            else
               deti[i]=det;
         }
         for(ii=0; ii<fNvar; ii++) {
            mstock(i,ii)=fMean(ii);
            for (jj=0; jj<fNvar; jj++)
               cstock(ii,jj+i*fNvar)=fCovariance(ii, jj);
         }
      }
      
      Int_t detind=TMath::LocMin(nbest, deti);
      for(ii=0; ii<fNvar; ii++) {
         fMean(ii)=mstock(detind,ii);
         
         for(jj=0; jj<fNvar; jj++)
            fCovariance(ii, jj)=cstock(ii,jj+detind*fNvar);
      }
      
      if (deti[detind]!=0) {
         //calculate robust distances and throw out the bad points
         Int_t nout = RDist(sscp);
         Double_t cutoff=kChiQuant[fNvar-1];
         
         fOut.Set(nout);
         
         j=0;
         for (i=0; i<fN; i++) {
            if(fRd(i)>cutoff) {
               fOut[j]=i;
               j++;
            }
         }
         
      } else {
         fExact=Exact(ndist);
      }
      delete [] index;
      delete [] ndist;
      delete [] deti;
      return;
      
   }
   /////////////////////////////////////////////////
  //if n>nmini, the dataset should be partitioned
  //partitioning
  ////////////////////////////////////////////////
   Int_t indsubdat[5];
   Int_t nsub;
   for (ii=0; ii<5; ii++)
      indsubdat[ii]=0;
   
   nsub = Partition(nmini, indsubdat);
   
   Int_t sum=0;
   for (ii=0; ii<5; ii++)
      sum+=indsubdat[ii];
   Int_t *subdat=new Int_t[sum];
   //printf("allocates subdat[ %d ]\n", sum);
   // init the subdat matrix
   for (int iii = 0; iii < sum; ++iii) subdat[iii] = -999;
   RDraw(subdat, nsub, indsubdat);
   for (int iii = 0; iii < sum; ++iii) {
      if (subdat[iii] < 0 || subdat[iii] >= fN ) {
         Error("Evaluate","subdat index is invalid subdat[%d] = %d",iii, subdat[iii] );
         R__ASSERT(0);
      }
   }
   //now the indexes of selected cases are in the array subdat
   //matrices to store best means and covariances
   Int_t nbestsub=nbest*nsub;
   TMatrixD mstockbig(nbestsub, fNvar);
   TMatrixD cstockbig(fNvar, fNvar*nbestsub);
   TMatrixD hyperplane(nbestsub, fNvar);
   for (i=0; i<nbestsub; i++) {
      for(j=0; j<fNvar; j++)
         hyperplane(i,j)=0;
   }
   Double_t *detibig = new Double_t[nbestsub];
   Int_t maxind;
   maxind=TMath::LocMax(5, indsubdat);
   TMatrixD dattemp(indsubdat[maxind], fNvar);
   
   Int_t k2=Int_t(k1/nsub);
   //construct h-subsets and perform 2 CSteps in subgroups
   
   for (Int_t kgroup=0; kgroup<nsub; kgroup++) {
      //printf("group #%d\n", kgroup);
      Int_t ntemp=indsubdat[kgroup];
      Int_t temp=0;
      for (i=0; i<kgroup; i++)
         temp+=indsubdat[i];
      Int_t par;
      
      for(i=0; i<ntemp; i++) {
         for (j=0; j<fNvar; j++) {
            dattemp(i,j)=fData[subdat[temp+i]][j];
         }
      }
      Int_t htemp=Int_t(fH*ntemp/fN);
      
      for (i=0; i<nbest; i++)
         deti[i]=1e16;
      
      for(k=0; k<k2; k++) {
         CreateSubset(ntemp, htemp, fNvar, index, dattemp, sscp, ndist);
         ClearSscp(sscp);
         for (i=0; i<htemp; i++) {
            for(j=0; j<fNvar; j++) {
               vec(j)=dattemp(index[i],j);
            }
            AddToSscp(sscp, vec);
         }
         Covar(sscp, fMean, fCovariance, fSd, htemp);
         det = fCovariance.Determinant();
         if (det<kEps) {
            par =Exact2(mstockbig, cstockbig, hyperplane, deti, nbest, kgroup, sscp,ndist);
            if(par==nbest+1) {
               
               delete [] detibig;
               delete [] deti;
               delete [] subdat;
               delete [] ndist;
               delete [] index;
               return;
            } else
               deti[par]=det;
         } else {
            det = CStep(ntemp, htemp, index, dattemp, sscp, ndist);
            if (det<kEps) {
               par=Exact2(mstockbig, cstockbig, hyperplane, deti, nbest, kgroup, sscp, ndist);
               if(par==nbest+1) {
                  
                  delete [] detibig;
                  delete [] deti;
                  delete [] subdat;
                  delete [] ndist;
                  delete [] index;
                  return;
               } else
                  deti[par]=det;
            } else {
               det=CStep(ntemp,htemp, index, dattemp, sscp, ndist);
               if(det<kEps){
                  par=Exact2(mstockbig, cstockbig, hyperplane, deti, nbest, kgroup, sscp,ndist);
                  if(par==nbest+1) {
                     
                     delete [] detibig;
                     delete [] deti;
                     delete [] subdat;
                     delete [] ndist;
                     delete [] index;
                     return;
                  } else {
                     deti[par]=det;
                  }
               } else {
                  maxind=TMath::LocMax(nbest, deti);
                  if(det<deti[maxind]) {
                     deti[maxind]=det;
                     for(i=0; i<fNvar; i++) {
                        mstockbig(nbest*kgroup+maxind,i)=fMean(i);
                        for(j=0; j<fNvar; j++) {
                           cstockbig(i,nbest*kgroup*fNvar+maxind*fNvar+j)=fCovariance(i,j);
                           
                        }
                     }
                  }
                  
               }
            }
         }
         
         maxind=TMath::LocMax(nbest, deti);
         if (deti[maxind]<kEps)
            break;
      }
      
      
      for(i=0; i<nbest; i++) {
         detibig[kgroup*nbest + i]=deti[i];
         
      }
      
   }
   
   //now the arrays mstockbig and cstockbig store nbest*nsub best means and covariances
   //detibig stores nbest*nsub their determinants
   //merge the subsets and carry out 2 CSteps on the merged set for all 50 best solutions
   
   TMatrixD datmerged(sum, fNvar);
   for(i=0; i<sum; i++) {
      for (j=0; j<fNvar; j++)
         datmerged(i,j)=fData[subdat[i]][j];
   }
   //  printf("performing calculations for merged set\n");
   Int_t hmerged=Int_t(sum*fH/fN);
   
   Int_t nh;
   for(k=0; k<nbestsub; k++) {
      //for all best solutions perform 2 CSteps and then choose the very best
      for(ii=0; ii<fNvar; ii++) {
         fMean(ii)=mstockbig(k,ii);
         for(jj=0; jj<fNvar; jj++)
            fCovariance(ii, jj)=cstockbig(ii,k*fNvar+jj);
      }
      if(detibig[k]==0) {
         for(i=0; i<fNvar; i++)
            fHyperplane(i)=hyperplane(k,i);
         CreateOrtSubset(datmerged,index, hmerged, sum, sscp, ndist);
         
      }
      det=CStep(sum, hmerged, index, datmerged, sscp, ndist);
      if (det<kEps) {
         nh= Exact(ndist);
         if (nh>=fH) {
            fExact = nh;
            
            delete [] detibig;
            delete [] deti;
            delete [] subdat;
            delete [] ndist;
            delete [] index;
            return;
         } else {
            CreateOrtSubset(datmerged, index, hmerged, sum, sscp, ndist);
         }
      }
      
      det=CStep(sum, hmerged, index, datmerged, sscp, ndist);
      if (det<kEps) {
         nh=Exact(ndist);
         if (nh>=fH) {
            fExact = nh;
            delete [] detibig;
            delete [] deti;
            delete [] subdat;
            delete [] ndist;
            delete [] index;
            return;
         }
      }
      detibig[k]=det;
      for(i=0; i<fNvar; i++) {
         mstockbig(k,i)=fMean(i);
         for(j=0; j<fNvar; j++) {
            cstockbig(i,k*fNvar+j)=fCovariance(i, j);
         }
      }
   }
   //now for the subset with the smallest determinant
   //repeat CSteps until convergence
   Int_t minind=TMath::LocMin(nbestsub, detibig);
   det=detibig[minind];
   for(i=0; i<fNvar; i++) {
      fMean(i)=mstockbig(minind,i);
      fHyperplane(i)=hyperplane(minind,i);
      for(j=0; j<fNvar; j++)
         fCovariance(i, j)=cstockbig(i,minind*fNvar + j);
   }
   if(det<kEps)
      CreateOrtSubset(fData, index, fH, fN, sscp, ndist);
   det=1;
   while (det>kEps) {
      det=CStep(fN, fH, index, fData, sscp, ndist);
      if(TMath::Abs(det-detibig[minind])<kEps) {
         break;
      } else {
         detibig[minind]=det;
      }
   }
   if(det<kEps) {
      Exact(ndist);
      fExact=kTRUE;
   }
   Int_t nout = RDist(sscp);
   Double_t cutoff=kChiQuant[fNvar-1];
   
   fOut.Set(nout);
   
   j=0;
   for (i=0; i<fN; i++) {
      if(fRd(i)>cutoff) {
         fOut[j]=i;
         j++;
      }
   }
   
   delete [] detibig;
   delete [] deti;
   delete [] subdat;
   delete [] ndist;
   delete [] index;
   return;
}

//___________________________________________________________________________________________
void TRobustEstimator::EvaluateUni(Int_t nvectors, Double_t *data, Double_t &mean, Double_t &sigma, Int_t hh)
{
   //for the univariate case
   //estimates of location and scatter are returned in mean and sigma parameters
   //the algorithm works on the same principle as in multivariate case -
   //it finds a subset of size hh with smallest sigma, and then returns mean and
   //sigma of this subset
   
   if (hh==0)
      hh=(nvectors+2)/2;
   Double_t faclts[]={2.6477,2.5092,2.3826,2.2662,2.1587,2.0589,1.9660,1.879,1.7973,1.7203,1.6473};
   Int_t *index=new Int_t[nvectors];
   TMath::Sort(nvectors, data, index, kFALSE);
   
   Int_t nquant;
   nquant=TMath::Min(Int_t(Double_t(((hh*1./nvectors)-0.5)*40))+1, 11);
   Double_t factor=faclts[nquant-1];
   
   Double_t *aw=new Double_t[nvectors];
   Double_t *aw2=new Double_t[nvectors];
   Double_t sq=0;
   Double_t sqmin=0;
   Int_t ndup=0;
   Int_t len=nvectors-hh;
   Double_t *slutn=new Double_t[len];
   for(Int_t i=0; i<len; i++)
      slutn[i]=0;
   for(Int_t jint=0; jint<len; jint++) {
      aw[jint]=0;
      for (Int_t j=0; j<hh; j++) {
         aw[jint]+=data[index[j+jint]];
         if(jint==0)
            sq+=data[index[j]]*data[index[j]];
      }
      aw2[jint]=aw[jint]*aw[jint]/hh;
      
      if(jint==0) {
         sq=sq-aw2[jint];
         sqmin=sq;
         slutn[ndup]=aw[jint];
         
      } else {
         sq=sq - data[index[jint-1]]*data[index[jint-1]]+
            data[index[jint+hh]]*data[index[jint+hh]]-
            aw2[jint]+aw2[jint-1];
         if(sq<sqmin) {
            ndup=0;
            sqmin=sq;
            slutn[ndup]=aw[jint];
            
         } else {
            if(sq==sqmin) {
               ndup++;
               slutn[ndup]=aw[jint];
            }
         }
      }
   }
   
   slutn[0]=slutn[Int_t((ndup)/2)]/hh;
   Double_t bstd=factor*TMath::Sqrt(sqmin/hh);
   mean=slutn[0];
   sigma=bstd;
   delete [] aw;
   delete [] aw2;
   delete [] slutn;
   delete [] index;
}

//_______________________________________________________________________
Int_t TRobustEstimator::GetBDPoint()
{
  //returns the breakdown point of the algorithm

   Int_t n;
   n=(fN-fH+1)/fN;
   return n;
}

//_______________________________________________________________________
Double_t TRobustEstimator::GetChiQuant(Int_t i) const
{
   //returns the chi2 quantiles

   if (i < 0 || i >= 50) return 0;
   return kChiQuant[i];
}

//_______________________________________________________________________
void TRobustEstimator::GetCovariance(TMatrixDSym &matr)
{
   //returns the covariance matrix

   if (matr.GetNrows()!=fNvar || matr.GetNcols()!=fNvar){
      Warning("GetCovariance","provided matrix is of the wrong size, it will be resized");
      matr.ResizeTo(fNvar, fNvar);
   }
   matr=fCovariance;
}

//_______________________________________________________________________
void TRobustEstimator::GetCorrelation(TMatrixDSym &matr)
{
   //returns the correlation matrix
   
   if (matr.GetNrows()!=fNvar || matr.GetNcols()!=fNvar) {
      Warning("GetCorrelation","provided matrix is of the wrong size, it will be resized");
      matr.ResizeTo(fNvar, fNvar);
   }
   matr=fCorrelation;
}

//____________________________________________________________________
const TVectorD* TRobustEstimator::GetHyperplane() const
{
   //if the points are on a hyperplane, returns this hyperplane

   if (fExact==0) {
      Error("GetHyperplane","the data doesn't lie on a hyperplane!\n");
      return 0;
   } else {
      return &fHyperplane;
   }
}

//______________________________________________________________________
void TRobustEstimator::GetHyperplane(TVectorD &vec)
{
   //if the points are on a hyperplane, returns this hyperplane

   if (fExact==0){
      Error("GetHyperplane","the data doesn't lie on a hyperplane!\n");
      return;
   }
   if (vec.GetNoElements()!=fNvar) {
      Warning("GetHyperPlane","provided vector is of the wrong size, it will be resized");
      vec.ResizeTo(fNvar);
   }
   vec=fHyperplane;
}

//________________________________________________________________________
void TRobustEstimator::GetMean(TVectorD &means)
{
   //return the estimate of the mean

   if (means.GetNoElements()!=fNvar) {
      Warning("GetMean","provided vector is of the wrong size, it will be resized");
      means.ResizeTo(fNvar);
   }
   means=fMean;
}

//_________________________________________________________________________
void TRobustEstimator::GetRDistances(TVectorD &rdist)
{
   //returns the robust distances (helps to find outliers)

   if (rdist.GetNoElements()!=fN) {
      Warning("GetRDistances","provided vector is of the wrong size, it will be resized");
      rdist.ResizeTo(fN);
   }
   rdist=fRd;
}

//__________________________________________________________________________
Int_t TRobustEstimator::GetNOut()
{
   //returns the number of outliers

   return fOut.GetSize();
}

//_________________________________________________________________________
void TRobustEstimator::AddToSscp(TMatrixD &sscp, TVectorD &vec)
{
  //update the sscp matrix with vector vec

   Int_t i, j;
   for (j=1; j<fNvar+1; j++) {
      sscp(0, j) +=vec(j-1);
      sscp(j, 0) = sscp(0, j);
   }
   for (i=1; i<fNvar+1; i++) {
      for (j=1; j<fNvar+1; j++) {
         sscp(i, j) += vec(i-1)*vec(j-1);
      }
   }
}

//__________________________________________________________________________
void TRobustEstimator::ClearSscp(TMatrixD &sscp)
{
   //clear the sscp matrix, used for covariance and mean calculation

   for (Int_t i=0; i<fNvar+1; i++) {
      for (Int_t j=0; j<fNvar+1; j++) {
         sscp(i, j)=0;
      }
   }
}

//_______________________________________________________________
void TRobustEstimator::Classic()
{
  //called when h=n. Returns classic covariance matrix
  //and mean
   TMatrixD sscp(fNvar+1, fNvar+1);
   TVectorD temp(fNvar);
   ClearSscp(sscp);
   for (Int_t i=0; i<fN; i++) {
      for (Int_t j=0; j<fNvar; j++)
         temp(j)=fData(i, j);
      AddToSscp(sscp, temp);
   }
   Covar(sscp, fMean, fCovariance, fSd, fN);
   Correl();
   
}

//____________________________________________________________________
void TRobustEstimator::Covar(TMatrixD &sscp, TVectorD &m, TMatrixDSym &cov, TVectorD &sd, Int_t nvec)
{
   //calculates mean and covariance

   Int_t i, j;
   Double_t f;
   for (i=0; i<fNvar; i++) {
      m(i)=sscp(0, i+1);
      sd[i]=sscp(i+1, i+1);
      f=(sd[i]-m(i)*m(i)/nvec)/(nvec-1);
      if (f>1e-14) sd[i]=TMath::Sqrt(f);
      else sd[i]=0;
      m(i)/=nvec;
   }
   for (i=0; i<fNvar; i++) {
      for (j=0; j<fNvar; j++) {
         cov(i, j)=sscp(i+1, j+1)-nvec*m(i)*m(j);
      cov(i, j)/=nvec-1;
      }
   }
}

//____________________________________________________________________
void TRobustEstimator::Correl()
{
  //transforms covariance matrix into correlation matrix

   Int_t i, j;
   Double_t *sd=new Double_t[fNvar];
   for(j=0; j<fNvar; j++)
      sd[j]=1./TMath::Sqrt(fCovariance(j, j));
   for(i=0; i<fNvar; i++) {
      for (j=0; j<fNvar; j++) {
         if (i==j)
            fCorrelation(i, j)=1.;
         else
            fCorrelation(i, j)=fCovariance(i, j)*sd[i]*sd[j];
      }
   }
   delete [] sd;
}

//____________________________________________________________________
void TRobustEstimator::CreateSubset(Int_t ntotal, Int_t htotal, Int_t p, Int_t *index, TMatrixD &data, TMatrixD &sscp, Double_t *ndist)
{
  //creates a subset of htotal elements from ntotal elements
  //first, p+1 elements are drawn randomly(without repetitions)
  //if their covariance matrix is singular, more elements are
  //added one by one, until their covariance matrix becomes regular
  //or it becomes clear that htotal observations lie on a hyperplane
  //If covariance matrix determinant!=0, distances of all ntotal elements
  //are calculated, using formula d_i=Sqrt((x_i-M)*S_inv*(x_i-M)), where
  //M is mean and S_inv is the inverse of the covariance matrix
  //htotal points with smallest distances are included in the returned subset.

   Double_t kEps = 1e-14;
   Int_t i, j;
   Bool_t repeat=kFALSE;
   Int_t nindex=0;
   Int_t num;
   for(i=0; i<ntotal; i++)
      index[i]=ntotal+1;
   
   for (i=0; i<p+1; i++) {
      num=Int_t(gRandom->Uniform(0, 1)*(ntotal-1));
      if (i>0){
         for(j=0; j<=i-1; j++) {
            if(index[j]==num)
            repeat=kTRUE;
         }
      }
      if(repeat==kTRUE) {
         i--;
         repeat=kFALSE;
      } else {
         index[i]=num;
         nindex++;
      }
   }
   
   ClearSscp(sscp);
   
   TVectorD vec(fNvar);
   Double_t det;
   for (i=0; i<p+1; i++) {
      for (j=0; j<fNvar; j++) {
         vec[j]=data[index[i]][j];
         
      }
      AddToSscp(sscp, vec);
   }
   
   Covar(sscp, fMean, fCovariance, fSd, p+1);
   det=fCovariance.Determinant();
   while((det<kEps)&&(nindex < htotal)) {
    //if covariance matrix is singular,another vector is added until
    //the matrix becomes regular or it becomes clear that all
    //vectors of the group lie on a hyperplane
      repeat=kFALSE;
      do{
         num=Int_t(gRandom->Uniform(0,1)*(ntotal-1));
         repeat=kFALSE;
         for(i=0; i<nindex; i++) {
            if(index[i]==num) {
               repeat=kTRUE;
               break;
            }
         }
      }while(repeat==kTRUE);
      
      index[nindex]=num;
      nindex++;
      //check if covariance matrix is singular
      for(j=0; j<fNvar; j++)
         vec[j]=data[index[nindex-1]][j];
      AddToSscp(sscp, vec);
      Covar(sscp, fMean, fCovariance, fSd, nindex);
      det=fCovariance.Determinant();
   }
   
   if(nindex!=htotal) {
      TDecompChol chol(fCovariance);
      fInvcovariance = chol.Invert();
      
      TVectorD temp(fNvar);
      for(j=0; j<ntotal; j++) {
         ndist[j]=0;
         for(i=0; i<fNvar; i++)
            temp[i]=data[j][i] - fMean(i);
         temp*=fInvcovariance;
         for(i=0; i<fNvar; i++)
            ndist[j]+=(data[j][i]-fMean(i))*temp[i];
      }
      KOrdStat(ntotal, ndist, htotal-1,index);
   }
   
}

//_____________________________________________________________________________
void TRobustEstimator::CreateOrtSubset(TMatrixD &dat,Int_t *index, Int_t hmerged, Int_t nmerged, TMatrixD &sscp, Double_t *ndist)
{
  //creates a subset of hmerged vectors with smallest orthogonal distances to the hyperplane
  //hyp[1]*(x1-mean[1])+...+hyp[nvar]*(xnvar-mean[nvar])=0
  //This function is called in case when less than fH samples lie on a hyperplane.

   Int_t i, j;
      TVectorD vec(fNvar);
   for (i=0; i<nmerged; i++) {
      ndist[i]=0;
      for(j=0; j<fNvar; j++) {
         ndist[i]+=fHyperplane[j]*(dat[i][j]-fMean[j]);
         ndist[i]=TMath::Abs(ndist[i]);
      }
   }
   KOrdStat(nmerged, ndist, hmerged-1, index);
   ClearSscp(sscp);
   for (i=0; i<hmerged; i++) {
      for(j=0; j<fNvar; j++)
         vec[j]=dat[index[i]][j];
      AddToSscp(sscp, vec);
   }
   Covar(sscp, fMean, fCovariance, fSd, hmerged);
}

//__________________________________________________________________________
Double_t TRobustEstimator::CStep(Int_t ntotal, Int_t htotal, Int_t *index, TMatrixD &data, TMatrixD &sscp, Double_t *ndist)
{
  //from the input htotal-subset constructs another htotal subset with lower determinant
  //
  //As proven by Peter J.Rousseeuw and Katrien Van Driessen, if distances for all elements
  //are calculated, using the formula:d_i=Sqrt((x_i-M)*S_inv*(x_i-M)), where M is the mean
  //of the input htotal-subset, and S_inv - the inverse of its covariance matrix, then
  //htotal elements with smallest distances will have covariance matrix with determinant
  //less or equal to the determinant of the input subset covariance matrix.
  //
  //determinant for this htotal-subset with smallest distances is returned

   Int_t i, j;
   TVectorD vec(fNvar);
   Double_t det;
   
   TDecompChol chol(fCovariance);
   fInvcovariance = chol.Invert();

   TVectorD temp(fNvar);
   for(j=0; j<ntotal; j++) {
      ndist[j]=0;
      for(i=0; i<fNvar; i++)
         temp[i]=data[j][i]-fMean[i];
      temp*=fInvcovariance;
      for(i=0; i<fNvar; i++)
         ndist[j]+=(data[j][i]-fMean[i])*temp[i];
   }
   
   //taking h smallest
   KOrdStat(ntotal, ndist, htotal-1, index);
   //writing their mean and covariance
   ClearSscp(sscp);
   for (i=0; i<htotal; i++) {
      for (j=0; j<fNvar; j++)
         temp[j]=data[index[i]][j];
      AddToSscp(sscp, temp);
   }
   Covar(sscp, fMean, fCovariance, fSd, htotal);
   det = fCovariance.Determinant();
   return det;
}

//_______________________________________________________________
Int_t TRobustEstimator::Exact(Double_t *ndist)
{
  //for the exact fit situaions
  //returns number of observations on the hyperplane

   Int_t i, j;
   
   TMatrixDSymEigen eigen(fCovariance);
   TVectorD eigenValues=eigen.GetEigenValues();
   TMatrixD eigenMatrix=eigen.GetEigenVectors();
   
   for (j=0; j<fNvar; j++) {
      fHyperplane[j]=eigenMatrix(j,fNvar-1);
   }
   //calculate and return how many observations lie on the hyperplane
   for (i=0; i<fN; i++) {
      ndist[i]=0;
      for(j=0; j<fNvar; j++) {
         ndist[i]+=fHyperplane[j]*(fData[i][j]-fMean[j]);
         ndist[i]=TMath::Abs(ndist[i]);
      }
   }
   Int_t nhyp=0;
   
   for (i=0; i<fN; i++) {
      if(ndist[i] < 1e-14) nhyp++;
   }
   return nhyp;
   
}

//____________________________________________________________________________
Int_t TRobustEstimator::Exact2(TMatrixD &mstockbig, TMatrixD &cstockbig, TMatrixD &hyperplane,
                             Double_t *deti, Int_t nbest, Int_t kgroup,
                             TMatrixD &sscp, Double_t *ndist)
{
  //This function is called if determinant of the covariance matrix of a subset=0.
  //
  //If there are more then fH vectors on a hyperplane,
  //returns this hyperplane and stops
  //else stores the hyperplane coordinates in hyperplane matrix

   Int_t i, j;
   
   TVectorD vec(fNvar);
   Int_t maxind = TMath::LocMax(nbest, deti);
   Int_t nh=Exact(ndist);
   //now nh is the number of observation on the hyperplane
   //ndist stores distances of observation from this hyperplane
   if(nh>=fH) {
      ClearSscp(sscp);
      for (i=0; i<fN; i++) {
         if(ndist[i]<1e-14) {
            for (j=0; j<fNvar; j++)
               vec[j]=fData[i][j];
            AddToSscp(sscp, vec);
         }
      }
      Covar(sscp, fMean, fCovariance, fSd, nh);
      
      fExact=nh;
      return nbest+1;
      
   } else {
      //if less than fH observations lie on a hyperplane,
      //mean and covariance matrix are stored in mstockbig
      //and cstockbig in place of the previous maximum determinant
      //mean and covariance
      for(i=0; i<fNvar; i++) {
         mstockbig(nbest*kgroup+maxind,i)=fMean(i);
         hyperplane(nbest*kgroup+maxind,i)=fHyperplane(i);
         for(j=0; j<fNvar; j++) {
            cstockbig(i,nbest*kgroup*fNvar+maxind*fNvar+j)=fCovariance(i,j);
         }   
      }      
      return maxind;
   }
}


//_____________________________________________________________________________
Int_t TRobustEstimator::Partition(Int_t nmini, Int_t *indsubdat)
{
  //divides the elements into approximately equal subgroups
  //number of elements in each subgroup is stored in indsubdat
  //number of subgroups is returned

   Int_t nsub;
   if ((fN>=2*nmini) && (fN<=(3*nmini-1))) {
      if (fN%2==1){
         indsubdat[0]=Int_t(fN*0.5);
      indsubdat[1]=Int_t(fN*0.5)+1;
      } else
         indsubdat[0]=indsubdat[1]=Int_t(fN/2);
      nsub=2;
   }
   else{
      if((fN>=3*nmini) && (fN<(4*nmini -1))) {
         if(fN%3==0){
            indsubdat[0]=indsubdat[1]=indsubdat[2]=Int_t(fN/3);
         } else {
            indsubdat[0]=Int_t(fN/3);
            indsubdat[1]=Int_t(fN/3)+1;
            if (fN%3==1) indsubdat[2]=Int_t(fN/3);
            else indsubdat[2]=Int_t(fN/3)+1;
         }
         nsub=3;
      }
      else{
         if((fN>=4*nmini)&&(fN<=(5*nmini-1))){
            if (fN%4==0) indsubdat[0]=indsubdat[1]=indsubdat[2]=indsubdat[3]=Int_t(fN/4);
            else {
               indsubdat[0]=Int_t(fN/4);
               indsubdat[1]=Int_t(fN/4)+1;
               if(fN%4==1) indsubdat[2]=indsubdat[3]=Int_t(fN/4);
               if(fN%4==2) {
                  indsubdat[2]=Int_t(fN/4)+1;
                  indsubdat[3]=Int_t(fN/4);
               }
               if(fN%4==3) indsubdat[2]=indsubdat[3]=Int_t(fN/4)+1;
            }
            nsub=4;
         } else {
            for(Int_t i=0; i<5; i++)
               indsubdat[i]=nmini;
            nsub=5;
         }
      }
   }
   return nsub;
}

//___________________________________________________________________________
Int_t TRobustEstimator::RDist(TMatrixD &sscp)
{
  //Calculates robust distances.Then the samples with robust distances
  //greater than a cutoff value (0.975 quantile of chi2 distribution with
  //fNvar degrees of freedom, multiplied by a correction factor), are given
  //weiht=0, and new, reweighted estimates of location and scatter are calculated
  //The function returns the number of outliers.

   Int_t i, j;
   Int_t nout=0;
   
   TVectorD temp(fNvar);
   TDecompChol chol(fCovariance);
   fInvcovariance = chol.Invert();
   

   for (i=0; i<fN; i++) {
      fRd[i]=0;
      for(j=0; j<fNvar; j++) {
         temp[j]=fData[i][j]-fMean[j];
      }
      temp*=fInvcovariance;
      for(j=0; j<fNvar; j++) {
         fRd[i]+=(fData[i][j]-fMean[j])*temp[j];
      }
   }
   
   Double_t med;
   Double_t chi = kChiMedian[fNvar-1];
   
   med=TMath::Median(fN, fRd.GetMatrixArray());
   med/=chi;
   fCovariance*=med;
   TDecompChol chol2(fCovariance);
   fInvcovariance = chol2.Invert();
   
   for (i=0; i<fN; i++) {
      fRd[i]=0;
      for(j=0; j<fNvar; j++) {
         temp[j]=fData[i][j]-fMean[j];
   }
      
      temp*=fInvcovariance;
      for(j=0; j<fNvar; j++) {
         fRd[i]+=(fData[i][j]-fMean[j])*temp[j];
      }
   }
   
   Double_t cutoff = kChiQuant[fNvar-1];
   
   ClearSscp(sscp);
   for(i=0; i<fN; i++) {
      if (fRd[i]<=cutoff) {
         for(j=0; j<fNvar; j++)
            temp[j]=fData[i][j];
         AddToSscp(sscp,temp);
      } else {
         nout++;
      }
   }
   
   Covar(sscp, fMean, fCovariance, fSd, fN-nout);
   return nout;
}

//____________________________________________________________________________
void TRobustEstimator::RDraw(Int_t *subdat, Int_t ngroup, Int_t *indsubdat)
{
  //Draws ngroup nonoverlapping subdatasets out of a dataset of size n
  //such that the selected case numbers are uniformly distributed from 1 to n

   Int_t jndex = 0;
   Int_t nrand;
   Int_t i, k, m, j;
   for (k=1; k<=ngroup; k++) {
      for (m=1; m<=indsubdat[k-1]; m++) {
         nrand = Int_t(gRandom->Uniform(0, 1) * double(fN-jndex))+1;
         //printf("nrand = %d - jndex %d\n",nrand,jndex);
         jndex++;
         if (jndex==1) {
            subdat[0]=nrand-1;  // in case nrand is equal to fN
         } else {
            subdat[jndex-1]=nrand+jndex-2;
            for (i=1; i<=jndex-1; i++) {
               if(subdat[i-1] > nrand+i-2) {
                  for(j=jndex; j>=i+1; j--) {
                     subdat[j-1]=subdat[j-2];
                  }
                  //printf("subdata[] i = %d - nrand %d\n",i,nrand);
                  subdat[i-1]=nrand+i-2;
                  break;  //breaking the loop for(i=1...
               }
            }
         }
      }
   }
}

//_____________________________________________________________________________
Double_t TRobustEstimator::KOrdStat(Int_t ntotal, Double_t *a, Int_t k, Int_t *work){
  //because I need an Int_t work array
   Bool_t isAllocated = kFALSE;
   const Int_t kWorkMax=100;
   Int_t i, ir, j, l, mid;
   Int_t arr;
   Int_t *ind;
   Int_t workLocal[kWorkMax];
   Int_t temp;


   if (work) {
      ind = work;
   } else {
      ind = workLocal;
      if (ntotal > kWorkMax) {
         isAllocated = kTRUE;
         ind = new Int_t[ntotal];
      }
   }

   for (Int_t ii=0; ii<ntotal; ii++) {
      ind[ii]=ii;
   }
   Int_t rk = k;
   l=0;
   ir = ntotal-1;
   for(;;) {
      if (ir<=l+1) { //active partition contains 1 or 2 elements
         if (ir == l+1 && a[ind[ir]]<a[ind[l]])
            {temp = ind[l]; ind[l]=ind[ir]; ind[ir]=temp;}
         Double_t tmp = a[ind[rk]];
         if (isAllocated)
            delete [] ind;
         return tmp;
      } else {
         mid = (l+ir) >> 1; //choose median of left, center and right
         {temp = ind[mid]; ind[mid]=ind[l+1]; ind[l+1]=temp;}//elements as partitioning element arr.
         if (a[ind[l]]>a[ind[ir]])  //also rearrange so that a[l]<=a[l+1]
            {temp = ind[l]; ind[l]=ind[ir]; ind[ir]=temp;}

         if (a[ind[l+1]]>a[ind[ir]])
            {temp=ind[l+1]; ind[l+1]=ind[ir]; ind[ir]=temp;}

         if (a[ind[l]]>a[ind[l+1]])
                {temp = ind[l]; ind[l]=ind[l+1]; ind[l+1]=temp;}

         i=l+1;        //initialize pointers for partitioning
         j=ir;
         arr = ind[l+1];
         for (;;) {
            do i++; while (a[ind[i]]<a[arr]);
            do j--; while (a[ind[j]]>a[arr]);
            if (j<i) break;  //pointers crossed, partitioning complete
               {temp=ind[i]; ind[i]=ind[j]; ind[j]=temp;}
         }
         ind[l+1]=ind[j];
         ind[j]=arr;
         if (j>=rk) ir = j-1; //keep active the partition that
         if (j<=rk) l=i;      //contains the k_th element
      }
   }
}
