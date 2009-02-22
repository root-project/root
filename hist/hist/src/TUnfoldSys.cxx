// Author: Stefan Schmitt
// DESY, 23/01/09

// Version 13, support for systematic errors

////////////////////////////////////////////////////////////////////////
//
// TUnfoldSys adds error propagation of systematic errors to TUnfold
//
// Example:
//    TH2D *histA,*histAsys1,*histAsys2;
//    TH1D *data;
//  assume the above histograms are filled:
//      histA: migration matrix from generator (x-axis) to detector (y-axis)
//           the errors of histA are the uncorrelated systematic errors
//      histAsys1: alternative migration matrix, when systematic #1 is applied
//      histAsys1: alternative migration matrix, when systematic #2 is applied
//
//  set up the unfolding:
//
//    TUnfoldSys unfold(histA,TUnfold::kHistMapOutputVert);
//    unfold.SetInput(input);
//    unfold.AddSysError(histAsys1,"syserror1",TUnfold::kHistMapOutputVert,
//                       TUnfoldSys::kSysErrModeMatrix);
//    unfold.AddSysError(histAsys2,"syserror2",TUnfold::kHistMapOutputVert,
//                       TUnfoldSys::kSysErrModeMatrix);
//
//  it is possible to specify the systematic errors as
//      TUnfoldSys::kSysErrModeMatrix:
//          alternative migration matrix
//      TUnfoldSys::kSysErrModeAbsolute:
//          matrix of absolute shifts wrt the original matrix
//      TUnfoldSys::kSysErrModeMatrix:
//          matrix of relative errors wrt the original matrix
//  
//  run the unfolding: see description of class TUnfold
//    unfold.ScanLcurve( ...)
//
// retrieve the output
//    unfold.GetOutput(output);     unfolding output with statistical errors
//    unfold.GetEmatrix(stat_error);             error matrix for stat.errors
//    unfold.GetEmatrixSysUncorr(uncorr_sys);    error matrix for uncorr.syst
//    unfold.GetEmatrixSysSource(sys1,"syserror1"); error matrix from source 1
//    unfold.GetEmatrixSysSource(sys2,"syserror2"); error matrix from source 2
//    unfold.GetEmatrixSysTotal(sys_total); error matrix for total sys.errors
//                                          (= uncorr_sys+sys1+sys2)
//    unfold.GetEmatrixTotal(err_total); error matrix with all errros added
//                                       (= sys_total+stat_error)
//    Double_t chi_2=GetChi2Sys();  chi**2 including systematic errors
//                                  compare to GetChi2A(), stat errors only
//
////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <TUnfoldSys.h>
#include <TMap.h>
#include <TMath.h>
#include <TObjString.h>
#include <RVersion.h>

ClassImp(TUnfoldSys)

TUnfoldSys::TUnfoldSys(void) {
   // set all pointers to zero
   InitTUnfoldSys();
}

TUnfoldSys::TUnfoldSys(TH2 const *hist_A, EHistMap histmap, ERegMode regmode)
   : TUnfold(hist_A,histmap,regmode) {
   // arguments:
   //    hist_A:  matrix that describes the migrations
   //    histmap: mapping of the histogram axes to the unfolding output 
   //    regmode: global regularisation mode
   // data members initialized to something different from zero:
   //    fDA2, fDAcol

   // save the errors on hist_A to the matrices fDA2 and fDAcol
   // and the content of the underflow/overflow rows
   InitTUnfoldSys();
   fAoutside = new TMatrixD(GetNx(),2);
   fDAcol = new TMatrixD(GetNx(),1);

   Int_t nmax=GetNx()*GetNy();
   Int_t *rowDA2 = new Int_t[nmax];
   Int_t *colDA2 = new Int_t[nmax];
   Double_t *dataDA2 = new Double_t[nmax];

   Int_t da2col_nonzero=0;
   Int_t da2_nonzero=0;
   for (Int_t ix = 0; ix < GetNx(); ix++) {
      Int_t ibinx = fXToHist[ix];
      for (Int_t ibiny = 0; ibiny <= GetNy()+1; ibiny++) {
         Double_t z,dz;
         if (histmap == kHistMapOutputHoriz) {
            z = hist_A->GetBinContent(ibinx, ibiny);
            dz = hist_A->GetBinError(ibinx, ibiny);
         } else {
            z = hist_A->GetBinContent(ibiny, ibinx);
            dz = hist_A->GetBinError(ibiny, ibinx);
         }
         if(ibiny==0) {
            // underflow bins
            (*fAoutside)(ix,0)=z;
            if(dz>0.0) {
               (*fDAcol)(ix,0) += dz*dz;
               da2col_nonzero++;
            }
         } else if(ibiny==GetNy()+1) {
            // overflow bins
            (*fAoutside)(ix,1)=z;
            if(dz>0.0) {
               (*fDAcol)(ix,0) += dz*dz;
               da2col_nonzero++;
            }
         } else {
            // normal bins
            Double_t sum= fSumOverY[ix];
            Double_t error =
               (sum-z)/sum/sum * dz;
            rowDA2[da2_nonzero]=ibiny-1;
            colDA2[da2_nonzero] = ix;
            dataDA2[da2_nonzero] = error*error;
            if(dataDA2[da2_nonzero]>0.0) da2_nonzero++;
         }
      }
   }
   if(da2_nonzero) {
      fDA2 = new TMatrixDSparse(GetNy(),GetNx());
      fDA2->SetMatrixArray(da2_nonzero,rowDA2,colDA2,dataDA2);
   }
   if(!da2col_nonzero) {
      delete fDAcol;
      fDAcol=0;
   } else {
      // normalize to the number of entries and take square root
      for (Int_t ix = 0; ix < GetNx(); ix++) {
         (*fDAcol)(ix,0) = TMath::Sqrt((*fDAcol)(ix,0))/fSumOverY[ix];
      }
   }
   delete [] rowDA2;
   delete [] colDA2;
   delete [] dataDA2;      
}

void TUnfoldSys::AddSysError(TH2 const *sysError,char const *name,
                             EHistMap histmap,ESysErrMode mode) {
   // add a correlated error source
   //    sysError: alternative matrix or matrix of absolute/relative shifts
   //    name: name of the error source
   //    histmap: mapping of the histogram axes to the unfolding output
   //    mode: format of the error source

   if(fSysIn->FindObject(name)) {
      std::cout<<"UnfoldSys::AddSysError \""<<name
               <<"\" has already been added, ignoring\n";
   } else {
      // a copy of fA is made. It can be accessed inside the loop
      // without having to take care that the sparse structure of *fA
      // may be accidentally destroyed by asking for an element which is zero.
      TMatrixD aCopy(*fA);

      Int_t nmax= GetNx()*GetNy();
      Double_t *data=new Double_t[nmax];
      Int_t *cols=new Int_t[nmax];
      Int_t *rows=new Int_t[nmax];
      nmax=0;
      for (Int_t ix = 0; ix < GetNx(); ix++) {
         Int_t ibinx = fXToHist[ix];
         Double_t sum=0.0;
         for(Int_t loop=0;loop<2;loop++) {
            for (Int_t ibiny = 0; ibiny <= GetNy()+1; ibiny++) {
               Double_t z;
               // get bin content, depending on histmap
               if (histmap == kHistMapOutputHoriz) {
                  z = sysError->GetBinContent(ibinx, ibiny);
               } else {
                  z = sysError->GetBinContent(ibiny, ibinx);
               }
               // correct to absolute numbers
               if(mode!=kSysErrModeMatrix) {
                  Double_t z0;
                  if((ibiny>0)&&(ibiny<=GetNy())) {
                     z0=aCopy(ibiny-1,ix)*fSumOverY[ix];
                  } else if(ibiny==0) {
                     z0=(*fAoutside)(ix,0);
                  } else {
                     z0=(*fAoutside)(ix,1);
                  }
                  if(mode==kSysErrModeShift) {
                     z += z0;
                  } else if(mode==kSysErrModeRelative) {
                     z = z0*(1.+z);
                  }
               }
               if(loop==0) {
                  // sum up all entries, including overflow bins
                  sum += z;
               } else {
                  if((ibiny>0)&&(ibiny<=GetNy())) {
                     // save normalized matrix of shifts,
                     // excluding overflow bins
                     rows[nmax]=ibiny-1;
                     cols[nmax]=ix;
                     if(sum>0.0) {
                        data[nmax]=z/sum-aCopy(ibiny-1,ix);
                     } else {
                        data[nmax]=0.0;
                     }
                     if(data[nmax] != 0.0) nmax++;
                  }
               }
            }
         }
      }
      std::cout<<nmax<<" "<<GetNy()*GetNx()<<"\n";

      TMatrixDSparse *dsys=new TMatrixDSparse(GetNy(),GetNx());
      if(nmax==0) {
         std::cout<<"UnfoldSys::AddSysError source \""<<name
                  <<"\" has no influence on the result.\n";
      }
      dsys->SetMatrixArray(nmax,rows,cols,data);
      delete[] data;
      delete[] rows;
      delete[] cols;
      fSysIn->Add(new TObjString(name),dsys);
   }
}

void TUnfoldSys::InitTUnfoldSys(void) {
   // initialize pointers and TMaps

   // input
   fDA2 = 0;
   fDAcol = 0;
   fAoutside = 0;
   fSysIn = new TMap();
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,20,00)
   fSysIn->SetOwnerKeyValue();
#else
   fSysIn->SetOwner();
#endif
   // results
   fVYAx = 0;
   fESparse = 0;
   fEAtV = 0;
   fErrorUncorrX = 0;
   fErrorUncorrAx = 0;
   fAE = 0;
   fAEAtV_one = 0;
   fErrorCorrX = new TMap();
   fErrorCorrAx = new TMap();
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,20,00)
   fErrorCorrX->SetOwnerKeyValue();
   fErrorCorrAx->SetOwnerKeyValue();
#else
   fErrorCorrX->SetOwner();
   fErrorCorrAx->SetOwner();
#endif
}

TUnfoldSys::~TUnfoldSys(void) {
   // delete all data members
   DeleteMatrix(&fDA2);
   delete fSysIn;
   ClearResults();
   delete fErrorCorrX;
   delete fErrorCorrAx;
}

void TUnfoldSys::ClearResults(void) {
   // clear all data members which depend on the unfolding results
   TUnfold::ClearResults();
   DeleteMatrix(&fVYAx);
   DeleteMatrix(&fESparse);
   DeleteMatrix(&fEAtV);
   DeleteMatrix(&fErrorUncorrX);
   DeleteMatrix(&fErrorUncorrAx);
   DeleteMatrix(&fAE);
   DeleteMatrix(&fAEAtV_one);
   fErrorCorrX->Clear();
   fErrorCorrAx->Clear();
}

void TUnfoldSys::PrepareSysError(void) {
   // calculations required for syst.error
   // data members modified
   //    fVYAx, fESparse, fEAtV, fAE, fAEAtV_one,
   //    fErrorUncorrX, fErrorUncorrAx, fErrorCorrX, fErrorCorrAx
   if(!fVYAx) {
      TMatrixD yAx(*fY,TMatrixD::kMinus,*fAx);
      fVYAx=MultiplyMSparseM(*fV,yAx);
   }
   if(!fESparse) {
      fESparse=new TMatrixDSparse(*fE);
   }
   if(!fEAtV) {
      fEAtV=MultiplyMSparseMSparse(*fESparse,*fAtV);
   }
   if(!fAE) {
      fAE = MultiplyMSparseMSparse(*fA,*fESparse);
   }
   if(!fAEAtV_one) {
      fAEAtV_one=MultiplyMSparseMSparse(*fA,*fEAtV);
      Int_t *rows_cols=new Int_t[GetNy()];
      Double_t *data=new Double_t[GetNy()];
      for(Int_t i=0;i<GetNy();i++) {
         rows_cols[i]=i;
         data[i]=1.0;
      }
      TMatrixDSparse one(GetNy(),GetNy());
      one.SetMatrixArray(GetNy(),rows_cols, rows_cols,data);
      AddMSparse(*fAEAtV_one,-1.,one);
      delete [] data;
      delete [] rows_cols;
   }
   if(!fErrorUncorrX) {
      fErrorUncorrX=PrepareUncorrEmat(fESparse,fEAtV);
   }
   if(!fErrorUncorrAx) {
      fErrorUncorrAx=PrepareUncorrEmat(fAE,fAEAtV_one);
   }
   TMapIter sysErrIn(fSysIn);
   TObjString const *key;
   for(key=(TObjString const *)sysErrIn.Next();key;
       key=(TObjString const *)sysErrIn.Next()) {
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,20,00)
      TMatrixDSparse const *dsys=
         (TMatrixDSparse const *)((TPair const *)*sysErrIn)->Value();
#else
      TMatrixDSparse const *dsys=
         (TMatrixDSparse const *)(fSysIn->GetValue(key->GetString()));
#endif
      TPair const *named_emat=(TPair const *)
         fErrorCorrX->FindObject(key->GetString());
      if(!named_emat) {
         TMatrixD *emat=PrepareCorrEmat(fESparse,fEAtV,dsys);
         fErrorCorrX->Add(new TObjString(*key),emat);
      }
      named_emat=(TPair const *)fErrorCorrAx->FindObject(key->GetString());
      if(!named_emat) {
         TMatrixD *emat=PrepareCorrEmat(fAE,fAEAtV_one,dsys);
         fErrorCorrAx->Add(new TObjString(*key),emat);
      }
   }
   
}
  
void TUnfoldSys::GetEmatrixSysUncorr(TH2 *ematrix,Int_t const *binMap,Bool_t clearEmat) {
   // get output error contribution from statistical fluctuations in A
   //   ematrix: output error matrix histogram
   //   binMap: see method GetEmatrix()
   //   clearEmat: set kTRUE to clear the histogram prior to adding the errors
   // data members modified:
   //   fVYAx, fESparse, fEAtV, fErrorAStat
   PrepareSysError();
   ErrorMatrixToHist(ematrix,fErrorUncorrX,binMap,clearEmat);
}

TMatrixD *TUnfoldSys::PrepareUncorrEmat(TMatrixDSparse const *m1,TMatrixDSparse const *m2) {
   // prepare matrix of uncorrelated systematic errors
   //   m1,m2 : coefficients for propagating the errors
   // the result depends on the errors
   //   fDA2, fDA2col

   TMatrixD *r=new TMatrixD(m1->GetNrows(),m1->GetNrows());

   // z is the output quantity, its derivative wrt a_ij is
   //    dz_m / da_ij =
   //
   //       (*m1)(m,j) * (*fVYAx)(i) - (*m2)(m,i) * (*fX)(j)
   //
   // The corresponding error matrix (*e)(m,n) is obtained by summing over i,j:
   //
   // sum{i,j}   {
   //   ((*m1)(m,j) * (*fVYAx)(i) - (*m2)(m,i) * (*fX)(j))*
   //   ((*m1)(n,j) * (*fVYAx)(i) - (*m2)(n,i) * (*fX)(j))*(*fDA2)(i,j)  }
   //
   // The sum is resolved into simpler matrix operations, such that loops over
   // 4 dimensions {i,j,m,n} are avoided:
   //
   // sum_j (*m1)(m,j)*(*m1)(n,j) * sum_i (*fDA2)(i,j)*(*fVYAx)(i)*(*fVYAx)(i)
   //+sum_i (*m2)(m,i)*(*m2)(n,i) * sum_j (*fDA2)(i,j)*(*fX)(j)*(*fX)(j)
   //-sum_j (*m1)(m,j)*(*fX)(j) * sum_i (*m2)(n,i)*(*fDA2)(i,j)*(*fVYAx)(i)
   //-sum_j (*m1)(n,j)*(*fX)(j) * sum_i (*m2)(m,i)*(*fDA2)(i,j)*(*fVYAx)(i)
   //
   // in addition, the error depends on entries of the original matrix
   // which only contribute to the column sum s_j = fSumOverY[j]
   // and a_ij = A_ij / s_j, so da_ij / ds_j = - a_ij / s_j
   //
   //   dz_m / de_j = sum_i { dz_m / da_ij * da_ij / de_j)
   //               = - sum_i { dz_m /da_ij * a_ij / s_j }
   //      =
   // - sum_i {((*m1)(m,j) * (*fVYAx)(i) - (*m2)(m,i) * (*fX)(j))*a_ij} /s_j

   if(fDA2) {
      const Int_t *DA2_rows=fDA2->GetRowIndexArray();
      const Int_t *DA2_cols=fDA2->GetColIndexArray();
      const Double_t *DA2_data=fDA2->GetMatrixArray();

      const Int_t *fVYAx_rows=fVYAx->GetRowIndexArray();
      const Double_t *fVYAx_data=fVYAx->GetMatrixArray();

      // sum_i { (*fDA2)(i,j)*(*fVYAx)(i)*(*fVYAx)(i) }
      TMatrixD DA2t_VYAx2(fDA2->GetNcols(),1);
      for(Int_t i=0;i<fVYAx->GetNrows();i++) {
         if(fVYAx_rows[i]==fVYAx_rows[i+1]) continue;
         for(Int_t index=DA2_rows[i];index<DA2_rows[i+1];index++) {
            Int_t j=DA2_cols[index];
            Double_t fVYAx_j=fVYAx_data[fVYAx_rows[i]];
            DA2t_VYAx2(j,0) += DA2_data[index]*fVYAx_j*fVYAx_j;
         }
      }
      // sum_j { (*fDA2)(i,j)*(*fX)(j)*(*fX)(j) }
      TMatrixD DA2_x2(fDA2->GetNrows(),1);
      for(Int_t i=0;i<fDA2->GetNrows();i++) {
         for(Int_t index=DA2_rows[i];index<DA2_rows[i+1];index++) {
            Int_t j=DA2_cols[index];
            DA2_x2(i,0) += DA2_data[index]*(*fX)(j,0)*(*fX)(j,0);
         }
      }
      // (*fDA2)(i,j)*(*fVYAx)(i)
      // same row/col structure as fDA2
      TMatrixDSparse mDA2_VYAx(*fDA2);
      Double_t *mDA2_VYAx_data=mDA2_VYAx.GetMatrixArray();
      for(Int_t i=0;i<fDA2->GetNrows();i++) {
         for(Int_t index=DA2_rows[i];index<DA2_rows[i+1];index++) {
            if(fVYAx_rows[i]==fVYAx_rows[i+1]) {
               mDA2_VYAx_data[index]=0.0;
            } else {
               mDA2_VYAx_data[index] *= fVYAx_data[fVYAx_rows[i]];
            }
         }
      }
      // (*m2)(n,i) * (*fDA2)(i,j)*(*fVYAx)(i)
      TMatrixDSparse *m2_mDA2_VYAx = MultiplyMSparseMSparse(*m2,mDA2_VYAx);

      const Int_t *m2_mDA2_VYAx_rows=m2_mDA2_VYAx->GetRowIndexArray();
      const Int_t *m2_mDA2_VYAx_cols=m2_mDA2_VYAx->GetColIndexArray();
      const Double_t *m2_mDA2_VYAx_data=m2_mDA2_VYAx->GetMatrixArray();   

      const Int_t *m1_rows=m1->GetRowIndexArray();
      const Int_t *m1_cols=m1->GetColIndexArray();
      const Double_t *m1_data=m1->GetMatrixArray();   
      
      const Int_t *m2_rows=m2->GetRowIndexArray();
      const Int_t *m2_cols=m2->GetColIndexArray();
      const Double_t *m2_data=m2->GetMatrixArray();   

      // add everything together
      for(Int_t m=0;m<r->GetNrows();m++) {
         for(Int_t n=0;n<r->GetNrows();n++) {
   // sum_j (*m1)(m,j)*(*m1)(n,j) * sum_i (*fDA2)(i,j)*(*fVYAx)(i)*(*fVYAx)(i)
            Int_t index_m=m1_rows[m];
            Int_t index_n=m1_rows[n];
            while((index_m<m1_rows[m+1])&&(index_n<m1_rows[n+1])) {
               Int_t j=m1_cols[index_m];
               Int_t delta=j-m1_cols[index_n];
               if(delta<0) {
                  index_m++;
               } else if(delta>0) {
                  index_n++;
               } else {
                  (*r)(m,n) +=
                     m1_data[index_m]*m1_data[index_n]*DA2t_VYAx2(j,0);
                  index_m++;
                  index_n++;
               }
            }
   //+sum_i (*m2)(m,i)*(*m2)(n,i) * sum_j (*fDA2)(i,j)*(*fX)(j)*(*fX)(j)
            index_m=m2_rows[m];
            index_n=m2_rows[n];
            while((index_m<m2_rows[m+1])&&(index_n<m2_rows[n+1])) {
               Int_t i=m2_cols[index_m];
               Int_t delta=i-m2_cols[index_n];
               if(delta<0) {
                  index_m++;
               } else if(delta>0) {
                  index_n++;
               } else {
                  (*r)(m,n) +=
                     m2_data[index_m]*m2_data[index_n]*DA2_x2(i,0);
                  index_m++;
                  index_n++;
               }
            }
         }
   //-sum_j (*m1)(m,j)*(*fX)(j) * sum_i (*m2)(n,i)*(*fDA2)(i,j)*(*fVYAx)(i)
   //-sum_j (*m1)(n,j)*(*fX)(j) * sum_i (*m2)(m,i)*(*fDA2)(i,j)*(*fVYAx)(i)
         for(Int_t n=0;n<m1->GetNrows();n++) {
            Int_t index_m=m2_mDA2_VYAx_rows[m];
            Int_t index_n=m1_rows[n];
            while((index_m<m2_mDA2_VYAx_rows[m+1])
                  &&(index_n<m1_rows[n+1])) {
               Int_t j=m2_mDA2_VYAx_cols[index_m];
               Int_t delta=j-m1_cols[index_n];
               if(delta<0) {
                  index_m++;
               } else if(delta>0) {
                  index_n++;
               } else {
                  Double_t d_mn = m1_data[index_n]*m2_mDA2_VYAx_data[index_n]*
                     (*fX)(j,0);
                  (*r)(m,n) -= d_mn; 
                  (*r)(n,m) -= d_mn; 
                  index_m++;
                  index_n++;
               }
            }
         }
      }
      delete m2_mDA2_VYAx;
   }
   if(fDAcol) {
      // error matrix  (dz_m/de_j)*(dz_n/de_j)  * (*fDAcol)(j,0)*(*fDAcol)(j,0)

      // -sum_i {((*m1)(m,j) * (*fVYAx)(i) - (*m2)(m,i) * (*fX)(j))*a_ij}
      
      // sum_i (*m2)(m,i)*a_ij
      TMatrixDSparse *delta=MultiplyMSparseMSparse(*m2,*fA);
      Double_t *delta_data=delta->GetMatrixArray();
      const Int_t *delta_rows=delta->GetRowIndexArray();
      const Int_t *delta_cols=delta->GetColIndexArray();
      // sum_i (*m2)(m,i)*a_ij * (*fX)(j)
      for(Int_t row=0;row<delta->GetNrows();row++) {
         for(Int_t index=delta_rows[row];index<delta_rows[row+1];index++) {
            Int_t col=delta_cols[index];
            delta_data[index] *= (*fX)(col,0);
         }
      }
      // sum_i { a_ij * (*fVYAx)(i) }
      TMatrixDSparse *AtVYAx=MultiplyMSparseTranspMSparse(*fA,*fVYAx);
      const Double_t *AtVYAx_data=AtVYAx->GetMatrixArray();
      const Int_t *AtVYAx_rows=AtVYAx->GetRowIndexArray();
 
      const Int_t *m1_rows=m1->GetRowIndexArray();
      const Int_t *m1_cols=m1->GetColIndexArray();
      const Double_t *m1_data=m1->GetMatrixArray();   
      Int_t nmax= m1_rows[m1->GetNrows()];
      if(nmax>0) {
         Double_t *delta2_data=new Double_t[nmax];
         Int_t *delta2_rows=new Int_t[nmax];
         Int_t *delta2_cols=new Int_t[nmax];
         nmax=0;
         // (*m1)(m,j) * sum_i { a_ij * (*fVYAx)(i) }
         for(Int_t row=0;row<m1->GetNrows();row++) {
            for(Int_t index=m1_rows[row];index<m1_rows[row+1];index++) {
               Int_t col=m1_cols[index];
               if(AtVYAx_rows[col]<AtVYAx_rows[col+1]) {
                  delta2_rows[nmax]=row;
                  delta2_cols[nmax]=col;
                  delta2_data[nmax]=m1_data[index]*
                     AtVYAx_data[AtVYAx_rows[col]];
                  if(delta2_data[nmax] !=0.0) nmax++;
               }
            }
         }
         if(nmax>0) {
            TMatrixDSparse delta2(m1->GetNrows(),m1->GetNcols());
            delta2.SetMatrixArray(nmax,delta2_rows,delta2_cols,delta2_data);
            // subtract from derivative matrix
            AddMSparse(*delta,-1.,delta2);
         }
         delete [] delta2_cols;
         delete [] delta2_rows;
         delete [] delta2_data;
      }
      delete AtVYAx;
      delta_data=delta->GetMatrixArray();
      delta_rows=delta->GetRowIndexArray();
      delta_cols=delta->GetColIndexArray();
      // multiply by systematic error
      for(Int_t row=0;row<delta->GetNrows();row++) {
         for(Int_t index=delta_rows[row];index<delta_rows[row+1];index++) {
            Int_t col=delta_cols[index];
            delta_data[index] *= (*fDAcol)(col,0);
         }
      }
      // delta * delta# is the error matrix
      for(Int_t row=0;row<delta->GetNrows();row++) {
         for(Int_t col=0;col<delta->GetNrows();col++) {
            Int_t index1=delta_rows[row];
            Int_t index2=delta_rows[col];
            while((index1<delta_rows[row+1])&&(index2<delta_rows[col+1])) {
               Int_t j=delta_cols[index1];
               Int_t dj=j-delta_cols[index2];
               if(dj<0) {
                  index1++;
               } else if(dj>0) {
                  index2++;
               } else {
                  (*r)(row,col) += delta_data[index1]*delta_data[index2];
                  index1++;
                  index2++;
               }
            }
         }
      }
      delete delta;
   }
   return r;
}

TMatrixD *TUnfoldSys::PrepareCorrEmat(TMatrixDSparse const *m1,TMatrixDSparse const *m2,TMatrixDSparse const *dsys) {
   // prepare error matrix of correlated systematic shifts
   //   m1,m2 : coefficients for propagating the errors
   //   dsys : matrix of correlated shifts from this source

   TMatrixD *r=new TMatrixD(m1->GetNrows(),m1->GetNrows());

   // delta_m = 
   //   sum{i,j}   {
   //      ((*m1)(m,j) * (*fVYAx)(i) - (*m2)(m,i) * (*fX)(j))*dsys(i,j) }
   //   =    sum_j (*m1)(m,j)  sum_i dsys(i,j) * (*fVYAx)(i)
   //     -  sum_i (*m2)(m,i)  sum_j dsys(i,j) * (*fX)(j)
   // emat_mn = delta_m*delta_n

   TMatrixDSparse *dsysT_VYAx = MultiplyMSparseTranspMSparse(*dsys,*fVYAx);
   TMatrixDSparse *delta =  MultiplyMSparseMSparse(*m1,*dsysT_VYAx);
   delete dsysT_VYAx;
   TMatrixDSparse *dsys_X = MultiplyMSparseM(*dsys,*fX);
   TMatrixDSparse *delta2 = MultiplyMSparseMSparse(*m2,*dsys_X);
   delete dsys_X;
   AddMSparse(*delta,-1.0,*delta2);
   delete delta2;
   const Double_t *delta_data=delta->GetMatrixArray();
   const Int_t *delta_rows=delta->GetRowIndexArray();
   for(Int_t row=0;row<delta->GetNrows();row++) {
      if(delta_rows[row]>=delta_rows[row+1]) continue;
      for(Int_t col=0;col<delta->GetNrows();col++) {
         if(delta_rows[col]>=delta_rows[col+1]) continue;
         (*r)(row,col)=
            delta_data[delta_rows[row]]*delta_data[delta_rows[col]];
      }
   }
   delete delta;
   return r;
}

void TUnfoldSys::GetEmatrixSysSource(TH2 *ematrix,char const *name,
                                     Int_t const *binMap,Bool_t clearEmat) {
   // calculate systematic error matrix from a given source
   //    ematrix: output
   //    source: name of the error source
   //    binMap: see method GetEmatrix()
   //    clearEmat: set kTRUE to clear the histogram prior to adding the errors
   PrepareSysError();
   TPair const *named_emat=(TPair const *)fErrorUncorrX->FindObject(name);
   TMatrixD *emat=0;
   if(named_emat) {
      emat=(TMatrixD *)named_emat->Value();
   }
   ErrorMatrixToHist(ematrix,emat,binMap,clearEmat);
}

void TUnfoldSys::GetEmatrixSysTotal(TH2 *ematrix,Int_t const *binMap,
                                    Bool_t clearEmat) {
   // calculate total systematic error matrix
   //    ematrix: output
   //    binMap: see method GetEmatrix()
   //    clearEmat: set kTRUE to clear the histogram prior to adding the errors
   GetEmatrixSysUncorr(ematrix,binMap,clearEmat);
   TMapIter sysErrPtr(fErrorCorrX);
   TObject const *key;
   for(key=sysErrPtr.Next();key;key=sysErrPtr.Next()) {
      ErrorMatrixToHist
         (ematrix,
          (TMatrixD const *)
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,20,00)
          ((TPair const *)*sysErrPtr)->Value()
#else
          (fErrorCorrX->GetValue(((TObjString const *)key)->GetString()))
#endif
          ,binMap,kFALSE);
   }
}


void TUnfoldSys::GetEmatrixTotal(TH2 *ematrix,Int_t const *binMap) {
   // get total error including statistical error
   //    ematrix: output
   //    binMap: see method GetEmatrix()
   GetEmatrix(ematrix,binMap);
   GetEmatrixSysTotal(ematrix,binMap,kFALSE);
}  

Double_t TUnfoldSys::GetChi2Sys(void) {
   // calculate total chi**2 including systematic errors
   PrepareSysError();
   TMatrixD *emat_sum_1=InvertMSparse(*fV);
   TMatrixDSparse emat_sum(*emat_sum_1);
   delete emat_sum_1;
   AddMSparse(emat_sum,1.0,*fErrorUncorrAx);
   TMapIter sysErrPtr(fErrorCorrAx);
   TObject const *key;
   for(key=sysErrPtr.Next();key;key=sysErrPtr.Next()) {
      AddMSparse(emat_sum,1.0,
                 *(TMatrixD const *)
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,20,00)
                 ((TPair const *)*sysErrPtr)->Value()
#else
                 fErrorCorrAx->GetValue(((TObjString const *)key)
                                          ->GetString())
#endif
                 );
   }
   TMatrixD *v=InvertMSparse(emat_sum);
   TMatrixD dy(*fY, TMatrixD::kMinus, *fAx);
   Double_t r=MultiplyVecMSparseVec(*v,dy);
   delete v;
   return r;
}

