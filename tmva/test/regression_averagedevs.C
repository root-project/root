#include "tmvaglob.C"

/*
this macro plots the quadratic deviation of the estimated from the target value, averaged over the first nevt events in test sample (all if Nevt=-1)
a) normal average
b) truncated average, using best 90%
 created January 2009, Eckhard von Toerne, University of Bonn, Germany
*/

void regression_averagedevs(TString fin, Int_t Nevt=-1, Bool_t useTMVAStyle = kTRUE )
{
   bool debug=false;
   if (Nevt <0)  Nevt=1000000; 
   Int_t type = 2;
   TMVAGlob::Initialize( useTMVAStyle );
   // checks if file with name "fin" is already open, and if not opens one
   TFile* file = TMVAGlob::OpenFile( fin );  
   TList jobDirList;
   TMVAGlob::GetListOfJobs(file,jobDirList);
   if (jobDirList.GetSize()==0) {
     cout << "error could not find jobs" << endl;
     return;
   }
   
   Bool_t __PLOT_LOGO__  = kTRUE;
   Bool_t __SAVE_IMAGE__ = kTRUE;

   TDirectory* dir0 = (TDirectory*) (jobDirList.At(0));
   //TDirectory* dir0 = (TDirectory*) (file->Get("InputVariables_Id"));   
   Int_t nTargets = TMVAGlob::GetNumberOfTargets( dir0);

   if (debug) cout << "found targets " << nTargets<<endl;
   TCanvas* c=0;
   for (Int_t itrgt = 0 ; itrgt < nTargets; itrgt++){
     if (debug) cout << "loop targets " << itrgt<<endl;
     TString xtit = "Method";
     TString ytit = "Average Quadratic Deviation";  
     TString ftit = ytit + " versus " + xtit + Form(" for target %d",itrgt);
     c = new TCanvas( Form("c%d",itrgt), ftit , 50+20*itrgt, 10*itrgt, 750, 650 );
     
     // global style settings
     c->SetGrid();
     c->SetTickx(1);
     c->SetTicky(0);
     c->SetTopMargin(0.28);
     c->SetBottomMargin(0.1);
     
     TString hNameRef(Form("regression_average_devs_target%d",itrgt));
     
     const Int_t maxMethods = 100;
     const Int_t maxTargets = 100;
     Float_t m[4][maxMethods]; // h0 train-all, h1 train-90%, h2 test-all, h3 test-90%
     Float_t em[4][maxMethods];
     Float_t x[4][maxMethods];
     Float_t ex[4][maxMethods];

     TIter next(&jobDirList);
     Float_t mymax=0., mymin=1.e40;
     TString mvaNames[maxMethods];
     TDirectory *jobDir;
     Int_t nMethods = 0;
     // loop over all methods
     while (jobDir = (TDirectory*)next()) {     
       TString methodTitle;
       TMVAGlob::GetMethodTitle(methodTitle,jobDir);
       mvaNames[nMethods]=methodTitle;
       if (debug) cout << "--- Found directory for method: " << methodTitle << endl;
       TIter keyIt(jobDir->GetListOfKeys());
       TKey *histKey;
       while ((histKey = (TKey*)keyIt())) {
	 if (histKey->ReadObj()->InheritsFrom("TH1F") ){
	   TString s(histKey->ReadObj()->GetName());
	   if( !s.Contains("Quadr_Dev") ) continue;
	   if( !s.Contains(Form("target_%d_",itrgt))) continue;
	   Int_t ihist = 0 ;
	   if( !s.Contains("best90perc") && s.Contains("train")) ihist=0;
	   if( s.Contains("best90perc") && s.Contains("train")) ihist=1;
	   if( !s.Contains("best90perc") && s.Contains("test")) ihist=2;
	   if( s.Contains("best90perc") && s.Contains("test")) ihist=3; 
	   if (debug) cout <<"using histogram" << s << ", ihist="<<ihist<<endl;
	   TH1F* h = (TH1F*) (histKey->ReadObj());
	   m[ihist][nMethods] = sqrt(h->GetMean());
	   em[ihist][nMethods] = h->GetRMS()/(sqrt(h->GetEntries())*2.*h->GetMean());
	   x[ihist][nMethods] = nMethods+0.44+0.12*ihist;
	   ex[ihist][nMethods] = 0.001;
	   mymax=  m[ihist][nMethods] > mymax ? m[ihist][nMethods] : mymax;
	   mymin=  m[ihist][nMethods] < mymin ? m[ihist][nMethods] : mymin;
	   if (debug) cout << "m"<< ihist << "="<<m[ihist][nMethods]<<endl;
	 }
       }
       nMethods++;
     }
     TH1F* haveragedevs= new TH1F(Form("haveragedevs%d",itrgt),ftit,nMethods,0.,nMethods);
     for (int i=0;i<nMethods;i++) haveragedevs->GetXaxis()->SetBinLabel(i+1, mvaNames[i]);
     haveragedevs->SetStats(0);
     TGraphErrors* graphTrainAv= new TGraphErrors(nMethods,x[0],m[0],ex[0],em[0]);
     TGraphErrors* graphTruncTrainAv= new TGraphErrors(nMethods,x[1],m[1],ex[1],em[1]);
     TGraphErrors* graphTestAv= new TGraphErrors(nMethods,x[2],m[2],ex[2],em[2]);
     TGraphErrors* graphTruncTestAv= new TGraphErrors(nMethods,x[3],m[3],ex[3],em[3]);
     
     Double_t xmax = 1.2 * mymax;
     Double_t xmin = 0.8 * mymin - (mymax - mymin)*0.05;
     Double_t xheader = 0.2;
     Double_t yheader = xmax*0.92;
     xmin = xmin > 0.? xmin : 0.;
     if (mymin > 1.e-20 && log10(mymax/mymin)>1.5){
       c->SetLogy();
       cout << "--- result differ significantly using log scale for display of regression results"<< endl;
       xmax = 1.5 * xmax;
       xmin = 0.75 * mymin;
       yheader = xmax*0.78;
     }
     Float_t x0L = 0.03,     y0H = 0.91;
     Float_t dxL = 0.457-x0L, dyH = 0.14;
     //     TLegend *legend = new TLegend( x0L, y0H-dyH, x0L+dxL, y0H , "Average Deviation = (#sum_{evts} (f_{MVA} - f_{target})^{2} )^{1/2}");
     TLegend *legend = new TLegend( x0L, y0H-dyH, x0L+dxL, y0H );
     legend->SetTextSize( 0.035 );
     legend->SetTextAlign(12);
     legend->SetMargin( 0.1 );

     TH1F *hr = c->DrawFrame(-1.,0.,nMethods+1, xmax);
     cout << endl;
     cout << "Training: Average Deviation between target " << itrgt <<" and estimate" << endl;
     cout << Form("%-15s%-15s%-15s", "Method","Average Dev.","trunc. Aver.(90%)") <<endl;
     for (int i=0;i<nMethods;i++){
	cout << Form("%-15s:%#10.3g%#10.3g",
		     (const char*)mvaNames[i], m[0][i],m[1][i])<<endl;
	//       cout << mvaNames[i] << "  " << m[0][i]<< "  "<< m[1][i]<<endl;
       hr->GetXaxis()->SetBinLabel(i+1," ");
     }
     cout << endl;
     cout << "Testing: Average Deviation between target " << itrgt <<" and estimate" << endl;
     cout << Form("%-15s%-15s%-15s", "Method","Average Dev.","trunc. Aver.(90%)") <<endl;
     for (int i=0;i<nMethods;i++){
	cout << Form("%-15s:%#10.3g%#10.3g",
		     (const char*)mvaNames[i], m[2][i],m[3][i])<<endl;
	//cout << mvaNames[i] << "  " << m[2][i]<< "  "<< m[3][i]<<endl;
     }

     haveragedevs->SetMinimum(xmin);
     haveragedevs->SetMaximum(xmax);
     haveragedevs->SetXTitle("Method");
     haveragedevs->SetYTitle("Deviation from target");
     haveragedevs->Draw();
     c->GetFrame()->SetFillColor(21);
     c->GetFrame()->SetBorderSize(12);
     graphTrainAv->SetMarkerSize(1.);
     graphTrainAv->SetMarkerColor(kBlue);
     graphTrainAv->SetMarkerStyle(25);
     graphTrainAv->Draw("P");
     
     graphTruncTrainAv->SetMarkerSize(1.);
     graphTruncTrainAv->SetMarkerColor(kBlack);
     graphTruncTrainAv->SetMarkerStyle(25);
     graphTruncTrainAv->Draw("P");

     graphTestAv->SetMarkerSize(1.);
     graphTestAv->SetMarkerColor(kBlue);
     graphTestAv->SetMarkerStyle(21);
     graphTestAv->Draw("P");
     
     graphTruncTestAv->SetMarkerSize(1.);
     graphTruncTestAv->SetMarkerColor(kBlack);
     graphTruncTestAv->SetMarkerStyle(21);
     graphTruncTestAv->Draw("P");
     legend->AddEntry(graphTrainAv,TString("Training Sample, Average Deviation"),"p");
     legend->AddEntry(graphTruncTrainAv,TString("Training Sample, truncated Average Dev. (best 90%)"),"p");
     legend->AddEntry(graphTestAv,TString("Test Sample, Average Deviation"),"p");
     legend->AddEntry(graphTruncTestAv,TString("Test Sample, truncated Average Dev. (best 90%)"),"p");

     legend->Draw();
     TLatex legHeader;
     legHeader.SetTextSize(0.035);
     legHeader.SetTextAlign(12);
     //legHeader.DrawLatex(x0L, y0H+0.01, "Average Deviation = (#sum (_{ } f_{MVA} - f_{target})^{2} )^{1/2}");
     legHeader.DrawLatex(xheader, yheader, "Average Deviation = (#sum (_{ } f_{MVA} - f_{target})^{2} )^{1/2}");     
     // ============================================================
     
     if (__PLOT_LOGO__) TMVAGlob::plot_logo();
     // ============================================================
     
     c->Update();
     TString fname = "plots/" + hNameRef;
     if (__SAVE_IMAGE__) TMVAGlob::imgconv( c, fname );   
   } // end loop itrgt
   return;
}

