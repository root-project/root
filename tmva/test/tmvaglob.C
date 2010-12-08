// global TMVA style settings
#ifndef TMVA_TMVAGLOB
#define TMVA_TMVAGLOB

#include <iostream>
#include <vector>

#include "TPad.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TSystem.h"
#include "TImage.h"
#include "TKey.h"
#include "TH1.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TObjArray.h"


#include "RVersion.h"

using std::cout;
using std::endl;

namespace TMVAGlob {

   // --------- S t y l e ---------------------------
   static Bool_t UsePaperStyle = 0;
   // -----------------------------------------------

   enum TypeOfPlot { kId = 0,
                     kNorm,
                     kDecorrelated,
                     kPCA,
                     kGaussDecorr,
                     kNumOfMethods };

   static Int_t c_Canvas         = TColor::GetColor( "#f0f0f0" );
   static Int_t c_FrameFill      = TColor::GetColor( "#fffffd" );
   static Int_t c_TitleBox       = TColor::GetColor( "#5D6B7D" );
   static Int_t c_TitleBorder    = TColor::GetColor( "#7D8B9D" );
   static Int_t c_TitleText      = TColor::GetColor( "#FFFFFF" );
   static Int_t c_SignalLine     = TColor::GetColor( "#0000ee" );
   static Int_t c_SignalFill     = TColor::GetColor( "#7d99d1" );
   static Int_t c_BackgroundLine = TColor::GetColor( "#ff0000" );
   static Int_t c_BackgroundFill = TColor::GetColor( "#ff0000" );
   static Int_t c_NovelBlue      = TColor::GetColor( "#2244a5" );

   // set the style
   void SetSignalAndBackgroundStyle( TH1* sig, TH1* bkg, TH1* all = 0 ) 
   {
      //signal
      // const Int_t FillColor__S = 38 + 150; // change of Color Scheme in ROOT-5.16.
      // convince yourself with gROOT->GetListOfColors()->Print()
      Int_t FillColor__S = c_SignalFill;
      Int_t FillStyle__S = 1001;
      Int_t LineColor__S = c_SignalLine;
      Int_t LineWidth__S = 2;

      // background
      //Int_t icolor = UsePaperStyle ? 2 + 100 : 2;
      Int_t FillColor__B = c_BackgroundFill;
      Int_t FillStyle__B = 3554;
      Int_t LineColor__B = c_BackgroundLine;
      Int_t LineWidth__B = 2;

      if (sig != NULL) {
         sig->SetLineColor( LineColor__S );
         sig->SetLineWidth( LineWidth__S );
         sig->SetFillStyle( FillStyle__S );
         sig->SetFillColor( FillColor__S );
      }
 
      if (bkg != NULL) {
         bkg->SetLineColor( LineColor__B );
         bkg->SetLineWidth( LineWidth__B );
         bkg->SetFillStyle( FillStyle__B );
         bkg->SetFillColor( FillColor__B );
      }

      if (all != NULL) {
         all->SetLineColor( LineColor__S );
         all->SetLineWidth( LineWidth__S );
         all->SetFillStyle( FillStyle__S );
         all->SetFillColor( FillColor__S );
      }
   }

   void SetMultiClassStyle( TObjArray* hists ) 
   {
      //signal
      // const Int_t FillColor__S = 38 + 150; // change of Color Scheme in ROOT-5.16.
      // convince yourself with gROOT->GetListOfColors()->Print()
      //Int_t FillColor__S = c_SignalFill;
      //Int_t FillStyle__S = 1001;
      //Int_t LineColor__S = c_SignalLine;
      //Int_t LineWidth__S = 2;

      // background
      //Int_t icolor = UsePaperStyle ? 2 + 100 : 2;
      //Int_t FillColor__B = c_BackgroundFill;
      //Int_t FillStyle__B = 3554;
      //Int_t LineColor__B = c_BackgroundLine;
      //Int_t LineWidth__B = 2;

      Int_t FillColors[10] = {38,2,3,6,7,8,9,11};
      Int_t LineColors[10] = {4,2,3,6,7,8,9,11};
      Int_t FillStyles[5] = {1001,3554,3003,3545,0};

      for(Int_t i=0; i<hists->GetEntriesFast(); ++i){
         ((TH1*)(*hists)[i])->SetFillColor(FillColors[i%10]);
         ((TH1*)(*hists)[i])->SetFillStyle(FillStyles[i%5]);
         ((TH1*)(*hists)[i])->SetLineColor(LineColors[i%10]);
         ((TH1*)(*hists)[i])->SetLineWidth(2);
      }
   }

   // set frame styles
   void SetFrameStyle( TH1* frame, Float_t scale = 1.0 )
   {
      frame->SetLabelOffset( 0.012, "X" );// label offset on x axis
      frame->SetLabelOffset( 0.012, "Y" );// label offset on x axis
      frame->GetXaxis()->SetTitleOffset( 1.25 );
      frame->GetYaxis()->SetTitleOffset( 1.22 );
      frame->GetXaxis()->SetTitleSize( 0.045*scale );
      frame->GetYaxis()->SetTitleSize( 0.045*scale );
      Float_t labelSize = 0.04*scale;
      frame->GetXaxis()->SetLabelSize( labelSize );
      frame->GetYaxis()->SetLabelSize( labelSize );

      // global style settings
      gPad->SetTicks();
      gPad->SetLeftMargin  ( 0.108*scale );
      gPad->SetRightMargin ( 0.050*scale );
      gPad->SetBottomMargin( 0.120*scale  );
   }

   void SetTMVAStyle() {
      
      TStyle *TMVAStyle = gROOT->GetStyle("TMVA");
      if(TMVAStyle!=0) {
         gROOT->SetStyle("TMVA");
         return;
      }
			
      TMVAStyle = new TStyle(*gROOT->GetStyle("Plain")); // our style is based on Plain
      TMVAStyle->SetName("TMVA");
      TMVAStyle->SetTitle("TMVA style based on \"Plain\" with modifications defined in tmvaglob.C");
      gROOT->GetListOfStyles()->Add(TMVAStyle);
      gROOT->SetStyle("TMVA");
			
      TMVAStyle->SetLineStyleString( 5, "[52 12]" );
      TMVAStyle->SetLineStyleString( 6, "[22 12]" );
      TMVAStyle->SetLineStyleString( 7, "[22 10 7 10]" );

      // the pretty color palette of old
      TMVAStyle->SetPalette((UsePaperStyle ? 18 : 1),0);

      // use plain black on white colors
      TMVAStyle->SetFrameBorderMode(0);
      TMVAStyle->SetCanvasBorderMode(0);
      TMVAStyle->SetPadBorderMode(0);
      TMVAStyle->SetPadColor(0);
      TMVAStyle->SetFillStyle(0);

      TMVAStyle->SetLegendBorderSize(0);

      // title properties
      // TMVAStyle->SetTitleW(.4);
      // TMVAStyle->SetTitleH(.10);
      // MVAStyle->SetTitleX(.5);
      // TMVAStyle->SetTitleY(.9);
      TMVAStyle->SetTitleFillColor( c_TitleBox );
      TMVAStyle->SetTitleTextColor( c_TitleText );
      TMVAStyle->SetTitleBorderSize( 1 );
      TMVAStyle->SetLineColor( c_TitleBorder );
      if (!UsePaperStyle) {
         TMVAStyle->SetFrameFillColor( c_FrameFill );
         TMVAStyle->SetCanvasColor( c_Canvas );
      }

      // set the paper & margin sizes
      TMVAStyle->SetPaperSize(20,26);
      TMVAStyle->SetPadTopMargin(0.10);
      TMVAStyle->SetPadRightMargin(0.05);
      TMVAStyle->SetPadBottomMargin(0.11);
      TMVAStyle->SetPadLeftMargin(0.12);

      // use bold lines and markers
      TMVAStyle->SetMarkerStyle(21);
      TMVAStyle->SetMarkerSize(0.3);
      TMVAStyle->SetHistLineWidth(2);
      TMVAStyle->SetLineStyleString(2,"[12 12]"); // postscript dashes

      // do not display any of the standard histogram decorations
      TMVAStyle->SetOptTitle(1);
      TMVAStyle->SetTitleH(0.052);

      TMVAStyle->SetOptStat(0);
      TMVAStyle->SetOptFit(0);

      // put tick marks on top and RHS of plots
      TMVAStyle->SetPadTickX(1);
      TMVAStyle->SetPadTickY(1);

   }

   void DestroyCanvases()
   {

      TList* loc = (TList*)gROOT->GetListOfCanvases();
      TListIter itc(loc);
      TObject *o(0);
      while ((o = itc())) delete o;
   }

   // set style and remove existing canvas'
   void Initialize( Bool_t useTMVAStyle = kTRUE )
   {
      // destroy canvas'
      DestroyCanvases();

      // set style
      if (!useTMVAStyle) {
         gROOT->SetStyle("Plain");
         gStyle->SetOptStat(0);
         return;
      }

      SetTMVAStyle();
   }

   // checks if file with name "fin" is already open, and if not opens one
   TFile* OpenFile( const TString& fin )
   {
      TFile* file = gDirectory->GetFile();
      if (file==0 || fin != file->GetName()) {
         if (file != 0) {
            gROOT->cd();
            file->Close();
         }
         cout << "--- Opening root file " << fin << " in read mode" << endl;
         file = TFile::Open( fin, "READ" );
      }
      else {
         file = gDirectory->GetFile();
      }

      file->cd();
      return file;
   }

   // used to create output file for canvas
   void imgconv( TCanvas* c, const TString & fname )
   {
      // return;
      if (NULL == c) {
         cout << "*** Error in TMVAGlob::imgconv: canvas is NULL" << endl;
      }
      else {
         // create directory if not existing
         TString f = fname;
         TString dir = f.Remove( f.Last( '/' ), f.Length() - f.Last( '/' ) );
         gSystem->mkdir( dir );

         TString pngName = fname + ".png";
         TString gifName = fname + ".gif";
         TString epsName = fname + ".eps";
         c->cd();

         // create eps (other option: c->Print( epsName ))
         if (UsePaperStyle) {
            c->Print(epsName);
         } 
         else {
            cout << "--- --------------------------------------------------------------------" << endl;
            cout << "--- If you want to save the image as eps, gif or png, please comment out " << endl;
            cout << "--- the corresponding lines (line no. 239-241) in tmvaglob.C" << endl;
            cout << "--- --------------------------------------------------------------------" << endl;
            c->Print(epsName);
            c->Print(pngName);
            // c->Print(gifName);
         }
      }
   }

   TImage * findImage(const char * imageName) 
   { 
      // looks for the image in macropath
      TString macroPath(gROOT->GetMacroPath()); // look for the image in here
      Ssiz_t curIndex(0);
      TImage *img(0);
      while(1) {
         Ssiz_t pathStart = curIndex;
         curIndex = macroPath.Index(":",curIndex);
         Ssiz_t pathEnd = (curIndex==-1)?macroPath.Length():curIndex;
         TString path(macroPath(pathStart,pathEnd-pathStart));
         
         gSystem->ExpandPathName(path);
         const char* fullName = Form("%s/%s", path.Data(), imageName);

         Bool_t fileFound = ! gSystem->AccessPathName(fullName);

         if(fileFound) {
            img = TImage::Open(fullName);
            break;
         }
         if(curIndex==-1) break;
         curIndex++;
      }
      return img;
   }

   void plot_logo( Float_t v_scale = 1.0, Float_t skew = 1.0 )
   {

      TImage *img = findImage("tmva_logo.gif");
      if (!img) {
         cout << "+++ Could not open image tmva_logo.gif" << endl;
         return;
      }
      
      img->SetConstRatio(kFALSE);
      UInt_t h_ = img->GetHeight();
      UInt_t w_ = img->GetWidth();

      Float_t r = w_/h_;
      gPad->Update();
      Float_t rpad = Double_t(gPad->VtoAbsPixel(0) - gPad->VtoAbsPixel(1))/(gPad->UtoAbsPixel(1) - gPad->UtoAbsPixel(0));
      r *= rpad;

      Float_t d = 0.055;
      // absolute coordinates
      Float_t x1R = 1 - gStyle->GetPadRightMargin(); 
      Float_t y1B = 1 - gStyle->GetPadTopMargin()+.01; // we like the logo to sit a bit above the histo 

      Float_t x1L = x1R - d*r/skew;
      Float_t y1T = y1B + d*v_scale*skew;
      if (y1T>0.99) y1T = 0.99;

      TPad *p1 = new TPad("imgpad", "imgpad", x1L, y1B, x1R, y1T );
      p1->SetRightMargin(0);
      p1->SetBottomMargin(0);
      p1->SetLeftMargin(0);
      p1->SetTopMargin(0);
      p1->Draw();

      Int_t xSizeInPixel = p1->UtoAbsPixel(1) - p1->UtoAbsPixel(0);
      Int_t ySizeInPixel = p1->VtoAbsPixel(0) - p1->VtoAbsPixel(1);
      if (xSizeInPixel<=25 || ySizeInPixel<=25) {
         delete p1;
         return; // ROOT doesn't draw smaller than this
      }

      p1->cd();
      img->Draw();
   } 

   void NormalizeHist( TH1* h ) 
   {
      if (h==0) return;
      if (h->GetSumw2N() == 0) h->Sumw2();
      if(h->GetSumOfWeights()!=0) {
         Float_t dx = (h->GetXaxis()->GetXmax() - h->GetXaxis()->GetXmin())/h->GetNbinsX();
         h->Scale( 1.0/h->GetSumOfWeights()/dx );
      }
   }
   void NormalizeHists( TH1* sig, TH1* bkg = 0 ) 
   {
      if (sig->GetSumw2N() == 0) sig->Sumw2();
      if (bkg && bkg->GetSumw2N() == 0) bkg->Sumw2();
      
      if(sig->GetSumOfWeights()!=0) {
         Float_t dx = (sig->GetXaxis()->GetXmax() - sig->GetXaxis()->GetXmin())/sig->GetNbinsX();
         sig->Scale( 1.0/sig->GetSumOfWeights()/dx );
      }
      if (bkg != 0 && bkg->GetSumOfWeights()!=0) {
         Float_t dx = (bkg->GetXaxis()->GetXmax() - bkg->GetXaxis()->GetXmin())/bkg->GetNbinsX();
         bkg->Scale( 1.0/bkg->GetSumOfWeights()/dx );
      }
   }

   // the following are tools to help handling different methods and titles


   void GetMethodName( TString & name, TKey * mkey ) {
      if (mkey==0) return;
      name = mkey->GetName();
      name.ReplaceAll("Method_","");
   }

   void GetMethodTitle( TString & name, TKey * ikey ) {
      if (ikey==0) return;
      name = ikey->GetName();
   }

   void GetMethodName( TString & name, TDirectory * mdir ) {
      if (mdir==0) return;
      name = mdir->GetName();
      name.ReplaceAll("Method_","");
   }

   void GetMethodTitle( TString & name, TDirectory * idir ) {
      if (idir==0) return;
      name = idir->GetName();
   }

   TKey *NextKey( TIter & keyIter, TString className) {
      TKey *key=(TKey *)keyIter.Next();
      TKey *rkey=0;
      Bool_t loop=(key!=0);
      //
      while (loop) {
         TClass *cl = gROOT->GetClass(key->GetClassName());
         if (cl->InheritsFrom(className.Data())) {
            loop = kFALSE;
            rkey = key;
         } else {
            key = (TKey *)keyIter.Next();
            if (key==0) loop = kFALSE;
         }
      }
      return rkey;
   }

   UInt_t GetListOfKeys( TList& keys, TString inherits, TDirectory *dir=0 )
   {
      // get a list of keys with a given inheritance
      // the list contains TKey objects
      if (dir==0) dir = gDirectory;
      TIter mnext(dir->GetListOfKeys());
      TKey *mkey;
      keys.Clear();
      keys.SetOwner(kFALSE);
      UInt_t ni=0;
      while ((mkey = (TKey*)mnext())) {
         // make sure, that we only look at TDirectory with name Method_<xxx>
         TClass *cl = gROOT->GetClass(mkey->GetClassName());
         if (cl->InheritsFrom(inherits)) {
            keys.Add(mkey);
            ni++;
         }
      }
      return ni;
   }

   Int_t GetNumberOfTargets( TDirectory *dir )
   {
      TIter next(dir->GetListOfKeys());
      TKey* key    = 0;
      Int_t noTrgts = 0;
         
      while ((key = (TKey*)next())) {
         if (key->GetCycle() != 1) continue;        
         if (TString(key->GetName()).Contains("__Regression_target")) noTrgts++;
      }
      return noTrgts;
   }

   Int_t GetNumberOfInputVariables( TDirectory *dir )
   {
      TIter next(dir->GetListOfKeys());
      TKey* key    = 0;
      Int_t noVars = 0;
         
      while ((key = (TKey*)next())) {
         if (key->GetCycle() != 1) continue;
         
         // count number of variables (signal is sufficient), exclude target(s)
         if (TString(key->GetName()).Contains("__Signal") || (TString(key->GetName()).Contains("__Regression") && !(TString(key->GetName()).Contains("__Regression_target")))) noVars++;
      }
      
      return noVars;
   }

   std::vector<TString> GetInputVariableNames(TDirectory *dir )
   {
      TIter next(dir->GetListOfKeys());
      TKey* key = 0;
      //set<std::string> varnames;
      std::vector<TString> names;
      
      while ((key = (TKey*)next())) {
         if (key->GetCycle() != 1) continue;
         TClass *cl = gROOT->GetClass(key->GetClassName());
         if (!cl->InheritsFrom("TH1")) continue;
         TString name(key->GetName());
         Int_t pos = name.First("__");
         name.Remove(pos);
         Bool_t hasname = false;
         std::vector<TString>::const_iterator iter = names.begin();
         while(iter != names.end()){
            if(name.CompareTo(*iter)==0)
               hasname=true;
            iter++;
         }
         if(!hasname)
            names.push_back(name);
      }
      return names;
   }

   Int_t GetNumberOfInputVariablesMultiClass( TDirectory *dir ){
      std::vector<TString> names(GetInputVariableNames(dir));
      return names.end() - names.begin();
   }
   
   std::vector<TString> GetClassNames(TDirectory *dir )
   {      
      
      TIter next(dir->GetListOfKeys());
      TKey* key = 0;
      //set<std::string> varnames;
      std::vector<TString> names;
      
      while ((key = (TKey*)next())) {
         if (key->GetCycle() != 1) continue;
         TClass *cl = gROOT->GetClass(key->GetClassName());
         if (!cl->InheritsFrom("TH1")) continue;
         TString name(key->GetName());
         name.ReplaceAll("_Deco","");
         name.ReplaceAll("_Gauss","");
         name.ReplaceAll("_PCA","");
         name.ReplaceAll("_Id","");
         name.ReplaceAll("_vs_","");
         char c = '_';
         Int_t pos = name.Last(c);
         name.Remove(0,pos+1);
         
         /*Int_t pos = name.First("__");
         name.Remove(0,pos+2);
         char c = '_';
         pos = name.Last(c);
         name.Remove(pos);
         if(name.Contains("Gauss")){
            pos = name.Last(c);
            name.Remove(pos);
         }
         pos = name.Last(c);
         if(pos!=-1)
            name.Remove(0,pos+1);
         */
         Bool_t hasname = false;
         std::vector<TString>::const_iterator iter = names.begin();
         while(iter != names.end()){
            if(name.CompareTo(*iter)==0)
               hasname=true;
            iter++;
         }
         if(!hasname)
            names.push_back(name);
      }
      return names;
   }


   TKey* FindMethod( TString name, TDirectory *dir=0 )
   {
      // find the key for a method
      if (dir==0) dir = gDirectory;
      TIter mnext(dir->GetListOfKeys());
      TKey *mkey;
      TKey *retkey=0;
      Bool_t loop=kTRUE;
      while (loop) {
         mkey = (TKey*)mnext();
         if (mkey==0) {
            loop = kFALSE;
         } 
         else {
            TString clname = mkey->GetClassName();
            TClass *cl = gROOT->GetClass(clname);
            if (cl->InheritsFrom("TDirectory")) {
               TString mname = mkey->GetName(); // method name
               TString tname = "Method_"+name;  // target name
               if (mname==tname) { // target found!
                  loop = kFALSE;
                  retkey = mkey;
               }
            }
         }
      }
      return retkey;
   }

   Bool_t ExistMethodName( TString name, TDirectory *dir=0 )
   {
      // find the key for a method
      if (dir==0) dir = gDirectory;
      TIter mnext(dir->GetListOfKeys());
      TKey *mkey;
      Bool_t loop=kTRUE;
      while (loop) {
         mkey = (TKey*)mnext();
         if (mkey==0) {
            loop = kFALSE;
         } 
         else {
            TString clname  = mkey->GetClassName();
            TString keyname = mkey->GetName();
            TClass *cl = gROOT->GetClass(clname);
            if (keyname.Contains("Method") && cl->InheritsFrom("TDirectory")) {

               TDirectory* d_ = (TDirectory*)dir->Get( keyname );
               if (!d_) {
                  cout << "HUUUGE TROUBLES IN TMVAGlob::ExistMethodName() --> contact authors" << endl;
                  return kFALSE;
               }

               TIter mnext_(d_->GetListOfKeys());
               TKey *mkey_;
               while ((mkey_ = (TKey*)mnext_())) {
                  TString clname_ = mkey_->GetClassName();
                  TClass *cl_ = gROOT->GetClass(clname_);
                  if (cl_->InheritsFrom("TDirectory")) {
                     TString mname = mkey_->GetName(); // method name
                     if (mname==name) { // target found!                  
                        return kTRUE;
                     }
                  }
               }
            }
         }
      }
      return kFALSE;
   }

   UInt_t GetListOfMethods( TList & methods, TDirectory *dir=0 )
   {
      // get a list of methods
      // the list contains TKey objects
      if (dir==0) dir = gDirectory;
      TIter mnext(dir->GetListOfKeys());
      TKey *mkey;
      methods.Clear();
      methods.SetOwner(kFALSE);
      UInt_t ni=0;
      while ((mkey = (TKey*)mnext())) {
         // make sure, that we only look at TDirectory with name Method_<xxx>
         TString name = mkey->GetClassName();
         TClass *cl = gROOT->GetClass(name);
         if (cl->InheritsFrom("TDirectory")) {
            if (TString(mkey->GetName()).BeginsWith("Method_")) {
               methods.Add(mkey);
               ni++;
            }
         }
      }
      cout << "--- Found " << ni << " classifier types" << endl;
      return ni;
   }

   UInt_t GetListOfJobs( TFile* file, TList& jobdirs)
   {
      // get a list of all jobs in all method directories
      // based on ideas by Peter and Joerg found in macro deviations.C
      TIter next(file->GetListOfKeys());
      TKey *key(0);   
      while ((key = (TKey*)next())) {
         
         if (TString(key->GetName()).BeginsWith("Method_")) {
            if (gROOT->GetClass(key->GetClassName())->InheritsFrom("TDirectory")) {

               TDirectory* mDir = (TDirectory*)key->ReadObj();
               
               TIter keyIt(mDir->GetListOfKeys());
               TKey *jobkey;
               while ((jobkey = (TKey*)keyIt())) {
                  if (!gROOT->GetClass(jobkey->GetClassName())->InheritsFrom("TDirectory")) continue;
                  
                  TDirectory *jobDir = (TDirectory *)jobkey->ReadObj();
                  cout << "jobdir name  " << jobDir->GetName() << endl;
                  jobdirs.Add(jobDir);
               }
            }
         }
      }
      return jobdirs.GetSize();
   }

   UInt_t GetListOfTitles( TDirectory *rfdir, TList & titles )
   {
      // get a list of titles (i.e TDirectory) given a method dir
      UInt_t ni=0;
      if (rfdir==0) return 0;
      TList *keys = rfdir->GetListOfKeys();
      if (keys==0) {
         cout << "+++ Directory '" << rfdir->GetName() << "' contains no keys" << endl;
         return 0;
      }
      //
      TIter rfnext(rfdir->GetListOfKeys());
      TKey *rfkey;
      titles.Clear();
      titles.SetOwner(kFALSE);
      while ((rfkey = (TKey*)rfnext())) {
         // make sure, that we only look at histograms
         TClass *cl = gROOT->GetClass(rfkey->GetClassName());
         if (cl->InheritsFrom("TDirectory")) {
            titles.Add(rfkey);
            ni++;
         }
      }
      cout << "--- Found " << ni << " instance(s) of the method " << rfdir->GetName() << endl;
      return ni;
   }

   UInt_t GetListOfTitles( TString & methodName, TList & titles, TDirectory *dir=0 )
   {
      // get the list of all titles for a given method
      // if the input dir is 0, gDirectory is used
      // returns a list of keys
      UInt_t ni=0;
      if (dir==0) dir = gDirectory;
      TDirectory* rfdir = (TDirectory*)dir->Get( methodName );
      if (rfdir==0) {
         cout << "+++ Could not locate directory '" << methodName << endl;
         return 0;
      }

      return GetListOfTitles( rfdir, titles );

      TList *keys = rfdir->GetListOfKeys();
      if (keys==0) {
         cout << "+++ Directory '" << methodName << "' contains no keys" << endl;
         return 0;
      }
      //
      TIter rfnext(rfdir->GetListOfKeys());
      TKey *rfkey;
      titles.Clear();
      titles.SetOwner(kFALSE);
      while ((rfkey = (TKey*)rfnext())) {
         // make sure, that we only look at histograms
         TClass *cl = gROOT->GetClass(rfkey->GetClassName());
         if (cl->InheritsFrom("TDirectory")) {
            titles.Add(rfkey);
            ni++;
         }
      }
      cout << "--- Found " << ni << " instance(s) of the method " << methodName << endl;
      return ni;
   }

   TDirectory *GetInputVariablesDir( TMVAGlob::TypeOfPlot type, TDirectory *dir=0 )
   {
      // get the InputVariables directory
      const TString directories[TMVAGlob::kNumOfMethods] = { "InputVariables_Id",
                                                             "InputVariables_Deco",
                                                             "InputVariables_PCA",
                                                             "InputVariables_Gauss_Deco" };
      if (dir==0) dir = gDirectory;

      // get top dir containing all hists of the variables
      dir = (TDirectory*)gDirectory->Get( directories[type] );
      if (dir==0) {
         cout << "+++ Could not locate input variable directory '" << directories[type] << endl;
         return 0;
      }
      return dir;
   }

   TDirectory *GetCorrelationPlotsDir( TMVAGlob::TypeOfPlot type, TDirectory *dir=0 )
   {
      // get the CorrelationPlots directory
      if (dir==0) dir = GetInputVariablesDir( type, 0 );
      if (dir==0) return 0;
      //
      TDirectory* corrdir = (TDirectory*)dir->Get( "CorrelationPlots" );
      if (corrdir==0) {
         cout << "+++ Could not find CorrelationPlots directory 'CorrelationPlots'" << endl;
         return 0;
      }
      return corrdir;
   }

}

#endif
