// global TMVA style settings
#ifndef TMVA_TMVAGLOB
#define TMVA_TMVAGLOB

#include "RVersion.h"

namespace TMVAGlob {

   // --------- S t y l e ---------------------------
   const Bool_t UsePaperStyle = 0;
   // -----------------------------------------------

   enum TypeOfPlot { kNormal = 0,
                     kDecorrelated,
                     kPCA,
                     kNumOfMethods };

   // set the style
   void SetSignalAndBackgroundStyle( TH1* sig, TH1* bkg, TH1* all = 0 ) 
   {

      //signal
      const Int_t FillColor__S = 38+150;
      const Int_t FillStyle__S = 1001;
      const Int_t LineColor__S = 104;
      const Int_t LineWidth__S = 2;

      // background
      const Int_t icolor = UsePaperStyle ? 2 + 100 : 2;
      const Int_t FillColor__B = icolor;
      const Int_t FillStyle__B = 3554;
      const Int_t LineColor__B = icolor;
      const Int_t LineWidth__B = 2;

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

   // set frame styles
   SetFrameStyle( TH1* frame, Float_t scale = 1.0 )
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

   // set style and remove existing canvas'
   void Initialize( Bool_t useTMVAStyle = kTRUE )
   {
      // set style
      if (!useTMVAStyle) {
         gROOT->SetStyle("Plain");
         gStyle->SetOptStat(0);
      }

      // destroy canvas'
      TList * loc = gROOT->GetListOfCanvases();
      TListIter itc(loc);
      TObject *o(0);
      while ((o = itc())) delete o;

      // define new line styles
      TStyle *TMVAStyle = gROOT->GetStyle("Plain"); // our style is based on Plain
      TMVAStyle->SetLineStyleString( 5, "[52 12]" );
      TMVAStyle->SetLineStyleString( 6, "[22 12]" );
      TMVAStyle->SetLineStyleString( 7, "[22 10 7 10]" );
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
         cout << "Opening root file " << fin << " in read mode" << endl;
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
         cout << "--- Error in TMVAGlob::imgconv: canvas is NULL" << endl;
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
         if (UsePaperStyle) c->Print(epsName);      
         //          cout << "If you want to save the image as gif or png, please comment out "
         //               << "the corresponding lines (line no. 142+143) in tmvaglob.C" << endl;
         c->Print(pngName);
      }
   }
   
   void plot_logo( Float_t v_scale = 1.0 )
   {
      TImage *img = TImage::Open("../macros/tmva_logo.gif");
      if (!img) {
         cut <<"Could not open image ../macros/tmva_logo.gif" << endl;
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

      Float_t x1L = x1R - d*r;
      Float_t y1T = y1B + d*v_scale;
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

   void NormalizeHists( TH1* sig, TH1* bkg = 0 ) 
   {
      if (sig->GetSumw2N() == 0) sig->Sumw2();
      if (bkg != 0) if (bkg->GetSumw2N() == 0) bkg->Sumw2();
      
      Float_t dx = (sig->GetXaxis()->GetXmax() - sig->GetXaxis()->GetXmin())/sig->GetNbinsX();
      sig->Scale( 1.0/sig->GetSumOfWeights()/dx );
      if (bkg != 0) bkg->Scale( 1.0/bkg->GetSumOfWeights()/dx );      
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
      TDirectory *dir=0;
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

   UInt_t GetListOfKeys( TList & keys, TString inherits, TDirectory *dir=0 )
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

   TKey *FindMethod( TString name, TDirectory *dir=0 )
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
         } else {
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
      cout << "--- Found " << ni << " methods(s)" << endl;
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
         cout << "Could not locate directory '" << methodName << endl;
         return 0;
      }

      return GetListOfTitles( rfdir, titles );

      TList *keys = rfdir->GetListOfKeys();
      if (keys==0) {
         cout << "Directory '" << methodName << "' contains no keys" << endl;
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

   UInt_t GetListOfTitles( TDirectory *rfdir, TList & titles )
   {
      // get a list of titles (i.e TDirectory) given a method dir
      UInt_t ni=0;
      if (rfdir==0) return 0;
      TList *keys = rfdir->GetListOfKeys();
      if (keys==0) {
         cout << "Directory '" << rfdir->GetName() << "' contains no keys" << endl;
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

   TDirectory *GetCorrelationPlotsDir( TMVAGlob::TypeOfPlot type, TDirectory *dir=0 )
   {
      // get the CorrelationPlots directory
      if (dir==0) dir = GetInputVariablesDir( 0, type );
      if (dir==0) return 0;
      //
      TDirectory* corrdir = (TDirectory*)dir->Get( "CorrelationPlots" );
      if (corrdir==0) {
         cout << "Could not CorrelationPlots directory '" << corrDirName << endl;
         return 0;
      }
      return corrdir;
   }

   TDirectory *GetInputVariablesDir( TMVAGlob::TypeOfPlot type, TDirectory *dir=0 )
   {
      // get the InputVariables directory
      const TString directories[TMVAGlob::kNumOfMethods] = { "InputVariables_NoTransform",
                                                             "InputVariables_DecorrTransform",
                                                             "InputVariables_PCATransform" };
      if (dir==0) dir = gDirectory;
      // get top dir containing all hists of the variables
      TDirectory* dir = (TDirectory*)gDirectory->Get( directories[type] );
      if (dir==0) {
         cout << "Could not locate input variable directory '" << directories[type] << endl;
         return 0;
      }
      return dir;
   }
}

#endif
