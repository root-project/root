void checkWarn(UInt_t opt) {
   switch (opt) {
      case 0:
         TFile::Open("data1.root");
         gROOT->ProcessLine(".L data2.C+");
         gROOT->GetClass("Tdata")->GetStreamerInfo()->ls();
         break;
      case 1:
         TFile::Open("data1.root");
         gROOT->GetClass("Tdata")->GetStreamerInfo()->ls();
         gROOT->ProcessLine(".L data2.C+");
         break;
      case 2:
         gROOT->ProcessLine(".L data2.C+");
         TFile::Open("data1.root");
         gROOT->GetClass("Tdata")->GetStreamerInfo()->ls();
         break;
      case 3:
         gROOT->ProcessLine(".L data2.C+");
         gROOT->GetClass("Tdata")->GetStreamerInfo()->ls();
         TFile::Open("data1.root");
         break;
      case 4:
         cerr << "No warning is expected from the 1st 4 files opening since we\n"
            << "can not distinguish an emulated ClassDef(name,1) from a foreign class\n";
         TFile::Open("data1.root");
         TFile::Open("data2.root");
         TFile::Open("data3.root");
         TFile::Open("data4.root");
         TFile::Open("data5.root");
         TFile::Open("data6.root");
         break;

      case 10:
         TFile::Open("data1.root");
         gROOT->ProcessLine(".L data2.C+");
         TFile::Open("data3.root");
         gROOT->GetClass("Tdata")->GetStreamerInfo()->ls();
         break;
      case 11:
         TFile::Open("data1.root");
         gROOT->GetClass("Tdata")->GetStreamerInfo()->ls();
         gROOT->ProcessLine(".L data2.C+");
         TFile::Open("data3.root");
         break;
      case 12:
         TFile::Open("data1.root");
         gROOT->ProcessLine(".L data2.C+");
         gROOT->GetClass("Tdata")->GetStreamerInfo()->ls();
         TFile::Open("data3.root");
         break;
      case 13:
         TFile::Open("data1.root");
         gROOT->GetClass("Tdata")->GetStreamerInfo()->ls();
         gROOT->ProcessLine(".L data2.C+");
         gROOT->GetClass("Tdata")->GetStreamerInfo()->ls();
         TFile::Open("data3.root");
         break;

      case 14:
         gROOT->ProcessLine(".L data2.C+");
         TFile::Open("data1.root");
         TFile::Open("data3.root");
         gROOT->GetClass("Tdata")->GetStreamerInfo()->ls();
         break;
      case 15:
         gROOT->ProcessLine(".L data2.C+");
         gROOT->GetClass("Tdata")->GetStreamerInfo()->ls();
         TFile::Open("data1.root");
         TFile::Open("data3.root");
         break;
      case 16:
         gROOT->ProcessLine(".L data2.C+");
         TFile::Open("data1.root");
         gROOT->GetClass("Tdata")->GetStreamerInfo()->ls();
         TFile::Open("data3.root");
         gROOT->GetClass("Tdata")->GetStreamerInfo()->ls();
         break;
      case 17:
         gROOT->ProcessLine(".L data2.C+");
         gROOT->GetClass("Tdata")->GetStreamerInfo()->ls();
         TFile::Open("data1.root");
         gROOT->GetClass("Tdata")->GetStreamerInfo()->ls();
         TFile::Open("data3.root");
         gROOT->GetClass("Tdata")->GetStreamerInfo()->ls();
         break;

      default:
         cout << "Test not implemented\n";
   }
};