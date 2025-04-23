// This script is developed to test new functionality of TDirectory
// 1. dir->SetName() should also rename keys structure
// 2. dir->ReadAll("dirs*") should read only subdirectories of the file
// 3. dir->rmdir("dirname") should delete directory and free all keys from all subdirectories

TFile* OpenFile(int mode, const char* openmode = "open")
{
//   cout << "OpenFile mode = " << mode << " openmode = " << openmode << endl;
    
   TFile* f = 0;
   switch(mode) {
      case 1: {
         f = TFile::Open("test.xml", openmode);
         break;
      }
//      case 2: {
//         f = new TSQLFile("mysql://hostname.domain:3306/test", openmode, "username", "userpass");
//         break;
//      }
      default: {
         f = TFile::Open("test.root", openmode);
         break;
      }
   }
   
   if (f==0) 
      cout << "Error opening file in mode: " << openmode << endl; 
                
   return f;
}

TH1* MakeHisto(const char* histoname) 
{
   // just create small histogram with content inside
   int nbins = 100; 
   TH1I* h1 = new TH1I(histoname,"histo title", nbins, 0., nbins);
   for (int n=0;n<nbins;n++)
      h1->SetBinContent(n+1, n+1);
   h1->SetDirectory(0);
   return h1;
}

void CreateSubdirs(int mode, int levels) 
{
   // create subdirectories with objects inside file
    
   TFile *f = OpenFile(mode, "update");
   if (f==0) return;
   
   TDirectory* last = f;
   TH1* h1 = 0;
   for (int level=0;level<levels;level++) {
      last = last->mkdir(Form("dir_%d",level));
      h1 = MakeHisto(Form("histo_%d",level));
      last->cd();
      h1->Write();
   }

   delete f;
}

void RemoveSubdirs(int mode, const char* removename)
{
   // delete subdirs, created by CreateSubdirs() functions
    
   TFile *f = OpenFile(mode, "update");
   if (f==0) return;
   
//   f->rmdir("dir_0");
   f->Delete(removename);
   
   delete f;
}

void ShowSubdirs(int mode, const char* opt = "-m") 
{
   // Read complete dirs structure and displays it 
    
   TFile *f = OpenFile(mode);
   if (f==0) return;
   
   f->ReadAll("dirs*");
   
   cout << "File size = " << f->GetSize() << endl;
   
   f->ls(opt);
   
   cout << endl;
   
   delete f;
}

void TestSubdirs(const char* msg, int mode, int maxsize, const char* testdir = 0) 
{
   // Read complete dirs structure and check file size and presence of subdirs
    
   TFile *f = OpenFile(mode);
   if (f==0) return;
   
   f->ReadAll("dirs*");
   
   if (f->GetSize() > maxsize) {
      cout << msg << endl;
      cout << "  Error: File size " << f->GetSize() << " too big. Expected not more than " << maxsize << endl;
   }
   
   if (testdir!=0) {
      TDirectory* dir = dynamic_cast<TDirectory*> (f->Get(testdir));
      if (dir==0) {
         cout << msg << endl;
         cout << "   Error: subdirectory " << testdir << " not found" << endl;
      }
   }
   
   delete f;
}


void RenameSubDir(int mode, const char* dirpath, const char* newname)
{
   TFile *f = OpenFile(mode, "update");
   if (f==0) return;
   
   //TDirectory* dir = f->GetDirectory("dir_0/dir_1/dir_2");
   TDirectory* dir = (TDirectory*) f->Get(dirpath);
   
   if (dir==0) 
      cout << "Error: Cannot find directory " << dirpath << endl;
   else 
      dir->SetName(newname);
   
   delete f;
}


void rundirdelete()
{
   int mode = 0; // here we will test only ROOT binray file
 
   // first create file with one subdirectory and objects inside
   TFile *f = OpenFile(mode, "recreate");
   if (f==0) return;
   
   TDirectory* dir3 = f->mkdir("dir3");
   dir3->cd();
   TH1* h1 = MakeHisto("histo3");
   h1->Write();
   
   f->cd();
   h1 = MakeHisto("histo4");
   h1->Write();
   
   delete f;
   
   // create subdirectory with 5 levels inside
   CreateSubdirs(mode, 5);

   // Test that subdirectories are created and existing
   TestSubdirs("Initial file creation", mode, 15000, "dir_0/dir_1/dir_2/dir_3/dir_4");

   // Rename one of subdirectories   
   RenameSubDir(mode,"dir_0/dir_1/dir_2","dir_2_newname");

   // Test that subdir is renamed
   TestSubdirs("After rename", mode, 15000, "dir_0/dir_1/dir_2_newname/dir_3/dir_4");

   // Delete subdirectories     
   RemoveSubdirs(mode, "dir_0;*");

   // Test that subdir is renamed, no subdirs tests
   TestSubdirs("After dir delete", mode, 15000);
   
   // now several time create and remove subdirectories from the file
   for (int n=0;n<20;n++) {
      CreateSubdirs(mode, 5); 
      RemoveSubdirs(mode, "dir_0;*"); 
   }

   // Test that size is not too much
   TestSubdirs("After 20 dir create/delete", mode, 30000);
}

