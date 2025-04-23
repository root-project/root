{
TFile *dirfile = new TFile("dirkeydelete.root?filetype=pcm", "RECREATE", "", 0);
dirfile->mkdir("subdir");
TObject *o = new TNamed("subdir","some title and more inf");
o->Write("",TObject::kWriteDelete);
cout << "File map after the object write/delete\n";
dirfile->Map("forcomp");
auto key = (TKey*)dirfile->GetListOfKeys()->FindObject("subdir");
key->Delete();
cout << "File map after the explicit key delete\n";
dirfile->Map("forcomp");
o->Write();
cout << "File map after the 2nd object write\n";
gFile->Map("forcomp");
gFile->Write();
cout << "Final file map\n";
gFile->Map("forcomp");
delete dirfile;

cout << "Reading the file\n";
TFile *file = TFile::Open("dirkeydelete.root");
file->ls();
file->Map("forcomp");
o = gFile->Get("subdir");
o->Print();
}
