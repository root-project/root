void work(int x, int y, int page){ 
//example of script to plot all histograms in a Root file
//on a Postcript file (x times y per page)
//The following logic can be used to loop on all the keys of a file:
// TFile f("myfile.root");
// TIter next(f1->GetListOfKeys());
// TKey *key;
// while ((key = (TKey*)next())) {
//    //key->GetClassName() returns the name of the object class
//    TObject *obj = key->ReadObj(); //read object from file
//    //obj->ClassName() should be equal to key->GetClassName()
//    //obj->InheritsFrom("someclass") test if obj inherits from someclass
// }    
 int page_cnt, hist_per_page, i;  
 TFile *f1 = new TFile("hsimple.root");
 TCanvas *c1 = new TCanvas("c1");
 TPostScript *ps = new TPostScript("file.ps",112);
 c1->Divide(x,y);
 hist_per_page = x*y; 
 TIter next(f1->GetListOfKeys());       //make an iterator on list of keys
 TKey *key;
 while (page_cnt < page) {
    ps->NewPage();
    i=1;
    while (hist_per_page >= i) {
       c1->cd(i);
       key = (TKey*)next();             //get next key on the file
       if (!key) break;                 //if no more keys, key=0
       TObject *obj = key->ReadObj();   //read object associated to key
       if (obj->InheritsFrom("TH1")) {  //interested by histograms only
          obj->Draw();                  //draw histogram with default option
          i++;
       }
    }
    c1->Update();
    if (!key) break;
    page_cnt++;     
 }
 ps->Close();
 //gSystem->Exec("lpr -Psmith2079 file.ps");
 //gSystem->Exec("gv file.ps");
}
