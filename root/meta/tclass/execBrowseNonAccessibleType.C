
template<class T>
void browse(TBrowser* b, const char* name){
   auto c=TClass::GetClass(name);
   T a;
   c->Browse(&a,b);
}

void execBrowseNonAccessibleType(){

TCanvas c("","",100,100);

TBrowser b;

browse<TString>(&b,"TString");
browse<TH1F>(&b,"TH1F");
browse<TCanvas>(&b,"TCanvas");
browse<TRolke>(&b,"TRolke");

}
