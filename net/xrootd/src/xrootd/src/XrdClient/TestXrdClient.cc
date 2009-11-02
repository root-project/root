//       $Id$

//const char *TestClientCVSID = "$Id$";
// Simple keleton for simple tests of Xrdclient and XrdClientAdmin
//

#include "XrdClient/XrdClient.hh"
#include "XrdClient/XrdClientAdmin.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdSys/XrdSysHeaders.hh"

int main(int argc, char **argv) {

//    EnvPutInt(NAME_DEBUG, 3);
  //EnvPutInt(NAME_READCACHESIZE, 100000000);


//   XrdClient *x = new XrdClient(argv[1]);
//   XrdClient *y = new XrdClient(argv[2]);
//   x->Open(0, 0);
    
//      for (int i = 0; i < 1000; i++)
//        x->Copy("/tmp/testcopy");
  
//   x->Close();

//   delete x;
//   x = 0;
   
//   y->Open(0, 0);
  
//      for (int i = 0; i < 1000; i++)
//        x->Copy("/tmp/testcopy");
  
//   y->Close();
  
//   delete y;

  XrdClientUrlInfo u;
  XrdClientAdmin *adm = new XrdClientAdmin(argv[1]);

  adm->Connect();

   string s;
   int i = 0;
   XrdClientLocate_Info loc;
   while (!cin.eof()) {
     cin >> s;

     if (!s.size()) continue;

     if (!adm->Locate((kXR_char*)s.c_str(), loc)) {
       cout << endl <<
	 " The server complained for file:" << endl <<
	 s.c_str() << endl << endl;
     }

     if (!(i % 100)) cout << i << "...";
     i++;
//     if (i == 9000) break;
   }

//  vecString vs;
//  XrdOucString os;
// string s;
//  int i = 0;
//  while (!cin.eof()) {
//    cin >> s;

//    if (!s.size()) continue;

//    os = s.c_str();
//    vs.Push_back(os);

//    if (!(i % 200)) {
//      cout << i << "...";
//      adm->Prepare(vs, kXR_stage, 0);
//      vs.Clear();
//    }

//    i++;

//  }



//  adm->Prepare(vs, 0, 0);
//  cout << endl << endl;

  delete adm;

      


}
