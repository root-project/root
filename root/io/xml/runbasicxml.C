{
// Fill out the code of the actual test
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   if (1) {
#endif
#ifndef SECOND_RUN
   gROOT->ProcessLine(".L test_classes.h+");
#endif

#if defined(ClingWorkAroundMissingDynamicScope) && !defined(SECOND_RUN)
#define SECOND_RUN
   gROOT->ProcessLine(".x runbasicxml.C");
#else
   TString sbuf;
   
   cout << endl << "Doing TXmlEx1 - basic data types" << endl;
   
   TXmlEx1* ex1 = new TXmlEx1;
   sbuf = TBufferXML::ConvertToXML(ex1, gROOT->GetClass("TXmlEx1"));
   delete ex1;
   ex1 = (TXmlEx1*) TBufferXML::ConvertFromXMLAny(sbuf);
   if (ex1==0) 
      cout << "TXmlEx1 cannot be read from xml" << endl;
   else 
      ex1->Print();
   delete ex1;

   cout << endl << "Doing TXmlEx2 - fixed-size arrays of basic types" << endl;
   
   TXmlEx2* ex2 = new TXmlEx2;
   sbuf = TBufferXML::ConvertToXML(ex2, gROOT->GetClass("TXmlEx2"));
   delete ex2;
   ex2 = (TXmlEx2*) TBufferXML::ConvertFromXMLAny(sbuf);
   if (ex2==0) 
      cout << "TXmlEx2 cannot be read from xml" << endl;
   else 
      ex2->Print();
   delete ex2;

   cout << endl << "Doing TXmlEx3 - dynamic arrays of basic types" << endl;
   
   TXmlEx3* ex3 = new TXmlEx3;
   sbuf = TBufferXML::ConvertToXML(ex3, gROOT->GetClass("TXmlEx3"));
   delete ex3;
   ex3 = (TXmlEx3*) TBufferXML::ConvertFromXMLAny(sbuf);
   if (ex3==0) 
      cout << "TXmlEx3 cannot be read from xml" << endl;
   else 
      ex3->Print();
   delete ex3;

   cout << endl << "Doing TXmlEx4 - inheritance and different strings" << endl;
   
   TXmlEx4* ex4 = new TXmlEx4(true);
   sbuf = TBufferXML::ConvertToXML(ex4, gROOT->GetClass("TXmlEx4"));
   delete ex4;
   ex4 = (TXmlEx4*) TBufferXML::ConvertFromXMLAny(sbuf);
   if (ex4==0) 
      cout << "TXmlEx4 cannot be read from xml" << endl;
   else 
      ex4->Print();
   delete ex4;

   cout << endl << "Doing TXmlEx5 - objects as data memebers" << endl;
   
   TXmlEx5* ex5 = new TXmlEx5(true);
   sbuf = TBufferXML::ConvertToXML(ex5, gROOT->GetClass("TXmlEx5"));
   delete ex5;
   ex5 = (TXmlEx5*) TBufferXML::ConvertFromXMLAny(sbuf);
   if (ex5==0) 
      cout << "TXmlEx5 cannot be read from xml" << endl;
   else 
      ex5->Print();
   delete ex5;

   cout << endl << "Doing TXmlEx6 - arrays of objects as data memebers" << endl;

   TXmlEx6* ex6 = new TXmlEx6(true);
   sbuf = TBufferXML::ConvertToXML(ex6, gROOT->GetClass("TXmlEx6"));
   delete ex6;
   ex6 = (TXmlEx6*) TBufferXML::ConvertFromXMLAny(sbuf);
   if (ex6==0) 
      cout << "TXmlEx6 cannot be read from xml" << endl;
   else 
      ex6->Print();
   delete ex6;

   cout << endl << "Doing TXmlEx7 - all kinds of STL containers" << endl;

   TXmlEx7* ex7 = new TXmlEx7(true);
   sbuf = TBufferXML::ConvertToXML(ex7, gROOT->GetClass("TXmlEx7"));
   delete ex7;
   ex7 = (TXmlEx7*) TBufferXML::ConvertFromXMLAny(sbuf);
   if (ex7==0) 
      cout << "TXmlEx7 cannot be read from xml" << endl;
   else 
      ex7->Print();
   delete ex7;

   cout << endl << "Doing TXmlEx8 - inheritance from std::vector<int>" << endl;

   TXmlEx8* ex8 = new TXmlEx8(true);
   sbuf = TBufferXML::ConvertToXML(ex8, gROOT->GetClass("TXmlEx8"));
   delete ex8;
   ex8 = (TXmlEx8*) TBufferXML::ConvertFromXMLAny(sbuf);
   if (ex8==0) 
      cout << "TXmlEx8 cannot be read from xml" << endl;
   else 
      ex8->Print();
   delete ex8;

   
   cout << "Done " << endl;
#endif
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
      }
#endif
#ifdef ClingWorkAroundBrokenUnnamedReturn
      gApplication->Terminate(0);
#else
      return 0;
#endif
}
