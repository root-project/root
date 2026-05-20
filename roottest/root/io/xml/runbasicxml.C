#include "test_classes.h"

void runbasicxml()
{
   gSystem->Load("libXmlTestDictionaries");

   TString sbuf;

   std::cout << std::endl << "Doing TXmlEx1 - basic data types" << std::endl;

   TXmlEx1 iex1;
   sbuf = TBufferXML::ConvertToXML(&iex1, gROOT->GetClass("TXmlEx1"));
   auto ex1 = (TXmlEx1 *) TBufferXML::ConvertFromXMLAny(sbuf);
   if (!ex1)
      std::cout << "TXmlEx1 cannot be read from xml" << std::endl;
   else
      ex1->Print();
   delete ex1;

   std::cout << std::endl << "Doing TXmlEx2 - fixed-size arrays of basic types" << std::endl;

   TXmlEx2 iex2;
   sbuf = TBufferXML::ConvertToXML(&iex2, gROOT->GetClass("TXmlEx2"));
   auto ex2 = (TXmlEx2 *) TBufferXML::ConvertFromXMLAny(sbuf);
   if (!ex2)
      std::cout << "TXmlEx2 cannot be read from xml" << std::endl;
   else
      ex2->Print();
   delete ex2;

   std::cout << std::endl << "Doing TXmlEx3 - dynamic arrays of basic types" << std::endl;

   TXmlEx3 iex3;
   sbuf = TBufferXML::ConvertToXML(&iex3, gROOT->GetClass("TXmlEx3"));
   auto ex3 = (TXmlEx3 *) TBufferXML::ConvertFromXMLAny(sbuf);
   if (!ex3)
      std::cout << "TXmlEx3 cannot be read from xml" << std::endl;
   else
      ex3->Print();
   delete ex3;

   std::cout << std::endl << "Doing TXmlEx4 - inheritance and different strings" << std::endl;

   TXmlEx4 iex4(true);
   sbuf = TBufferXML::ConvertToXML(&iex4, gROOT->GetClass("TXmlEx4"));
   auto ex4 = (TXmlEx4 *) TBufferXML::ConvertFromXMLAny(sbuf);
   if (!ex4)
      std::cout << "TXmlEx4 cannot be read from xml" << std::endl;
   else
      ex4->Print();
   delete ex4;

   std::cout << std::endl << "Doing TXmlEx5 - objects as data memebers" << std::endl;

   TXmlEx5 iex5(true);
   sbuf = TBufferXML::ConvertToXML(&iex5, gROOT->GetClass("TXmlEx5"));
   auto ex5 = (TXmlEx5 *) TBufferXML::ConvertFromXMLAny(sbuf);
   if (!ex5)
      std::cout << "TXmlEx5 cannot be read from xml" << std::endl;
   else
      ex5->Print();
   delete ex5;

   std::cout << std::endl << "Doing TXmlEx6 - arrays of objects as data memebers" << std::endl;

   TXmlEx6 iex6(true);
   sbuf = TBufferXML::ConvertToXML(&iex6, gROOT->GetClass("TXmlEx6"));
   auto ex6 = (TXmlEx6 *) TBufferXML::ConvertFromXMLAny(sbuf);
   if (!ex6)
      std::cout << "TXmlEx6 cannot be read from xml" << std::endl;
   else
      ex6->Print();
   delete ex6;

   std::cout << std::endl << "Doing TXmlEx7 - all kinds of STL containers" << std::endl;

   TXmlEx7 iex7(true);
   sbuf = TBufferXML::ConvertToXML(&iex7, gROOT->GetClass("TXmlEx7"));
   auto ex7 = (TXmlEx7*) TBufferXML::ConvertFromXMLAny(sbuf);
   if (!ex7)
      std::cout << "TXmlEx7 cannot be read from xml" << std::endl;
   else
      ex7->Print();
   delete ex7;

   std::cout << std::endl << "Doing TXmlEx8 - inheritance from std::vector<int>" << std::endl;

   TXmlEx8 iex8(true);
   sbuf = TBufferXML::ConvertToXML(&iex8, gROOT->GetClass("TXmlEx8"));
   auto ex8 = (TXmlEx8 *) TBufferXML::ConvertFromXMLAny(sbuf);
   if (!ex8)
      std::cout << "TXmlEx8 cannot be read from xml" << std::endl;
   else
      ex8->Print();
   delete ex8;

   std::cout << "Done " << std::endl;
}
