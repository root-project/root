#ifndef test_classes_xml_h
#define test_classes_xml_h

#include <string>
#include <vector>
#include <list>
#include <deque>
#include <map>
#include <set>
#include <iostream>

#include "TNamed.h"

class TXmlEx1 {
   public:
     int        GetI() { return fInt; }
     long       GetLong() { return fLong; }

     void       SetI(int zn) { fInt = zn; }
     void       SetLong(int zn) { fLong = zn; }

     TXmlEx1()
     {
        fBool = false;
        fChar = 'C';
        fShort = 123;
        fInt = 123456;
        fLong = 7654321;
        fFloat = 1.2;
        fDouble = 3.8;
     }
     virtual ~TXmlEx1() {}

     virtual void Print()
     {
         std::cout << "   fBool = " << fBool << std::endl;
         std::cout << "   fChar = " << fChar << std::endl;
         std::cout << "   fShort = " << fShort << std::endl;
         std::cout << "   fInt = " << fInt << std::endl;
         std::cout << "   fLong = " << fLong << std::endl;
         std::cout << "   fFloat = " << fFloat << std::endl;
         std::cout << "   fDouble = " << fDouble << std::endl;
     }

   protected:
     bool       fBool;
     char       fChar;
     short      fShort;
     int        fInt;      // *OPTION={SetMethod="SetI";GetMethod="GetI"}
     long       fLong;
     float      fFloat;
     double     fDouble;
};

// _______________________________________________________________

class TXmlEx2 {
   protected:
     int        fTest1[4];
     int        fTest2[2][2];
     bool       fBool[2][3][4];
     char       fChar[2][3][4];
     short      fShort[2][3][4];
     int        fInt[2][3][4];
     long       fLong[2][3][4];
     float      fFloat[2][3][4];
     double     fDouble[2][3][4];
   public:
     TXmlEx2()
     {
        fTest1[0] = 11111;
        fTest1[1] = 22222;
        fTest1[2] = 33333;
        fTest1[3] = 44444;

        fTest2[0][0] = 1;
        fTest2[0][1] = 2;
        fTest2[1][0] = 3;
        fTest2[1][1] = 4;

        for (int i=0;i<2;i++)
          for (int j=0;j<3;j++)
            for (int k=0;k<4;k++) {
               fBool[i][j][k] = (i+j) % 2 !=0;
               fChar[i][j][k] = 48 + i+j+k;
               fShort[i][j][k] = i+j+k;
               fInt[i][j][k] = i*j+k;
               fLong[i][j][k] = i*j*k;
               fFloat[i][j][k] = i*j*k;
               fDouble[i][j][k] = i*j*k;
            }
     }

     virtual ~TXmlEx2() {}

     void Print()
     {
         for (int i=0;i<4;i++)
           std::cout << "   fTest1[" << i << "] = " << fTest1[i] << std::endl;

         for (int i=0;i<2;i++)
         for (int j=0;j<3;j++)
         for (int k=0;k<4;k++) {
            std::cout << "  for indexes ["<<i<<"]["<<j<<"]["<<k<<"]" << std::endl;
            std::cout << "     fBool = " << fBool[i][j][k] << std::endl;
            std::cout << "     fChar = " << fChar[i][j][k] << std::endl;
            std::cout << "     fShort = " << fShort[i][j][k] << std::endl;
            std::cout << "     fInt = " << fInt[i][j][k] << std::endl;
            std::cout << "     fLong = " << fLong[i][j][k] << std::endl;
            std::cout << "     fFloat = " << fFloat[i][j][k] << std::endl;
            std::cout << "     fDouble = " << fDouble[i][j][k] << std::endl;
         }
     }
};

// _______________________________________________________________

class TXmlEx3 {
   protected:
     int        fSize;
     bool       *fBool;     // [fSize]
     char       *fChar;     // [fSize]
     short      *fShort;    // [fSize]
     int        *fInt;      // [fSize]
     long       *fLong;     // [fSize]
     float      *fFloat;    // [fSize]
     double     *fDouble;   // [fSize]
   public:

     TXmlEx3()
     {
        fSize = 5;
        fBool = new bool[fSize];
        fChar = new char[fSize];
        fShort = new short[fSize];
        fInt = new int[fSize];
        fLong = new long[fSize];
        fFloat = new float[fSize];
        fDouble = new double[fSize];

        for (int n=0;n<fSize;n++) {
           fBool[n] = false;
           fChar[n] = 49 + n;
           fShort[n] = n*2;
           fInt[n] = n*5;
           fLong[n] = n*123;
           fFloat[n] = n*3;
           fDouble[n] = n*7;
        }

     }
     virtual ~TXmlEx3()
     {
        delete[] fBool;
        delete[] fChar;
        delete[] fShort;
        delete[] fInt;
        delete[] fLong;
        delete[] fFloat;
        delete[] fDouble;
     }

     void Print()
     {
         for (int i=0;i<fSize;i++) {
            std::cout << "  index = " << i << std::endl;
            std::cout << "     fBool = " << fBool[i] << std::endl;
            std::cout << "     fChar = " << fChar[i] << std::endl;
            std::cout << "     fShort = " << fShort[i] << std::endl;
            std::cout << "     fInt = " << fInt[i] << std::endl;
            std::cout << "     fLong = " << fLong[i] << std::endl;
            std::cout << "     fFloat = " << fFloat[i] << std::endl;
            std::cout << "     fDouble = " << fDouble[i] << std::endl;
         }
     }

};

// _______________________________________________________________

class TXmlEx4 : public TXmlEx1 {
   protected:
      char        fStr1[100];
      const char* fStr2;
      int         fDummy2;
      const char* fStr3;
   public:
      TXmlEx4(bool setvalues = false) : TXmlEx1()
      {
        memset(fStr1, 0, sizeof(fStr1));
        fStr2 = 0;
        fDummy2 = 0;
        fStr3 = 0;
        if (setvalues) {
           strcpy(fStr1, "Value of string 1");
           fDummy2 = 1234567;
           fStr3 = new char[1000];
           strcpy((char*)fStr3, "***\t\n/t/n************** Long Value of string 3 *****************************************************************************************************************************************************************************************************************************************************************************************************************");
        }
      }
      virtual ~TXmlEx4()
      {
         delete[] fStr2;
         delete[] fStr3;
      }

      void Print() override
      {
          TXmlEx1::Print();
          std::cout << "   fStr1 = " << fStr1 << std::endl;
          std::cout << "   fStr2 = " << (fStr2 ? fStr2 : "null") << std::endl;
          std::cout << "   fDummy2 = " << fDummy2 << std::endl;
          std::cout << "   fStr3 = " << fStr3 << std::endl;
      }
};

// _______________________________________________________________

class TXmlEx5 {
  protected:

     TXmlEx1    fObj1;
     TXmlEx2    fObj2;

     TXmlEx1    *fPtr1;
     TXmlEx2    *fPtr2;

     TXmlEx1    *fSafePtr1;   //->
     TXmlEx2    *fSafePtr2;   //->

   public:
     TXmlEx1    fObj3;
     TXmlEx1    *fPtr3;
     TXmlEx1    *fSafePtr3;   //->

     TXmlEx5(bool setvalues = false)
     {
        fPtr1 = 0;
        fPtr2 = 0;
        fPtr3 = 0;

        fSafePtr1 = new TXmlEx1;
        fSafePtr2 = new TXmlEx2;
        fSafePtr3 = new TXmlEx1;

        if (setvalues) {
           fPtr1 = new TXmlEx1;
           fPtr2 = new TXmlEx2;
           fPtr3 = fPtr1;
        }
     }

     virtual ~TXmlEx5()
     {
        delete fSafePtr1;
        delete fSafePtr2;
        delete fSafePtr3;
     }

     TXmlEx2&  GetObj2() { return fObj2; }
     TXmlEx2*  GetPtr2() { return fPtr2; }
     void      SetPtr2(TXmlEx2* ptr) { fPtr2 = ptr; }
     TXmlEx2*  GetSafePtr2() { return fSafePtr2; }


     void Print()
     {
        std::cout << std::endl << "!!!!!!!!!! fObj1 !!!!!!!!!!!" << std::endl;
        fObj1.Print();

        std::cout << std::endl << "!!!!!!!!!! fObj2 !!!!!!!!!!!" << std::endl;
        fObj2.Print();

        std::cout << std::endl << "!!!!!!!!!! fObj3 !!!!!!!!!!!" << std::endl;
        fObj3.Print();

        std::cout << std::endl << "!!!!!!!!!! fPtr1 !!!!!!!!!!!" << std::endl;
        if (fPtr1) fPtr1->Print();

        std::cout << std::endl << "!!!!!!!!!! fPtr2 !!!!!!!!!!!" << std::endl;
        if (fPtr2) fPtr2->Print();

        std::cout << std::endl << "!!!!!!!!!! fPtr3 !!!!!!!!!!!" << std::endl;
        if (fPtr3) fPtr3->Print();

        std::cout << std::endl << "!!!!!!!!!! fSafePtr1 !!!!!!!!!!!" << std::endl;
        if (fSafePtr1) fSafePtr1->Print();

        std::cout << std::endl << "!!!!!!!!!! fSafePtr2 !!!!!!!!!!!" << std::endl;
        if (fSafePtr2) fSafePtr2->Print();

        std::cout << std::endl << "!!!!!!!!!! fSafePtr3 !!!!!!!!!!!" << std::endl;
        if (fSafePtr3) fSafePtr3->Print();
     }
};


// _______________________________________________________________


class TXmlEx6 {
  protected:

     TXmlEx1    fObj1[3];
     TXmlEx1    fObj2[3];

     TXmlEx1*   fPtr1[3];
     TXmlEx1*   fPtr2[3];

     TXmlEx1*   fSafePtr1[3];   //->
     TXmlEx1*   fSafePtr2[3];   //->

   public:
     TXmlEx1    fObj3[3];
     TXmlEx1*   fPtr3[3];

     TXmlEx1*   fSafePtr3[3];   //->

     TXmlEx1*   GetObj2() { return fObj2; }
     TXmlEx1**  GetPtr2() { return fPtr2; }
     TXmlEx1**  GetSafePtr2() { return fSafePtr2; }

     TXmlEx6(bool setvalues = false)
     {
        for (int n=0;n<3;n++) {
           fPtr1[n] = 0;
           fPtr2[n] = 0;
           fPtr3[n] = 0;

           fSafePtr1[n] = new TXmlEx1();
           fSafePtr2[n] = new TXmlEx1();
           fSafePtr3[n] = new TXmlEx1();
         }

         if (setvalues)
            for (int n=0;n<3;n++) {
              fPtr1[n] = new TXmlEx1();
              fPtr2[n] = fPtr1[n];
              fPtr3[n] = fPtr1[n];
            }

     }
     virtual ~TXmlEx6()
     {
     }
     void Print()
     {
        for (int n=0;n<3;n++) {
           std::cout << std::endl << "!!!!!!!!!! fObj1["<<n<<"] !!!!!!!!!!!" << std::endl;
           fObj1[n].Print();

           std::cout << std::endl << "!!!!!!!!!! fObj2["<<n<<"] !!!!!!!!!!!" << std::endl;
           fObj2[n].Print();

           std::cout << std::endl << "!!!!!!!!!! fObj3["<<n<<"] !!!!!!!!!!!" << std::endl;
           fObj3[n].Print();

           std::cout << std::endl << "!!!!!!!!!! fPtr1["<<n<<"] !!!!!!!!!!!" << std::endl;
           if (fPtr1[n]) fPtr1[n]->Print();

           std::cout << std::endl << "!!!!!!!!!! fPtr2["<<n<<"] !!!!!!!!!!!" << std::endl;
           if (fPtr2[n]) fPtr2[n]->Print();

           std::cout << std::endl << "!!!!!!!!!! fPtr3["<<n<<"] !!!!!!!!!!!" << std::endl;
           if (fPtr3[n]) fPtr3[n]->Print();

           std::cout << std::endl << "!!!!!!!!!! fSafePtr1["<<n<<"] !!!!!!!!!!!" << std::endl;
           if (fSafePtr1[n]) fSafePtr1[n]->Print();

           std::cout << std::endl << "!!!!!!!!!! fSafePtr2["<<n<<"] !!!!!!!!!!!" << std::endl;
           if (fSafePtr2[n]) fSafePtr2[n]->Print();

           std::cout << std::endl << "!!!!!!!!!! fSafePtr3["<<n<<"] !!!!!!!!!!!" << std::endl;
           if (fSafePtr3[n]) fSafePtr3[n]->Print();
        }
     }
};

// _______________________________________________________________

class TXmlEx7 {
   public:
      std::string            fStr1;
      std::string            fStr2;
      std::string           *fStrPtr1;
      std::string           *fStrPtr2;

      std::string            fStrArr[3];
      std::string           *fStrPtrArr[3];

      bool                        fBoolArr[10];

      std::vector<double>         fVectDouble;
      std::vector<bool>           fVectBool;
      std::vector<TXmlEx1>        fVectEx1;
      std::vector<TXmlEx1*>       fVectEx1Ptr;

      std::vector< double >         fVectDoubleArr[3];
      std::vector< double >         *fVectPtrDouble;
      std::vector< double >         *fVectPtrDoubleArr[3];

      std::vector<std::string>         fVectString;
      std::vector<std::string*>        fVectStringPtr;

      std::list<double>           fListDouble;
      std::list<bool>             fListBool;
      std::list<TXmlEx1>          fListEx1;
      std::list<TXmlEx1*>         fListEx1Ptr;

      std::deque<double>          fDequeDouble;
      std::deque<bool>            fDequeBool;
      std::deque<TXmlEx1>         fDequeEx1;
      std::deque<TXmlEx1*>        fDequeEx1Ptr;

      std::map<int,double>        fMapIntDouble;
      std::map<int,TXmlEx1*>      fMapIntEx1Ptr;
      std::multimap<int,double>   fMultimapIntDouble;

      std::set<double>            fSetDouble;
      std::multiset<double>       fMultisetDouble;

   TXmlEx7(bool setvalues = false)
   {
      fStrPtr1 = 0;
      fStrPtr2 = 0;

      fVectPtrDouble = 0;
      for (int n=0;n<3;n++) {
        fStrPtrArr[n] = 0;
        fVectPtrDoubleArr[n] = 0;
      }

      if (!setvalues) return;

      fStr1 = "String with special characters: \" & < >";
      fStr2 = "Very long Value of STL string// ***********************************************************8// ***********************************************************8// ***********************************************************8// ***********************************************************8// ***********************************************************8// ***********************************************************8// ***********************************************************8// ***********************************************************8// ***********************************************************8";

      fStrPtr1 = 0;
      fStrPtr2 = 0;

      fStrArr[0] = "Value of fStrArr[0]";
      fStrArr[1] = "Value of fStrArr[1]";
      fStrArr[2] = "Value of fStrArr[2]";
      fStrPtrArr[0] = 0;
      fStrPtrArr[1] = 0;
      fStrPtrArr[2] = 0;

      fVectPtrDouble = 0;
      for (int i=0;i<3;i++)
        fVectPtrDoubleArr[i] = 0;

      for (int n=0;n<10;n++) {
         fVectDouble.push_back(n*3);
         fVectBool.push_back(n%2 == 1);
         fBoolArr[n] = (n%2 == 1);
         fVectEx1.push_back(TXmlEx1());

         fVectDoubleArr[0].push_back(n*3);
         fVectDoubleArr[1].push_back(n*3);
         fVectDoubleArr[2].push_back(n*3);

         fListDouble.push_back(n*4);
         fListBool.push_back(n%2 == 1);
         fListEx1.push_back(TXmlEx1());
         fListEx1Ptr.push_back(new TXmlEx1);

         fDequeDouble.push_back(n*4);
         fDequeBool.push_back(n%2 == 1);
         fDequeEx1.push_back(TXmlEx1());
         fDequeEx1Ptr.push_back(new TXmlEx1);

         fMapIntDouble[n] = n*5;
         fMapIntEx1Ptr[n] = new TXmlEx1;
         fMultimapIntDouble.insert(std::pair<int,double>(n,n*6));
   //      fStackDouble.push(n*7);

         fSetDouble.insert(n);
         fMultisetDouble.insert(n);
      }

      for (int n=0;n<3;n++) {
        TXmlEx1 *ex1 = new TXmlEx1;
        fVectEx1Ptr.push_back(ex1);
        fVectEx1Ptr.push_back(ex1);
        fVectEx1Ptr.push_back(ex1);
      }

      for (int n=0;n<5;n++) {
        fVectString.push_back("string as content");
        fVectStringPtr.push_back(new std::string("string pointer as content"));
      }

     fStrPtr1 = new std::string("Value of < >  &lt; &gt; string fStrPtr1");
     fStrPtr2 = new std::string("Value of string fStrPtr2");

     fStrPtrArr[0] = new std::string("value of fStrPtrArr[0]");
     fStrPtrArr[1] = new std::string("value of fStrPtrArr[1]");
     fStrPtrArr[2] = new std::string("value of fStrPtrArr[2]");

     fVectPtrDouble = new std::vector<double>;
     for (int n=0;n<10;n++)
        fVectPtrDouble->push_back(n*3);

     for (int i=0;i<2;i++) {
        fVectPtrDoubleArr[i] = new std::vector<double>;
        for (int n=0;n<10;n++)
           fVectPtrDoubleArr[i]->push_back(n*3);
     }
   }


   virtual ~TXmlEx7()
   {
   }

   void Print()
   {
      std::cout << "Do not print everything, just something..." << std::endl;


      std::cout << " fStr1 = " << fStr1 << std::endl;

      std::cout << " fStrArr[1] = " << fStrArr[1] << std::endl;

      std::cout << " fVectEx1.back().Print()" << std::endl;
      fVectEx1.back().Print();

      std::cout << " fVectEx1Ptr.size() = " << fVectEx1Ptr.size() << std::endl;

      std::cout << " fVectStringPtr.size() = " << fVectStringPtr.size() << std::endl;

      std::cout << " fMapIntEx1Ptr.size() = " << fMapIntEx1Ptr.size() << std::endl;
   }
};

// _______________________________________________________________

class TXmlEx8 : public std::vector<int> {
   public:
     int                    fInt;
     std::string            fStdString;

   TXmlEx8(bool setvalues = false)
   {
      if (setvalues) {
         for (int n=0;n<10;n++)
           push_back(n*14);
         fInt = 12345;
         fStdString = "Value of STL string";
      }

   }
   virtual ~TXmlEx8() {}

   void Print()
   {
      std::cout << "vector size = " << size() << std::endl;
      for (unsigned n=0;n<size(); n++)
        std::cout << "vector.at(" << n << ") = " << at(n) << std::endl;
      std::cout << " fInt = " << fInt << std::endl;
      std::cout << " fStdString = " << fStdString << std::endl;
   }
};

#endif
