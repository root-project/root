/// \file
/// \ingroup tutorial_xml
///
/// ROOT implementation of a XML DOM Parser
///
/// This is an example of how Dom Parser works. It will parse the xml file
/// (person.xml) to the Person object.
/// A DTD validation will be run on this example.
///
/// To run this program
/// ~~~{.cpp}
/// .x DOMParsePerson.C+
/// ~~~
///
/// Requires: person.xml and person.dtd
///
/// \macro_output
/// \macro_code
///
/// \author Sergey Linev


#include <Riostream.h>
#include <TDOMParser.h>
#include <TXMLAttr.h>
#include <TXMLNode.h>
#include <TList.h>


class Date {
public:
   Date() : day(0), month(0), year(0) { }
   Date(Int_t d, Int_t m, Int_t y) : day(d), month(m), year(y) { }
   Int_t GetDay() const { return day; }
   Int_t GetMonth() const { return month; }
   Int_t GetYear() const { return year; }
   void SetDay(Int_t d) { day=d; }
   void SetMonth(Int_t m) { month=m;}
   void SetYear(Int_t y) { year=y;}
private:
   Int_t day;
   Int_t month;
   Int_t year;
};

class Address {
public:
   Address() { }
   Address(TString s, TString p, TString c) :
      street(s), postalCode(p), country(c) { }
   TString GetStreet() const { return street; }
   TString GetPostalCode() const { return postalCode; }
   TString GetCountry() const { return country; }
   void SetStreet(const TString &s) { street = s; }
   void SetPostalCode(const TString &p) { postalCode = p; }
   void SetCountry(const TString &c) { country = c; }
private:
   TString street;
   TString postalCode;
   TString country;
};

class Person : public TObject {
public:
   Person() { }
   Person(Int_t i, TString f, TString l, Char_t g, Date * d, Address * a) :
      id(i), firstName(f), lastName(l), gender(g), dateOfBirth(d), address(a){ }

   ~Person() {
      delete dateOfBirth;
      delete address;
   }

   TString GetFirstName() const { return firstName; }
   TString GetLastName() const { return lastName; }
   Char_t GetGender() const { return gender; }
   Date *GetDate() const { return dateOfBirth; }
   Address *GetAddress() const { return address; }
   Int_t GetID() const { return id; }

   friend ostream & operator << (ostream& out, const Person& p) {
      out << "ID: " << p.id << endl;
      out << "First name: " << p.firstName << endl;
      out << "Last name: " << p.lastName << endl;
      out << "Sex: " << p.gender << endl;
      out << "Date of birth: " << p.dateOfBirth->GetDay() << "/"
          << p.dateOfBirth->GetMonth() << "/"
          << p.dateOfBirth->GetYear() << endl;
      out << "Address: " << p.address->GetStreet() << endl;
      out << "\t" << p.address->GetPostalCode() << endl;
      out << "\t" << p.address->GetCountry() << endl;
      out << endl;
      return out;
   }

private:
   Int_t     id;
   TString   firstName;
   TString   lastName;
   Char_t    gender;
   Date     *dateOfBirth;
   Address  *address;
};

class PersonList {
public:
   PersonList() {
      listOfPerson = new TList();
   }

   Int_t ParseFile(TString filename) {
      TDOMParser *domParser = new TDOMParser();
      Int_t parsecode = domParser->ParseFile(filename);

      if (parsecode < 0) {
         cerr << domParser->GetParseCodeMessage(parsecode) << endl;
         return -1;
      }

      TXMLNode * node = domParser->GetXMLDocument()->GetRootNode();

      ParsePersonList(node);

      return 0;
   }

   void ParsePersonList(TXMLNode *node) {
      for (; node; node = node->GetNextNode()) {
         if (node->GetNodeType() == TXMLNode::kXMLElementNode) { // Element Node
            if (strcmp(node->GetNodeName(), "Person") == 0) {
               Int_t id=0;
               if (node->HasAttributes()) {
                  TList *attrList = node->GetAttributes();
                  TXMLAttr *attr = 0;
                  TIter next(attrList);
                  while ((attr=(TXMLAttr*)next())) {
                     if (strcmp(attr->GetName(), "ID") == 0) {
                        id = atoi(attr->GetValue());
                        break;
                     }
                  }
               }
               listOfPerson->Add(ParsePerson(node->GetChildren(), id));
            }
         }
         ParsePersonList(node->GetChildren());
      }
   }

   Date *ParseDate(TXMLNode *node) {
      Int_t d=0, m=0, y=0;
      for ( ; node; node = node->GetNextNode()) {
         if (node->GetNodeType() == TXMLNode::kXMLElementNode) { // Element Node
            if (strcmp(node->GetNodeName(), "Day") == 0) {
               d = atoi(node->GetText());
            }
            if (strcmp(node->GetNodeName(), "Month") == 0) {
               m = atoi(node->GetText());
            }
            if (strcmp(node->GetNodeName(), "Year") == 0) {
               y = atoi(node->GetText());
            }
         }
      }
      return new Date(d, m, y);
   }

   Address *ParseAddress(TXMLNode *node) {
      TString s, p, c;
      for( ; node!=NULL; node = node->GetNextNode()){
         if (node->GetNodeType() == TXMLNode::kXMLElementNode) { // Element Node
            if (strcmp(node->GetNodeName(), "Street") == 0) {
               s = node->GetText();
            }
            if (strcmp(node->GetNodeName(), "PostalCode") == 0) {
               p = node->GetText();
            }
            if (strcmp(node->GetNodeName(), "Country") == 0) {
               c = node->GetText();
            }
         }
     }
     return new Address(s, p, c);
   }

   Person *ParsePerson(TXMLNode *node, Int_t id) {
      TString firstName, lastName;
      char gender = ' ';
      Date *date;
      Address *address;

      for ( ; node; node = node->GetNextNode()) {
         if (node->GetNodeType() == TXMLNode::kXMLElementNode) { // Element Node
            if (strcmp(node->GetNodeName(), "FirstName") == 0)
               firstName = node->GetText();
            if (strcmp(node->GetNodeName(), "LastName") == 0)
               lastName = node->GetText();
            if (strcmp(node->GetNodeName(), "Gender") == 0)
               gender = node->GetText()[0];
            if (strcmp(node->GetNodeName(), "DateOfBirth") == 0)
               date = ParseDate(node->GetChildren());
            if (strcmp(node->GetNodeName(), "Address") == 0)
               address = ParseAddress(node->GetChildren());
         }
     }

     return new Person(id, firstName, lastName, gender, date, address);
   }

   friend ostream& operator << (ostream& out, const PersonList & pl) {
      TIter next(pl.listOfPerson);
      Person *p;
      while ((p =(Person*)next())){
         out << *p << endl;
      }
      return out;
   }

   void PrintPerson() {
      TIter next(listOfPerson);
      Person *p;
      while ((p =(Person*)next())) {
         cout << *p << endl;
      }
   }

private:
   Int_t  numberOfPersons;
   TList *listOfPerson;
};


void DOMParsePerson()
{
   PersonList personlist;
   gROOT->ProcessLine(".O 0");
   TString dir = gROOT->GetTutorialDir();
   if (personlist.ParseFile(dir+"/xml/person.xml") == 0)
      cout << personlist << endl;
}
