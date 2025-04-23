// BNoClassDef.C

#include "BNoClassDef.h"

#include <cstring>
#include <iostream>

using namespace std;

B::B()
: fName(0), fTitle(0), fX(10), fY(20.0)
{
  fName = (char*) new char[1];
  if (fName) {
    fName[0] = '\0';
  }
  fTitle = (char*) new char[1];
  if (fTitle) {
    fTitle[0] = '\0';
  }
}

B::B(char const* name, char const* title, int x, double y)
: fName(0), fTitle(0), fX(x), fY(y)
{
  int len = strlen(name);
  fName = (char*) new char[len+1];
  if (fName) {
    strcpy(fName, name);
  }
  len = strlen(title);
  fTitle = (char*) new char[len+1];
  if (fTitle) {
    strcpy(fTitle, title);
  }
}

B::B(const B& b)
: fName(0), fTitle(0), fX(b.fX), fY(b.fY)
{
  if (b.fName) {
    int len = strlen(b.fName);
    fName = new char[len+1];
    if (fName) {
      strcpy(fName, b.fName);
    }
  }
  if (b.fTitle) {
    int len = strlen(b.fTitle);
    fTitle = new char[len+1];
    if (fTitle) {
      strcpy(fTitle, b.fTitle);
    }
  }
}

B& B::operator=(const B& b) {
  if (&b != this) {
    fX = b.fX;
    fY = b.fY;
    if (strcmp(b.fName, fName)) {
      delete[] fName;
      fName = 0;
      if (b.fName) {
        int len = strlen(b.fName);
        fName = new char[len+1];
        if (fName) {
          strcpy(fName, b.fName);
        }
      }
    }
    if (strcmp(b.fTitle, fTitle)) {
      delete[] fTitle;
      fTitle = 0;
      if (b.fTitle) {
        int len = strlen(b.fTitle);
        fTitle = new char[len+1];
        if (fTitle) {
          strcpy(fTitle, b.fTitle);
        }
      }
    }
  }
  return *this;
}

B::~B() {
  if (fName) {
    delete[] fName;
  }
  if (fTitle) {
    delete[] fTitle;
  }
}

void B::SetName(char const* name)
{
  if (strcmp(name, fName)) {
    if (fName) {
      delete[] fName;
      fName = 0;
    }
    if (name) {
      int len = strlen(name);
      fName = new char[len+1];
      if (fName) {
        strcpy(fName, name);
      }
    }
  }
}

void B::SetTitle(char const* title)
{
  if (strcmp(title, fTitle)) {
    if (fTitle) {
      delete[] fTitle;
      fTitle = 0;
    }
    if (title) {
      int len = strlen(title);
      fTitle = new char[len+1];
      if (fTitle) {
        strcpy(fTitle, title);
      }
    }
  }
}

void B::repr() const
{
  if (fName) {
    cout << fName;
  }
  cout << " ";
  if (fTitle) {
    cout << fTitle;
  }
  cout  << " fX: " << fX << " fY: " << fY << endl;
}

