//How to add a function to the class context menu
//Author: Ilka Antcheva

#include "TH1.h"
#include "TClassMenuItem.h"
#include "TList.h"

void poptest1(int a=5, int b = 56);
void poptest2(int a, int b, TObject* c);
void poptest2bis(TObject* c);
int  poptest3();
void poptest4(char* ch);


void customContextMenu()
{
   TH1F *h;
   TH1F *h2;
   TClassMenuItem *n;
   TList *l;

   // Create test histograms
   h = new TH1F("h","Schtroumpf",100,-4,4);
   h->FillRandom("gaus");
   h->Draw();

   h2 = new TH1F("h2","h2",1000,-4,4);
   h2->FillRandom("gaus",30000);

   // Retrieve menu list from TClass
   TClass *cl = h->IsA();
   l = cl->GetMenuList();

   // Add some items to the popup menus
   n = new TClassMenuItem(TClassMenuItem::kPopupUserFunction,cl,
                          "Test object, draw a second h","Draw",h2,"Option_t*");
   l->AddFirst(n);
   n = new TClassMenuItem(TClassMenuItem::kPopupSeparator,cl);
   l->AddFirst(n);

   n = new TClassMenuItem(TClassMenuItem::kPopupUserFunction,cl,
                          "test no 4","poptest4",0,"const char*");
   l->AddFirst(n);
   n = new TClassMenuItem(TClassMenuItem::kPopupUserFunction,cl,
                          "test no 3","poptest3",0,"");
   l->AddFirst(n);
   n = new TClassMenuItem(TClassMenuItem::kPopupUserFunction,cl,
                          "test no 2 bis","poptest2bis",0,"TObject*",2);
   l->AddFirst(n);
   n = new TClassMenuItem(TClassMenuItem::kPopupUserFunction,cl,
                          "test no 2","poptest2",0,"int,int,TObject*",2);
   l->AddFirst(n);
   n = new TClassMenuItem(TClassMenuItem::kPopupUserFunction,cl,
                          "test no 1","poptest1",0,"int,int");
   l->AddFirst(n);
}

void poptest1(int a, int b)
{
   printf("poptest1 %d %d\n",a,b);
}

void poptest2(int a, int b, TObject *c)
{
   // Remark: the object c will not be asked for since it is the selected object
   printf("poptest2 %d %d\n",a,b);
   printf("histo : %lx\n",(Long_t)c);
   printf("title of the histogram : %s\n",((TH1F*)c)->GetTitle());
}

void poptest2bis(TObject *c)
{
   // Remark: the object c will not be asked for since it is the selected object
   printf("poptest2bis\n");
   printf("histo : %lx\n",(Long_t)c);
   printf("title of the histogram : %s\n",((TH1F*)c)->GetTitle());
}

int poptest3()
{
   printf("poptest3\n");
   return 12;
}

void poptest4(const char *ch)
{
   printf("poptest4 %s\n",ch);
}
