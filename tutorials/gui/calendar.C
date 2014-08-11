// Author: Valeriy Onuchin   24/08/2007
//
// This macro gives an example of how to use html widget
// to display tabular data.
//
// To run it do either:
// .x calendar.C
// .x calendar.C++


#include "TDatime.h"
#include "TTimeStamp.h"
#include "TGComboBox.h"
#include "TGNumberEntry.h"
#include "TGLabel.h"
#include "TGColorSelect.h"
#include "TGHtml.h"
#include "TApplication.h"
#include "TROOT.h"
#include "TColor.h"


/////////////////////////// HTML calendar //////////////////////////////////////

TString monthNames[12] = {"January", "February", "March", "April",
                          "May", "June", "July", "August", "September",
                          "October", "November", "December"};

////////////////////////////////////////////////////////////////////////////////
class HtmlDayName {
public:                 // make them public for shorter code
   TString fDay;        // day name, e.g. "Sunday"
   TString fAlign;      // name align inside table cell
   TString fBgColor;    // cell background color
   TString fFontSize;   // text font size
   TString fFontColor;  // text color
   TString fHtml;       // HTML output code

public:
   HtmlDayName(const char *day);
   virtual ~HtmlDayName() {}

   TString Html() const { return fHtml; }

   ClassDef(HtmlDayName, 0);
};

//______________________________________________________________________________
HtmlDayName::HtmlDayName(const char *day) : fDay(day), fAlign("middle"),
   fBgColor("#000000"), fFontSize("4"), fFontColor("#FFFFFF")
{
   // ctor.

   fHtml += "<TH  width=14%";
   fHtml += " align=" + fAlign;
   fHtml += " bgcolor=" + fBgColor + ">";
   fHtml += "<font size=" + fFontSize;
   fHtml += " color=" + fFontColor + ">";
   fHtml += fDay;
   fHtml += "</font></TH>\n";
}


////////////////////////////////////////////////////////////////////////////////
class HtmlMonthTable {
public:                    // make them public for shorter code
   Int_t    fYear;         // year
   Int_t    fMonth;        // month

   TString  fBorder;       // border width
   TString  fBgColor;      // background color
   TString  fCellpadding;  // cell padding
   TString  fCellFontSize; // cell font size
   TString  fCellBgcolor;  // cell background color
   TString  fTodayColor;   // background color of cell correspondent today date

   TDatime  fToday;        // today's date
   TString  fHtml;         // HTML output code

   void Build();
   void BuildDayNames();
   void BuildDays();

public:
   HtmlMonthTable(Int_t year, Int_t month);
   virtual ~HtmlMonthTable() {}

   void SetDate(Int_t year, Int_t month);
   TString Html() const { return fHtml; }

   ClassDef(HtmlMonthTable, 0);
};

//______________________________________________________________________________
HtmlMonthTable::HtmlMonthTable(Int_t year, Int_t month) : fYear(year),
   fMonth(month), fBorder("2"), fBgColor("#aaaaaa"), fCellpadding("5"),
   fCellFontSize("3"), fCellBgcolor("#eeeeee"), fTodayColor("#ffff00")
{
   // Constructor.

   Build();
}

//______________________________________________________________________________
void HtmlMonthTable::SetDate(Int_t year, Int_t month)
{
   // Set date.

   fYear = year;
   fMonth = month;
   Build();
}

//______________________________________________________________________________
void HtmlMonthTable::Build()
{
   // Build HTML code.

   fHtml = "<TABLE width=100%";
   fHtml += " border=" + fBorder;
   fHtml += " bgcolor=" + fBgColor;
   fHtml += " cellpadding=" + fCellpadding;
   fHtml += "><TBODY>";

   BuildDayNames();
   BuildDays();

   fHtml += "</TBODY></TABLE>\n";
}

//______________________________________________________________________________
void HtmlMonthTable::BuildDayNames()
{
   // Build table header with day names.

   fHtml += "<TR>";
   fHtml += HtmlDayName("Sunday").Html();
   fHtml += HtmlDayName("Monday").Html();
   fHtml += HtmlDayName("Tuesday").Html();
   fHtml += HtmlDayName("Wednesday").Html();
   fHtml += HtmlDayName("Thursday").Html();
   fHtml += HtmlDayName("Friday").Html();
   fHtml += HtmlDayName("Saturday").Html();
   fHtml += "</TR>\n";
}

//______________________________________________________________________________
void HtmlMonthTable::BuildDays()
{
   // Build part of table with day numbers.

   static Int_t maxdays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

   Int_t maxday = maxdays[fMonth-1];
   if ((fMonth == 2) && TTimeStamp::IsLeapYear(fYear)) maxday = 29;

   Int_t first = TTimeStamp::GetDayOfWeek(1, fMonth, fYear);

   // fill html table
   for (int week = 0; week < 6; week++) {
      fHtml += "<TR>";

      for (int weekday = 0; weekday < 7; weekday++) {//
         Int_t day = week*7 + weekday - first + 1;

         if ((day > maxday) && !weekday) break; //

         fHtml += "<TD align=left width=14% ";

         // hightlight today's cell
         if ((fToday.GetYear() == fYear) &&
             (fToday.GetMonth() == fMonth) &&
             (fToday.GetDay() == day)) {
            fHtml += " bgcolor=" + fTodayColor;
         } else {
            fHtml += " bgcolor=" + fCellBgcolor;
         }
         fHtml += ">";

         //skip week days which are not of this month
         if ((day <= 0) || (day > maxday)) {
            fHtml += "&nbsp;</TD>";
            continue;
         }

         fHtml += "<font size=" + fCellFontSize + ">";
         fHtml += Form("%d", day);
         fHtml += "</font></TD>\n";
      }
      fHtml += "</TR>\n";
   }
}


////////////////////////////////////////////////////////////////////////////////
class HtmlCalendar {
public:                          // make them public for shorter code
   Int_t          fYear;         // year
   Int_t          fMonth;        // month
   HtmlMonthTable fMonthTable;   // HTML table presenting month days
   TString        fHeader;       // HTML header
   TString        fFooter;       // HTML footer
   TString        fHtml;         // output HTML string
   TString        fTitle;        // page title

   void MakeHeader();
   void MakeFooter();

public:
   HtmlCalendar(Int_t year, Int_t month);
   virtual ~HtmlCalendar() {}

   void SetDate(Int_t year, Int_t month);
   TString Html() const { return fHtml; }

   ClassDef(HtmlCalendar, 0);
};

//______________________________________________________________________________
HtmlCalendar::HtmlCalendar(Int_t year, Int_t month) : fMonthTable(year, month)
{
   // Constructor.

   fYear = year;
   fMonth = month;

   MakeHeader();
   MakeFooter();

   fHtml = fHeader;
   fHtml += fMonthTable.Html();
   fHtml += fFooter;
}

//______________________________________________________________________________
void HtmlCalendar::SetDate(Int_t year, Int_t month)
{
   // Create calendar for month/year.

   fYear = year;
   fMonth = month;

   fMonthTable.SetDate(year, month);
   MakeHeader();
   MakeFooter();
   fHtml = fHeader;
   fHtml += fMonthTable.Html();
   fHtml += fFooter;
}

//______________________________________________________________________________
void HtmlCalendar::MakeHeader()
{
   // Make HTML header.

   fTitle = monthNames[fMonth-1] + Form(" %d", fYear);
   fHeader = "<html><head><title>";
   fHeader += fTitle;
   fHeader += "</title></head><body>\n";
   fHeader += "<center><H2>" + fTitle + "</H2></center>";
}

//______________________________________________________________________________
void HtmlCalendar::MakeFooter()
{
   // Make HTML footer.

   fFooter = "<br><p><br><center><strong><font size=2 color=#2222ee>";
   fFooter += "Example of using Html widget to display tabular data.";
   fFooter += "</font></strong></center></body></html>";
}

//////////////////////// end of HTML calendar //////////////////////////////////



class CalendarWindow {
private:
   TGMainFrame    *fMain;       // main frame
   HtmlCalendar   *fHtmlText;   // calendar HTML table
   TGHtml         *fHtml;       // html widget to display HTML calendar
   TGComboBox     *fMonthBox;   // month selector
   TGNumberEntry  *fYearEntry;  // year selector
   TGNumberEntry  *fFontEntry;  // font size selector
   TGColorSelect  *fTableColor; // selector of background color of table
   TGColorSelect  *fCellColor;  // selector of background color of table's cells

public:
   CalendarWindow();
   virtual ~CalendarWindow();

   void UpdateHTML();

   ClassDef(CalendarWindow, 0);
};


//______________________________________________________________________________
CalendarWindow::~CalendarWindow()
{
   // Destructor.

   delete fHtmlText;
   delete fMain;
}

//______________________________________________________________________________
CalendarWindow::CalendarWindow()
{
   // Main  window.

   fMain = new TGMainFrame(gClient->GetRoot(), 10, 10, kVerticalFrame);
   fMain->SetCleanup(kDeepCleanup); // delete all subframes on exit

   // Controls
   TGHorizontalFrame *controls = new TGHorizontalFrame(fMain);
   fMain->AddFrame(controls, new TGLayoutHints(kLHintsCenterX, 1, 1, 1, 1));

   // generate HTML calendar table
   TDatime today;
   fHtmlText = new HtmlCalendar(today.GetYear(), today.GetMonth());

   // create HTML widget
   fHtml = new TGHtml(fMain, 1, 1);
   fMain->AddFrame(fHtml, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
                                            5, 5, 2, 2));

   // parse HTML context of HTML calendar table
   fHtml->ParseText((char*)fHtmlText->Html().Data());

   TGLabel *dateLabel = new TGLabel(controls, "Date:");
   controls->AddFrame(dateLabel, new TGLayoutHints(kLHintsLeft|kLHintsCenterY,
                                                   5, 2, 2, 2));

   //
   fMonthBox = new TGComboBox(controls);
   for (int i = 0; i < 12; i++) {
      fMonthBox->AddEntry(monthNames[i].Data(), i+1);
   }
   fMonthBox->Select(today.GetMonth());
   controls->AddFrame(fMonthBox, new TGLayoutHints(kLHintsLeft, 5, 5, 2, 2));

   fYearEntry = new TGNumberEntry(controls, today.GetYear(), 5, -1,
                                  TGNumberFormat::kNESInteger,
                                  TGNumberFormat::kNEAPositive,
                                  TGNumberFormat::kNELLimitMin, 1995);
   controls->AddFrame(fYearEntry, new TGLayoutHints(kLHintsLeft, 5, 5, 2, 2));

   fMonthBox->Resize(100, fYearEntry->GetHeight());

   TGLabel *fontLabel = new TGLabel(controls, "Font Size:");
   controls->AddFrame(fontLabel, new TGLayoutHints(kLHintsLeft|kLHintsCenterY,
                                                   30, 2, 2, 2));

   Int_t fontsize = atoi(fHtmlText->fMonthTable.fCellFontSize.Data());
   fFontEntry = new TGNumberEntry(controls, fontsize, 2, -1,
                                  TGNumberFormat::kNESInteger,
                                  TGNumberFormat::kNEAPositive,
                                  TGNumberFormat::kNELLimitMax, 0, 7);
   controls->AddFrame(fFontEntry, new TGLayoutHints(kLHintsLeft, 5, 5, 2, 2));

   TGLabel *tableLabel = new TGLabel(controls, "Table:");
   controls->AddFrame(tableLabel, new TGLayoutHints(kLHintsLeft|kLHintsCenterY,
                                                    5, 2, 2, 2));

   Pixel_t color;

   gClient->GetColorByName(fHtmlText->fMonthTable.fBgColor.Data(), color);
   fTableColor = new TGColorSelect(controls, color);
   controls->AddFrame(fTableColor, new TGLayoutHints(kLHintsLeft|kLHintsCenterY,
                                                     5, 2, 2, 2));

   TGLabel *cellLabel = new TGLabel(controls, "Cell:");
   controls->AddFrame(cellLabel, new TGLayoutHints(kLHintsLeft|kLHintsCenterY,
                                                   5, 2, 2, 2));

   gClient->GetColorByName(fHtmlText->fMonthTable.fCellBgcolor.Data(), color);
   fCellColor = new TGColorSelect(controls, color);
   controls->AddFrame(fCellColor, new TGLayoutHints(kLHintsLeft|kLHintsCenterY,
                                                    5, 2, 2, 2));

   // connect signals
   fMonthBox->Connect("Selected(Int_t)", "CalendarWindow", this,
                      "UpdateHTML()");
   fYearEntry->GetNumberEntry()->Connect("TextChanged(char*)", "CalendarWindow",
                                         this, "UpdateHTML()");
   fFontEntry->GetNumberEntry()->Connect("TextChanged(char*)", "CalendarWindow",
                                         this, "UpdateHTML()");
   fTableColor->Connect("ColorSelected(Pixel_t)", "CalendarWindow", this,
                        "UpdateHTML()");
   fCellColor->Connect("ColorSelected(Pixel_t)", "CalendarWindow", this,
                       "UpdateHTML()");

   // terminate ROOT session when window is closed
   fMain->Connect("CloseWindow()", "TApplication", gApplication, "Terminate()");
   fMain->DontCallClose();

   fMain->MapSubwindows();
   fMain->Resize(600, 333);

   // set  minimum size of main window
   fMain->SetWMSizeHints(controls->GetDefaultWidth(), fMain->GetDefaultHeight(),
                         1000, 1000, 0 ,0);

   TString title = "Calendar for ";
   title += fHtmlText->fTitle;
   fMain->SetWindowName(title.Data());
   fMain->MapRaised();
}

//______________________________________________________________________________
void CalendarWindow::UpdateHTML()
{
   // Update HTML table on user's input.

   Int_t month = fMonthBox->GetSelected();
   Int_t year = atoi(fYearEntry->GetNumberEntry()->GetText());
   fHtmlText->fMonthTable.fCellFontSize = fFontEntry->GetNumberEntry()->GetText();

   Pixel_t pixel = 0;
   TColor *color = 0;

   // table background
   pixel = fTableColor->GetColor();
   color = gROOT->GetColor(TColor::GetColor(pixel));

   if (color) {
      fHtmlText->fMonthTable.fBgColor = color->AsHexString();
   }

   // cell background
   pixel = fCellColor->GetColor();
   color = gROOT->GetColor(TColor::GetColor(pixel));

   if (color) {
      fHtmlText->fMonthTable.fCellBgcolor = color->AsHexString();
   }

   // update HTML context
   fHtmlText->SetDate(year, month);

   // parse new HTML context of HTML calendar table
   fHtml->Clear();
   fHtml->ParseText((char*)fHtmlText->Html().Data());
   fHtml->Layout();

   // update window title
   TString title = "Calendar for ";
   title += fHtmlText->fTitle;
   fMain->SetWindowName(title.Data());
}

////////////////////////////////////////////////////////////////////////////////
void calendar()
{
   // Main program.

   new CalendarWindow();
}



