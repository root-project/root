#ifndef ExampleWidget_h
#define ExampleWidget_h

#include <QWidget>
#include <QWebEngineView>
#include "ui_ExampleWidget.h"

class TH1F;
class TCanvas;

class ExampleWidget : public QWidget, public Ui::ExampleWidget
{
   Q_OBJECT

   public:

      ExampleWidget(QWidget *parent = 0, const char* name = 0);

      virtual ~ExampleWidget();

   protected:
      virtual void resizeEvent(QResizeEvent * e);
      void SetPrivateCanvasFields(TCanvas *canv, bool on_init);

      QWebEngineView *fView{nullptr};
      TCanvas *fCanvas{nullptr};
      TH1F *fHisto{nullptr};

   public slots:

      void InfoButton_clicked();
      void ExitButton_clicked();

};

#endif
