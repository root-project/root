#ifndef ExampleWidget_h
#define ExampleWidget_h

#include <QWidget>
#include "ui_ExampleWidget.h"

class TH1F;

class ExampleWidget : public QWidget, public Ui::ExampleWidget
{
   Q_OBJECT

   protected:

      TH1F *fHisto{nullptr};

   public:

      ExampleWidget(QWidget *parent = 0, const char* name = 0);

      virtual ~ExampleWidget();

   public slots:

      void InfoButton_clicked();
      void ExitButton_clicked();

};

#endif
