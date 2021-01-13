#ifndef ExampleWidget_h
#define ExampleWidget_h

#include <QWidget>
#include "ui_ExampleWidget.h"

#include <memory>

class TH1F;
class TH2I;

class ExampleWidget : public QWidget, public Ui::ExampleWidget
{
   Q_OBJECT

   protected:

      TH1F *fHisto{nullptr};  ///< histogram for display in TCanvas
      std::shared_ptr<TH2I> fHisto2; ///< histogram for display in RCanvas

   public:

      ExampleWidget(QWidget *parent = 0, const char* name = 0);

      virtual ~ExampleWidget();

   public slots:

      void InfoButton_clicked();
      void ExitButton_clicked();

};

#endif
