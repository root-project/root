#ifndef RCanvasWidget_H
#define RCanvasWidget_H

#include <QWidget>
#include <QWebEngineView>

namespace ROOT {
namespace Experimental {
class RCanvas;
}
}

class RCanvasWidget : public QWidget {

   Q_OBJECT

public:
   RCanvasWidget(QWidget *parent = nullptr);
   virtual ~RCanvasWidget();

   /// returns canvas shown in the widget
   ROOT::Experimental::RCanvas *getCanvas() { return fCanvas; }

protected:

   void resizeEvent(QResizeEvent *event) override;

   QWebEngineView *fView{nullptr};  ///< qt webwidget to show

   ROOT::Experimental::RCanvas *fCanvas{nullptr};
};

#endif
