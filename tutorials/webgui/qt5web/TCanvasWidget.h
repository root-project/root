#ifndef TCanvasWidget_H
#define TCanvasWidget_H

#include <QWidget>
#include <QWebEngineView>

class TCanvas;
class TPad;
class TObject;

class TCanvasWidget : public QWidget {

   Q_OBJECT

public:
   TCanvasWidget(QWidget *parent = 0);
   virtual ~TCanvasWidget();

   /// returns canvas shown in the widget
   TCanvas *getCanvas() { return fCanvas; }

signals:

   void CanvasUpdated();

   void SelectedPadChanged(TPad*);

   void PadClicked(TPad*,int,int);

   void PadDblClicked(TPad*,int,int);

public slots:

   void activateEditor(TPad *pad = nullptr, TObject *obj = nullptr);

   void activateStatusLine();

   void setEditorVisible(bool flag = true);

protected:

   void resizeEvent(QResizeEvent *event) override;

   void SetPrivateCanvasFields(bool on_init);

   QWebEngineView *fView{nullptr};  ///< qt webwidget to show

   TCanvas *fCanvas{nullptr};
};

#endif
