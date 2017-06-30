#ifndef RootWebView_h
#define RootWebView_h

#include <QWebEngineView>
#include <QWebEnginePage>


class RootWebPage : public QWebEnginePage {
   Q_OBJECT
protected:
   virtual void javaScriptConsoleMessage(QWebEnginePage::JavaScriptConsoleMessageLevel level,
                      const QString &message, int lineNumber, const QString &sourceID);

public:

   RootWebPage(QObject *parent = 0) : QWebEnginePage(parent) {}
   virtual ~RootWebPage() {}

};

class RootWebView : public QWebEngineView {
   Q_OBJECT
protected:


public:
   RootWebView(QWidget* parent = 0);
   virtual ~RootWebView();

};





#endif
