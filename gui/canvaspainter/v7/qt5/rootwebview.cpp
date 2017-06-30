#include "rootwebview.h"



void RootWebPage::javaScriptConsoleMessage(JavaScriptConsoleMessageLevel level,
                      const QString &message, int lineNumber, const QString &sourceID)
{
   QByteArray ba = message.toLatin1();
   QByteArray src = sourceID.toLatin1();

   printf("CONSOLE %s:%d: %s\n", src.data(), lineNumber, ba.data());
}


RootWebView::RootWebView(QWidget* parent) : QWebEngineView(parent)
{
   setPage(new RootWebPage());

   //connect(this, SIGNAL(javaScriptConsoleMessage(JavaScriptConsoleMessageLevel, const QString &, int, const QString &)),
   //        this, SLOT(doConsole(JavaScriptConsoleMessageLevel, const QString &, int, const QString &)));

   // connect(this, &QWebEngineView::javaScriptConsoleMessage, this, &RootWebView::doConsole);
}

RootWebView::~RootWebView()
{

}
