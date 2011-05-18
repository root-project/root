#ifndef ROOT_TQTCANVASPAINTER
#define ROOT_TQTCANVASPAINTER

/////////////////////////////////////////////////////////////////////////////
//                                                                         //
// TQtCanvasPainter                                                        //
//                                                                         //
// TQtCanvasPainter is abstarct visitor interface                          //
// to customize TQtWidget painting                                         //
// It allows the arbitrary low level Qt painting onto the TQtWidget face   //
// on the top of  TCanvas image                                            //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////

#include <QtCore/QObject>
 
class TQtCanvasPainter : public QObject 
{
    protected:
       TQtCanvasPainter(){}
   	public:
       TQtCanvasPainter(QObject *p) : QObject(p) {}
	   virtual ~TQtCanvasPainter() {}
	   virtual void paintEvent(QPainter &painter, QPaintEvent *e=0) = 0;
};

#endif
