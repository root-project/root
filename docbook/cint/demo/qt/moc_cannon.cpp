/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/****************************************************************************
** CannonField meta object code from reading C++ file 'cannon.h'
**
** Created: Sun Jan 4 19:50:57 2004
**      by: The Qt MOC ($Id: moc_cannon.cpp,v 1.2 2005/11/25 04:58:08 pcanal Exp $)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#define Q_MOC_CannonField
#if !defined(Q_MOC_OUTPUT_REVISION)
#define Q_MOC_OUTPUT_REVISION 8
#elif Q_MOC_OUTPUT_REVISION != 8
#error "Moc format conflict - please regenerate all moc files"
#endif

#include "cannon.h"
#include <qmetaobject.h>
#include <qapplication.h>

#if defined(Q_SPARCWORKS_FUNCP_BUG)
#define Q_AMPERSAND
#else
#define Q_AMPERSAND &
#endif


const char *CannonField::className() const
{
    return "CannonField";
}

QMetaObject *CannonField::metaObj = 0;

void CannonField::initMetaObject()
{
    if ( metaObj )
	return;
    if ( strcmp(QWidget::className(), "QWidget") != 0 )
	badSuperclassWarning("CannonField","QWidget");
    (void) staticMetaObject();
}

#ifndef QT_NO_TRANSLATION
QString CannonField::tr(const char* s)
{
    return ((QNonBaseApplication*)qApp)->translate("CannonField",s);
}

#endif // QT_NO_TRANSLATION
QMetaObject* CannonField::staticMetaObject()
{
    if ( metaObj )
	return metaObj;
    (void) QWidget::staticMetaObject();
#ifndef QT_NO_PROPERTIES
#endif // QT_NO_PROPERTIES
    typedef void(CannonField::*m1_t0)(int);
    m1_t0 v1_0 = Q_AMPERSAND CannonField::setAngle;
    QMetaData *slot_tbl = QMetaObject::new_metadata(1);
    QMetaData::Access *slot_tbl_access = QMetaObject::new_metaaccess(1);
    slot_tbl[0].name = "setAngle(int)";
    slot_tbl[0].ptr = *((QMember*)&v1_0);
    slot_tbl_access[0] = QMetaData::Public;
    typedef void(CannonField::*m2_t0)(int);
    m2_t0 v2_0 = Q_AMPERSAND CannonField::angleChanged;
    QMetaData *signal_tbl = QMetaObject::new_metadata(1);
    signal_tbl[0].name = "angleChanged(int)";
    signal_tbl[0].ptr = *((QMember*)&v2_0);
    metaObj = QMetaObject::new_metaobject(
	"CannonField", "QWidget",
	slot_tbl, 1,
	signal_tbl, 1,
#ifndef QT_NO_PROPERTIES
	0, 0,
	0, 0,
#endif // QT_NO_PROPERTIES
	0, 0 );
    metaObj->set_slot_access( slot_tbl_access );
#ifndef QT_NO_PROPERTIES
#endif // QT_NO_PROPERTIES
    return metaObj;
}

// SIGNAL angleChanged
void CannonField::angleChanged( int t0 )
{
    activate_signal( "angleChanged(int)", t0 );
}
