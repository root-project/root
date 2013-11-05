#!/bin/sh
#
# Build a single large pcm for the entire basic set of ROOT libraries.
# Script takes as optional argument the source directory path.
#
# Copyright (c) 2013 Rene Brun and Fons Rademakers
# Author: Fons Rademakers, 19/2/2013

srcdir=.
if [ $# -eq 1 ]; then
   srcdir=$1
fi

rm -f include/allHeaders.h include/allHeaders.h.pch include/allLinkDef.h

# create allHeaders.h including all headers from the include directory
#find include -name \*.h | sed -e 's|include/|#include "|' -e 's|$|"|' \
cat $srcdir/build/unix/gminimalHeaders.list | sed -e 's|include/|#include "|' -e 's|$|"|' \
 -e /Bits.h/d \
 -e /CocoaGuiTypes.h/d \
 -e /CocoaPrivate.h/d \
 -e /CocoaUtils.h/d \
 -e /config.h/d \
 -e /FontCache.h/d \
 -e /FTBBox.h/d \
 -e /FTBitmapGlyph.h/d \
 -e /FTCharmap.h/d \
 -e /FTCharToGlyphIndexMap.h/d \
 -e /FTContour.h/d \
 -e /FTExtrdGlyph.h/d \
 -e /FTFace.h/d \
 -e /FTFont.h/d \
 -e /FTGL.h/d \
 -e /FTGLBitmapFont.h/d \
 -e /FTGLExtrdFont.h/d \
 -e /FTGLOutlineFont.h/d \
 -e /FTGLPixmapFont.h/d \
 -e /FTGLPolygonFont.h/d \
 -e /FTGLTextureFont.h/d \
 -e /FTGlyph.h/d \
 -e /FTGlyphContainer.h/d \
 -e /FTLibrary.h/d \
 -e /FTList.h/d \
 -e /FTOutlineGlyph.h/d \
 -e /FTPixmapGlyph.h/d \
 -e /FTPoint.h/d \
 -e /FTPolyGlyph.h/d \
 -e /FTSize.h/d \
 -e /FTTextureGlyph.h/d \
 -e /FTVector.h/d \
 -e /FTVectoriser.h/d \
 -e /glew.h/d \
 -e /glxew.h/d \
 -e /wglew.h/d \
 -e /gl2ps.h/d \
 -e /BinaryOperators.h/d \
 -e /VectorUtil_Cint.h/d \
 -e /MenuLoader.h/d \
 -e /proofdp.h/d \
 -e /QuartzFillArea.h/d \
 -e /QuartzLine.h/d \
 -e /QuartzMarker.h/d \
 -e /QuartzPixmap.h/d \
 -e /QuartzText.h/d \
 -e /QuartzUtils.h/d \
 -e /QuartzWindow.h/d \
 -e /ROOTApplicationDelegate.h/d \
 -e /rootdp.h/d \
 -e /ROOTOpenGLView.h/d \
 -e /rpdconn.h/d \
 -e /rpddefs.h/d \
 -e /rpderr.h/d \
 -e /rpdp.h/d \
 -e /rpdpriv.h/d \
 -e /RtypesCint.h/d \
 -e /RtypesImp.h/d \
 -e /TAtomicCountGcc.h/d \
 -e /TAtomicCountPthread.h/d \
 -e /TChair.h/d \
 -e /TColumnView.h/d \
 -e /TGenericTable.h/d \
 -e /TGLContextPrivate.h/d \
 -e /TGLIncludes.h/d \
 -e /TGLWSIncludes.h/d \
 -e /TIndexTable.h/d \
 -e /TMemStatHook.h/d \
 -e /TMemStatMng.h/d \
 -e /TMemStatShow.h/d \
 -e /TMetaUtils.h/d \
 -e /TMonaLisaWriter.h/d \
 -e /TMySQLResult.h/d \
 -e /TMySQLStatement.h/d \
 -e /TMySQLRow.h/d \
 -e /TMySQLServer.h/d \
 -e /TODBCResult.h/d \
 -e /TODBCRow.h/d \
 -e /TODBCServer.h/d \
 -e /TODBCStatement.h/d \
 -e /TOracleResult.h/d \
 -e /TOracleRow.h/d \
 -e /TOracleServer.h/d \
 -e /TOracleStatement.h/d \
 -e /TPgSQLResult.h/d \
 -e /TPgSQLRow.h/d \
 -e /TPgSQLServer.h/d \
 -e /TPgSQLStatement.h/d \
 -e /TPythia8.h/d \
 -e /TPythia8Decayer.h/d \
 -e /TResponseTable.h/d \
 -e /TGQt.h/d \
 -e /TObjectExecute.h/d \
 -e /TQMimeTypes.h/d \
 -e /TQUserEvent.h/d \
 -e /TQtApplication.h/d \
 -e /TQtBrush.h/d \
 -e /TQtCanvasPainter.h/d \
 -e /TQtClientFilter.h/d \
 -e /TQtClientGuard.h/d \
 -e /TQtClientWidget.h/d \
 -e /TQtEmitter.h/d \
 -e /TQtEvent.h/d \
 -e /TQtEventQueue.h/d \
 -e /TQtLock.h/d \
 -e /TQtLockGuard.h/d \
 -e /TQtMarker.h/d \
 -e /TQtPadFont.h/d \
 -e /TQtPen.h/d \
 -e /TQtRConfig.h/d \
 -e /TQtRootApplication.h/d \
 -e /TQtRootSlot.h/d \
 -e /TQtSymbolCodec.h/d \
 -e /TQtTimer.h/d \
 -e /TQtUtil.h/d \
 -e /TQtWidget.h/d \
 -e /TVirtualX.interface.h/d \
 -e /TWaitCondition.h/d \
 -e /TQApplication.h/d \
 -e /TQCanvasImp.h/d \
 -e /TQCanvasMenu.h/d \
 -e /TQRootApplication.h/d \
 -e /TQRootCanvas.cw/d \
 -e /TQRootCanvas.h/d \
 -e /TQRootDialog.h/d \
 -e /TQRootGuiFactory.h/d \
 -e /TQtRootGuiFactory.h/d \
 -e /TTable.h/d \
 -e /TTable3Points.h/d \
 -e /TTableDescriptor.h/d \
 -e /TTableIter.h/d \
 -e /TTableMap.h/d \
 -e /TTablePadView3D.h/d \
 -e /TTablePoints.h/d \
 -e /TTableSorter.h/d \
 -e /Ttypes.h/d \
 -e /TXNetSystem.h/d \
 -e /TXSocket.h/d \
 -e /TXSocketHandler.h/d \
 -e /TXUnixSocket.h/d \
 -e /TX11GL.h/d \
 -e /Windows4Root.h/d \
 -e /X11Buffer.h/d \
 -e /X11Colors.h/d \
 -e /X11Drawable.h/d \
 -e /X11Events.h/d \
 -e /x3d.h/d \
 -e /ZIP.h/d \
 -e /ZDeflate.h/d \
 -e /ZTrees.h/d \
 -e /TGX11.h/d \
 -e /TGX11TTF.h/d \
 -e /crc32.h/d \
 -e /deflate.h/d \
 -e /inffast.h/d \
 -e /inffixed.h/d \
 -e /inflate.h/d \
 -e /inftrees.h/d \
 -e /trees.h/d \
 -e /zconf.h/d \
 -e /zlib.h/d \
 -e /gzguts.h/d \
 -e /zutil.h/d \
 -e /TGCocoa.h/d \
 -e /TGQuartz.h/d \
 -e /THbook.*.h/d \
 -e /TMacOSXSystem.h/d \
 -e /X11Atoms.h/d \
 -e /XLFDParser.h/d \
> all.h

echo '#include "cling/Interpreter/DynamicLookupRuntimeUniverse.h"' >> all.h

mv all.h include/allHeaders.h

cxxflags="-D__CLING__ -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -Iinclude -Ietc -Ietc/cling"
# generate the pch to test if all includes are consistent
# clang++ -x c++-header $cxxflags include/allHeaders.h -o include/allHeaders.h.pch

err=$?
if [ $err -ne 0 ]; then
   echo "Failed to make one pch"
   exit $err
fi

# create one large allLinkDef.h
rm -f include/allLinkDef.h

# create allLinkDef.h including all LinkDefs
find $srcdir -path '*/test' -prune -o \
             -path '*/roottest' -prune -o \
             -path '*/include' -prune -o \
             -path '*/roofit/*' -prune -o \
             -path '*/io/castor/*' -prune -o \
             -path '*/io/xmlparser/*' -prune -o \
             -path '*/io/chirp/*' -prune -o \
             -path '*/io/dcache/*' -prune -o \
             -path '*/io/gfal/*' -prune -o \
             -path '*/io/hdfs/*' -prune -o \
             -path '*/io/rfio/*' -prune -o \
             -path '*/net/alien/*' -prune -o \
             -path '*/proof/proofx/*' -prune -o \
             -path '*/core/meta/inc/LinkDefCling.h' -prune -o \
             -path '*/core/winnt/*' -prune -o \
             -path '*/sql/*' -prune -o \
             -path '*/bindings/pyroot/*' -prune -o \
             -path '*/bindings/ruby/*' -prune -o \
             -path '*/core/macosx/*' -prune -o \
             -path '*/geom/gdml/*' -prune -o \
             -path '*/geom/geocad/*' -prune -o \
             -path '*/graf2d/asimage/*' -prune -o \
             -path '*/graf2d/cocoa/*' -prune -o \
             -path '*/graf2d/gviz/*' -prune -o \
             -path '*/graf2d/qt/*' -prune -o \
             -path '*/graf2d/fitsio/*' -prune -o \
             -path '*/graf2d/win32gdk/*' -prune -o \
             -path '*/graf2d/x11/*' -prune -o \
             -path '*/graf2d/x11ttf/*' -prune -o \
             -path '*/graf3d/eve/*' -prune -o \
             -path '*/graf3d/gl/*' -prune -o \
             -path '*/graf3d/gviz3d/*' -prune -o \
             -path '*/graf3d/x3d/*' -prune -o \
             -path '*/gui/qtgsi/*' -prune -o \
             -path '*/gui/qtroot/*' -prune -o \
             -path '*/hist/hbook/*' -prune -o \
             -path '*/math/fftw/*' -prune -o \
             -path '*/math/genetic/*' -prune -o \
             -path '*/math/genvector/*' -prune -o \
             -path '*/math/mathmore/*' -prune -o \
             -path '*/math/minuit2/*' -prune -o \
             -path '*/math/unuran/*' -prune -o \
             -path '*/misc/memstat/*' -prune -o \
             -path '*/misc/table/*' -prune -o \
             -path '*/montecarlo/pythia6/*' -prune -o \
             -path '*/montecarlo/pythia8/*' -prune -o \
             -path '*/montecarlo/vmc/*' -prune -o \
             -path '*/net/auth/*' -prune -o \
             -path '*/net/bonjour/*' -prune -o \
             -path '*/net/glite/*' -prune -o \
             -path '*/net/krb5auth/*' -prune -o \
             -path '*/net/ldap/*' -prune -o \
             -path '*/net/monalisa/*' -prune -o \
             -path '*/net/net/*' -prune -o \
             -path '*/net/netx/*' -prune -o \
             -path '*/tree/treeviewer/*' -prune -o \
             -path '*/tmva/*' -prune -o \
             -follow \
             -name '*LinkDef*.h' -print | \
  grep -v -e 'base/inc/LinkDef.h$' -e '/RooFitCore_LinkDef.h$' -e "$srcdir/[[:alnum:]]*LinkDef*.h$" | \
  sed -e 's|^|#include "|' -e 's|$|"|' > alldefs.h

mv alldefs.h include/allLinkDef.h

# generate one large pcm
rm -f allDict.* lib/allDict_rdict.pc*
touch etc/allDict.cxx.h
core/utils/src/rootcling_tmp -1 -f etc/allDict.cxx -c $cxxflags -I$srcdir allHeaders.h include/allLinkDef.h
res=$?
if [ $res -eq 0 ] ; then
  mv etc/allDict_rdict.pch etc/allDict.cxx.pch
  res=$?

  # actually we won't need the allDict.[h,cxx] files
  #rm -f allDict.*
fi

exit $res
