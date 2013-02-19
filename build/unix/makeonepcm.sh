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
find include -name \*.h | sed -e 's|include/|#include "|' -e 's|$|"|' \
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
 -e /TPythia8.h/d \
 -e /TPythia8Decayer.h/d \
 -e /TResponseTable.h/d \
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
> all.h

mv all.h include/allHeaders.h

# generate the pch to test if all includes are consistent
clang++ -x c++-header -Iinclude include/allHeaders.h -o include/allHeaders.h.pch

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
             -follow \
             -name '*LinkDef*.h' -print | \
  grep -v -e '/RooFitCore_LinkDef.h$' | \
  sed -e 's|^|#include "|' -e 's|$|"|' > alldefs.h

mv alldefs.h include/allLinkDef.h

# generate one large pcm
rm -f core/base/src/allDict.* lib/allDict_rdict.pcm
core/utils/src/rootcling_tmp -1 -f core/base/src/allDict.cxx -c -I$srcdir -Iinclude allHeaders.h include/allLinkDef.h

exit $?
