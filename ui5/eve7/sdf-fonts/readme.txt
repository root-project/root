SDF fonts for REve
==================

This directory holds the signed-distance-field atlases REve uses to render text.
Each font is a pair of files:

    <name>.png     the distance field itself
    <name>.js.gz   the glyph metrics (advances, kerning, rectangles)

Both are generated files and are not kept in git. A fresh checkout has no
atlases until something calls REveText::AssertSdfFont().

Generating
----------

REveText::AssertSdfFont(font_name, ttf_path) is the entry point. It does
nothing when both files for font_name are present. Otherwise it builds them
from the TTF with TGLSdfFontMaker from the RGL module, which it calls through
the interpreter so that REve does not link against RGL.

Generation needs three things:

 1. REveManager must already exist, because the font directory is registered
    with it. Called earlier, AssertSdfFont() prints an error and returns
    false; calling it again after REveManager::Create() works.

 2. A real GL context, so generation cannot happen in batch mode. `root.exe -b`
    fails with "TGLWidget::CreateWindow: Display is not set!". Run once with a
    display to populate the directory; after that batch sessions are fine,
    because AssertSdfFont() sees the files and does nothing.

 3. A writable target directory. Two defaults are tried in order, and the
    first writable one is used. A missing directory is created. The first
    default is eve7/sdf-fonts/ under WebGui.RootUi5Path, or under
    $ROOTSYS/ui5 when that is not set. The second is ./sdf-fonts/.
    REveText::SetSdfFontDir(dir, require_write_access) overrides them. Pass
    false for require_write_access when pointing at a directory that is
    already populated.

Which fonts to ask for
----------------------

Only faces that ship in TROOT::GetDataDir()/fonts can be relied on. They
include LiberationMono-Regular and LiberationSerif-Regular, arial and arialbd,
verdana, georgia, comic, comicbd and BlackChancery. ROOT ships no Liberation
Sans in any weight. A system path such as /usr/share/fonts/liberation-sans
ties the code to one distribution's layout.

LiberationSerif-Regular is REveText's default. REveViewer::SetAxesType()
generates it, and the client's Axis3D.js uses it for the viewer axes.

For bold, prefer REveText::SetFontWeight() over a second atlas: weight is applied
by moving the SDF threshold, so it is continuous and needs no extra font. Several
of the faces above have no bold variant in ROOT at all.

See also: REveText, and tutorials/visualisation/eve7/texts.C and texts_grid.C.
