SDF fonts for REve
==================

This directory holds signed-distance-field atlases REve uses to render text.
Each font is a pair of files:

    <name>.png     the distance field itself
    <name>.js.gz   the glyph metrics (advances, kerning, rectangles)

Both are generated files and are not kept in git. The fonts shipped with ROOT
are in ../assets/sdf-fonts/, which the build fills from the RenderCore assets
tarball. By default, fonts generated at run time go here.

Lookup
------

The server serves the fonts under sdf-fonts/ from several directories, and
REveText::AssertSdfFont() looks for them in the same order:

 1. The directory of REveText::SetSdfFontDir(dir, require_write_access).
    Without that call it is this directory, eve7/sdf-fonts/ under the ui5
    directory, when it is writable or can be created, else ./sdf-fonts/ in the
    current working directory. The ui5 directory is $ROOTUI5SYS, else
    WebGui.RootUi5Path, else ui5/ in ROOT's data directory, as for the
    server's rootui5sys/. ./sdf-fonts/ is created only when a font is
    generated into it.

 2. The directories of REveText::AddSdfFontDir(dir), in the order they were
    added. Fonts are only read from them.

 3. eve7/assets/sdf-fonts/ under the ui5 directory, the fonts shipped with
    ROOT. It may be read-only.

A file is served from the first directory that has it. The directories can be
set before REveManager::Create().

Generating
----------

REveText::AssertSdfFont(font_name, ttf_path) is the entry point. It does
nothing when the first directory that has either file of font_name has both.
Otherwise it builds them from the TTF with TGLSdfFontMaker from the RGL
module, which it calls through the interpreter so that REve does not link
against RGL. The files are written into the directory of point 1 above, which
is searched first. Requiring both files in one directory keeps a lone file in
a directory searched earlier from being served with the other file of a later
directory, which would be a mismatched pair.

Generation needs a real GL context, so it cannot happen in batch mode.
AssertSdfFont() then warns and returns false. Run once with a display to
populate the directory; after that batch sessions are fine, because
AssertSdfFont() finds the files and does nothing.

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
