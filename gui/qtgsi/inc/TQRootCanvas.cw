<!DOCTYPE CW><CW>
<customwidgets>
    <customwidget>
        <class>TQRootCanvas</class>
        <header location="global">TQRootCanvas.h</header>
        <sizehint>
            <width>-1</width>
            <height>-1</height>
        </sizehint>
        <container>0</container>
        <sizepolicy>
            <hordata>5</hordata>
            <verdata>5</verdata>
        </sizepolicy>
        <signal>destroyed()</signal>
        <signal>destroyed(QObject*)</signal>
        <signal>SelectedPadChanged(TPad*)</signal>
        <slot access="public">deleteLater()</slot>
        <slot access="public">setEnabled(bool)</slot>
        <slot access="public">setDisabled(bool)</slot>
        <slot access="public">setCaption(const QString&amp;)</slot>
        <slot access="public">setIcon(const QPixmap&amp;)</slot>
        <slot access="public">setIconText(const QString&amp;)</slot>
        <slot access="public">setMouseTracking(bool)</slot>
        <slot access="public">setFocus()</slot>
        <slot access="public">clearFocus()</slot>
        <slot access="public">setUpdatesEnabled(bool)</slot>
        <slot access="public">update()</slot>
        <slot access="public">update(int,int,int,int)</slot>
        <slot access="public">update(const QRect&amp;)</slot>
        <slot access="public">repaint()</slot>
        <slot access="public">repaint(bool)</slot>
        <slot access="public">repaint(int,int,int,int)</slot>
        <slot access="public">repaint(int,int,int,int,bool)</slot>
        <slot access="public">repaint(const QRect&amp;)</slot>
        <slot access="public">repaint(const QRect&amp;,bool)</slot>
        <slot access="public">repaint(const QRegion&amp;)</slot>
        <slot access="public">repaint(const QRegion&amp;,bool)</slot>
        <slot access="public">show()</slot>
        <slot access="public">hide()</slot>
        <slot access="public">setShown(bool)</slot>
        <slot access="public">setHidden(bool)</slot>
        <slot access="public">iconify()</slot>
        <slot access="public">showMinimized()</slot>
        <slot access="public">showMaximized()</slot>
        <slot access="public">showFullScreen()</slot>
        <slot access="public">showNormal()</slot>
        <slot access="public">polish()</slot>
        <slot access="public">constPolish()</slot>
        <slot access="public">close()</slot>
        <slot access="public">raise()</slot>
        <slot access="public">lower()</slot>
        <slot access="public">stackUnder(QWidget*)</slot>
        <slot access="public">move(int,int)</slot>
        <slot access="public">move(const QPoint&amp;)</slot>
        <slot access="public">resize(int,int)</slot>
        <slot access="public">resize(const QSize&amp;)</slot>
        <slot access="public">setGeometry(int,int,int,int)</slot>
        <slot access="public">setGeometry(const QRect&amp;)</slot>
        <slot access="public">adjustSize()</slot>
        <slot access="public">cd()</slot>
        <slot access="public">cd(Int_t)</slot>
        <slot access="public">Browse(TBrowser*)</slot>
        <slot access="public">Clear()</slot>
        <slot access="public">Clear(Option_t*)</slot>
        <slot access="public">Close()</slot>
        <slot access="public">Close(Option_t*)</slot>
        <slot access="public">Draw()</slot>
        <slot access="public">Draw(Option_t*)</slot>
        <slot access="public">DrawClone()</slot>
        <slot access="public">DrawClone(Option_t*)</slot>
        <slot access="public">DrawClonePad()</slot>
        <slot access="public">EditorBar()</slot>
        <slot access="public">EnterLeave(TPad*,TObject*)</slot>
        <slot access="public">FeedbackMode(Bool_t)</slot>
        <slot access="public">Flush()</slot>
        <slot access="public">UseCurrentStyle()</slot>
        <slot access="public">ForceUpdate()</slot>
        <slot access="public">GetDISPLAY()</slot>
        <slot access="public">GetContextMenu()</slot>
        <slot access="public">GetDoubleBuffer()</slot>
        <slot access="public">GetEvent()</slot>
        <slot access="public">GetEventX()</slot>
        <slot access="public">GetEventY()</slot>
        <slot access="public">GetHighLightColor()</slot>
        <slot access="public">GetPadSave()</slot>
        <slot access="public">GetSelected()</slot>
        <slot access="public">GetSelectedOpt()</slot>
        <slot access="public">GetSelectedPad()</slot>
        <slot access="public">GetShowEventStatus()</slot>
        <slot access="public">GetAutoExec()</slot>
        <slot access="public">GetXsizeUser()</slot>
        <slot access="public">GetYsizeUser()</slot>
        <slot access="public">GetXsizeReal()</slot>
        <slot access="public">GetYsizeReal()</slot>
        <slot access="public">GetCanvasID()</slot>
        <slot access="public">GetWindowTopX()</slot>
        <slot access="public">GetWindowTopY()</slot>
        <slot access="public">GetWindowWidth()</slot>
        <slot access="public">GetWindowHeight()</slot>
        <slot access="public">GetWw()</slot>
        <slot access="public">GetWh()</slot>
        <slot access="public">GetCanvasPar(Int_t&amp;,Int_t&amp;,UInt_t&amp;,UInt_t&amp;)</slot>
        <slot access="public">HandleInput(EEventType,Int_t,Int_t)</slot>
        <slot access="public">HasMenuBar()</slot>
        <slot access="public">Iconify()</slot>
        <slot access="public">IsBatch()</slot>
        <slot access="public">IsRetained()</slot>
        <slot access="public">ls()</slot>
        <slot access="public">ls(Option_t*)</slot>
        <slot access="public">MoveOpaque()</slot>
        <slot access="public">MoveOpaque(Int_t)</slot>
        <slot access="public">OpaqueMoving()</slot>
        <slot access="public">OpaqueResizing()</slot>
        <slot access="public">Paint()</slot>
        <slot access="public">Paint(Option_t*)</slot>
        <slot access="public">Pick(Int_t,Int_t,TObjLink*&amp;)</slot>
        <slot access="public">Pick(Int_t,Int_t,TObject*)</slot>
        <slot access="public">Resize()</slot>
        <slot access="public">Resize(Option_t*)</slot>
        <slot access="public">ResizeOpaque()</slot>
        <slot access="public">ResizeOpaque(Int_t)</slot>
        <slot access="public">SaveSource()</slot>
        <slot access="public">SaveSource(const char*)</slot>
        <slot access="public">SaveSource(const char*,Option_t*)</slot>
        <slot access="public">SetCursor(ECursor)</slot>
        <slot access="public">SetDoubleBuffer()</slot>
        <slot access="public">SetDoubleBuffer(Int_t)</slot>
        <slot access="public">SetWindowPosition(Int_t,Int_t)</slot>
        <slot access="public">SetWindowSize(UInt_t,UInt_t)</slot>
        <slot access="public">SetCanvasSize(UInt_t,UInt_t)</slot>
        <slot access="public">SetHighLightColor(Color_t)</slot>
        <slot access="public">SetSelected(TObject*)</slot>
        <slot access="public">SetSelectedPad(TPad*)</slot>
        <slot access="public">Show()</slot>
        <slot access="public">Size()</slot>
        <slot access="public">Size(Float_t)</slot>
        <slot access="public">Size(Float_t,Float_t)</slot>
        <slot access="public">SetBatch()</slot>
        <slot access="public">SetBatch(Bool_t)</slot>
        <slot access="public">SetRetained()</slot>
        <slot access="public">SetRetained(Bool_t)</slot>
        <slot access="public">SetTitle()</slot>
        <slot access="public">SetTitle(const char*)</slot>
        <slot access="public">ToggleEventStatus()</slot>
        <slot access="public">ToggleAutoExec()</slot>
        <slot access="public">Update()</slot>
        <slot access="public">NeedsResize()</slot>
        <slot access="public">SetNeedsResize(Bool_t)</slot>
        <property type="CString">name</property>
        <property type="Bool">enabled</property>
        <property type="Rect">geometry</property>
        <property type="SizePolicy">sizePolicy</property>
        <property type="Size">minimumSize</property>
        <property type="Size">maximumSize</property>
        <property type="Size">sizeIncrement</property>
        <property type="Size">baseSize</property>
        <property type="Color">paletteForegroundColor</property>
        <property type="Color">paletteBackgroundColor</property>
        <property type="Pixmap">paletteBackgroundPixmap</property>
        <property type="Palette">palette</property>
        <property type="BackgroundOrigin">backgroundOrigin</property>
        <property type="Font">font</property>
        <property type="Cursor">cursor</property>
        <property type="String">caption</property>
        <property type="Pixmap">icon</property>
        <property type="String">iconText</property>
        <property type="Bool">mouseTracking</property>
        <property type="FocusPolicy">focusPolicy</property>
        <property type="Bool">acceptDrops</property>
    </customwidget>
</customwidgets>
</CW>
