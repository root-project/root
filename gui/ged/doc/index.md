\defgroup ged ROOT Graphics Editor
\ingroup gui
\brief Classes forming the Graphics Editor (GED) of ROOT and the basic classes of so-called object editors.

## The ROOT Graphics Editor (GED)


Everything drawn in a ROOT canvas is an object. There are classes for
all objects, and they fall into hierarchies. In addition, the ROOT has
fully cross-platform GUI classes and provides all standard components
for an application environment with common ‘look and feel'. The
object-oriented, event-driven programming model supports the modern
signals/slots communication mechanism. It handles user interface actions
and allows total independence of interacting objects and classes. This
mechanism uses the ROOT dictionary information and the Cling the C++
Interpreter to connect signals to slots methods.

Therefore, all necessary elements for an object-oriented editor design
are in place. The editor complexity can be reduced by splitting it into
discrete units of so-called *`object`* *`editors`*. Any object editor
provides an object specific GUI. The main purpose of the ROOT graphics
editor is the organization of the object editors' appearance and the
task sequence between them.

### Object Editors


Every object editor follows a simple naming convention: to have as a
name the object class name concatenated with ‘*`Editor`*' (e.g. for
**`TGraph`** objects the object editor is **`TGraphEditor`**). Thanks to
the signals/slots communication mechanism and to the method
`DistanceToPrimitive()` that computes a ‘‘distance'' to an object from
the mouse position, it was possible to implement a signal method of the
canvas that says which is the selected object and to which pad it
belongs. Having this information the graphics editor loads the
corresponding object editor and the user interface is ready for use.
This way after a click on ‘axis'—the axis editor is active; a click on a
‘pad' activates the pad editor, etc.

The algorithm in use is simple and is based on the object-oriented
relationship and communication. When the user activates the editor,
according to the selected object **`<obj>`** in the canvas it looks for
a class name **`<obj>Editor`**. For that reason, the correct naming is
very important. If a class with this name is found, the editor verifies
that this class derives from the base editor class **`TGedFrame`**. If
all checks are satisfied, the editor makes an instance of the object
editor. Then, it scans all object base classes searching the
corresponding object editors. When it finds one, it makes an instance of
the base class editor too.

Once the object editor is in place, it sets the user interface elements
according to the object's status. After that, it is ready to interact
with the object following the user actions.

The graphics editor gives an intuitive way to edit objects in a canvas
with immediate feedback. Complexity of some object editors is reduced by
hiding GUI elements and revealing them only on users' requests.

An object in the canvas is selected by clicking on it with the left
mouse button. Its name is displayed on the top of the editor frame in
red color. If the editor frame needs more space than the canvas window,
a vertical scroll bar appears for easy navigation.

\image html ged.png width=800px

**Histogram, pad and axis editors**

### Editor Design Elements


The next rules describe the path to follow when creating your own object
editor that will be recognized and loaded by the graphics editor in
ROOT, i.e. it will be included as a part of it.

(a) Derive the code of your object editor from the base editor class
**`TGedFrame`**.

(b) Keep the correct naming convention: the name of the object editor
should be the object class name concatenated with the word `‘Editor'`.

(c) Provide a default constructor.

(d) Use the signals/slots communication mechanism for event processing.

(e) Implement the virtual method `SetModel(TObject *obj)` where all
widgets are set with the current object's attributes. This method is
called when the editor receives a signal from the canvas saying that an
object is the selected.

(f) Implement all necessary slots and connect them to appropriate
signals that GUI widgets send out. The GUI classes in ROOT are developed
to emit signals whenever they change a state that others might be
interested. As we noted already, the signals/slots communication
mechanism allows total independence of the interacting classes.

#### Creation and Destruction

GED-frames are constructed during traversal of class hierarchy of the
selected object, executed from method **`TGedEditor::SetModel()`**.
When a new object of a different class is selected, the unneeded
GED-frames are cached in memory for potential reuse. The frames are
deleted automatically when the editor is closed.

Note: A deep cleanup is assumed for all frames put into the editor. This
implies:

-   do not share the layout-hints among GUI components;

-   do not delete child widgets in the destructor as this is done
    automatically.

#### Using Several Tabs

Sometimes you might need to use several tabs to organize properly your
class-editor. Each editor tab is a resource shared among all the
class-editors. Tabs must be created from the constructor of your
editor-class by using the method:

``` {.cpp}
TGVerticalFrame* TGedFrame::CreateEditorTabSubFrame(const Text_t *name),
```

It returns a pointer to a new tab container frame ready for use in your
class. If you need to hide/show this frame depending on the object's
status, you should store it in a data member. See for examples:
**`TH1Editor`**, **`TH2Editor`**.

#### Base-Class Editors Control

Full control over base-class editors can be achieved by re-implementing
virtual method void `TGedFrame::ActivateBaseClassEditors(TClass` `*cl)`.
It is called during each compound editor rebuild and the default
implementation simply offers all base-classes to the publishing
mechanism.

To prevent inclusion of a base-class into the compound editor, call:

``` {.cpp}
void TGedEditor::ExcludeClassEditor(TClass* class, Bool_t recurse)
```

Pointer to the compound GED-editor is available in **`TGedFrame`**‘s
data-member:

``` {.cpp}
TGedEditor *fGedEditor
```

Ordering of base-class editor frames follows the order of the classes in
the class hierarchy. This order can be changed by modifying the value of
**`TGedFrame`**'s data member `Int_t fPriority`. The default value is
50; smaller values move the frame towards to the top. This priority
should be set in the editor constructor.
