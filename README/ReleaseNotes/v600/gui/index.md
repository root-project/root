## GUI Libraries


### Attributes editors

-   The transparency sliders change the transparency only for the currently edited object.

### Object editing on canvas

-   The "move opaque" way to edit object on canvas, has been extended to all kind of objects.
-   When in move opaque mode, a canvas can show guide lines to place object relatively to each other. A new resource in `etc/system.rootrc` allows to turn the feature on or off: `Canvas.ShowGuideLines`.
-   For a fine adjustment at the pixel level, the arrow keys can be used to move object on pad.
-   The zoom on axis and on 2D histogram has be improved. A shaded area is shown instead on simple lines. Also it is possible to zoom a 2D histogram with a shaded rectangle.

### Saving Files
-  When saving files from a canvas, the default file type is now .pdf instead of .ps, since pdf is probably becoming more popular than ps.
-  In the "File Save Dialog", there is now a default file name and its extension (if a specific one is selected), and the name is highlighted, so when the user types something, only the file name is changed.
-  The default file type can be changed with a new `Canvas.SaveAsDefaultType` option in `etc/system.rootrc` (default being pdf).

### ROOT browser and pad editor
-  The Pad Editor is now embedded in the left tab of the browser instead of inside the canvas itself, so the layout of the canvas remains untouched when opening the editor.


