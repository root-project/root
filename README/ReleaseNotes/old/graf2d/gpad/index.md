\defgroup Graphics Graphics
\brief The graphics related classes
\defgroup Graphics2D 2D Graphics
\ingroup Graphics
\brief The 2D graphics related classes
\defgroup gpad Graphics pad
\ingroup Graphics2D
\brief The TPad related classes

TPad and TPad's related classes' usages are illustrated by the following examples:

  - [The Graphics Pad.](http://root.cern.ch/drupal/content/graphics-pad)
  - [How to Draw objects.](http://root.cern.ch/drupal/content/how-draw-objects)
  - [How to Pick objects.](http://root.cern.ch/drupal/content/how-pick-objects)
  - [Dividing a canvas with no margins between pads.](http://root.cern.ch/root/html/tutorials/graphs/zones.C.html)
  - [Using transparent pads.](http://root.cern.ch/root/html/tutorials/hist/transpad.C.html)

\defgroup GraphicsAtt Graphics attributes
\ingroup Graphics
\brief The graphics attributes related classes

Graphics attributes, are parameters that affect the way
[graphics primitives](https://root.cern.ch/basic-graphics-primitives) are displayed.

A ROOT object get graphics attributes by inheritance from the `TAttXXX` classes.

 For example, lines can be dotted or dashed, fat or thin, blue or orange. If
an object inherits form the class TAttLine it will get these attributes.
 Areas might be filled with one color or with a multicolor pattern. If
an object inherits form the class TAttFill it will get these attribute.
Text can appear with an angle, displayed in different fonts, colors, and sizes.
 If an object inherits form the class TAttText it will get these attribute.

