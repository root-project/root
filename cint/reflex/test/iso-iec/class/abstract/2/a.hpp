#ifndef SAMPLE_HPP
#define SAMPLE_HPP

class point { /* ... */ };

/**
 * @class shape
 *
 * See 10.4.2 [class.abstract]
 * An <i>abstract class</i> is a class that can be used only as a base class of some other class;
 * no objects of an abstract base class can be created except as sub-objects of a class derived
 * from it.
 */
class shape {
   point center;
   // ...
public:
   point where() { return center; }
   void move(point p) { center = p; draw(); }
   virtual void rotate(int) = 0; // pure virtual
   virtual void draw() = 0; // pure virtual
   // ...
};

#endif // SAMPLE_HPP
