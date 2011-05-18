#if !defined		(_ACCESS_SPECIFIERS_HPP__)
#define				 _ACCESS_SPECIFIERS_HPP__

/**
 * @class FloatingPointPromotion
 *
 * See 4.6.1 [conv.fpprom]
 * An rvalue of type <code>float</code> can be converted to an rvalue of type <code>double</code>.
 * The value is unchanged.
 *
 * We should be able to set a double value via the Reflex API using a float type.
 */
class FloatingPointPromotion
{
public:
	void setDouble(double _double) { this->_double = _double; }
	double getDouble() 			   { return this->_double;    }
private:
	double _double;
};

#endif			   //_ACCESS_SPECIFIERS_HPP__
