// Bindings
#include "CPyCppyy.h"
#include "TypeManip.h"

// Standard
#include <algorithm>
#include <sstream>
#include <ctype.h>


//- helpers ------------------------------------------------------------------
static inline
bool is_varchar(char c) {
    return isalnum((int)c) || c == '_' || c == ')' || c == '(' /* for (anonymous)/(unnamed) */;
}

static inline
std::string::size_type find_qualifier_index(const std::string& name)
{
// Find the first location that is not part of the class name proper.
    std::string::size_type i = name.size() - 1;
    bool arr_open = false;
    for ( ; 0 < i; --i) {
        std::string::value_type c = name[i];
        if (!arr_open && (is_varchar(c) || c == '>')) {
            if (c == 't' && 6 < i && !is_varchar(name[i-5]) && name.substr(i-4, 5) == "const")
                i -= 4;      // this skips 'const' on a pointer type
            else
                break;
        } else if (c == ']') {
            arr_open = true;
        } else if (c == '[') {
            arr_open = false;
        }
    }

    return i+1;
}

static inline void erase_const(std::string& name)
{
// Find and remove all occurrence of 'const'.
    if (name.empty())
        return;

    std::string::size_type spos = std::string::npos;
    std::string::size_type start = 0;
    while ((spos = name.find("const", start)) != std::string::npos) {
    // make sure not to erase 'const' as part of the name: if it is
    // connected, before or after, to a variable name, then keep it
        std::string::size_type after = spos+5;
        if (after < name.size() && is_varchar(name[after])) {
            start = after;
            continue;
        } else if (after == name.size()) {
            if (spos > 0 && is_varchar(name[spos - 1]))
                break;
        }

        std::string::size_type i = 5;
        while (name[spos+i] == ' ') ++i;
        name.swap(name.erase(spos, i));
    }
}

static inline void rstrip(std::string& name)
{
// Remove space from the right side of name.
    std::string::size_type i = name.size();
    for ( ; 0 < i; --i) {
       if (!isspace(name[i-1]))
           break;
    }

    if (i != name.size())
        name = name.substr(0, i);
}

//----------------------------------------------------------------------------
std::string CPyCppyy::TypeManip::remove_const(const std::string& cppname)
{
// Remove 'const' qualifiers from the given C++ name.
    std::string::size_type tmplt_start = cppname.find('<');
    std::string::size_type type_stop   = cppname.rfind('>');
    if (cppname.find("::", type_stop+1) != std::string::npos) // e.g. klass<T>::some_typedef
        type_stop = cppname.find(' ', type_stop+1);
    if (tmplt_start != std::string::npos && cppname[tmplt_start+1] != '<') {
    // only replace const qualifying cppname, not in template parameters
        std::string pre = cppname.substr(0, tmplt_start);
        erase_const(pre);
        std::string post = "";
        if (type_stop != std::string::npos && type_stop != cppname.size()-1) {
            post = cppname.substr(type_stop+1, std::string::npos);
            erase_const(post);
        }

        return pre + cppname.substr(tmplt_start, type_stop+1-tmplt_start) + post;
    }

    std::string clean_name = cppname;
    erase_const(clean_name);
    return clean_name;
}

//----------------------------------------------------------------------------
std::string CPyCppyy::TypeManip::clean_type(
    const std::string& cppname, bool template_strip, bool const_strip)
{
// Strip C++ name from all qualifiers and compounds.
    std::string::size_type i = find_qualifier_index(cppname);
    std::string name = cppname.substr(0, i);
    rstrip(name);

    if (name.back() == ']') {                      // array type?
    // TODO: this fails templates instantiated on arrays (not common)
        name = name.substr(0, name.find('['));
    } else if (template_strip && name.back() == '>') {
        name = name.substr(0, name.find('<'));
    }

    if (const_strip) {
        if (template_strip)
            erase_const(name);
        else
            name = remove_const(name);
    }

    return name;
}

//----------------------------------------------------------------------------
std::string CPyCppyy::TypeManip::template_base(const std::string& cppname)
{
// If this is a template, return the underlying template name w/o arguments
    if (cppname.empty() || cppname.back() != '>')
        return cppname;

    int tpl_open = 0;
    for (std::string::size_type pos = cppname.size()-1; 0 < pos; --pos) {
        std::string::value_type c = cppname[pos];

    // count '<' and '>' to be able to skip template contents
        if (c == '>')
            --tpl_open;
        else if (c == '<' && cppname[pos+1] != '<')
            ++tpl_open;

        if (tpl_open == 0)
            return cppname.substr(0, pos);
    }

    return cppname;
}

//----------------------------------------------------------------------------
std::string CPyCppyy::TypeManip::compound(const std::string& name)
{
// Break down the compound of a fully qualified type name.
    std::string cleanName = remove_const(name);
    auto idx = find_qualifier_index(cleanName);

    const std::string& cpd = cleanName.substr(idx, std::string::npos);

// for easy identification of fixed size arrays
    if (!cpd.empty() && cpd.back() == ']') {
        if (cpd.front() == '[')
            return "[]";    // fixed array any; dimensions handled separately

        std::ostringstream scpd;
        scpd << cpd.substr(0, cpd.find('[')) << "[]";
        return scpd.str();
    }

    return cpd;
}

//----------------------------------------------------------------------------
void CPyCppyy::TypeManip::cppscope_to_pyscope(std::string& cppscope)
{
// Change '::' in C++ scope into '.' as in a Python scope.
    std::string::size_type pos = 0;
    while ((pos = cppscope.find("::", pos)) != std::string::npos) {
        cppscope.replace(pos, 2, ".");
        pos += 1;
    }
}

//----------------------------------------------------------------------------
void CPyCppyy::TypeManip::cppscope_to_legalname(std::string& cppscope)
{
// Change characters illegal in a variable name into '_' to form a legal name.
    for (char& c : cppscope) {
        for (char needle : {':', '>', '<', ' ', ',', '&', '=', '*'})
            if (c == needle) c = '_';
    }
}

//----------------------------------------------------------------------------
std::string CPyCppyy::TypeManip::extract_namespace(const std::string& name)
{
// Find the namespace the named class lives in, take care of templates
    if (name.empty())
        return name;

    int tpl_open = 0;
    for (std::string::size_type pos = name.size()-1; 0 < pos; --pos) {
        std::string::value_type c = name[pos];

    // count '<' and '>' to be able to skip template contents
        if (c == '>')
            --tpl_open;
        else if (c == '<' && name[pos+1] != '<')
            ++tpl_open;

    // collect name up to "::"
        else if (tpl_open == 0 && c == ':' && name[pos-1] == ':') {
        // found the extend of the scope ... done
            return name.substr(0, pos-1);
        }
    }

// no namespace; assume outer scope
    return "";
}

//----------------------------------------------------------------------------
std::vector<std::string> CPyCppyy::TypeManip::extract_arg_types(const std::string& sig)
{
// break out the argument types from the signature string
    std::vector<std::string> result;

    if (sig.empty() || sig == "()")
        return result;

    int tpl_open = 0;
    std::string::size_type start = 1;
    for (std::string::size_type pos = 1; pos < sig.size()-1; ++pos) {
        std::string::value_type c = sig[pos];

    // count '<' and '>' to be able to skip template contents
        if (c == '>')
            --tpl_open;
        else if (c == '<' && sig[pos+1] != '<')
            ++tpl_open;

    // collect type name up to ',' or end ')'
        else if (tpl_open == 0 && c == ',') {
        // found the extend of the scope ... done
            result.push_back(sig.substr(start, pos-start));
            start = pos+1;
        }
    }

// add last type
    result.push_back(sig.substr(start, sig.rfind(")")-start));

    return result;
}

//----------------------------------------------------------------------------
Py_ssize_t CPyCppyy::TypeManip::array_size(const std::string& name)
{
// Extract the array size from a given type name (assumes 1D arrays)
    std::string cleanName = remove_const(name);
    if (cleanName[cleanName.size()-1] == ']') {
        std::string::size_type idx = cleanName.rfind('[');
        if (idx != std::string::npos) {
            const std::string asize = cleanName.substr(idx+1, cleanName.size()-2);
            return strtoul(asize.c_str(), nullptr, 0);
        }
    }

    return -1;
}
