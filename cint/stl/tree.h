/* -*- C++ -*- */

/************************************************************************
 *
 * Copyright(c) 1995~2006  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

/*
 *
 * Copyright (c) 1994
 * Hewlett-Packard Company
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Hewlett-Packard Company makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 *
 */

#ifndef TREE_H
#define TREE_H

/*

Red-black tree class, designed for use in implementing STL
associative containers (set, multiset, map, and multimap). The
insertion and deletion algorithms are based on those in Cormen,
Leiserson, and Rivest, Introduction to Algorithms (MIT Press, 1990),
except that

(1) the header cell is maintained with links not only to the root
but also to the leftmost node of the tree, to enable constant time
begin(), and to the rightmost node of the tree, to enable linear time
performance when used with the generic set algorithms (set_union,
etc.);

(2) when a node being deleted has two children its successor node is
relinked into its place, rather than copied, so that the only
iterators invalidated are those referring to the deleted node.

*/

#include <algobase.h>
#include <_iterator.h>
#include <function.h>
#include <bool.h>
#include <projectn.h>

#ifndef __CINT__
#ifndef rb_tree 
#define rb_tree rb_tree
#endif
#endif

template <class Key, class Value, class KeyOfValue, class Compare>
class rb_tree {
protected:
    enum color_type {red, black};
    typedef Allocator<void>::pointer void_pointer;
    struct rb_tree_node;
    friend rb_tree_node;
    struct rb_tree_node {
        color_type color_field; 
        void_pointer parent_link;
        void_pointer left_link;
        void_pointer right_link;
        Value value_field;
    };
    static Allocator<rb_tree_node> rb_tree_node_allocator;
    static Allocator<Value> value_allocator;
public:
    typedef Key key_type;
    typedef Value value_type;
    typedef Allocator<Value>::pointer pointer;
    typedef Allocator<Value>::reference reference;
    typedef Allocator<Value>::const_reference const_reference;
    typedef Allocator<rb_tree_node> rb_tree_node_allocator_type;
    typedef Allocator<rb_tree_node>::pointer link_type;
    typedef Allocator<rb_tree_node>::size_type size_type;
    typedef Allocator<rb_tree_node>::difference_type difference_type;
protected:
    size_type buffer_size() {
        return rb_tree_node_allocator.init_page_size();
    }
    struct rb_tree_node_buffer;
    friend rb_tree_node_buffer;
    struct rb_tree_node_buffer {
        void_pointer next_buffer;
        link_type buffer;
    };
public:
    typedef Allocator<rb_tree_node_buffer> buffer_allocator_type;
    typedef Allocator<rb_tree_node_buffer>::pointer buffer_pointer;     
protected:
    static Allocator<rb_tree_node_buffer> buffer_allocator;
    static buffer_pointer buffer_list;
    static link_type free_list;
    static link_type next_avail;
    static link_type last;
    void add_new_buffer() {
        buffer_pointer tmp = buffer_allocator.allocate((size_type)1);
        tmp->buffer = rb_tree_node_allocator.allocate(buffer_size());
        tmp->next_buffer = buffer_list;
        buffer_list = tmp;
        next_avail = buffer_list->buffer;
        last = next_avail + buffer_size();
    }
    static size_type number_of_trees;
    void deallocate_buffers();
    link_type get_node() {
        link_type tmp = free_list;
        return free_list ? 
            (free_list = (link_type)(free_list->right_link), tmp) 
                : (next_avail == last ? (add_new_buffer(), next_avail++) 
                   : next_avail++);
        // ugly code for inlining - avoids multiple returns
    }
    void put_node(link_type p) {
        p->right_link = free_list;
        free_list = p;
    }
protected:
    link_type header;  
    link_type& root() { return parent(header); }
    link_type& root() const { return parent(header); }
    link_type& leftmost() { return left(header); }
    link_type& leftmost() const { return left(header); }
    link_type& rightmost() { return right(header); }
    link_type& rightmost() const { return right(header); }
    size_type node_count; // keeps track of size of tree
    bool insert_always;  // controls whether an element already in the
                         // tree is inserted again
//public:
    Compare key_compare;
    static link_type NIL;
    static link_type& left(link_type x) { 
        return (link_type&)((*x).left_link);
    }
    static link_type& right(link_type x) {
        return (link_type&)((*x).right_link); 
    }
    static link_type& parent(link_type x) {
        return (link_type&)((*x).parent_link);
    }
    static reference value(link_type x) { return (*x).value_field; }
    static Allocator<Key>::const_reference key(link_type x) {
        return KeyOfValue()(value(x));
    }
    static color_type& color(link_type x) { 
        return (color_type&)(*x).color_field; }
    static link_type minimum(link_type x) {
        while (left(x) != NIL)
            x = left(x);
        return x;
    }
    static link_type maximum(link_type x) {
        while (right(x) != NIL)
            x = right(x);
        return x;
    }
public:
    class iterator;
    friend iterator;
    class const_iterator;
    friend const_iterator;
    class iterator : public bidirectional_iterator<Value, difference_type> {
    friend class rb_tree<Key, Value, KeyOfValue, Compare>;
    friend class const_iterator;
/*      
    friend bool operator==(const iterator& x, const iterator& y) {
        return x.node == y.node;
    }
*/
    protected:
        link_type node;
        iterator(link_type x) : node(x) {}
    public:
        iterator() {}
        bool operator==(const iterator& y) const { return node == y.node; }
        reference operator*() const { return value(node); }
        iterator& operator++() {
            if (right(node) != NIL) {
                node = right(node);
                while (left(node) != NIL)
                    node = left(node);
            } else {
                link_type y = parent(node);
                while (node == right(y)) {
                    node = y;
                    y = parent(y);
                }
                if (right(node) != y) // necessary because of rightmost 
                    node = y;
            }
            return *this;
        }
        iterator operator++(int) {
            iterator tmp = *this;
            ++*this;
            return tmp;
        }
        iterator& operator--() {
            if (color(node) == red && parent(parent(node)) == node)  
                // check for header
                node = right(node);   // return rightmost
            else if (left(node) != NIL) {
                link_type y = left(node);
                while (right(y) != NIL)
                    y = right(y);
                node = y;
            } else {
                link_type y = parent(node);
                while (node == left(y)) {
                    node = y;
                    y = parent(y);
                }
                node = y;
            }
            return *this;
        }
        iterator operator--(int) {
            iterator tmp = *this;
            --*this;
            return tmp;
        }
    };
    class const_iterator 
        : public bidirectional_iterator<Value,difference_type> {
    friend class rb_tree<Key, Value, KeyOfValue, Compare>;
    friend class iterator;
/*      
    friend bool operator==(const const_iterator& x, const const_iterator& y) {
        return x.node == y.node;
    }
*/
    protected:
        link_type node;
        const_iterator(link_type x) : node(x) {}
    public:
        const_iterator() {}
        const_iterator(const iterator& x) : node(x.node) {}
        bool operator==(const const_iterator& y) const { 
            return node == y.node; 
        }
        bool operator!=(const const_iterator& y) const { 
            return node != y.node; 
        }
        const_reference operator*() const { return value(node); }
        const_iterator& operator++() {
            if (right(node) != NIL) {
                node = right(node);
                while (left(node) != NIL)
                    node = left(node);
            } else {
                link_type y = parent(node);
                while (node == right(y)) {
                    node = y;
                    y = parent(y);
                }
                if (right(node) != y) // necessary because of rightmost 
                    node = y;
            }
            return *this;
        }
        const_iterator operator++(int) {
            const_iterator tmp = *this;
            ++*this;
            return tmp;
        }
        const_iterator& operator--() {
            if (color(node) == red && parent(parent(node)) == node)  
                // check for header
                node = right(node);   // return rightmost
            else if (left(node) != NIL) {
                link_type y = left(node);
                while (right(y) != NIL)
                    y = right(y);
                node = y;
            } else {
                link_type y = parent(node);
                while (node == left(y)) {
                    node = y;
                    y = parent(y);
                }
                node = y;
            }
            return *this;
        }
        const_iterator operator--(int) {
            const_iterator tmp = *this;
            --*this;
            return tmp;
        }
    };
    typedef reverse_bidirectional_iterator<iterator, value_type, reference,
                                           difference_type>
        reverse_iterator; 
    typedef reverse_bidirectional_iterator<const_iterator, value_type,
                                           const_reference, difference_type>
	const_reverse_iterator;
private:
    iterator __insert(link_type x, link_type y, const value_type& v);
    void init() {
        ++number_of_trees;
        if (NIL == 0) {
            NIL = get_node();
            color(NIL) = black;
            parent(NIL) = 0;
            left(NIL) = 0;
            right(NIL) = 0;
        }
        header = get_node();
        color(header) = red;  // used to distinguish header from root,
                              // in iterator.operator++
        root() = NIL;
        leftmost() = header;
        rightmost() = header;
    }
public:
    
// allocation/deallocation
    
    rb_tree(const Compare& comp = Compare(), bool always = true) 
           : node_count(0) {
        key_compare = comp;
        insert_always = always;
        init();
    }
    rb_tree(const value_type* first, const value_type* last, 
            const Compare& comp = Compare(), bool always = true)
          : node_count(0) {
        key_compare = comp;
        insert_always = always;
        init();
        insert(first, last);
    }
    rb_tree(const rb_tree<Key, Value, KeyOfValue, Compare>& x, 
            bool always = true) : node_count(0) {
        key_compare = x.key_compare;
        insert_always = always;
        init();
        for (rb_tree<Key, Value, KeyOfValue, Compare>::const_iterator 
                i = x.begin(); i != x.end(); ++i)
            insert(end(), *i);
    }
    ~rb_tree() {
        erase(begin(), end());
        put_node(header);
        if (--number_of_trees == 0) {
            put_node(NIL);
            NIL = 0;
            deallocate_buffers();
            free_list = 0;    
            next_avail = 0;
            last = 0;
        }
    }
    rb_tree<Key, Value, KeyOfValue, Compare>& 
        operator=(const rb_tree<Key, Value, KeyOfValue, Compare>& x);
    
// accessors:

    Compare key_comp() const { return key_compare; }
    iterator begin() { return leftmost(); }
    const_iterator begin() const { return leftmost(); }
    iterator end() { return header; }
    const_iterator end() const { return header; }
    reverse_iterator rbegin() { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const { 
        return const_reverse_iterator(end()); 
    }
    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const { 
        return const_reverse_iterator(begin());
    } 
    bool empty() const { return node_count == 0; }
    size_type size() const { return node_count; }
    size_type max_size() const { 
        return rb_tree_node_allocator.max_size(); 
    }
    void swap(rb_tree<Key, Value, KeyOfValue, Compare>& t) {
        ::swap(header, t.header);
        ::swap(node_count, t.node_count);
        ::swap(insert_always, t.insert_always);
        ::swap(key_compare, t.key_compare);
    }
    
// insert/erase

    typedef  pair<iterator, bool> pair_iterator_bool; 
    // typedef done to get around compiler bug
    pair_iterator_bool insert(const value_type& x);
    iterator insert(iterator position, const value_type& x);
    void insert(iterator first, iterator last);
    void insert(const value_type* first, const value_type* last);
    void erase(iterator position);
    size_type erase(const key_type& x);
    void erase(iterator first, iterator last);
    void erase(const key_type* first, const key_type* last);

// set operations:

    iterator find(const key_type& x);
    const_iterator find(const key_type& x) const;
    size_type count(const key_type& x) const;
    iterator lower_bound(const key_type& x);
    const_iterator lower_bound(const key_type& x) const;
    iterator upper_bound(const key_type& x);
    const_iterator upper_bound(const key_type& x) const;
    typedef  pair<iterator, iterator> pair_iterator_iterator; 
    // typedef done to get around compiler bug
    pair_iterator_iterator equal_range(const key_type& x);
    typedef  pair<const_iterator, const_iterator> pair_citerator_citerator; 
    // typedef done to get around compiler bug
    pair_citerator_citerator equal_range(const key_type& x) const;
    inline void rotate_left(link_type x);
    inline void rotate_right(link_type x);
};

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::buffer_pointer 
        rb_tree<Key, Value, KeyOfValue, Compare>::buffer_list = 0;

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::link_type 
        rb_tree<Key, Value, KeyOfValue, Compare>::free_list = 0;

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::link_type 
        rb_tree<Key, Value, KeyOfValue, Compare>::next_avail = 0;

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::link_type 
        rb_tree<Key, Value, KeyOfValue, Compare>::last = 0;

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::size_type 
        rb_tree<Key, Value, KeyOfValue, Compare>::number_of_trees = 0;

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::rb_tree_node_allocator_type 
        rb_tree<Key, Value, KeyOfValue, Compare>::rb_tree_node_allocator;

template <class Key, class Value, class KeyOfValue, class Compare>
Allocator<Value> rb_tree<Key, Value, KeyOfValue, Compare>::value_allocator;

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::buffer_allocator_type 
        rb_tree<Key, Value, KeyOfValue, Compare>::buffer_allocator;

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::link_type 
        rb_tree<Key, Value, KeyOfValue, Compare>::NIL = 0;

template <class Key, class Value, class KeyOfValue, class Compare>
void rb_tree<Key, Value, KeyOfValue, Compare>::deallocate_buffers() {
    while (buffer_list) {
        buffer_pointer tmp = buffer_list;
        buffer_list = (buffer_pointer)(buffer_list->next_buffer);
        rb_tree_node_allocator.deallocate(tmp->buffer);
        buffer_allocator.deallocate(tmp);
    }
}

template <class Key, class Value, class KeyOfValue, class Compare>
inline bool operator==(const rb_tree<Key, Value, KeyOfValue, Compare>& x, 
                       const rb_tree<Key, Value, KeyOfValue, Compare>& y) {
    return x.size() == y.size() && equal(x.begin(), x.end(), y.begin());
}

template <class Key, class Value, class KeyOfValue, class Compare>
inline bool operator<(const rb_tree<Key, Value, KeyOfValue, Compare>& x, 
                      const rb_tree<Key, Value, KeyOfValue, Compare>& y) {
    return lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
}

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>& 
rb_tree<Key, Value, KeyOfValue, Compare>::
operator=(const rb_tree<Key, Value, KeyOfValue, Compare>& x) {
    if (this != &x) {
        // can't be done as in list because Key may be contant type
        erase(begin(), end());
        for (rb_tree<Key, Value, KeyOfValue, Compare>::const_iterator 
                i = x.begin(); i != x.end(); ++i)
            insert(end(), *i);
    }
    return *this;
}

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::iterator
rb_tree<Key, Value, KeyOfValue, Compare>::
__insert(link_type x, link_type y, const Value& v) {
    ++node_count;
    link_type z = get_node();
    construct(value_allocator.address(value(z)), v);
    if (y == header || x != NIL || key_compare(KeyOfValue()(v), key(y))) {
        left(y) = z;  // also makes leftmost() = z when y == header
        if (y == header) {
            root() = z;
            rightmost() = z;
        } else if (y == leftmost())
            leftmost() = z;   // maintain leftmost() pointing to minimum node
    } else {
        right(y) = z;
        if (y == rightmost())
            rightmost() = z;   // maintain rightmost() pointing to maximum node
    }
    parent(z) = y;
    left(z) = NIL;
    right(z) = NIL;
    x = z;  // recolor and rebalance the tree
    color(x) = red;
    while (x != root() && color(parent(x)) == red) 
        if (parent(x) == left(parent(parent(x)))) {
            y = right(parent(parent(x)));
            if (color(y) == red) {
                color(parent(x)) = black;
                color(y) = black;
                color(parent(parent(x))) = red;
                x = parent(parent(x));
            } else {
                if (x == right(parent(x))) {
                    x = parent(x);
                    rotate_left(x);
                }
                color(parent(x)) = black;
                color(parent(parent(x))) = red;
                rotate_right(parent(parent(x)));
            }
        } else {
            y = left(parent(parent(x)));
            if (color(y) == red) {
                color(parent(x)) = black;
                color(y) = black;
                color(parent(parent(x))) = red;
                x = parent(parent(x));
            } else {
                if (x == left(parent(x))) {
                    x = parent(x);
                    rotate_right(x);
                }
                color(parent(x)) = black;
                color(parent(parent(x))) = red;
                rotate_left(parent(parent(x)));
            }
        }
    color(root()) = black;
    return iterator(z);
}

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::pair_iterator_bool
rb_tree<Key, Value, KeyOfValue, Compare>::insert(const Value& v) {
    link_type y = header;
    link_type x = root();
    while (x != NIL) {
        y = x;
        if (key_compare(KeyOfValue()(v), key(y)))
            x = left(x);
        else {
            x = right(x);
            if (!key_compare(key(y), KeyOfValue()(v))) {
                // value(y) and v are equivalent according to key_compare
                if (!insert_always) 
                    return pair_iterator_bool(iterator(y), false);
                if (x != NIL) {
                    y = minimum(x);
                    break;
                }
            }
        }
    }
    return pair_iterator_bool(__insert(x, y, v), true);
}

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::iterator 
rb_tree<Key, Value, KeyOfValue, Compare>::insert(iterator position,
                                                 const Value& v) {
    if (position == iterator(begin()))
        if (size() > 0 && key_compare(KeyOfValue()(v), key(position.node)))
            return __insert(position.node, position.node, v);
            // first argument just needs to be non-NIL 
        else
            return insert(v).first;
    else if (position == iterator(end()))
        if (key_compare(key(rightmost()), KeyOfValue()(v)))
            return __insert(NIL, rightmost(), v);
        else
            return insert(v).first;
    else {
        iterator before = --position;
        if (key_compare(key(before.node), KeyOfValue()(v))
            && key_compare(KeyOfValue()(v), key(position.node)))
            if (right(before.node) == NIL)
                return __insert(NIL, before.node, v); 
            else
                return __insert(position.node, position.node, v);
                // first argument just needs to be non-NIL 
        else
            return insert(v).first;
    }
}

template <class Key, class Value, class KeyOfValue, class Compare>
void rb_tree<Key, Value, KeyOfValue, Compare>::insert(iterator first, 
                                                      iterator last) {
    while (first != last) insert(*first++);
}

template <class Key, class Value, class KeyOfValue, class Compare>
void rb_tree<Key, Value, KeyOfValue, Compare>::insert(const Value* first, 
                                                      const Value* last) {
    while (first != last) insert(*first++);
}
         
template <class Key, class Value, class KeyOfValue, class Compare>
void rb_tree<Key, Value, KeyOfValue, Compare>::erase(iterator position) {
    link_type z = position.node;
    link_type y = z;
    link_type x;
    if (left(y) == NIL)
        x = right(y);
    else
        if (right(y) == NIL) 
            x = left(y);
        else {
            y = right(y);
            while (left(y) != NIL)
                y = left(y);
            x = right(y);
        }
    if (y != z) { // relink y in place of z
        parent(left(z)) = y; 
        left(y) = left(z);
        if (y != right(z)) {
            parent(x) = parent(y); // possibly x == NIL
            left(parent(y)) = x;   // y must be a left child
            right(y) = right(z);
            parent(right(z)) = y;
        } else
            parent(x) = y;  // needed in case x == NIL
        if (root() == z)
            root() = y;
        else if (left(parent(z)) == z)
            left(parent(z)) = y;
        else 
            right(parent(z)) = y;
        parent(y) = parent(z);
        ::swap(color(y), color(z));
        ::swap(y, z);  
                       // y points to node to be actually deleted,
                       // z points to old z's former successor
    } else {  // y == z
        parent(x) = parent(y);   // possibly x == NIL
        if (root() == z)
            root() = x;
        else 
            if (left(parent(z)) == z)
                left(parent(z)) = x;
            else
                right(parent(z)) = x;
        if (leftmost() == z) 
            if (right(z) == NIL)  // left(z) must be NIL also
                leftmost() = parent(z);
                // makes leftmost() == header if z == root()
        else
            leftmost() = minimum(x);
        if (rightmost() == z)  
            if (left(z) == NIL) // right(z) must be NIL also
                rightmost() = parent(z);  
                // makes rightmost() == header if z == root()
        else  // x == left(z)
            rightmost() = maximum(x);
    }
    if (color(y) != red) { 
        while (x != root() && color(x) == black)
            if (x == left(parent(x))) {
                link_type w = right(parent(x));
                if (color(w) == red) {
                    color(w) = black;
                    color(parent(x)) = red;
                    rotate_left(parent(x));
                    w = right(parent(x));
                }
                if (color(left(w)) == black && color(right(w)) == black) {
                    color(w) = red;
                    x = parent(x);
                } else {
                    if (color(right(w)) == black) {
                        color(left(w)) = black;
                        color(w) = red;
                        rotate_right(w);
                        w = right(parent(x));
                    }
                    color(w) = color(parent(x));
                    color(parent(x)) = black;
                    color(right(w)) = black;
                    rotate_left(parent(x));
                    break;
                }
            } else {  // same as then clause with "right" and "left" exchanged
                link_type w = left(parent(x));
                if (color(w) == red) {
                    color(w) = black;
                    color(parent(x)) = red;
                    rotate_right(parent(x));
                    w = left(parent(x));
                }
                if (color(right(w)) == black && color(left(w)) == black) {
                    color(w) = red;
                    x = parent(x);
                } else {
                    if (color(left(w)) == black) {
                        color(right(w)) = black;
                        color(w) = red;
                        rotate_left(w);
                        w = left(parent(x));
                    }
                    color(w) = color(parent(x));
                    color(parent(x)) = black;
                    color(left(w)) = black;
                    rotate_right(parent(x));
                    break;
                }
            }
        color(x) = black;
    }
    destroy(value_allocator.address(value(y)));
    put_node(y);
    --node_count;
}

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::size_type 
rb_tree<Key, Value, KeyOfValue, Compare>::erase(const Key& x) {
    pair_iterator_iterator p = equal_range(x);
    size_type n = 0;
    distance(p.first, p.second, n);
    erase(p.first, p.second);
    return n;
}

template <class Key, class Value, class KeyOfValue, class Compare>
void rb_tree<Key, Value, KeyOfValue, Compare>::erase(iterator first, 
                                                     iterator last) {
    while (first != last) erase(first++);
}

template <class Key, class Value, class KeyOfValue, class Compare>
void rb_tree<Key, Value, KeyOfValue, Compare>::erase(const Key* first, 
                                                     const Key* last) {
    while (first != last) erase(*first++);
}

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::iterator 
rb_tree<Key, Value, KeyOfValue, Compare>::find(const Key& k) {
    link_type x = root();
    while (x != NIL) {
        if (key_compare(k, key(x)))
            x = left(x);
        else if (key_compare(key(x), k))
            x = right(x);
        else 
            return iterator(x);
    }
    return end();
}

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::const_iterator 
rb_tree<Key, Value, KeyOfValue, Compare>::find(const Key& k) const {
    link_type x = root();
    while (x != NIL) {
        if (key_compare(k, key(x)))
            x = left(x);
        else if (key_compare(key(x), k))
            x = right(x);
        else 
            return const_iterator(x);
    }
    return end();
}

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::size_type 
rb_tree<Key, Value, KeyOfValue, Compare>::count(const Key& k) const {
    pair<const_iterator, const_iterator> p = equal_range(k);
    size_type n = 0;
    distance(p.first, p.second, n);
    return n;
}

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::iterator 
rb_tree<Key, Value, KeyOfValue, Compare>::lower_bound(const Key& k) {
    link_type y = header;
    link_type x = root();
    while (x != NIL) {
        y = x;
        if (key_compare(key(x), k))
            x = right(x);
        else 
            x = left(x);
    }
    if (y == header || !key_compare(key(y), k)) 
        return iterator(y);
    iterator j = iterator(y);
    return ++j;
}

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::const_iterator 
rb_tree<Key, Value, KeyOfValue, Compare>::lower_bound(const Key& k) const {
    link_type y = header;
    link_type x = root();
    while (x != NIL) {
        y = x;
        if (key_compare(key(x), k))
            x = right(x);
        else 
            x = left(x);
    }
    if (y == header || !key_compare(key(y), k)) 
        return const_iterator(y);
    const_iterator j = const_iterator(y);
    return ++j;
}

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::iterator 
rb_tree<Key, Value, KeyOfValue, Compare>::upper_bound(const Key& k) {
    link_type y = header;
    link_type x = root();
    while (x != NIL) {
        y = x;
        if (key_compare(k, key(x)))
            x = left(x);
        else 
            x = right(x);
    }
    if (y == header || key_compare(k, key(y)))
        return iterator(y);
    iterator j = iterator(y);
    return ++j;
}

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::const_iterator 
rb_tree<Key, Value, KeyOfValue, Compare>::upper_bound(const Key& k) const {
    link_type y = header;
    link_type x = root();
    while (x != NIL) {
        y = x;
        if (key_compare(k, key(x)))
            x = left(x);
        else 
            x = right(x);
    }
    if (y == header || key_compare(k, key(y)))
        return const_iterator(y);
    const_iterator j = const_iterator(y);
    return ++j;
}

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::pair_iterator_iterator 
rb_tree<Key, Value, KeyOfValue, Compare>::equal_range(const Key& k) {
    return pair_iterator_iterator(lower_bound(k), upper_bound(k));
}

template <class Key, class Value, class KeyOfValue, class Compare>
rb_tree<Key, Value, KeyOfValue, Compare>::pair_citerator_citerator 
rb_tree<Key, Value, KeyOfValue, Compare>::equal_range(const Key& k) const {
    return pair_citerator_citerator(lower_bound(k), upper_bound(k));
}

template <class Key, class Value, class KeyOfValue, class Compare>
inline void 
rb_tree<Key, Value, KeyOfValue, Compare>::rotate_left(link_type x) {
    link_type y = right(x);
    right(x) = left(y);
    if (left(y) != NIL)
        parent(left(y)) = x;
    parent(y) = parent(x);
    if (x == root())
        root() = y;
    else if (x == left(parent(x)))
        left(parent(x)) = y;
    else
        right(parent(x)) = y;
    left(y) = x;
    parent(x) = y;
}

template <class Key, class Value, class KeyOfValue, class Compare>
inline void 
rb_tree<Key, Value, KeyOfValue, Compare>::rotate_right(link_type x) {
    link_type y = left(x);
    left(x) = right(y);
    if (right(y) != NIL)
        parent(right(y)) = x;
    parent(y) = parent(x);
    if (x == root())
        root() = y;
    else if (x == right(parent(x)))
        right(parent(x)) = y;
    else
        left(parent(x)) = y;
    right(y) = x;
    parent(x) = y;
}

#endif

