/* GLIB - Library of useful routines for C programming
 * Copyright (C) 1995-1997  Peter Mattis, Spencer Kimball and Josh MacDonald
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

/*
 * Modified by the GLib Team and others 1997-2000.  See the AUTHORS
 * file for a list of people on the GLib Team.  See the ChangeLog
 * files for a list of changes.  These files are distributed with
 * GLib at ftp://ftp.gtk.org/pub/gtk/. 
 */

/* 
 * MT safe
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "glib.h"


typedef struct _GTreeNode  GTreeNode;

struct _GTree
{
  GTreeNode *root;
  GCompareDataFunc key_compare;
  GDestroyNotify   key_destroy_func;
  GDestroyNotify   value_destroy_func;
  gpointer         key_compare_data;
};

struct _GTreeNode
{
  gint balance;      /* height (left) - height (right) */
  GTreeNode *left;   /* left subtree */
  GTreeNode *right;  /* right subtree */
  gpointer key;      /* key for this node */
  gpointer value;    /* value stored at this node */
};


static GTreeNode* g_tree_node_new                   (gpointer          key,
						     gpointer          value);
static void       g_tree_node_destroy               (GTreeNode        *node,
                                                     GDestroyNotify    key_destroy_func,
						     GDestroyNotify    value_destroy_func);
static GTreeNode* g_tree_node_insert                (GTree            *tree,
                                                     GTreeNode        *node,
						     gpointer          key,
						     gpointer          value,
                                                     gboolean          replace,
						     gboolean         *inserted);
static GTreeNode* g_tree_node_remove                (GTree            *tree,
                                                     GTreeNode        *node,
						     gconstpointer     key,
                                                     gboolean          notify);
static GTreeNode* g_tree_node_balance               (GTreeNode        *node);
static GTreeNode* g_tree_node_remove_leftmost       (GTreeNode        *node,
						     GTreeNode       **leftmost);
static GTreeNode* g_tree_node_restore_left_balance  (GTreeNode        *node,
						     gint              old_balance);
static GTreeNode* g_tree_node_restore_right_balance (GTreeNode        *node,
						     gint              old_balance);
static GTreeNode* g_tree_node_lookup                (GTreeNode        *node,
						     GCompareDataFunc  compare,
						     gpointer          comp_data,
						     gconstpointer     key);
static gint       g_tree_node_count                 (GTreeNode        *node);
static gint       g_tree_node_pre_order             (GTreeNode        *node,
						     GTraverseFunc     traverse_func,
						     gpointer          data);
static gint       g_tree_node_in_order              (GTreeNode        *node,
						     GTraverseFunc     traverse_func,
						     gpointer          data);
static gint       g_tree_node_post_order            (GTreeNode        *node,
						     GTraverseFunc     traverse_func,
						     gpointer          data);
static gpointer   g_tree_node_search                (GTreeNode        *node,
						     GCompareFunc      search_func,
						     gconstpointer     data);
static gint       g_tree_node_height                (GTreeNode        *node);
static GTreeNode* g_tree_node_rotate_left           (GTreeNode        *node);
static GTreeNode* g_tree_node_rotate_right          (GTreeNode        *node);
static void       g_tree_node_check                 (GTreeNode        *node);


G_LOCK_DEFINE_STATIC (g_tree_global);
static GMemChunk *node_mem_chunk = NULL;
static GTreeNode *node_free_list = NULL;


static GTreeNode*
g_tree_node_new (gpointer key,
		 gpointer value)
{
  GTreeNode *node;

  G_LOCK (g_tree_global);
  if (node_free_list)
    {
      node = node_free_list;
      node_free_list = node->right;
    }
  else
    {
      if (!node_mem_chunk)
	node_mem_chunk = g_mem_chunk_new ("GLib GTreeNode mem chunk",
					  sizeof (GTreeNode),
					  1024,
					  G_ALLOC_ONLY);

      node = g_chunk_new (GTreeNode, node_mem_chunk);
   }
  G_UNLOCK (g_tree_global);

  node->balance = 0;
  node->left = NULL;
  node->right = NULL;
  node->key = key;
  node->value = value;

  return node;
}

static void
g_tree_node_destroy (GTreeNode      *node,
		     GDestroyNotify  key_destroy_func,
		     GDestroyNotify  value_destroy_func)
{
  if (node)
    {
      g_tree_node_destroy (node->right,
			   key_destroy_func, value_destroy_func);
      g_tree_node_destroy (node->left,
			   key_destroy_func, value_destroy_func);

      if (key_destroy_func)
	key_destroy_func (node->key);
      if (value_destroy_func)
	value_destroy_func (node->value);
      
#ifdef ENABLE_GC_FRIENDLY
      node->left = NULL;
      node->key = NULL;
      node->value = NULL;
#endif /* ENABLE_GC_FRIENDLY */

      G_LOCK (g_tree_global);
      node->right = node_free_list;
      node_free_list = node;
      G_UNLOCK (g_tree_global);
   }
}

/**
 * g_tree_new:
 * @key_compare_func: the function used to order the nodes in the #GTree.
 *   It should return values similar to the standard 
 *   <function>strcmp()</function> function -
 *   0 if the two arguments are equal, a negative value if the first argument 
 *   comes before the second, or a positive value if the first argument comes 
 *   after the second.
 * 
 * Creates a new #GTree.
 * 
 * Return value: a new #GTree.
 **/
GTree*
g_tree_new (GCompareFunc key_compare_func)
{
  g_return_val_if_fail (key_compare_func != NULL, NULL);

  return g_tree_new_full ((GCompareDataFunc) key_compare_func, NULL,
                          NULL, NULL);
}

/**
 * g_tree_new_with_data:
 * @key_compare_func: qsort()-style comparison function.
 * @key_compare_data: data to pass to comparison function.
 * 
 * Creates a new #GTree with a comparison function that accepts user data.
 * See g_tree_new() for more details.
 * 
 * Return value: a new #GTree.
 **/
GTree*
g_tree_new_with_data (GCompareDataFunc key_compare_func,
 		      gpointer         key_compare_data)
{
  g_return_val_if_fail (key_compare_func != NULL, NULL);
  
  return g_tree_new_full (key_compare_func, key_compare_data, 
 			  NULL, NULL);
}

/**
 * g_tree_new_full:
 * @key_compare_func: qsort()-style comparison function.
 * @key_compare_data: data to pass to comparison function.
 * @key_destroy_func: a function to free the memory allocated for the key 
 *   used when removing the entry from the #GTree or #NULL if you don't
 *   want to supply such a function.
 * @value_destroy_func: a function to free the memory allocated for the 
 *   value used when removing the entry from the #GTree or #NULL if you 
 *   don't want to supply such a function.
 * 
 * Creates a new #GTree like g_tree_new() and allows to specify functions 
 * to free the memory allocated for the key and value that get called when 
 * removing the entry from the #GTree.
 * 
 * Return value: a new #GTree.
 **/
GTree*	 
g_tree_new_full (GCompareDataFunc key_compare_func,
 		 gpointer         key_compare_data, 		 
                 GDestroyNotify   key_destroy_func,
 		 GDestroyNotify   value_destroy_func)
{
  GTree *tree;
  
  g_return_val_if_fail (key_compare_func != NULL, NULL);
  
  tree = g_new (GTree, 1);
  tree->root               = NULL;
  tree->key_compare        = key_compare_func;
  tree->key_destroy_func   = key_destroy_func;
  tree->value_destroy_func = value_destroy_func;
  tree->key_compare_data   = key_compare_data;
  
  return tree;
}

/**
 * g_tree_destroy:
 * @tree: a #GTree.
 * 
 * Destroys the #GTree. If keys and/or values are dynamically allocated, you 
 * should either free them first or create the #GTree using g_tree_new_full().
 * In the latter case the destroy functions you supplied will be called on 
 * all keys and values before destroying the #GTree.
 **/
void
g_tree_destroy (GTree *tree)
{
  g_return_if_fail (tree != NULL);

  g_tree_node_destroy (tree->root,
                       tree->key_destroy_func,
 		       tree->value_destroy_func);

  g_free (tree);
}

/**
 * g_tree_insert:
 * @tree: a #Gtree.
 * @key: the key to insert.
 * @value: the value corresponding to the key.
 * 
 * Inserts a key/value pair into a #GTree. If the given key already exists 
 * in the #GTree its corresponding value is set to the new value. If you 
 * supplied a value_destroy_func when creating the #GTree, the old value is 
 * freed using that function. If you supplied a key_destroy_func when 
 * creating the #GTree, the passed key is freed using that function.
 *
 * The tree is automatically 'balanced' as new key/value pairs are added,
 * so that the distance from the root to every leaf is as small as possible.
 **/
void
g_tree_insert (GTree    *tree,
	       gpointer  key,
	       gpointer  value)
{
  gboolean   inserted;

  g_return_if_fail (tree != NULL);

  inserted = FALSE;
  tree->root = g_tree_node_insert (tree,
                                   tree->root,
				   key, value, 
				   FALSE, &inserted);
}

/**
 * g_tree_replace:
 * @tree: a #Gtree.
 * @key: the key to insert.
 * @value: the value corresponding to the key.
 * 
 * Inserts a new key and value into a #GTree similar to g_tree_insert(). 
 * The difference is that if the key already exists in the #GTree, it gets 
 * replaced by the new key. If you supplied a value_destroy_func when 
 * creating the #GTree, the old value is freed using that function. If you 
 * supplied a key_destroy_func when creating the #GTree, the old key is 
 * freed using that function. 
 *
 * The tree is automatically 'balanced' as new key/value pairs are added,
 * so that the distance from the root to every leaf is as small as possible.
 **/
void
g_tree_replace (GTree    *tree,
		gpointer  key,
		gpointer  value)
{
  gboolean   inserted;

  g_return_if_fail (tree != NULL);

  inserted = FALSE;
  tree->root = g_tree_node_insert (tree,
                                   tree->root,
				   key, value, 
				   TRUE, &inserted);
}

/**
 * g_tree_remove:
 * @tree: a #Gtree.
 * @key: the key to remove.
 * 
 * Removes a key/value pair from a #GTree.
 *
 * If the #GTree was created using g_tree_new_full(), the key and value 
 * are freed using the supplied destroy_functions, otherwise you have to 
 * make sure that any dynamically allocated values are freed yourself.
 **/
void
g_tree_remove (GTree         *tree,
	       gconstpointer  key)
{
  g_return_if_fail (tree != NULL);

  tree->root = g_tree_node_remove (tree, tree->root, key, TRUE);
}

/**
 * g_tree_steal:
 * @tree: a #Gtree.
 * @key: the key to remove.
 * 
 * Removes a key and its associated value from a #GTree without calling 
 * the key and value destroy functions.
 **/
void
g_tree_steal (GTree         *tree,
              gconstpointer  key)
{
  g_return_if_fail (tree != NULL);

  tree->root = g_tree_node_remove (tree, tree->root, key, FALSE);
}

/**
 * g_tree_lookup:
 * @tree: a #GTree.
 * @key: the key to look up.
 * 
 * Gets the value corresponding to the given key. Since a #GTree is 
 * automatically balanced as key/value pairs are added, key lookup is very 
 * fast.
 *
 * Return value: the value corresponding to the key.
 **/
gpointer
g_tree_lookup (GTree         *tree,
	       gconstpointer  key)
{
  GTreeNode *node;

  g_return_val_if_fail (tree != NULL, NULL);

  node = g_tree_node_lookup (tree->root, 
                             tree->key_compare, tree->key_compare_data, key);

  return node ? node->value : NULL;
}

/**
 * g_tree_lookup_extended:
 * @tree: a #GTree.
 * @lookup_key: the key to look up.
 * @orig_key: returns the original key.
 * @value: returns the value associated with the key.
 * 
 * Looks up a key in the #GTree, returning the original key and the
 * associated value and a gboolean which is TRUE if the key was found. This 
 * is useful if you need to free the memory allocated for the original key, 
 * for example before calling g_tree_remove().
 * 
 * Return value: #TRUE if the key was found in the #GTree.
 **/
gboolean
g_tree_lookup_extended (GTree         *tree,
                        gconstpointer  lookup_key,
                        gpointer      *orig_key,
                        gpointer      *value)
{
  GTreeNode *node;
  
  g_return_val_if_fail (tree != NULL, FALSE);
  
  node = g_tree_node_lookup (tree->root, 
                             tree->key_compare, tree->key_compare_data, lookup_key);

  if (node)
    {
      if (orig_key)
        *orig_key = node->key;
      if (value)
        *value = node->value;
      return TRUE;
    }
  else
    return FALSE;
}

/**
 * g_tree_foreach:
 * @tree: a #GTree.
 * @func: the function to call for each node visited. If this function
 *   returns TRUE, the traversal is stopped.
 * @user_data: user data to pass to the function.
 * 
 * Calls the given function for each of the key/value pairs in the #GTree.
 * The function is passed the key and value of each pair, and the given
 * @data parameter.
 **/
void
g_tree_foreach (GTree         *tree,
                GTraverseFunc  func,
                gpointer       user_data)
{
  g_return_if_fail (tree != NULL);
  
  if (!tree->root)
    return;

  g_tree_node_in_order (tree->root, func, user_data);
}

/**
 * g_tree_traverse:
 * @tree: a #GTree.
 * @traverse_func: the function to call for each node visited. If this 
 *   function returns TRUE, the traversal is stopped.
 * @traverse_type: the order in which nodes are visited, one of %G_IN_ORDER,
 *   %G_PRE_ORDER and %G_POST_ORDER.
 * @user_data: user data to pass to the function.
 * 
 * Calls the given function for each node in the GTree. This function is
 * deprecated, use g_tree_foreach() instead.
 **/
void
g_tree_traverse (GTree         *tree,
		 GTraverseFunc  traverse_func,
		 GTraverseType  traverse_type,
		 gpointer       user_data)
{
  g_return_if_fail (tree != NULL);

  if (!tree->root)
    return;

  switch (traverse_type)
    {
    case G_PRE_ORDER:
      g_tree_node_pre_order (tree->root, traverse_func, user_data);
      break;

    case G_IN_ORDER:
      g_tree_node_in_order (tree->root, traverse_func, user_data);
      break;

    case G_POST_ORDER:
      g_tree_node_post_order (tree->root, traverse_func, user_data);
      break;
    
    case G_LEVEL_ORDER:
      g_warning ("g_tree_traverse(): traverse type G_LEVEL_ORDER isn't implemented.");
      break;
    }
}

/**
 * g_tree_search:
 * @tree: a #GTree.
 * @search_func: the comparison function used to search the #GTree. 
 * @user_data: the data passed as the second argument to the @search_func 
 * function.
 * 
 * Searches a #GTree using an alternative form of the comparison function.
 *
 * This function is not as useful as it sounds.
 * It allows you to use a different function for performing the lookup of
 * a key. However, since the tree is ordered according to the @key_compare_func
 * function passed to g_tree_new(), the function you pass to g_tree_search() 
 * must return exactly the same value as would be returned by the comparison 
 * function, for each pair of tree nodes, or the search will not work.
 * 
 * To search for a specific value, you can use g_tree_foreach() or 
 * g_tree_traverse().
 *
 * Return value: the value corresponding to the found key, or NULL if the key 
 * is not found.
 **/
gpointer
g_tree_search (GTree         *tree,
	       GCompareFunc   search_func,
	       gconstpointer  user_data)
{
  g_return_val_if_fail (tree != NULL, NULL);

  if (tree->root)
    return g_tree_node_search (tree->root, search_func, user_data);
  else
    return NULL;
}

/**
 * g_tree_height:
 * @tree: a #GTree.
 * 
 * Gets the height of a #GTree.
 *
 * If the #GTree contains no nodes, the height is 0.
 * If the #GTree contains only one root node the height is 1.
 * If the root node has children the height is 2, etc.
 * 
 * Return value: the height of the #GTree.
 **/
gint
g_tree_height (GTree *tree)
{
  g_return_val_if_fail (tree != NULL, 0);

  if (tree->root)
    return g_tree_node_height (tree->root);
  else
    return 0;
}

/**
 * g_tree_nnodes:
 * @tree: a #GTree.
 * 
 * Gets the number of nodes in a #GTree.
 * 
 * Return value: the number of nodes in the #GTree.
 **/
gint
g_tree_nnodes (GTree *tree)
{
  g_return_val_if_fail (tree != NULL, 0);

  if (tree->root)
    return g_tree_node_count (tree->root);
  else
    return 0;
}

static GTreeNode*
g_tree_node_insert (GTree     *tree,
                    GTreeNode *node,
		    gpointer   key,
		    gpointer   value,
                    gboolean   replace,
		    gboolean  *inserted)
{
  gint  old_balance;
  gint  cmp;

  if (!node)
    {
      *inserted = TRUE;
      return g_tree_node_new (key, value);
    }

  cmp = tree->key_compare (key, node->key, tree->key_compare_data);
  if (cmp == 0)
    {
      *inserted = FALSE;

      if (tree->value_destroy_func)
	tree->value_destroy_func (node->value);

      node->value = value;
      
      if (replace)
	{
	  if (tree->key_destroy_func)
	    tree->key_destroy_func (node->key);

	  node->key = key;
	}
      else
	{
	  /* free the passed key */
	  if (tree->key_destroy_func)
	    tree->key_destroy_func (key);
	}

      return node;
    }

  if (cmp < 0)
    {
      if (node->left)
	{
	  old_balance = node->left->balance;
	  node->left = g_tree_node_insert (tree,
                                           node->left,
					   key, value,
					   replace, inserted);

	  if ((old_balance != node->left->balance) && node->left->balance)
	    node->balance -= 1;
	}
      else
	{
	  *inserted = TRUE;
	  node->left = g_tree_node_new (key, value);
	  node->balance -= 1;
	}
    }
  else if (cmp > 0)
    {
      if (node->right)
	{
	  old_balance = node->right->balance;
	  node->right = g_tree_node_insert (tree,
                                            node->right,
					    key, value, 
					    replace, inserted);

	  if ((old_balance != node->right->balance) && node->right->balance)
	    node->balance += 1;
	}
      else
	{
	  *inserted = TRUE;
	  node->right = g_tree_node_new (key, value);
	  node->balance += 1;
	}
    }

  if (*inserted)
    {
      if ((node->balance < -1) || (node->balance > 1))
	node = g_tree_node_balance (node);
    }

  return node;
}

static GTreeNode*
g_tree_node_remove (GTree         *tree,
                    GTreeNode     *node,
		    gconstpointer  key,
                    gboolean       notify)
{
  GTreeNode *new_root;
  gint old_balance;
  gint cmp;

  if (!node)
    return NULL;

  cmp = tree->key_compare (key, node->key, tree->key_compare_data);
  if (cmp == 0)
    {
      GTreeNode *garbage;

      garbage = node;

      if (!node->right)
	{
	  node = node->left;
	}
      else
	{
	  old_balance = node->right->balance;
	  node->right = g_tree_node_remove_leftmost (node->right, &new_root);
	  new_root->left = node->left;
	  new_root->right = node->right;
	  new_root->balance = node->balance;
	  node = g_tree_node_restore_right_balance (new_root, old_balance);
	}

      if (notify)
        {
          if (tree->key_destroy_func)
            tree->key_destroy_func (garbage->key);
          if (tree->value_destroy_func)
            tree->value_destroy_func (garbage->value);
        }

#ifdef ENABLE_GC_FRIENDLY
      garbage->left = NULL;
      garbage->key = NULL;
      garbage->value = NULL;
#endif /* ENABLE_GC_FRIENDLY */

      G_LOCK (g_tree_global);
      garbage->right = node_free_list;
      node_free_list = garbage;
      G_UNLOCK (g_tree_global);
   }
  else if (cmp < 0)
    {
      if (node->left)
	{
	  old_balance = node->left->balance;
	  node->left = g_tree_node_remove (tree, node->left, key, notify);
	  node = g_tree_node_restore_left_balance (node, old_balance);
	}
    }
  else if (cmp > 0)
    {
      if (node->right)
	{
	  old_balance = node->right->balance;
	  node->right = g_tree_node_remove (tree, node->right, key, notify);
	  node = g_tree_node_restore_right_balance (node, old_balance);
	}
    }

  return node;
}

static GTreeNode*
g_tree_node_balance (GTreeNode *node)
{
  if (node->balance < -1)
    {
      if (node->left->balance > 0)
	node->left = g_tree_node_rotate_left (node->left);
      node = g_tree_node_rotate_right (node);
    }
  else if (node->balance > 1)
    {
      if (node->right->balance < 0)
	node->right = g_tree_node_rotate_right (node->right);
      node = g_tree_node_rotate_left (node);
    }

  return node;
}

static GTreeNode*
g_tree_node_remove_leftmost (GTreeNode  *node,
			     GTreeNode **leftmost)
{
  gint old_balance;

  if (!node->left)
    {
      *leftmost = node;
      return node->right;
    }

  old_balance = node->left->balance;
  node->left = g_tree_node_remove_leftmost (node->left, leftmost);
  return g_tree_node_restore_left_balance (node, old_balance);
}

static GTreeNode*
g_tree_node_restore_left_balance (GTreeNode *node,
				  gint       old_balance)
{
  if (!node->left)
    node->balance += 1;
  else if ((node->left->balance != old_balance) &&
	   (node->left->balance == 0))
    node->balance += 1;

  if (node->balance > 1)
    return g_tree_node_balance (node);
  return node;
}

static GTreeNode*
g_tree_node_restore_right_balance (GTreeNode *node,
				   gint       old_balance)
{
  if (!node->right)
    node->balance -= 1;
  else if ((node->right->balance != old_balance) &&
	   (node->right->balance == 0))
    node->balance -= 1;

  if (node->balance < -1)
    return g_tree_node_balance (node);
  return node;
}

static GTreeNode *
g_tree_node_lookup (GTreeNode        *node,
		    GCompareDataFunc  compare,
		    gpointer          compare_data,
		    gconstpointer     key)
{
  gint cmp;

  if (!node)
    return NULL;

  cmp = (* compare) (key, node->key, compare_data);
  if (cmp == 0)
    return node;

  if (cmp < 0)
    {
      if (node->left)
	return g_tree_node_lookup (node->left, compare, compare_data, key);
    }
  else if (cmp > 0)
    {
      if (node->right)
	return g_tree_node_lookup (node->right, compare, compare_data, key);
    }

  return NULL;
}

static gint
g_tree_node_count (GTreeNode *node)
{
  gint count;

  count = 1;
  if (node->left)
    count += g_tree_node_count (node->left);
  if (node->right)
    count += g_tree_node_count (node->right);

  return count;
}

static gint
g_tree_node_pre_order (GTreeNode     *node,
		       GTraverseFunc  traverse_func,
		       gpointer       data)
{
  if ((*traverse_func) (node->key, node->value, data))
    return TRUE;
  if (node->left)
    {
      if (g_tree_node_pre_order (node->left, traverse_func, data))
	return TRUE;
    }
  if (node->right)
    {
      if (g_tree_node_pre_order (node->right, traverse_func, data))
	return TRUE;
    }

  return FALSE;
}

static gint
g_tree_node_in_order (GTreeNode     *node,
		      GTraverseFunc  traverse_func,
		      gpointer       data)
{
  if (node->left)
    {
      if (g_tree_node_in_order (node->left, traverse_func, data))
	return TRUE;
    }
  if ((*traverse_func) (node->key, node->value, data))
    return TRUE;
  if (node->right)
    {
      if (g_tree_node_in_order (node->right, traverse_func, data))
	return TRUE;
    }

  return FALSE;
}

static gint
g_tree_node_post_order (GTreeNode     *node,
			GTraverseFunc  traverse_func,
			gpointer       data)
{
  if (node->left)
    {
      if (g_tree_node_post_order (node->left, traverse_func, data))
	return TRUE;
    }
  if (node->right)
    {
      if (g_tree_node_post_order (node->right, traverse_func, data))
	return TRUE;
    }
  if ((*traverse_func) (node->key, node->value, data))
    return TRUE;

  return FALSE;
}

static gpointer
g_tree_node_search (GTreeNode     *node,
		    GCompareFunc   search_func,
		    gconstpointer  data)
{
  gint dir;

  if (!node)
    return NULL;

  do {
    dir = (* search_func) (node->key, data);
    if (dir == 0)
      return node->value;

    if (dir < 0)
      node = node->left;
    else if (dir > 0)
      node = node->right;
  } while (node);

  return NULL;
}

static gint
g_tree_node_height (GTreeNode *node)
{
  gint left_height;
  gint right_height;

  if (node)
    {
      left_height = 0;
      right_height = 0;

      if (node->left)
	left_height = g_tree_node_height (node->left);

      if (node->right)
	right_height = g_tree_node_height (node->right);

      return MAX (left_height, right_height) + 1;
    }

  return 0;
}

static GTreeNode*
g_tree_node_rotate_left (GTreeNode *node)
{
  GTreeNode *right;
  gint a_bal;
  gint b_bal;

  right = node->right;

  node->right = right->left;
  right->left = node;

  a_bal = node->balance;
  b_bal = right->balance;

  if (b_bal <= 0)
    {
      if (a_bal >= 1)
	right->balance = b_bal - 1;
      else
	right->balance = a_bal + b_bal - 2;
      node->balance = a_bal - 1;
    }
  else
    {
      if (a_bal <= b_bal)
	right->balance = a_bal - 2;
      else
	right->balance = b_bal - 1;
      node->balance = a_bal - b_bal - 1;
    }

  return right;
}

static GTreeNode*
g_tree_node_rotate_right (GTreeNode *node)
{
  GTreeNode *left;
  gint a_bal;
  gint b_bal;

  left = node->left;

  node->left = left->right;
  left->right = node;

  a_bal = node->balance;
  b_bal = left->balance;

  if (b_bal <= 0)
    {
      if (b_bal > a_bal)
	left->balance = b_bal + 1;
      else
	left->balance = a_bal + 2;
      node->balance = a_bal - b_bal + 1;
    }
  else
    {
      if (a_bal <= -1)
	left->balance = b_bal + 1;
      else
	left->balance = a_bal + b_bal + 2;
      node->balance = a_bal + 1;
    }

  return left;
}

static void
g_tree_node_check (GTreeNode *node)
{
  gint left_height;
  gint right_height;
  gint balance;
  
  if (node)
    {
      left_height = 0;
      right_height = 0;
      
      if (node->left)
	left_height = g_tree_node_height (node->left);
      if (node->right)
	right_height = g_tree_node_height (node->right);
      
      balance = right_height - left_height;
      if (balance != node->balance)
	g_log (g_log_domain_glib, G_LOG_LEVEL_INFO,
	       "g_tree_node_check: failed: %d ( %d )\n",
	       balance, node->balance);
      
      if (node->left)
	g_tree_node_check (node->left);
      if (node->right)
	g_tree_node_check (node->right);
    }
}
