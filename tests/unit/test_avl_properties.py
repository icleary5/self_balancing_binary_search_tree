"""Unit tests for AVL tree properties using Hypothesis for property-based testing.

Each test checks a fundamental invariant of AVL (self-balancing binary search) trees:
- test_bst_ordering: verifies the BST property (left < node < right for all nodes)
- test_height_consistency: verifies that the stored height at each node matches the computed height
- test_balance_factor: verifies that the AVL balance factor property holds (|left height - right height| <= 1)
- test_inorder_sorted_uniqueness: verifies that in-order traversal yields a sorted list of unique keys
- test_search_consistency: verifies that search returns True for present keys and False for absent keys after insertions and deletions
"""

from hypothesis import given, strategies as st
from sbbst import sbbst, TreeNode

import math

# Hypothesis strategy for generating pairs of (insertions, deletions) lists.
# - Generates a list of unique integers (insertions) in the range [0, 1000], up to 100 elements.
# - For each insertions list, generates a deletions list:
#     - Also unique integers in [0, 1000].
#     - Length up to the number of insertions.
#     - Deletions may or may not overlap with insertions.
# This strategy is used to test AVL tree properties under various insertion and deletion scenarios.
inserts_and_deletes = st.lists(
    st.integers(min_value=0, max_value=1000), unique=True, max_size=100
).flatmap(
    lambda inserts: st.tuples(
        st.just(inserts),
        st.lists(
            st.integers(min_value=0, max_value=1000),
            unique=True,
            max_size=len(inserts),
        ),
    )
)

def construct_tree(insertions, deletions):
    """Helper function to construct a tree from insertions and deletions."""
    tree = sbbst()
    for x in insertions:
        tree.insert(x)
    for x in deletions:
        tree.delete(x)
    return tree


def is_bst(node, min_key=-math.inf, max_key=math.inf) -> bool:
    """Check if the tree rooted at `node` satisfies the BST property."""
    if node is None:
        return True
    if not (min_key < node.val < max_key):
        return False
    return is_bst(node.left, min_key, node.val) and is_bst(
        node.right, node.val, max_key
    )


def compute_height(node) -> int:
    """Compute the height of the tree rooted at `node`."""
    if node is None:
        return 0
    return 1 + max(compute_height(node.left), compute_height(node.right))


@given(inserts_and_deletes)
def test_bst_ordering(data):
    """Test that the tree maintains the binary search tree (BST) ordering property after insertions."""
    insertions, deletions = data
    tree = construct_tree(insertions, deletions)
    assert is_bst(tree.head)

@given(inserts_and_deletes)
def test_height_consistency(data):
    insertions, deletions = data
    """Test that the stored height at each node matches the computed height from its children."""
    tree = construct_tree(insertions, deletions)

    def check(node):
        if node is None:
            return
        lh = compute_height(node.left)
        rh = compute_height(node.right)
        # The height property of each node should be 1 + max(left height, right height)
        assert node.height == 1 + max(lh, rh)
        check(node.left)
        check(node.right)

    check(tree.head)


@given(inserts_and_deletes)
def test_balance_factor(data):
    """Test that the tree maintains the AVL balance factor property after insertions."""
    insertions, deletions = data
    tree = construct_tree(insertions, deletions)

    def check(node):
        if node is None:
            return
        lh = compute_height(node.left)
        rh = compute_height(node.right)
        # The balance factor of each node should be |left height - right height| <= 1
        assert abs(lh - rh) <= 1
        check(node.left)
        check(node.right)

    check(tree.head)


@given(inserts_and_deletes)
def test_inorder_sorted_uniqueness(data):
    """Test that in-order traversal of the tree yields a sorted list of unique keys."""
    insertions, deletions = data
    tree = construct_tree(insertions, deletions)
    result = tree.inOrder()  # should return list of keys in-order
    expected = sorted(set(insertions) - set(deletions))
    assert result == expected, f"Expected {expected}, got {result}"


@given(inserts_and_deletes)
def test_search_consistency(data):
    """Test that search returns correct results after a series of insertions and deletions."""
    insertions, deletions = data
    tree = sbbst()
    for x in insertions:
        tree.insert(x)
    for x in deletions:
        tree.delete(x)
    expected = set(insertions) - set(deletions)
    for key in expected:
        assert (
            tree.search(key) is True
        ), f"Key {key} should be found in the tree"
    for key in set(deletions) - set(insertions):
        assert (
            tree.search(key) is False
        ), f"Key {key} should not be found in the tree"
