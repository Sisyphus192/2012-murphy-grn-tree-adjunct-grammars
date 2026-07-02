"""Tree-Adjunct Grammatical Evolution (TAGE) genotype->phenotype mapping.

Faithful implementation of the TAGE derivation from Murphy, O'Neill,
Galvan-Lopez & Brabazon, "Tree-Adjunct Grammatical Evolution" (IEEE WCCI 2010) --
the algorithm the 2012 GRN paper extends.

A context-free grammar is transformed into a tree-adjoining grammar (TAG):

* **Initial trees** ``I`` -- every derivation tree rooted at the start symbol
  built using ONLY non-recursive productions, fully expanded so every leaf is a
  terminal (the grammars here are lexicalised: no substitution nodes).
* **Auxiliary trees** ``A`` of type ``X`` -- minimal recursive structures: take a
  recursive production containing ``X``, make one ``X`` occurrence the *foot*
  (a leaf re-labelled ``X``), and expand every other child via non-recursive
  productions (cartesian product over the choices).

Only the **adjunction** composition operation is used: adjoining an auxiliary
tree of type ``X`` at an ``X``-node ``v`` replaces ``v`` with a copy of the
auxiliary tree and re-attaches the subtree formerly at ``v`` at the auxiliary
tree's foot. Because initial trees are complete (all-terminal frontier) and
adjunction preserves completeness, the derived tree is a VALID phenotype at every
stage -- so any codon string maps to something runnable regardless of length
(the property the 2012 GRN front-end relies on: the GRN emits a variable number
of codons). Running out of codons simply stops the derivation with the current
complete tree -- no dangling non-terminals, no padding, no forced wrapping.

Codon consumption (paper sec. III-B, ``i mod c`` GE mapping):
  1. one codon selects the initial tree: ``I[codon % len(I)]``;
  2. each adjunction consumes two codons -- one selects the adjunction address
     ``N[codon % len(N)]`` from the current adjunctable nodes, one selects the
     auxiliary tree ``A_l[codon % len(A_l)]`` whose root matches that node's
     label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from grntage.grammar.definitions import Grammar


@dataclass
class TreeNode:
    """A node in an elementary or derived tree.

    ``is_foot`` marks the single non-terminal leaf of an auxiliary tree (the foot
    node, labelled the same as the tree's root); it is re-attached to during
    adjunction and never persists in a derived tree.
    """

    label: str
    children: list[TreeNode] = field(default_factory=list)
    is_foot: bool = False

    def copy(self) -> TreeNode:
        """Deep copy (elementary trees are templates; each use gets a fresh copy)."""
        return TreeNode(
            self.label, [c.copy() for c in self.children], is_foot=self.is_foot
        )


def _preorder(root: TreeNode) -> list[TreeNode]:
    """Nodes of a tree in pre-order (root, then children left-to-right)."""
    out: list[TreeNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        out.append(node)
        # push children reversed so they pop left-to-right (pre-order)
        stack.extend(reversed(node.children))
    return out


def _frontier(root: TreeNode) -> list[str]:
    """Left-to-right terminal leaves of the derived tree (the phenotype tokens)."""
    out: list[str] = []
    for node in _preorder(root):
        if not node.children:
            out.append(node.label)
    return out


def _find_foot(node: TreeNode) -> TreeNode | None:
    for n in _preorder(node):
        if n.is_foot:
            return n
    return None


class TAG:
    """A tree-adjoining grammar derived from a context-free :class:`Grammar`.

    Attributes:
        initial_trees: list of complete trees rooted at the start symbol (``I``)
        aux_trees: maps a non-terminal label to its auxiliary trees (``A``)
        adjunctable_labels: labels that have at least one auxiliary tree
    """

    def __init__(self, grammar: Grammar) -> None:
        self.grammar = grammar
        self._nonterminals = set(grammar.productions.keys())

        recursive, non_recursive = self._classify_productions()
        self._non_recursive = non_recursive

        self.initial_trees: list[TreeNode] = self._generate_initial(
            grammar.start_symbol
        )
        if not self.initial_trees:
            raise ValueError(
                f"grammar {grammar.name!r} produced no initial trees "
                "(every non-terminal needs a non-recursive production)"
            )

        self.aux_trees: dict[str, list[TreeNode]] = {}
        for lhs, prods in recursive.items():
            trees = self._generate_auxiliary(lhs, prods)
            if trees:
                self.aux_trees[lhs] = trees
        self.adjunctable_labels = frozenset(self.aux_trees)

    # -- CFG -> TAG construction ------------------------------------------------

    def _reachable(self) -> dict[str, set[str]]:
        """For each non-terminal, the set of non-terminals reachable from it."""
        succ: dict[str, set[str]] = {nt: set() for nt in self._nonterminals}
        for lhs, prods in self.grammar.productions.items():
            for rhs in prods:
                for sym in rhs:
                    if sym in self._nonterminals:
                        succ[lhs].add(sym)
        reach: dict[str, set[str]] = {}
        for nt in self._nonterminals:
            seen: set[str] = set()
            stack = list(succ[nt])
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                stack.extend(succ[cur] - seen)
            reach[nt] = seen
        return reach

    def _classify_productions(
        self,
    ) -> tuple[dict[str, list[tuple[str, ...]]], dict[str, list[tuple[str, ...]]]]:
        """Split each non-terminal's productions into (recursive, non-recursive).

        A production ``A -> rhs`` is recursive if some non-terminal ``B`` in
        ``rhs`` can reach ``A`` (so ``A -> ... -> A`` forms a cycle through it).
        """
        reach = self._reachable()
        recursive: dict[str, list[tuple[str, ...]]] = {}
        non_recursive: dict[str, list[tuple[str, ...]]] = {}
        for lhs, prods in self.grammar.productions.items():
            for rhs in prods:
                is_rec = any(
                    sym in self._nonterminals and lhs in reach[sym] for sym in rhs
                )
                if is_rec:
                    # Only direct self-recursion (A in its own rhs) is supported;
                    # multi-non-terminal cycles would need the general base-cycle
                    # construction. None of the paper's grammars use them.
                    if lhs not in rhs:
                        raise NotImplementedError(
                            f"indirect recursion in {lhs!r} -> {rhs!r} is not "
                            "supported (only direct self-recursion)"
                        )
                    recursive.setdefault(lhs, []).append(rhs)
                else:
                    non_recursive.setdefault(lhs, []).append(rhs)
        return recursive, non_recursive

    def _generate_initial(self, symbol: str) -> list[TreeNode]:
        """All complete trees from ``symbol`` using only non-recursive productions."""
        if self.grammar.is_terminal(symbol):
            return [TreeNode(symbol)]
        trees: list[TreeNode] = []
        for rhs in self._non_recursive.get(symbol, []):
            child_options = [self._generate_initial(s) for s in rhs]
            for combo in product(*child_options):
                trees.append(TreeNode(symbol, [c.copy() for c in combo]))
        return trees

    def _generate_auxiliary(
        self, lhs: str, recursive_prods: list[tuple[str, ...]]
    ) -> list[TreeNode]:
        """Minimal recursive (auxiliary) trees rooted at ``lhs`` with one foot."""
        trees: list[TreeNode] = []
        for rhs in recursive_prods:
            foot_positions = [i for i, s in enumerate(rhs) if s == lhs]
            for foot_pos in foot_positions:
                child_options: list[list[TreeNode]] = []
                for i, sym in enumerate(rhs):
                    if i == foot_pos:
                        child_options.append([TreeNode(lhs, is_foot=True)])
                    else:
                        # non-foot children expand via non-recursive productions
                        child_options.append(self._generate_initial(sym))
                for combo in product(*child_options):
                    trees.append(TreeNode(lhs, [c.copy() for c in combo]))
        return trees

    # -- derivation -------------------------------------------------------------

    def derive(self, codons: list[int]) -> list[str]:
        """Map a codon list to a phenotype token list via TAGE derivation."""
        if not codons:
            # Defensive: an empty codon stream still yields a valid phenotype.
            return _frontier(self.initial_trees[0].copy())

        root = self.initial_trees[codons[0] % len(self.initial_trees)].copy()
        idx = 1

        if self.adjunctable_labels:
            n_codons = len(codons)
            # Each adjunction needs two codons (address, auxiliary tree).
            while n_codons - idx >= 2:
                addresses = [
                    node
                    for node in _preorder(root)
                    if node.label in self.adjunctable_labels and not node.is_foot
                ]
                if not addresses:
                    break
                target = addresses[codons[idx] % len(addresses)]
                idx += 1
                options = self.aux_trees[target.label]
                beta = options[codons[idx] % len(options)]
                idx += 1
                self._adjoin(target, beta)

        return _frontier(root)

    @staticmethod
    def _adjoin(node: TreeNode, beta_template: TreeNode) -> None:
        """Adjoin a copy of ``beta_template`` at ``node`` (in place).

        ``node`` becomes the auxiliary tree's root; the subtree formerly at
        ``node`` is re-attached at the auxiliary tree's foot.
        """
        beta = beta_template.copy()
        foot = _find_foot(beta)
        if foot is None:  # pragma: no cover - auxiliary trees always have a foot
            raise ValueError("auxiliary tree has no foot node")
        # detach the old subtree at `node`
        old_label, old_children = node.label, node.children
        # place the old subtree at the foot
        foot.label = old_label
        foot.children = old_children
        foot.is_foot = False
        # `node` becomes beta's root (same label as `node`)
        node.label = beta.label
        node.children = beta.children


_TAG_CACHE: dict[str, TAG] = {}


def build_tag(grammar: Grammar) -> TAG:
    """Return the TAG for ``grammar`` (cached per grammar; the TAG is fixed)."""
    tag = _TAG_CACHE.get(grammar.name)
    if tag is None:
        tag = TAG(grammar)
        _TAG_CACHE[grammar.name] = tag
    return tag
