use ff::{BitIterator, Field, PrimeField};
use pairing::bn256::{Bn256, Fr};
use sapling_crypto::babyjubjub::JubjubBn256;

/// Binary Tree where leaves hold a stand-alone value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tree {
    Empty {
        hash: pairing::bn256::Fr,
        parent: Option<Box<Tree>>,
    },
    Node {
        hash: pairing::bn256::Fr,
        left: Box<Tree>,
        right: Box<Tree>,
        parent: Option<Box<Tree>>,
    },
}

impl Tree {
    /// Returns a hash from the tree.
    pub fn hash(&self) -> &pairing::bn256::Fr {
        match *self {
            Tree::Empty { ref hash, .. } => hash,
            Tree::Node { ref hash, .. } => hash,
        }
    }
}

#[derive(Debug)]
pub struct MerkleTree {
    pub root: Tree,
}

pub fn create_leaf_from_preimage(
    nullifier: pairing::bn256::Fr,
    secret: pairing::bn256::Fr,
) -> Tree {
    let params = &JubjubBn256::new();
    let mut lhs: Vec<bool> = BitIterator::new(nullifier.into_repr()).collect();
    let mut rhs: Vec<bool> = BitIterator::new(secret.into_repr()).collect();
    lhs.reverse();
    rhs.reverse();
    let hash = sapling_crypto::baby_pedersen_hash::pedersen_hash::<Bn256, _>(
        sapling_crypto::baby_pedersen_hash::Personalization::NoteCommitment,
        lhs.into_iter()
            .take(Fr::NUM_BITS as usize)
            .chain(rhs.into_iter().take(Fr::NUM_BITS as usize)),
        params,
    )
    .into_xy()
    .0;
    return Tree::Empty { hash, parent: None };
}

pub fn create_leaf_list(mut nodes: Vec<pairing::bn256::Fr>, depth: usize) -> Vec<Box<Tree>> {
    for _ in 0..((2 << (depth - 1)) - nodes.len()) {
        nodes.push(<pairing::bn256::Fr>::zero());
    }

    let mut tree_nodes: Vec<Box<Tree>> = vec![];
    for i in 0..nodes.len() {
        tree_nodes.push(Box::new(Tree::Empty {
            hash: nodes[i],
            parent: None,
        }));
    }

    return tree_nodes;
}

#[allow(dead_code)]
pub fn build_merkle_tree_with_proof(
    nodes: Vec<Box<Tree>>,
    depth: usize,
    top_depth: usize,
    target_node: pairing::bn256::Fr,
    curr_list: Vec<Option<(bool, pairing::bn256::Fr)>>,
) -> (MerkleTree, Vec<Option<(bool, pairing::bn256::Fr)>>) {
    let (mut new_nodes, target_node, new_curr_list) =
        hash_nodes_rec(nodes, depth, top_depth, target_node, curr_list);
    if new_nodes.len() == 1 {
        let root = new_nodes.remove(0);
        return (MerkleTree { root: *root }, new_curr_list);
    } else {
        return build_merkle_tree_with_proof(
            new_nodes,
            depth - 1,
            top_depth,
            target_node,
            new_curr_list,
        );
    }
}

#[allow(dead_code)]
pub fn hash_nodes_rec(
    mut nodes: Vec<Box<Tree>>,
    depth: usize,
    top_depth: usize,
    mut target_node: pairing::bn256::Fr,
    mut curr_list: Vec<Option<(bool, pairing::bn256::Fr)>>,
) -> (
    Vec<Box<Tree>>,
    pairing::bn256::Fr,
    Vec<Option<(bool, pairing::bn256::Fr)>>,
) {
    if nodes.len() == 2 {
        let left = nodes.remove(0);
        let right = nodes.remove(0);
        let temp_bool = target_node == *left.hash() || target_node == *right.hash();
        let mut val = vec![];
        if target_node == *left.hash() {
            val.push(Some((true, *right.hash())));
        }

        if target_node == *right.hash() {
            val.push(Some((false, *left.hash())));
        }

        let cur = hash_leaf_pair(top_depth - depth, *left, *right);
        if temp_bool {
            target_node = *cur.hash();
        }
        if depth == 1 {
            curr_list.append(&mut val);
            return (vec![cur], target_node, curr_list);
        } else {
            return (vec![cur], target_node, val);
        }
    } else {
        let (mut left_new_nodes, left_target_node, mut left_new_curr_list) = hash_nodes_rec(
            nodes[..(nodes.len() / 2)].to_vec(),
            depth,
            top_depth,
            target_node,
            curr_list.clone(),
        );
        let (mut right_new_nodes, right_target_node, mut right_new_curr_list) = hash_nodes_rec(
            nodes[(nodes.len() / 2)..].to_vec(),
            depth,
            top_depth,
            target_node,
            curr_list.clone(),
        );

        if left_target_node == target_node {
            target_node = right_target_node;
        } else {
            target_node = left_target_node;
        }

        left_new_nodes.append(&mut right_new_nodes);
        curr_list.append(&mut left_new_curr_list);
        curr_list.append(&mut right_new_curr_list);

        return (left_new_nodes, target_node, curr_list);
    }
}

pub fn hash_leaf_pair(index: usize, lhs: Tree, rhs: Tree) -> Box<Tree> {
    let params = &JubjubBn256::new();
    let mut lhs_bool: Vec<bool> = BitIterator::new((lhs).hash().into_repr()).collect();
    let mut rhs_bool: Vec<bool> = BitIterator::new((rhs).hash().into_repr()).collect();
    lhs_bool.reverse();
    rhs_bool.reverse();
    let personalization =
        sapling_crypto::baby_pedersen_hash::Personalization::MerkleTree(index as usize);
    let hash = sapling_crypto::baby_pedersen_hash::pedersen_hash::<Bn256, _>(
        personalization,
        lhs_bool
            .clone()
            .into_iter()
            .take(Fr::NUM_BITS as usize)
            .chain(rhs_bool.clone().into_iter().take(Fr::NUM_BITS as usize)),
        params,
    )
    .into_xy()
    .0;
    return Box::new(Tree::Node {
        hash,
        left: Box::new(lhs),
        right: Box::new(rhs),
        parent: None,
    });
}

pub fn compute_root_from_proof(
    leaf: pairing::bn256::Fr,
    path: Vec<Option<(bool, pairing::bn256::Fr)>>,
) -> pairing::bn256::Fr {
    let mut hash = leaf;
    for i in 0..path.len() {
        match path[i] {
            Some((right_side, pt)) => {
                if right_side {
                    hash = *hash_leaf_pair(
                        i,
                        Tree::Empty {
                            hash: hash,
                            parent: None,
                        },
                        Tree::Empty {
                            hash: pt,
                            parent: None,
                        },
                    )
                    .hash();
                } else {
                    hash = *hash_leaf_pair(
                        i,
                        Tree::Empty {
                            hash: pt,
                            parent: None,
                        },
                        Tree::Empty {
                            hash: hash,
                            parent: None,
                        },
                    )
                    .hash();
                }
            }
            None => {}
        }
    }

    return hash;
}
