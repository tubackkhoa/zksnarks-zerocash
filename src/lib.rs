extern crate bellman;
extern crate blake2_rfc;
extern crate ff;
extern crate num_bigint;
extern crate num_traits;
extern crate pairing;
extern crate rand;
extern crate sapling_crypto;
extern crate time;
extern crate wasm_bindgen;

#[macro_use]
extern crate serde_derive;

extern crate hex;

use wasm_bindgen::prelude::*;

use bellman::{Circuit, ConstraintSystem, SynthesisError};

use ff::{Field, PrimeField};
use sapling_crypto::{
    babyjubjub::JubjubEngine,
    circuit::{
        baby_pedersen_hash,
        boolean::{AllocatedBit, Boolean},
        num::AllocatedNum,
    },
};

use pairing::bn256::Fr;

mod blake_circuit;
mod blake_merkle_tree;
mod merkle_tree;
mod zk_util;

use zk_util::{generate, prove, verify};

/// Circuit for proving knowledge of preimage of leaf in merkle tree
struct MerkleTreeCircuit<'a, E: JubjubEngine> {
    // nullifier
    nullifier: Option<E::Fr>,
    // secret
    secret: Option<E::Fr>,
    proof: Vec<Option<(bool, E::Fr)>>,
    params: &'a E::Params,
}

/// Our demo circuit implements this `Circuit` trait which
/// is used during paramgen and proving in order to
/// synthesize the constraint system.
impl<'a, E: JubjubEngine> Circuit<E> for MerkleTreeCircuit<'a, E> {
    fn synthesize<CS: ConstraintSystem<E>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        // nullifier is the left side of the preimage
        let nullifier = AllocatedNum::alloc(cs.namespace(|| "nullifier"), || {
            Ok(match self.nullifier {
                Some(n) => n,
                None => E::Fr::zero(),
            })
        })?;
        nullifier.inputize(cs.namespace(|| "public input nullifier"))?;
        // secret is the right side of the preimage
        let secret = AllocatedNum::alloc(cs.namespace(|| "secret"), || {
            Ok(match self.secret {
                Some(s) => s,
                None => E::Fr::zero(),
            })
        })?;
        // construct preimage using [nullifier_bits|secret_bits] concatenation
        let mut preimage = vec![];
        preimage.extend(
            nullifier
                .into_bits_le_strict(cs.namespace(|| "nullifier bits"))?
                .into_iter()
                .take(Fr::NUM_BITS as usize),
        );
        preimage.extend(
            secret
                .into_bits_le_strict(cs.namespace(|| "secret bits"))?
                .into_iter()
                .take(Fr::NUM_BITS as usize),
        );
        // compute leaf hash using pedersen hash of preimage
        let mut hash = baby_pedersen_hash::pedersen_hash(
            cs.namespace(|| "computation of leaf pedersen hash"),
            baby_pedersen_hash::Personalization::NoteCommitment,
            &preimage,
            self.params,
        )?
        .get_x()
        .clone();
        // reconstruct merkle root hash using the private merkle path
        for i in 0..self.proof.len() {
            if let Some((ref side, ref element)) = self.proof[i] {
                let elt =
                    AllocatedNum::alloc(cs.namespace(|| format!("elt {}", i)), || Ok(*element))?;
                let right_side = Boolean::from(
                    AllocatedBit::alloc(
                        cs.namespace(|| format!("position bit {}", i)),
                        Some(*side),
                    )
                    .unwrap(),
                );
                // Swap the two if the current subtree is on the right
                let (xl, xr) = AllocatedNum::conditionally_reverse(
                    cs.namespace(|| format!("conditional reversal of preimage {}", i)),
                    &elt,
                    &hash,
                    &right_side,
                )?;
                // build preimage of merkle hash as concatenation of left and right nodes
                let mut preimage = vec![];
                preimage.extend(
                    xl.into_bits_le_strict(cs.namespace(|| format!("xl into bits {}", i)))?,
                );
                preimage.extend(
                    xr.into_bits_le_strict(cs.namespace(|| format!("xr into bits {}", i)))?,
                );
                // Compute the new subtree value
                let personalization = baby_pedersen_hash::Personalization::MerkleTree(i as usize);
                hash = baby_pedersen_hash::pedersen_hash(
                    cs.namespace(|| format!("computation of pedersen hash {}", i)),
                    personalization,
                    &preimage,
                    self.params,
                )?
                .get_x()
                .clone(); // Injective encoding
            }
        }
        hash.inputize(cs)?;
        println!("THE ROOT HASH {:?}", hash.get_value());
        Ok(())
    }
}

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen(catch)]
pub fn generate_tree(seed_slice: &[u32], depth: u32) -> Result<JsValue, JsValue> {
    let res = generate(seed_slice, depth);
    if res.is_ok() {
        Ok(JsValue::from_serde(&res.ok().unwrap()).unwrap())
    } else {
        Err(JsValue::from_str(&res.err().unwrap().to_string()))
    }
}

#[wasm_bindgen(catch)]
pub fn prove_tree(
    seed_slice: &[u32],
    params: &str,
    nullifier_hex: &str,
    secret_hex: &str,
    proof_path_hex: &str,
    proof_path_sides: &str,
) -> Result<JsValue, JsValue> {
    let res = prove(
        seed_slice,
        params,
        nullifier_hex,
        secret_hex,
        proof_path_hex,
        proof_path_sides,
    );
    if res.is_ok() {
        Ok(JsValue::from_serde(&res.ok().unwrap()).unwrap())
    } else {
        Err(JsValue::from_str(&res.err().unwrap().to_string()))
    }
}

#[wasm_bindgen(catch)]
pub fn verify_tree(
    params: &str,
    proof: &str,
    nullifier_hex: &str,
    root_hex: &str,
) -> Result<JsValue, JsValue> {
    let res = verify(params, proof, nullifier_hex, root_hex);
    if res.is_ok() {
        Ok(JsValue::from_serde(&res.ok().unwrap()).unwrap())
    } else {
        Err(JsValue::from_str(&res.err().unwrap().to_string()))
    }
}

#[cfg(test)]
mod test {
    use ff::PrimeField;
    use pairing::bn256::{Bn256, Fr};
    use rand::{ChaChaRng, SeedableRng};
    use sapling_crypto::babyjubjub::JubjubBn256;
    use sapling_crypto::circuit;
    use sapling_crypto::circuit::multipack;
    use sapling_crypto::circuit::sapling::Spend;
    use sapling_crypto::jubjub;
    use sapling_crypto::pedersen_hash;
    use sapling_crypto::primitives;
    use sapling_crypto::primitives::ValueCommitment;
    use std::fs;

    use bellman::Circuit;
    use rand::Rand;
    use rand::Rng;
    use sapling_crypto::circuit::test::TestConstraintSystem;

    use crate::test;

    use super::{generate, prove, verify, MerkleTreeCircuit};
    use blake_circuit::BlakeTreeCircuit;
    use merkle_tree::{build_merkle_tree_with_proof, create_leaf_from_preimage, create_leaf_list};
    use time::PreciseTime;

    #[test]
    fn test_merkle_circuit() {
        let mut cs = TestConstraintSystem::<Bn256>::new();
        let seed_slice = &[1u32, 1u32, 1u32, 1u32];
        let rng = &mut ChaChaRng::from_seed(seed_slice);
        println!("generating setup...");
        let start = PreciseTime::now();

        let mut proof_vec = vec![];
        for _ in 0..32 {
            proof_vec.push(Some((true, Fr::rand(rng))));
        }

        let j_params = &JubjubBn256::new();
        let m_circuit = MerkleTreeCircuit {
            params: j_params,
            nullifier: Some(Fr::rand(rng)),
            secret: Some(Fr::rand(rng)),
            proof: proof_vec,
        };

        m_circuit.synthesize(&mut cs).unwrap();
        println!(
            "setup generated in {} s",
            start.to(PreciseTime::now()).num_milliseconds() as f64 / 1000.0
        );
        println!("num constraints: {}", cs.num_constraints());
        println!("num inputs: {}", cs.num_inputs());
    }

    #[test]
    fn test_generate_params() {
        // let mut cs = TestConstraintSystem::<Bn256>::new();
        let seed_slice = &[1u32, 1u32, 1u32, 1u32];
        let rng = &mut ChaChaRng::from_seed(seed_slice);
        println!("generating setup...");
        let nullifier = Fr::rand(rng);
        let secret = Fr::rand(rng);
        let leaf = *create_leaf_from_preimage(nullifier, secret).hash();
        let mut leaves = vec![leaf];
        for _ in 0..7 {
            leaves.push(Fr::rand(rng));
        }
        let tree_nodes = create_leaf_list(leaves, 3);
        let (_r, proof) = build_merkle_tree_with_proof(tree_nodes, 3, 3, leaf, vec![]);
        println!("THE ROOT HASH IN TEST{:?}", _r.root.hash());
        // let j_params = &JubjubBn256::new();
        // let m_circuit = MerkleTreeCircuit {
        //     params: j_params,
        //     nullifier: Some(nullifier),
        //     secret: Some(secret),
        //     proof: proof.clone(),
        // };
        // m_circuit.synthesize(&mut cs).unwrap();

        let nullifier_hex = &nullifier.to_hex();
        let secret_hex = &secret.to_hex();
        let root_hex = &_r.root.hash().to_hex();
        let mut proof_path_hex: String = "".to_string();
        let mut proof_path_sides: String = "".to_string();
        for inx in 0..proof.len() {
            match proof[inx] {
                Some((right_side, pt)) => {
                    proof_path_hex.push_str(&pt.to_hex());
                    proof_path_sides.push_str(if right_side { &"1" } else { &"0" });
                }
                None => {}
            }
        }
        let params = generate(seed_slice, proof.len() as u32).unwrap().params;
        let proof_hex = prove(
            seed_slice,
            &params,
            nullifier_hex,
            secret_hex,
            &proof_path_hex,
            &proof_path_sides,
        )
        .unwrap();

        fs::write("test/test.params", params).unwrap();
        fs::write("test/test.proof", proof_hex.proof).unwrap();
        let parameters = &String::from_utf8(fs::read("test/test.params").unwrap()).unwrap();
        let the_proof = &String::from_utf8(fs::read("test/test.proof").unwrap()).unwrap();

        // let h = &String::from_utf8(fs::read("test/test_tree.h").unwrap()).unwrap();
        let verify = verify(parameters, the_proof, &nullifier_hex, &root_hex).unwrap();
        // println!("{:?}", cs.which_is_unsatisfied());
        println!("Did the circuit work!? {:?}", verify.result);
    }

    use merkle_tree::compute_root_from_proof;

    #[test]
    fn test_proof_creation() {
        let start = PreciseTime::now();
        let rng = &mut ChaChaRng::from_seed(&[1u32, 1u32, 1u32, 1u32]);
        println!(
            "\nsetup generated in {} s\n\n",
            start.to(PreciseTime::now()).num_milliseconds() as f64 / 1000.0
        );

        let target_leaf = Fr::rand(rng);
        println!(
            "random target created in {} s",
            start.to(PreciseTime::now()).num_milliseconds() as f64 / 1000.0
        );
        let mut leaves: Vec<pairing::bn256::Fr> = vec![1, 2, 3, 4, 5, 6, 7]
            .iter()
            .map(|_| Fr::rand(rng))
            .collect();
        println!(
            "leaves created in {} s",
            start.to(PreciseTime::now()).num_milliseconds() as f64 / 1000.0
        );
        leaves.push(target_leaf);
        println!(
            "leaves pushed in {} s",
            start.to(PreciseTime::now()).num_milliseconds() as f64 / 1000.0
        );
        let tree_nodes = create_leaf_list(leaves, 3);
        println!(
            "leaf list generated in {} s",
            start.to(PreciseTime::now()).num_milliseconds() as f64 / 1000.0
        );
        let (_r, proof) = build_merkle_tree_with_proof(tree_nodes, 3, 3, target_leaf, vec![]);
        println!(
            "tree proof generated in {} s",
            start.to(PreciseTime::now()).num_milliseconds() as f64 / 1000.0
        );
        let _computed_root = compute_root_from_proof(target_leaf, proof);
        println!(
            "computed root generated in {} s",
            start.to(PreciseTime::now()).num_milliseconds() as f64 / 1000.0
        );
        assert!(_computed_root == *_r.root.hash());
    }

    #[test]
    fn create_deposit() {
        // let rng = &mut ChaChaRng::from_seed(&[1u32, 1u32, 1u32, 1u32]);
        let nullifier =
            Fr::from_hex("ab808e80fcdc0f4d598289769260e9c427ce75a42b4b77f3c5a135ed448c55").unwrap();
        let secret =
            Fr::from_hex("32065632608bba12d37a9fa672751a633c3c108cadf963fbfe263ba1be995a").unwrap();
        let leaf = *create_leaf_from_preimage(nullifier, secret).hash();
        println!("hash : {:?}", leaf.to_hex());
    }

    #[test]
    fn test_nullifier_proof() {
        let start = PreciseTime::now();
        let rng = &mut ChaChaRng::from_seed(&[1u32, 1u32, 1u32, 1u32]);
        println!(
            "\nsetup generated in {} s\n\n",
            start.to(PreciseTime::now()).num_milliseconds() as f64 / 1000.0
        );

        let nullifier = Fr::rand(rng);
        let secret = Fr::rand(rng);
        let leaf = *create_leaf_from_preimage(nullifier, secret).hash();
        println!(
            "\nrandom target created in {} s\n\n",
            start.to(PreciseTime::now()).num_milliseconds() as f64 / 1000.0
        );
        let mut leaves = vec![leaf];
        for _ in 0..7 {
            leaves.push(Fr::rand(rng));
        }
        println!(
            "\nleaves created in {} s\n\n",
            start.to(PreciseTime::now()).num_milliseconds() as f64 / 1000.0
        );
        println!(
            "\nleaves pushed in {} s\n\n",
            start.to(PreciseTime::now()).num_milliseconds() as f64 / 1000.0
        );
        let tree_nodes = create_leaf_list(leaves, 3);
        println!(
            "\nleaf list generated in {} s\n\n",
            start.to(PreciseTime::now()).num_milliseconds() as f64 / 1000.0
        );
        let (_r, proof) = build_merkle_tree_with_proof(tree_nodes, 3, 3, leaf, vec![]);
        println!("\nProof\n{:?}", proof);
        println!("\nRoot\n{:?}", *_r.root.hash());

        println!(
            "\ntree proof generated in {} s\n\n",
            start.to(PreciseTime::now()).num_milliseconds() as f64 / 1000.0
        );
        let _computed_root = compute_root_from_proof(leaf, proof);
        println!(
            "\ncomputed root generated in {} s\n\n",
            start.to(PreciseTime::now()).num_milliseconds() as f64 / 1000.0
        );
        println!("\nComputed root{:?}\n", _computed_root);
        assert!(_computed_root == *_r.root.hash());
    }

    #[test]
    fn test_blake_merkle_circuit() {
        let mut cs = TestConstraintSystem::<Bn256>::new();
        let seed_slice = &[1u32, 1u32, 1u32, 1u32];
        let rng = &mut ChaChaRng::from_seed(seed_slice);
        println!("generating setup...");
        let start = PreciseTime::now();

        let depth = 3;

        let mut proof_elts = vec![];
        for _ in 0..depth {
            proof_elts.push(Some((true, rng.gen::<[u8; 32]>())));
        }

        let _j_params = &JubjubBn256::new();

        let nullifier = rng.gen::<[u8; 32]>();
        let secret = rng.gen::<[u8; 32]>();
        let m_circuit = BlakeTreeCircuit {
            nullifier: Some(nullifier),
            secret: Some(secret),
            proof: proof_elts,
        };

        m_circuit
            .synthesize(&mut cs)
            .expect("circuit must synthesize");
        println!(
            "setup generated in {} s",
            start.to(PreciseTime::now()).num_milliseconds() as f64 / 1000.0
        );
        println!("num constraints: {}", cs.num_constraints());
        println!("num inputs: {}", cs.num_inputs());

        // we can use internal tools to check for some variables being unconstrained (e.g. declared, but not used)
        let unconstrained = cs.find_unconstrained();
        println!("{}", unconstrained);
        assert!(unconstrained == "");

        // let's check that our constraints are satisfied with a current assignment
        let unsatisfied = cs.which_is_unsatisfied();
        if unsatisfied.is_some() {
            panic!("{}", unsatisfied.unwrap());
        }
        println!("Constraint system is satisfied");
    }

    #[test]
    fn test_bmt_sequence() {
        use blake_circuit;
        use blake_merkle_tree;

        let mut _cs = TestConstraintSystem::<Bn256>::new();
        let seed_slice = &[1u32, 1u32, 1u32, 1u32];
        let rng = &mut ChaChaRng::from_seed(seed_slice);

        let nullifier = rng.gen::<[u8; 32]>();
        let secret = rng.gen::<[u8; 32]>();
        println!("Nullifier: {:?}", nullifier);
        println!("Secret: {:?}", secret);
        let leaf = blake_merkle_tree::create_leaf_from_preimage(nullifier, secret);
        println!("Leaf hash: {:?}\n", leaf.hash());

        let bmt_leaves = blake_merkle_tree::create_leaf_list(vec![*leaf.hash()], 3);
        let (_r, proof_path) =
            blake_merkle_tree::build_merkle_tree_with_proof(bmt_leaves, 3, 3, *leaf.hash(), vec![]);
        println!("Path {:?}", proof_path);
        println!("Root hash: {:?}\n", _r.root.hash());

        let params = blake_circuit::generate(seed_slice, proof_path.len() as u32)
            .unwrap()
            .params;
        println!("Circuit params{:?}", params);
        let proof_hex =
            blake_circuit::prove(seed_slice, &params, &nullifier, &secret, proof_path).unwrap();

        fs::write("test/test.params", params).unwrap();
        fs::write("test/test.proof", proof_hex.proof).unwrap();
        let parameters = &String::from_utf8(fs::read("test/test.params").unwrap()).unwrap();
        let the_proof = &String::from_utf8(fs::read("test/test.proof").unwrap()).unwrap();

        // let h = &String::from_utf8(fs::read("test/test_tree.h").unwrap()).unwrap();
        let verify = verify(
            parameters,
            the_proof,
            &hex::encode(nullifier),
            &hex::encode(_r.root.hash()),
        )
        .unwrap();
        println!("Did the circuit work!? {:?}", verify.result);
    }

    #[test]
    fn test_input_circuit_with_bls12_381() {
        use self::jubjub::{edwards, fs, JubjubBls12};
        use ff::{BitIterator, Field};
        use pairing::bls12_381::*;
        use rand::{Rng, SeedableRng, XorShiftRng};

        let params = &JubjubBls12::new();
        let rng = &mut XorShiftRng::from_seed([0x3dbe6259, 0x8d313d76, 0x3237db17, 0xe5bc0654]);

        let tree_depth = 32;

        for _ in 0..1 {
            let value_commitment = ValueCommitment {
                value: rng.gen(),
                randomness: rng.gen(),
            };

            let nsk: fs::Fs = rng.gen();
            let ak = edwards::Point::rand(rng, params).mul_by_cofactor(params);

            let proof_generation_key = self::primitives::ProofGenerationKey {
                ak: ak.clone(),
                nsk: nsk.clone(),
            };

            let viewing_key = proof_generation_key.into_viewing_key(params);

            let payment_address;

            loop {
                let diversifier = test::primitives::Diversifier(rng.gen());

                if let Some(p) = viewing_key.into_payment_address(diversifier, params) {
                    payment_address = p;
                    break;
                }
            }

            let g_d = payment_address.diversifier.g_d(params).unwrap();
            let commitment_randomness: fs::Fs = rng.gen();
            let auth_path = vec![Some((rng.gen(), rng.gen())); tree_depth];
            let ar: fs::Fs = rng.gen();

            {
                let rk = viewing_key.rk(ar, params).into_xy();
                let expected_value_cm = value_commitment.cm(params).into_xy();
                let note = test::primitives::Note {
                    value: value_commitment.value,
                    g_d: g_d.clone(),
                    pk_d: payment_address.pk_d.clone(),
                    r: commitment_randomness.clone(),
                };

                let mut position = 0u64;
                let cm: Fr = note.cm(params);
                let mut cur = cm.clone();

                for (i, val) in auth_path.clone().into_iter().enumerate() {
                    let (uncle, b) = val.unwrap();

                    let mut lhs = cur;
                    let mut rhs = uncle;

                    if b {
                        ::std::mem::swap(&mut lhs, &mut rhs);
                    }

                    let mut lhs: Vec<bool> = BitIterator::new(lhs.into_repr()).collect();
                    let mut rhs: Vec<bool> = BitIterator::new(rhs.into_repr()).collect();

                    lhs.reverse();
                    rhs.reverse();

                    cur = self::pedersen_hash::pedersen_hash::<Bls12, _>(
                        self::pedersen_hash::Personalization::MerkleTree(i),
                        lhs.into_iter()
                            .take(Fr::NUM_BITS as usize)
                            .chain(rhs.into_iter().take(Fr::NUM_BITS as usize)),
                        params,
                    )
                    .into_xy()
                    .0;

                    if b {
                        position |= 1 << i;
                    }
                }

                let expected_nf = note.nf(&viewing_key, position, params);
                let expected_nf = multipack::bytes_to_bits_le(&expected_nf);
                let expected_nf = multipack::compute_multipacking::<Bls12>(&expected_nf);
                assert_eq!(expected_nf.len(), 2);

                let mut cs = TestConstraintSystem::<Bls12>::new();

                let instance = Spend {
                    params,
                    value_commitment: Some(value_commitment.clone()),
                    proof_generation_key: Some(proof_generation_key.clone()),
                    payment_address: Some(payment_address.clone()),
                    commitment_randomness: Some(commitment_randomness),
                    ar: Some(ar),
                    auth_path: auth_path.clone(),
                    anchor: Some(cur),
                };

                instance.synthesize(&mut cs).unwrap();

                assert!(cs.is_satisfied());
                assert_eq!(cs.num_constraints(), 98777);
                assert_eq!(
                    cs.hash(),
                    "d37c738e83df5d9b0bb6495ac96abf21bcb2697477e2c15c2c7916ff7a3b6a89"
                );

                assert_eq!(cs.get("randomization of note commitment/x3/num"), cm);

                assert_eq!(cs.num_inputs(), 8);
                assert_eq!(cs.get_input(0, "ONE"), Fr::one());
                assert_eq!(cs.get_input(1, "rk/x/input variable"), rk.0);
                assert_eq!(cs.get_input(2, "rk/y/input variable"), rk.1);
                assert_eq!(
                    cs.get_input(3, "value commitment/commitment point/x/input variable"),
                    expected_value_cm.0
                );
                assert_eq!(
                    cs.get_input(4, "value commitment/commitment point/y/input variable"),
                    expected_value_cm.1
                );
                assert_eq!(cs.get_input(5, "anchor/input variable"), cur);
                assert_eq!(cs.get_input(6, "pack nullifier/input 0"), expected_nf[0]);
                assert_eq!(cs.get_input(7, "pack nullifier/input 1"), expected_nf[1]);
            }
        }
    }
}
