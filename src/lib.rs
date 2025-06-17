use byteorder::{ByteOrder, LittleEndian};
use numpy::{PyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;

/// Reads the Dice output binary file and returns three arrays that reprsent
/// a SCI wavefunction:
/// (1) 2D SCI coefficients (amplitudes),
/// (2) unique and sorted alpha CIs (determinants), and
/// (3) unique and sorted beta CIs.
#[pyfunction]
fn from_bin_file_to_sci(py: Python, path: &str) -> PyResult<PyObject> {
    let (amps, dets_a, dets_b) =
        py.allow_threads(|| -> PyResult<(Vec<f64>, Vec<u64>, Vec<u64>)> {
            // read file and validate
            let mut file = File::open(path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("File error: {}", e)))?;

            let mut header = [0u8; 8];
            file.read_exact(&mut header).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Header read error: {}", e))
            })?;

            let num_records = LittleEndian::read_u32(&header[0..4]);
            let string_length = LittleEndian::read_u32(&header[4..8]) as usize;

            if string_length > 64 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "String length {} exceeds maximum of 64",
                    string_length
                )));
            }

            let file_size = file.metadata()?.len();
            let header_size = 8;
            let data_size = file_size.saturating_sub(header_size);

            let record_size = 8usize + string_length;

            if data_size % record_size as u64 != 0 {
                return Err(pyo3::exceptions::PyIOError::new_err(format!(
                    "Data size {} not divisible by record size {}",
                    data_size, record_size
                )));
            }

            let calculated_records = (data_size / record_size as u64) as u32;
            if calculated_records != num_records {
                return Err(pyo3::exceptions::PyIOError::new_err(format!(
                    "Header claims {} records but found {}",
                    num_records, calculated_records
                )));
            }

            let mut buffer = vec![0u8; data_size as usize];
            file.read_exact(&mut buffer).map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Data read error: {}", e))
            })?;

            // process bytes data to get amplitudes (f64) and alpha and beta
            // determinants (as u64 each)
            Ok(process_data(
                buffer,
                record_size,
                string_length,
                num_records,
            ))
        })?;

    // deduplicate and sort the alpha and beta determinants and
    // reorganize the amplitudes into a 2D Vec.
    let (ci_vec, unique_a, unique_b) = construct_ci_vec(&amps, &dets_a, &dets_b);

    // convert Rust Vecs to PyArrays to reduce copy
    // overhead from Rust to Python space
    let ci_vec_array2d = PyArray2::from_vec2_bound(py, &ci_vec).unwrap();
    let unique_a_array = unique_a.to_pyarray_bound(py);
    let unique_b_array = unique_b.to_pyarray_bound(py);

    Ok(PyTuple::new_bound(
        py,
        &[
            ci_vec_array2d.to_object(py),
            unique_a_array.to_object(py),
            unique_b_array.to_object(py),
        ],
    )
    .to_object(py))
}

/// Process bytes to compute amplitudes and alpha and beta determinants.
/// The bytes are separated in fixed size chunks, and each chunk is processed
/// in parallel.
///
/// It is recommended that you set `RAYON_NUM_THREADS` environment
/// variable to control the number of parallel threads.
///
/// Max. chunk size (`records_per_chunk`) is fixed to 1000 records.
/// Too small or too large number of records per chunk can hurt performance.
/// While there are opportunities to fine-tune this parameter,
/// current max chunk size of 1000 gives reasonably fast performance.
///
/// There can be remainder records after constructing fixed size chunks.
/// For example, for 2500 records and `records_per_chunk=1000`, there will be
/// 500 remainder records (1000 in the 1st chunk + 1000 in the 2nd chunk +
/// 500 remainder). Those records are handled separately.
fn process_data(
    buffer: Vec<u8>,
    record_size: usize,
    string_length: usize,
    num_records: u32,
) -> (Vec<f64>, Vec<u64>, Vec<u64>) {
    let records_per_chunk = std::cmp::min(1000, num_records as usize);

    let chunks_iter = buffer.par_chunks_exact(record_size * records_per_chunk);
    let remainder = chunks_iter.remainder();

    let batch_results: Vec<(Vec<f64>, Vec<u64>, Vec<u64>)> = chunks_iter
        .map(|batch| process_batch(batch, record_size, string_length))
        .collect();

    let mut amps = Vec::with_capacity(num_records as usize);
    let mut dets_a = Vec::with_capacity(num_records as usize);
    let mut dets_b = Vec::with_capacity(num_records as usize);

    for (batch_amps, batch_a, batch_b) in batch_results {
        amps.extend(batch_amps);
        dets_a.extend(batch_a);
        dets_b.extend(batch_b);
    }

    // handle remainder
    for record in remainder.chunks_exact(record_size) {
        let amp = LittleEndian::read_f64(&record[0..8]);
        let dets = process_string(&record[8..8 + string_length]);
        amps.push(amp);
        dets_a.push(dets[0]);
        dets_b.push(dets[1]);
    }

    (amps, dets_a, dets_b)
}

/// Processes a batch of records. Each record consists of 8 bytes that represent
/// a floating-point amplitude + `string_length` (which is equal to the number
/// of spatial orbitals) bytes representing a string. The string is converted into
/// two u64 integers that represent alpha and beta determinants.
fn process_batch(
    batch: &[u8],
    record_size: usize,
    string_length: usize,
) -> (Vec<f64>, Vec<u64>, Vec<u64>) {
    let mut batch_amps = Vec::with_capacity(batch.len() / record_size);
    let mut batch_dets_a = Vec::with_capacity(batch.len() / record_size);
    let mut batch_dets_b = Vec::with_capacity(batch.len() / record_size);

    for record in batch.chunks_exact(record_size) {
        let float = LittleEndian::read_f64(&record[0..8]);
        let dets = process_string(&record[8..8 + string_length]);

        batch_amps.push(float);
        batch_dets_a.push(dets[0]);
        batch_dets_b.push(dets[1]);
    }

    (batch_amps, batch_dets_a, batch_dets_b)
}

/// Converts bytes into two u64s that represent alpha and beta determinants.
/// Dice represents both alpha and beta spin-orbital occupancy succintly using
/// a single string consisting of characters '2', 'a', 'b', and '0'. A '2' means
/// both spin-orbitals at that index are occupied. An 'a' ('b') means only the
/// alpha (beta) orbital is occupied. Finally, a '0' means both are empty.
/// Instead of first creating two bitstrings explicitly from this concise
/// representaion then converting them to integers, this routine iterates over
/// bytes and efficiently computes integers.
fn process_string(bytes: &[u8]) -> [u64; 2] {
    let mut dets = [0u64; 2];

    for (i, &byte) in bytes.iter().enumerate() {
        let pos = i as u32;

        match byte {
            b'2' => {
                dets[0] |= 1u64 << pos;
                dets[1] |= 1u64 << pos;
            }
            b'a' => {
                dets[0] |= 1u64 << pos;
            }
            b'b' => {
                dets[1] |= 1u64 << pos;
            }
            b'0' => {}
            _ => panic!("Invalid character '{}' at position {}", byte as char, i),
        }
    }

    dets
}

/// Gets dedupicated and sorted alpha and beta determinants, and
/// reorganizes the amplitudes in a 2D Vec. Each element of the
// `ci_vec` will represent the amplitude of a state with specific
/// alpha and beta occupancy.
fn construct_ci_vec(
    amps: &[f64],
    dets_a: &[u64],
    dets_b: &[u64],
) -> (Vec<Vec<f64>>, Vec<u64>, Vec<u64>) {
    let (unique_a, map_a) = get_unique_and_map(dets_a);
    let (unique_b, map_b) = get_unique_and_map(dets_b);

    let mut ci_vec = vec![vec![0.0; unique_b.len()]; unique_a.len()];

    for ((&amp, &a), &b) in amps.iter().zip(dets_a).zip(dets_b) {
        let i = map_a[&a];
        let j = map_b[&b];
        ci_vec[i][j] = amp;
    }

    (ci_vec, unique_a, unique_b)
}

/// Deduplicates and sorts a Vec and creates a map of elem to index.
fn get_unique_and_map(values: &[u64]) -> (Vec<u64>, HashMap<u64, usize>) {
    let mut unique: Vec<u64> = values
        .iter()
        .cloned()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    unique.sort_unstable();
    let map = unique.iter().enumerate().map(|(i, &v)| (v, i)).collect();

    (unique, map)
}

#[pymodule]
fn dice_outfile_reader(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(from_bin_file_to_sci, m)?)?;
    Ok(())
}
