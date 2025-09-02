use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;
use ndarray::Array2;
use numpy::{PyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;
use std::collections::HashMap;
use std::env;
use std::fs::File;

/// Reads the Dice output binary file and returns three arrays that represent
/// a SCI wavefunction using memory-mapped I/O.
/// The three arrays are:
/// (1) 2D SCI coefficients (amplitudes),
/// (2) unique and sorted alpha CIs (determinants), and
/// (3) unique and sorted beta CIs.
#[pyfunction]
fn from_bin_file_to_sci(py: Python, path: &str) -> PyResult<PyObject> {
    let (amps, dets_a, dets_b) =
        py.allow_threads(|| -> PyResult<(Vec<f64>, Vec<u64>, Vec<u64>)> {
            let file = File::open(path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("File error: {}", e)))?;
            let mmap = unsafe { Mmap::map(&file) }
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Mmap error: {}", e)))?;
            let data = &mmap[..];

            if data.len() < 8 {
                return Err(pyo3::exceptions::PyIOError::new_err(
                    "File too small for header",
                ));
            }

            let num_records = LittleEndian::read_u32(&data[0..4]);
            let string_length = LittleEndian::read_u32(&data[4..8]) as usize;

            if string_length > 64 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "String length {} exceeds maximum of 64",
                    string_length
                )));
            }

            let data_section = &data[8..];
            let record_size = 8 + string_length;
            let total_records = data_section.len() / record_size;

            if total_records != num_records as usize {
                return Err(pyo3::exceptions::PyIOError::new_err(format!(
                    "Header claims {} records but found {}",
                    num_records, total_records
                )));
            }

            Ok(process_data(
                data_section,
                record_size,
                string_length,
                num_records as usize,
            ))
        })?;

    let (ci_vec_flat, nrows, ncols, unique_a, unique_b) = construct_ci_vec(&amps, &dets_a, &dets_b);

    let array = Array2::from_shape_vec((nrows, ncols), ci_vec_flat).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Array shape error: {}", e))
    })?;

    let ci_vec_array = PyArray::from_owned_array_bound(py, array);
    let unique_a_array = PyArray1::from_vec_bound(py, unique_a);
    let unique_b_array = PyArray1::from_vec_bound(py, unique_b);

    Ok(PyTuple::new_bound(
        py,
        &[
            ci_vec_array.to_object(py),
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
    data: &[u8],
    record_size: usize,
    string_length: usize,
    num_records: usize,
) -> (Vec<f64>, Vec<u64>, Vec<u64>) {
    // Configurable batch size with environment variable
    let default_batch_size = 1000;
    let records_per_chunk = env::var("BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_batch_size)
        .min(num_records);

    let chunks_iter = data.par_chunks_exact(record_size * records_per_chunk);
    let remainder = chunks_iter.remainder();

    let batch_results: Vec<(Vec<f64>, Vec<u64>, Vec<u64>)> = chunks_iter
        .map(|batch| process_batch(batch, record_size, string_length))
        .collect();

    let mut amps = Vec::with_capacity(num_records);
    let mut dets_a = Vec::with_capacity(num_records);
    let mut dets_b = Vec::with_capacity(num_records);

    for (batch_amps, batch_a, batch_b) in batch_results {
        amps.extend(batch_amps);
        dets_a.extend(batch_a);
        dets_b.extend(batch_b);
    }

    // Process remainder without parallelization
    for record in remainder.chunks_exact(record_size) {
        let amp = LittleEndian::read_f64(&record[0..8]);
        let dets = process_string(&record[8..8 + string_length]);
        amps.push(amp);
        dets_a.push(dets[0]);
        dets_b.push(dets[1]);
    }

    (amps, dets_a, dets_b)
}

/// Processes a batch of records. Each record consists of (8+n) bytes. The first 8
/// bytes represent a floating-point amplitude. The next n bytes reprsent a string,
/// where n = `string_length` (which is equal to the number of spatial orbitals).
/// Finally, the string is converted into two u64 integers that represent alpha
/// and beta determinants.
fn process_batch(
    batch: &[u8],
    record_size: usize,
    string_length: usize,
) -> (Vec<f64>, Vec<u64>, Vec<u64>) {
    let num_records = batch.len() / record_size;
    let mut batch_amps = Vec::with_capacity(num_records);
    let mut batch_dets_a = Vec::with_capacity(num_records);
    let mut batch_dets_b = Vec::with_capacity(num_records);

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
            b'a' => dets[0] |= 1u64 << pos,
            b'b' => dets[1] |= 1u64 << pos,
            b'0' => {}
            _ => panic!("Invalid character '{}' at position {}", byte as char, i),
        }
    }
    dets
}

/// Gets dedupicated and sorted alpha and beta determinants, and
/// organizes the amplitudes in a 1D Vec. Each element of the
// `ci_vec` will represent the amplitude of a state with specific
/// alpha and beta occupancy.
fn construct_ci_vec(
    amps: &[f64],
    dets_a: &[u64],
    dets_b: &[u64],
) -> (Vec<f64>, usize, usize, Vec<u64>, Vec<u64>) {
    let (unique_a, map_a) = get_unique_and_map(dets_a);
    let (unique_b, map_b) = get_unique_and_map(dets_b);

    let nrows = unique_a.len();
    let ncols = unique_b.len();
    let mut ci_vec_flat = vec![0.0; nrows * ncols];

    for ((&amp, &a), &b) in amps.iter().zip(dets_a).zip(dets_b) {
        let i = map_a[&a];
        let j = map_b[&b];
        ci_vec_flat[i * ncols + j] = amp;
    }

    (ci_vec_flat, nrows, ncols, unique_a, unique_b)
}

/// Deduplicates and sorts a Vec and creates a map of elem to index.
fn get_unique_and_map(values: &[u64]) -> (Vec<u64>, HashMap<u64, usize>) {
    let mut unique = values.to_vec();
    unique.sort_unstable();
    unique.dedup();
    let map = unique
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    (unique, map)
}

#[pymodule]
fn dice_outfile_reader(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(from_bin_file_to_sci, m)?)?;
    Ok(())
}
