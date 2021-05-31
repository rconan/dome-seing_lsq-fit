use indicatif::ProgressBar;
use na::{DMatrix, DVector};
use nalgebra as na;
use ndarray::{array, Array, Array1, Array2};
use ndarray_npy::{NpzReader, NpzWriter};
use num_cpus;
use std::{env, fs::File, path::Path, sync::Arc, thread};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check environment
    let cfd_data_path = env::var("CFD_DATA_PATH").map_err(|e| {
        format!(
            "Environment variable CFD_DATA_PATH is missing\n Caused by {:}",
            e
        )
    })?;

    // LOAD KL DATA
    let mut data = NpzReader::new(File::open("../kls.npz")?)?;
    println!("data: {:#?}", data.names()?);

    // Read KL modes in matrix [n_pupil_points X n_mode] (n_mode=675*7)
    let modes = Arc::new({
        let modes: Array2<f64> = data.by_name("M.npy")?;
        let shape = modes.shape();
        println!("mode shape: {:?}", modes.shape());
        //let q = modes.as_slice().unwrap().to_owned();
        //println!("M: {:#?}", &q[..10]);
        na::DMatrix::from_row_slice(shape[0], shape[1], modes.as_slice().unwrap())
    });

    // Read pupil defined on a 512x512 mesh
    let pupil = {
        let pupil: Array1<bool> = data.by_name("p.npy")?;
        println!("pupil shape: {:?}", pupil.shape());
        pupil.as_slice().unwrap().to_owned()
    };
    let nnz = pupil.iter().fold(0usize, |s, &x| s + x as usize);
    println!("Pupil nnz #: {}", nnz);

    // Read segment pupil mask 7 vectors of vector of length n_pupil_points
    let seg_mask_p: Arc<Vec<Vec<bool>>> = Arc::new({
        // Read mask into matrix
        let piston_mask = {
            let piston_mask: Array2<bool> = data.by_name("piston mask.npy")?;
            let shape = piston_mask.shape();
            println!("piston mask shape: {:?}", piston_mask.shape());
            //let q = modes.as_slice().unwrap().to_owned();
            //println!("M: {:#?}", &q[..10]);
            na::DMatrix::from_row_slice(shape[0], shape[1], piston_mask.as_slice().unwrap())
        };
        // Filter row based on the pupil mask
        piston_mask
            .row_iter()
            .map(|r| {
                r.iter()
                    .cloned()
                    .zip(pupil.iter())
                    .filter(|(_, &p)| p)
                    .map(|(x, _)| x)
                    .collect::<Vec<bool>>()
            })
            .collect()
    });
    // Check the # of points in each segment
    let seg_pupil_size = seg_mask_p
        .iter()
        .map(|r| r.iter().fold(0usize, |s, x| s + *x as usize))
        .collect::<Vec<usize>>();
    println!(
        "Segment pupil sizes: {:?} , total: {}",
        seg_pupil_size,
        seg_pupil_size.iter().sum::<usize>()
    );

    /* Split the KL modes into 7 matrices,
       one per segment and reduced to only the values defined in the segment
       compute the SVD of each segment KL matrix
    */
    let n_mode_outer = 675;
    let n_mode_center = n_mode_outer - 114;
    let mut handle = vec![];
    for k in 0..7 {
        let seg_mask_p = Arc::clone(&seg_mask_p);
        let modes = Arc::clone(&modes);
        let seg_pupil_size = seg_pupil_size.clone();
        handle.push(thread::spawn(move || {
            println!("Seg # {}", k + 1);
            let segment_mask = &seg_mask_p[k];
            let n_mode_seg = if k == 6 { n_mode_center } else { n_mode_outer };
            let q: Vec<_> = modes
                .column_iter()
                .skip(k * n_mode_outer)
                .take(n_mode_seg)
                .map(|c| {
                    na::DVector::from_iterator(
                        seg_pupil_size[k],
                        c.iter()
                            .cloned()
                            .zip(segment_mask.iter())
                            .filter(|(_, &m)| m)
                            .map(|(x, _)| x),
                    )
                })
                .collect();
            let buffer = DMatrix::from_columns(&q);
            let svd = buffer.clone().svd(true, true);
            (buffer, svd)
        }));
    }
    let (__modes_seg, __svd): (Vec<_>, Vec<_>) =
        handle.into_iter().map(|c| c.join().unwrap()).unzip();
    let modes_seg = Arc::new(__modes_seg);
    let svd = Arc::new(__svd);

    // ####### DOME SEEING OPD PROCESSING #######

    let n_thread = num_cpus::get();
    println!("thread #: {}", n_thread);
    // Read directory and select dome seeing OPD data files
    let opd_paths: Vec<String> = Path::new(&cfd_data_path)
        .read_dir()?
        .filter(|x| match x {
            Ok(x) => x
                .file_name()
                .to_str()
                .map(|x| x.starts_with("OPDData_OPD_Data_"))
                .unwrap(),
            Err(_) => false,
        })
        .filter(|x| match x {
            Ok(x) => match x.path().extension() {
                None => false,
                Some(ext) => match ext.to_str() {
                    Some("npz") => true,
                    _ => false,
                },
            },
            Err(_) => false,
        })
        .map(|d| d.unwrap().path().to_str().unwrap().to_owned())
        .collect();
    println!("OPD file #: {}", opd_paths.len());
    // Process OPD in batches, each batch running parallel processes
    let n_pool = 1 + opd_paths.len() / n_thread;
    let bar = ProgressBar::new(opd_paths.len() as u64);
    for k in 0..n_pool {
        let mut handle = vec![];
        for opd_file in opd_paths
            .iter()
            .map(|x| x.to_owned())
            .skip(k * n_pool)
            .take(n_thread)
        {
            let cfd_data_path = cfd_data_path.clone();
            let modes_seg = Arc::clone(&modes_seg);
            let svd = Arc::clone(&svd);
            let pupil = pupil.clone();
            let seg_mask_p = Arc::clone(&seg_mask_p);
            let seg_pupil_size = seg_pupil_size.clone();
            handle.push(thread::spawn(move || {
                let opd_path = Path::new(&opd_file);
                //println!("OPD file: {:#?}", opd_path);
                // Load dome seeing OPD
                let w_seg: Vec<_> = {
                    let mut cfd = NpzReader::new(File::open(opd_path).unwrap()).unwrap();
                    //println!("cfd: {:#?}", cfd.names()?);
                    // Reduced the OPD to the pupil
                    let w: Vec<_> = {
                        let w: Array1<f64> = cfd.by_name("opd.npy").unwrap();
                        //println!("opd shape: {:?}", w.shape());
                        w.as_slice()
                            .unwrap()
                            .iter()
                            .map(|&x| if x.is_nan() { 0f64 } else { x })
                            .zip(pupil.iter())
                            .filter(|(_, &p)| p)
                            .map(|(x, _)| x)
                            .collect()
                    };
                    // Split and reduced the OPD to each segment
                    seg_mask_p
                        .iter()
                        .zip(seg_pupil_size.iter())
                        .map(|(segment_mask, &n)| {
                            DVector::from_iterator(
                                n,
                                w.iter()
                                    .zip(segment_mask.iter())
                                    .filter(|(_, &m)| m)
                                    .map(|(&x, _)| x),
                            )
                        })
                        .collect()
                };

                // Least square fit the segment OPD to the segment KL
                let x: Vec<_> = svd
                    .iter()
                    .zip(w_seg.iter())
                    .map(|(a, b)| a.solve(&b, 1e-6).unwrap())
                    .collect();
                // Compute the residual dome seeing OPD
                let res: Vec<_> = w_seg
                    .iter()
                    .zip(modes_seg.iter().zip(x.iter()))
                    .map(|(s, (m, x))| (s - m * x).as_slice().to_vec())
                    .collect();
                // Compute the residual wavefront error RMS
                let wfe_rms = {
                    let wfe: Vec<f64> = res.iter().flat_map(|x| x.clone()).collect();
                    let n = wfe.len() as f64;
                    let wfe_mean = wfe.iter().sum::<f64>() / n;
                    (wfe.iter().fold(0f64, |s, x| s + (x - wfe_mean).powi(2)) / n).sqrt()
                };
                //println!("WFE RMS: {:.3}nm", wfe_rms * 1e9);
                // Reconstruct the wavefront on the original 512x512 mesh
                let mut wfe = vec![f64::NAN; pupil.len()];
                {
                    // Reconstruct the wavefront in the pupik
                    let seg_wfe_p: Vec<_> = seg_mask_p.iter().zip(res.into_iter()).fold(
                        vec![0f64; nnz],
                        |s, (m, r)| {
                            let mut ri = r.into_iter();
                            s.iter()
                                .zip(m.iter())
                                .map(|(&s, &m)| s + if m { ri.next().unwrap() } else { 0f64 })
                                .collect::<Vec<f64>>()
                        },
                    );
                    // Reconstruct the wavefront in the mesh
                    wfe.iter_mut()
                        .zip(pupil.iter())
                        .filter(|(_, &p)| p)
                        .map(|(w, _)| w)
                        .zip(seg_wfe_p.iter())
                        .for_each(|(w, w_p)| *w = *w_p);
                };

                // Save data (opd map and rms) in Numpy npz file
                let file_name = format!(
                    "high-contrast_dome-seeing_residual_{}.npz",
                    opd_path.file_stem().unwrap().to_str().unwrap()
                );
                let out_path = Path::new(&cfd_data_path).join(Path::new(&file_name));
                //println!("Residue path: {:#?}", out_path);
                let mut npz = NpzWriter::new(File::create(out_path).unwrap());
                npz.add_array("residual_opd", &Array::from_vec(wfe))
                    .unwrap();
                npz.add_array("residual_opd_rms", &array![wfe_rms]).unwrap();
                npz.finish().unwrap();
            }));
        }
        for (k, c) in handle.into_iter().enumerate() {
            if let Err(e) = c.join() {
                println!("Error in thread #{}: {:#?}", k, e);
            }
            bar.inc(1);
        }
    }
    bar.finish();
    Ok(())
}
