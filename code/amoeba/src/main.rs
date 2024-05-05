
use std::f32;
use rand_distr::StandardNormal;
use rand::prelude::*;
use rand_seeder::Seeder;
use rand_pcg::Pcg64;
use image::{RgbImage, Rgb};
use image::io::Reader as ImageReader;
use rayon::prelude::*;
use std::fs::{OpenOptions, read_dir};
use std::io::Write;
use std::env;
use libm;

const DOM_WIDTH: usize = 200;
const DOM_HEIGHT: usize = 200;
const DT: f32 = 0.0020; // temporal step 0.0025;
const DX: f32 = 0.15; // spatial step in um // old 0.1
const STOP_TIME: f32 = 1200.0;
const SHIFT_DOMAIN: bool = true;
const HISTOGRAM_RESOLUTION: usize = 360;
const SAVE_STEP: i32 = 100;
const IMAGE_SAVE_MULTIPLE: i32 = 4;

// Phase field parameters
const RADIUS: f32 = 40.0;
const VOL: f32 = f32::consts::PI * RADIUS * RADIUS;
const THRESHOLD: f32 = 0.001;

// const ALPHA0: f32 = 3.0;     //Active force
// const ALPHA0: f32 = 1.2;     //Active force
const ALPHA0: f32 = 1.0;     //Active force
const EPSILON: f32 = 0.750;
const GAMMA: f32 = 2.0;   //Tension of the membrane
// const GAMMA: f32 = 1.8;   //Tension of the membrane
const TAU: f32 = 3.0;           //Time scale of the whole equation for the membrane dynamics
const MA: f32 = 0.5;     //Volume conservation
const MB: f32 = 1.0;            //Repulsion among cells
// const DELTA_M: f32 = 0.045;
const DELTA_M: f32 = 10.0/VOL;
const BETA: f32 = 22.2;

// Parameters for the Noise
const TAUXI: f32 = 10.0; // in s
const TAUXI_INV: f32 = 1.0 / TAUXI;
// const SIGMA: f32 = 0.122474487;
const SIGMA: f32 = 0.387298335;
const SIGMA2: f32 = SIGMA * SIGMA; // 0.15
const SEED: u64 = 30499;
// const MEMBRANE_NOISE: f32 = 0.5; //1.0 FOR the case of NO dependence on curvature, 0.4 for dependence
// const MEMBRANE_NOISE: f32 = 0.25; //1.0 FOR the case of NO dependence on curvature, 0.4 for dependence
// const MEMBRANE_NOISE: f32 = 0.7; //1.0 FOR the case of NO dependence on curvature, 0.4 for dependence
// const GLOBAL_NOISE: f32 = 0.0;
// const GLOBAL_NOISE: f32 = 0.1;
// const MEMBRANE_NOISE: f32 = 1.0; //1.0 FOR the case of NO dependence on curvature, 0.4 for dependence
const MEMBRANE_NOISE: f32 = 0.5; //1.0 FOR the case of NO dependence on curvature, 0.4 for dependence
const GLOBAL_NOISE: f32 = 0.0;

// Parameters for Sawai Model
const D_B: f32 = 0.5 ; // in um2/s.
// const DECAYX: f32 = 0.01; //
const DECAYX: f32 = 0.1; //
// const KACT: f32 = 1.0; // 1.0	 //1.0 for Starving full movile cells and 0.2-0.5 for vegetative cells
const KACT: f32 = 2.0; // 1.0	 //1.0 for Starving full movile cells and 0.2-0.5 for vegetative cells
const REACT: f32 = 1.0;

// Parameters of the activator conservation
const DELTA02: f32 = 0.5;
const VOL2: f32 = 0.6*VOL;
const RADIUS2: f32 = 0.5*RADIUS;
const TAUDELTA: f32 = 0.010 ;
const TAUMA: f32 = 1.0 ; // tauma = 0.010 ;

const CURVATURE_DEPENDENCE: bool = true;
// const CURVATURE_LIMIT: f32 = 20.0;
// const MIN_CURVATURE: f32 = 0.1667;
const MIN_CURVATURE: f32 = 0.166667 * 1.0;
// const MAX_CURVATURE: f32 = 0.8333;
// const MAX_CURVATURE: f32 = 1.0;
const MAX_CURVATURE: f32 = 0.166667 * 2.5;

type DomainMatrix = [[f32; DOM_HEIGHT]; DOM_WIDTH];

enum Shape {Circle, Flat, BumpyCircle}
const SHAPE: Shape = Shape::Circle;


fn distance(x1: f32, y1: f32, x2: f32, y2: f32) -> f32{
    ((x1 - x2).powi(2) + (y1 - y2).powi(2)).sqrt()
}

fn boundary_get(field: &DomainMatrix, x: usize, y: usize, dx: i32, dy: i32) -> f32{
    match SHAPE {
        Shape::Circle => periodic_border(field, x, y, dx, dy),
        Shape::Flat => semiperiodic_border(field, x, y, dx, dy),
        Shape::BumpyCircle => periodic_border(field, x, y, dx, dy),
    }
}

fn periodic_border(field: &DomainMatrix, x: usize, y: usize, dx: i32, dy: i32) -> f32{
    let px = (x as i32 + dx + DOM_WIDTH as i32) % (DOM_WIDTH as i32);
    let py = (y as i32 + dy + DOM_HEIGHT as i32) % (DOM_HEIGHT as i32);
    field[px as usize][py as usize]
}

fn semiperiodic_border(field: &DomainMatrix, x: usize, y: usize, dx: i32, dy: i32) -> f32{
    let px = (x as i32 + dx + DOM_WIDTH as i32) % (DOM_WIDTH as i32);
    let py = (y as i32 + dy).max(0).min(DOM_HEIGHT as i32 - 1);
    field[px as usize][py as usize]
}

fn gradient(u: &DomainMatrix, grad: &mut DomainMatrix) {
    // grad.par_iter_mut().enumerate().for_each(|(ix, row)| {
    //     for iy in 0..DOM_WIDTH {
    //         row[iy] = 
    //             ((boundary_get(&u, ix, iy, 1, 0) - boundary_get(&u, ix, iy, -1, 0)).powi(2) +
    //              (boundary_get(&u, ix, iy, 0, 1) - boundary_get(&u, ix, iy, 0, -1)).powi(2))
    //             .sqrt() / (2.0 * DX);
    //     }
    // })
    for ix in 0..DOM_WIDTH {
        for iy in 0..DOM_HEIGHT {
            grad[ix][iy] =
                ((boundary_get(&u, ix, iy, 1, 0) - boundary_get(&u, ix, iy, -1, 0)).powi(2) +
                 (boundary_get(&u, ix, iy, 0, 1) - boundary_get(&u, ix, iy, 0, -1)).powi(2))
                .sqrt() / (2.0 * DX);
        }
    }
}

fn laplacian(u: &DomainMatrix, lapl: &mut DomainMatrix) {
    lapl.par_iter_mut().enumerate().for_each(|(ix, row)| {
        for iy in 0..DOM_HEIGHT {
            row[iy] =
                (boundary_get(&u, ix, iy, 1, 0) + boundary_get(&u, ix, iy, -1, 0) +
                 boundary_get(&u, ix, iy, 0, 1) + boundary_get(&u, ix, iy, 0, -1) - 
                 4.0 * u[ix][iy]) / DX.powi(2);
        }
    })
    // for ix in 0..DOM_WIDTH {
    //     for iy in 0..DOM_HEIGHT {
    //         lapl[ix][iy] =
    //             (boundary_get(&u, ix, iy, 1, 0) + boundary_get(&u, ix, iy, -1, 0) +
    //              boundary_get(&u, ix, iy, 0, 1) + boundary_get(&u, ix, iy, 0, -1) - 
    //              4.0 * u[ix][iy]) / DX.powi(2);
    //     }
    // }
}

// Derivate phase field model: nabla*(phi nabla u)
fn derivate_phase_field(u: &DomainMatrix, phi: &DomainMatrix, out: &mut DomainMatrix) {
    out.par_iter_mut().enumerate().for_each(|(ix, row)| {
        for iy in 0..DOM_HEIGHT {
            let dphi_up = phi[ix][iy] + boundary_get(&phi, ix, iy, 0, 1);
            let dphi_down = phi[ix][iy] + boundary_get(&phi, ix, iy, 0, -1);
            let dphi_right = phi[ix][iy] + boundary_get(&phi, ix, iy, 1, 0);
            let dphi_left = phi[ix][iy] + boundary_get(&phi, ix, iy, -1, 0);
            row[iy] =
                (dphi_up * (boundary_get(&u, ix, iy, 0, 1) - u[ix][iy])
                 + dphi_down * (boundary_get(&u, ix, iy, 0, -1) - u[ix][iy])
                 + dphi_right * (boundary_get(&u, ix, iy, 1, 0) - u[ix][iy])
                 + dphi_left * (boundary_get(&u, ix, iy, -1, 0) - u[ix][iy]))
                * 0.5 / DX.powi(2);
        }
    })
    // for ix in 0..DOM_WIDTH {
    //     for iy in 0..DOM_HEIGHT {
    //         let dphi_up = phi[ix][iy] + boundary_get(&phi, ix, iy, 0, 1);
    //         let dphi_down = phi[ix][iy] + boundary_get(&phi, ix, iy, 0, -1);
    //         let dphi_right = phi[ix][iy] + boundary_get(&phi, ix, iy, 1, 0);
    //         let dphi_left = phi[ix][iy] + boundary_get(&phi, ix, iy, -1, 0);
    //         out[ix][iy] =
    //             (dphi_up * (boundary_get(&u, ix, iy, 0, 1) - u[ix][iy])
    //              + dphi_down * (boundary_get(&u, ix, iy, 0, -1) - u[ix][iy])
    //              + dphi_right * (boundary_get(&u, ix, iy, 1, 0) - u[ix][iy])
    //              + dphi_left * (boundary_get(&u, ix, iy, -1, 0) - u[ix][iy]))
    //             * 0.5 / DX.powi(2);
    //     }
    // }
}

fn curvature(u: &DomainMatrix, gradu: &DomainMatrix, out: &mut DomainMatrix, max_curv: &mut f32) {
    let mut unit_x: DomainMatrix = [[1.0; DOM_HEIGHT]; DOM_WIDTH];  // Phase field
    let mut unit_y: DomainMatrix = [[1.0; DOM_HEIGHT]; DOM_WIDTH];  // Phase field
    *max_curv = 0.0;
    for ix in 0..DOM_WIDTH {
        for iy in 0..DOM_HEIGHT {
            if gradu[ix][iy] > 0.0001 {
                unit_x[ix][iy]= (boundary_get(u, ix, iy, 1, 0)-boundary_get(u, ix, iy, -1, 0))/(2.0*DX*gradu[ix][iy]);
                unit_y[ix][iy]= (boundary_get(u, ix, iy, 0, 1)-boundary_get(u, ix, iy, 0, -1))/(2.0*DX*gradu[ix][iy]);
            }
        }
    }
    for ix in 0..DOM_WIDTH {
        for iy in 0..DOM_HEIGHT {
            if gradu[ix][iy] > 0.005 {
                // let ux_p1 = if boundary_get(gradu, ix, iy, 1, 0) > 0.001 {(boundary_get(u, ix, iy, 2, 0) - u[ix][iy])/(DX*boundary_get(gradu, ix, iy, 1, 0))} else {0.0};
                // let ux_m1 = if boundary_get(gradu, ix, iy, -1, 0) > 0.001 {(u[ix][iy] - boundary_get(u, ix, iy, -2, 0))/(DX*boundary_get(gradu, ix, iy, -1, 0))} else {0.0};
                // let uy_p1 = if boundary_get(gradu, ix, iy, 0, 1) > 0.001 {(boundary_get(u, ix, iy, 0, 2) - u[ix][iy])/(DX*boundary_get(gradu, ix, iy, 0, 1))} else {0.0};
                // let uy_m1 = if boundary_get(gradu, ix, iy, 0, -1) > 0.001 {(u[ix][iy] - boundary_get(u, ix, iy, 0, -2))/(DX*boundary_get(gradu, ix, iy, 0, -1))} else {0.0};
                // out[ix][iy] = (ux_p1 - ux_m1 + uy_p1 - uy_m1) / DX;
                out[ix][iy] = (boundary_get(&unit_x, ix, iy, -1, 0) - boundary_get(&unit_x, ix, iy, 1, 0))/(2.0*DX) + (boundary_get(&unit_y, ix, iy, 0, -1) - boundary_get(&unit_y, ix, iy, 0, 1))/(2.0*DX);
                out[ix][iy] = out[ix][iy].min(100.0);
                // *max_curv = max_curv.max(out[ix][iy]);
                if u[ix][iy] > 0.1 && u[ix][iy] < 0.9 {*max_curv = max_curv.max(out[ix][iy])}
            } else {
                out[ix][iy] = 0.0;
            }
        }
    }
}

// Generation of field: circular
fn generate_field_circular(phi: &mut DomainMatrix, xcenter: f32, ycenter: f32, radius: f32) {
    for ix in 0..DOM_WIDTH {
        for iy in 0..DOM_HEIGHT {
            let r = DX * distance(xcenter, ycenter, ix as f32, iy as f32);
            phi[ix][iy] += 0.5 + 0.5 * ((radius - r) / (DX * EPSILON)).tanh();
        }
    }
}

fn initialize_circular(phi: &mut DomainMatrix, bct: &mut DomainMatrix, rng: &mut Pcg64) {
    let ro = RADIUS * DX;
    let xcenter = (DOM_WIDTH as f32)/2.0;
    let ycenter = (DOM_HEIGHT as f32)/2.0;
    generate_field_circular(phi, xcenter, ycenter, ro);
    // for ix in 0..DOM_WIDTH {
    //     for iy in 0..DOM_HEIGHT {
    //         bct[ix][iy] = phi[ix][iy] * rng.gen::<f32>();
    //     }
    // }
    let xran: f32 = -4.0 + 8.0*rng.gen::<f32>();
    let yran: f32 = -4.0 + 8.0*rng.gen::<f32>();
    generate_field_circular(bct, xcenter+xran+20.0, ycenter+yran, RADIUS2*DX);
}

fn initialize_bumpy_circle(phi: &mut DomainMatrix, bct: &mut DomainMatrix, rng: &mut Pcg64) {
    let ro = RADIUS * DX;
    let xcenter = (DOM_WIDTH as f32)/2.0;
    let ycenter = (DOM_HEIGHT as f32)/2.0;
    generate_field_circular(phi, xcenter, ycenter, ro);
    let r2 = 0.2;
    // generate_field_circular(phi, xcenter+RADIUS*(1.0+r2/2.0), ycenter, r2*RADIUS*DX);
    let xran: f32 = 0.0;
    let yran: f32 = 0.0;
    generate_field_circular(bct, xcenter+xran+20.0, ycenter+yran, RADIUS2*DX);
}

fn initialize_flat(phi: &mut DomainMatrix, bct: &mut DomainMatrix, rng: &mut Pcg64) {
    for ix in 0..DOM_WIDTH {
        for iy in 0..DOM_HEIGHT {
            phi[ix][iy] = 0.5 - 0.5 * (50.0 * (0.5-(iy as f32)/DOM_HEIGHT as f32) / (DX * EPSILON) + 20.0 * (4.0 * 2.0 * f32::consts::PI * ix as f32/DOM_WIDTH as f32).cos()).tanh();
        }
    }
    let xcenter = (DOM_WIDTH as f32)/2.0;
    let ycenter = (DOM_HEIGHT as f32)/2.0;
    // generate_field_circular(phi, xcenter - 25.0, ycenter, RADIUS*DX);
    // generate_field_circular(bct, xcenter, ycenter, RADIUS2*DX);
}

fn f_phi(x: f32) -> f32{
    72.0 * x * (1.0-x) * (x-0.5)
}

fn sum_matrix(u: &DomainMatrix) -> f32 {
    let mut sum = 0.0;
    for ix in 0..DOM_WIDTH {
        for iy in 0..DOM_HEIGHT {
            sum += u[ix][iy];
        }
    }
    sum
}

fn compute_centroid(u: &DomainMatrix) -> (f32, f32) {
    let mut cx = 0.0;
    let mut cy = 0.0;
    for ix in 0..DOM_WIDTH {
        for iy in 0..DOM_HEIGHT {
            cx += u[ix][iy] * ix as f32;
            cy += u[ix][iy] * iy as f32;
        }
    }
    let usum = sum_matrix(u);
    cx /= usum;
    cy /= usum;
    (cx, cy)
}

fn shift_matrix(u: &mut DomainMatrix, sx: i32, sy: i32) {
    let mut newu: DomainMatrix = [[0.0; DOM_HEIGHT]; DOM_WIDTH];
    for ix in 0..(DOM_WIDTH as i32) {
        for iy in 0..(DOM_HEIGHT as i32) {
            if 0 <= ix + sx && ix + sx < DOM_WIDTH as i32 && 0 <= iy + sy && iy + sy < DOM_HEIGHT as i32 {
                newu[ix as usize][iy as usize] = u[(ix+sx) as usize][(iy+sy) as usize];
            }
        }
    }
    *u = newu;
}

fn curvature_factor(c: f32) -> f32 {
    if CURVATURE_DEPENDENCE {
        let f = 4.0 * (c - MIN_CURVATURE) / (MAX_CURVATURE - MIN_CURVATURE)
            - 4.0 * (c - MIN_CURVATURE).powi(2) * (MAX_CURVATURE - MIN_CURVATURE).powi(-2);
        return f.max(0.0)
    }
    return 1.0
}

fn save_images(outdir: &str, phi: &DomainMatrix, bct: &DomainMatrix, curv: &DomainMatrix, vn: &DomainMatrix, step: i32) {
    fn f32_to_grayscale_pos(x: f32) -> Rgb<u8> {
        let c = (x*255.0) as u8;
        Rgb([c, c, c])
    }
    // fn f32_to_grayscale(x: f32) -> u8 {
    //     ((x+1.0)/2.0*50.0) as u8
    // }
    fn f32_to_curv_color(x: f32) -> Rgb<u8> {
        if x == 0.0 {return Rgb([0,0,0])};
        let c = x as i32;
        // if x > 100.0 * MAX_CURVATURE {return Rgb([(-c.min(0) as u8).min(255), 255, (c.max(0) as u8).min(255)]);}
        // if x > 100.0 * MIN_CURVATURE {return Rgb([(-c.min(0) as u8).min(255), 128, (c.max(0) as u8).min(255)]);}
        // Rgb([(-c.min(0) as u8).min(255), 64, (c.max(0) as u8).min(255)])
        if x > 100.0 * MAX_CURVATURE {return Rgb([255, 0, 0]);}
        if x > 100.0 * (MAX_CURVATURE + MIN_CURVATURE) / 2.0 {return Rgb([255, 255, 0]);}
        if x > 100.0 * MIN_CURVATURE {return Rgb([0, 255, 0]);}
        if x > 0.0 * MIN_CURVATURE {return Rgb([0, 0, 255]);}
        Rgb([64, 0, 255])
    }
    fn f32_to_color(x: f32) -> Rgb<u8> {
        if x == 0.0 {return Rgb([0,0,0])};
        let c = x as i32;
        Rgb([(-c.min(0) as u8).min(255), 64, (c.max(0) as u8).min(255)])
    }
    fn f32_to_color_two(x: f32, y: f32) -> Rgb<u8> {
        if x == 0.0 {return Rgb([0,0,0])};
        Rgb([(y*255.0) as u8, 0, (x*255.0*0.3 + y*255.0*0.5) as u8])
    }
    let mut image_phi = RgbImage::new(DOM_WIDTH as u32, DOM_HEIGHT as u32);
    let mut image_bct = RgbImage::new(DOM_WIDTH as u32, DOM_HEIGHT as u32);
    let mut image_display = RgbImage::new(DOM_WIDTH as u32, DOM_HEIGHT as u32);
    let mut image_curv = RgbImage::new(DOM_WIDTH as u32, DOM_HEIGHT as u32);
    let mut image_vn = RgbImage::new(DOM_WIDTH as u32, DOM_HEIGHT as u32);
    for ix in 0..DOM_WIDTH {
        for iy in 0..DOM_HEIGHT {
            image_phi.put_pixel(ix as u32, iy as u32, f32_to_grayscale_pos(phi[ix][iy]));
            image_bct.put_pixel(ix as u32, iy as u32, f32_to_grayscale_pos(bct[ix][iy] * phi[ix][iy]));
            image_display.put_pixel(ix as u32, iy as u32, f32_to_color_two(phi[ix][iy], bct[ix][iy] * phi[ix][iy]));
            image_curv.put_pixel(ix as u32, iy as u32, f32_to_curv_color(100.0*curv[ix][iy]));
            image_vn.put_pixel(ix as u32, iy as u32, f32_to_color(500.0 * vn[ix][iy]));
        }
    }
    image_phi.save(format!("{}/phi_{:0>6}.png", outdir, step/500)).expect("image could not be saved");
    image_bct.save(format!("{}/bct_{:0>6}.png", outdir, step/500)).expect("image could not be saved");
    image_display.save(format!("{}/display_{:0>6}.png", outdir, step/500)).expect("image could not be saved");
    image_curv.save(format!("{}/curv_{:0>6}.png", outdir, step/500)).expect("image could not be saved");
    image_vn.save(format!("{}/vn_{:0>6}.png", outdir, step/500)).expect("image could not be saved");
}

fn log_write(path: &str, string: String) {
    let mut log_file = match OpenOptions::new().append(true).open(path) {
        Ok(f) => f,
        Err(_e) => OpenOptions::new().create_new(true).append(true).open(path).expect("Error creating file")
    };
    log_file.write(string.as_bytes());
}

fn save_angle_histogram(u: &DomainMatrix, centroid: (f32, f32), fname: &str, threshold: f32) {
    let mut hist = [0.0; HISTOGRAM_RESOLUTION];
    let mut count = [0.0; HISTOGRAM_RESOLUTION];
    for ix in 0..DOM_WIDTH {
        for iy in 0..DOM_HEIGHT {
            if u[ix][iy].abs() > threshold {
                let mut angle = libm::atan2(centroid.0 as f64 - ix as f64, centroid.1 as f64 - iy as f64) as f32;
                angle = angle / (2.0 * f32::consts::PI);
                angle = ((angle * HISTOGRAM_RESOLUTION as f32) + HISTOGRAM_RESOLUTION as f32) % HISTOGRAM_RESOLUTION as f32;
                let angle = angle as usize;
                hist[angle] += u[ix][iy];
                count[angle] += 1.0;
            }
        }
    }
    let mut histogram_file = match OpenOptions::new().append(true).open(fname) {
        Ok(f) => f,
        Err(e) => OpenOptions::new().create_new(true).append(true).open(fname).expect("Error creating file")
    };
    for i in 0..HISTOGRAM_RESOLUTION {
        if count[i] > 0.0 {
            hist[i] /= count[i];
        }
    }
    let hist_as_str = hist.into_iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" ");
    writeln!(histogram_file, "{}", hist_as_str);
}

fn save_matrix(u: &DomainMatrix, fname: &str) {
    let mut file = match OpenOptions::new().append(true).open(fname) {
        Ok(f) => f,
        Err(e) => OpenOptions::new().create_new(true).append(true).open(fname).expect("Error creating file")
    };
    let matrix_as_str = u.into_iter().map(|row| row.into_iter().map(|x| format!("{:.5}", x)).collect::<Vec<String>>().join(" ")).collect::<Vec<String>>().join("; ");
    writeln!(file, "{}", matrix_as_str);
}

fn amoeba(args: Vec<String>) {
    let mut rng: Pcg64 = Seeder::from(SEED).make_rng();
    let outdir = match args.len() {
        1 => "out",
        2 => &args[1],
        _ => panic!("Too many command line parameters!")
    };
    println!("Output dir: {}", outdir);

    let mut phi: DomainMatrix = [[0.0; DOM_HEIGHT]; DOM_WIDTH];  // Phase field
    let mut bct: DomainMatrix = [[0.0; DOM_HEIGHT]; DOM_WIDTH];  // Concentration
    let mut xi: DomainMatrix = [[0.0; DOM_HEIGHT]; DOM_WIDTH];
    match SHAPE {
        Shape::Circle => initialize_circular(&mut phi, &mut bct, &mut rng),
        Shape::Flat => initialize_flat(&mut phi, &mut bct, &mut rng),
        Shape::BumpyCircle => initialize_bumpy_circle(&mut phi, &mut bct, &mut rng),
    }

    let mut grad_phi: DomainMatrix = [[0.0; DOM_HEIGHT]; DOM_WIDTH];
    let mut lap_phi: DomainMatrix = [[0.0; DOM_HEIGHT]; DOM_WIDTH];
    let mut d_bct_d_phi: DomainMatrix = [[0.0; DOM_HEIGHT]; DOM_WIDTH];
    let mut curv_phi: DomainMatrix = [[0.0; DOM_HEIGHT]; DOM_WIDTH];
    let mut shift: (i32, i32) = (0, 0);

    let mut prevphi: DomainMatrix = [[0.0; DOM_HEIGHT]; DOM_WIDTH];  // Phase field
    for ix in 0..DOM_WIDTH {
        for iy in 0..DOM_HEIGHT {
            prevphi[ix][iy] = phi[ix][iy]
        }
    }
    let mut prev_centroid: (f32, f32) = (0.0, 0.0);
    
    let mut ma2 = MA;
    let mut current_volume = match SHAPE {
        Shape::Circle => VOL,
        Shape::Flat => DOM_HEIGHT as f32 * DOM_WIDTH as f32 / 2.0,
        Shape::BumpyCircle => VOL,
    };
    let mut current_perimeter;
    let amplitude: f32 = (2.0 * SIGMA2 * DT).sqrt() / (DX);
    let mut step = 0;
    let mut time = 0.0;
    let mut delta2 = DELTA02;
    let mut alpha = ALPHA0;
    let mut max_curv = 0.0;
    while time < STOP_TIME {
        step += 1;
        time += DT;
        let target_volume = match SHAPE {
            Shape::Circle => VOL,
            Shape::Flat => DOM_HEIGHT as f32 * DOM_WIDTH as f32 / 2.0,
            Shape::BumpyCircle => VOL,
        };
        ma2 += (DT / TAUMA) * (current_volume - target_volume);

        let mut phi2 = phi.clone();
        let mut bct2 = bct.clone();
        gradient(&phi, &mut grad_phi);
        laplacian(&phi, &mut lap_phi);
        derivate_phase_field(&bct, &phi, &mut d_bct_d_phi);
        if step % 10 == 0 {
            curvature(&phi, &grad_phi, &mut curv_phi, &mut max_curv);
            // if min_curv < -30.0 {alpha = 1.2;}
            // else {alpha = ALPHA0;}
        }

        phi2.par_iter_mut().enumerate().for_each(|(ix, row)| {
        // for ix in 0..DOM_WIDTH {
            for iy in 0..DOM_HEIGHT {
                let phic = phi[ix][iy];
                let bctc = bct[ix][iy];
                let dphidt = 0.0
                    + (GAMMA/TAU) * (f_phi(phic) / EPSILON.powi(2) + lap_phi[ix][iy])
                    // - (MB/TAU) * phic * (1.0-phic) * grad_phi[ix][iy]
                    - BETA * (ma2/TAU) * grad_phi[ix][iy]
                    // + (alpha/TAU) * bctc * phic * grad_phi[ix][iy] 
                    // - (alpha/TAU) * bctc * phic * grad_phi[ix][iy] * curv_phi[ix][iy]
                    // + (alpha/TAU) * bctc * phic * grad_phi[ix][iy] * curv_phi[ix][iy].powi(2) * 0.005
                    // + 0.5 * (alpha/TAU) * bctc * phic * grad_phi[ix][iy] 
                    // - 8.0 * (alpha/TAU) * bctc * phic * grad_phi[ix][iy] * curvature_0actor(curv_phi[ix][iy])
                    + 4.0 * (alpha/TAU) * bctc * phic * grad_phi[ix][iy] * curvature_factor(curv_phi[ix][iy])
                    ;
                row[iy] += dphidt * DT;
                row[iy] = row[iy].min(1.0);
                if row[iy] < 0.00 {
                    // println!("Ostia");
                    // println!("{} {}, phi {} -> {}, bct {} -> {}", ix, iy, phi[ix][iy], row[iy], bct[ix][iy], bct2[ix][iy]);
                    // if true {println!("{} {} {} {}", (GAMMA/TAU) * (f_phi(phic) / EPSILON.powi(2) + lap_phi[ix][iy])
                    //     ,- (MB/TAU) * phic * (1.0-phic) * grad_phi[ix][iy]
                    //     ,- (ma2/TAU) * grad_phi[ix][iy]
                    //     , (alpha/TAU) * bctc * phic * grad_phi[ix][iy]
                    //     )};
                    // println!("{} {} {}", lap_phi[ix][iy], f_phi(phic), EPSILON.powi(2));
                    // return;
                }
            }
        });
        // }
        for ix in 0..DOM_WIDTH {
            for iy in 0..DOM_HEIGHT {
                // DO NOT parallellise this or results will be irreproducible
                xi[ix][iy] = xi[ix][iy] * (1.0 - DT * TAUXI_INV) +  amplitude * rng.sample::<f32,_>(StandardNormal);
                // xi[ix][iy] = 0.0;
            }
        }
        bct2.par_iter_mut().enumerate().for_each(|(ix, row)| {
        // for ix in 0..DOM_WIDTH {
            for iy in 0..DOM_HEIGHT {
                let phic = phi[ix][iy];
                let bctc = bct[ix][iy];
                if phic > THRESHOLD {
                    // let dbct = 0.0
                    //     + D_B * d_bct_d_phi[ix][iy] / phic
                    //     - REACT * DECAYX * bctc
                    //     // - (1.0 - REACT) * phic * xi[ix][iy]
                    //     - (1.0 - REACT) * phic
                    //     + REACT * KACT * bctc * (1.0-bctc) * (bctc - delta2)
                    //     + MEMBRANE_NOISE * REACT * phic * (1.0-phic) * xi[ix][iy]
                    //     + GLOBAL_NOISE * phic * xi[ix][iy]
                    //     + bctc * d_bct_d_phi[ix][iy]
                    //     ;
                    let dbct = 0.0
                        + D_B * d_bct_d_phi[ix][iy] / phic
                        + REACT * KACT * bctc * (1.0-bctc) * (bctc - delta2)
                        - REACT * DECAYX * bctc
                        + MEMBRANE_NOISE * REACT * (1.0-phic) * xi[ix][iy]
                        + bctc * d_bct_d_phi[ix][iy]
                        ;
                    row[iy] += DT * dbct;
                } else {
                    row[iy] += -DT * bctc * DECAYX;
                }
                row[iy] = row[iy].max(0.0).min(1.0);
            }
        });
        // }
        phi = phi2;
        bct = bct2;
        current_volume = sum_matrix(&phi);
        current_perimeter = sum_matrix(&grad_phi);

        let mut tot = 0.0;
        if step % SAVE_STEP == 1 {
            if SHIFT_DOMAIN {
                let centroid = compute_centroid(&phi);
                let newshift = (centroid.0 as i32 - (DOM_WIDTH/2) as i32, centroid.1 as i32 - (DOM_HEIGHT/2) as i32);
                shift_matrix(&mut phi, newshift.0, newshift.1);
                shift_matrix(&mut bct, newshift.0, newshift.1);
                shift_matrix(&mut curv_phi, newshift.0, newshift.1);
                shift_matrix(&mut prevphi, newshift.0, newshift.1);
                shift = (shift.0 + newshift.0, shift.1 + newshift.1);
            }

            tot = 0.0;
            for ix in 0..DOM_WIDTH {
                for iy in 0..DOM_HEIGHT {
                    tot += phi[ix][iy] * bct[ix][iy]
                }
            }
            // delta2 = match SHAPE {
            //     Shape::Circle => delta2 + (DT/TAUDELTA) * (-delta2 + 0.5 + 0.001*(tot-VOL2)),
            //     Shape::Flat => delta2 + (DT/TAUDELTA) * (-delta2 + 0.5 + 0.001*(tot-VOL2*5.0)),
            //     Shape::BumpyCircle => delta2 + (DT/TAUDELTA) * (-delta2 + 0.5 + 0.001*(tot-VOL2)),
            // };
            // delta2 = delta2.min(1.0);
            delta2 = 0.5 + DELTA_M*(tot-VOL2);
        }

        // if time > 0.1 {break;}
        // println!("\n")
        if step % SAVE_STEP == 1 {
            let centroid = compute_centroid(&phi);
            let prev_centroid = compute_centroid(&prevphi);
            save_angle_histogram(&curv_phi, centroid, format!("{}/curv_histogram.txt", outdir).as_str(), THRESHOLD);
            let mut chbct: DomainMatrix = [[0.0; DOM_HEIGHT]; DOM_WIDTH];  // Concentration
            for ix in 0..DOM_WIDTH {
                for iy in 0..DOM_HEIGHT {
                    chbct[ix][iy] = if phi[ix][iy] < 0.999 && phi[ix][iy] > 0.001 {bct[ix][iy].max(THRESHOLD*1.01)} else {0.0};
                }
            }
            let mut vn: DomainMatrix = [[0.0; DOM_HEIGHT]; DOM_WIDTH];  // Normal velocity
            let centroid_velocity = distance(prev_centroid.0, prev_centroid.1, centroid.0, centroid.1) * DX / (DT * SAVE_STEP as f32);
            let centroid_angle = libm::atan2((centroid.0 - prev_centroid.0) as f64, (centroid.1 - prev_centroid.1) as f64) as f32;
            let mut vnb: DomainMatrix = [[0.0; DOM_HEIGHT]; DOM_WIDTH];  // Normal velocity
            for ix in 0..DOM_WIDTH {
                for iy in 0..DOM_HEIGHT {
                    if phi[ix][iy] < 0.9 && phi[ix][iy] > 0.1 {
                        let angle = libm::atan2(centroid.0 as f64 - ix as f64, centroid.1 as f64 - iy as f64) as f32;
                        let centroid_factor = centroid_velocity * (centroid_angle - angle).cos();
                        let avn = (phi[ix][iy] - prevphi[ix][iy]) / (DT * SAVE_STEP as f32) / (grad_phi[ix][iy]).abs();
                        // vn[ix][iy] += centroid_velocity * (centroid_angle - angle).cos();
                        vnb[ix][iy] = avn;
                        vn[ix][iy] = avn + centroid_factor;
                    }
                }
            }
            for ix in 0..DOM_WIDTH {
                for iy in 0..DOM_HEIGHT {
                    prevphi[ix][iy] = phi[ix][iy];
                }
            }
            save_angle_histogram(&chbct, centroid, format!("{}/bct_histogram.txt", outdir).as_str(), THRESHOLD);
            save_angle_histogram(&vn, centroid, format!("{}/vn_histogram.txt", outdir).as_str(), 0.0);
            save_angle_histogram(&vnb, centroid, format!("{}/vnb_histogram.txt", outdir).as_str(), 0.0);
            save_matrix(&phi, format!("{}/phi.txt", outdir).as_str());
            save_matrix(&curv_phi, format!("{}/curv.txt", outdir).as_str());
            if step % (SAVE_STEP * IMAGE_SAVE_MULTIPLE) == 1 {save_images(outdir, &phi, &bct, &curv_phi, &vn, step);}
            let logstring = format!("{}. Area: {}, Perimeter: {}, Conc: {}, Delta2: {}, Alpha: {}, Max curv: {}, Shift: ({}, {})", time, current_volume, current_perimeter, tot/current_volume, delta2, alpha, max_curv, shift.0, shift.1);
            println!("{}", logstring);
            log_write(format!("{}/log.txt", outdir).as_str(), logstring);
            log_write(format!("{}/shift.txt", outdir).as_str(), format!("{} {}\n", shift.0, shift.1));
            log_write(format!("{}/centroid.txt", outdir).as_str(), format!("{} {}\n", centroid.0, centroid.1));
            log_write(format!("{}/time.txt", outdir).as_str(), format!("{}\n", time));
        }
        
    }

}


fn main() {
    let args: Vec<String> = env::args().collect();

    ctrlc::set_handler(move || {
        println!("Exit.");
        std::process::exit(0);
    })
    .expect("Error setting Ctrl-C handler");

    amoeba(args);
}
