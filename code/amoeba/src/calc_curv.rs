
use image::{RgbImage, Rgb};
use image::io::Reader as ImageReader;
use std::fs::{OpenOptions, read_dir};

fn calc_curvatures(path: &str) {
    println!("{}", path);
    let mut filelist: Vec<_> = read_dir(path).unwrap().map(|r| r.unwrap()).collect();
    filelist.sort_by_key(|f| f.path());
    for file in filelist {
        if file.path().file_name().expect("Error") == "shifts.txt" {
            continue;
        }
        let img = ImageReader::open(file.path()).expect("Error").decode();
        break;
    }
}


fn main() {
    calc_curvatures("/home/ferran/tfm/analysis/tmp6/b5/");
}
