use std::{io::{Read, Write}, fs::File};
use csv::ReaderBuilder;
use flate2::read::GzDecoder;
use linfa::{Dataset, prelude::{ToConfusionMatrix}, traits::{Fit, Predict, Transformer}};
use ndarray::prelude::*;
use ndarray_csv::*;
use linfa_preprocessing::linear_scaling::LinearScaler;
use ndarray_rand::rand::SeedableRng;
use rand::rngs::SmallRng;
use linfa_trees::{DecisionTree, SplitQuality};

pub fn array_from_csv<R: Read>(
    csv: R,
    has_headers: bool,
    seperator: u8,

)-> Result<Array2<f64>, ReadError>{
    let mut reader = ReaderBuilder::new()
        .has_headers(has_headers)
        .delimiter(seperator)
        .from_reader(csv);

    // extract ndarray
    reader.deserialize_array2_dynamic()
}

pub fn array_from_csv_gz<R: Read>(
    gz: R,
    has_headers: bool,
    seperator: u8,
)-> Result<Array2<f64>, ReadError>{
  let file = GzDecoder::new(gz);
  array_from_csv(file, has_headers, seperator)
}

pub fn winequality() -> Dataset<f64, usize, Ix1> {
    let data = include_bytes!("../winequality-red.csv.gz");
    let array = array_from_csv_gz(&data[..],true,b',').unwrap();

    let (data, targets) = (
        array.slice(s![..,0..11]).to_owned(),
        array.column(11).to_owned(),
    );

    let feature_names = vec![
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ];

    Dataset::new(data, targets)
        .map_targets(|x| *x as usize)
        .with_feature_names(feature_names)
}

pub fn decision_tree_classification_linear_scaler() -> linfa_trees::Result<()>{
    let mut rng = SmallRng::seed_from_u64(42);

    let (train, test) = winequality()
        .shuffle(&mut rng)
        .split_with_ratio(0.8);

    // preprocessing linear scaler
    let scaler = LinearScaler::standard().fit(&train).unwrap();
    let train_pre = scaler.transform(train);
    let test_pre = scaler.transform(test);

    // gini criterion
    let gini_model = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(100))
        .min_weight_split(1.0)
        .min_weight_leaf(1.0)
        .fit(&train_pre)?;

    let gini_pred_y = gini_model.predict(&test_pre);
    let cm = gini_pred_y.confusion_matrix(&test_pre)?;

    println!("{:?}", cm);

    println!(
        "Test accuracy with Gini criterion: {:.2}%",
        100.0 * cm.accuracy()
    );

    let feats = gini_model.features();
    println!("Features trained in this tree {:?}", feats);

    println!("Training model with entropy criterion ...");
    let entropy_model = DecisionTree::params()
        .split_quality(SplitQuality::Entropy)
        .max_depth(Some(100))
        .min_weight_split(10.0)
        .min_weight_leaf(10.0)
        .fit(&train_pre)?;
    
    let entropy_pred_y = entropy_model.predict(&test_pre);
    let cm = entropy_pred_y.confusion_matrix(&test_pre)?;
    
    println!("{:?}", cm);
    
    println!(
        "Test accuracy with Entropy criterion: {:.2}%",
        100.0 * cm.accuracy()
    );
    
    let feats = entropy_model.features();
    println!("Features trained in this tree {:?}", feats);

    println!("---------------------------------------------------------------------------------------------");

    let mut tikz = File::create("decision_tree_example.tex").unwrap();
    tikz.write_all(
        gini_model
            .export_to_tikz()
            .with_legend()
            .to_string()
            .as_bytes(),
    )
    .unwrap();
    println!(" => generate Gini tree description with `latex decision_tree_example.tex`!");
    
    Ok(())
}

pub fn decision_tree_classification()-> linfa_trees::Result<()>{
    // load dataset
    let mut rng = SmallRng::seed_from_u64(42);

    let (train, test) = winequality()
        .shuffle(&mut rng)
        .split_with_ratio(0.8);

    println!("Training model with Gini criterion ...");
    let gini_model = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .max_depth(Some(100))
        .min_weight_split(1.0)
        .min_weight_leaf(1.0)
        .fit(&train)?;
    
    let gini_pred_y = gini_model.predict(&test);
    let cm = gini_pred_y.confusion_matrix(&test)?;
    
    println!("{:?}", cm);
    
    println!(
        "Test accuracy with Gini criterion: {:.2}%",
        100.0 * cm.accuracy()
    );
    
    let feats = gini_model.features();
    println!("Features trained in this tree {:?}", feats);
    
    println!("Training model with entropy criterion ...");
    let entropy_model = DecisionTree::params()
        .split_quality(SplitQuality::Entropy)
        .max_depth(Some(100))
        .min_weight_split(10.0)
        .min_weight_leaf(10.0)
        .fit(&train)?;
    
    let entropy_pred_y = entropy_model.predict(&test);
    let cm = entropy_pred_y.confusion_matrix(&test)?;
    
    println!("{:?}", cm);
    
    println!(
        "Test accuracy with Entropy criterion: {:.2}%",
        100.0 * cm.accuracy()
    );
    
    let feats = entropy_model.features();
    println!("Features trained in this tree {:?}", feats);
    
    let mut tikz = File::create("decision_tree_example.tex").unwrap();
    tikz.write_all(
        gini_model
            .export_to_tikz()
            .with_legend()
            .to_string()
            .as_bytes(),
    )
    .unwrap();
    println!(" => generate Gini tree description with `latex decision_tree_example.tex`!");
    
    Ok(())
}

fn main() {
    decision_tree_classification_linear_scaler();
    decision_tree_classification();
}
