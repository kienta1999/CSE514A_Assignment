<div>
  <h1>Installation</h1>
  <p>Make sure you have python3 installed</p>
  <p>Install the following packages:</p>
  <ul>
    <li>Pandas</li>
    <li>NumPy</li>
    <li>Sklearn</li>
    <li>Tensorflow</li>
    <li>Pandas</li>
  </ul>
  <h1>Code instruction</h1>
  <p>
    Order to read python files: <i>data.py</i>, <i>train.py</i>,
    <i>final_validation.py</i>
  </p>
  <p>
    <i>data.py</i> preprocesses and splits data into train and validation sets,
    and output the train/test data for the training and validation files
    (<i>train.py</i>, <i>final_validation.py</i>)
  </p>
  <p>
    Run <i>train.py</i> will print out to the console the best hyperparameter
    for each model and output the plots of the cross validation process.
  </p>
  <p>
    Run <i>final_validation.py</i> will train the best selected models on the
    train data as a whole and test the result on the final validation data. The
    summary of the final validation process will be saved as csv file at
    <i>final_validation_output.csv</i>
  </p>
</div>
