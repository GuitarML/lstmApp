# lstmApp

A standalone executable for running [GuitarLSTM](https://github.com/GuitarML/GuitarLSTM) models on wav files
using frugally-deep. This was created in the process of developing
the [SmartAmpPro](https://github.com/GuitarML/SmartAmpPro), while investigating neural net inference
methods. I evenutally decided to write my own inference for the
plugin, but wanted to share my work here for anyone interested. 

Only source code and example .wav and model are included, it's up to the user on how to build.
```
After compiling the executable:

Usage:  lstm_app.exe <in_wav> <out_wav> <model.json> <input_size>


Example using the example wav and model: 
 ./lstm-app.exe x_test.wav output.wav ts9_fdeep100.json 100


Note: "input_size" is a parameter of the GuitarLSTM model.
       Since it's not part of Keras, it needs to be a separate argument.
```

## Dependencies and Custom Modifications

[Frugally Deep](https://github.com/Dobiasd/frugally-deep): Header-only library for using Keras models in C++.<br>
    Requires: [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page), [JSON](https://github.com/nlohmann/json), and [FunctionalPlus](https://github.com/Dobiasd/FunctionalPlus)

First convert your GuitarLSTM .h5 model into a .json model using
a modified version of frugally-deep that includes the "error_to_signal"
function. This allows the converter to include the custom loss function.

In the frugally-deep repository, modify the "keras_export/convert_model.py" 
with the following:


```
# At the beginning of the file (after imports and info section):

def pre_emphasis_filter(x, coeff=0.95):
    return tf.concat([x, x - coeff * x], 1)
    
def error_to_signal(y_true, y_pred): 
    """
    Error to signal ratio with pre-emphasis filter:
    """
    y_true, y_pred = pre_emphasis_filter(y_true), pre_emphasis_filter(y_pred)
    return K.sum(tf.pow(y_true - y_pred, 2), axis=0) / K.sum(tf.pow(y_true, 2), axis=0) + 1e-10
```


```
# In the "convert" function:

798     model = load_model(in_path, custom_objects={'error_to_signal' : error_to_signal})
```

Then run the conversion on your GuitarLSTM trained .h5 model as instructed in [FrugallyDeep](https://github.com/Dobiasd/frugally-deep).
Use the newly created .json file with lstm_app.exe to run the model on a properly formatted 
wav file (44.1k samplerate, mono, and float32 format wav)

Note: The .json file created here is only compatible with the Frugally-Deep interface, and will not run in SmartGuitarAmp or SmartAmpPro plugins.

