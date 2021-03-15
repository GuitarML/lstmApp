//////////////////////////////////////////////////////////////////////////
// lstm_app.cpp 
//
//
//  A standalone executable for running GuitarLSTM models on wav files
//    using frugally-deep. 
//
//   Usage:  lstm_app.exe <in_wav> <out_wav> <model.json> <input_size>
//      
//   Example:
//          ./lstm-app.exe x_test.wav output.wav ts9_fdeep100.json 100
//
//   First convert your GuitarLSTM .h5 model into a .json model using
//     a modified version of frugally-deep that includes the "error_to_signal"
//     function.
//     
//////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "AudioFile.h"
#include "fdeep/fdeep.hpp"
#include "fplus/stopwatch.hpp"


int main(int argc, char** argv)
{
    if (argc < 5) {
        std::cout << "Usage:  tf_test.exe <in_wav> <out_wav> <model.json> <input_size>" << std::endl;
        return (0);
    }
    int input_size = atoi(argv[4]);
    
    AudioFile<double> audioFile;
    AudioFile<double> audioFile_out;
    AudioFile<double>::AudioBuffer buffer;
    audioFile.load(argv[1]);
    audioFile.printSummary();

    int channel = 0;
    int numSamples = audioFile.getNumSamplesPerChannel();
    buffer.resize(1);
    buffer[0].resize(numSamples - input_size);
    
    const auto model = fdeep::load_model(argv[3]);

    std::cout << "Running prediction on input wav file.." << std::endl;

    fplus::stopwatch stopwatch;
   
    std::cout << "Number of samples: " << numSamples << std::endl;
    fplus::stopwatch stopwatch2;
    for (int i = 0; i < numSamples - input_size; i++)
    {     
        // Set the range of samples of length "input_size" for input to neural net
        std::vector<float> range(&audioFile.samples[0][i], &audioFile.samples[0][i] + input_size);
        
        // Convert vector to frugally deep tensor and run prediction
        const fdeep::tensor input = fdeep::tensor(fdeep::tensor_shape(input_size, 1), range);
        const auto result = model.predict({ input });

        // Add prediction to output buffer
        buffer[0][i] = result.front().to_vector()[0];
    }

    std::cout << "Time test (average single sample predition time): " << stopwatch.elapsed() / numSamples << std::endl;
    
    std::cout << "Writing output wav from buffer.." << std::endl;
    bool ok = audioFile_out.setAudioBuffer(buffer);
    int numChannels = 1;
    // Set both the number of channels and number of samples per channel
    audioFile_out.setAudioBufferSize(numChannels, numSamples);

    // Set the number of samples per channel
    audioFile_out.setNumSamplesPerChannel(numSamples);

    // Set the number of channels
    audioFile_out.setNumChannels(numChannels);
    //Set bit depth and sample rate
    audioFile_out.setBitDepth(16);
    audioFile_out.setSampleRate(44100);
    // Save to output wav file
    audioFile_out.save(argv[2]);
}
