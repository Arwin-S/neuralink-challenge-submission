
#include "autoencoder.h"
#include "serialize.h"
#include <chrono>
#include <fstream>
#include <filesystem>

// TODO: build in Release mode
int main(int argc, char* argv[]){
    /**
     * 1. Parse Inputs
     */
    const std::string file_in_name = argv[1];
    const std::string file_out_name = argv[2];

    /**
     * 2. Initialize arrays
     */
    drwav wav;

    std::vector<int16_t> samples(0);
    std::vector<float>  latent(0);
    std::vector<float> predictions(0);
    std::vector<int16_t> residuals(0);
    vector<uint8_t> encoded_bytes(0);

    /**
     * 3. Model inference
     */
    Autoencoder autoencoder;

    autoencoder.readWav(file_in_name, wav, samples);
    autoencoder.preprocess(samples);
    autoencoder.encode(samples, latent);
    autoencoder.decode(latent, predictions);
    autoencoder.getResiduals(samples, predictions, residuals);
    
    /**
     * 4. Encode to byte array and write to file
     */
    serializeEncoding(file_in_name, latent, residuals, encoded_bytes);
    drwav_uninit(&wav);

    std::ofstream outputFile(file_out_name, std::ios::binary);
    outputFile.write(reinterpret_cast<const char*>(encoded_bytes.data()), encoded_bytes.size());
    outputFile.close();

    return 0;
}