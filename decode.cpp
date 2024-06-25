#include "autoencoder.h"
#include "serialize.h"

#include <fstream>


int main(int argc, char* argv[]){
    /**
     * 1. Parse Inputs
     */
    const std::string file_in_name = argv[1];
    const std::string file_out_name = argv[2];

    /**
     * 2. Initialize arrays
     */
    std::vector<uint8_t> decoded_bytes(0);
    std::vector<uint8_t> decoded_wav_header;
    std::vector<float> decoded_latent(0);
    std::vector<int16_t> decoded_residuals(0);
    std::vector<int16_t> reconstructed_samples(0);


    /**
     * 3. Read input file to byte array and deserialize bytes
     */

    std::ifstream inputFile(file_in_name, std::ios::binary);
        
    // Get file size
    inputFile.seekg(0, std::ios::end);
    std::streamsize fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);
    
    // Read the file bytes into vector
    std::vector<uint8_t> encoded_bytes(fileSize);
    inputFile.read(reinterpret_cast<char*>(encoded_bytes.data()), fileSize);
    inputFile.close();

    // Deserialize into latent, residuals, and wav file header
    deserializeEncoding(encoded_bytes,decoded_wav_header, decoded_latent, decoded_residuals);
    
    /**
     * 4. Decode encoded bytes using decoder model, latent vector, and residuals vector
     */
    Autoencoder autoencoder;
    autoencoder.reconstructSamples(decoded_latent, decoded_residuals, reconstructed_samples);
    autoencoder.postprocess(reconstructed_samples);
    
    /**
     * 5. Write to wav file
     */
    writeWavFile(file_out_name, decoded_wav_header, reconstructed_samples);


    return 0;
}