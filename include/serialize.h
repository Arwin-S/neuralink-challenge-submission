#include <iostream>
#include <fstream>
#include <bitset>
#include <cstdint>
#include <vector>
#include <deque>

using namespace std;

/**
 * @brief Converts a vector of int16_t values to a packed 10-bit bit array.
 *
 * This function takes a vector of int16_t values, each of which is expected to be within the range of -512 to 511.
 * Each value is converted to its 10-bit binary representation and stored in a bit array.
 *
 * The conversion is performed by taking the least significant 10 bits of each int16_t value.
 * The resulting bit array will have a length of 10 times the size of the input vector.
 *
 * @param int16_array A vector of int16_t values, each expected to be within the range of -512 to 511.
 * @return A vector of bools representing the packed 10-bit binary values of the input vector.
 */
std::vector<bool> int16ToBitArray(const std::vector<int16_t> &int16_array) {
    std::vector<bool> bit_array(int16_array.size() * 10, 0);
    for (size_t i = 0; i < int16_array.size(); i++) {
        std::bitset<10> int10_bits(int16_array[i] & 0x3FF);
        for (int j = 9; j >= 0; --j) {
            bit_array[i * 10 + (9 - j)] = int10_bits[j];
        }
    }
    return bit_array;
}

/**
 * @brief Converts a bit array to a byte array.
 *
 * This function takes a vector of bools representing bits and converts it into a vector of uint8_t values (bytes).
 * Each group of 8 bits from the input bit array is combined to form a byte in the output vector.
 * If the bit array's size is not a multiple of 8, the remaining bits are filled with zeros in the last byte.
 *
 * @param bit_array A vector of bools representing the bit array to be converted.
 * @return A vector of uint8_t values representing the converted bytes from the bit array.
 */
std::vector<uint8_t> bitArrayToBytes(const std::vector<bool>& bit_array) {
    std::vector<uint8_t> bytes((bit_array.size() + 7) / 8, 0);
    for (size_t i = 0; i < bit_array.size(); ++i) {
        bytes[i / 8] |= bit_array[i] << (7 - (i % 8));
    }
    return bytes;
}

/**
 * @brief Converts a byte array to a packed bit array.
 *
 * This function takes a vector of uint8_t values (bytes) and converts it into a vector of bools representing bits.
 * The number of bits to be extracted from the byte array is specified by the bit_count parameter.
 *
 * Each byte is processed to extract individual bits, which are then stored in the output bit array.
 * The function assumes that the total number of bits specified by bit_count is valid and can be extracted from the provided byte array.
 *
 * @param bytes A vector of uint8_t values representing the byte array to be converted.
 * @param bit_count The number of bits to be extracted from the byte array.
 * @return A vector of bools representing the packed bit array converted from the byte array.
 */
std::vector<bool> bytesToBitArray(const std::vector<uint8_t>& bytes, size_t bit_count) {
    std::vector<bool> bit_array(bit_count, 0);
    for (size_t i = 0; i < bit_count; ++i) {
        bit_array[i] = (bytes[i / 8] >> (7 - (i % 8))) & 1;
    }
    return bit_array;
}

/**
 * @brief Converts a packed 10-bit bit array to a vector of int16_t values.
 *
 * This function takes a vector of bools representing a packed 10-bit bit array and converts it into a vector of int16_t values.
 * Each group of 10 bits from the bit array is interpreted as an integer, and the resulting values are stored in the output vector.
 *
 * The function also handles sign extension for the 10-bit values to ensure they are correctly represented as 16-bit integers.
 * If the sign bit (bit 9) of a 10-bit value is set, the function converts it to the corresponding negative value.
 *
 * @param bit_array A vector of bools representing the packed 10-bit bit array to be converted.
 * @return A vector of int16_t values converted from the bit array.
 */
std::vector<int16_t> bitArrayToInt16(const std::vector<bool>& bit_array) {
    std::vector<int16_t> int16_arr(bit_array.size() / 10, 0);
    for (size_t i = 0; i < int16_arr.size(); i++) {
        int16_t value = 0;
        for (int j = 0; j < 10; ++j) {
            value = (value << 1) | bit_array[i * 10 + j];
        }
        if (value & 0x200) { // If the sign bit (bit 9) is set
            value = value - 0x400; // Convert to negative value
        }
        int16_arr[i] = value;
    }
    return int16_arr;
}

/**
 * @brief Serializes a vector of int16_t residuals into a byte array with noise region handling.
 *
 * This function serializes a given vector of int16_t residuals into a byte array. It handles two types of residuals:
 * small integers in the range of -128 to 127, and noisy values outside this range. Small integers are converted to
 * unsigned 8-bit values, while noisy values are processed in chunks, converted to a packed 10-bit bit array, and then
 * serialized into bytes. The start positions and lengths of these noisy regions are recorded in the bounds deque.
 *
 * @param residuals A vector of int16_t values representing the residuals to be serialized.
 * @param serialized_residuals A vector of uint8_t values where the serialized residuals will be stored.
 * @param bounds A deque of pairs where each pair represents the start position and length of a noisy region in the serialized data.
 */
void serializeResiduals(const std::vector<int16_t>& residuals, std::vector<uint8_t>& serialized_residuals, std::deque<std::pair<int32_t, int32_t>>& bounds) {
    
    auto convertToUnsigned = [](int16_t value) -> uint8_t {
        return static_cast<uint8_t>(value + 128);
    };
    
    size_t i = 0;
    while (i < residuals.size()) {
        if (residuals[i] < -128 || residuals[i] > 127) {
            // We found the start of a noisy region
            size_t noise_start = serialized_residuals.size();
            size_t noise_end = i;
            std::vector<int16_t> noisy_values;
            
            // Collect the noisy region
            while (i < residuals.size() && (residuals[i] < -128 || residuals[i] > 127)) {
                noisy_values.push_back(residuals[i]);
                ++i;
            }
            
            // Convert the noisy region to bit array and then to bytes
            std::vector<bool> noisy_bits = int16ToBitArray(noisy_values);
            std::vector<uint8_t> noisy_bytes = bitArrayToBytes(noisy_bits);
            serialized_residuals.insert(serialized_residuals.end(), noisy_bytes.begin(), noisy_bytes.end());
            
            // Record the noisy region
            size_t noise_length = serialized_residuals.size() - noise_start;
            bounds.push_back(std::make_pair(noise_start, noise_length));
        } else {
            // This is a small integer
            serialized_residuals.push_back(convertToUnsigned(residuals[i]));
            ++i;
        }
    }
}

/**
 * @brief Deserializes a byte array into a vector of int16_t residuals with noise region handling.
 *
 * This function deserializes a given byte array into a vector of int16_t residuals. It handles two types of data:
 * 1) Small integers that were originally in the range of -128 to 127, and noisy values that were serialized in chunks.
 * 2) The start positions and lengths of these noisy regions are provided in the bounds deque.
 *
 * @param serialized_residuals A vector of uint8_t values representing the serialized residuals.
 * @param bounds A deque of pairs where each pair represents the start position and length of a noisy region in the serialized data.
 * @param deserialized_residuals A vector of int16_t values where the deserialized residuals will be stored.
 */
void deserializeResiduals(const std::vector<uint8_t>& serialized_residuals, std::deque<std::pair<int32_t, int32_t>>& bounds, std::vector<int16_t>& deserialized_residuals) {
    size_t byte_pos = 0;
    
    while (byte_pos < serialized_residuals.size()) {
        // Check if the current position is the start of a noisy region
        if (!bounds.empty() && byte_pos == bounds.front().first) {
            int32_t noise_len = bounds.front().second;
            bounds.pop_front();
            std::vector<uint8_t> noise_bytes(serialized_residuals.begin() + byte_pos, serialized_residuals.begin() + byte_pos + noise_len);
            std::vector<bool> noisy_bits = bytesToBitArray(noise_bytes, noise_len * 8);
            std::vector<int16_t> noisy_residuals = bitArrayToInt16(noisy_bits);
            deserialized_residuals.insert(deserialized_residuals.end(), noisy_residuals.begin(), noisy_residuals.end());
            byte_pos += noise_len;
        } else {
            deserialized_residuals.push_back(static_cast<int16_t>(serialized_residuals[byte_pos] - 128));
            ++byte_pos;
        }
    }
    
}

/**
 * @brief Serializes a deque of boundary pairs into a byte array.
 *
 * This function serializes a given deque of pairs, where each pair consists of two int32_t values representing the 
 * start position and length of a noisy region. The serialized data includes the size of the deque followed by the 
 * serialized pairs of int32_t values.
 *
 * The function first clears the output vector, then calculates the size needed for serialization, and reserves 
 * the required space. It then serializes the size of the deque followed by the pairs of int32_t values.
 *
 * @param bounds A deque of pairs where each pair represents the start position and length of a noisy region.
 * @param serialized_bounds A vector of uint8_t values where the serialized boundary pairs will be stored.
 */
void serializeBoundaries(const std::deque<std::pair<int32_t, int32_t>>& bounds, std::vector<uint8_t>& serialized_bounds) {
    serialized_bounds.clear();

    // Calculate the size needed: size of the uint32_t (size of the deque) + size of the data
    size_t size_in_bytes = sizeof(uint32_t) + bounds.size() * sizeof(std::pair<int32_t, int32_t>);
    serialized_bounds.reserve(size_in_bytes);

    // Serialize the size of the deque
    uint32_t size = static_cast<uint32_t>(bounds.size());
    uint8_t* size_ptr = reinterpret_cast<uint8_t*>(&size);
    serialized_bounds.insert(serialized_bounds.end(), size_ptr, size_ptr + sizeof(uint32_t));

    // Serialize the pairs of int32_t
    for (const auto& pair : bounds) {
        const uint8_t* first_ptr = reinterpret_cast<const uint8_t*>(&pair.first);
        serialized_bounds.insert(serialized_bounds.end(), first_ptr, first_ptr + sizeof(int32_t));

        const uint8_t* second_ptr = reinterpret_cast<const uint8_t*>(&pair.second);
        serialized_bounds.insert(serialized_bounds.end(), second_ptr, second_ptr + sizeof(int32_t));
    }
}

/**
 * @brief Deserializes a byte array into a deque of boundary pairs.
 *
 * This function deserializes a given byte array into a deque of pairs, where each pair consists of two int32_t values 
 * representing the start position and length of a noisy region. The byte array includes the size of the deque followed 
 * by the serialized pairs of int32_t values.
 *
 * The function first reads the size of the deque from the byte array, resizes the deque accordingly, and then reads 
 * each pair of int32_t values from the byte array into the deque.
 *
 * @param serialized_bounds A vector of uint8_t values representing the serialized boundary pairs.
 * @param deserialized_bounds A deque of pairs where the deserialized bounds bytes will be stored.
 */
void deserializeBoundaries(const std::vector<uint8_t>& serialized_bounds, std::deque<std::pair<int32_t, int32_t>>& deserialized_bounds) {
    // Read the size of the deque
    uint32_t size;
    std::memcpy(&size, serialized_bounds.data(), sizeof(size));

    // Resize the deque to hold the pairs
    deserialized_bounds.resize(size);

    // Read the pairs of int32_t from the buffer
    size_t offset = sizeof(size);
    for (auto& pair : deserialized_bounds) {
        std::memcpy(&pair.first, serialized_bounds.data() + offset, sizeof(pair.first));
        offset += sizeof(pair.first);
        std::memcpy(&pair.second, serialized_bounds.data() + offset, sizeof(pair.second));
        offset += sizeof(pair.second);
    }
}

/**
 * @brief Serializes the header of a WAV file into a byte array.
 *
 * This function reads the header of a WAV file and stores it in a byte array. The WAV header is typically the first 
 * 44 bytes of the file. The function opens the file in binary mode, reads the first 44 bytes, and stores them in the 
 * provided byte array. If the header cannot be read, the byte array is cleared.
 *
 * @param file_path The path to the WAV file.
 * @param header_bytes A vector of type uint8_t where the WAV header bytes will be stored.
 */
void serializeWavHeader(const std::string& file_path, std::vector<uint8_t>& header_bytes) {

    // Open the file in binary mode.
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);


    // Determine the size of the file.
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // The WAV header is typically the first 44 bytes.
    const std::streamsize headerSize = 44;

    // Resize the vector to hold the header bytes.
    header_bytes.resize(headerSize);

    // Read the header bytes.
    if (!file.read(reinterpret_cast<char*>(header_bytes.data()), headerSize)) {
        std::cerr << "Failed to read WAV header from file: " << file_path << std::endl;
        header_bytes.clear();
    }
}


/**
 * @brief Deserializes a byte array into a WAV header.
 *
 * This function takes a byte array containing serialized data and extracts the first 44 bytes as the WAV header. 
 * If the byte array does not contain at least 44 bytes, the function outputs an error message and clears the 
 * header byte array.
 *
 * @param serialized_bytes A vector of type uint8_t representing the serialized data containing the WAV header.
 * @param header_bytes A vector of type uint8_t where the extracted WAV header bytes will be stored.
 */
void deserializeWavHeader(const std::vector<uint8_t>& serialized_bytes, std::vector<uint8_t>& header_bytes) {
    // The WAV header is typically the first 44 bytes.
    const size_t headerSize = 44;

    // Check if the serialized_bytes contains at least 44 bytes.
    if (serialized_bytes.size() < headerSize) {
        std::cerr << "Insufficient data to extract WAV header." << std::endl;
        header_bytes.clear();
        return;
    }

    // Resize the vector to hold the header bytes.
    header_bytes.resize(headerSize);

    // Copy the first 44 bytes from the serialized_bytes to header_bytes.
    std::memcpy(header_bytes.data(), serialized_bytes.data(), headerSize);
}

/**
 * @brief Serializes a vector of float latent values into a byte array.
 *
 * This function serializes a given vector of float latent values into a byte array. The serialized data includes the 
 * size of the vector followed by the float values. The function first clears the output byte array, reserves the 
 * required space, serializes the size of the vector, and then serializes the float values.
 *
 * @param latent A vector of float values representing the latent data to be serialized.
 * @param serialized_latent A vector of uint8_t where the serialized latent data will be stored.
 */
void serializeLatent(const std::vector<float>& latent, std::vector<uint8_t>& serialized_latent) {
    serialized_latent.clear();

    // Reserve enough space for the size and the data
    size_t size_in_bytes = sizeof(uint32_t) + latent.size() * sizeof(float);
    serialized_latent.reserve(size_in_bytes);

    // Serialize the size
    uint32_t size = static_cast<uint32_t>(latent.size());
    uint8_t* size_ptr = reinterpret_cast<uint8_t*>(&size);
    serialized_latent.insert(serialized_latent.end(), size_ptr, size_ptr + sizeof(uint32_t));

    // Serialize the data
    const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(latent.data());
    serialized_latent.insert(serialized_latent.end(), data_ptr, data_ptr + latent.size() * sizeof(float));
}

/**
 * @brief Deserializes a byte array into a vector of float latent values.
 *
 * This function deserializes a given byte array into a vector of float latent values. The byte array includes the size 
 * of the vector followed by the serialized float values. The function first clears the output float vector, checks if 
 * the byte array contains enough data, deserializes the size of the vector, and then deserializes the float values.
 *
 * @param serialized_latent A vector of uint8_t representing the serialized latent data.
 * @param latent A vector of float values where the deserialized latent data will be stored.
 */
void deserializeLatent(const std::vector<uint8_t>& serialized_latent, std::vector<float>& latent) {
    latent.clear();

    // Deserialize the size
    uint32_t size;
    const uint8_t* size_ptr = serialized_latent.data();
    std::memcpy(&size, size_ptr, sizeof(uint32_t));

    // Resize the latent vector to the correct size
    latent.resize(size);

    // Deserialize the data
    const uint8_t* data_ptr = size_ptr + sizeof(uint32_t);
    std::memcpy(latent.data(), data_ptr, size * sizeof(float));
}

/**
 * @brief Serializes WAV header, latent vector, residuals, and boundary data into a single byte vector.
 *
 * This function serializes the WAV header from the specified file, a vector of float latent values, a vector of int16_t residuals,
 * and their boundary data into a single byte vector. It first serializes each component individually, then combines them into 
 * the output byte vector in the following order: WAV header, latent vector, boundaries, and residuals.
 *
 * @param wav_fname The file path to the WAV file.
 * @param latent A vector of float values representing the latent data to be serialized.
 * @param residuals A vector of int16_t values representing the residuals to be serialized.
 * @param encoded_bytes A vector of uint8_t where the combined serialized data will be stored.
 */
void serializeEncoding(const std::string wav_fname, const std::vector<float>& latent,
                        const std::vector<int16_t>& residuals, std::vector<uint8_t>& encoded_bytes){
    std::vector<uint8_t> wav_header;
    std::vector<uint8_t> serialized_latent;
    std::vector<uint8_t> serialized_residuals;
    std::vector<uint8_t> serialized_bounds;

    std::deque<std::pair<int32_t, int32_t>> bounds;

    serializeWavHeader(wav_fname, wav_header);
    serializeLatent(latent, serialized_latent);
    serializeResiduals(residuals, serialized_residuals, bounds);
    serializeBoundaries(bounds, serialized_bounds);


    // Combine all components into a single byte vector
    encoded_bytes.clear();
    encoded_bytes.insert(encoded_bytes.end(), wav_header.begin(), wav_header.end());
    encoded_bytes.insert(encoded_bytes.end(), serialized_latent.begin(), serialized_latent.end());
    encoded_bytes.insert(encoded_bytes.end(), serialized_bounds.begin(), serialized_bounds.end());
    encoded_bytes.insert(encoded_bytes.end(), serialized_residuals.begin(), serialized_residuals.end());
}

/**
 * @brief Deserializes encoding components from a single byte vector.
 *
 * This function deserializes a single byte vector containing the WAV header, latent vector, residuals, and boundary data
 * into their respective components. It extracts and deserializes each component from the byte vector in the following order:
 * WAV header, latent vector, boundaries, and residuals.
 *
 * @param encoded_bytes A vector of uint8_t representing the combined serialized data.
 * @param wav_header A vector of uint8_t where the deserialized WAV header will be stored.
 * @param latent A vector of float values where the deserialized latent data will be stored.
 * @param residuals A vector of int16_t values where the deserialized residuals will be stored.
 */
void deserializeEncoding(const std::vector<uint8_t>& encoded_bytes, std::vector<uint8_t> &wav_header, std::vector<float>& latent, std::vector<int16_t>& residuals){
    
    deserializeWavHeader(encoded_bytes, wav_header);
    size_t byte_pos = 44;

    // latent
    std::vector<uint8_t> serialized_latent(encoded_bytes.begin() + byte_pos, encoded_bytes.end());
    deserializeLatent(serialized_latent, latent);
    byte_pos += sizeof(uint32_t) + latent.size() * sizeof(float);

    // bounds
    std::deque<std::pair<int32_t, int32_t>> bounds;
    std::vector<uint8_t> serialized_bounds(encoded_bytes.begin() + byte_pos, encoded_bytes.end());
    deserializeBoundaries(serialized_bounds, bounds);
    byte_pos += sizeof(uint32_t) + bounds.size() * sizeof(std::pair<int32_t, int32_t>);

    // residuals
    std::vector<uint8_t> serialized_residuals(encoded_bytes.begin() + byte_pos, encoded_bytes.end());
    deserializeResiduals(serialized_residuals, bounds, residuals);
}

/**
 * @brief Writes a WAV file with the given header and audio samples.
 *
 * This function writes a WAV file at the specified file path using the provided WAV header and audio samples.
 * It opens the file in binary mode, writes the header bytes, and then writes the audio samples. The function 
 * assumes the header size to be 44 bytes, which is typical for a WAV file header. If the header size is not 44 bytes, 
 * an error message is displayed. The function also handles file opening and writing errors.
 *
 * @param filePath The path to the output WAV file.
 * @param header_bytes A vector of uint8_t containing the WAV header bytes (expected to be 44 bytes).
 * @param samples A vector of int16_t containing the audio samples to be written to the WAV file.
 */
void writeWavFile(const std::string& filePath, const std::vector<uint8_t>& header_bytes, const std::vector<int16_t>& samples) {
    // Open the file in binary mode.
    std::ofstream file(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return;
    }

    // Write the WAV header
    if (header_bytes.size() != 44) {
        std::cerr << "Invalid WAV header size. Expected 44 bytes." << std::endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(header_bytes.data()), header_bytes.size());

    // Write the samples
    file.write(reinterpret_cast<const char*>(samples.data()), samples.size() * sizeof(int16_t));

    file.close();
    if (!file) {
        std::cerr << "Failed to write data to file: " << filePath << std::endl;
    }
}