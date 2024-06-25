#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <string>
#include <vector>
#include <cassert>

#define DR_WAV_IMPLEMENTATION // make sure this is above the include dr_wav.h
#include "dr_wav.h"

class Autoencoder
{
private:
    void *env;
    void *session_options;
    void *encoder_session;
    void *decoder_session;

public:
    Autoencoder();
    ~Autoencoder();

    void readWav(const std::string fname, drwav &wav, std::vector<int16_t> &samples);
    void preprocess(std::vector<int16_t> &samples);
    void postprocess(std::vector<int16_t> &samples);
    void encode(const std::vector<int16_t> &samples, std::vector<float> &latent);
    void decode(const std::vector<float> &latent, std::vector<float> &decoding);

    void getResiduals(const std::vector<int16_t> &samples, const std::vector<float> &decoding, std::vector<int16_t> &residuals);
    void reconstructSamples(const std::vector<float> &latent, const std::vector<int16_t> &residuals, std::vector<int16_t> &reconstructed_samples);
};

/**
 * @brief Constructs an Autoencoder object.
 * 
 * This constructor initializes the environment and session options for ONNX Runtime,
 * and loads the encoder and decoder ONNX models from specified paths.
 */
Autoencoder::Autoencoder()
{
    // Initialize environment and session options
    env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
    session_options = new Ort::SessionOptions();
    static_cast<Ort::SessionOptions *>(session_options)->SetIntraOpNumThreads(1);
    static_cast<Ort::SessionOptions *>(session_options)->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    // Load encoder session
    const std::string encoder_path = std::string(MODEL_DIR) + "/encoder.onnx";
    encoder_session = new Ort::Session(*static_cast<Ort::Env *>(env), encoder_path.c_str(), *static_cast<Ort::SessionOptions *>(session_options));

    // Load decoder session
    const std::string decoder_path = std::string(MODEL_DIR) + "/decoder.onnx";
    decoder_session = new Ort::Session(*static_cast<Ort::Env *>(env), decoder_path.c_str(), *static_cast<Ort::SessionOptions *>(session_options));

    // std::cout << "Encoder/Decoder models loaded successfully!" << std::endl;
}

/**
 * @brief Destructs the Autoencoder object.
 * 
 * This destructor releases the memory allocated for the ONNX Runtime environment,
 * session options, encoder session, and decoder session.
 */
Autoencoder::~Autoencoder()
{
    // onnx env variavles
    delete static_cast<Ort::Session *>(encoder_session);
    delete static_cast<Ort::Session *>(decoder_session);
    delete static_cast<Ort::SessionOptions *>(session_options);
    delete static_cast<Ort::Env *>(env);
}

/**
 * @brief Reads a WAV file and extracts PCM samples.
 * 
 * This function initializes a WAV file using dr_wav library, reads the PCM samples,
 * and stores them in the provided vector.
 * 
 * @param fname The file name of the WAV file to read.
 * @param wav Reference to a drwav structure to hold WAV file information.
 * @param samples Reference to a vector to store the extracted PCM samples.
 */
void Autoencoder::readWav(const std::string fname, drwav &wav, std::vector<int16_t> &samples)
{
    // Initialize the WAV file
    if (!drwav_init_file(&wav, fname.c_str(), nullptr))
    {
        std::cerr << "Failed to open WAV file: " << fname << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read the PCM samples into the allocated memory
    samples.resize(wav.totalPCMFrameCount, 0);
    drwav_read_pcm_frames_s16(&wav, wav.totalPCMFrameCount, samples.data());
}

/**
 * @brief Preprocesses PCM samples from int16 to int10.
 * 
 * This function scales down the PCM samples by dividing each sample by 64 and flooring the result.
 * 
 * @param samples Reference to a vector of PCM samples to preprocess.
 */
void Autoencoder::preprocess(std::vector<int16_t> &samples){
    for (int i = 0; i < samples.size(); ++i)
    {
        samples[i] = static_cast<int16_t>(std::floor(static_cast<double>(samples[i]) / 64.0));
    }
}

/**
 * @brief Postprocesses PCM samples from int10 to int16.
 * 
 * This function scales up the PCM samples by multiplying each positive sample by 64.061577 
 * and adding 31.034184, and similarly adjusting negative samples.
 * 
 * @param samples Reference to a vector of PCM samples to postprocess.
 */
void Autoencoder::postprocess(std::vector<int16_t> &samples)
{
    for (int i = 0; i < samples.size(); ++i)
    {
        if (samples[i] >= 0) {
		    samples[i] = std::round(samples[i] * 64.061577 + 31.034184);
	    }
        else {
            samples[i] = -1 * std::round((-samples[i] -1) * 64.061577 + 31.034184) - 1;
        }
    }
}


/**
 * @brief Encodes the given samples using the encoder model and stores the result in the latent tensor.
 *
 * This function divides the input samples into windows of 256 samples each (padding the
 * last window with 0's), processes each window using the encoder model (with latent size of 8),
 * and stores the  * encoded results in the latent tensor. The latent tensor is initialized with the
 * appropriate shape to hold the encoded data for all windows.
 *
 * @param samples A vector of int16_t containing the input samples to be encoded.
 * @param latent A reference to an Ort::Value that will store the encoded output tensor (float32).
 */
void Autoencoder::encode(const std::vector<int16_t> &samples, std::vector<float> &latent)
{
    // number of 256 sized subarrays ("windows") in samples (rounded up)
    size_t num_windows = (samples.size() + 255) / 256;

    latent.resize(num_windows * 8, 0.0f);

    // Create memory info
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    for (size_t window = 0; window < num_windows; ++window)
    {
        // get window start,end points in terms of sample index
        size_t start_index = window * 256;
        size_t end_index = std::min(start_index + 256, samples.size());

        // padding is done automatically since array is all zeros by default
        std::vector<float> model_input_vector(256, 0.0f);
        for (size_t i = start_index; i < end_index; ++i)
        {
            model_input_vector[i - start_index] = static_cast<float>(samples[i]);
            /*  if end_index < samples.size(), the unassigned elements will be
                0, thereby padding by itself   */
        }
        std::array<int64_t, 3> model_input_shape = {1, 1, 256};

        // Create the model input tensor
        Ort::Value model_input_tensor = Ort::Value::CreateTensor<float>(memory_info, model_input_vector.data(), model_input_vector.size(), model_input_shape.data(), model_input_shape.size());

        // Prepare model output tensor
        std::vector<const char *> model_input_node_name = {"input"};
        std::vector<const char *> model_output_node_name = {"output"};
        auto model_output_tensor = static_cast<Ort::Session *>(encoder_session)->Run(Ort::RunOptions{nullptr}, model_input_node_name.data(), &model_input_tensor, 1, model_output_node_name.data(), 1);

        // save model output tensor to latent tensor's data ptr
        float *model_output_array = model_output_tensor[0].GetTensorMutableData<float>();

        /** Example:
         * if window = 10, then start_index, end_index = 80,88
         * window*8 = 80, so elements 80-87 (inclusive) of latent
         * vector will be modified
         */
        std::copy(model_output_array, model_output_array + 8, latent.data() + (window * 8));
    }
}

/**
 * @brief Decodes the given latent tensor using the decoder model and stores the result in the decoding vector.
 *
 * This function processes each 8-element segment of the latent tensor using the decoder model,
 * and stores the decoded results in the decoding vector. The decoding vector is initialized
 * with the appropriate shape to hold the decoded data for all windows.
 *
 * @param latent A vector of float containing the encoded latent representation to be decoded.
 * @param decoding A reference to a vector that will store the decoded output samples (float).
 */
void Autoencoder::decode(const std::vector<float> &latent, std::vector<float> &decoding)
{
    size_t num_windows = latent.size() / 8;
    decoding.resize(num_windows * 256, 0.0f);

    if (latent.size() % 8 != 0)
    {
        std::cerr << "Latent length is not a multiple of 8!\n";
        exit(EXIT_FAILURE);
    }

    // Create memory info
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    for (size_t window = 0; window < num_windows; ++window)
    {
        // get window start,end points in terms of latent index
        size_t start_index = window * 8;
        size_t end_index = start_index + 8;

        // padding is done automatically since array is all zeros by default
        std::vector<float> model_input_vector(8, 0.0f);
        for (size_t i = start_index; i < end_index; ++i)
            model_input_vector[i - start_index] = static_cast<float>(latent[i]);

        std::array<int64_t, 3> model_input_shape = {1, 1, 8};

        // Create the model input tensor
        Ort::Value model_input_tensor = Ort::Value::CreateTensor<float>(memory_info, model_input_vector.data(), model_input_vector.size(), model_input_shape.data(), model_input_shape.size());

        // Prepare model output tensor
        std::vector<const char *> model_input_node_name = {"input"};
        std::vector<const char *> model_output_node_name = {"output"};
        auto model_output_tensor = static_cast<Ort::Session *>(decoder_session)->Run(Ort::RunOptions{nullptr}, model_input_node_name.data(), &model_input_tensor, 1, model_output_node_name.data(), 1);

        // save model output tensor to latent tensor's data ptr
        float *model_output_array = model_output_tensor[0].GetTensorMutableData<float>();

        std::copy(model_output_array, model_output_array + 256, decoding.data() + (window * 256));
    }
}

/**
 * @brief Calculates the residuals between the original samples and the decoded samples.
 *
 * This function calculates the difference between each original sample and its corresponding
 * decoded sample (rounded to the nearest integer), and stores the result in the residuals vector.
 * The residuals vector is resized to match the size of the samples vector.
 *
 * @param samples A vector of int16_t containing the original input samples.
 * @param decoding A vector of float containing the decoded output samples.
 * @param residuals A reference to a vector that will store the computed residuals (int16_t).
 */
void Autoencoder::getResiduals(const std::vector<int16_t> &samples, const std::vector<float> &decoding, std::vector<int16_t> &residuals)
{
    /**
     *  resize residuals to sample size, so we don't have to worry about
     *  padded values in the decoding vector
     */
    residuals.resize(samples.size(), 0);
    for (size_t i = 0; i < samples.size(); ++i)
    {
        int rounded_decoding = std::round(decoding[i]);
        int residual = samples[i] - rounded_decoding;
        assert(residual >= -512 && residual <= 511);
        residuals[i] = static_cast<int16_t>(residual);
    }
}

/**
 * @brief Reconstructs the original samples from the latent tensor and residuals.
 *
 * This function decodes the latent tensor to get the decoded samples, then adds the
 * residuals to the decoded samples to reconstruct the original samples. The reconstructed
 * samples vector is resized to match the size of the residuals vector.
 *
 * @param latent A vector of float containing the encoded latent representation.
 * @param residuals A vector of int16_t containing the residuals between the original and decoded samples.
 * @param reconstructed_samples A reference to a vector that will store the reconstructed samples (int16_t).
 */
void Autoencoder::reconstructSamples(const std::vector<float> &latent, const std::vector<int16_t> &residuals, std::vector<int16_t> &reconstructed_samples){
    
    reconstructed_samples.resize(residuals.size(), 0);
    std::vector<float> decoding(0);

    decode(latent, decoding);
    decoding.resize(residuals.size()); // clip unneeded values in decoding (0-255 elements)

    assert(residuals.size() == decoding.size());
    for (size_t i = 0; i < residuals.size(); ++i)
        reconstructed_samples[i] = std::round(decoding[i]) + residuals[i]; 
}

