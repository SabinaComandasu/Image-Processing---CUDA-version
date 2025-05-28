#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define KERNEL_SIZE 5
#define KERNEL_RADIUS (KERNEL_SIZE / 2)

const std::string INPUT_DIR = "inputs\\";
const std::string OUTPUT_DIR = "outputs\\";

// Precomputed Gaussian kernel (sigma ≈ 1.0)
__constant__ float d_gaussianKernel[KERNEL_SIZE * KERNEL_SIZE] = {
    1,  4,  7,  4, 1,
    4, 16, 26, 16, 4,
    7, 26, 41, 26, 7,
    4, 16, 26, 16, 4,
    1,  4,  7,  4, 1
};

__global__ void invert_filter(unsigned char* img, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) * channels;

    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c)
            img[idx + c] = 255 - img[idx + c];
    }
}

__global__ void gaussian_blur(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) * channels;

    if (x >= KERNEL_RADIUS && x < (width - KERNEL_RADIUS) &&
        y >= KERNEL_RADIUS && y < (height - KERNEL_RADIUS)) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            float norm = 0.0f;

            for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ++ky) {
                for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; ++kx) {
                    int px = x + kx;
                    int py = y + ky;
                    int kernelIdx = (ky + KERNEL_RADIUS) * KERNEL_SIZE + (kx + KERNEL_RADIUS);
                    int pixelIdx = (py * width + px) * channels + c;
                    float weight = d_gaussianKernel[kernelIdx];
                    sum += weight * input[pixelIdx];
                    norm += weight;
                }
            }
            output[idx + c] = static_cast<unsigned char>(sum / norm);
        }
    }
}

int main() {
    std::string inputFile, outputFile;
    int operation;

    std::cout << "Enter input image filename (e.g. image.jpg): ";
    std::getline(std::cin, inputFile);

    std::cout << "Enter output image filename (e.g. output.jpg): ";
    std::getline(std::cin, outputFile);

    std::string inputPath = INPUT_DIR + inputFile;
    std::string outputPath = OUTPUT_DIR + outputFile;

    std::cout << "\nChoose operation:\n"
        << "1. Rotate (any degrees)\n"
        << "2. Resize (custom size)\n"
        << "3. Invert filter (CUDA)\n"
        << "4. Gaussian blur (CUDA)\n"
        << "Choice: ";
    std::cin >> operation;

    cv::Mat img = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error loading image from: " << inputPath << "\n";
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    if (operation == 1) {
        double angle;
        std::cout << "Enter rotation angle (in degrees): ";
        std::cin >> angle;

        cv::Point2f center(img.cols / 2.0f, img.rows / 2.0f);
        cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
        cv::warpAffine(img, img, rot, img.size());
    }
    else if (operation == 2) {
        int new_width, new_height;
        std::cout << "Enter new width: ";
        std::cin >> new_width;
        std::cout << "Enter new height: ";
        std::cin >> new_height;

        if (new_width <= 0 || new_height <= 0) {
            std::cerr << "Invalid dimensions.\n";
            return -1;
        }

        cv::resize(img, img, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);
    }
    else {
        int width = img.cols;
        int height = img.rows;
        int channels = img.channels();
        size_t img_size = width * height * channels;

        unsigned char* d_input;
        unsigned char* d_output = nullptr;

        cudaMalloc((void**)&d_input, img_size);
        cudaMemcpy(d_input, img.data, img_size, cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);

        if (operation == 3) {
            invert_filter << <grid, block >> > (d_input, width, height, channels);
            cudaMemcpy(img.data, d_input, img_size, cudaMemcpyDeviceToHost);
        }
        else if (operation == 4) {
            cudaMalloc((void**)&d_output, img_size);
            gaussian_blur << <grid, block >> > (d_input, d_output, width, height, channels);
            cudaMemcpy(img.data, d_output, img_size, cudaMemcpyDeviceToHost);
            cudaFree(d_output);
        }
        else {
            std::cerr << "Invalid operation.\n";
            return -1;
        }

        cudaFree(d_input);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    if (!cv::imwrite(outputPath, img)) {
        std::cerr << "Error saving output image to: " << outputPath << "\n";
        return -1;
    }

    std::cout << "Image processed and saved to " << outputPath << "\n";
    std::cout << "Processing time: " << elapsed << " ms\n";

    return 0;
}
