import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from PIL import Image
from google.colab import files

grayscale_kernel_code = """
__global__ void rgb_to_grayscale_with_shared(unsigned char *input, unsigned char *output, int width, int height, float R_WEIGHT, float G_WEIGHT, float B_WEIGHT) {
    extern __shared__ unsigned char shared_data[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    int x = block_x * blockDim.x + tx;
    int y = block_y * blockDim.y + ty;

    int shared_width = blockDim.x + 2;
    int shared_height = blockDim.y + 2;

    int global_idx = (y * width + x) * 3;

    if (x < width && y < height) {
        shared_data[(ty * shared_width + tx) * 3 + 0] = input[global_idx + 0];
        shared_data[(ty * shared_width + tx) * 3 + 1] = input[global_idx + 1];
        shared_data[(ty * shared_width + tx) * 3 + 2] = input[global_idx + 2];
    }

    __syncthreads();

    if (x < width && y < height) {
        unsigned char r = shared_data[(ty * shared_width + tx) * 3 + 0];
        unsigned char g = shared_data[(ty * shared_width + tx) * 3 + 1];
        unsigned char b = shared_data[(ty * shared_width + tx) * 3 + 2];

        unsigned char gray = (unsigned char)(R_WEIGHT * r + G_WEIGHT * g + B_WEIGHT * b);
        int output_idx = y * width + x;
        output[output_idx] = gray;
    }
}
"""

brightness_kernel_code = """
__global__ void adjust_brightness(unsigned char *input, unsigned char *output, int width, int height, float average, float scale_factor) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    int x = block_x * blockDim.x + tx;
    int y = block_y * blockDim.y + ty;

    __shared__ unsigned char shared_input[%(block_size)s * %(block_size)s * 3];

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        shared_input[(ty * blockDim.x + tx) * 3 + 0] = input[idx + 0];
        shared_input[(ty * blockDim.x + tx) * 3 + 1] = input[idx + 1];
        shared_input[(ty * blockDim.x + tx) * 3 + 2] = input[idx + 2];
    }

    __syncthreads();

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        for (int c = 0; c < 3; c++) {
            float value = shared_input[(ty * blockDim.x + tx) * 3 + c];
            float adjusted = average + (value - average) * scale_factor;
            output[idx + c] = min(max((int)adjusted, 0), 255);
        }
    }
}
"""

gaussian_kernel_code = """
__global__ void gaussian_blur(float *input, float *output, float *kernel, int width, int height, int channels, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int color = threadIdx.z;

    int half_kernel = kernel_size / 2;
    
    if(x > width || y > height || color > channels) return;

    float sum = 0.0;
    for (int i = -half_kernel; i <= half_kernel; i++) {
        for (int j = -half_kernel; j <= half_kernel; j++) {
            int img_x = min(max(x + i, 0), width - 1);
            int img_y = min(max(y + j, 0), height - 1);
            int idx = (img_y * width + img_x) * channels + color;
            int kernel_idx = (i + half_kernel) * kernel_size + (j + half_kernel);
            sum += input[idx] * kernel[kernel_idx];
        }
    }
    int out_idx = (y * width + x) * channels + color;
    output[out_idx] = sum;
}
"""

def load_image(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")
    return np.array(img)

def convert_to_grayscale(image_path, block_size=(16, 16)):
    img = load_image(image_path)
    height, width, _ = img.shape
    img = img.astype(np.uint8)

    input_mem = cuda.mem_alloc(img.nbytes)
    output_mem = cuda.mem_alloc(width * height)

    cuda.memcpy_htod(input_mem, img)

    mod = SourceModule(grayscale_kernel_code)
    func = mod.get_function("rgb_to_grayscale_with_shared")

    R_WEIGHT = np.float32(0.299)
    G_WEIGHT = np.float32(0.587)
    B_WEIGHT = np.float32(0.114)

    grid_dim = (int(np.ceil(width / block_size[0])), int(np.ceil(height / block_size[1])))
    block_dim = (block_size[0], block_size[1], 1)
    shared_memory_size = (block_size[0] + 2) * (block_size[1] + 2) * 3

    func(input_mem, output_mem, np.int32(width), np.int32(height), R_WEIGHT, G_WEIGHT, B_WEIGHT,
         block=block_dim, grid=grid_dim, shared=shared_memory_size)

    result = np.empty((height, width), dtype=np.uint8)
    cuda.memcpy_dtoh(result, output_mem)

    return result

def adjust_brightness(image_path, scale_factor, block_size=(16, 16)):
    img = load_image(image_path)
    height, width, _ = img.shape
    img = img.astype(np.uint8)

    average_intensity = np.mean(img)

    input_mem = cuda.mem_alloc(img.nbytes)
    output_mem = cuda.mem_alloc(img.nbytes)

    cuda.memcpy_htod(input_mem, img)

    mod = SourceModule(brightness_kernel_code % {'block_size': block_size[0]})
    func = mod.get_function("adjust_brightness")

    grid_dim = (int(np.ceil(width / block_size[0])), int(np.ceil(height / block_size[1])))
    block_dim = (block_size[0], block_size[1], 1)

    func(input_mem, output_mem, np.int32(width), np.int32(height), np.float32(average_intensity), np.float32(scale_factor),
         block=block_dim, grid=grid_dim)

    result = np.empty_like(img, dtype=np.uint8)
    cuda.memcpy_dtoh(result, output_mem)

    return result

def apply_gaussian_blur(image, kernel_size=3, sigma=1.0):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    total = 0.0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - kernel_size // 2, j - kernel_size // 2
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            total += kernel[i, j]
    kernel = kernel / total

    height, width, channels = image.shape

    input_mem = cuda.mem_alloc(image.nbytes)
    output_mem = cuda.mem_alloc(image.nbytes)
    kernel_mem = cuda.mem_alloc(kernel.nbytes)

    cuda.memcpy_htod(input_mem, image.astype(np.float32))
    cuda.memcpy_htod(kernel_mem, kernel.astype(np.float32))

    threads_per_block = (16, 16, channels)
    blocks_per_grid = ((width + threads_per_block[0]) // threads_per_block[0],
                        (height + threads_per_block[1]) // threads_per_block[1], 1)

    result = np.empty((height, width, channels), dtype=np.float32)

    mod = SourceModule(gaussian_kernel_code)
    gaussian_blur = mod.get_function("gaussian_blur")

    gaussian_blur(input_mem, output_mem, kernel_mem, np.int32(width), np.int32(height), np.int32(channels), np.int32(kernel_size),
                  block=threads_per_block, grid=blocks_per_grid)

    cuda.memcpy_dtoh(result, output_mem)

    return result.astype(np.uint8)

uploaded = files.upload()
image_path = next(iter(uploaded))

grayscale_img = convert_to_grayscale(image_path)
grayscale_img = Image.fromarray(grayscale_img)
grayscale_img.save('grayscale_output.jpg')
files.download('grayscale_output.jpg')

brightness_img = adjust_brightness(image_path, scale_factor=5)
brightness_img = Image.fromarray(brightness_img)
brightness_img.save('brightness_adjusted_output.jpg')
files.download('brightness_adjusted_output.jpg')

image = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
blurred_img = apply_gaussian_blur(image, kernel_size=9, sigma=2.5)
blurred_img = Image.fromarray(blurred_img)
blurred_img.save('blurred_output.jpg')
files.download('blurred_output.jpg')
