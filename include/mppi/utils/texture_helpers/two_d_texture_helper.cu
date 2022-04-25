//
// Created by jason on 1/5/22.
//

#include "two_d_texture_helper.cuh"

template <class DATA_T>
void TwoDTextureHelper<DATA_T>::allocateCudaTexture(int index)
{
  TextureHelper<TwoDTextureHelper<DATA_T>, DATA_T>::allocateCudaTexture(index);

  TextureParams<DATA_T>* param = &this->textures_[index];

  int w = param->extent.width;
  int h = param->extent.height;

  HANDLE_ERROR(cudaMallocArray(&(param->array_d), &(param->channelDesc), w, h));
}

template <class DATA_T>
void TwoDTextureHelper<DATA_T>::updateTexture(const int index, std::vector<DATA_T>& values, bool column_major)
{
  TextureParams<DATA_T>* param = &this->textures_buffer_[index];
  int w = param->extent.width;
  int h = param->extent.height;

  // check that the sizes are correct
  if (values.size() != w * h)
  {
    throw std::runtime_error(std::string("Error: invalid size to updateTexture ") + std::to_string(values.size()) +
                             " != " + std::to_string(w * h));
  }

  // copy over values to cpu side holder
  // TODO handle row/column major data
  this->cpu_buffer_values_[index].resize(w * h);
  if (column_major)
  {
    for (int j = 0; j < w; j++)
    {
      for (int i = 0; i < h; i++)
      {
        int columnMajorIndex = j * h + i;
        int rowMajorIndex = i * w + j;
        this->cpu_buffer_values_[index][rowMajorIndex] = values[columnMajorIndex];
      }
    }
  }
  else
  {
    // std::copy(values.begin(), values.end(), cpu_buffer_values_[index].begin());
    this->cpu_buffer_values_[index] = std::move(values);
  }
  // tells the object to copy it over next time that happens
  param->update_data = true;
}

template <class DATA_T>
void TwoDTextureHelper<DATA_T>::updateTexture(const int index, std::vector<DATA_T>& data, cudaExtent& extent,
                                              bool column_major)
{
  setExtent(index, extent);
  updateTexture(index, data, column_major);
}

template <class DATA_T>
__host__ __device__ DATA_T TwoDTextureHelper<DATA_T>::queryTexture(const int index, const float3& point)
{
#ifdef __CUDA_ARCH__
  return tex2D<DATA_T>(this->textures_d_[index].tex_d, point.x, point.y);
#else
  std::cout << "\n\ninput point " << point.x << ", " << point.y << std::endl;
  TextureParams<DATA_T>* param = &this->textures_[index];

  // convert normalized to array index
  float2 query = make_float2(point.x * param->extent.width, point.y * param->extent.height);
  std::cout << "after converting to pixels at " << query.x << ", " << query.y << std::endl;
  // correctly handle being above the location of the top index
  if (query.x < param->extent.width)
  {
    query.x = std::max(query.x - 0.5, 0.0);
  }
  if (query.y < param->extent.height)
  {
    query.y = std::max(query.y - 0.5, 0.0);
  }
  if (param->texDesc.addressMode[0] == cudaAddressModeClamp)
  {
    if (query.x > param->extent.width - 1)
    {
      query.x = param->extent.width - 1;
    }
    else if (query.x <= 0.0)
    {
      query.x = 0.0;
    }
  }
  else
  {
    throw std::runtime_error(std::string("using unsupported address mode on the CPU in texture utils"));
  }
  if (param->texDesc.addressMode[1] == cudaAddressModeClamp)
  {
    if (query.y > param->extent.height - 1)
    {
      query.y = param->extent.height - 1;
    }
    else if (query.y <= 0.0)
    {
      query.y = 0.0;
    }
  }
  else
  {
    throw std::runtime_error(std::string("using unsupported address mode on the CPU in texture utils"));
  }
  std::cout << "query " << query.x << ", " << query.y << std::endl;
  int w = param->extent.width;
  if (param->texDesc.filterMode == cudaFilterModeLinear)
  {
    // the value is distributed evenly in the space starting at half a cell from 0.0
    int2 x_min_y_min = make_int2(std::floor(query.x), std::floor(query.y));
    int2 x_max_y_min = make_int2(std::ceil(query.x), std::floor(query.y));
    int2 x_min_y_max = make_int2(std::floor(query.x), std::ceil(query.y));
    int2 x_max_y_max = make_int2(std::ceil(query.x), std::ceil(query.y));

    std::cout << "got  indexes\n";
    std::cout << "x_min_y_min: " << x_min_y_min.x << ", " << x_min_y_min.y << std::endl;
    std::cout << "x_max_y_min: " << x_max_y_min.x << ", " << x_max_y_min.y << std::endl;
    std::cout << "x_min_y_max: " << x_min_y_max.x << ", " << x_min_y_max.y << std::endl;
    std::cout << "x_max_y_max: " << x_max_y_max.x << ", " << x_max_y_max.y << std::endl;

    DATA_T x_min_y_min_val = this->cpu_values_[index][x_min_y_min.y * w + x_min_y_min.x];
    DATA_T x_max_y_min_val = this->cpu_values_[index][x_max_y_min.y * w + x_max_y_min.x];
    DATA_T x_min_y_max_val = this->cpu_values_[index][x_min_y_max.y * w + x_min_y_max.x];
    DATA_T x_max_y_max_val = this->cpu_values_[index][x_max_y_max.y * w + x_max_y_max.x];

    std::cout << "got data\n";
    std::cout << "x_min_y_min: " << x_min_y_min_val.x << ", " << x_min_y_min_val.y << std::endl;
    std::cout << "x_max_y_min: " << x_max_y_min_val.x << ", " << x_max_y_min_val.y << std::endl;
    std::cout << "x_min_y_max: " << x_min_y_max_val.x << ", " << x_min_y_max_val.y << std::endl;
    std::cout << "x_max_y_max: " << x_max_y_max_val.x << ", " << x_max_y_max_val.y << std::endl;

    // computes distance from query to each point
    float x_min_y_min_dist = 2 * sqrtf(2) - sqrtf(powf(query.x - x_min_y_min.x, 2) + powf(query.y - x_min_y_min.y, 2));
    float x_max_y_min_dist = 2 * sqrtf(2) - sqrtf(powf(query.x - x_max_y_min.x, 2) + powf(query.y - x_max_y_min.y, 2));
    float x_min_y_max_dist = 2 * sqrtf(2) - sqrtf(powf(query.x - x_min_y_max.x, 2) + powf(query.y - x_min_y_max.y, 2));
    float x_max_y_max_dist = 2 * sqrtf(2) - sqrtf(powf(query.x - x_max_y_max.x, 2) + powf(query.y - x_max_y_max.y, 2));
    std::cout << "got distance\n";
    std::cout << "x_min_y_min: " << x_min_y_min_dist << std::endl;
    std::cout << "x_max_y_min: " << x_max_y_min_dist << std::endl;
    std::cout << "x_min_y_max: " << x_min_y_max_dist << std::endl;
    std::cout << "x_max_y_max: " << x_max_y_max_dist << std::endl;

    float total_dist = x_min_y_min_dist + x_max_y_min_dist + x_min_y_max_dist + x_max_y_max_dist;
    std::cout << "total dist " << total_dist << std::endl;

    if (total_dist <= 0)
    {
      x_min_y_min_dist = 1.0;
    }
    else
    {
      x_min_y_min_dist /= total_dist;
      x_max_y_min_dist /= total_dist;
      x_min_y_max_dist /= total_dist;
      x_max_y_max_dist /= total_dist;
    }

    std::cout << "got distance\n";
    std::cout << "x_min_y_min: " << x_min_y_min_dist << std::endl;
    std::cout << "x_max_y_min: " << x_max_y_min_dist << std::endl;
    std::cout << "x_min_y_max: " << x_min_y_max_dist << std::endl;
    std::cout << "x_max_y_max: " << x_max_y_max_dist << std::endl;

    float4 x_min_y_min_final = x_min_y_min_val * x_min_y_min_dist;
    float4 x_max_y_min_final = x_max_y_min_val * x_max_y_min_dist;
    float4 x_min_y_max_final = x_min_y_max_val * x_min_y_max_dist;
    float4 x_max_y_max_final = x_max_y_max_val * x_max_y_max_dist;

    std::cout << x_min_y_min_final.x << " + " << x_min_y_min_final.y << " + " << x_min_y_min_final.z << " + "
              << x_min_y_min_final.w << std::endl;
    std::cout << x_max_y_min_final.x << " + " << x_max_y_min_final.y << " + " << x_max_y_min_final.z << " + "
              << x_max_y_min_final.w << std::endl;
    std::cout << x_min_y_max_final.x << " + " << x_min_y_max_final.y << " + " << x_min_y_max_final.z << " + "
              << x_min_y_max_final.w << std::endl;
    std::cout << x_max_y_max_final.x << " + " << x_max_y_max_final.y << " + " << x_max_y_max_final.z << " + "
              << x_max_y_max_final.w << std::endl;

    float4 result = x_min_y_min_final + x_max_y_min_final + x_min_y_max_final + x_max_y_max_final;

    std::cout << "result: " << result.x << ", " << result.y << ", " << result.z << ", " << result.w << "\n\n"
              << std::endl;

    // does the actual interpolation
    return result;
  }
  else if (param->texDesc.filterMode == cudaFilterModePoint)
  {
    int rowMajorIndex = std::round(query.y) * w + std::round(query.x);
    return this->cpu_values_[index][rowMajorIndex];
  }
#endif
}

template <class DATA_T>
bool TwoDTextureHelper<DATA_T>::setExtent(int index, cudaExtent& extent)
{
  if (extent.depth != 0)
  {
    throw std::runtime_error(std::string("Error: extent in setExtent invalid,"
                                         " cannot use depth != 0 in 2D texture: using ") +
                             std::to_string(extent.depth));
  }

  return TextureHelper<TwoDTextureHelper<DATA_T>, DATA_T>::setExtent(index, extent);
}

template <class DATA_T>
void TwoDTextureHelper<DATA_T>::copyDataToGPU(int index, bool sync)
{
  TextureParams<DATA_T>* param = &this->textures_[index];
  int w = param->extent.width;
  int h = param->extent.height;
  HANDLE_ERROR(cudaMemcpy2DToArrayAsync(param->array_d, 0, 0, this->cpu_values_[index].data(), w * sizeof(DATA_T),
                                        w * sizeof(DATA_T), h, cudaMemcpyHostToDevice, this->stream_));
  if (sync)
  {
    cudaStreamSynchronize(this->stream_);
  }
  param->update_data = false;
}

template <class DATA_T>
void TwoDTextureHelper<DATA_T>::updateTexture(
    const int index,
    const Eigen::Ref<const Eigen::Matrix<DATA_T, Eigen::Dynamic, Eigen::Dynamic>, 0,
                     Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        values,
    bool column_major)
{
  TextureParams<DATA_T>* param = &this->textures_buffer_[index];
  int w = param->extent.width;
  int h = param->extent.height;
  this->cpu_buffer_values_[index].resize(w * h);

  if (column_major)
  {
    // if we are column major transform to row major
    for (int j = 0; j < w; j++)
    {
      for (int i = 0; i < h; i++)
      {
        int columnMajorIndex = j * h + i;
        int rowMajorIndex = i * w + j;
        this->cpu_buffer_values_[index][rowMajorIndex] = values.data()[columnMajorIndex];
      }
    }
  }
  else
  {
    // if we row major copy directly
    memcpy(this->cpu_buffer_values_[index].data(), values.data(), values.size() * sizeof(DATA_T));
  }
  // tells the object to copy it over next time that happens
  param->update_data = true;
}

template <class DATA_T>
void TwoDTextureHelper<DATA_T>::updateTexture(
    int index,
    const Eigen::Ref<const Eigen::Matrix<DATA_T, Eigen::Dynamic, Eigen::Dynamic>, 0,
                     Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        values,
    cudaExtent& extent, bool column_major)
{
  setExtent(index, extent);
  updateTexture(index, values);
}
