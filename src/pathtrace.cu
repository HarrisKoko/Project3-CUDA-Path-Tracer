#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define samplesPerPixel 16

// Optimization Toggles
#define STREAM_COMPACTION 0
#define MATERIAL_SORTING 0

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

// Functor for stream compaction
struct isRayAlive {
    __host__ __device__
        bool operator()(const PathSegment& path) const {
        return path.remainingBounces > 0;
    }
};

// Comparator for sorting intersections by materialId
struct materialsCmp {
    __host__ __device__
        bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const {
        return a.materialId < b.materialId;
    }
};

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    // We only use the first pixelcount slots, but extra capacity is fine.
    cudaMalloc(&dev_paths, pixelcount * samplesPerPixel * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, int numSamples)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= cam.resolution.x || y >= cam.resolution.y) return;

    int index = x + y * cam.resolution.x;
    PathSegment& segment = pathSegments[index];

    segment.ray.origin = cam.position;
    segment.color = glm::vec3(1.0f);
    segment.pixelIndex = index;
    segment.remainingBounces = traceDepth;

    thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

    glm::vec3 accumulatedDir(0.0f);

    for (int s = 0; s < numSamples; ++s) {
        float jitterX = u01(rng) - 0.5f;
        float jitterY = u01(rng) - 0.5f;

        float px = (float)x + jitterX;
        float py = (float)y + jitterY;

        glm::vec3 rayDir = glm::normalize(
            cam.view
            - cam.right * cam.pixelLength.x * (px - cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (py - cam.resolution.y * 0.5f)
        );

        accumulatedDir += rayDir;
    }

    segment.ray.direction = glm::normalize(accumulatedDir / float(numSamples));
}

// Intersections only (no shading)
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms
        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // add more primitives as needed

            // track closest hit
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    int bounces)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths || pathSegments[idx].remainingBounces <= 0) return;

    ShadeableIntersection intersection = shadeableIntersections[idx];

    if (intersection.t > 0.0f) { // hit something
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, bounces);
        Material material = materials[intersection.materialId];

        glm::vec3 hitPoint = pathSegments[idx].ray.origin +
            intersection.t * pathSegments[idx].ray.direction;

        scatterRay(pathSegments[idx], hitPoint, intersection.surfaceNormal, material, rng);
    }
    else { // miss
        pathSegments[idx].color = glm::vec3(0.0f);
        pathSegments[idx].remainingBounces = 0;
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D for primary rays
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    const int blockSize1d = 128;

    // Generate primary rays
    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths, samplesPerPixel);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount; 
    int num_paths = (int)(dev_path_end - dev_paths);    

    bool iterationComplete = false;
    int bounces = 0;
    bool firstBounce = true;

    while (!iterationComplete)
    {
        // Do stream compaction before material sorting to decrease overhead of material sorting
#if STREAM_COMPACTION
        if (!firstBounce) {
            num_paths = thrust::partition(
                thrust::device,
                dev_paths,
                dev_paths + num_paths,
                isRayAlive()) - dev_paths;
        }
#endif

        if (num_paths <= 0) {
            iterationComplete = true;
            break;
        }

        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            (int)hst_scene->geoms.size(),
            dev_intersections
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        // Sort rays by material 
#if MATERIAL_SORTING
        thrust::sort_by_key(
            thrust::device,
            dev_intersections,
            dev_intersections + num_paths,
            dev_paths,
            materialsCmp());
#endif

        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            bounces
            );
        checkCUDAError("shade");

        depth++;
        bounces++;
        firstBounce = false;

        if (depth >= traceDepth) {
            iterationComplete = true;
        }

        if (guiData != NULL) {
            guiData->TracedDepth = depth;
        }
    }

#if STREAM_COMPACTION
    num_paths = (int)(dev_path_end - dev_paths); 
#endif

    dim3 numBlocksPixels = (num_paths + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);
    checkCUDAError("finalGather");

    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
