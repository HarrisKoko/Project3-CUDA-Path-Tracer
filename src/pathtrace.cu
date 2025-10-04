// pathtrace.cu

#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define samplesPerPixel 16

// Performance optimization toggles
#define SORT_MATERIAL_ID 0
#define STREAM_COMPACTION 1

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
#endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

// Kernels

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

        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

// Global device memory pointers
static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;

// Mesh data on device
static glm::vec3* dev_vertices = nullptr;
static uint32_t* dev_indices = nullptr;
static glm::vec3* dev_normals = nullptr;
static int dev_index_count = 0;

// BVH acceleration structure on device
static BVHNode* dev_bvh = nullptr;
static glm::uvec3* dev_triTriplets = nullptr;
static int dev_triCount = 0;

// Functors for thrust operations

struct isRayAlive {
    __host__ __device__
        bool operator()(const PathSegment& path) const {
        return path.remainingBounces > 0;
    }
};

struct materialsCmp {
    __host__ __device__
        bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const {
        return a.materialId < b.materialId;
    }
};

// Initialization

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

    cudaMalloc(&dev_paths, pixelcount * samplesPerPixel * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    if (!scene->vertices.empty()) {
        cudaMalloc(&dev_vertices, scene->vertices.size() * sizeof(glm::vec3));
        cudaMemcpy(dev_vertices, scene->vertices.data(),
            scene->vertices.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    }
    if (!scene->indices.empty()) {
        cudaMalloc(&dev_indices, scene->indices.size() * sizeof(uint32_t));
        cudaMemcpy(dev_indices, scene->indices.data(),
            scene->indices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        dev_index_count = static_cast<int>(scene->indices.size());
    }
    if (!scene->normals.empty()) {
        cudaMalloc(&dev_normals, scene->normals.size() * sizeof(glm::vec3));
        cudaMemcpy(dev_normals, scene->normals.data(),
            scene->normals.size() * sizeof(glm::vec3),
            cudaMemcpyHostToDevice);
    }

    if (!scene->bvhNodes.empty()) {
        cudaMalloc(&dev_bvh, scene->bvhNodes.size() * sizeof(BVHNode));
        cudaMemcpy(dev_bvh, scene->bvhNodes.data(),
            scene->bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
    }
    if (!scene->triIndexTriplets.empty()) {
        cudaMalloc(&dev_triTriplets, scene->triIndexTriplets.size() * sizeof(glm::uvec3));
        cudaMemcpy(dev_triTriplets, scene->triIndexTriplets.data(),
            scene->triIndexTriplets.size() * sizeof(glm::uvec3), cudaMemcpyHostToDevice);
        dev_triCount = (int)scene->triIndexTriplets.size();
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);

    cudaFree(dev_vertices);
    cudaFree(dev_normals);
    cudaFree(dev_indices);

    cudaFree(dev_bvh);
    cudaFree(dev_triTriplets);

    checkCUDAError("pathtraceFree");
}

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

// Compute ray-scene intersections
__global__ void computeIntersections(
    int currentDepth,
    int numActivePaths,
    PathSegment* pathSegments,
    Geom* sceneGeometry,
    int geometryCount,
    const glm::vec3* meshVertices,
    const uint32_t* meshIndices,
    int totalIndexCount,
    const glm::vec3* meshNormals,
    const BVHNode* bvhTree,
    const glm::uvec3* triangleIndices,
    ShadeableIntersection* intersectionResults)
{
    int pathIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (pathIndex >= numActivePaths)
        return;

    const PathSegment currentPath = pathSegments[pathIndex];
    const Ray worldRay = currentPath.ray;

    // Track closest hit
    float closestDistance = FLT_MAX;
    int hitGeometryIndex = -1;
    glm::vec3 hitNormalWorld(0.0f);

    // Test ray against each geometry object
    for (int i = 0; i < geometryCount; ++i)
    {
        const Geom& geom = sceneGeometry[i];

        float hitDistance = -1.0f;
        glm::vec3 hitPointObject, normalObject;
        glm::vec3 normalWorld;
        bool isOutside = true;

        if (geom.type == CUBE)
        {
            hitDistance = boxIntersectionTest(
                const_cast<Geom&>(geom),
                worldRay,
                hitPointObject,
                normalWorld,
                isOutside
            );

            if (hitDistance > 0.0f && hitDistance < closestDistance)
            {
                closestDistance = hitDistance;
                hitGeometryIndex = i;
                hitNormalWorld = normalWorld;
            }
        }
        else if (geom.type == SPHERE)
        {
            hitDistance = sphereIntersectionTest(
                const_cast<Geom&>(geom),
                worldRay,
                hitPointObject,
                normalWorld,
                isOutside
            );

            if (hitDistance > 0.0f && hitDistance < closestDistance)
            {
                closestDistance = hitDistance;
                hitGeometryIndex = i;
                hitNormalWorld = normalWorld;
            }
        }
        else if (geom.type == MESH)
        {
            // Transform ray to object space
            Ray objectRay;
            objectRay.origin = glm::vec3(
                geom.inverseTransform * glm::vec4(worldRay.origin, 1.0f)
            );
            objectRay.direction = glm::normalize(
                glm::vec3(geom.inverseTransform * glm::vec4(worldRay.direction, 0.0f))
            );

            float closestTriDistance = FLT_MAX;
            glm::vec3 closestTriNormal(0.0f);
            bool hitMesh = false;

            // Traverse BVH using a stack
            int stack[64];
            int stackPointer = 0;
            stack[stackPointer++] = 0; // Start at root

            while (stackPointer > 0)
            {
                int nodeIndex = stack[--stackPointer];
                BVHNode node = bvhTree[nodeIndex];

                // Test against bounding box
                float tEntry, tExit;
                bool hitsBox = intersectAABB(objectRay, node.box, tEntry, tExit);

                if (!hitsBox || tEntry > closestTriDistance)
                    continue;

                if (node.triCount > 0)
                {
                    // Leaf node: test triangles
                    for (int k = 0; k < node.triCount; ++k)
                    {
                        const glm::uvec3 tri = triangleIndices[node.firstTri + k];
                        const glm::vec3& v0 = meshVertices[tri.x];
                        const glm::vec3& v1 = meshVertices[tri.y];
                        const glm::vec3& v2 = meshVertices[tri.z];

                        float triDistance;
                        float baryU, baryV;
                        glm::vec3 faceNormal;

                        if (intersectTriangleBarycentric(objectRay, v0, v1, v2, triDistance, baryU, baryV, faceNormal))
                        {
                            if (triDistance > 0.0f && triDistance < closestTriDistance)
                            {
                                closestTriDistance = triDistance;
                                hitMesh = true;

                                // Interpolate smooth normal using barycentric coordinates
                                if (meshNormals)
                                {
                                    float baryW = 1.0f - baryU - baryV;
                                    closestTriNormal = glm::normalize(
                                        baryW * meshNormals[tri.x] +
                                        baryU * meshNormals[tri.y] +
                                        baryV * meshNormals[tri.z]
                                    );
                                }
                                else
                                {
                                    closestTriNormal = glm::normalize(faceNormal);
                                }
                            }
                        }
                    }
                }
                else
                {
                    // Internal node: add children to stack
                    if (node.right >= 0) stack[stackPointer++] = node.right;
                    if (node.left >= 0) stack[stackPointer++] = node.left;
                }
            }

            // Convert mesh hit to world space and compare with other geometry
            if (hitMesh)
            {
                const glm::vec3 hitPointObject = objectRay.origin + closestTriDistance * objectRay.direction;
                const glm::vec3 hitPointWorld = glm::vec3(geom.transform * glm::vec4(hitPointObject, 1.0f));

                // Calculate world space distance along ray
                const float worldDistance = glm::dot(hitPointWorld - worldRay.origin, glm::normalize(worldRay.direction));

                if (worldDistance > 0.0f && worldDistance < closestDistance)
                {
                    closestDistance = worldDistance;
                    hitGeometryIndex = i;
                    hitNormalWorld = glm::normalize(glm::vec3(geom.invTranspose * glm::vec4(closestTriNormal, 0.0f)));
                }
            }
        }
    }

    // Store results
    if (hitGeometryIndex < 0)
    {
        intersectionResults[pathIndex].t = -1.0f;
    }
    else
    {
        intersectionResults[pathIndex].t = closestDistance;
        intersectionResults[pathIndex].materialId = sceneGeometry[hitGeometryIndex].materialid;
        intersectionResults[pathIndex].surfaceNormal = hitNormalWorld;
    }
}

// Shade materials and scatter rays
__global__ void shadeMaterial(
    int currentIteration,
    int numActivePaths,
    ShadeableIntersection* intersections,
    PathSegment* pathSegments,
    Material* materials,
    int bounceCount)
{
    int pathIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (pathIndex >= numActivePaths || pathSegments[pathIndex].remainingBounces <= 0)
        return;

    ShadeableIntersection intersection = intersections[pathIndex];

    if (intersection.t > 0.0f)
    {
        // Valid hit: scatter ray based on material
        thrust::default_random_engine rng = makeSeededRandomEngine(currentIteration, pathIndex, bounceCount);
        Material material = materials[intersection.materialId];

        glm::vec3 hitPoint = pathSegments[pathIndex].ray.origin +
            intersection.t * pathSegments[pathIndex].ray.direction;

        scatterRay(pathSegments[pathIndex], hitPoint, intersection.surfaceNormal, material, rng);
    }
    else
    {
        // No hit: terminate ray
        pathSegments[pathIndex].color = glm::vec3(0.0f);
        pathSegments[pathIndex].remainingBounces = 0;
    }
}

// Accumulate path colors into final image
__global__ void finalGather(int numPaths, glm::vec3* image, PathSegment* paths)
{
    int pathIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (pathIndex < numPaths)
    {
        PathSegment path = paths[pathIndex];
        image[path.pixelIndex] += path.color;
    }
}

// Main path tracing loop
void pathtrace(uchar4* pbo, int frame, int currentIteration)
{
    const int maxBounces = hst_scene->state.traceDepth;
    const Camera& camera = hst_scene->state.camera;
    const int totalPixels = camera.resolution.x * camera.resolution.y;

    const dim3 blockSize2D(8, 8);
    const dim3 numBlocks2D(
        (camera.resolution.x + blockSize2D.x - 1) / blockSize2D.x,
        (camera.resolution.y + blockSize2D.y - 1) / blockSize2D.y
    );

    const int blockSize1D = 128;

    // Generate camera rays
    generateRayFromCamera << <numBlocks2D, blockSize2D >> > (
        camera, currentIteration, maxBounces, dev_paths, samplesPerPixel
        );
    checkCUDAError("generate camera ray");

    int currentDepth = 0;
    PathSegment* pathsEnd = dev_paths + totalPixels;
    int numActivePaths = pathsEnd - dev_paths;

    bool tracingComplete = false;
    int bounceCount = 0;

    while (!tracingComplete)
    {
        // Clear intersection data
        cudaMemset(dev_intersections, 0, totalPixels * sizeof(ShadeableIntersection));

        dim3 numBlocks1D = (numActivePaths + blockSize1D - 1) / blockSize1D;

        // Find intersections
        computeIntersections << <numBlocks1D, blockSize1D >> > (
            currentDepth,
            numActivePaths,
            dev_paths,
            dev_geoms,
            (int)hst_scene->geoms.size(),
            dev_vertices,
            dev_indices,
            dev_index_count,
            dev_normals,
            dev_bvh,
            dev_triTriplets,
            dev_intersections
            );

        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        currentDepth++;

#if SORT_MATERIAL_ID
        // Sort rays by material for better coherence
        thrust::sort_by_key(
            thrust::device,
            dev_intersections,
            dev_intersections + numActivePaths,
            dev_paths,
            materialsCmp()
        );
#endif

        // Shade materials and scatter rays
        shadeMaterial << <numBlocks1D, blockSize1D >> > (
            currentIteration,
            numActivePaths,
            dev_intersections,
            dev_paths,
            dev_materials,
            bounceCount
            );
        checkCUDAError("shade");

#if STREAM_COMPACTION
        // Remove terminated rays
        numActivePaths = thrust::partition(
            thrust::device,
            dev_paths,
            dev_paths + numActivePaths,
            isRayAlive()
        ) - dev_paths;
#endif

        // Check termination conditions
        if (currentDepth >= maxBounces || numActivePaths == 0)
        {
            tracingComplete = true;
        }
        bounceCount++;

        if (guiData != NULL)
        {
            guiData->TracedDepth = currentDepth;
        }
    }

#if STREAM_COMPACTION
    numActivePaths = (int)(pathsEnd - dev_paths);
#endif

    // Accumulate path results into image
    dim3 numBlocksForGather = (numActivePaths + blockSize1D - 1) / blockSize1D;
    finalGather << <numBlocksForGather, blockSize1D >> > (numActivePaths, dev_image, dev_paths);
    checkCUDAError("finalGather");

    // Send image to display
    sendImageToPBO << <numBlocks2D, blockSize2D >> > (pbo, camera.resolution, currentIteration, dev_image);

    // Copy image back to host
    cudaMemcpy(
        hst_scene->state.image.data(),
        dev_image,
        totalPixels * sizeof(glm::vec3),
        cudaMemcpyDeviceToHost
    );

    checkCUDAError("pathtrace");
}