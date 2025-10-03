#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include "gltf/tiny_gltf.h"


using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else if (ext == ".gltf") {
        loadFromGLTF(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        if (p.contains("HAS_REFRACTIVE") && p["HAS_REFRACTIVE"] > 0)
        {
            newMaterial.hasRefractive = 1.0f;
            newMaterial.indexOfRefraction = p["IOR"];
        }

        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube") {
            newGeom.type = CUBE;
        }
        else if (type == "sphere") {
            newGeom.type = SPHERE;
        }
        else if (type == "gltf") {
            newGeom.type = MESH;
            std::string filepath = p["FILE"];

            // Get transformation parameters from JSON
            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];
            glm::vec3 translation = glm::vec3(trans[0], trans[1], trans[2]);
            glm::vec3 rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            glm::vec3 scaleVec = glm::vec3(scale[0], scale[1], scale[2]);

            loadFromGLTF(filepath, translation, rotation, scaleVec);

            continue; 
        }


        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    // Build BVH after all objects are loaded
    buildBVH();
}

// Helper functions for BVH construction
namespace
{
    inline AABB mergeBoundingBoxes(const AABB& boxA, const AABB& boxB)
    {
        AABB result;
        result.bmin = glm::min(boxA.bmin, boxB.bmin);
        result.bmax = glm::max(boxA.bmax, boxB.bmax);
        return result;
    }

    inline AABB getTriangleBounds(const glm::vec3& vertexA, const glm::vec3& vertexB, const glm::vec3& vertexC)
    {
        AABB bounds;
        bounds.bmin = glm::min(vertexA, glm::min(vertexB, vertexC));
        bounds.bmax = glm::max(vertexA, glm::max(vertexB, vertexC));
        return bounds;
    }

    inline glm::vec3 getTriangleCentroid(const glm::vec3& vertexA, const glm::vec3& vertexB, const glm::vec3& vertexC)
    {
        return (vertexA + vertexB + vertexC) * (1.0f / 3.0f);
    }
}

void Scene::buildBVH()
{
    if (triIndexTriplets.empty())
    {
        bvhNodes.clear();
        return;
    }

    // Create a permutation array to track triangle ordering during splits
    std::vector<int> triangleOrder(triIndexTriplets.size());
    std::iota(triangleOrder.begin(), triangleOrder.end(), 0);

    // Keep a read-only copy of the original triangle data
    // This prevents us from overwriting data while building the tree
    const std::vector<glm::uvec3> originalTriangles = triIndexTriplets;

    struct BuildTask
    {
        int startIndex;
        int triangleCount;
        int parentNodeIndex;
        bool isLeftChild;
    };

    bvhNodes.clear();
    bvhNodes.reserve(triIndexTriplets.size() * 2); // Reserve space for worst case

    // Create root node
    const int rootIndex = (int)bvhNodes.size();
    bvhNodes.push_back(BVHNode{});

    std::vector<BuildTask> taskStack;
    taskStack.push_back({ 0, (int)triangleOrder.size(), -1, true });

    const int maxTrianglesPerLeaf = 8;

    // Lambda function to compute bounds for a range of triangles
    auto computeRangeBounds = [&](int start, int count, AABB& outBounds, AABB& outCentroidBox)
        {
            AABB triangleBounds, centroidBounds;
            triangleBounds.bmin = glm::vec3(FLT_MAX);
            triangleBounds.bmax = glm::vec3(-FLT_MAX);
            centroidBounds = triangleBounds;

            for (int i = 0; i < count; ++i)
            {
                const glm::uvec3 triangle = originalTriangles[triangleOrder[start + i]];
                const glm::vec3& v0 = vertices[triangle.x];
                const glm::vec3& v1 = vertices[triangle.y];
                const glm::vec3& v2 = vertices[triangle.z];

                // Expand triangle bounds
                AABB triBounds;
                triBounds.bmin = glm::min(v0, glm::min(v1, v2));
                triBounds.bmax = glm::max(v0, glm::max(v1, v2));
                triangleBounds.bmin = glm::min(triangleBounds.bmin, triBounds.bmin);
                triangleBounds.bmax = glm::max(triangleBounds.bmax, triBounds.bmax);

                // Expand centroid bounds
                const glm::vec3 centroid = (v0 + v1 + v2) * (1.0f / 3.0f);
                centroidBounds.bmin = glm::min(centroidBounds.bmin, centroid);
                centroidBounds.bmax = glm::max(centroidBounds.bmax, centroid);
            }

            outBounds = triangleBounds;
            outCentroidBox = centroidBounds;
        };

    // Build BVH using a stack-based approach
    while (!taskStack.empty())
    {
        BuildTask task = taskStack.back();
        taskStack.pop_back();

        AABB nodeBounds, centroidBox;
        computeRangeBounds(task.startIndex, task.triangleCount, nodeBounds, centroidBox);

        // Determine node index
        int currentNodeIndex = (task.parentNodeIndex == -1) ? rootIndex : (int)bvhNodes.size();

        // Link to parent
        if (task.parentNodeIndex != -1)
        {
            if (task.isLeftChild)
                bvhNodes[task.parentNodeIndex].left = currentNodeIndex;
            else
                bvhNodes[task.parentNodeIndex].right = currentNodeIndex;

            bvhNodes.push_back(BVHNode{});
        }

        BVHNode& currentNode = bvhNodes[currentNodeIndex];
        currentNode.box = nodeBounds;
        currentNode.left = -1;
        currentNode.right = -1;
        currentNode.firstTri = -1;
        currentNode.triCount = 0;

        // Create leaf node if triangle count is small enough
        if (task.triangleCount <= maxTrianglesPerLeaf)
        {
            currentNode.firstTri = task.startIndex;
            currentNode.triCount = task.triangleCount;

            // Copy triangles to their final positions
            for (int i = 0; i < task.triangleCount; ++i)
            {
                int destinationIndex = task.startIndex + i;
                int sourceIndex = triangleOrder[destinationIndex];
                triIndexTriplets[destinationIndex] = originalTriangles[sourceIndex];
            }
            continue;
        }

        // Split node - find the widest axis of the centroid bounding box
        glm::vec3 centroidExtent = centroidBox.bmax - centroidBox.bmin;
        int splitAxis = 0; // Default to X axis

        if (centroidExtent.y > centroidExtent.x && centroidExtent.y >= centroidExtent.z)
            splitAxis = 1; // Y axis
        else if (centroidExtent.z > centroidExtent.x && centroidExtent.z >= centroidExtent.y)
            splitAxis = 2; // Z axis

        float splitPosition = 0.5f * (centroidBox.bmin[splitAxis] + centroidBox.bmax[splitAxis]);

        // Partition triangles around the split position
        int leftPointer = task.startIndex;
        int rightPointer = task.startIndex + task.triangleCount - 1;

        while (leftPointer <= rightPointer)
        {
            const glm::uvec3 triangle = originalTriangles[triangleOrder[leftPointer]];
            const glm::vec3 centroid = (vertices[triangle.x] + vertices[triangle.y] + vertices[triangle.z]) * (1.0f / 3.0f);

            if (centroid[splitAxis] < splitPosition)
                ++leftPointer;
            else
                std::swap(triangleOrder[leftPointer], triangleOrder[rightPointer--]);
        }

        int leftChildCount = leftPointer - task.startIndex;

        // Handle degenerate case where all triangles go to one side
        if (leftChildCount == 0 || leftChildCount == task.triangleCount)
            leftChildCount = task.triangleCount / 2;

        // Push child tasks (right first so left is processed next)
        taskStack.push_back({
            task.startIndex + leftChildCount,
            task.triangleCount - leftChildCount,
            currentNodeIndex,
            false
            });
        taskStack.push_back({
            task.startIndex,
            leftChildCount,
            currentNodeIndex,
            true
            });
    }
}

void Scene::loadFromGLTF(const std::string& gltfFilePath,
    const glm::vec3& translation,
    const glm::vec3& rotation,
    const glm::vec3& scale)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string errorMessage, warningMessage;

    bool isBinaryFormat = gltfFilePath.size() >= 4 &&
        gltfFilePath.substr(gltfFilePath.size() - 4) == ".glb";

    bool loadedSuccessfully = isBinaryFormat
        ? loader.LoadBinaryFromFile(&model, &errorMessage, &warningMessage, gltfFilePath)
        : loader.LoadASCIIFromFile(&model, &errorMessage, &warningMessage, gltfFilePath);

    if (!warningMessage.empty())
        std::cout << "GLTF warning: " << warningMessage << "\n";

    if (!errorMessage.empty())
        std::cerr << "GLTF error: " << errorMessage << "\n";

    if (!loadedSuccessfully)
        throw std::runtime_error("Failed to load glTF file: " + gltfFilePath);

    // Process each mesh in the model
    for (const auto& mesh : model.meshes)
    {
        for (const auto& primitive : mesh.primitives)
        {
            // Load vertex positions
            auto positionIterator = primitive.attributes.find("POSITION");
            if (positionIterator == primitive.attributes.end())
                continue; // Skip primitives without positions

            const tinygltf::Accessor& positionAccessor = model.accessors[positionIterator->second];
            const tinygltf::BufferView& positionBufferView = model.bufferViews[positionAccessor.bufferView];
            const tinygltf::Buffer& positionBuffer = model.buffers[positionBufferView.buffer];

            const unsigned char* positionDataStart = positionBuffer.data.data() +
                positionBufferView.byteOffset +
                positionAccessor.byteOffset;
            size_t positionStride = positionAccessor.ByteStride(positionBufferView)
                ? positionAccessor.ByteStride(positionBufferView)
                : sizeof(float) * 3;

            size_t baseVertexIndex = vertices.size();

            for (size_t i = 0; i < positionAccessor.count; ++i)
            {
                const float* position = reinterpret_cast<const float*>(positionDataStart + i * positionStride);
                vertices.emplace_back(position[0], position[1], position[2]);
            }

            // Load vertex normals (if available)
            auto normalIterator = primitive.attributes.find("NORMAL");
            if (normalIterator != primitive.attributes.end())
            {
                const auto& normalAccessor = model.accessors[normalIterator->second];
                const auto& normalBufferView = model.bufferViews[normalAccessor.bufferView];
                const auto& normalBuffer = model.buffers[normalBufferView.buffer];

                const unsigned char* normalDataStart = normalBuffer.data.data() +
                    normalBufferView.byteOffset +
                    normalAccessor.byteOffset;
                size_t normalStride = normalAccessor.ByteStride(normalBufferView)
                    ? normalAccessor.ByteStride(normalBufferView)
                    : sizeof(float) * 3;

                for (size_t i = 0; i < normalAccessor.count; ++i)
                {
                    const float* normal = reinterpret_cast<const float*>(normalDataStart + i * normalStride);
                    normals.emplace_back(glm::normalize(glm::vec3(normal[0], normal[1], normal[2])));
                }
            }
            else
            {
                normals.resize(vertices.size(), glm::vec3(0, 1, 0));
            }

            // Load indices
            size_t baseIndexPosition = indices.size();

            if (primitive.indices >= 0)
            {
                const auto& indexAccessor = model.accessors[primitive.indices];
                const auto& indexBufferView = model.bufferViews[indexAccessor.bufferView];
                const auto& indexBuffer = model.buffers[indexBufferView.buffer];
                const unsigned char* indexDataStart = indexBuffer.data.data() +
                    indexBufferView.byteOffset +
                    indexAccessor.byteOffset;

                indices.resize(baseIndexPosition + indexAccessor.count);

                // Helper to read different index types
                auto readIndex = [&](size_t k) -> uint32_t
                    {
                        switch (indexAccessor.componentType)
                        {
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                            return reinterpret_cast<const uint8_t*>(indexDataStart)[k];
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                            return reinterpret_cast<const uint16_t*>(indexDataStart)[k];
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                            return reinterpret_cast<const uint32_t*>(indexDataStart)[k];
                        default:
                            throw std::runtime_error("Unsupported index component type.");
                        }
                    };

                for (size_t k = 0; k < indexAccessor.count; ++k)
                    indices[baseIndexPosition + k] = readIndex(k) + static_cast<uint32_t>(baseVertexIndex);
            }
            else
            {
                // Generate sequential indices if none provided
                indices.resize(baseIndexPosition + positionAccessor.count);
                for (size_t k = 0; k < positionAccessor.count; ++k)
                    indices[baseIndexPosition + k] = static_cast<uint32_t>(baseVertexIndex + k);
            }

            // Build triangles from indices
            for (size_t k = baseIndexPosition; k + 2 < indices.size(); k += 3)
            {
                triIndexTriplets.emplace_back(indices[k], indices[k + 1], indices[k + 2]);
            }

            // Load material
            Material newMaterial{};

            if (primitive.material >= 0)
            {
                const auto& gltfMaterial = model.materials[primitive.material];
                auto baseColorFactor = gltfMaterial.pbrMetallicRoughness.baseColorFactor;
                newMaterial.color = glm::vec3(baseColorFactor[0], baseColorFactor[1], baseColorFactor[2]);

                // Check for base color texture
                if (gltfMaterial.pbrMetallicRoughness.baseColorTexture.index >= 0)
                {
                    int textureIndex = gltfMaterial.pbrMetallicRoughness.baseColorTexture.index;
                    int imageIndex = model.textures[textureIndex].source;
                    newMaterial.color = glm::vec3(1); // Will be replaced by sampled texture
                    // TODO: Store material-to-texture mapping
                }
            }
            else
            {
                // Default gray material
                newMaterial.color = glm::vec3(0.8f);
            }

            int materialID = materials.size();
            materials.push_back(newMaterial);

            // Create geometry for this primitive with applied transforms
            Geom geometry{};
            geometry.type = MESH;
            geometry.materialid = materialID;
            geometry.translation = translation;  
            geometry.rotation = rotation;        
            geometry.scale = scale;              
            geometry.transform = utilityCore::buildTransformationMatrix(
                geometry.translation, geometry.rotation, geometry.scale);
            geometry.inverseTransform = glm::inverse(geometry.transform);
            geometry.invTranspose = glm::inverseTranspose(geometry.transform);

            geoms.push_back(geometry);
        }
    }
    std::cout << "GLTF loaded: "
        << vertices.size() << " verts, "
        << normals.size() << " normals, "
        << triIndexTriplets.size() << " tris, "
        << std::endl;
}