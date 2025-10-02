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
            loadFromGLTF(filepath);
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
}


namespace {
    inline AABB merge(const AABB& a, const AABB& b) {
        AABB r; r.bmin = glm::min(a.bmin, b.bmin); r.bmax = glm::max(a.bmax, b.bmax); return r;
    }
    inline AABB triBounds(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
        AABB r;
        r.bmin = glm::min(a, glm::min(b, c));
        r.bmax = glm::max(a, glm::max(b, c));
        return r;
    }
    inline glm::vec3 triCentroid(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
        return (a + b + c) * (1.0f / 3.0f);
    }

    // compute bounds & centroid-bounds of a [start,count) range over "order"
    inline void computeRangeBounds(
        const std::vector<glm::uvec3>& tris,
        const std::vector<glm::vec3>& verts,
        const std::vector<int>& order,
        int start, int count,
        AABB& outBounds, AABB& outCBox)
    {
        AABB b{}, cb{};
        b.bmin = glm::vec3(FLT_MAX); b.bmax = glm::vec3(-FLT_MAX);
        cb = b;
        for (int i = 0; i < count; ++i) {
            const glm::uvec3 t = tris[order[start + i]];
            const glm::vec3& v0 = verts[t.x];
            const glm::vec3& v1 = verts[t.y];
            const glm::vec3& v2 = verts[t.z];
            b = merge(b, triBounds(v0, v1, v2));
            AABB cc; const glm::vec3 c = triCentroid(v0, v1, v2);
            cc.bmin = cc.bmax = c;
            cb = merge(cb, cc);
        }
        outBounds = b; outCBox = cb;
    }
}

void Scene::buildBVH() {
    if (triIndexTriplets.empty()) { bvhNodes.clear(); return; }

    // 1) Permutation over triangles
    std::vector<int> order(triIndexTriplets.size());
    std::iota(order.begin(), order.end(), 0);

    // 2) READ-ONLY staging copy to avoid clobbering sources while writing leaves
    const std::vector<glm::uvec3> triSrc = triIndexTriplets;

    struct Task { int start, count; int parent; bool isLeft; };

    bvhNodes.clear();
    bvhNodes.reserve(int(triIndexTriplets.size() * 2));

    // root
    const int root = (int)bvhNodes.size();
    bvhNodes.push_back(BVHNode{}); // zero-init: box={min=+inf,max=-inf}, left/right=-1, firstTri=-1, triCount=0

    std::vector<Task> stack;
    stack.push_back({ 0, (int)order.size(), -1, true });

    const int LeafThreshold = 8;

    auto computeRangeBounds = [&](int start, int count, AABB& outBounds, AABB& outCBox) {
        AABB b, cb;
        b.bmin = glm::vec3(FLT_MAX); b.bmax = glm::vec3(-FLT_MAX);
        cb = b;
        for (int i = 0; i < count; ++i) {
            const glm::uvec3 t = triSrc[order[start + i]];
            const glm::vec3& v0 = vertices[t.x];
            const glm::vec3& v1 = vertices[t.y];
            const glm::vec3& v2 = vertices[t.z];
            // tri bounds
            AABB tb;
            tb.bmin = glm::min(v0, glm::min(v1, v2));
            tb.bmax = glm::max(v0, glm::max(v1, v2));
            b.bmin = glm::min(b.bmin, tb.bmin);
            b.bmax = glm::max(b.bmax, tb.bmax);
            // centroid box
            const glm::vec3 c = (v0 + v1 + v2) * (1.0f / 3.0f);
            cb.bmin = glm::min(cb.bmin, c);
            cb.bmax = glm::max(cb.bmax, c);
        }
        outBounds = b; outCBox = cb;
        };

    while (!stack.empty()) {
        Task t = stack.back(); stack.pop_back();

        AABB nodeB, cbox;
        computeRangeBounds(t.start, t.count, nodeB, cbox);

        int nodeIdx = (t.parent == -1) ? root : (int)bvhNodes.size();
        if (t.parent != -1) {
            if (t.isLeft) bvhNodes[t.parent].left = nodeIdx;
            else          bvhNodes[t.parent].right = nodeIdx;
            bvhNodes.push_back(BVHNode{});
        }

        BVHNode& node = bvhNodes[nodeIdx];
        node.box = nodeB;
        node.left = node.right = -1;
        node.firstTri = -1;
        node.triCount = 0;

        // Leaf
        if (t.count <= LeafThreshold) {
            node.firstTri = t.start;
            node.triCount = t.count;
            // WRITE triangles for this leaf from triSrc -> triIndexTriplets
            for (int i = 0; i < t.count; ++i) {
                const int dst = t.start + i;
                const int src = order[dst];
                triIndexTriplets[dst] = triSrc[src];  // <-- crucial: read from triSrc, not the array we're mutating
            }
            continue;
        }

        // Split by widest centroid axis
        glm::vec3 ext = cbox.bmax - cbox.bmin;
        int axis = (ext.y > ext.x && ext.y >= ext.z) ? 1 : (ext.z > ext.x && ext.z >= ext.y ? 2 : 0);
        float splitPos = 0.5f * (cbox.bmin[axis] + cbox.bmax[axis]);

        int i = t.start, j = t.start + t.count - 1;
        while (i <= j) {
            const glm::uvec3 tri = triSrc[order[i]];
            const glm::vec3 c = (vertices[tri.x] + vertices[tri.y] + vertices[tri.z]) * (1.0f / 3.0f);
            if (c[axis] < splitPos) ++i; else std::swap(order[i], order[j--]);
        }
        int leftCount = i - t.start;
        if (leftCount == 0 || leftCount == t.count) leftCount = t.count / 2;

        // Push children (right first so left is processed last)
        stack.push_back({ t.start + leftCount, t.count - leftCount, nodeIdx, false });
        stack.push_back({ t.start,             leftCount,           nodeIdx, true });
    }
}



void Scene::loadFromGLTF(const std::string& gltfName) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    bool isGlb = gltfName.size() >= 4 && gltfName.substr(gltfName.size() - 4) == ".glb";
    bool loaded = isGlb
        ? loader.LoadBinaryFromFile(&model, &err, &warn, gltfName)
        : loader.LoadASCIIFromFile(&model, &err, &warn, gltfName);

    if (!warn.empty()) std::cout << "GLTF warning: " << warn << "\n";
    if (!err.empty())  std::cerr << "GLTF error: " << err << "\n";
    if (!loaded) throw std::runtime_error("Failed to load glTF: " + gltfName);

    // --- Load textures into our list ---
    textures.clear();
    for (const auto& img : model.images) {
        TextureInfo tex;
        tex.filepath = img.uri; // relative path
        textures.push_back(tex);
    }

    // --- Loop over meshes & primitives ---
    for (const auto& mesh : model.meshes) {
        for (const auto& prim : mesh.primitives) {
            // POSITION
            auto posIt = prim.attributes.find("POSITION");
            if (posIt == prim.attributes.end()) continue;

            const tinygltf::Accessor& posAcc = model.accessors[posIt->second];
            const tinygltf::BufferView& posBv = model.bufferViews[posAcc.bufferView];
            const tinygltf::Buffer& posBuf = model.buffers[posBv.buffer];
            const unsigned char* posStart = posBuf.data.data() + posBv.byteOffset + posAcc.byteOffset;
            size_t posStride = posAcc.ByteStride(posBv) ? posAcc.ByteStride(posBv) : sizeof(float) * 3;

            size_t baseVertex = vertices.size();
            for (size_t i = 0; i < posAcc.count; ++i) {
                const float* p = reinterpret_cast<const float*>(posStart + i * posStride);
                vertices.emplace_back(p[0], p[1], p[2]);
            }

            // NORMAL
            auto normIt = prim.attributes.find("NORMAL");
            if (normIt != prim.attributes.end()) {
                const auto& nAcc = model.accessors[normIt->second];
                const auto& nBv  = model.bufferViews[nAcc.bufferView];
                const auto& nBuf = model.buffers[nBv.buffer];
                const unsigned char* nStart = nBuf.data.data() + nBv.byteOffset + nAcc.byteOffset;
                size_t nStride = nAcc.ByteStride(nBv) ? nAcc.ByteStride(nBv) : sizeof(float) * 3;

                for (size_t i = 0; i < nAcc.count; ++i) {
                    const float* n = reinterpret_cast<const float*>(nStart + i * nStride);
                    normals.emplace_back(glm::normalize(glm::vec3(n[0], n[1], n[2])));
                }
            } else {
                normals.resize(vertices.size(), glm::vec3(0,1,0));
            }

            // UVs
            auto uvIt = prim.attributes.find("TEXCOORD_0");
            if (uvIt != prim.attributes.end()) {
                const auto& uvAcc = model.accessors[uvIt->second];
                const auto& uvBv  = model.bufferViews[uvAcc.bufferView];
                const auto& uvBuf = model.buffers[uvBv.buffer];
                const unsigned char* uvStart = uvBuf.data.data() + uvBv.byteOffset + uvAcc.byteOffset;
                size_t uvStride = uvAcc.ByteStride(uvBv) ? uvAcc.ByteStride(uvBv) : sizeof(float) * 2;

                for (size_t i = 0; i < uvAcc.count; ++i) {
                    const float* uv = reinterpret_cast<const float*>(uvStart + i * uvStride);
                    uvs.emplace_back(uv[0], uv[1]);
                }
            } else {
                uvs.resize(vertices.size(), glm::vec2(0));
            }

            // INDICES
            size_t baseIndex = indices.size();
            if (prim.indices >= 0) {
                const auto& iAcc = model.accessors[prim.indices];
                const auto& iBv  = model.bufferViews[iAcc.bufferView];
                const auto& iBuf = model.buffers[iBv.buffer];
                const unsigned char* iStart = iBuf.data.data() + iBv.byteOffset + iAcc.byteOffset;

                indices.resize(baseIndex + iAcc.count);
                auto readIdx = [&](size_t k)->uint32_t {
                    switch (iAcc.componentType) {
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  return reinterpret_cast<const uint8_t*>(iStart)[k];
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: return reinterpret_cast<const uint16_t*>(iStart)[k];
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:   return reinterpret_cast<const uint32_t*>(iStart)[k];
                        default: throw std::runtime_error("Unsupported index type.");
                    }
                };
                for (size_t k = 0; k < iAcc.count; ++k)
                    indices[baseIndex + k] = readIdx(k) + static_cast<uint32_t>(baseVertex);
            } else {
                indices.resize(baseIndex + posAcc.count);
                for (size_t k = 0; k < posAcc.count; ++k)
                    indices[baseIndex + k] = static_cast<uint32_t>(baseVertex + k);
            }

            // Triangles
            for (size_t k = baseIndex; k + 2 < indices.size(); k += 3) {
                triIndexTriplets.emplace_back(indices[k], indices[k+1], indices[k+2]);
            }

            // Material
            Material newMat{};
            if (prim.material >= 0) {
                const auto& gltfMat = model.materials[prim.material];
                auto f = gltfMat.pbrMetallicRoughness.baseColorFactor;
                newMat.color = glm::vec3(f[0], f[1], f[2]);

                if (gltfMat.pbrMetallicRoughness.baseColorTexture.index >= 0) {
                    int texIndex = gltfMat.pbrMetallicRoughness.baseColorTexture.index;
                    int imgIndex = model.textures[texIndex].source;
                    newMat.color = glm::vec3(1); // will be replaced by sampled texture
                    // Save texture association
                    // e.g. store materialID -> imgIndex mapping
                }
            } else {
                newMat.color = glm::vec3(0.8f);
            }

            int matID = materials.size();
            materials.push_back(newMat);

            // Geom for this primitive
            Geom geom{};
            geom.type = MESH;
            geom.materialid = matID;
            geom.translation = glm::vec3(0);
            geom.rotation = glm::vec3(0);
            geom.scale = glm::vec3(1);
            geom.transform = utilityCore::buildTransformationMatrix(geom.translation, geom.rotation, geom.scale);
            geom.inverseTransform = glm::inverse(geom.transform);
            geom.invTranspose = glm::inverseTranspose(geom.transform);

            geoms.push_back(geom);
        }
    }

    buildBVH();
}
