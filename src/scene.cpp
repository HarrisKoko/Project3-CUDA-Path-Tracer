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

void Scene::loadFromGLTF(const std::string& gltfName) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err, warn;

    const bool isGlb = gltfName.size() >= 4 &&
        gltfName.substr(gltfName.size() - 4) == ".glb";
    bool loaded = isGlb
        ? loader.LoadBinaryFromFile(&model, &err, &warn, gltfName)
        : loader.LoadASCIIFromFile(&model, &err, &warn, gltfName);

    if (!warn.empty()) std::cout << "GLTF warning: " << warn << "\n";
    if (!err.empty())  std::cerr << "GLTF error: " << err << "\n";
    if (!loaded) throw std::runtime_error("Failed to load glTF: " + gltfName);
    if (model.meshes.empty() || model.meshes[0].primitives.empty())
        throw std::runtime_error("No mesh data in glTF.");

    // Just take the first primitive for now
    const auto& prim = model.meshes[0].primitives[0];

    // --- POSITION ---
    auto posIt = prim.attributes.find("POSITION");
    if (posIt == prim.attributes.end())
        throw std::runtime_error("Primitive has no POSITION.");

    const tinygltf::Accessor& posAcc = model.accessors[posIt->second];
    const tinygltf::BufferView& posBv = model.bufferViews[posAcc.bufferView];
    const tinygltf::Buffer& posBuf = model.buffers[posBv.buffer];

    if (posAcc.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT || posAcc.type != TINYGLTF_TYPE_VEC3)
        throw std::runtime_error("POSITION must be float vec3.");

    const size_t posStride = posAcc.ByteStride(posBv) ? posAcc.ByteStride(posBv) : sizeof(float) * 3;
    const unsigned char* posStart = posBuf.data.data() + posBv.byteOffset + posAcc.byteOffset;

    const size_t baseVertex = vertices.size();
    vertices.reserve(vertices.size() + posAcc.count);
    for (size_t i = 0; i < posAcc.count; ++i) {
        const float* p = reinterpret_cast<const float*>(posStart + i * posStride);
        vertices.emplace_back(p[0], p[1], p[2]);
    }

    // --- NORMAL ---
    auto normIt = prim.attributes.find("NORMAL");
    if (normIt != prim.attributes.end()) {
        const tinygltf::Accessor& nAcc = model.accessors[normIt->second];
        const tinygltf::BufferView& nBv = model.bufferViews[nAcc.bufferView];
        const tinygltf::Buffer& nBuf = model.buffers[nBv.buffer];

        if (nAcc.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT || nAcc.type != TINYGLTF_TYPE_VEC3)
            throw std::runtime_error("NORMAL must be float vec3.");

        const size_t nStride = nAcc.ByteStride(nBv) ? nAcc.ByteStride(nBv) : sizeof(float) * 3;
        const unsigned char* nStart = nBuf.data.data() + nBv.byteOffset + nAcc.byteOffset;

        normals.reserve(normals.size() + nAcc.count);
        for (size_t i = 0; i < nAcc.count; ++i) {
            const float* n = reinterpret_cast<const float*>(nStart + i * nStride);
            normals.emplace_back(glm::normalize(glm::vec3(n[0], n[1], n[2])));
        }
    }
    else {
        // Fallback: fill with dummy normals for new verts
        normals.resize(vertices.size(), glm::vec3(0, 1, 0));
    }

    // --- INDICES ---
    size_t baseIndex = indices.size();
    if (prim.indices >= 0) {
        const auto& iAcc = model.accessors[prim.indices];
        const auto& iBv = model.bufferViews[iAcc.bufferView];
        const auto& iBuf = model.buffers[iBv.buffer];
        const unsigned char* iStart = iBuf.data.data() + iBv.byteOffset + iAcc.byteOffset;

        indices.resize(baseIndex + iAcc.count);

        auto readIdx = [&](size_t k)->uint32_t {
            switch (iAcc.componentType) {
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                return reinterpret_cast<const uint8_t*>(iStart)[k];
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                return reinterpret_cast<const uint16_t*>(iStart)[k];
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                return reinterpret_cast<const uint32_t*>(iStart)[k];
            default:
                throw std::runtime_error("Unsupported index type.");
            }
            };

        for (size_t k = 0; k < iAcc.count; ++k)
            indices[baseIndex + k] = readIdx(k) + static_cast<uint32_t>(baseVertex);
    }
    else {
        // No index buffer: synthesize a trivial 0..N-1 index list
        indices.resize(baseIndex + posAcc.count);
        for (size_t k = 0; k < posAcc.count; ++k)
            indices[baseIndex + k] = static_cast<uint32_t>(baseVertex + k);
    }

    // --- Geom ---
    Geom geom{};
    geom.type = MESH;
    geom.materialid = 0;
    geom.translation = glm::vec3(0);
    geom.rotation = glm::vec3(0);
    geom.scale = glm::vec3(1);
    geom.transform = utilityCore::buildTransformationMatrix(geom.translation, geom.rotation, geom.scale);
    geom.inverseTransform = glm::inverse(geom.transform);
    geom.invTranspose = glm::inverseTranspose(geom.transform);

    geoms.push_back(geom);

    const size_t addedVerts = vertices.size() - baseVertex;
    const size_t addedIdx = indices.size() - baseIndex;
    std::cout << "Loaded " << addedVerts << " vertices, "
        << normals.size() - baseVertex << " normals, "
        << addedIdx << " indices from " << gltfName << "\n";
}
