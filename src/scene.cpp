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
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
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

    bool loaded = (gltfName.size() >= 4 &&
        gltfName.substr(gltfName.size() - 4) == ".glb")
        ? loader.LoadBinaryFromFile(&model, &err, &warn, gltfName)
        : loader.LoadASCIIFromFile(&model, &err, &warn, gltfName);

    if (!warn.empty()) std::cout << "GLTF warning: " << warn << "\n";
    if (!err.empty())  std::cerr << "GLTF error: " << err << "\n";
    if (!loaded)       throw std::runtime_error("Failed to load glTF: " + gltfName);

    if (model.meshes.empty() || model.meshes[0].primitives.empty())
        throw std::runtime_error("No mesh data in glTF.");

    const auto& prim = model.meshes[0].primitives[0];

    auto posIt = prim.attributes.find("POSITION");
    if (posIt == prim.attributes.end())
        throw std::runtime_error("Primitive has no POSITION.");

    const tinygltf::Accessor& accessor = model.accessors[posIt->second];
    const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
    const tinygltf::Buffer& buffer = model.buffers[view.buffer];

    const size_t stride = accessor.ByteStride(view);
    const unsigned char* dataStart =
        buffer.data.data() + view.byteOffset + accessor.byteOffset;

    Geom geom{};
    geom.type = MESH;
    geom.materialid = 0; 

    vertices.reserve(accessor.count);
    for (size_t i = 0; i < accessor.count; ++i) {
        const float* p = reinterpret_cast<const float*>(dataStart + i * stride);
        vertices.emplace_back(p[0], p[1], p[2]);
    }

    if (prim.indices >= 0) {
        const auto& iAcc = model.accessors[prim.indices];
        const auto& iView = model.bufferViews[iAcc.bufferView];
        const auto& iBuf = model.buffers[iView.buffer];

        const unsigned char* iStart =
            iBuf.data.data() + iView.byteOffset + iAcc.byteOffset;

        auto readIndex = [&](size_t idx)->uint32_t {
            switch (iAcc.componentType) {
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                return reinterpret_cast<const uint16_t*>(iStart)[idx];
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                return reinterpret_cast<const uint32_t*>(iStart)[idx];
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                return reinterpret_cast<const uint8_t*>(iStart)[idx];
            default:
                throw std::runtime_error("Unsupported index type.");
            }
            };

        indices.resize(iAcc.count);
        for (size_t i = 0; i < iAcc.count; ++i)
            indices[i] = readIndex(i);
    }

    geoms.push_back(std::move(geom));
    std::cout << "Loaded " << vertices.size()
        << " vertices from " << gltfName << "\n";
}
