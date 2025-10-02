#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromGLTF(const std::string& gltfName);
    void buildBVH();
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;

    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<unsigned int> indices;

    std::vector<Triangle> tris;

    std::vector<glm::uvec3> triIndexTriplets;
    std::vector<BVHNode>    bvhNodes;        


};
