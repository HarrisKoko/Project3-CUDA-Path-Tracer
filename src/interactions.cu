#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ float schlickFresnel(float cosTheta, float etaI, float etaT) {
    float r0 = (etaI - etaT) / (etaI + etaT);
    r0 = r0 * r0;
    float x = 1.0f - cosTheta;
    return r0 + (1.0f - r0) * powf(x, 5.0f);
}

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    glm::vec3 I = glm::normalize(pathSegment.ray.direction);
    glm::vec3 newDir;

    // Emissive: terminate path
    if (m.emittance > 0.0f) {
        pathSegment.color *= m.color * m.emittance;
        pathSegment.remainingBounces = 0;
        return;
    }

    // Reflective
    if (m.hasReflective > 0.0f) {
        newDir = glm::reflect(I, normal);
        pathSegment.ray.origin = intersect + 1e-4f * normal;
        pathSegment.ray.direction = glm::normalize(newDir);
        pathSegment.color *= m.color;
        pathSegment.remainingBounces--;
        return;
    }

    if (m.hasRefractive > 0.0f) {
        float etaI = 1.0f, etaT = m.indexOfRefraction;
        glm::vec3 n = normal;

        bool entering = glm::dot(I, normal) < 0.0f;
        if (!entering) {
            n = -normal;
            etaI = m.indexOfRefraction;
            etaT = 1.0f;
        }

        float cosi = glm::clamp(glm::dot(-I, n), -1.0f, 1.0f);
        float eta = etaI / etaT;

        // Fresnel reflectance
        float F = schlickFresnel(fabsf(cosi), etaI, etaT);

        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
        float toss = u01(rng);

        float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
        if (k < 0.0f || toss < F) {
            newDir = glm::reflect(I, n);
        }
        else {
            newDir = eta * I + (eta * cosi - sqrtf(k)) * n;

            if (!entering) {
                float dist = intersect.t;  // distance traveled in medium
                glm::vec3 absorb = glm::exp(-m.color * dist);
                pathSegment.color *= absorb;
            }
        }

        // Offset origin along the chosen ray direction (safer for refraction)
        float bias = 1e-4f;
        pathSegment.ray.origin = intersect + bias * newDir;
        pathSegment.ray.direction = glm::normalize(newDir);
        pathSegment.remainingBounces--;
        return;
    }




    // Diffuse
    newDir = calculateRandomDirectionInHemisphere(normal, rng);
    pathSegment.ray.origin = intersect + 1e-4f * normal;
    pathSegment.ray.direction = glm::normalize(newDir);
    pathSegment.color *= m.color;
    pathSegment.remainingBounces--;
}




