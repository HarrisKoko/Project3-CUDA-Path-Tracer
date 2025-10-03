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

// Schlick's approximation of Fresnel reflectance
// Returns probability of reflection vs transmission at a dielectric interface
__host__ __device__ float schlickFresnel(float cosTheta, float etaIncident, float etaTransmitted)
{
    float r0 = (etaIncident - etaTransmitted) / (etaIncident + etaTransmitted);
    r0 = r0 * r0;
    float oneMinusCos = 1.0f - cosTheta;
    return r0 + (1.0f - r0) * powf(oneMinusCos, 5.0f);
}

// Scatter a ray based on material properties
__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 hitPoint,
    glm::vec3 surfaceNormal,
    const Material& material,
    thrust::default_random_engine& rng)
{
    glm::vec3 incomingDirection = glm::normalize(pathSegment.ray.direction);
    glm::vec3 scatteredDirection;

    // Emissive material
    if (material.emittance > 0.0f)
    {
        pathSegment.color *= material.color * material.emittance;
        pathSegment.remainingBounces = 0;
        return;
    }

    // Reflective material
    if (material.hasReflective > 0.0f)
    {
        scatteredDirection = glm::reflect(incomingDirection, surfaceNormal);
        pathSegment.ray.origin = hitPoint + 1e-4f * surfaceNormal;
        pathSegment.ray.direction = glm::normalize(scatteredDirection);
        pathSegment.color *= material.color;
        pathSegment.remainingBounces--;
        return;
    }

    // Refractive material
    if (material.hasRefractive > 0.0f)
    {
        // Determine if ray is entering or exiting the material
        bool isEntering = glm::dot(incomingDirection, surfaceNormal) < 0.0f;
        glm::vec3 outwardNormal = isEntering ? surfaceNormal : -surfaceNormal;

        float etaIncident = isEntering ? 1.0f : material.indexOfRefraction;
        float etaTransmitted = isEntering ? material.indexOfRefraction : 1.0f;
        float etaRatio = etaIncident / etaTransmitted;

        // Calculate Fresnel reflectance
        float cosIncident = glm::clamp(glm::dot(-incomingDirection, outwardNormal), 0.0f, 1.0f);
        float fresnelReflectance = schlickFresnel(cosIncident, etaIncident, etaTransmitted);

        // Compute reflection and refraction directions
        glm::vec3 reflectedDirection = glm::reflect(incomingDirection, outwardNormal);
        glm::vec3 refractedDirection = glm::refract(incomingDirection, outwardNormal, etaRatio);

        // Check for total internal reflection
        bool shouldReflect = (glm::length(refractedDirection) * glm::length(refractedDirection) < 1e-12f);
        if (!shouldReflect)
        {
            // Use Fresnel to stochastically choose between reflection and refraction
            thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
            shouldReflect = (u01(rng) < fresnelReflectance);
        }

        glm::vec3 finalDirection = shouldReflect ? reflectedDirection : refractedDirection;

        // Apply Beer-Lambert absorption for transmitted rays
        if (!shouldReflect)
        {
            float travelDistance = glm::length(hitPoint - pathSegment.ray.origin);

            // Compute absorption coefficient from material color
            glm::vec3 transmittance = glm::clamp(material.color, glm::vec3(1e-6f), glm::vec3(0.999f));
            glm::vec3 absorptionCoefficient = -glm::log(transmittance);
            pathSegment.color *= glm::exp(-absorptionCoefficient * travelDistance);
        }

        const float rayBias = 1e-3f;
        pathSegment.ray.direction = glm::normalize(finalDirection);
        pathSegment.ray.origin = hitPoint + rayBias * pathSegment.ray.direction;

        pathSegment.remainingBounces--;
        return;
    }

    // Diffuse material: scatter in random hemisphere direction
    scatteredDirection = calculateRandomDirectionInHemisphere(surfaceNormal, rng);
    pathSegment.ray.origin = hitPoint + 1e-3f * surfaceNormal;
    pathSegment.ray.direction = glm::normalize(scatteredDirection);
    pathSegment.color *= material.color;
    pathSegment.remainingBounces--;
}