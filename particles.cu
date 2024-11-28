
#include <SFML/Graphics.hpp>
#include <iostream>
#include <cstdint>
#include "tensor/tensor.hpp"
#include <chrono>
#include <cmath>
#include "ops/ops.h"
#include <cuda_runtime.h>
       // For general CUDA runtime functions
// opengl
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
// glcreateprogram
#include <cuda_gl_interop.h>   


auto width = 256 ;
auto height = 256;
auto depth = 256;

std::string vertexShader = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;

// MATRIX UNIFORMS
uniform mat4 aWorldViewProjection;
uniform mat4 aWorld;
uniform mat4 aView;
uniform mat4 aProjection;


out vec3 ourColor;
varying vec4 position;

void main()
{
    position = vec4(aPos, 1.0);
    gl_Position = aWorldViewProjection * vec4(aPos, 1.0);
    ourColor = aColor;
}
)";

std::string fragmentShader = R"(
#version 330 core

// buffer for tensorfield
#extension GL_ARB_shader_storage_buffer_object : require
layout(std430, binding = 0) buffer tensorField
{
    vec4 data[];
};

out vec4 FragColor;
varying vec4 position;
in vec3 ourColor;
uniform ivec3 size;



void main()
{
    vec4 tensor = data[int(position.z+0.1) + int(position.x+0.1) * size.x + int(position.y+0.1) * size.x * size.y];
    vec4 tensor2 = data[int(position.z-0.1) + int(position.x-0.1) * size.x + int(position.y-0.1) * size.x * size.y];
    if (tensor.w > 0.5 || tensor2.w > 0.5)
    {
        FragColor = vec4(tensor.xyz*0.5+0.5, 1.0);
    }
    else
    {
        
        discard;
        
    }
    
}
)";


auto loadShader(std::string vertex, std::string fragment)
{
    auto vertexShader = glCreateShader(GL_VERTEX_SHADER);
    const char *vertexShaderSource = vertex.c_str();
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    auto fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    const char *fragmentShaderSource = fragment.c_str();
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    // get compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n"
                  << infoLog << std::endl;
    }

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n"
                  << infoLog << std::endl;
    }
    auto shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // print compile errors if any
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::COMPILATION_FAILED\n"
                  << infoLog << std::endl;
    }

    // set current shader
    glUseProgram(shaderProgram);
    
    // set to blend mode add
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    return shaderProgram;
}



auto initGL()
{
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        std::cout << "Failed to initialize GLEW" << std::endl;
        return -1;
    }
    
   
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    glEnable(GL_DEPTH_TEST);
    // make add
    glEnable(GL_BLEND);
    return 0;
}




auto repulsion = 0.02;
float gravity = -0.01;
auto warp = 0.0;
auto scale = 1;

#define SIZE 1
int size = SIZE;
auto numthreads = 1;

int particlecount = 100000000;

void draw_circle(Tensor &screenbuffer, int x, int y, int radius, int r, int g, int b, int a)
{
    for (int i = -radius; i < radius; i++)
    {
        for (int j = -radius; j < radius; j++)
        {
            if (i * i + j * j < radius * radius && x + j > 0 && x + j < screenbuffer.shape[1] && y + i > 0 && y + i < screenbuffer.shape[0])
            {

                uint64_t *pixel = screenbuffer.get<uint64_t>(y + i, x + j);
                uint8_t *pixel8 = (uint8_t *)pixel;
                pixel8[0] += r;
                pixel8[1] += g;
                pixel8[2] += b;
                pixel8[3] = 255;
            }
        }
    }
};


#define FADEOFF(io, jo, ko) 1
// #define FADEOFF(io, jo) 1.0
// #define FADEOFF(io,jo) sqrt(io*io+jo*jo + 1.0)


__global__ void ProcessParticlesKernel(float4 *particles, float4 *tensorField, int numParticles,
                                       int width, int height, int depth, float4 gravity,
                                       float repulsion, int numblocks, float friction)
{

    for (int ioo = 0; ioo < numblocks; ioo++)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x + ioo * blockDim.x * gridDim.x;
        if (tid >= numParticles)
            return;

        float4 particle = particles[tid];
        if (particle.x == 0 || particle.y == 0)
            return;

        

        float4 momentum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        for (int io = -SIZE; io <= SIZE; io++)
        {
            for (int jo = -SIZE; jo <= SIZE; jo++)
            {
                for (int k = -SIZE; k <= SIZE; k++)
                {
                    int4 offset = make_int4(io, jo, k, 0);
                    float4 *fieldPtr = tensorField + ((int(particle.y) + offset.y) * width + (int(particle.x) + offset.x)) * depth + (int(particle.z) + offset.z);
                    momentum.x += fieldPtr->x;
                    momentum.y += fieldPtr->y;
                    momentum.z += fieldPtr->z;
                    fieldPtr->x = 0;
                    fieldPtr->y = 0;
                    fieldPtr->z = 0;
                }
            }
        }
        float4* myloc = tensorField + ((int(particle.y)) * width + (int(particle.x))) * depth + (int(particle.z));
        myloc->w = 0;
        float4 oldloc = {particle.x, particle.y, particle.z, 0};
        // momentum.y += gravity;
        particle.x += momentum.x;
        particle.y += momentum.y;
        particle.z += momentum.z;

              
        
        momentum.y += gravity.y;
        momentum.x += gravity.x;
        momentum.z += gravity.z;
       

        // momentum.x *= friction;
        // momentum.y *= friction;
        // momentum.z *= friction;

        for (int io = -SIZE; io <= SIZE; io++)
        {
            for (int jo = -SIZE; jo <= SIZE; jo++)
            {
                for (int k = -SIZE; k <= SIZE; k++)
                {
                    if (particle.x + io <= 1.0 + SIZE || particle.y + jo <= 1.0 + SIZE || particle.z + k <= 1.0 + SIZE || particle.x + io >= width - (1.0 + SIZE) || particle.y + jo >= height - (1.0 + SIZE) || particle.z + k >= depth - (1.0 + SIZE))
                    {
                        continue;
                    }
                    int4 offset = make_int4(io, jo, k, 0);
                    // float fade = FADEOFF(io, jo, k);
                    float4 offsetto;
                    offsetto.x = io;
                    offsetto.y = jo;
                    offsetto.z = k;

                    float4 moffset;
                    
                    moffset.x = (offsetto.x * repulsion + momentum.x/27.0f) ;// fade;
                    moffset.y = (offsetto.y * repulsion + momentum.y/27.0f) ;// fade;
                    moffset.z = (offsetto.z * repulsion + momentum.z/27.0f) ;// fade;
                    float *fieldPtr = (&tensorField->x) + 4 * (((int(particle.y) + offset.y) * width + (int(particle.x) + offset.x)) * depth + (int(particle.z) + offset.z));
                    *(fieldPtr) += moffset.x;
                    *(fieldPtr + 1) += moffset.y;
                    *(fieldPtr + 2) += moffset.z;
                    
                }
            }
        }
        myloc = tensorField + ((int(particle.y)) * width + (int(particle.x))) * depth + (int(particle.z));
        if(particle.x < 1.0 + SIZE || particle.y < 1.0 + SIZE || particle.z < 1.0 + SIZE || particle.x > width - (1.0 + SIZE) || particle.y > height - (1.0 + SIZE) || particle.z > depth - (1.0 + SIZE) ||
            
            myloc->w >= 0.75){
            particle.x = oldloc.x;
            particle.y = oldloc.y;
            particle.z = oldloc.z;
        }
        myloc = tensorField + ((int(particle.y)) * width + (int(particle.x))) * depth + (int(particle.z));
        myloc->w = 1;
        

        particles[tid].x = particle.x;
        particles[tid].y = particle.y;
        particles[tid].z = particle.z;
        particles[tid].w = particle.w;
    }
}

void LaunchProcessParticlesKernel(float4 *d_particles, float4 *d_tensorField, int numParticles,
                                  int width, int height, int depth, float4 gravity, float repulsion, float friction)
{
    int blockSize = 1024; // Adjust this based on your GPU's capabilities
    int numBlocks = (numParticles + blockSize - 1) / blockSize;

    ProcessParticlesKernel<<<numBlocks, blockSize>>>(d_particles, d_tensorField, numParticles,
                                                     width, height, depth, gravity, repulsion, 1, friction);

    cudaDeviceSynchronize(); // Ensure all threads have finished
}

struct thread
{
    Tensor *Particles;
    float4 *TensorField;
    GLuint TensorFieldBuffer;
    cudaGraphicsResource* TensorFieldResource = 0;
    int numParticles;
    float4 *gpuparticles;
    float friction = 0.0;
    // std::thread t;

    thread(int numParticles)
    {
        this->Particles = new Tensor({numParticles, 4}, kFLOAT_32);

        cudaMalloc(&gpuparticles, numParticles * sizeof(float4));
        
        glGenBuffers(1, &TensorFieldBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, TensorFieldBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float4) *  width * height * depth, NULL,  GL_DYNAMIC_READ);
        // check for errors
        auto glbuffererror = glGetError();
        if (glbuffererror != GL_NO_ERROR)
        {
            std::cout << "GLBufferError: " << glbuffererror << std::endl;
        }

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        auto glerror = glGetError();
        if (glerror != GL_NO_ERROR)
        {
            std::cout << "GLError: " << glerror << std::endl;
        }
        

        auto error = cudaGraphicsGLRegisterBuffer(&TensorFieldResource, TensorFieldBuffer, cudaGraphicsRegisterFlagsNone);
        if (error != cudaSuccess)
        {
            std::cout << "RegisterError: " << cudaGetErrorString(error) << ": " << error << std::endl;
        }

        size_t size;
        size = width * height * depth * sizeof(float4);
        cudaGraphicsMapResources(1, &TensorFieldResource);
        auto ptrerror = cudaGraphicsResourceGetMappedPointer((void**)&TensorField, &size, TensorFieldResource);

        if (ptrerror != cudaSuccess)
        {
            std::cout << "GetPtrError: " << cudaGetErrorString(ptrerror) << std::endl;
        }
        


        *Particles = 0;
        this->numParticles = numParticles;
        
        for (int i = -SIZE; i <= SIZE; i++)
        {
            for (int j = -SIZE; j <= SIZE; j++)
            {
                for (int k = -SIZE; k <= SIZE; k++)
                {
                    friction += 1.0f / FADEOFF(i, j, k);
                }
            }
        }
        friction = 1.0f / friction;
    }

    void run(int activepartivles)
    {
        // copy particles to device
        float4 gravity4 = {0.f, gravity, 0.f, 0.f};
        if (activepartivles > 1000000)
            gravity4 = {gravity, 0.f, 0.f, 0.f};
        LaunchProcessParticlesKernel((float4 *)this->gpuparticles, this->TensorField, numParticles, width, height, depth, gravity4, repulsion, friction);
    }

    void update()
    {
        // copy particles to host
        cudaMemcpy(Particles->data, gpuparticles, numParticles * sizeof(float4), cudaMemcpyDeviceToHost);
    }

    void flushToGPU()
    {
        cudaMemcpy(gpuparticles, Particles->data, numParticles * sizeof(float4), cudaMemcpyHostToDevice);
    }

    
};

struct Camera
{
    float4 position;
    float4 rotation;
    float4 scaling;
    float4 offset;
    Tensor projectionMatrix = Tensor({4, 4}, kFLOAT_32);
    Tensor rotationMatrix = Tensor({4, 4}, kFLOAT_32);
    Tensor rotationTranslationMatrix = Tensor({4, 4}, kFLOAT_32);
    Tensor translationMatrix = Tensor({4, 4}, kFLOAT_32);
    Tensor worldViewProjectionMatrix = Tensor({4, 4}, kFLOAT_32);
    Tensor identityMatrix = Tensor({4, 4}, kFLOAT_32);
    float fov = 3.1415 / 4.0;
    float aspect = 1;
    float near = 1;
    float far = 10000;
    bool orthographic = false;

    Camera()
    {
        position = {40, 40, 2000, 0};
        rotation = {0, 0, 0, 0};
        scaling = {1, 1, 1, 1};
        offset = {0, 0, 0, 0};
        identityMatrix = 0;
        identityMatrix[0][0] = 1;
        identityMatrix[1][1] = 1;
        identityMatrix[2][2] = 1;
        identityMatrix[3][3] = 1;

    }

    void updateProjectionMatrix()
    {
        projectionMatrix = identityMatrix;
        if (!orthographic)
        {
            // Perspective Projection Matrix
            float tanHalfFov = tan(fov / 2.0f);
            projectionMatrix[0][0] = 1.0f / (aspect * tanHalfFov);
            projectionMatrix[1][1] = 1.0f / tanHalfFov;
            projectionMatrix[2][2] = -(far + near) / (far - near);
            projectionMatrix[2][3] = -1.0f;
            projectionMatrix[3][2] = -(2.0f * far * near) / (far - near);
        }
        else
        {
            // Orthographic Projection Matrix
            projectionMatrix[0][0] = 1.0f / (aspect * tan(fov / 2.0f));
            projectionMatrix[1][1] = 1.0f / tan(fov / 2.0f);
            projectionMatrix[2][2] = 2.0f / (far - near);
            projectionMatrix[2][3] = 0.0f;
            projectionMatrix[3][2] = -(far + near) / (far - near);
        }
    }

    void updateWorldViewProjectionMatrix()
    {

        // Rotation Matrix
        rotationMatrix = identityMatrix;
        rotationMatrix[0][0] = cos(rotation.y) * cos(rotation.z);
        rotationMatrix[0][1] = cos(rotation.z) * sin(rotation.x) * sin(rotation.y) - cos(rotation.x) * sin(rotation.z);
        rotationMatrix[0][2] = cos(rotation.x) * cos(rotation.z) * sin(rotation.y) + sin(rotation.x) * sin(rotation.z);
        rotationMatrix[1][0] = cos(rotation.y) * sin(rotation.z);

        rotationMatrix[1][1] = cos(rotation.x) * cos(rotation.z) + sin(rotation.x) * sin(rotation.y) * sin(rotation.z);
        rotationMatrix[1][2] = -cos(rotation.z) * sin(rotation.x) + cos(rotation.x) * sin(rotation.y) * sin(rotation.z);
        rotationMatrix[2][0] = -sin(rotation.y);

        rotationMatrix[2][1] = cos(rotation.y) * sin(rotation.x);
        rotationMatrix[2][2] = cos(rotation.x) * cos(rotation.y);

        // Translation Matrix
        translationMatrix = identityMatrix;
        translationMatrix[3][0] = -position.x;
        translationMatrix[3][1] = -position.y;
        translationMatrix[3][2] = -position.z;

        worldViewProjectionMatrix = 0;
        rotationTranslationMatrix = 0;

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                for (int k = 0; k < 4; k++)
                {
                    *rotationTranslationMatrix[j][i].as<float>() += *rotationMatrix[k][i].as<float>() * *translationMatrix[j][k].as<float>();
                }
            }
        }

        // worldViewProjectionMatrix = worldViewProjectionMatrix * translationMatrix;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                for (int k = 0; k < 4; k++)
                {
                    *worldViewProjectionMatrix[j][i].as<float>() += *projectionMatrix[k][i].as<float>() * *rotationTranslationMatrix[j][k].as<float>();
                }
            }
        }
    }

    void update()
    {
            // reset mouse position

            // lastMousePos = {float(localPosition.x), float(localPosition.y), 0., 0.};
            float speed = 10;
             if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
            {
                speed = 20;
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift))
            {
                speed = 1;
            }


            if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
            {
                position.z -= cos(rotation.y) * speed * cos(rotation.x);
                position.x -= sin(rotation.y) * speed * cos(rotation.x);
                position.y += sin(rotation.x) * speed;
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
            {
                position.z += cos(rotation.y) * speed * cos(rotation.x);
                position.x += sin(rotation.y) * speed * cos(rotation.x);
                position.y -= sin(rotation.x) * speed;
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
            {
                position.x -= cos(rotation.y) * speed;
                position.z += sin(rotation.y) * speed;
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
            {
                position.x += cos(rotation.y) * speed;
                position.z -= sin(rotation.y) * speed;
            }
        updateProjectionMatrix();
        updateWorldViewProjectionMatrix();
    }

    float4 apply(float4 input)
    {
        float4 out = {0, 0, 0, 0};

        out.x += *worldViewProjectionMatrix[0][0].as<float>() * input.x;
        out.y += *worldViewProjectionMatrix[0][1].as<float>() * input.x;
        out.z += *worldViewProjectionMatrix[0][2].as<float>() * input.x;
        out.w += *worldViewProjectionMatrix[0][3].as<float>() * input.x;
        out.x += *worldViewProjectionMatrix[1][0].as<float>() * input.y;
        out.y += *worldViewProjectionMatrix[1][1].as<float>() * input.y;
        out.z += *worldViewProjectionMatrix[1][2].as<float>() * input.y;
        out.w += *worldViewProjectionMatrix[1][3].as<float>() * input.y;
        out.x += *worldViewProjectionMatrix[2][0].as<float>() * input.z;
        out.y += *worldViewProjectionMatrix[2][1].as<float>() * input.z;
        out.z += *worldViewProjectionMatrix[2][2].as<float>() * input.z;
        out.w += *worldViewProjectionMatrix[2][3].as<float>() * input.z;
        out.x += *worldViewProjectionMatrix[3][0].as<float>() * input.w;
        out.y += *worldViewProjectionMatrix[3][1].as<float>() * input.w;
        out.z += *worldViewProjectionMatrix[3][2].as<float>() * input.w;
        out.w += *worldViewProjectionMatrix[3][3].as<float>() * input.w;

        out.x = (out.x - 0.25) / (out.z + 0.0001) + 0.25;
        out.y = (out.y - 0.25) / (out.z + 0.0001) + 0.25;

        out.x = out.x * 2;
        out.y = out.y * 2;

        out.x *= 1024;
        out.y *= 1024;
        return out;
    }
};


auto drawbox(float x, float y, float z, const int w, const int h, const int d, unsigned int shaderProgram, Camera &camera, thread& t)
{

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    int height = int(h);
    int width = int(w);
    int depth = int(d);
    float vertices[24 * (h + w + d)];
    for (int i = 0; i < h; i++)
     {
        // positions          // colors
        vertices[i * 24] = x;
        vertices[i * 24 + 1] = y+i;
        vertices[i * 24 + 2] = z;

        vertices[i * 24 + 3] = 0.0f;
        vertices[i * 24 + 4] = 1.0f;
        vertices[i * 24 + 5] = 1.0f;

        vertices[i * 24 + 6] = x;
        vertices[i * 24 + 7] = y+i;
        vertices[i * 24 + 8] = z + d;

        vertices[i * 24 + 9] = 0.0f;
        vertices[i * 24 + 10] = 1.0f;
        vertices[i * 24 + 11] = 1.0f;

        vertices[i * 24 + 12] = x + w;
        vertices[i * 24 + 13] = y+i;
        vertices[i * 24 + 14] = z + d;

        vertices[i * 24 + 15] = 0.0f;
        vertices[i * 24 + 16] = 1.0f;
        vertices[i * 24 + 17] = 1.0f;

        vertices[i * 24 + 18] = x + w;
        vertices[i * 24 + 19] = y+i;
        vertices[i * 24 + 20] = z;

        vertices[i * 24 + 21] = 0.0f;
        vertices[i * 24 + 22] = 1.0f;
        vertices[i * 24 + 23] = 1.0f;
        

    };

    for (int i = h; i < h + w; i++)
    {
        uint loc = i - h;
        // positions          // colors
        vertices[i * 24] = x+loc;
        vertices[i * 24 + 1] = y;
        vertices[i * 24 + 2] = z;

        vertices[i * 24 + 3] = 1.0f;
        vertices[i * 24 + 4] = 0.0f;
        vertices[i * 24 + 5] = 1.0f;

        vertices[i * 24 + 6] = x+loc;
        vertices[i * 24 + 7] = y;
        vertices[i * 24 + 8] = z + d;

        vertices[i * 24 + 9] = 1.0f;
        vertices[i * 24 + 10] = 0.0f;
        vertices[i * 24 + 11] = 1.0f;

        vertices[i * 24 + 12] = x+loc;
        vertices[i * 24 + 13] = y + h;
        vertices[i * 24 + 14] = z + d;

        vertices[i * 24 + 15] = 1.0f;
        vertices[i * 24 + 16] = 0.0f;
        vertices[i * 24 + 17] = 1.0f;

        vertices[i * 24 + 18] = x+loc;
        vertices[i * 24 + 19] = y + h;
        vertices[i * 24 + 20] = z;

        vertices[i * 24 + 21] = 1.0f;
        vertices[i * 24 + 22] = 0.0f;
        vertices[i * 24 + 23] = 1.0f;
    };

    for (int i = h + w; i < h + w + d; i++)
    {
        uint loc = i - h - w;
        // positions          // colors
        vertices[i * 24] = x;
        vertices[i * 24 + 1] = y;
        vertices[i * 24 + 2] = z+loc;

        vertices[i * 24 + 3] = 0.0f;
        vertices[i * 24 + 4] = 1.0f;
        vertices[i * 24 + 5] = 0.0f;

        vertices[i * 24 + 6] = x;
        vertices[i * 24 + 7] = y + h;
        vertices[i * 24 + 8] = z+loc;

        vertices[i * 24 + 9] = 0.0f;
        vertices[i * 24 + 10] = 1.0f;
        vertices[i * 24 + 11] = 0.0f;

        vertices[i * 24 + 12] = x + w;
        vertices[i * 24 + 13] = y + h;
        vertices[i * 24 + 14] = z+loc;

        vertices[i * 24 + 15] = 0.0f;
        vertices[i * 24 + 16] = 1.0f;
        vertices[i * 24 + 17] = 0.0f;

        vertices[i * 24 + 18] = x + w;
        vertices[i * 24 + 19] = y;
        vertices[i * 24 + 20] = z+loc;

        vertices[i * 24 + 21] = 0.0f;
        vertices[i * 24 + 22] = 1.0f;
        vertices[i * 24 + 23] = 0.0f;
    };

    unsigned int numquads = sizeof(vertices) / sizeof(float) / 6;

    unsigned int indices[numquads];
    for (int i = 0; i < numquads; i++)
    {
        indices[i] = i;
    }

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // camera
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "aWorldViewProjection"), 1, GL_FALSE, (float*)camera.worldViewProjectionMatrix.data);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "aWorld"), 1, GL_FALSE, (float*)camera.rotationTranslationMatrix.data);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "aView"), 1, GL_FALSE, (float*)camera.rotationMatrix.data);
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "aProjection"), 1, GL_FALSE, (float*)camera.projectionMatrix.data);

    // set uniform for tensorfield size
    glUniform3i(glGetUniformLocation(shaderProgram, "size"), width, height, depth);
    

    // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // set uniform for tensorfield
    // unbind the buffer
    // sync cuda
    cudaDeviceSynchronize();
    cudaGraphicsUnmapResources(1, &t.TensorFieldResource);
    size_t size = width * height * depth * sizeof(float4);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, t.TensorFieldBuffer);

    

    

    // remember: do NOT unbind the EBO while a VAO is active as the bound element buffer object IS stored in the VAO; keep the EBO bound.
    // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
    // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
    // glBindVertexArray(0);
    // glUseProgram(shaderProgram);

    // draw our first triangle
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO);
    // make sure depth testing is enabled
    
    glDrawElements(GL_QUADS, sizeof(indices) / sizeof(unsigned int), GL_UNSIGNED_INT, 0);
    glEnd();

    // remap the buffer
    cudaGraphicsMapResources(1, &t.TensorFieldResource);
    cudaGraphicsResourceGetMappedPointer((void**)&t.TensorField, &size, t.TensorFieldResource);
}


int main()
{

    // auto friction = ((1.0-0.0)/ pow(size*2 + 1,2));

    // create the window
    sf::RenderWindow window(sf::VideoMode(1024, 1024), "Some Funky Title", sf::Style::Default, sf::ContextSettings(32));
    sf::Texture texture;
    texture.create(1024, 1024);

    initGL();
    auto shaderProgram = loadShader(vertexShader, fragmentShader);

    // create a texture

    auto camera = Camera();

    std::vector<thread> threads;

    for (int i = 0; i < numthreads; i++)
    {
        threads.push_back(thread(particlecount / numthreads));
    }

    window.setMouseCursorGrabbed(true);
    window.setMouseCursorVisible(false);
    window.setMouseCursorGrabbed(true);

    auto count = 0;
    // The colour we will fill the window with
    unsigned char red = 0;
    unsigned char blue = 255;
    auto currenttime = std::chrono::system_clock::now();

    // run the program as long as the window is open
    auto lastFrame = std::chrono::system_clock::now();
    auto fps = 60;
    bool ispressed = false;
    int frame = 0;
    bool but = 0;
    uint32_t brushsize = 20;

    auto lastMousePos = sf::Mouse::getPosition(window);

    while (window.isOpen())
    {
        threads[0].run(count);
        // check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
        while (window.pollEvent(event))
        {
            // "close requested" event: we close the window
            if (event.type == sf::Event::Closed)
                window.close();

            // if esc is pressed close the window
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                window.close();
            if (sf::Event(event).type == sf::Event::MouseWheelScrolled)
            {
                brushsize += int(sf::Event(event).mouseWheelScroll.delta);
            }
            if (sf::Event(event).type == sf::Event::MouseMoved)
            {
                // if(!window.hasFocus())continue;
                auto mpos = sf::Event(event).mouseMove;
                int2 delta = {mpos.x - lastMousePos.x, mpos.y - lastMousePos.y};
                // if(delta.x==0 && delta.y==0)continue;

                lastMousePos = {mpos.x, mpos.y};
                if (mpos.x == 512 && mpos.y == 512)
                    continue;

                camera.rotation.x -= float(delta.y) / 300;
                camera.rotation.y -= float(delta.x) / 300;


                //
                // localPosition = sf::Mouse::getPosition(window);
            }
        }

        // clear the window with black color
        auto show = (std::chrono::system_clock::now() - lastFrame) > std::chrono::milliseconds(1000 / fps);
        if (show)
        {
            lastFrame = std::chrono::system_clock::now();

            window.clear(sf::Color::Black);

            // set screenbuffergpu to black
            // cudaMemset(screenbufferGPU, 0, 1024 * 1024 * 4 * sizeof(uint8_t));
        }


        
        bool lpressed = sf::Mouse::isButtonPressed(sf::Mouse::Left);
        bool rpressed = sf::Mouse::isButtonPressed(sf::Mouse::Right);
        bool mpressed = sf::Mouse::isButtonPressed(sf::Mouse::Middle);

        // scroll for brush size

        bool pressed = lpressed || rpressed;
        // if (pressed && !ispressed)
        // {
        //     startdragpos = {localPosition.x + rand() % scale, localPosition.y};
        //     but = lpressed;
        // }

        float4 mmpos = {width / 2, height / 2, depth / 2, 0};
        if (pressed && show)
        {

            float4 startdragpos = {mmpos.x - brushsize/2, mmpos.y - brushsize / 2, mmpos.z - brushsize / 2, 0.};
            but = lpressed;
            float4 enddragpos = {mmpos.x + brushsize/2, mmpos.y + brushsize / 2, mmpos.z + brushsize / 2, 0.};
            pressed = 0;
            ispressed = 1;

            // threads[0].update();
            float4 pos = {0, 0, 0, 1.0};

            for (int i = startdragpos.x; i < enddragpos.x; i += (size * 2) * but + 1)
            {
                for (int j = startdragpos.y; j < enddragpos.y; j += (size * 2) * but + 1)
                {
                    for (int k = startdragpos.z; k < enddragpos.z; k += (size * 2) * but + 1)
                    {
                        auto part = threads[0].gpuparticles + count;
                        // *part = double2{i * scale, j*scale };
                        pos.x = i;
                        pos.y = j;
                        pos.z = k;

                        cudaMemcpy(part, &pos, sizeof(float4), cudaMemcpyHostToDevice);
                        count += 1;
                    }
                }
            }
            // threads[count % numthreads].flushToGPU();
        }

        ispressed = pressed;

        // Update screen
        if (show)
        {

            
            sf::Mouse::setPosition(sf::Vector2i(512, 512), window);
           
            camera.update();
            drawbox(0, 0, 0, width, height, depth, shaderProgram, camera, threads[0]);

           

            window.display();
            auto currtime = std::chrono::system_clock::now();
            auto camerpos = camera.position;
            auto zeropos = camera.apply({0, 0, 0, 1});
            // std::cout << "frames per second(0dp): " << int(1.0 / std::chrono::duration<double>(currtime - currenttime).count()) << " : Active Particles: " << std::min(count, particlecount) << " : Brush Size: " << brushsize << "\r";
            std::cout << "camera position: " << camerpos.x << " " << camerpos.y << " " << camerpos.z << " : zero position: " << zeropos.x << " " << zeropos.y << " " << zeropos.z << " : frames per second(0dp): " << int(1.0 / std::chrono::duration<double>(currtime - currenttime).count()) << " : Active Particles: " << std::min(count, particlecount) << " : Brush Size: " << brushsize << "\r";
            std::cout << std::flush;
            currenttime = currtime;
            frame++;
        }
        // if(frame==1000)break;
    }

    return 0;
}