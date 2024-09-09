
#include <SFML/Graphics.hpp>
#include <iostream>
#include <cstdint>
#include "tensor/tensor.hpp"
#include <chrono>
#include <cmath>
#include "ops/ops.h"
#include "vector/vectors.hpp"

void draw_circle(Tensor &screenbuffer, int x, int y, int radius, int r, int g, int b, int a)
{
    for (int i = -radius; i < radius; i++)
    {
        for (int j = -radius; j < radius; j++)
        {
            if (i * i + j * j < radius * radius && x + j > 0 && x + j < screenbuffer.shape[1] && y + i > 0 && y + i < screenbuffer.shape[0])
            {
                
                uint64_t* pixel = screenbuffer.get<uint64_t>(y + i,x + j);
                uint8_t* pixel8 = (uint8_t*)pixel;
                pixel8[0] += r;
                pixel8[1] += g;
                pixel8[2] += b;
                pixel8[3]  = 255;
            }
        }
    }
};



int main()
{
   
    auto repulsion = 0.2;
    auto gravity = 0.02;
    auto warp = 0.0;
    auto scale = 2;
    int size = 1;
    const int numParticles = 4096 * 100;
    const int width = 512*scale;
    const int height = 512*scale;

    auto friction = ((1.0-0.0)/ pow(size*2 + 1,2));

    // create the window
    sf::RenderWindow window(sf::VideoMode(width/scale, height/scale), "Some Funky Title");

    // create a texture
    sf::Texture texture;
    texture.create(width, height);

    // Create a pixel buffer to fill with RGBA data
    Tensor screenbuffer = Tensor({width, height, 4}, kUINT_8);
    Tensor Particles = Tensor({numParticles, 4}, kFLOAT_32);
    Particles = 0;
    Tensor TensorField = Tensor({width, height, 4}, kFLOAT_32);

   
    screenbuffer = 0;
    auto count = 0;
    TensorField = 0;

    // The colour we will fill the window with
    unsigned char red = 0;
    unsigned char blue = 255;
    auto currenttime = std::chrono::system_clock::now();

    // run the program as long as the window is open
    auto lastFrame = std::chrono::system_clock::now();
    auto fps = 60;
    bool ispressed = false;
    double2 startdragpos = {0, 0};
    int frame = 0;
    bool but = 0;
    while (window.isOpen())
    { 
        // check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
        while (window.pollEvent(event))
        {
            // "close requested" event: we close the window
            if (event.type == sf::Event::Closed)
                window.close();
        }
        

        // clear the window with black color
        auto show = (std::chrono::system_clock::now() - lastFrame) > std::chrono::milliseconds(1000 / fps);
        if (show)
        {
            lastFrame = std::chrono::system_clock::now();

            window.clear(sf::Color::Black);

            screenbuffer = 0;
        }

        
        // Create RGBA value to fill screen with.
        // Increment red and decrement blue on each cycle. Leave green=0, and make opaque
        // uint32_t RGBA;
        // Stuff data into buffer
        // get mouse position in the window
        sf::Vector2i localPosition = sf::Mouse::getPosition(window);

        // TensorFieldLast = TensorFieldLast*0.5;
        // TensorField[] = a;
        bool lpressed = sf::Mouse::isButtonPressed(sf::Mouse::Left);
        bool rpressed = sf::Mouse::isButtonPressed(sf::Mouse::Right);
        bool pressed = lpressed || rpressed;
        if(pressed && !ispressed){
            startdragpos = {localPosition.x, localPosition.y};
            but = lpressed;
        }

        if(!pressed && ispressed){
            auto enddragpos = int2{localPosition.x, localPosition.y};
            for (int i = startdragpos.x; i < enddragpos.x; i+=(size*2) + but)
            {
                for (int j = startdragpos.y; j < enddragpos.y; j+=(size*2) + but)
                {
                    auto part = Particles[count++].as<float4>();
                    // *part = double2{i * scale, j*scale };
                    part->x = i * scale;
                    part->y = j * scale;
                    part->w = but ? 1 : 0;
                }
            }
        }

        ispressed = pressed;
     
    
// #pragma omp parallel for
        for (int i = 0; i < numParticles; i++)
        {

            float4 &particle = *Particles[i].as<float4>();
            if (particle.x == 0 || particle.y == 0)
            {
                continue;
            }
            auto mm = particle.w;

            if(mm > 0){

                float4 momentum = {0, 0, 0, 0};
                
                // #pragma omp parallel for
                for (int io = -size; io < size+1; io++)
                {
                    for (int jo = -size; jo < size+1; jo++)
                    {
                        int2 offset = {io , jo };
                        momentum += *TensorField.get<float4>(particle.y + offset.y, particle.x + offset.x);
                        *TensorField.get<float4>(particle.y + offset.y, particle.x + offset.x) *= 0.0;
                                
                    }
                }
            
                momentum.y += gravity;

                momentum *= mm;

                if(particle.x + momentum.x < 1 || particle.x + momentum.x > width-2){
                    momentum.x = -momentum.x;
                }
                if(particle.y + momentum.y < 1 || particle.y + momentum.y > height-2){
                    momentum.y = -momentum.y;
                }
            
                particle += momentum;

                
                particle.x = std::max(1.0f + size, std::min(float(width-(1+size)), particle.x));
                particle.y = std::max(1.0f + size, std::min(float(height-(1+size)), particle.y));

                
                auto aa = (momentum ) * friction;
                // }
                // #pragma omp parallel for
                for (int io = -size; io < size+1; io++)
                {
                    for (int jo = -size; jo < size+1; jo++)
                    {
                        int2 offset = {io, jo};
                        float4 offsetto = {io, jo,0,0};
                        int2 poffset = {particle.x + offset.x, particle.y + offset.y};
                        float4 moffset = offsetto*repulsion/((io*io+jo*jo + 1.0)) + aa;
                        *TensorField.get<float4>(poffset.y, poffset.x) += moffset;           
                    }
                }
            }else{
                // #pragma omp parallel for
                for (int io = -size; io < size+1; io++)
                {
                    for (int jo = -size; jo < size+1; jo++)
                    {
                        int2 offset = {io, jo};
                        int2 poffset = {particle.x + offset.x, particle.y + offset.y};
                        *TensorField.get<float4>(poffset.y, poffset.x) *= -1;           
                    }
                }
            }

    
            if (show)
            {
                draw_circle(screenbuffer, particle.x/scale, particle.y/scale, size+1, mm, 120, 120, 255);
            }
            // 
        }
        // Update screen
        if (show)
        {
           
            texture.update((uint8_t *)screenbuffer.data);
            sf::Sprite sprite(texture);
            window.draw(sprite);

            // end the current frame
            window.display();
            auto currtime = std::chrono::system_clock::now();
            std::cout << "frames per second(0dp): " << int(1.0 / std::chrono::duration<double>(currtime - currenttime).count()) << " : Active Particles: " << count << "\r";
            std::cout << std::flush;
            currenttime = currtime;
            frame++;
        }
        // if(frame==1000)break;
    }

    return 0;
}