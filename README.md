# Algebra_project
Ok now i will code a libray similar to numpy in c++ using SIMD (Single Instruction, Multiple Data) 
I want to point out that i am not a engineer and even less a computer engineer so maybe this is not the most efficient way to do this but 
the reason why i am doing this is beacuse i want to learn. 
This libreary will have matrices / transposed matrices and vectors of type double, float, int, uint64_t, int64_t and bool, i will explain firts the type bool.

The point of the vector or matrix boolean are mainly three: Logical operations, 
Replace elements based on conditions (somthing like numpy.where) and count the number of elements that are true, 
the last two are the reason why i will need to use uint8_t to emulate bool since some simd instructions such as _mm256_blend_pd and _mm256_movemask_epi8 
require the most significant bit to be one for true and 0 for false so 
i can not use bool because for some reason the c++ compile sometimes convert value that i want to assign to a bool variable into true for example 
if i want to assign 0b11111111 which is basically 255 in binary the compiler will assign just 1 or true.
The logical operations and blend operations are pretty easy to implement(even though i haven't implemented the blend operations yet :) ), 
the real challenge is count the number of elements that are true in a fast way. 
Somehow I had an idea that I can use _mm256_movemask_epi8 to extract the most significant bits of 32 uint8_t and save them in an array of 8 32-bit integers and then 
load them into a SIMD register then count the number of bits that are one in each 32-bit integer and finally add everything, with this
I can process 256 elements in a single iteration, which according to the The tests I did it is almost 4 times faster than iterating element by element
I think there is nothing remarkable in the double type maybe that i implemented an algorithm that allows me to store the result of a comparation between 
two 64 bits numers in an array of uint8_t with some shifts and permutes, 
in the uint64_t I implemented an algorithm to calculate the power of integers quickly and with simd.
Since i started to develop this library a couple of weeks ago and most of the time i have been planning the structure and implementation of each function, 
i have not finish this libray i still need to implement Dot product / matrix multiplication, 
a function to compute the determinant, other types of vectors and matrices and I need to check that everything works well, 
I had already implemented most of this but due to the restructuring I have done I must implement it from scratch.
Maybe when someone sees this I will have already implemented what is missing.
My ultimate goal is finish this and then start to code the optimization libreary in c++.
I am thinking of adding multiprocessing so that different types of optimization algorithms can be used at the same time 
and they communicate by sharing memory or sharing a database like MySql, it would be interesting to see a tpe and a genetic algorithm together.
