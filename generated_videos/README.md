# Generated Videos Using Text-to-Video Models

This directory contains the video output of various open-source Text-to-Video(T2V) models.

### T2V Models
This includes the output of the models:

- Aphantasia [GitHub](https://github.com/eps696/aphantasia)
- ModelScope Text2Video Synthesis [GitHub](https://github.com/modelscope/modelscope)
- Tune-a-video [GitHub](https://tuneavideo.github.io/) [Paper](https://arxiv.org/pdf/2212.11565.pdf) 
- VideoCrafter[GitHub](https://github.com/VideoCrafter/VideoCrafter)
- ModelScope VideoFusion[GitHub](https://github.com/modelscope/modelscope) [Paper](https://arxiv.org/pdf/2303.08320v3.pdf)

Each of the models use the prompts in text file "./prompts.txt" to create 35 unique videos.
The prompts used are based on the prompts presented by Meta and Google for their closed-source T2V models Make-a-Video and Imagen Video respectively.

The code used to generate the videos can be found in the directory "../T2V_models/$MODEL_TYPE".
Where $MODEL_TYPE is one of the five models presented above.

---