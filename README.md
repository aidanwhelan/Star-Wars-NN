# Star Wars Neural Network  
An Artificial Neural Network to generate a film script in the style of Star Wars.  
This model and approach is drawn from Machine Learning with Phil (channel linked below) as a study in Natural Language Processing and its applications for film scripts.  

### Motivation  
**1)** To optimize the movie-making pipeline by assisting screenwriters in  generating script segments.  
**2)** Sequel scripts are notoriously challenging to write, and may benefit from the continuity of the style of their predecessors.  
**3)** Star Wars is a *series* of well-written, popular films. Multiple scripts provides more data for training a model.  
**4)** AI / NN models have already been written for generating stage scenes in the style of William Shakespeare. With some modification, similar models could be used to generate film scripts.  
**5)** As a huge Star Wars fan, generating new content offers new (unofficial) stories between official Disney content releases.

### Instructions  
**starwars_nn.py:** Full model training, checkpoint generation, text generation, and loss metric visualization.  
- After cloning, remove training_checkpoints directory
- Training: ~140 minutes
- Returns:  
  - 1500 characters of a scene in the style of Star Wars Ep. IV: A New Hope  
  - Summary of model structure
  - Plot of training loss over 192 epochs  

**quick_gen.py:** Uses existing checkpoints of training weights and biases to generate text (no additional training required)  
- Returns:  
  - 1500 characters of a scene in the style of Star Wars Ep. IV: A New Hope
  - Summary of model structure

### Credits  
[Machine Learning with Phil](https://www.youtube.com/watch?v=xs6dOWlpQbM&t=748s) - Shakespeare Text Generation NN Tutorial  

[gastonstat](https://github.com/gastonstat/StarWars) - Repository of Star Wars film scripts  

[scifiscripts.com](http://www.scifiscripts.com/scripts/swd1_5-74.txt) - Early draft of The Star Wars by George Lucas    
