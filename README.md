# trainplicate

A python application to take care of your FLUX training through the replicate API.

## Prerequisites

Get the model [xtuner_llava-phi-3-mini-hf](https://huggingface.co/xtuner/llava-phi-3-mini-hf) from Huggingface. You can use other models ofcause but you will be on your own if you need any help. Rember where you stored the directory you will need it later.

## Installing.

Git clone this repository and change to its directory. This will become your workplace.

Run the install file:

```zsh
./install.zsh
```

It will setup the environment. If it ever becomes corrupt because you tinkered with is delete and repeat.

## Pre run

Open the trainplicate.py file and look for the variables owner and mpath you cant mis them promise. Change these to reflect your settings.

## Run

Not so fast, we do a test first to see if everything went fine. As you can see I included a directory with images. We are going to process them. At the end the directory should contain a txt file for each image and a zip file containing the images and txt files. The program stopped short before the final stage.

Run the test with:

```zsh
./startz.zsh flux1.lora.watercolors wcolors "Beautiful colors, many colors."
```

If the test did not give any errors you can open trainplicate.py again and remove the exit statement at the bottom. While scrolling down you might read a few of the comments.

Now you are all set to run your own set. just create a directory reflecting your lora name and throw in your pictures. Usually 20 is just fine. Then you can start again. You have to give 3 parameters. 

* The name of your lora which should be the directory name.
* The token. It can be used in flux to recognize your Lora and will be prepended to your captions.
* Desciption. A description only used for Replicate. 
 
Thats it beautiful people. There are timestamps throughout the output so you can track your script running.

Have fun.


