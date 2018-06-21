# Image assembling
## Use a set of images to assemble another image that you input

Many people from 90's or before remember the movie The Truman Show. I was very impressed by it's poster, so impressed that I remember it till this day:

![The Truman Show Poster](https://thesmithresponse.files.wordpress.com/2014/09/5030336_f520.jpg)

However, if you look closely, you will find the poster actually puts a mask on a set of images to achieve a better look. This project is to help people achieving such effect without just putting a mask over images.

For example, we all love the show Rick and Morty, let us look at the poster for season three:
![The original poster](https://github.com/txstc55/image_assembling/blob/master/rick_morty_pos.jpg)

And here is the sample generated from all episodes from season three:
![The sample output](https://github.com/txstc55/image_assembling/blob/master/rick_morty_pos_out.jpg)

Feel free to download this image to look at the details. DUE TO THE LIMIT OF FILE SIZE OF GITHUB, I CANNOT GET A BETTER ONE UPLOADED.

### Now let us talk code:
This project contains two main file (may change later) and I intend for it to stay that way. Those files are pre_process_images.py and assemble_image.py. The minimum packages you need for running those codes are:
+ json
+ ast
+ PIL
+ math
+ itertools
+ operator
+ functools
+ numpy
+ random
+ multiprocessing
+ os
+ sys
+ optparse
+ shutil

If you have anaconda3, ignore those. I have tried to use PIL as much as possible since cv2 does not come with Anaconda.
The environment is PYTHON 3.
PLEASE RUN THOSE CODE UNDER LINUX ENVIRONMENT! I HAD PROBLEM WITH MULTIPROCESSING MODULE IN WINDOWS, WILL FIX THAT LATER BUT NOT NOW!

The functions of the two files are quite straight forward:
preprocess_images.py pre processes a set of images (those images should be in a folder), and will create a json file that contains the color information of each image.
assemble_image.py assembles an image using the color information generated from before.

The reason I want them to stay seperate is so that you can pre process different sets of images, and after that, decide which set you want to use for assembling.
Ok now let us really talk about code.

### Pre Processing
As I said before, this file is just for pre processing, so no fancy inputs, just do:
```bash
python pre_process_images.py your/image/directory/
```
While running, you will see outputs telling you what image has been processed. So far, only jpg and png files are supported. However, as long as PIL library supports the file format, you can edit line 40:
```python
if (image_file.lower().endswith("png") or image_file.lower().endswith("jpg")):
```
to adjust your need. Yes I know this is a very cheap way, I was just trying to avoid video files back then.

The code will automatically evaluate the colors in the middle part (so if it is a 1920*1080 image, only the middle 108*1080 will be evaluated as the color key for the picture), later there will be support for evaluating the entire image, more reason about why I did so will be explained in assembling part.

The genius part of the color evaluating part is that it does not use knn (we don't have that time), or just averaging the color (that is really cheap, but since we are dealing with real life pictures, this does not work all the time), the idea that I thought of works this way:
1. Find all the pixel values and group them, this can be easily scquired using: ```pythonim.getcolors()``` where im is a PIL.Image object.
2. Group the same colors, and add their counts. This is where we can use reduce by key (lambda rules): ```python [reduce(my_reduce, group) for _, group in groupby(sorted(colors), key=itemgetter(1))][::-1] ``` where colors is the returned value from ```python getcolors()``` function. We have to do this step because sometimes the returned value is not always grouped by the key.
3. Find the similar colors. This part can be defined in various ways, the way I did it is: if a color and another color's cartesian distance is small, and the variance of each rgb value's square distance is small, we say they are the same color. You can change those numbers to a smaller number for a more strict color classification.
4. Sort the list by count, pick the most frequent color, easy to understand why I did this.
5. For the rest colors, loop through them, if the count difference with the most frequent one has less than 20% (which can be changed) difference, we say that color is also dominant. We keep looping until we find one that is not.

Now if you have more questions about how I did grouping, or you have a better idea, you are very welcome to leave a message. 

In the end, you will get a folder called data, and inside you will have xxx_data.json, where xxx is the name of the directory that contains your images.

### Assemble Image
The assemble_image.py takes two mandatory inputs (in order), the image that you want to assemble, and the data that you want to use.
For example. I have pre processed a directory called rick_morty/ that contains lots of images of rick and morty, then in data directory, you will see a file called rick_morty_data.json. To use that data to assemble an image called rick_and_morty.jpg, this is what you should input:
```bash
python assemble_image.py rick_and_morty.jpg rick_morty
```
Yes just like that, and you can watch your terminal flooded with information, don't worry, I left those line uncommented so that you know it is working, since the process will take fairly long.
In the middle of processing there will be directory made to store thumbnail images, it will be gone eventually.

So far there are three optional inputs: overlap, cut and enlarge.
#### Overlap
For overlap, if set to True (please only set them to True or False, not true or false, not 1 or 0, not y or n, because I was too lazy to change for now), Then instead of cutting and pasting only the middle part of an image, it will paste the entire image. Hence there will be overlaps, an example output for Rick and Morty is:
![Sample output with overlap](https://github.com/txstc55/image_assembling/blob/master/rick_morty_pos_out_overlap.jpg)

Well it is not perfect (of course). However, this is something that the next to optional inputs may help.
To enable overlap, do this:
```bash
python assemble_image.py rick_and_morty.jpg rick_morty -o True
```
The default is set to false.

#### Cut
Cut is very straight forward, how large should each cut be, the default is 10, so if you input 1920*1080 image, it will be cut to 192*108 squares.
To change the value to 5, do:
```bash
python assemble_image.py rick_and_morty.jpg rick_morty -c 5
```

#### Enlarge
Enlarge is also very straight forward, how much you want to enlarge each square for pasting. So if you have the cut set to 10, then each square will be 10 pixel by 10 pixel. When you paste the image, it will also shrink to fit that square. Of course you don't want to do that, and this is where cut jumps in.
For example, if you set enlarge to 8 by doing:
```bash
python assemble_image.py rick_and_morty.jpg rick_morty -c 5 -e 8
```
Then each pasted image will occupy 40\*40 pixels. As a result, the output image will have a larger pixel count than the original one. The default enlarge value is set to 10.

Like I mentioned, you can always optimize the result (get a better detail) by setting a samll cut value and a fairly large enlarge value.

The algorithm will put a mask on it (yes I know I said it was lame but I did it anyway). The origin:mask ration is 0.75:0.25, this can be changed in line 252:
```python
img = np.uint8(0.75*img+0.25*origin_fragmant)
```

Later I will make that an optional value too.


Alright that will be all. If you have any questions, I guess you can open an issue (I guess). Or email txstc55@gmail.com 

Have fun and make some images!
I cannot show off my girlfriend's image because it is too large, but I can put up a gif!
![A gif to show off](https://github.com/txstc55/image_assembling/blob/master/gf.gif)
