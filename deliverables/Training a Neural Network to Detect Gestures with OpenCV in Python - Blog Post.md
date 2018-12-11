# Training a Neural Network to Detect Gestures with OpenCV in Python

### Background

I've been curious about gesture detection for a long time. I remember when the first Microsoft Kinect came out -  I had an absolute blast playing games and controlling the screen with just a wave of the hand. As time went on and devices like Google Home and Amazon Alexa were released, it seemed that gesture detection fell of the radar. Still, I wondered whether it was likely to experience a renaissance, now that video devices like the Facebook Portal are coming out. With this in mind, I wanted to see if it was possible to build a neural network that could recognize my gestures in real time!

### Data, and my early models

I was excited about this idea and moved quickly to implement it, like I'd been shot out of a cannon. I started working with a hand gesture recognition database on [Kaggle.com](https://www.kaggle.com/benenharrington/hand-gesture-recognition-database-with-cnn), and exploring the data. It consists of 20,000 labeled hand gestures, like the ones found below.

![Hand images](https://github.com/athena15/project_kojak/raw/master/deliverables/hand_images.jpg)

The data are organized into 10 folders, each of which represents a single person's images. Each person then took 200 images for each gesture. This requires walking through the file tree, recording which person and gesture is represented in the image. I then read in the images like so:

```python
img = Image.open(path)
img = img.resize((224, 224))
arr = np.array(img)
X_data.append(arr)
```
Once I read the images in, the first problem I ran into was that my images were black & white. This means the NumPy arrays have only one channel, instead of three (i.e., the shape of each array is (224, 224, 1)). As a result, I was unable to use these images with the VGG-16 pretrained model, as that model requires an RGB, 3-channel image. This was solved by using np.stack on the list of images, X_data:

```python
X_data = np.array(X_data, dtype = 'float32')
X_data = np.stack((X_data,)*3, axis=-1)
```

Once I got through that hurdle, I used Keras to build a neural network of my own, as well as a model that used VGG-16. I was having incredible success, too. After doing a classic train-test split, I was still getting test accuracy of 99%!

Still, I was anxious to see if my model was good enough to work not only with data from the Kaggle dataset, but also with images I created by using my Macbook Pro's webcam. I created functions to apply the same transformations to my new, self-created images as I had with the Kaggle images. Then, one by one, I fed in my new images and hoped that my model would be smart enough to recognize those as well!

Unfortunately, it was not. I couldn't believe that my model was supposedly as accurate as it was, yet when I gave it new images to predict, it seemingly fell flat. What was going on?

It seemed there were two problems. One was that the source images from Kaggle were rather weird - very pale, ghost-like images with black backgrounds - and it was hard to replicate the look of these images with images I generated myself.

The second problem was that I had leaky data. I had suspected as much when I was achieving 99-100% accuracy right off the bat. I hadn't taken into account the fact that my train-test split *had included all 10 of the individuals in the images*. Compounding the problem, many of the images in sequential order were strikingly similar to each other. They looked like they might even be screenshots of a video, they were so similar. My model had seen very, very similar images to the ones in the test set, and simply remembered which category they belonged to. The good news was, I was enough of a skeptic to know my results were too good to be true!

To remedy these problems, I set about building a new model, using a train-test split that *completely held out* photos from 2 of the 10 individuals in them. This made my train-test split valid. After re-running the model built on the VGG-16 architecture, my model achieved an F1 score of 0.74 overall. This was a much more believable result, I thought - and quite good, given that random guessing over 10 classes would result in only 10% accuracy, on average.

However, training a model to recognize images from a dataset is one thing. Training it to recognize images from a webcam that it hasn't seen before is another. I hoped that my new model would generalize well enough to predict the images I provided via webcam. I tried adjusting the lighting of my photos, and using a dark background - mimicking the photos the model had trained on. Although this model did better than before, I was still having unpredictable - and in my mind unacceptable - results.

I tried image augmentation - flips, skews, rotations. This got me closer than before, but I still had the nagging feeling that I needed to rethink the problem.

***Takeaway: Train your model on images that are as close as possible to the images it is likely to see in the real world.***

#### Rethinking the problem

I decided to pivot and try something new. It seemed to me that there was a clear disconnect between training and real-world data. I decided I'd try building my own dataset.

I had been working with OpenCV, an open source computer vision library, and I needed an engineer a solution that would turn on my webcam, grab an image from the screen, then resize and transform the image into a NumPy array that my model could understand. The methods I used to transform my data are as follows:

```python
from keras import load_model
model = load_model(path) #open saved model/weights from .h5 file

def predict_image(image):
	image = np.array(image, dtype='float32')
	image /= 255
	pred_array = model.predict(image)

    '''
    model.predict() returns an array of probabilities - 
    np.argmax grabs the index of the highest probability.
    '''
	result = gesture_names[np.argmax(pred_array)]
    
    '''
    A bit of magic here - the score is a float,
    but I wanted to display just 2 digits beyond the decimal
    point.
    '''
	score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(f'Result: {result}, Score: {score}')
	return result, score
```
In a nutshell, once you get the camera up and running using:

```python
#starts the webcam, uses it as video source
camera = cv2.VideoCapture(0) 

while camera.isOpened():
    #ret returns True if camera is running, frame grabs each frame of the video feed
	ret, frame = camera.read()
```

you can grab the frame, transform it, and get a prediction from your model:

```python
target = np.stack((frame,)*3, axis=-1)
target = cv2.resize(target, (224, 224))
target = target.reshape(1, 224, 224, 3)
prediction, score = predict_image(target)
```

Getting the pipeline connected between the webcam and my model was a big success. I started to think about what would be the ideal image to feed in to my model. One clear obstacle was that it's difficult to separate the area of interest (in our case, a hand) from the background.

#### Extracting the gesture

The approach that I took was one that is familiar to anyone who has played around with Photoshop at all - background subtraction. It's a beautiful thing! In essence, if you take a photo of a scene before your hand is in it, you can create a "mask" that will remove everything in the image except the new object.

Once I had subtracted the background from my images, I then used binary thresholding to make the target gesture totally white, and the background totally black. I chose this approach for two reasons: it made the outline of the hand crisp and clear, and it made the model easier to generalize across users with different skin colors. This created the telltale "silhouette"-like photos that I ultimately trained my model on.

#### Building a new dataset

Now that I could accurately detect my hand in images, I decided to try something new. My old model didn't generalize well, and my ultimate goal was to build a model that could recognize my gestures in real time - so I decided to build my own dataset!

I chose to focus on 5 gestures:

![5 hand gestures](https://github.com/athena15/project_kojak/raw/master/deliverables/Background%20Masking.png)

I strategically chose 4 gestures that were also included in the Kaggle data set, so I could cross-validate my model against those images later. I also added the peace sign, although that gesture did not have an analogue in the Kaggle data set.

From here, I built a dataset by setting up my webcam, and creating a click binding in OpenCV to capture and save images with unique filenames. In no time, I had built a dataset with 550 silhouette images each.

#### Training the new model

I then built a convolutional neural network using Keras & TensorFlow. In order to cross-validate my model, I decided to see whether my trained model could also accurately predict images from the Kaggle data set.  In order to do this, I applied the same transformations to each Kaggle image that I had applied to my training data - background subtraction and binary thresholding. This gave them a similar "look" that my model was familiar with.

#### Results

The model's performance exceeded my expectations. It classified nearly every gesture in the test set correctly, ending up with a 98% F1 score, as well as 98% precision and accuracy scores. This was great news!

As any well-seasoned researcher knows, though, a model that performs well in the lab but not in real life isn't worth much. Having experienced the same failure with my initial model, I was cautiously optimistic that this model would perform well on gestures in real time.

#### Smart Home Integration

Before testing my model, I wanted to add another twist. I've always been a bit of a smart home devotee, and my vision had always been to control my Sonos and Philips Hue lights using just my gestures. To easily access the Philips Hue and Sonos APIs, I used the phue and SoCo libraries, respectively. They were both extremely simple to use, as seen below:

```python
# Philips Hue Settings
bridge_ip = '192.168.0.103'
b = Bridge(bridge_ip)

on_command =  {'transitiontime' : 0, 'on' : True, 'bri' : 254}
off_command =  {'transitiontime' : 0, 'on' : False, 'bri' : 254}

# Turn lights on
b.set_light(6, on_command)

#Turn lights off
b.set_light(6, off_command)
```

Using SoCo to control Sonos via the web API was arguably even easier:

```python
sonos_ip = '192.168.0.104'
sonos = SoCo(sonos_ip)

# Play
sonos.play()

#Pause
sonos.pause()
```

I then created bindings for different gestures to do different things with my smart home devices:

```python
# boolean set earlier to turn smart home functionality on/off
if smart_home:
    if prediction == 'Palm':
        try:
            action = "Lights on, music on"
            sonos.play()
        # turn off smart home actions if devices are not responding    
        except ConnectionError:
            smart_home = False
            pass
    # etc. etc.
```

When I finally tested my model in real time, I was extremely pleased with the results. My model was accurately predcting my gestures the vast majority of the time. See the video below for a demo:

[Gestures Demo Video](https://www.youtube.com/watch?v=CsMcAxD7FxM&feature=youtu.be&vq=720)

I hope you enjoy the results!