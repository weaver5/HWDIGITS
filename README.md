Handwritten Digits 0-9 with Convolutional Neural Networking !!!✨
<pre>CNN model with Raspberry Pi Code to take in image data for digits 0-9
See model code, written in Juypter notebook
See Pi code, written in .py file on the Raspberry Pi 3
</pre>
<pre>
In the process of setting up the environment for this project, I chose to use a Raspberry Pi 3 Model B Plus Rev 1.3 as the hardware platform, running the default Raspberry Pi OS based on Debian. Python 3.7.12 was selected as the programming language, and TensorFlow 2.4.0 was installed as the deep learning framework. <br />
However, I encountered challenges in ensuring compatibility between Python, TensorFlow, and the Raspberry Pi model. Specifically, I faced issues when trying to obtain the correct versions for TensorFlow and Python. Despite my efforts to install the appropriate versions and confirming compatibility with my 32-bit OS, I persistently encountered the error "No module named 'tensorflow'." <br />
 After some troubleshooting, I was able to resolve this issue by opting to reinstall the entire environment. This corrective step has proven effective, and I was able to continue with the project. <br />
 Additionally, the camera used for image capture in this project is the Pi Camera with the imx219 sensor, providing the necessary imaging capabilities despite the initial setup issues I encountered.I also had to make sure my 1.19.5 numpy version was compatible with my tensorflow.
</pre>
`Hardware: Raspberry Pi 3 Model B Plus Rev 1.3`
`Operating System: Raspberry Pi Debian - Default Pi OS 32 bit`
`Python: Python 3.7.12`
`TensorFlow: 2.4.0`
`Camera: Pi Cam-imx219`
<pre>
The `imgprocessed` function oversees the preprocessing of images, predicts the digit, and presents the outcome along with confidence and processing time. 
I was able to use the np.argmax function to accurately show the predicted digit's accuracy. Through the application of np.argmax on the prediction array, the script chooses the index associated with the highest confidence or probability. This index directly corresponds to the digit that the model predicts with the greatest certainty. 
The confidence level is determined by getting the specific confidence value linked to the predicted digit from the prediction array. In my code, after making predictions using the model.predict method, the outcomes are stored in the variable predic. The confidence level for the predicted digit is then used by indexing into the array using predic[0][pred_val].
 The display_sensehat function visually shows the predicted digit on the sense hat. A continuous video capture loop allows me to start predictions by pressing 'c' and stop the program with 'q'. 
The output from the video each time when I pressed the button looked like the following image. I utilized a monitor with digitally handwritten digits, and my demo can be found at this link: https://www.youtube.com/watch?v=7rFuzUy9Gkg&ab_channel=MichaelaCrego
</pre>
